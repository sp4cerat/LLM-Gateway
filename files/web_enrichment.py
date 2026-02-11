"""
LLM Gateway — Web Search Enrichment & Fact-Checking
=====================================================

Two-stage pipeline for complex queries:

  Stage 1 (Classification): Cheap model decides via tool-calling (existing)
  Stage 2 (Deep Search + Synthesis): This module

Flow:
  1. DDG snippets (fast, ~200ms)
  2. Trafilatura full-text extraction (top 3 URLs, ~1-3s)
  3. Enriched context injected into LLM call for synthesis
  4. Optional: post-response fact-check with cross-reference

Cost (Gemini 2.0 Flash):
  - Snippets: ~1k-2k tokens → <$0.0003
  - Full-text (3 pages): ~5k-15k tokens → <$0.001
  - Total per complex query: <$0.002

Usage:
  from web_enrichment import get_web_enricher
  enricher = get_web_enricher()

  # Pre-query enrichment (before escalated LLM call)
  result = await enricher.enrich_query(query)
  # → inject result.enriched_context into messages

  # Post-response fact-check
  fc = await enricher.fact_check(query, llm_response)
  # → if fc and fc.has_data: re-call LLM with fc.enriched_context
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

log = logging.getLogger("gateway.web_enrichment")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MAX_FULL_TEXT_PAGES = 3
MAX_PAGE_CHARS = 4000
PAGE_FETCH_TIMEOUT = 8.0
DDG_MAX_RESULTS = 5
FACT_CHECK_MIN_QUERY_LEN = 50

SKIP_DOMAINS = {
    # Social media (no useful full text)
    "youtube.com", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "reddit.com", "linkedin.com", "pinterest.com",
    # Search engines / homepages (DDG sometimes returns these as results)
    "google.com", "google.de", "bing.com", "yahoo.com", "yandex.com",
    "duckduckgo.com", "baidu.com",
    # News homepages that return generic content instead of articles
    "bild.de", "t-online.de",
    # Generic portals that won't have useful full text
    "amazon.com", "amazon.de", "ebay.com", "ebay.de",
}

# Additional filter: skip URLs that are clearly just homepages (no path)
def _is_homepage(url: str) -> bool:
    """Detect if URL is just a homepage (e.g., https://bild.de/) rather than an article."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return path == "" or path == "/" or len(path) < 3
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    full_text: Optional[str] = None
    domain: str = ""
    fetch_time_ms: float = 0


@dataclass
class EnrichmentResult:
    query: str
    search_results: list[SearchResult] = field(default_factory=list)
    enriched_context: str = ""
    token_estimate: int = 0
    search_time_ms: float = 0
    fetch_time_ms: float = 0
    total_time_ms: float = 0
    sources_fetched: int = 0
    method: str = "none"       # "none" | "snippets" | "deep" | "fact_check"
    has_data: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Query Classification
# ═══════════════════════════════════════════════════════════════════════════════

_FACT_PATTERNS = [
    r"(ist|sind|hat|haben)\s+.+\s+(noch|immer noch|aktuell|derzeit)",
    r"(is|are|has|have)\s+.+\s+(still|currently|now)",
    r"(stimmt|wahr|richtig|korrekt)\s+(es|das|dass)",
    r"(is it true|is it correct|really|actually)\b",
    r"\d+[\s.]*(prozent|percent|milliard|billion|million|%|euro|dollar)",
    r"(gestern|letzte woche|kürzlich|neulich|vor kurzem|seit wann)",
    r"(yesterday|last week|recently|just now|since when)",
]
_FACT_RE = [re.compile(p, re.IGNORECASE) for p in _FACT_PATTERNS]

_DEEP_KEYWORDS = {
    "vergleich", "analyse", "unterschied", "vor- und nachteile",
    "compare", "analysis", "difference", "pros and cons",
    "erkläre", "warum", "wieso", "wie funktioniert",
    "explain", "why", "how does", "mechanism",
    "auswirkungen", "konsequenzen", "bedeutung",
    "impact", "consequences", "implications",
}


def classify_needs_enrichment(query: str) -> dict:
    """Classify if query benefits from web enrichment / fact-check."""
    q_lower = query.lower().strip()

    for pat in _FACT_RE:
        if pat.search(q_lower):
            return {"needs_deep": True, "needs_fact_check": True, "reason": "factual_claim"}

    for kw in _DEEP_KEYWORDS:
        if kw in q_lower:
            return {"needs_deep": True, "needs_fact_check": False, "reason": f"keyword:{kw}"}

    if len(query) > FACT_CHECK_MIN_QUERY_LEN:
        return {"needs_deep": False, "needs_fact_check": True, "reason": "complex_query"}

    return {"needs_deep": False, "needs_fact_check": False, "reason": "simple"}


# ═══════════════════════════════════════════════════════════════════════════════
#  DDG Search
# ═══════════════════════════════════════════════════════════════════════════════

def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


async def ddg_search(
    query: str, max_results: int = DDG_MAX_RESULTS, time_filter: str = "w",
    append_date: bool = True,
) -> list[SearchResult]:
    """DuckDuckGo search → structured results.

    Args:
        time_filter: 'd' (24h), 'w' (week), 'm' (month), None (no limit)
        append_date: If True, append current month/year to query (good for news,
                     bad for research). Set False for deep/thorough searches.
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    def _search():
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            try:
                from ddgs import DDGS
            except ImportError:
                log.warning("No DDG search lib. Install: pip install ddgs")
                return []
        try:
            # Only append date for news-style queries, not for research
            if append_date:
                search_q = f"{query} {datetime.now().strftime('%B %Y')}"
            else:
                search_q = query
            # Fetch extra results to compensate for filtering
            fetch_count = max_results + 5
            with DDGS() as ddgs:
                raw = list(ddgs.text(
                    search_q, max_results=fetch_count,
                    timelimit=time_filter,  # None = no time restriction
                ))

            # Filter out garbage: skip domains, homepages, clearly irrelevant
            filtered = []
            for r in raw:
                url = r.get("href", "")
                dom = _domain(url)
                if dom in SKIP_DOMAINS:
                    log.debug(f"DDG filter: skip domain {dom}")
                    continue
                if _is_homepage(url):
                    log.debug(f"DDG filter: skip homepage {url[:60]}")
                    continue
                filtered.append(r)
                if len(filtered) >= max_results:
                    break

            if len(filtered) < len(raw):
                log.info(f"DDG filter: {len(raw)} → {len(filtered)} results "
                         f"(removed {len(raw)-len(filtered)} garbage)")

            return [
                SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    url=r.get("href", ""),
                    domain=_domain(r.get("href", "")),
                ) for r in filtered
            ]
        except Exception as e:
            log.warning(f"DDG search failed: {e}")
            return []

    results = await loop.run_in_executor(None, _search)
    elapsed = (time.time() - start) * 1000
    log.info(f"DDG: '{query[:60]}' → {len(results)} results, {elapsed:.0f}ms"
             f" | date={'yes' if append_date else 'no'} tf={time_filter}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Full-Text Extraction (trafilatura + fallback)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_full_text(url: str) -> Optional[str]:
    """Fetch + extract main text. Primary: trafilatura. Fallback: HTML strip."""
    if _domain(url) in SKIP_DOMAINS:
        return None

    try:
        async with httpx.AsyncClient(
            timeout=PAGE_FETCH_TIMEOUT, follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "de,en;q=0.9",
            },
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            html = r.text
    except Exception as e:
        log.debug(f"Fetch failed {url[:60]}: {e}")
        return None

    loop = asyncio.get_event_loop()

    def _extract():
        # trafilatura (best quality)
        try:
            import trafilatura
            text = trafilatura.extract(
                html, include_comments=False, include_tables=True,
                favor_precision=True,
            )
            if text and len(text) > 50:
                return text[:MAX_PAGE_CHARS]
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"trafilatura: {e}")

        # Fallback: basic HTML strip
        try:
            from html.parser import HTMLParser

            class _T(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.p, self._s = [], False
                def handle_starttag(self, tag, a):
                    if tag in ("script","style","nav","header","footer","aside"):
                        self._s = True
                def handle_endtag(self, tag):
                    if tag in ("script","style","nav","header","footer","aside"):
                        self._s = False
                def handle_data(self, d):
                    if not self._s and len(d.strip()) > 3:
                        self.p.append(d.strip())

            ext = _T()
            ext.feed(html[:60000])
            text = "\n".join(ext.p)
            return text[:MAX_PAGE_CHARS] if len(text) > 50 else None
        except Exception:
            return None

    text = await loop.run_in_executor(None, _extract)
    if text:
        log.debug(f"Extracted {len(text)} chars from {_domain(url)}")
    return text


async def fetch_pages(results: list[SearchResult], max_pages: int = MAX_FULL_TEXT_PAGES) -> int:
    """Fetch full text for top N results in parallel. Returns count fetched."""
    to_fetch = [
        r for r in results[:max_pages + 2]
        if r.url and not r.full_text and r.domain not in SKIP_DOMAINS
    ][:max_pages]

    if not to_fetch:
        return 0

    start = time.time()
    tasks = [fetch_full_text(r.url) for r in to_fetch]
    texts = await asyncio.gather(*tasks, return_exceptions=True)

    fetched = 0
    for result, text in zip(to_fetch, texts):
        if isinstance(text, str) and text:
            result.full_text = text
            fetched += 1

    log.info(f"Full-text: {fetched}/{len(to_fetch)} pages, {(time.time()-start)*1000:.0f}ms")
    return fetched


# ═══════════════════════════════════════════════════════════════════════════════
#  Context Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_context(
    query: str, results: list[SearchResult],
    mode: str = "synthesis", language: str = "de",
    depth: str = "deep",
) -> str:
    """
    Build structured context string from search results for LLM consumption.

    The context includes analytical instructions that scale with depth:
    - snippets: simple "answer from these sources"
    - deep: analytical synthesis with quantification
    - thorough: full macro-analytical framework
    """
    if not results:
        return ""

    parts = []
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    is_de = language.startswith("de")

    if mode == "fact_check":
        parts.append(
            f"[{'FAKTENCHECK' if is_de else 'FACT-CHECK'} — {ts}]\n"
            f"{'Originalfrage' if is_de else 'Question'}: {query}\n\n"
            + ("Prüfe die folgenden Quellen und korrigiere Fehler. "
               "Bei Widersprüchen bevorzuge aktuelle Quellen.\n"
               if is_de else
               "Verify against these sources. Prefer current sources over training data.\n")
        )
    elif depth == "thorough":
        if is_de:
            parts.append(
                f"[WEBRECHERCHE — {ts}]\n"
                f"Frage: {query}\n\n"
                "ANALYSE-ANWEISUNGEN:\n"
                "1. Extrahiere quantitative Daten (Zahlen, %, Prognosen) aus den Quellen\n"
                "2. Priorisiere Studien und Reports über Zeitungsartikel\n"
                "3. Synthetisiere zu einer kohärenten Analyse — NICHT Quelle für Quelle zusammenfassen\n"
                "4. Strukturiere nach thematischen Aspekten, nicht nach Quellen\n"
                "5. Identifiziere Konsens und Widersprüche zwischen den Quellen\n"
                "6. Gib quantitative Aussagen (Spannbreiten) statt vager Qualifizierungen\n"
                "7. Quellenangaben inline (z.B. 'laut OECD...', 'McKinsey schätzt...')\n"
            )
        else:
            parts.append(
                f"[WEB RESEARCH — {ts}]\n"
                f"Question: {query}\n\n"
                "ANALYSIS INSTRUCTIONS:\n"
                "1. Extract quantitative data (numbers, %, forecasts) from sources\n"
                "2. Prioritize studies and reports over news articles\n"
                "3. Synthesize into a coherent analysis — do NOT summarize source by source\n"
                "4. Structure by thematic aspects, not by sources\n"
                "5. Identify consensus and contradictions between sources\n"
                "6. Provide quantitative ranges instead of vague qualifiers\n"
                "7. Cite sources inline (e.g. 'according to OECD...', 'McKinsey estimates...')\n"
            )
    elif depth == "deep":
        if is_de:
            parts.append(
                f"[WEBRECHERCHE — {ts}]\n"
                f"Frage: {query}\n\n"
                "Nutze die folgenden Quellen für eine fundierte Antwort.\n"
                "WICHTIG: Synthetisiere die Informationen — fasse NICHT jede Quelle einzeln zusammen.\n"
                "Extrahiere konkrete Zahlen und Fakten. Gib Quellen inline an.\n"
            )
        else:
            parts.append(
                f"[WEB RESEARCH — {ts}]\n"
                f"Question: {query}\n\n"
                "Use these sources for a well-founded answer.\n"
                "IMPORTANT: Synthesize the information — do NOT summarize each source separately.\n"
                "Extract concrete numbers and facts. Cite sources inline.\n"
            )
    else:  # snippets
        parts.append(
            f"[{'WEBRECHERCHE' if is_de else 'WEB RESEARCH'} — {ts}]\n"
            f"{'Frage' if is_de else 'Question'}: {query}\n\n"
            + ("Beantworte anhand der folgenden Quellen.\n"
               if is_de else
               "Answer based on these sources.\n")
        )

    for i, r in enumerate(results, 1):
        block = f"\n── Quelle {i}: {r.title} ──\n"
        block += f"URL: {r.url}\n"
        if r.full_text:
            block += f"\n{r.full_text}\n"
        elif r.snippet:
            block += f"{r.snippet}\n"
        parts.append(block)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fact-Check Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_verify_queries(query: str, response: str) -> list[str]:
    """Extract verifiable claims from LLM response."""
    queries = [query]
    sentences = re.split(r'[.!?]\s+', response)
    for s in sentences:
        s = s.strip()
        if 20 < len(s) < 150 and re.search(r'\d', s) and len(queries) < 3:
            queries.append(re.sub(r'[„""\'«»\[\]]', '', s)[:100])
    return queries[:3]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class WebEnricher:
    """
    Web Search Enrichment & Fact-Check Pipeline.

    enrich_query() → pre-LLM: search + extract + build context
    fact_check()   → post-LLM: verify claims against web sources
    """

    async def enrich_query(
        self, query: str, deep: bool = False, language: str = "de",
        depth: str = "auto",
    ) -> EnrichmentResult:
        """
        Full enrichment: DDG search → [trafilatura full-text] → context.

        Args:
            depth: "auto" (from classification), "snippets", "deep", "thorough"
            deep: Legacy flag — if True, acts like depth="deep"
        """
        start = time.time()
        result = EnrichmentResult(query=query)

        classification = classify_needs_enrichment(query)

        # Resolve depth
        if depth == "auto":
            if deep or classification["needs_deep"]:
                depth = "deep"
            else:
                depth = "snippets"
        use_deep = depth in ("deep", "thorough")

        # DDG search
        t0 = time.time()
        ddg_max = 5 if depth == "thorough" else 5
        search_results = await ddg_search(query, max_results=ddg_max)
        result.search_time_ms = (time.time() - t0) * 1000
        result.search_results = search_results

        if not search_results:
            result.total_time_ms = (time.time() - start) * 1000
            return result

        # Deep/thorough: fetch full text via trafilatura
        if use_deep:
            max_pg = 5 if depth == "thorough" else 3
            t0 = time.time()
            fetched = await fetch_pages(search_results, max_pages=max_pg)
            result.fetch_time_ms = (time.time() - t0) * 1000
            result.sources_fetched = fetched
            result.method = depth if fetched > 0 else "snippets"
        else:
            result.method = "snippets"

        # Build context with analytical framing
        mode = "fact_check" if classification["needs_fact_check"] else "synthesis"
        result.enriched_context = build_context(
            query, search_results, mode=mode, language=language, depth=depth,
        )
        result.token_estimate = len(result.enriched_context) // 4
        result.has_data = bool(result.enriched_context)
        result.total_time_ms = (time.time() - start) * 1000

        log.info(
            f"Enrichment: '{query[:50]}' | {result.method} (depth={depth}) | "
            f"{len(search_results)} results/{result.sources_fetched} deep | "
            f"~{result.token_estimate} tok | {result.total_time_ms:.0f}ms | "
            f"reason={classification['reason']}"
        )
        return result

    async def fact_check(
        self, query: str, llm_response: str, language: str = "de",
    ) -> Optional[EnrichmentResult]:
        """
        Post-response verification: extract claims → search → build verification context.
        Returns None if no verification data found.
        """
        start = time.time()

        queries = _extract_verify_queries(query, llm_response)
        if not queries:
            return None

        # Search claims in parallel
        all_results: list[SearchResult] = []
        tasks = [ddg_search(q, max_results=3, time_filter="m") for q in queries]
        search_batches = await asyncio.gather(*tasks, return_exceptions=True)
        for batch in search_batches:
            if isinstance(batch, list):
                all_results.extend(batch)

        if not all_results:
            return None

        # Deduplicate
        seen = set()
        unique = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        # Fetch full text
        fetched = await fetch_pages(unique, max_pages=3)

        context = build_context(query, unique, mode="fact_check", language=language)

        result = EnrichmentResult(
            query=query, search_results=unique,
            enriched_context=context,
            token_estimate=len(context) // 4,
            sources_fetched=fetched,
            method="fact_check",
            has_data=bool(context),
            total_time_ms=(time.time() - start) * 1000,
        )
        log.info(
            f"Fact-check: {len(queries)} queries → {len(unique)} sources, "
            f"{fetched} deep | ~{result.token_estimate} tok | {result.total_time_ms:.0f}ms"
        )
        return result


# Singleton
_enricher: Optional[WebEnricher] = None

def get_web_enricher() -> WebEnricher:
    global _enricher
    if _enricher is None:
        _enricher = WebEnricher()
    return _enricher
