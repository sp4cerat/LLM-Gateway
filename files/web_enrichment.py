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

MAX_FULL_TEXT_PAGES = 8
MAX_PAGE_CHARS = 12000       # Default; overridden by config.web_search.max_page_chars
PAGE_FETCH_TIMEOUT = 8.0     # Default; overridden by config.web_search.page_fetch_timeout
DDG_MAX_RESULTS = 8
FACT_CHECK_MIN_QUERY_LEN = 50
MAX_CONCURRENT_DOWNLOADS = 3  # Limit parallel httpx connections (prevents C-level crashes)


def _get_page_limits() -> tuple[int, float]:
    """Get max_page_chars and page_fetch_timeout from config (with fallback to module defaults)."""
    try:
        from config import config
        ws = config.web_search
        return ws.max_page_chars, ws.page_fetch_timeout
    except Exception:
        return MAX_PAGE_CHARS, PAGE_FETCH_TIMEOUT

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
#  Google/Bing Scraping (no API key needed)
# ═══════════════════════════════════════════════════════════════════════════════

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}


async def google_search(
    query: str, max_results: int = 5, time_filter: str = None,
) -> list[SearchResult]:
    """Google search via HTML scraping (no API key). Fragile but free."""
    from urllib.parse import quote_plus
    import re as _re

    # Build URL with optional time filter
    tbs = ""
    if time_filter == "d":
        tbs = "&tbs=qdr:d"
    elif time_filter == "w":
        tbs = "&tbs=qdr:w"
    elif time_filter == "m":
        tbs = "&tbs=qdr:m"

    url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results + 5}&hl=de{tbs}"

    try:
        async with httpx.AsyncClient(
            timeout=8.0, follow_redirects=True, headers=_SCRAPE_HEADERS,
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                log.warning(f"Google scrape: HTTP {r.status_code}")
                return []
            html = r.text
    except Exception as e:
        log.warning(f"Google scrape failed: {e}")
        return []

    # Parse results from HTML — Google wraps results in specific patterns
    results = []
    # Pattern: <a href="/url?q=REAL_URL&..." followed by title text
    url_pattern = _re.compile(r'<a[^>]+href="/url\?q=([^&"]+)&')
    # Also try direct href pattern used in some Google layouts
    alt_pattern = _re.compile(
        r'<a[^>]+href="(https?://(?!google\.com|accounts\.google)[^"]+)"[^>]*>'
        r'(?:<h3[^>]*>)?([^<]+)'
    )

    for m in alt_pattern.finditer(html):
        link, title = m.group(1), m.group(2).strip()
        dom = _domain(link)
        if dom in SKIP_DOMAINS or _is_homepage(link) or not title:
            continue
        # Extract snippet: text near the URL
        pos = m.end()
        chunk = html[pos:pos + 500]
        snippet_m = _re.search(r'<span[^>]*>([^<]{20,300})</span>', chunk)
        snippet = snippet_m.group(1).strip() if snippet_m else ""

        results.append(SearchResult(
            title=title, snippet=snippet, url=link, domain=dom,
        ))
        if len(results) >= max_results:
            break

    log.info(f"Google: '{query[:50]}' → {len(results)} results (scraped)")
    return results


async def bing_search(
    query: str, max_results: int = 5, time_filter: str = None,
) -> list[SearchResult]:
    """Bing search via HTML scraping (no API key). More reliable than Google scraping."""
    from urllib.parse import quote_plus
    import re as _re

    # Build URL with optional freshness filter
    freshness = ""
    if time_filter == "d":
        freshness = "&filters=ex1%3a%22ez1%22"  # Last 24h
    elif time_filter == "w":
        freshness = "&filters=ex1%3a%22ez2%22"  # Last week
    elif time_filter == "m":
        freshness = "&filters=ex1%3a%22ez3%22"  # Last month

    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results + 5}{freshness}"

    try:
        async with httpx.AsyncClient(
            timeout=8.0, follow_redirects=True, headers=_SCRAPE_HEADERS,
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                log.warning(f"Bing scrape: HTTP {r.status_code}")
                return []
            html = r.text
    except Exception as e:
        log.warning(f"Bing scrape failed: {e}")
        return []

    # Bing uses <li class="b_algo"> for organic results
    results = []
    # Pattern: <h2><a href="URL">TITLE</a></h2> inside b_algo blocks
    block_pattern = _re.compile(
        r'<li\s+class="b_algo">(.*?)</li>', _re.DOTALL
    )
    link_pattern = _re.compile(r'<a\s+href="(https?://[^"]+)"[^>]*>(.*?)</a>', _re.DOTALL)
    snippet_pattern = _re.compile(r'<p[^>]*>(.*?)</p>', _re.DOTALL)

    for block_m in block_pattern.finditer(html):
        block = block_m.group(1)
        link_m = link_pattern.search(block)
        if not link_m:
            continue

        link = link_m.group(1)
        title = _re.sub(r'<[^>]+>', '', link_m.group(2)).strip()
        dom = _domain(link)

        if dom in SKIP_DOMAINS or _is_homepage(link) or not title:
            continue

        snippet_m = snippet_pattern.search(block)
        snippet = _re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip() if snippet_m else ""

        results.append(SearchResult(
            title=title, snippet=snippet[:300], url=link, domain=dom,
        ))
        if len(results) >= max_results:
            break

    log.info(f"Bing: '{query[:50]}' → {len(results)} results (scraped)")
    return results


async def multi_engine_search(
    query: str,
    engines: list[str] = None,
    max_results: int = 5,
    time_filter: str = "w",
    append_date: bool = True,
) -> list[SearchResult]:
    """
    Search across multiple engines in parallel, deduplicate by URL.
    Engines: "ddg" (default), "google", "bing"
    """
    if not engines:
        engines = ["ddg"]

    engine_map = {
        "ddg": lambda: ddg_search(query, max_results=max_results,
                                  time_filter=time_filter, append_date=append_date),
        "google": lambda: google_search(query, max_results=max_results,
                                        time_filter=time_filter if time_filter != "none" else None),
        "bing": lambda: bing_search(query, max_results=max_results,
                                    time_filter=time_filter if time_filter != "none" else None),
    }

    tasks = []
    engine_names = []
    for eng in engines:
        fn = engine_map.get(eng)
        if fn:
            tasks.append(fn())
            engine_names.append(eng)

    if not tasks:
        return []

    start = time.time()
    results_per_engine = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge + deduplicate
    seen_urls = set()
    merged = []
    for eng_name, eng_results in zip(engine_names, results_per_engine):
        if isinstance(eng_results, Exception):
            log.warning(f"Engine {eng_name} failed: {eng_results}")
            continue
        for r in eng_results:
            url_key = r.url.rstrip("/").lower()
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                merged.append(r)

    elapsed = (time.time() - start) * 1000
    log.info(f"Multi-engine [{','.join(engine_names)}]: '{query[:50]}' "
             f"→ {len(merged)} unique results, {elapsed:.0f}ms")
    return merged[:max_results * 2]  # Keep generous pool for filtering


# ═══════════════════════════════════════════════════════════════════════════════
#  Full-Text Extraction (trafilatura + fallback)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Playwright (optional JS rendering) ──

_PLAYWRIGHT_AVAILABLE = None  # lazy check

async def _fetch_html_playwright(url: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch fully rendered HTML via headless Chromium. Returns None if unavailable."""
    global _PLAYWRIGHT_AVAILABLE
    
    if _PLAYWRIGHT_AVAILABLE is False:
        return None
    
    try:
        from playwright.async_api import async_playwright
        _PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        _PLAYWRIGHT_AVAILABLE = False
        return None
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                locale="de-DE",
            )
            await page.goto(url, timeout=int(timeout * 1000), wait_until="networkidle")
            # Wait a bit for dynamic content to load
            await page.wait_for_timeout(1500)
            html = await page.content()
            await browser.close()
            log.info(f"Playwright: {len(html)} chars from {_domain(url)}")
            return html
    except Exception as e:
        log.debug(f"Playwright failed for {url[:60]}: {e}")
        return None


async def fetch_full_text(url: str, use_playwright: bool = False) -> Optional[str]:
    """Fetch + extract main text.
    
    Extraction layers (all combined, not fallback):
      1. JSON-LD / Schema.org (structured event data)
      2. trafilatura (main text, precision then recall)
      3. Basic HTML strip (last resort)
    
    If use_playwright=True, fetches HTML via headless Chromium (for JS-heavy SPAs).
    """
    if _domain(url) in SKIP_DOMAINS:
        return None

    max_chars, fetch_timeout = _get_page_limits()
    html = None

    # ── Fetch HTML ──
    if use_playwright:
        html = await _fetch_html_playwright(url, timeout=fetch_timeout)
    
    if not html:
        try:
            async with httpx.AsyncClient(
                timeout=fetch_timeout, follow_redirects=True,
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

    if not html:
        return None

    loop = asyncio.get_event_loop()

    def _extract():
        parts = []

        # ── Layer 1: JSON-LD / Schema.org (best for event/product sites) ──
        try:
            import json as _json
            import re as _re
            ld_blocks = _re.findall(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                html[:200000], _re.DOTALL | _re.IGNORECASE
            )
            events_text = []
            for block in ld_blocks:
                try:
                    data = _json.loads(block.strip())
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        _type = item.get("@type", "")
                        if _type in ("Event", "MusicEvent", "TheaterEvent",
                                     "Festival", "SocialEvent", "ScreeningEvent"):
                            name = item.get("name", "")
                            date = item.get("startDate", "")
                            loc = item.get("location", {})
                            if isinstance(loc, dict):
                                loc_name = loc.get("name", "")
                                loc_addr = loc.get("address", "")
                                if isinstance(loc_addr, dict):
                                    loc_addr = loc_addr.get("streetAddress", "")
                            else:
                                loc_name = str(loc)
                                loc_addr = ""
                            price = ""
                            offers = item.get("offers", {})
                            if isinstance(offers, dict):
                                p = offers.get("price", "")
                                if p:
                                    currency = offers.get("priceCurrency", "EUR")
                                    price = f" — {p} {currency}"
                            elif isinstance(offers, list) and offers:
                                p = offers[0].get("price", "")
                                if p:
                                    currency = offers[0].get("priceCurrency", "EUR")
                                    price = f" — ab {p} {currency}"
                            desc = item.get("description", "")[:200]
                            if name:
                                line = f"• {name}"
                                if date:
                                    line += f" | {date[:16]}"
                                if loc_name:
                                    line += f" | {loc_name}"
                                if loc_addr:
                                    line += f", {loc_addr}"
                                if price:
                                    line += price
                                if desc:
                                    line += f"\n  {desc}"
                                events_text.append(line)
                        elif _type == "ItemList":
                            for elem in item.get("itemListElement", []):
                                inner = elem.get("item", elem)
                                if inner.get("@type", "") in ("Event", "MusicEvent",
                                        "TheaterEvent", "Festival"):
                                    name = inner.get("name", "")
                                    date = inner.get("startDate", "")
                                    if name:
                                        line = f"• {name}"
                                        if date:
                                            line += f" | {date[:16]}"
                                        events_text.append(line)
                except (_json.JSONDecodeError, TypeError, KeyError):
                    continue

            if events_text:
                ld_text = "Strukturierte Event-Daten:\n" + "\n".join(events_text[:30])
                parts.append(ld_text)
                log.info(f"JSON-LD: {len(events_text)} events from {_domain(url)}")
        except Exception as e:
            log.debug(f"JSON-LD extraction failed: {e}")

        # ── Layer 2: trafilatura (general text extraction) ──
        try:
            import trafilatura
            text = trafilatura.extract(
                html, include_comments=False, include_tables=True,
                favor_precision=True,
            )
            # If too thin (<500 chars), retry with recall mode
            if not text or len(text) < 500:
                text_recall = trafilatura.extract(
                    html, include_comments=False, include_tables=True,
                    favor_recall=True,
                )
                if text_recall and len(text_recall) > len(text or ""):
                    text = text_recall
            if text and len(text) > 50:
                parts.append(text[:max_chars])
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"trafilatura: {e}")

        # ── Layer 3: basic HTML strip (last resort) ──
        if not parts:
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
                ext.feed(html[:80000])
                text = "\n".join(ext.p)
                if text and len(text) > 50:
                    parts.append(text[:max_chars])
            except Exception:
                pass

        # Combine all layers
        if parts:
            combined = "\n\n".join(parts)
            return combined[:max_chars]
        return None

    text = await loop.run_in_executor(None, _extract)
    if text:
        log.info(f"Extracted {len(text)} chars (limit={max_chars}) from {_domain(url)}")
    return text


async def fetch_pages(results: list[SearchResult], max_pages: int = MAX_FULL_TEXT_PAGES) -> int:
    """Fetch full text for top N results in parallel. Returns count fetched.
    
    Uses a semaphore to limit concurrent downloads (prevents C-level crashes
    from too many simultaneous httpx/SSL connections).
    If Playwright is available, retries pages that got <300 chars with JS rendering.
    """
    to_fetch = [
        r for r in results[:max_pages + 2]
        if r.url and not r.full_text and r.domain not in SKIP_DOMAINS
    ][:max_pages]

    if not to_fetch:
        return 0

    start = time.time()
    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def _fetch_with_sem(url: str) -> Optional[str]:
        async with sem:
            return await fetch_full_text(url)

    tasks = [_fetch_with_sem(r.url) for r in to_fetch]
    texts = await asyncio.gather(*tasks, return_exceptions=True)

    fetched = 0
    thin_results = []  # Results with <300 chars → candidates for Playwright retry
    for result, text in zip(to_fetch, texts):
        if isinstance(text, str) and text:
            result.full_text = text
            fetched += 1
            if len(text) < 300:
                thin_results.append(result)

    log.info(f"Full-text: {fetched}/{len(to_fetch)} pages, {(time.time()-start)*1000:.0f}ms")

    # ── Playwright retry for thin results ──
    if thin_results and _PLAYWRIGHT_AVAILABLE is not False:
        # Only retry up to 2 pages (Playwright is slow, ~3-5s each)
        pw_candidates = thin_results[:2]
        log.info(f"Playwright retry: {len(pw_candidates)} thin pages "
                 f"({[r.domain for r in pw_candidates]})")
        
        pw_start = time.time()
        pw_tasks = [fetch_full_text(r.url, use_playwright=True) for r in pw_candidates]
        pw_texts = await asyncio.gather(*pw_tasks, return_exceptions=True)
        
        pw_improved = 0
        for result, text in zip(pw_candidates, pw_texts):
            if isinstance(text, str) and text and len(text) > len(result.full_text or ""):
                result.full_text = text
                pw_improved += 1
        
        if pw_improved:
            log.info(f"Playwright retry: improved {pw_improved}/{len(pw_candidates)} pages, "
                     f"{(time.time()-pw_start)*1000:.0f}ms")

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
                "1. Extrahiere KONKRETE Fakten, Daten, Events, Namen, Termine aus den Quellen\n"
                "2. Priorisiere Studien und Reports über Zeitungsartikel\n"
                "3. Synthetisiere zu einer kohärenten Analyse — NICHT Quelle für Quelle zusammenfassen\n"
                "4. Strukturiere nach thematischen Aspekten, nicht nach Quellen\n"
                "5. Identifiziere Konsens und Widersprüche zwischen den Quellen\n"
                "6. Gib quantitative Aussagen (Spannbreiten) statt vager Qualifizierungen\n"
                "7. Quellenangaben als Nummern: [1], [2], [3] — NIEMALS URLs oder Seitennamen inline\n"
                "8. NIEMALS den Nutzer auf Webseiten verweisen — extrahiere die Infos SELBST\n"
            )
        else:
            parts.append(
                f"[WEB RESEARCH — {ts}]\n"
                f"Question: {query}\n\n"
                "ANALYSIS INSTRUCTIONS:\n"
                "1. Extract CONCRETE facts, data, events, names, dates from sources\n"
                "2. Prioritize studies and reports over news articles\n"
                "3. Synthesize into a coherent analysis — do NOT summarize source by source\n"
                "4. Structure by thematic aspects, not by sources\n"
                "5. Identify consensus and contradictions between sources\n"
                "6. Provide quantitative ranges instead of vague qualifiers\n"
                "7. Cite sources as numbers: [1], [2], [3] — NEVER inline URLs or site names\n"
                "8. NEVER refer the user to websites — extract the information YOURSELF\n"
            )
    elif depth == "deep":
        if is_de:
            parts.append(
                f"[WEBRECHERCHE — {ts}]\n"
                f"Frage: {query}\n\n"
                "Nutze die folgenden Quellen für eine fundierte Antwort.\n"
                "WICHTIG:\n"
                "- Extrahiere KONKRETE Informationen (Namen, Adressen, Telefon, Preise, Termine).\n"
                "- KOMBINIERE Daten aus verschiedenen Quellen zum selben Eintrag.\n"
                "- Verweise den Nutzer NIEMALS auf Webseiten — liefere die Inhalte SELBST.\n"
                "- Quellenangaben als Nummern: [1], [2], [3]\n"
            )
        else:
            parts.append(
                f"[WEB RESEARCH — {ts}]\n"
                f"Question: {query}\n\n"
                "Use these sources for a well-founded answer.\n"
                "IMPORTANT:\n"
                "- Extract CONCRETE information (names, addresses, phone, prices, dates).\n"
                "- MERGE data from different sources about the same entity.\n"
                "- NEVER refer the user to websites — provide the content YOURSELF.\n"
                "- Cite sources as numbers: [1], [2], [3]\n"
            )
    else:  # snippets
        parts.append(
            f"[{'WEBRECHERCHE' if is_de else 'WEB RESEARCH'} — {ts}]\n"
            f"{'Frage' if is_de else 'Question'}: {query}\n\n"
            + ("Beantworte anhand der folgenden Quellen. Extrahiere konkrete Fakten.\n"
               if is_de else
               "Answer based on these sources. Extract concrete facts.\n")
        )

    # Filter, sort by quality, and cap sources (shared with build_source_footer)
    filtered_results = prepare_sources(results, depth, query=query)
    
    # Tell model exactly how many sources it has
    _n_sources = len(filtered_results)
    if is_de:
        parts.append(
            f"\nDu hast {_n_sources} Quellen ([1]-[{_n_sources}]). "
            f"Verwende NUR [1]-[{_n_sources}] als Referenzen. "
            f"ERFINDE KEINE zusätzlichen Quellen oder URLs. "
            f"Erstelle KEIN eigenes Quellenverzeichnis am Ende.\n"
        )
    else:
        parts.append(
            f"\nYou have {_n_sources} sources ([1]-[{_n_sources}]). "
            f"Use ONLY [1]-[{_n_sources}] as references. "
            f"Do NOT invent additional sources or URLs. "
            f"Do NOT create your own source list at the end.\n"
        )

    total_content_chars = 0
    sources_with_real_content = 0  # full_text > 500 chars
    for i, r in enumerate(filtered_results, 1):
        block = f"\n── Quelle [{i}]: {r.title} ──\n"
        block += f"URL: {r.url}\n"
        if r.full_text:
            block += f"\n{r.full_text}\n"
            total_content_chars += len(r.full_text)
            if len(r.full_text) > 500:
                sources_with_real_content += 1
        elif r.snippet:
            block += f"{r.snippet}\n"
            total_content_chars += len(r.snippet)
        parts.append(block)

    # Content quality gate: if very little actual content was extracted,
    # instruct model to supplement with training knowledge
    avg_chars_per_source = total_content_chars / max(len(filtered_results), 1)
    _thin_content = (
        total_content_chars < 2000
        or avg_chars_per_source < 300
        or sources_with_real_content == 0
    )
    if _thin_content:
        if is_de:
            parts.append(
                f"\n⚠️ HINWEIS: Die {_n_sources} heruntergeladenen Webseiten lieferten wenig extrahierbaren Text "
                "(vermutlich JavaScript-basierte Seiten). "
                "Du MUSST trotzdem eine VOLLSTÄNDIGE und KONKRETE Antwort geben. "
                "NUTZE DEIN TRAININGSWISSEN um die Frage zu beantworten: "
                "Liste konkrete Events, Termine, Veranstaltungsorte, Sehenswürdigkeiten etc. auf. "
                "Informationen aus deinem Trainingswissen OHNE [N]-Referenz schreiben — "
                f"[1]-[{_n_sources}] NUR für Infos die tatsächlich in den Quellen oben stehen. "
                "Eine Antwort die nur Webseiten auflistet oder erfundene URLs enthält ist NICHT akzeptabel.\n"
            )
        else:
            parts.append(
                f"\n⚠️ NOTE: The {_n_sources} downloaded pages yielded very little extractable text "
                "(likely JavaScript-heavy sites). "
                "You MUST still provide a COMPLETE and CONCRETE answer. "
                "USE YOUR TRAINING KNOWLEDGE to answer the question. "
                "Write information from training knowledge WITHOUT [N] references — "
                f"use [1]-[{_n_sources}] ONLY for info actually present in the sources above. "
                "An answer that only lists websites or contains made-up URLs is NOT acceptable.\n"
            )
        log.info(f"Content quality gate triggered: {total_content_chars} chars, "
                 f"avg {avg_chars_per_source:.0f}/source, "
                 f"{sources_with_real_content} rich sources "
                 f"→ added training knowledge instruction")

    return "\n".join(parts)


def filter_irrelevant_results(results: list[SearchResult],
                              query: str = "") -> list[SearchResult]:
    """Filter out obviously irrelevant search results (wrong language domains, spam, etc.).
    
    If query is provided, also filters results with low topical relevance.
    This function MUST be used by both build_context() and build_source_footer()
    to ensure consistent numbering between the LLM context and the source footer.
    """
    from urllib.parse import urlparse
    
    _IRRELEVANT_TLDS = {".ru", ".pro", ".ua"}
    _IRRELEVANT_DOMAINS = {
        "hltv.org", "mail.ru", "tv.mail.ru", "kinodraiv.pro",
        "o-politico.ru", "pndexam.ru",
    }
    # Low-quality content domains (lifestyle blogs, generic listicles)
    _LOW_QUALITY_DOMAINS = {
        "pinterest.com", "pinterest.de", "instagram.com",
        "facebook.com", "twitter.com", "x.com",
        "tiktok.com", "youtube.com",  # video platforms (no text)
    }
    
    # Extract significant query terms for relevance check
    _query_terms = set()
    if query:
        import re as _re_fir
        # Include 3+ char words (catches DNA, RNA, UV, etc.)
        _words = _re_fir.findall(r'\b\w{3,}\b', query.lower())
        # Remove common stop words (language-agnostic: very common function words)
        _stop = {"welche", "welcher", "welchem", "dinge", "sachen",
                 "things", "which", "what", "does", "have", "some",
                 "eine", "einem", "einen", "einer", "eines",
                 "the", "that", "this", "with", "from", "into",
                 "nicht", "noch", "auch", "oder", "aber", "wenn",
                 "wie", "was", "wer", "warum", "wann", "wohin",
                 "gibt", "sind", "werden", "kann", "hat", "ist",
                 "wird", "sein", "war", "were", "been", "being",
                 "die", "der", "das", "des", "dem", "den",
                 "und", "für", "von", "bei", "aus", "auf",
                 "mit", "zum", "zur", "ins", "ans", "ums",
                 "and", "for", "the", "are", "can", "how"}
        _query_terms = {w for w in _words if w not in _stop}
    
    filtered = []
    seen_urls = set()
    for r in results:
        # Deduplicate
        url_key = r.url.rstrip("/").lower()
        if url_key in seen_urls or not r.url:
            continue
        
        # Filter irrelevant domains
        try:
            domain = urlparse(r.url).hostname or ""
            tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
            if tld in _IRRELEVANT_TLDS or domain in _IRRELEVANT_DOMAINS:
                continue
            if domain in _LOW_QUALITY_DOMAINS:
                log.debug(f"Relevance filter: skip low-quality domain {domain}")
                continue
        except Exception:
            pass
        
        # Relevance check: do key query terms appear in title+snippet?
        if _query_terms and len(_query_terms) >= 2:
            _source_text = f"{r.title} {r.snippet}".lower()
            _matched = sum(1 for t in _query_terms if t in _source_text)
            # Need at least 2 term matches, or 40% of terms, whichever is lower
            _min_matches = min(2, max(1, int(len(_query_terms) * 0.4)))
            if _matched < _min_matches:
                log.info(f"Relevance filter: skip {r.domain} "
                         f"(matched={_matched}/{len(_query_terms)}, need≥{_min_matches}, "
                         f"title='{r.title[:60]}')")
                continue
        
        seen_urls.add(url_key)
        filtered.append(r)
    
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  Source Footer Builder
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_sources(results: list[SearchResult], depth: str = "deep",
                    query: str = "") -> list[SearchResult]:
    """Filter, deduplicate, sort by quality, remove stale results, and cap sources.
    
    MUST be used by both build_context() and build_source_footer()
    to ensure consistent numbering.
    """
    filtered = filter_irrelevant_results(results, query=query)

    # ── Stale result filter ──
    # DDG sometimes returns old results despite time_filter.
    # Detect and deprioritize results with obviously old dates (>1 year old).
    current_year = datetime.now().year
    stale_years = {str(y) for y in range(2018, current_year - 1)}  # 2018-2024 are stale in 2026
    
    fresh = []
    stale = []
    for r in filtered:
        text_to_check = f"{r.title} {r.snippet} {r.url}"
        # Check if result mentions a stale year prominently
        is_stale = False
        for year in stale_years:
            if year in text_to_check:
                # But not if current year is also mentioned (could be a comparison)
                if str(current_year) not in text_to_check and str(current_year - 1) not in text_to_check:
                    is_stale = True
                    break
        if is_stale:
            stale.append(r)
        else:
            fresh.append(r)
    
    if stale:
        log.info(f"Stale filter: {len(stale)} results from old years removed "
                 f"({[r.domain for r in stale]})")
    
    # Use fresh results; only add stale if we'd have too few
    if len(fresh) >= 2:
        filtered = fresh
    else:
        filtered = fresh + stale  # Keep stale as backup

    # ── Cap: too many thin sources overwhelm the model ──
    _max_sources = {
        "snippets": 5,
        "deep": 6,
        "thorough": 8,
    }.get(depth, 6)

    if len(filtered) > _max_sources:
        # Sort: sources with full_text first (by length desc), then snippet-only
        # Use (length, url) as sort key for deterministic ordering
        filtered.sort(
            key=lambda r: (
                len(r.full_text) if r.full_text else len(r.snippet or ""),
                r.url,  # secondary key for determinism
            ),
            reverse=True,
        )
        dropped = len(filtered) - _max_sources
        filtered = filtered[:_max_sources]
        log.info(f"Source cap: kept {_max_sources} best sources, dropped {dropped} thin ones")

    return filtered


def build_source_footer(results: list[SearchResult], language: str = "de",
                        depth: str = "deep", query: str = "") -> str:
    """
    Build a numbered source list from search results.
    Uses the same prepare_sources() as build_context()
    to ensure numbering matches exactly.
    """
    if not results:
        return ""

    # Use the SAME filter+cap as build_context — critical for number consistency
    filtered = prepare_sources(results, depth, query=query)
    if not filtered:
        return ""

    is_de = language.startswith("de")
    lines = ["\n---", f"{'Quellen' if is_de else 'Sources'}:"]
    for idx, r in enumerate(filtered, 1):
        title_part = f" — {r.title}" if r.title else ""
        lines.append(f"[{idx}] {r.url}{title_part}")

    return "\n".join(lines) if len(lines) > 2 else ""


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
