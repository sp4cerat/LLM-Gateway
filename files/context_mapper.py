"""
LLM Gateway - Context Mapper
=============================
Builds a structured "map" of large contexts so the router sees the full picture
without consuming the entire token budget.

Pipeline (no LLM needed):
  1. Split text into chunks (~2000 tokens each)
  2. Extract headings/structure via regex
  3. Extract keywords via TF-IDF approximation
  4. Build JSON context map with offsets

The router receives:  query + context_map (compact)
Instead of:           query + full_document (expensive)
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from context import estimate_tokens

log = logging.getLogger("gateway.context_mapper")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: int
    start_char: int
    end_char: int
    start_token_est: int
    end_token_est: int
    headings: list[str] = field(default_factory=list)
    first_sentences: str = ""
    keywords: list[str] = field(default_factory=list)
    token_count: int = 0

    def to_summary(self) -> str:
        """Compact single-line summary for the router."""
        parts = [f"[Chunk {self.chunk_id}]"]
        if self.headings:
            parts.append(f"Headings: {'; '.join(self.headings)}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords[:8])}")
        if self.first_sentences:
            parts.append(f"Preview: {self.first_sentences[:120]}")
        parts.append(f"({self.token_count} tok)")
        return " | ".join(parts)


@dataclass
class ContextMap:
    """Structured representation of a large document."""
    total_chars: int = 0
    total_tokens_est: int = 0
    chunk_count: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    global_headings: list[str] = field(default_factory=list)
    document_type: str = "unknown"  # code, legal, prose, log, structured
    hash: str = ""

    def to_router_prompt(self, max_lines: int = 40) -> str:
        """
        Build a compact text representation for the router.
        This replaces sending the full document.
        """
        lines = [
            f"=== CONTEXT MAP ({self.total_tokens_est} tokens, {self.chunk_count} chunks, type={self.document_type}) ===",
        ]

        if self.global_headings:
            lines.append(f"Document structure: {' → '.join(self.global_headings[:15])}")
            lines.append("")

        for chunk in self.chunks:
            lines.append(chunk.to_summary())
            if len(lines) >= max_lines:
                lines.append(f"... ({self.chunk_count - chunk.chunk_id - 1} more chunks)")
                break

        lines.append("=== END CONTEXT MAP ===")
        return "\n".join(lines)

    def get_chunks_by_ids(self, chunk_ids: list[int], context_margin_chars: int = 500) -> list[dict]:
        """Retrieve specific chunks by ID with optional margin."""
        results = []
        for cid in chunk_ids:
            for chunk in self.chunks:
                if chunk.chunk_id == cid:
                    results.append({
                        "chunk_id": cid,
                        "start_char": max(0, chunk.start_char - context_margin_chars),
                        "end_char": chunk.end_char + context_margin_chars,
                    })
                    break
        return results


# ─── Heading Extraction ──────────────────────────────────────────────────────

# Patterns for common heading styles
HEADING_PATTERNS = [
    # Markdown headers
    re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
    # Numbered sections: "1.", "1.1", "1.1.1", "Section 1:"
    re.compile(r'^(\d+(?:\.\d+)*\.?)\s+(.+)$', re.MULTILINE),
    re.compile(r'^(?:Section|Kapitel|Abschnitt|Chapter|Article|§)\s*(\d+[.\d]*)[:\s]*(.+)$',
               re.MULTILINE | re.IGNORECASE),
    # ALLCAPS headings (min 3 words to avoid false positives)
    re.compile(r'^([A-ZÄÖÜ][A-ZÄÖÜ\s]{8,})$', re.MULTILINE),
    # Legal clause IDs: "(a)", "(i)", "(1)"
    re.compile(r'^\(([a-z]|\d+|[ivxlc]+)\)\s+(.{10,80})$', re.MULTILINE),
]


def extract_headings(text: str) -> list[str]:
    """Extract structural headings from text using regex patterns."""
    headings = []
    for pattern in HEADING_PATTERNS:
        for match in pattern.finditer(text):
            heading = match.group(0).strip()
            # Clean markdown hashes
            heading = re.sub(r'^#+\s*', '', heading)
            if 3 < len(heading) < 200:
                headings.append(heading)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for h in headings:
        normalized = h.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(h)

    return unique


# ─── Keyword Extraction ──────────────────────────────────────────────────────

# Common stopwords (EN + DE)
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "i", "we", "you", "they", "he", "she", "my",
    "your", "our", "their", "his", "her", "for", "to", "from", "with",
    "in", "on", "at", "by", "of", "and", "or", "but", "not", "no",
    "if", "then", "else", "when", "where", "which", "who", "what", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "just", "also", "so", "as",
    # German
    "der", "die", "das", "ein", "eine", "und", "oder", "aber", "nicht",
    "ist", "sind", "war", "hat", "mit", "von", "für", "auf", "zu",
    "den", "dem", "des", "im", "am", "um", "aus", "bei", "nach",
    "über", "unter", "vor", "zwischen", "durch", "ohne", "gegen",
    "wird", "werden", "wurde", "können", "müssen", "sollen", "darf",
    "sich", "auch", "noch", "schon", "nur", "dann", "wenn", "weil",
    "dass", "dieser", "diese", "dieses", "jeder", "jede", "jedes",
})

# Token pattern: alphanumeric words, allowing hyphens and underscores
WORD_PATTERN = re.compile(r'\b[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß0-9_-]{2,}\b')


def extract_keywords(text: str, top_n: int = 12) -> list[str]:
    """
    Fast keyword extraction using term frequency with stopword filtering.
    No external dependencies (spaCy/KeyBERT not needed for routing).
    """
    words = WORD_PATTERN.findall(text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    # Term frequency
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Boost multi-word patterns (bigrams from original text)
    bigrams = _extract_bigrams(text)
    for bg in bigrams:
        freq[bg] = freq.get(bg, 0) + 3  # Boost compound terms

    # Sort by frequency, return top N
    sorted_kw = sorted(freq.items(), key=lambda x: -x[1])
    return [kw for kw, _ in sorted_kw[:top_n]]


def _extract_bigrams(text: str) -> list[str]:
    """Extract meaningful bigrams (compound terms like 'rate limit', 'API key')."""
    words = WORD_PATTERN.findall(text.lower())
    bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in STOPWORDS and w2 not in STOPWORDS and len(w1) > 2 and len(w2) > 2:
            bigrams.append(f"{w1} {w2}")

    # Only keep recurring bigrams
    freq: dict[str, int] = {}
    for bg in bigrams:
        freq[bg] = freq.get(bg, 0) + 1

    return [bg for bg, count in freq.items() if count >= 2]


# ─── Document Type Detection ─────────────────────────────────────────────────

def detect_document_type(text: str) -> str:
    """Classify document type heuristically for routing hints."""
    sample = text[:5000]

    code_indicators = sum([
        sample.count('{') + sample.count('}') > 20,
        sample.count('def ') + sample.count('function ') + sample.count('class ') > 3,
        sample.count('import ') + sample.count('#include') + sample.count('require(') > 3,
        bool(re.search(r'(if|for|while|return)\s*[\(\{]', sample)),
    ])
    if code_indicators >= 2:
        return "code"

    legal_indicators = sum([
        bool(re.search(r'§\s*\d+', sample)),
        bool(re.search(r'\b(hereby|whereas|thereof|pursuant|notwithstanding)\b', sample, re.I)),
        bool(re.search(r'\b(Vertrag|Klausel|Haftung|Gewährleistung|Vertragspartei)\b', sample)),
        bool(re.search(r'\(\s*[a-z]\s*\)\s+\w', sample)),
    ])
    if legal_indicators >= 2:
        return "legal"

    log_indicators = sum([
        bool(re.search(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', sample)),
        bool(re.search(r'\b(ERROR|WARN|INFO|DEBUG|CRITICAL)\b', sample)),
        bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', sample)),
    ])
    if log_indicators >= 2:
        return "log"

    if sample.count('|') > 10 and sample.count('\n') > 5:
        return "structured"

    return "prose"


# ─── Main Context Mapper ────────────────────────────────────────────────────

class ContextMapper:
    """
    Builds a compact "map" of a document for the router.

    Usage:
        mapper = ContextMapper(chunk_size_tokens=2000)
        ctx_map = mapper.build(full_document_text)
        router_prompt = ctx_map.to_router_prompt()
    """

    def __init__(self, chunk_size_tokens: int = 2000, overlap_tokens: int = 100):
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def build(self, text: str) -> ContextMap:
        """Build a context map from raw text."""
        if not text or len(text.strip()) < 50:
            return ContextMap()

        total_tokens = estimate_tokens(text)
        doc_hash = hashlib.md5(text[:10000].encode()).hexdigest()[:12]
        doc_type = detect_document_type(text)
        global_headings = extract_headings(text)

        # Split into chunks
        chunks = self._split_into_chunks(text)

        # Enrich each chunk
        for chunk in chunks:
            chunk_text = text[chunk.start_char:chunk.end_char]
            chunk.headings = extract_headings(chunk_text)
            chunk.keywords = extract_keywords(chunk_text, top_n=8)
            chunk.first_sentences = self._extract_first_sentences(chunk_text, n=2)
            chunk.token_count = estimate_tokens(chunk_text)

        return ContextMap(
            total_chars=len(text),
            total_tokens_est=total_tokens,
            chunk_count=len(chunks),
            chunks=chunks,
            global_headings=global_headings,
            document_type=doc_type,
            hash=doc_hash,
        )

    def _split_into_chunks(self, text: str) -> list[Chunk]:
        """Split text into chunks, preferring natural boundaries."""
        chunks = []
        # Approximate chars per token
        chars_per_token = max(1, len(text) / max(1, estimate_tokens(text)))
        chunk_size_chars = int(self.chunk_size_tokens * chars_per_token)
        overlap_chars = int(self.overlap_tokens * chars_per_token)

        pos = 0
        chunk_id = 0
        token_offset = 0

        while pos < len(text):
            end = min(pos + chunk_size_chars, len(text))

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', pos + chunk_size_chars // 2, end + 200)
                if para_break > pos:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sent_break = max(
                        text.rfind('. ', pos + chunk_size_chars // 2, end + 100),
                        text.rfind('.\n', pos + chunk_size_chars // 2, end + 100),
                    )
                    if sent_break > pos:
                        end = sent_break + 2

            chunk_text = text[pos:end]
            chunk_tokens = estimate_tokens(chunk_text)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                start_char=pos,
                end_char=end,
                start_token_est=token_offset,
                end_token_est=token_offset + chunk_tokens,
                token_count=chunk_tokens,
            ))

            token_offset += chunk_tokens
            pos = end - overlap_chars if end < len(text) else end
            chunk_id += 1

        return chunks

    def _extract_first_sentences(self, text: str, n: int = 2) -> str:
        """Extract first N sentences from chunk."""
        text = text.strip()
        sentences = re.split(r'(?<=[.!?])\s+', text[:500])
        result = ' '.join(sentences[:n])
        return result[:200]

    def retrieve_chunks(self, full_text: str, context_map: ContextMap,
                        chunk_ids: list[int], margin_chars: int = 400) -> str:
        """
        Retrieve specific chunks from the original text by their IDs.
        Used when the router says "I need chunks [12, 13, 41]".
        """
        if not chunk_ids:
            return ""

        parts = []
        for cid in sorted(set(chunk_ids)):
            for chunk in context_map.chunks:
                if chunk.chunk_id == cid:
                    start = max(0, chunk.start_char - margin_chars)
                    end = min(len(full_text), chunk.end_char + margin_chars)
                    parts.append(f"--- Chunk {cid} (tokens ~{chunk.token_count}) ---\n")
                    parts.append(full_text[start:end])
                    parts.append("\n\n")
                    break

        return "".join(parts)


# ─── Singleton ───────────────────────────────────────────────────────────────

context_mapper = ContextMapper()
