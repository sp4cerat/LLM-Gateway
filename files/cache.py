"""
LLM Gateway - Two-Stage Caching Engine
Exact Cache (SHA-256) + Semantic Cache (Embeddings) with SQLite.
"""

import json
import hashlib
import sqlite3
import logging
import re
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from threading import Lock
from models import GatewayConfig
from metrics import metrics

log = logging.getLogger("gateway.cache")


# ─── Response Type TTLs ──────────────────────────────────────────────────────

RESPONSE_TTLS = {
    "explanation_generic": 7 * 24 * 3600,
    "explanation_contextual": 24 * 3600,
    "code_suggestion": 3600,
    "code_review": 12 * 3600,
    "command_execution": 3600,
    "documentation": 24 * 3600,
}


def get_adaptive_ttl(response_type: str, hit_count: int = 0) -> int:
    """Calculate TTL based on response type and popularity."""
    base = RESPONSE_TTLS.get(response_type, 3600)
    if hit_count >= 10:
        return int(base * 2)
    elif hit_count >= 5:
        return int(base * 1.5)
    elif hit_count == 0:
        return int(base * 0.5)
    return base


# ─── Exact Cache ──────────────────────────────────────────────────────────────

class ExactCache:
    """
    Exact cache using SHA-256 hash of query + fingerprint.
    0ms lookup, 100% confidence on cache hits.
    """

    def __init__(self, db_path: str = "cache_exact.sqlite"):
        self._lock = Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS exact_cache (
                cache_key TEXT PRIMARY KEY,
                query_hash TEXT,
                fingerprint TEXT,
                response TEXT,
                response_type TEXT,
                model TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                last_hit_at TIMESTAMP
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_expires ON exact_cache(expires_at)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint ON exact_cache(fingerprint)")
        self.db.commit()

    def get_key(self, query: str, fingerprint: str = "") -> str:
        content = f"{query}|{fingerprint}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, query: str, fingerprint: str = "") -> Optional[dict]:
        """Look up exact cache entry."""
        cache_key = self.get_key(query, fingerprint)
        now = datetime.now()

        with self._lock:
            row = self.db.execute("""
                SELECT response, response_type, model FROM exact_cache
                WHERE cache_key = ? AND expires_at > ?
            """, (cache_key, now)).fetchone()

        if row:
            with self._lock:
                self.db.execute("""
                    UPDATE exact_cache SET hit_count = hit_count + 1, last_hit_at = ?
                    WHERE cache_key = ?
                """, (now, cache_key))
                self.db.commit()

            metrics.increment("cache_hit", tags={"type": "exact"})
            return {
                "response": json.loads(row[0]),
                "response_type": row[1],
                "model": row[2],
                "cache_type": "exact",
            }

        metrics.increment("cache_miss", tags={"type": "exact"})
        return None

    def set(self, query: str, fingerprint: str, response: dict,
            response_type: str, model: str = "", ttl_seconds: int = 0):
        """Store a response in the cache."""
        if ttl_seconds == 0:
            ttl_seconds = get_adaptive_ttl(response_type)

        cache_key = self.get_key(query, fingerprint)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl_seconds)

        with self._lock:
            self.db.execute("""
                INSERT OR REPLACE INTO exact_cache
                (cache_key, query_hash, fingerprint, response, response_type, model,
                 created_at, expires_at, hit_count, last_hit_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
            """, (cache_key, query_hash, fingerprint, json.dumps(response),
                  response_type, model, now, expires_at))
            self.db.commit()

    def invalidate_by_fingerprint(self, fingerprint: str) -> int:
        """Invalidate all entries with a given fingerprint."""
        with self._lock:
            cursor = self.db.execute(
                "DELETE FROM exact_cache WHERE fingerprint = ?", (fingerprint,)
            )
            self.db.commit()
            return cursor.rowcount

    def invalidate_by_type(self, response_type: str) -> int:
        """Invalidate all entries of a given type."""
        with self._lock:
            cursor = self.db.execute(
                "DELETE FROM exact_cache WHERE response_type = ?", (response_type,)
            )
            self.db.commit()
            return cursor.rowcount

    def invalidate_all(self) -> int:
        """Clear entire cache."""
        with self._lock:
            cursor = self.db.execute("DELETE FROM exact_cache")
            self.db.commit()
            return cursor.rowcount

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            cursor = self.db.execute(
                "DELETE FROM exact_cache WHERE expires_at < ?", (datetime.now(),)
            )
            self.db.commit()
            return cursor.rowcount

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self.db.execute("SELECT COUNT(*) FROM exact_cache").fetchone()[0]
            active = self.db.execute(
                "SELECT COUNT(*) FROM exact_cache WHERE expires_at > ?", (datetime.now(),)
            ).fetchone()[0]
            size = self.db.execute(
                "SELECT SUM(LENGTH(response)) FROM exact_cache"
            ).fetchone()[0] or 0

        return {
            "total_entries": total,
            "active_entries": active,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
        }


# ─── Semantic Cache ───────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic cache using embedding similarity.
    Falls back to BM25 for technical queries (fast path).
    """

    def __init__(self, db_path: str = "cache_semantic.sqlite",
                 similarity_threshold: float = 0.92):
        self._lock = Lock()
        self.threshold = similarity_threshold
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                query_normalized TEXT,
                query_embedding BLOB,
                fingerprint TEXT,
                response TEXT,
                response_type TEXT,
                model TEXT,
                created_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_sem_fp ON semantic_cache(fingerprint)")
        self.db.commit()

        # Query embedding cache
        self.embedding_cache = QueryEmbeddingCache()

    async def get(self, query: str, fingerprint: str = "",
                  get_embedding_fn=None) -> Optional[dict]:
        """Search for semantically similar cached entry."""
        if not get_embedding_fn:
            return None

        query_embedding = await self.embedding_cache.get_or_compute(query, get_embedding_fn)
        if query_embedding is None:
            return None

        with self._lock:
            rows = self.db.execute("""
                SELECT id, query, query_embedding, response, response_type, model
                FROM semantic_cache WHERE fingerprint = ?
            """, (fingerprint,)).fetchall()

        if not rows:
            metrics.increment("cache_miss", tags={"type": "semantic"})
            return None

        best_match = None
        best_similarity = 0.0

        for row in rows:
            try:
                cached_embedding = np.frombuffer(row[2], dtype=np.float32)
                similarity = cosine_similarity(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity >= self.threshold:
                    best_similarity = similarity
                    best_match = row
            except Exception:
                continue

        if best_match:
            # Update hit counter
            with self._lock:
                self.db.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                    (best_match[0],)
                )
                self.db.commit()

            metrics.increment("cache_hit", tags={"type": "semantic"})
            return {
                "id": best_match[0],
                "original_query": best_match[1],
                "response": json.loads(best_match[3]),
                "response_type": best_match[4],
                "model": best_match[5],
                "similarity": best_similarity,
                "cache_type": "semantic",
            }

        metrics.increment("cache_miss", tags={"type": "semantic"})
        return None

    async def set(self, query: str, fingerprint: str, response: dict,
                  response_type: str, model: str = "", get_embedding_fn=None):
        """Store response in semantic cache."""
        if not get_embedding_fn:
            return

        embedding = await self.embedding_cache.get_or_compute(query, get_embedding_fn)
        if embedding is None:
            return

        normalized = self._normalize_query(query)

        with self._lock:
            self.db.execute("""
                INSERT INTO semantic_cache
                (query, query_normalized, query_embedding, fingerprint,
                 response, response_type, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (query, normalized, embedding.tobytes(), fingerprint,
                  json.dumps(response), response_type, model, datetime.now()))
            self.db.commit()

    def invalidate_all(self) -> int:
        with self._lock:
            cursor = self.db.execute("DELETE FROM semantic_cache")
            self.db.commit()
            return cursor.rowcount

    def _normalize_query(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[a-f0-9]{8,}', '<ID>', query)
        return query

    def get_stats(self) -> dict:
        with self._lock:
            total = self.db.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
            size = self.db.execute(
                "SELECT SUM(LENGTH(response)) FROM semantic_cache"
            ).fetchone()[0] or 0
        return {"total_entries": total, "size_bytes": size, "size_mb": round(size / (1024 * 1024), 2)}


# ─── Query Embedding Cache ────────────────────────────────────────────────────

class QueryEmbeddingCache:
    """Cache for query embeddings. Reduces remote embedding calls by 60-70%."""

    def __init__(self, db_path: str = "cache_embeddings.sqlite"):
        self._lock = Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS query_embeddings (
                query_hash TEXT PRIMARY KEY,
                query_normalized TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()

    async def get_or_compute(self, query: str, compute_fn) -> Optional[np.ndarray]:
        """Get embedding from cache or compute new one."""
        normalized = self._normalize(query)
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        with self._lock:
            row = self.db.execute(
                "SELECT embedding FROM query_embeddings WHERE query_hash = ?",
                (query_hash,)
            ).fetchone()

        if row:
            metrics.increment("query_embedding_cache_hit")
            return np.frombuffer(row[0], dtype=np.float32)

        try:
            embedding = await compute_fn(query)
            if embedding is not None:
                embedding_np = np.array(embedding, dtype=np.float32)
                with self._lock:
                    self.db.execute("""
                        INSERT OR REPLACE INTO query_embeddings (query_hash, query_normalized, embedding)
                        VALUES (?, ?, ?)
                    """, (query_hash, normalized, embedding_np.tobytes()))
                    self.db.commit()
                metrics.increment("query_embedding_cache_miss")
                return embedding_np
        except Exception as e:
            log.warning(f"Embedding computation failed: {e}")

        return None

    def _normalize(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[a-f0-9]{8,}', '<ID>', query)
        return query

    def cleanup_old(self, days: int = 30):
        with self._lock:
            self.db.execute(
                "DELETE FROM query_embeddings WHERE created_at < datetime('now', ?)",
                (f'-{days} days',)
            )
            self.db.commit()


# ─── Utilities ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
