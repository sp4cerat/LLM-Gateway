"""
LLM Gateway - Metrics & Monitoring
Lightweight metrics collection without external dependencies.
"""

import time
import threading
import logging
from collections import defaultdict
from datetime import datetime, date
from typing import Optional

log = logging.getLogger("gateway.metrics")


class Metrics:
    """
    Thread-safe in-memory metrics collector.
    Tracks counters, gauges, histograms for gateway operations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()
        self._today: date = date.today()

    def _check_day_reset(self):
        """Reset daily metrics at midnight."""
        today = date.today()
        if today != self._today:
            with self._lock:
                self._today = today
                # Reset daily counters
                keys_to_reset = [k for k in self._counters if k.startswith("daily_")]
                for k in keys_to_reset:
                    self._counters[k] = 0
                # Reset daily histograms
                hist_to_reset = [k for k in self._histograms if k.startswith("daily_")]
                for k in hist_to_reset:
                    self._histograms[k] = []
                log.info("Daily metrics reset")

    def increment(self, name: str, value: float = 1, tags: Optional[dict] = None):
        """Increment a counter."""
        self._check_day_reset()
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._counters[f"daily_{key}"] += value

    def gauge(self, name: str, value: float, tags: Optional[dict] = None):
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, tags: Optional[dict] = None):
        """Record a histogram observation."""
        self._check_day_reset()
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
            self._histograms[f"daily_{key}"].append(value)
            # Keep only last 10000 observations
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-5000:]

    def get_counter(self, name: str, tags: Optional[dict] = None) -> float:
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)

    def get_daily_counter(self, name: str, tags: Optional[dict] = None) -> float:
        self._check_day_reset()
        key = f"daily_{self._make_key(name, tags)}"
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, tags: Optional[dict] = None) -> float:
        key = self._make_key(name, tags)
        return self._gauges.get(key, 0)

    def get_histogram_stats(self, name: str, tags: Optional[dict] = None) -> dict:
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "count": n,
            "avg": sum(sorted_vals) / n,
            "p50": sorted_vals[int(n * 0.5)],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
        }

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def get_all_counters(self) -> dict[str, float]:
        with self._lock:
            return dict(self._counters)

    def get_all_gauges(self) -> dict[str, float]:
        with self._lock:
            return dict(self._gauges)

    def get_summary(self) -> dict:
        """Get a full metrics summary for the dashboard."""
        self._check_day_reset()

        total_requests = self.get_counter("requests_total")
        cache_exact_hits = self.get_counter("cache_hit", {"type": "exact"})
        cache_semantic_hits = self.get_counter("cache_hit", {"type": "semantic"})
        cache_misses = self.get_counter("cache_miss")
        total_cache_checks = cache_exact_hits + cache_semantic_hits + cache_misses

        tier_counts = {}
        for tier in ["local", "cheap", "premium"]:
            tier_counts[tier] = self.get_counter("requests_by_tier", {"tier": tier})

        total_tier = sum(tier_counts.values()) or 1

        return {
            "uptime_seconds": self.get_uptime(),
            "total_requests": int(total_requests),
            "daily_requests": int(self.get_daily_counter("requests_total")),
            "requests_by_tier": {k: int(v) for k, v in tier_counts.items()},
            "cache_hits": {
                "exact": int(cache_exact_hits),
                "semantic": int(cache_semantic_hits),
            },
            "cache_hit_rate": (cache_exact_hits + cache_semantic_hits) / max(total_cache_checks, 1),
            "premium_ratio": tier_counts.get("premium", 0) / total_tier,
            "daily_cost_usd": self.get_gauge("daily_spend_usd"),
            "latency": {
                tier: self.get_histogram_stats("request_latency_ms", {"tier": tier})
                for tier in ["local", "cheap", "premium", "cached"]
            },
            "errors_today": int(self.get_daily_counter("errors")),
            "policy_blocks_today": int(self.get_daily_counter("policy_blocks")),
            "rate_limit_hits_today": int(self.get_daily_counter("rate_limit_hits")),
            "router_latency": self.get_histogram_stats("router_latency_ms"),
        }

    @staticmethod
    def _make_key(name: str, tags: Optional[dict] = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# Global singleton
metrics = Metrics()
