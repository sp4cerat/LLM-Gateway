"""
LLM Gateway - Rate Limiter & Budget Guard
Token-aware rate limiting with three-tier kill switch.
"""

import logging
from datetime import datetime, timedelta, date
from collections import defaultdict
from threading import Lock
from typing import Optional
from models import BudgetStatus, GatewayConfig
from metrics import metrics

log = logging.getLogger("gateway.rate_limiter")


class RateLimiter:
    """
    Token-aware rate limiter with per-tier limits.
    Tracks RPM, TPM, and daily budget per tier.
    """

    def __init__(self, config: GatewayConfig):
        self._lock = Lock()
        self.limits = {
            "local": {
                "rpm": config.rate_limits.local_rpm,
                "tpm": config.rate_limits.local_tpm,
                "daily_usd": 0,
                "burst": 10,
            },
            "cheap": {
                "rpm": config.rate_limits.cheap_rpm,
                "tpm": config.rate_limits.cheap_tpm,
                "daily_usd": 5,
                "burst": 8,
            },
            "cheap_plus": {
                "rpm": config.rate_limits.cheap_rpm,
                "tpm": config.rate_limits.cheap_tpm,
                "daily_usd": 10,
                "burst": 6,
            },
            "medium": {
                "rpm": config.rate_limits.medium_rpm,
                "tpm": config.rate_limits.medium_tpm,
                "daily_usd": 20,
                "burst": 5,
            },
            "premium": {
                "rpm": config.rate_limits.premium_rpm,
                "tpm": config.rate_limits.premium_tpm,
                "daily_usd": 50,
                "burst": 3,
            },
        }
        self.minute_requests: dict[str, list[datetime]] = defaultdict(list)
        self.minute_tokens: dict[str, int] = defaultdict(int)
        self.daily_spend: dict[str, float] = defaultdict(float)
        self.last_reset = datetime.now()

    def check(self, tier: str, estimated_tokens: int) -> tuple[bool, str, float]:
        """
        Check rate limits for a tier.
        Returns: (allowed, reason, delay_seconds)
        """
        with self._lock:
            now = datetime.now()
            limits = self.limits.get(tier, self.limits["cheap"])

            # Daily reset
            if now.date() != self.last_reset.date():
                self.daily_spend.clear()
                self.minute_tokens.clear()
                self.last_reset = now

            # Cleanup old timestamps (>1 minute)
            cutoff = now - timedelta(minutes=1)
            self.minute_requests[tier] = [
                ts for ts in self.minute_requests[tier] if ts > cutoff
            ]

            # RPM check
            current_rpm = len(self.minute_requests[tier])
            if current_rpm >= limits["rpm"]:
                if current_rpm < limits["rpm"] + limits["burst"]:
                    return (True, "burst_allowed", 0)
                metrics.increment("rate_limit_hits", tags={"tier": tier, "type": "rpm"})
                return (False, "rpm_exceeded", 60)

            # TPM check
            if self.minute_tokens[tier] + estimated_tokens > limits["tpm"]:
                metrics.increment("rate_limit_hits", tags={"tier": tier, "type": "tpm"})
                return (False, "tpm_exceeded", 30)

            # Daily budget check (for premium)
            if limits["daily_usd"] > 0:
                estimated_cost = self._estimate_cost(tier, estimated_tokens)
                if self.daily_spend[tier] + estimated_cost > limits["daily_usd"]:
                    metrics.increment("rate_limit_hits", tags={"tier": tier, "type": "daily_budget"})
                    return (False, "daily_budget_exceeded", 0)

            return (True, "ok", 0)

    def record(self, tier: str, tokens_used: int, cost_usd: float):
        """Record a successful request."""
        with self._lock:
            self.minute_requests[tier].append(datetime.now())
            self.minute_tokens[tier] += tokens_used
            self.daily_spend[tier] += cost_usd

    def _estimate_cost(self, tier: str, tokens: int) -> float:
        prices = {
            "local": 0,
            "cheap": 0.10 / 1_000_000,
            "medium": 2.00 / 1_000_000,
            "premium": 9.00 / 1_000_000,
        }
        return tokens * prices.get(tier, 0)


class BudgetGuard:
    """
    Three-tier budget system with global kill switch.
    Soft → Warning, Medium → Throttle, Hard → Kill Premium
    """

    def __init__(self, config: GatewayConfig):
        self._lock = Lock()
        self.limits = {
            "soft": config.budget.daily_soft_limit,
            "medium": config.budget.daily_medium_limit,
            "hard": config.budget.daily_hard_limit,
        }
        self.today_spend = 0.0
        self.premium_disabled = False
        self.last_reset = date.today()

    def check(self, tier: str, estimated_cost: float) -> dict:
        """
        Check budget for a request.
        Returns: {"allowed": bool, "delay": float, "reason": str}
        """
        with self._lock:
            today = date.today()
            if today != self.last_reset:
                self.today_spend = 0.0
                self.premium_disabled = False
                self.last_reset = today
                log.info("Daily budget reset")

            projected = self.today_spend + estimated_cost

            # Kill switch active?
            if self.premium_disabled:
                if tier == "premium":
                    return {"allowed": False, "delay": 0, "reason": "kill_switch_active"}
                return {"allowed": True, "delay": 0, "reason": "non_premium_allowed"}

            # Hard limit → Activate kill switch
            if projected > self.limits["hard"]:
                self.premium_disabled = True
                log.critical(f"KILL SWITCH ACTIVATED: ${projected:.2f} > ${self.limits['hard']}")
                metrics.increment("kill_switch_activated")
                return {"allowed": False, "delay": 0, "reason": "hard_limit_kill_switch"}

            # Medium limit → Throttle
            if projected > self.limits["medium"]:
                log.error(f"Budget THROTTLE: ${projected:.2f} > ${self.limits['medium']}")
                return {"allowed": True, "delay": 5.0, "reason": "throttle_medium_limit"}

            # Soft limit → Warning
            if projected > self.limits["soft"]:
                log.warning(f"Budget WARNING: ${projected:.2f} > ${self.limits['soft']}")

            return {"allowed": True, "delay": 0, "reason": "ok"}

    def record_spend(self, cost: float):
        """Record actual spend."""
        with self._lock:
            self.today_spend += cost
            metrics.gauge("daily_spend_usd", self.today_spend)

    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        with self._lock:
            if self.premium_disabled:
                level = "kill"
            elif self.today_spend > self.limits["medium"]:
                level = "throttle"
            elif self.today_spend > self.limits["soft"]:
                level = "warning"
            else:
                level = "ok"

            return BudgetStatus(
                today_spend=self.today_spend,
                soft_limit=self.limits["soft"],
                medium_limit=self.limits["medium"],
                hard_limit=self.limits["hard"],
                premium_disabled=self.premium_disabled,
                level=level,
            )

    def force_kill(self):
        """Manually activate kill switch."""
        with self._lock:
            self.premium_disabled = True
            log.critical("KILL SWITCH manually activated")

    def force_reset(self):
        """Manually reset kill switch."""
        with self._lock:
            self.premium_disabled = False
            log.info("Kill switch manually reset")


class IdempotencyGuard:
    """
    Prevents duplicate API calls during retries/timeouts.
    Caches response for identical requests for 5 minutes.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._lock = Lock()
        self.cache: dict[str, tuple[dict, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def get_key(self, messages: list, model: str = "", temperature: float = 1) -> str:
        import hashlib, json
        content = json.dumps({
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check(self, key: str) -> Optional[dict]:
        """Check if request was already processed."""
        self._cleanup()
        with self._lock:
            if key in self.cache:
                response, _ = self.cache[key]
                metrics.increment("idempotency_hits")
                return response
        return None

    def store(self, key: str, response: dict):
        """Store response for key."""
        with self._lock:
            self.cache[key] = (response, datetime.now())

    def _cleanup(self):
        now = datetime.now()
        with self._lock:
            expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
            for k in expired:
                del self.cache[k]
