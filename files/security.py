"""
LLM Gateway - Security Module
Hard Policy Gate, authentication, IP rate limiting, and input validation.
"""

import re
import hashlib
import logging
import time
import unicodedata
from collections import defaultdict
from typing import Optional
from models import PolicyViolation

log = logging.getLogger("gateway.security")


# ═══════════════════════════════════════════════════════════════════════════════
#  IP-Based Rate Limiter (brute force / abuse protection)
# ═══════════════════════════════════════════════════════════════════════════════

class IPRateLimiter:
    """
    Sliding-window IP rate limiter.
    Protects against brute force, bot spam, and resource abuse.
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_hour: int = 300,
        burst_limit: int = 10,         # Max requests in 5-second window
        ban_threshold: int = 5,        # Consecutive 401s before temp ban
        ban_duration: int = 600,       # 10 min ban
        max_body_bytes: int = 20_000_000,  # 20 MB max request body
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.burst_limit = burst_limit
        self.ban_threshold = ban_threshold
        self.ban_duration = ban_duration
        self.max_body_bytes = max_body_bytes

        # Tracking dicts
        self._minute_hits: dict[str, list[float]] = defaultdict(list)
        self._hour_hits: dict[str, list[float]] = defaultdict(list)
        self._burst_hits: dict[str, list[float]] = defaultdict(list)
        self._auth_failures: dict[str, int] = defaultdict(int)
        self._banned_until: dict[str, float] = {}
        self._last_cleanup = time.time()

    def _cleanup(self):
        """Periodic cleanup of expired entries (every 5 min)."""
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now

        cutoff_minute = now - 60
        cutoff_hour = now - 3600

        for ip in list(self._minute_hits.keys()):
            self._minute_hits[ip] = [t for t in self._minute_hits[ip] if t > cutoff_minute]
            if not self._minute_hits[ip]:
                del self._minute_hits[ip]

        for ip in list(self._hour_hits.keys()):
            self._hour_hits[ip] = [t for t in self._hour_hits[ip] if t > cutoff_hour]
            if not self._hour_hits[ip]:
                del self._hour_hits[ip]

        for ip in list(self._banned_until.keys()):
            if self._banned_until[ip] < now:
                del self._banned_until[ip]
                self._auth_failures.pop(ip, None)

    def check(self, ip: str) -> tuple[bool, Optional[str]]:
        """
        Check if IP is allowed. Returns (allowed, reason).
        Localhost (127.0.0.1) gets relaxed limits for benchmarks.
        """
        now = time.time()
        self._cleanup()

        # Localhost gets 10x higher limits (benchmarks, testing)
        is_local = ip in ("127.0.0.1", "::1", "localhost")
        effective_rpm = self.rpm * 10 if is_local else self.rpm
        effective_rph = self.rph * 10 if is_local else self.rph
        effective_burst = self.burst_limit * 5 if is_local else self.burst_limit

        # Check ban (not for localhost)
        if not is_local and ip in self._banned_until:
            if now < self._banned_until[ip]:
                remaining = int(self._banned_until[ip] - now)
                log.warning(f"Banned IP blocked: {ip} ({remaining}s remaining)")
                return False, f"IP temporarily banned. Retry in {remaining}s."
            else:
                del self._banned_until[ip]
                self._auth_failures.pop(ip, None)

        # Burst check (5-second window)
        cutoff_burst = now - 5
        self._burst_hits[ip] = [t for t in self._burst_hits[ip] if t > cutoff_burst]
        if len(self._burst_hits[ip]) >= effective_burst:
            log.warning(f"Burst limit hit: {ip} ({len(self._burst_hits[ip])}/{effective_burst} in 5s)")
            return False, "Too many requests. Slow down."

        # Per-minute check
        cutoff_minute = now - 60
        self._minute_hits[ip] = [t for t in self._minute_hits[ip] if t > cutoff_minute]
        if len(self._minute_hits[ip]) >= effective_rpm:
            log.warning(f"RPM limit hit: {ip} ({len(self._minute_hits[ip])}/{effective_rpm})")
            return False, f"Rate limit: {effective_rpm} requests/min exceeded."

        # Per-hour check
        cutoff_hour = now - 3600
        self._hour_hits[ip] = [t for t in self._hour_hits[ip] if t > cutoff_hour]
        if len(self._hour_hits[ip]) >= effective_rph:
            log.warning(f"RPH limit hit: {ip} ({len(self._hour_hits[ip])}/{effective_rph})")
            return False, f"Rate limit: {effective_rph} requests/hour exceeded."

        # Record hit
        self._burst_hits[ip].append(now)
        self._minute_hits[ip].append(now)
        self._hour_hits[ip].append(now)
        return True, None

    def record_auth_failure(self, ip: str):
        """Record an auth failure. Auto-ban after threshold."""
        self._auth_failures[ip] = self._auth_failures.get(ip, 0) + 1
        if self._auth_failures[ip] >= self.ban_threshold:
            self._banned_until[ip] = time.time() + self.ban_duration
            log.warning(f"IP auto-banned: {ip} after {self._auth_failures[ip]} auth failures "
                        f"({self.ban_duration}s)")

    def record_auth_success(self, ip: str):
        """Reset auth failure counter on success."""
        self._auth_failures.pop(ip, None)

    def get_stats(self) -> dict:
        """Return current rate limiter stats."""
        return {
            "tracked_ips": len(self._minute_hits),
            "banned_ips": len(self._banned_until),
            "auth_failures": dict(self._auth_failures),
        }


# Module-level instance
ip_limiter = IPRateLimiter()


# ═══════════════════════════════════════════════════════════════════════════════
#  Security Headers
# ═══════════════════════════════════════════════════════════════════════════════

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store, no-cache, must-revalidate",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORS Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def get_cors_origins() -> list[str]:
    """
    Get allowed CORS origins from environment.
    Set CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
    Default: restrictive (only localhost for development).
    """
    import os
    origins = os.environ.get("CORS_ORIGINS", "")
    if origins:
        return [o.strip() for o in origins.split(",") if o.strip()]
    # Default: localhost only (safe for development)
    return [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]


# ─── Dangerous Shell Commands ─────────────────────────────────────────────────

DANGEROUS_COMMANDS = [
    # Destructive operations
    r"\brm\s+(-[rf]+\s+)*(/|~|\$HOME|\*)",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r"\b:(){.*};\s*:",
    r"\bchmod\s+(-R\s+)?[0-7]*777",
    r"\bchown\s+-R\s+.*\s+/",
    # Network attacks
    r"\bnc\s+.*-e\s+/bin/(ba)?sh",
    r"\bcurl\s+.*\|\s*(ba)?sh",
    r"\bwget\s+.*-O\s*-\s*\|\s*(ba)?sh",
    # Crypto mining
    r"\b(xmrig|minerd|cgminer|bfgminer)\b",
    # Service manipulation
    r"\bsystemctl\s+(stop|disable|mask)\s+(ssh|sshd|ufw|iptables)",
    r"\bservice\s+\w+\s+(stop|disable)",
    # Indirect destructive
    r"\bfind\s+.*-delete",
    r"\btruncate\s+--size\s*0",
]

# ─── Sensitive Data Patterns ──────────────────────────────────────────────────

SENSITIVE_PATTERNS = [
    r"(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w-]{20,}",
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    r"aws_secret_access_key\s*=",
    r"ghp_[a-zA-Z0-9]{36}",
    r"sk-[a-zA-Z0-9]{48}",
    r"sk-ant-[a-zA-Z0-9-]{95}",
    r"gsk_[a-zA-Z0-9]{50,}",
]

# ─── Forbidden Paths ─────────────────────────────────────────────────────────

FORBIDDEN_PATHS = [
    r"(cat|less|head|tail|nano|vim?|read|type|get-content)\s+/etc/(passwd|shadow|sudoers)",
    r"(cat|less|head|tail|nano|vim?|read|type|get-content)\s+(/root/|~/\.ssh/)",
    r"(cat|less|head|tail|nano|vim?|read|type|get-content)\s+\S*\.(env|pem|key|crt)\b",
    r"(curl|wget|scp|rsync)\s+\S*/etc/(passwd|shadow)",
    r"(curl|wget|scp|rsync)\s+\S*/root/",
]

# ─── Prompt Injection Patterns ────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(DAN|jailbreak)",
    r"pretend\s+you\s+are\s+(?!a\s+(senior|junior|software))",
    r"system\s*:\s*you\s+are",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"\[\/INST\]",
]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode tricks (invisible chars, lookalikes)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
    return text


def check_hard_policy(query: str, enable_sensitive: bool = True) -> Optional[PolicyViolation]:
    """
    Check query against hard policy rules.
    Returns None if OK, PolicyViolation otherwise.
    Runs BEFORE any LLM call to save costs and block dangerous requests.
    """
    normalized = normalize_unicode(query.lower())

    # 1. Check dangerous commands
    for pattern in DANGEROUS_COMMANDS:
        if re.search(pattern, normalized, re.IGNORECASE):
            log.warning(f"Policy block: dangerous_command matched '{pattern}'")
            return PolicyViolation(
                category="dangerous_command",
                pattern=pattern,
                message="Dangerous command detected"
            )

    # 2. Check sensitive data (on original, not just normalized)
    if enable_sensitive:
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                log.warning(f"Policy block: sensitive_data matched")
                return PolicyViolation(
                    category="sensitive_data",
                    pattern="[redacted]",
                    message="Sensitive data detected in request"
                )

    # 3. Check forbidden paths
    for pattern in FORBIDDEN_PATHS:
        if re.search(pattern, normalized, re.IGNORECASE):
            log.warning(f"Policy block: forbidden_path matched '{pattern}'")
            return PolicyViolation(
                category="forbidden_path",
                pattern=pattern,
                message="Access to protected path"
            )

    # 4. Check prompt injection attempts
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            log.warning(f"Policy block: prompt_injection detected")
            return PolicyViolation(
                category="prompt_injection",
                pattern="[redacted]",
                message="Potential prompt injection detected"
            )

    return None


def validate_api_key(provided_key: str, expected_key: str) -> bool:
    """Constant-time API key comparison to prevent timing attacks."""
    if not expected_key:
        return True  # No key configured = open access

    import hmac
    return hmac.compare_digest(provided_key, expected_key)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    import uuid
    return f"gw-{uuid.uuid4().hex[:16]}"


def hash_content(content: str) -> str:
    """SHA-256 hash for cache keys."""
    return hashlib.sha256(content.encode()).hexdigest()
