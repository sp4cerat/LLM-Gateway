"""
LLM Gateway - Configuration Loader
Loads config from config.yaml + .env with validation and defaults.
"""

import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from models import GatewayConfig, ProviderConfig, CacheConfig, BudgetConfig, RateLimitConfig, SecurityConfig, WebSearchConfig

log = logging.getLogger("gateway.config")

# Default config path
CONFIG_DIR = Path(os.environ.get("GATEWAY_CONFIG_DIR", Path(__file__).parent))
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"


def load_config() -> GatewayConfig:
    """
    Load configuration from config.yaml and .env
    Priority: ENV vars > config.yaml > defaults
    """
    # 1. Load .env file
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
        log.info(f"Loaded .env from {ENV_FILE}")
    else:
        load_dotenv()  # Try default locations

    # 2. Load config.yaml
    yaml_config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            yaml_config = yaml.safe_load(f) or {}
        log.info(f"Loaded config from {CONFIG_FILE}")
    else:
        log.warning(f"No config.yaml found at {CONFIG_FILE}, using defaults")

    # 3. Build config with precedence: ENV > YAML > defaults
    providers_yaml = yaml_config.get("providers", {})
    cache_yaml = yaml_config.get("cache", {})
    budget_yaml = yaml_config.get("budget", {})
    rate_limits_yaml = yaml_config.get("rate_limits", {})
    security_yaml = yaml_config.get("security", {})
    web_search_yaml = yaml_config.get("web_search", {})

    config = GatewayConfig(
        host=os.environ.get("GATEWAY_HOST", yaml_config.get("host", "0.0.0.0")),
        port=int(os.environ.get("GATEWAY_PORT", yaml_config.get("port", 8000))),
        debug=os.environ.get("GATEWAY_DEBUG", str(yaml_config.get("debug", False))).lower() == "true",
        routing_strategy=os.environ.get("ROUTING_STRATEGY", yaml_config.get("routing_strategy", "cost_optimized")),
        mock_mode=os.environ.get("MOCK_MODE", str(yaml_config.get("mock_mode", False))).lower() == "true",
        providers=ProviderConfig(
            router_provider=providers_yaml.get("router_provider", "openrouter"),
            router_model=providers_yaml.get("router_model", "google/gemini-2.0-flash-001"),
            cheap_provider=providers_yaml.get("cheap_provider", "openrouter"),
            cheap_model=providers_yaml.get("cheap_model", "google/gemini-2.0-flash-001"),
            cheap_plus_provider=providers_yaml.get("cheap_plus_provider", "openrouter"),
            cheap_plus_model=providers_yaml.get("cheap_plus_model", "google/gemini-3-flash-preview"),
            medium_provider=providers_yaml.get("medium_provider", "openrouter"),
            medium_model=providers_yaml.get("medium_model", "anthropic/claude-3.5-haiku"),
            premium_provider=providers_yaml.get("premium_provider", "openrouter"),
            premium_model=providers_yaml.get("premium_model", "anthropic/claude-sonnet-4-20250514"),
            embedding_provider=providers_yaml.get("embedding_provider", "openrouter"),
            embedding_model=providers_yaml.get("embedding_model", "openai/text-embedding-3-small"),
        ),
        cache=CacheConfig(**{**CacheConfig().model_dump(), **cache_yaml}),
        budget=BudgetConfig(
            daily_soft_limit=float(os.environ.get("DAILY_BUDGET_SOFT", budget_yaml.get("daily_soft_limit", 5.0))),
            daily_medium_limit=float(os.environ.get("DAILY_BUDGET_MEDIUM", budget_yaml.get("daily_medium_limit", 15.0))),
            daily_hard_limit=float(os.environ.get("DAILY_BUDGET_HARD", budget_yaml.get("daily_hard_limit", 50.0))),
        ),
        rate_limits=RateLimitConfig(**{**RateLimitConfig().model_dump(), **rate_limits_yaml}),
        security=SecurityConfig(**{**SecurityConfig().model_dump(), **security_yaml}),
        web_search=WebSearchConfig(**{**WebSearchConfig().model_dump(), **web_search_yaml}),
    )

    return config


def get_api_key(provider: str) -> str:
    """Get API key for a provider from environment."""
    key_map = {
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(f"Missing API key: {env_var}")

    return key


def get_gateway_secret() -> str:
    """Get the gateway authentication secret."""
    return os.environ.get("GATEWAY_SECRET", "")


def generate_default_config() -> str:
    """Generate a default config.yaml content."""
    return """# LLM Gateway Configuration v1.3
# Cost-Optimized AI Routing

host: "0.0.0.0"
port: 8000
debug: false
routing_strategy: "cost_optimized"  # cost_optimized | quality_first | local_only
mock_mode: false  # true = no real API calls (for testing)

providers:
  router_provider: "groq"
  router_model: "llama-3.1-8b-instant"
  cheap_provider: "anthropic"
  cheap_model: "claude-3-5-haiku-20241022"
  premium_provider: "anthropic"
  premium_model: "claude-sonnet-4-20250514"
  embedding_provider: "openai"
  embedding_model: "text-embedding-3-small"

cache:
  exact_cache_enabled: true
  semantic_cache_enabled: true
  semantic_similarity_threshold: 0.92
  embedding_cache_ttl_days: 30
  max_cache_size_mb: 500

budget:
  daily_soft_limit: 5.0    # Warning threshold (USD)
  daily_medium_limit: 15.0  # Throttle threshold (USD)
  daily_hard_limit: 50.0    # Kill switch threshold (USD)

rate_limits:
  local_rpm: 100
  local_tpm: 50000
  cheap_rpm: 50
  cheap_tpm: 100000
  premium_rpm: 20
  premium_tpm: 50000

security:
  require_api_key: true
  enable_policy_gate: true
  block_sensitive_data: true
"""


def generate_default_env() -> str:
    """Generate a default .env template."""
    return """# LLM Gateway Environment Variables
# IMPORTANT: Never commit this file to version control!

# === Provider Mode ===
# Option A: OpenRouter-only (single key for all models)
OPENROUTER_API_KEY=sk-or-your_openrouter_key_here

# Option B: Multi-provider (comment out OpenRouter, uncomment these)
# GROQ_API_KEY=gsk_your_groq_key_here
# ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
# OPENAI_API_KEY=sk-your_openai_key_here  # Only for embeddings fallback

# === Gateway Security ===
GATEWAY_SECRET=change_me_to_a_random_string

# === Budget Overrides (optional, config.yaml values used if not set) ===
# DAILY_BUDGET_SOFT=5.0
# DAILY_BUDGET_MEDIUM=15.0
# DAILY_BUDGET_HARD=50.0

# === Server Overrides (optional) ===
# GATEWAY_HOST=0.0.0.0
# GATEWAY_PORT=8000
# GATEWAY_DEBUG=false
# MOCK_MODE=false
"""
