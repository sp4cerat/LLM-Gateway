"""
LLM Gateway - Data Models
Pydantic models for requests, responses, configuration, and internal types.
"""

from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ─── Router Enums & Models ───────────────────────────────────────────────────

class RouterAction(str, Enum):
    CACHE_ONLY = "cache_only"
    LOCAL = "local"
    CHEAP = "cheap"
    CHEAP_PLUS = "cheap_plus"
    MEDIUM = "medium"
    PREMIUM = "premium"


class RouterResult(BaseModel):
    action: RouterAction
    confidence: float = Field(ge=0.0, le=1.0)
    response_type: str = "explanation_generic"
    reason: str = ""
    is_code_generation: bool = False
    needs_web: bool = False


# ─── API Request/Response (OpenAI-Compatible) ────────────────────────────────

class ChatMessage(BaseModel):
    """
    OpenAI-compatible chat message with multimodal and tool-calling support.
    
    content can be:
      - str: Plain text message
      - list: Multimodal content (OpenAI format):
          [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "file", "file": {"url": "data:application/pdf;base64,..."}},
          ]
      - None: When message only contains tool_calls
    """
    role: str = Field(..., description="Role: system, user, assistant, tool, developer")
    content: Optional[Union[str, list]] = Field(default=None, description="Message content (text or multimodal list)")
    # OpenAI tool calling fields
    tool_calls: Optional[list[dict]] = Field(default=None, description="Tool calls from assistant")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID for tool result messages")
    name: Optional[str] = Field(default=None, description="Function name for tool result messages")

    @property
    def text_content(self) -> str:
        """Extract only the text from content, regardless of format."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts = []
            for item in self.content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
            return "\n".join(parts)
        return str(self.content)

    @property
    def has_media(self) -> bool:
        """Check if message contains images, files, or other media."""
        if self.content is None or isinstance(self.content, str):
            return False
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t in ("image_url", "image", "file", "document"):
                        return True
            return False
        return False

    @property
    def media_types(self) -> list[str]:
        """Return list of media types in this message (e.g. ['image', 'pdf'])."""
        types = []
        if not isinstance(self.content, list):
            return types
        for item in self.content:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "")
            if t == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if "image/png" in url:
                    types.append("png")
                elif "image/jpeg" in url or "image/jpg" in url:
                    types.append("jpeg")
                elif "image/gif" in url:
                    types.append("gif")
                elif "image/webp" in url:
                    types.append("webp")
                else:
                    types.append("image")
            elif t == "image":
                types.append("image")
            elif t in ("file", "document"):
                url = ""
                if "file" in item:
                    url = item["file"].get("url", "")
                elif "source" in item:
                    url = item["source"].get("media_type", "")
                if "pdf" in url:
                    types.append("pdf")
                else:
                    types.append("file")
        return types

    @property
    def media_token_estimate(self) -> int:
        """Estimate tokens for media content (images ~765 tok, PDFs ~1000 tok/page)."""
        if not self.has_media:
            return 0
        tokens = 0
        if not isinstance(self.content, list):
            return 0
        for item in self.content:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "")
            if t in ("image_url", "image"):
                # OpenAI: ~85 tokens (low), ~765 (high/auto) per image
                detail = "auto"
                if t == "image_url":
                    detail = item.get("image_url", {}).get("detail", "auto")
                tokens += 85 if detail == "low" else 765
            elif t in ("file", "document"):
                # Rough estimate: ~1000 tokens per PDF page, assume 3 pages avg
                tokens += 3000
        return tokens

    def to_api_format(self) -> dict:
        """Convert to OpenAI API format dict."""
        return {"role": self.role, "content": self.content}


class ChatRequest(BaseModel):
    model: str = Field(default="auto", description="Model name or 'auto' for routing")
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32000)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    # OpenAI-compatible tool fields (used by agents like OpenClaw)
    tools: Optional[list[dict]] = Field(default=None, description="Tool/function definitions")
    tool_choice: Optional[Any] = Field(default=None, description="Tool choice strategy")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Allow parallel tool calls")
    # Gateway-specific extensions
    fingerprint: Optional[str] = Field(default=None, description="Working-tree fingerprint")
    project_path: Optional[str] = Field(default=None, description="Project path for context")
    file_context: Optional[dict] = Field(default=None, description="Active file context")
    # Enhanced routing extensions (v2)
    customer_tier: Optional[str] = Field(default="standard", description="Customer tier: standard, premium, enterprise")
    task_priority: Optional[str] = Field(default="normal", description="Task priority: low, normal, high, critical")


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    # Gateway metadata
    gateway_metadata: Optional[dict] = None


# ─── Security Models ─────────────────────────────────────────────────────────

class PolicyViolation(BaseModel):
    category: str
    pattern: str
    message: str


# ─── Cache Models ─────────────────────────────────────────────────────────────

class CacheInvalidationEvent(BaseModel):
    event: str = Field(..., description="Event type: git_commit, manual, file_change")
    commit: Optional[str] = None
    files: Optional[list[str]] = None


# ─── Budget & Rate Limiting ──────────────────────────────────────────────────

class BudgetStatus(BaseModel):
    today_spend: float
    soft_limit: float
    medium_limit: float
    hard_limit: float
    premium_disabled: bool
    level: str  # ok, warning, throttle, kill


class RateLimitStatus(BaseModel):
    tier: str
    rpm_current: int
    rpm_limit: int
    tpm_current: int
    tpm_limit: int
    daily_spend: float
    daily_limit: float


# ─── Monitoring Models ────────────────────────────────────────────────────────

class GatewayStats(BaseModel):
    uptime_seconds: float
    total_requests: int
    requests_by_tier: dict[str, int]
    cache_hits: dict[str, int]
    cache_misses: dict[str, int]
    total_cost_today_usd: float
    avg_latency_ms: dict[str, float]
    premium_ratio: float
    cache_hit_rate: float
    budget_status: BudgetStatus
    errors_today: int
    last_updated: datetime


# ─── Configuration Models ────────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    router_provider: str = "openrouter"
    router_model: str = "google/gemini-2.0-flash-001"
    cheap_provider: str = "openrouter"
    cheap_model: str = "google/gemini-2.0-flash-001"
    cheap_plus_provider: str = "openrouter"
    cheap_plus_model: str = "google/gemini-3-flash-preview"
    medium_provider: str = "openrouter"
    medium_model: str = "anthropic/claude-3.5-haiku"
    premium_provider: str = "openrouter"
    premium_model: str = "anthropic/claude-sonnet-4-20250514"
    embedding_provider: str = "openrouter"
    embedding_model: str = "openai/text-embedding-3-small"


class CacheConfig(BaseModel):
    exact_cache_enabled: bool = True
    semantic_cache_enabled: bool = True
    semantic_similarity_threshold: float = 0.92
    embedding_cache_ttl_days: int = 30
    max_cache_size_mb: int = 500


class BudgetConfig(BaseModel):
    daily_soft_limit: float = 5.0
    daily_medium_limit: float = 15.0
    daily_hard_limit: float = 50.0


class RateLimitConfig(BaseModel):
    local_rpm: int = 100
    local_tpm: int = 50000
    cheap_rpm: int = 60
    cheap_tpm: int = 100000
    medium_rpm: int = 40
    medium_tpm: int = 80000
    premium_rpm: int = 20
    premium_tpm: int = 50000


class SecurityConfig(BaseModel):
    require_api_key: bool = True
    enable_policy_gate: bool = True
    block_sensitive_data: bool = True


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    providers: ProviderConfig = ProviderConfig()
    cache: CacheConfig = CacheConfig()
    budget: BudgetConfig = BudgetConfig()
    rate_limits: RateLimitConfig = RateLimitConfig()
    security: SecurityConfig = SecurityConfig()
    routing_strategy: str = "cost_optimized"
    mock_mode: bool = False
    web_search: "WebSearchConfig" = None  # type: ignore
    code_generation: "CodeGenerationConfig" = None  # type: ignore
    tools: "ToolsConfig" = None  # type: ignore

    def model_post_init(self, __context):
        if self.web_search is None:
            self.web_search = WebSearchConfig()
        if self.code_generation is None:
            self.code_generation = CodeGenerationConfig()
        if self.tools is None:
            self.tools = ToolsConfig()


class WebSearchConfig(BaseModel):
    """Web search pipeline configuration."""
    # Result counts
    max_snippets: int = 5            # DDG snippet results to fetch
    min_pages: int = 2               # MINIMUM pages to always download (even for 'min' depth)
    max_pages_deep: int = 5          # Full-text pages for 'deep' depth
    max_pages_thorough: int = 8      # Full-text pages for 'thorough' depth

    # Content extraction
    max_page_chars: int = 12000      # Max chars extracted per page (~3000 tokens)
    page_fetch_timeout: float = 8.0  # Timeout per page fetch in seconds

    # Source citations
    show_sources: bool = True        # Append source URLs to LLM response
    source_format: str = "footer"    # "footer" = [1] URL at end, "none" = disabled

    # Search engines (order = priority)
    engines: list[str] = ["ddg"]     # Available: "ddg", "google", "bing"

    # Multi-query
    multi_query: bool = True         # Allow model to issue multiple search queries
    max_queries: int = 3             # Max parallel queries per search call


class CodeGenerationConfig(BaseModel):
    """Code generation routing configuration."""
    # Minimum tier for code generation tasks in agent sessions
    # Options: "cheap", "medium", "premium", "custom"
    min_tier: str = "medium"
    # Custom OpenRouter model (only used when min_tier="custom")
    custom_model: str = ""
    # Detection: which agent tools indicate code generation
    code_tool_names: list[str] = ["exec", "write", "create_file", "edit_file",
                                   "patch_file", "sub_agent"]


class ToolsConfig(BaseModel):
    """Enable/disable individual tool skills."""
    web_search: bool = True      # Web search (DDG/Bing/Google scraping)
    weather: bool = True         # Weather via Open-Meteo
    stocks: bool = True          # Stock/crypto prices via yfinance
    news: bool = True            # News headlines via NewsAPI
    vision: bool = True          # Image analysis / OCR
    transcription: bool = True   # Audio transcription via Whisper
    pdf: bool = True             # PDF text extraction
    docx: bool = True            # DOCX/DOC text extraction
    zip: bool = True             # ZIP/TAR archive extraction
