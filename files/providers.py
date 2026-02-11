"""
LLM Gateway - LLM Provider Integrations
Anthropic (with prompt caching), Groq, OpenAI, and Mock providers.
"""

import json
import time
import asyncio
import logging
import httpx
from typing import Optional
from models import ChatMessage, GatewayConfig
from metrics import metrics

log = logging.getLogger("gateway.providers")


# ─── Provider Exceptions ─────────────────────────────────────────────────────

class ProviderRateLimitError(Exception):
    """Raised when a provider returns 429 Too Many Requests."""
    def __init__(self, provider: str, model: str, retry_after: Optional[float] = None,
                 message: str = ""):
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        super().__init__(message or f"{provider}/{model}: Rate limited"
                         f"{f' (retry after {retry_after}s)' if retry_after else ''}")


class ProviderOverloadError(Exception):
    """Raised when a provider returns 503/529 (overloaded)."""
    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        super().__init__(message or f"{provider}/{model}: Service overloaded")


def _check_response(response: httpx.Response, provider: str, model: str):
    """Check HTTP response, raise specific exceptions for 429/503."""
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        retry_secs = float(retry_after) if retry_after else None
        raise ProviderRateLimitError(provider, model, retry_secs,
                                     response.text[:200])
    if response.status_code in (503, 529):
        raise ProviderOverloadError(provider, model, response.text[:200])
    response.raise_for_status()


def _normalize_content(content):
    """
    Flatten text-only content arrays to plain strings.
    Many providers (OpenRouter, Groq, OpenAI for non-vision models)
    don't support content arrays when there are no images.
    """
    if not isinstance(content, list):
        return content
    has_media = any(
        isinstance(b, dict) and b.get("type") in ("image_url", "image", "document")
        for b in content
    )
    if has_media:
        return content  # Keep as array — provider needs to handle image blocks
    # Text-only → join into string
    return "\n\n".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


# ─── Static System Prompt (cached by Anthropic) ──────────────────────────────

STATIC_SYSTEM_PROMPT = """You are a senior software engineer assistant with deep expertise in:
- Web Development (React, Next.js, Vue, Node.js, Python, Go, Rust)
- DevOps (Docker, Kubernetes, CI/CD, Terraform)
- Databases (PostgreSQL, MySQL, MongoDB, Redis, SQLite)
- Cloud (AWS, GCP, Azure)
- System Design and Architecture

Your communication style:
- Precise and technically correct
- Code examples as complete, runnable snippets
- For code changes: Use unified diff format when modifying existing files
- When uncertain: State it explicitly
- Concise explanations, avoid unnecessary verbosity

Output format for code changes:
```diff
--- a/path/to/file
+++ b/path/to/file
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
```

Rules:
- Include 3 lines of context above and below each change
- For multiple changes in the same file, use multiple hunks
- NEVER output the entire file when only a few lines change
- For new files only: output the full file content"""


# ─── Diff Instruction (injected for premium code tasks) ──────────────────────

DIFF_INSTRUCTION = """
OUTPUT FORMAT RULE:
When modifying existing files, you MUST respond with a unified diff — NOT the full file.
Use standard unified diff format with 3 lines of context.
NEVER output the entire file when only a few lines change.
For new files only: output the full file content.
"""


# ─── Cascade System Prompt (cheap model tries first, escalates if needed) ────

ESCALATION_MARKER = "[ESCALATE]"
ESCALATION_MARKER_CHEAP_PLUS = "[ESCALATE_TO_CHEAP_PLUS]"
ESCALATION_MARKER_MEDIUM = "[ESCALATE_TO_MEDIUM]"
ESCALATION_MARKER_PREMIUM = "[ESCALATE_TO_PREMIUM]"

CASCADE_SYSTEM_PROMPT = """You are a fast, helpful assistant. Answer the user's question directly when you can.

ESCALATION RULES — if the task exceeds your abilities, respond with ONLY one of these markers and nothing else:

""" + ESCALATION_MARKER_CHEAP_PLUS + """ — Use when the question needs REAL-TIME or CURRENT DATA that you don't have:
- Current weather, forecasts, temperature
- Live stock prices, exchange rates, crypto prices
- Today's news, recent events, election results
- Current sports scores, standings, schedules
- Current opening hours, availability, status of services
- Any question using "right now", "today", "currently", "live", "latest"

""" + ESCALATION_MARKER_MEDIUM + """ — Use when the task needs:
- ANY code writing task (even short functions, classes, scripts)
- Multi-step reasoning, logic puzzles, brain teasers, riddles
- Creative writing (poems, stories, jokes, limericks) — ALWAYS escalate these
- Detailed explanations of complex or hard topics
- Summarizing or analyzing longer text
- Comparisons, pros/cons analysis
- Translation between languages (German ↔ English, technical terms)

""" + ESCALATION_MARKER_PREMIUM + """ — Use ONLY when the task needs:
- Multi-file code changes, large diffs, architectural refactors (>100 lines)
- Complex system design or architecture documents
- Deep domain expertise requiring nuanced judgment (legal, financial, medical)
- Tasks explicitly requesting the "best" or "most thorough" answer

HANDLE DIRECTLY (no escalation):
- Greetings, small talk, simple questions
- Factual lookups, definitions, very short explanations (1-3 sentences)
- Simple math, conversions, formatting help
- Historical facts, general knowledge (not requiring current data)
- Command lookups, one-liner code snippets (e.g. "how to list files in bash")
- Simple translations of single words or short phrases

IMPORTANT RULES:
1. When in doubt, ESCALATE TO MEDIUM. It is better to escalate than to give a mediocre answer.
2. For questions about current/live/real-time data, ALWAYS use """ + ESCALATION_MARKER_CHEAP_PLUS + """.
3. For ANY reasoning, creative, or code task beyond a single command, use """ + ESCALATION_MARKER_MEDIUM + """.
4. Reserve """ + ESCALATION_MARKER_PREMIUM + """ for truly complex tasks. Most tasks are well-served by MEDIUM."""


# ─── Base Provider ────────────────────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False,
                   web_search: bool = False, tools: list = None,
                   **kwargs) -> dict:
        raise NotImplementedError

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        raise NotImplementedError


# ─── Anthropic Provider ──────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider with prompt caching support.
    Uses direct HTTP calls for maximum control over caching headers.
    """

    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25, "cache_read": 0.025, "cache_write": 0.3125},
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False) -> dict:
        start = time.time()

        # Build system prompt with caching
        system_block = []
        if system_prompt:
            block = {"type": "text", "text": system_prompt}
            if use_cache:
                block["cache_control"] = {"type": "ephemeral"}
            system_block.append(block)

        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                # Merge into system block
                system_block.append({"type": "text", "text": msg.text_content})
            else:
                # Support multimodal: convert OpenAI format → Anthropic format
                if msg.has_media and isinstance(msg.content, list):
                    anthropic_content = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                anthropic_content.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    # data:image/png;base64,... → extract media_type and data
                                    header, b64data = url.split(",", 1) if "," in url else (url, "")
                                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": b64data}
                                    })
                                else:
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {"type": "url", "url": url}
                                    })
                            elif part.get("type") in ("file", "document"):
                                # PDF/document support
                                file_info = part.get("file", part.get("source", {}))
                                url = file_info.get("url", "")
                                if url.startswith("data:"):
                                    header, b64data = url.split(",", 1) if "," in url else (url, "")
                                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "application/pdf"
                                    anthropic_content.append({
                                        "type": "document",
                                        "source": {"type": "base64", "media_type": media_type, "data": b64data}
                                    })
                    anthropic_messages.append({"role": msg.role, "content": anthropic_content})
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.text_content})

        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        if system_block:
            request_body["system"] = system_block

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=120.0,
            )
            _check_response(response, "anthropic", model)
            data = response.json()

        latency = (time.time() - start) * 1000

        # Extract usage and cost
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)

        cost = self._calculate_cost(model, input_tokens, output_tokens, cache_read, cache_write)

        # Log cache stats
        if cache_read > 0 or cache_write > 0:
            log.info(f"Prompt cache: read={cache_read}, write={cache_write}, saved=${(cache_read * 2.7 / 1_000_000):.4f}")

        metrics.histogram("provider_latency_ms", latency, tags={"provider": "anthropic", "model": model})

        content = ""
        if data.get("content"):
            content = data["content"][0].get("text", "")

        return {
            "content": content,
            "model": model,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
            "cost_usd": cost,
            "latency_ms": latency,
        }

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int,
                        cache_read: int, cache_write: int) -> float:
        prices = self.PRICING.get(model, {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75})
        # Regular input tokens (subtract cached tokens)
        regular_input = max(0, input_tokens - cache_read - cache_write)
        cost = (
            regular_input * prices["input"] / 1_000_000 +
            cache_read * prices["cache_read"] / 1_000_000 +
            cache_write * prices["cache_write"] / 1_000_000 +
            output_tokens * prices["output"] / 1_000_000
        )
        return cost


# ─── Groq Provider ────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """Groq provider for cheap/fast inference (Llama, Mixtral)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False) -> dict:
        start = time.time()

        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": _normalize_content(msg.content)})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            _check_response(response, "groq", model)
            data = response.json()

        latency = (time.time() - start) * 1000
        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]

        # Groq is very cheap
        total_tokens = usage.get("total_tokens", 0)
        cost = total_tokens * 0.06 / 1_000_000

        metrics.histogram("provider_latency_ms", latency, tags={"provider": "groq", "model": model})

        return {
            "content": content,
            "model": model,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": total_tokens,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "cost_usd": cost,
            "latency_ms": latency,
        }


# ─── OpenAI Provider (Embeddings) ────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI provider, primarily for embeddings."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": "text-embedding-3-small", "input": text},
                    timeout=10.0,
                )
                _check_response(response, "openai", model)
                data = response.json()
                return data["data"][0]["embedding"]
        except Exception as e:
            log.warning(f"OpenAI embedding failed: {e}")
            return None

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False) -> dict:
        start = time.time()
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": _normalize_content(msg.content)})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60.0,
            )
            _check_response(response, "openai", model)
            data = response.json()

        latency = (time.time() - start) * 1000
        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]

        return {
            "content": content,
            "model": model,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "cost_usd": 0,
            "latency_ms": latency,
        }


# ─── OpenRouter Provider (Single API Key for All Models) ────────────────────

class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider — access all major LLMs through a single API key.
    Uses the OpenAI-compatible API at https://openrouter.ai/api/v1.
    Pricing is per-model, tracked via OpenRouter's usage headers.
    """

    # Approximate pricing per 1M tokens (input/output) for common models
    PRICING = {
        # Anthropic via OpenRouter
        "anthropic/claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
        "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.0},
        # Meta Llama (cheap/router)
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.30, "output": 0.40},
        # Google
        "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
        "google/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
        "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.0},
        # OpenAI
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai/gpt-4o": {"input": 2.50, "output": 10.0},
        # xAI Grok
        "x-ai/grok-2-1212": {"input": 2.0, "output": 10.0},
        "x-ai/grok-3-mini-beta": {"input": 0.30, "output": 0.50},
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324": {"input": 0.14, "output": 0.28},
        # Qwen
        "qwen/qwen-2.5-72b-instruct": {"input": 0.30, "output": 0.40},
    }

    def __init__(self, api_key: str, site_url: str = "", site_name: str = "LLM Gateway"):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.site_url = site_url
        self.site_name = site_name

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False,
                   web_search: bool = False,
                   tools: list[dict] = None,
                   raw_messages: list[dict] = None) -> dict:
        """
        Chat with OpenRouter. Supports:
          - tools: OpenAI function calling format (list of tool definitions)
          - raw_messages: Pre-formatted OpenAI messages (for tool result follow-ups)
          - web_search: OpenRouter :online plugin for paid web search
        """
        start = time.time()

        # Use raw_messages if provided (for tool call follow-up), else build from ChatMessage
        if raw_messages:
            openai_messages = raw_messages
        else:
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                openai_messages.append({"role": msg.role, "content": _normalize_content(msg.content)})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        request_body = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Function calling: pass tool definitions
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"

        # Enable web search via OpenRouter :online plugin
        # This gives any model real-time data (weather, stocks, news)
        if web_search:
            request_body["plugins"] = [{"id": "web", "max_results": 5}]

        # Retry with exponential backoff for 429/503
        max_retries = 3
        data = None
        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=request_body,
                    timeout=120.0,
                )
                if response.status_code == 429 and attempt < max_retries:
                    retry_after = response.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else (2 ** attempt)
                    wait = min(wait, 15)  # Cap at 15s
                    log.warning(f"OpenRouter 429 (attempt {attempt+1}/{max_retries+1}), "
                                f"retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                if response.status_code == 503 and attempt < max_retries:
                    wait = 2 ** attempt
                    log.warning(f"OpenRouter 503 (attempt {attempt+1}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                _check_response(response, "openrouter", model)
                data = response.json()
                break

        latency = (time.time() - start) * 1000
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Calculate cost from our pricing table or use OpenRouter's reported cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        # Add web search cost (~$0.02 per request with 5 results via Exa)
        if web_search:
            cost += 0.02

        content = ""
        tool_calls = None
        finish_reason = "stop"
        if data.get("choices") and len(data["choices"]) > 0:
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", None)
            finish_reason = choice.get("finish_reason", "stop")

        metrics.histogram("provider_latency_ms", latency, tags={"provider": "openrouter", "model": model})

        result = {
            "content": content,
            "model": model,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "cost_usd": cost,
            "latency_ms": latency,
            "finish_reason": finish_reason,
        }

        # Include tool_calls if the model requested them
        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        return (
            input_tokens * prices["input"] / 1_000_000 +
            output_tokens * prices["output"] / 1_000_000
        )

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding via OpenRouter (limited model support)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "openai/text-embedding-3-small",
                        "input": text,
                    },
                    timeout=10.0,
                )
                _check_response(response, "openrouter", model)
                data = response.json()
                return data["data"][0]["embedding"]
        except Exception as e:
            log.warning(f"OpenRouter embedding failed: {e} (semantic cache disabled)")
            return None


# ─── Mock Provider (for testing) ─────────────────────────────────────────────

class MockProvider(LLMProvider):
    """Mock provider for testing without real API calls."""

    async def chat(self, messages: list[ChatMessage], model: str,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   system_prompt: str = "", use_cache: bool = False,
                   tools: list = None, **kwargs) -> dict:
        import asyncio
        await asyncio.sleep(0.1)  # Simulate latency

        user_msg = messages[-1].content if messages else ""

        return {
            "content": f"[MOCK RESPONSE] Received: '{user_msg[:100]}...' | Model: {model}",
            "model": f"mock-{model}",
            "usage": {
                "prompt_tokens": len(user_msg.split()) * 2,
                "completion_tokens": 50,
                "total_tokens": len(user_msg.split()) * 2 + 50,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            "cost_usd": 0.0,
            "latency_ms": 100,
        }

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        import numpy as np
        # Return a deterministic fake embedding based on text hash
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(1536).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        return embedding.tolist()


# ─── Provider Factory ─────────────────────────────────────────────────────────

def create_provider(provider_name: str, api_key: str = "", mock: bool = False) -> LLMProvider:
    """Create an LLM provider instance."""
    if mock:
        return MockProvider()

    providers = {
        "anthropic": lambda: AnthropicProvider(api_key),
        "groq": lambda: GroqProvider(api_key),
        "openai": lambda: OpenAIProvider(api_key),
        "openrouter": lambda: OpenRouterProvider(api_key),
    }

    factory = providers.get(provider_name)
    if not factory:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")

    return factory()
