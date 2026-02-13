"""
LLM Gateway - Context Budgeting & Compression
Token budget management, log compression, and output strategy.
"""

import re
import logging
from typing import Optional

log = logging.getLogger("gateway.context")


# ─── Token Estimation ─────────────────────────────────────────────────────────

def estimate_tokens(content) -> int:
    """
    Estimate token count for text or multimodal content.
    
    Args:
        content: str, list (OpenAI multimodal format), or ChatMessage
    """
    if not content:
        return 0
    
    # Handle ChatMessage objects
    if hasattr(content, 'text_content'):
        text_tokens = estimate_tokens(content.text_content)
        media_tokens = getattr(content, 'media_token_estimate', 0)
        return text_tokens + media_tokens
    
    # Handle multimodal list format
    if isinstance(content, list):
        tokens = 0
        for item in content:
            if isinstance(item, str):
                tokens += estimate_tokens(item)
            elif isinstance(item, dict):
                t = item.get("type", "")
                if t == "text":
                    tokens += estimate_tokens(item.get("text", ""))
                elif t in ("image_url", "image"):
                    detail = "auto"
                    if t == "image_url":
                        detail = item.get("image_url", {}).get("detail", "auto")
                    tokens += 85 if detail == "low" else 765
                elif t in ("file", "document"):
                    tokens += 3000  # ~3 pages avg
        return tokens
    
    # Handle plain text
    text = str(content)
    if not text:
        return 0
    words = len(text.split())
    # Code tends to have more tokens per word
    if any(c in text for c in ['{', '}', '(', ')', ';', '//', '/*']):
        return int(words * 1.5)
    return int(words * 1.3)


# ─── Context Budget ──────────────────────────────────────────────────────────

class ContextBudget:
    """Limits context size per tier to control costs."""

    LIMITS = {
        "local":      {"input": 4000,   "output": 2000},
        "cheap":      {"input": 4000,   "output": 2000},
        "cheap_long": {"input": 100000, "output": 4000},  # Doc QA on Gemini Flash (1M context)
        "medium":     {"input": 8000,   "output": 4000},
        "premium":    {"input": 16000,  "output": 8000},
    }

    def get_limits(self, tier: str) -> dict:
        return self.LIMITS.get(tier, self.LIMITS["cheap"])

    def apply(self, tier: str, messages: list, system_prompt: str = "") -> tuple[list, int]:
        """
        Apply token budget to messages.
        Returns (trimmed_messages, max_output_tokens).
        """
        limits = self.get_limits(tier)
        max_input = limits["input"]
        max_output = limits["output"]

        system_tokens = estimate_tokens(system_prompt)
        remaining = max_input - system_tokens

        # Always keep the last user message complete
        if not messages:
            return messages, max_output

        trimmed = []
        total_tokens = 0

        # Process messages in reverse (keep most recent)
        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg.content)
            if total_tokens + msg_tokens <= remaining:
                trimmed.insert(0, msg)
                total_tokens += msg_tokens
            else:
                # Truncate this message to fit (only text, keep media intact)
                available = remaining - total_tokens
                if available > 100:
                    if msg.has_media:
                        # Keep media messages intact (don't truncate images/files)
                        trimmed.insert(0, msg)
                    else:
                        truncated_content = truncate_to_tokens(msg.text_content, available)
                        from models import ChatMessage
                        trimmed.insert(0, ChatMessage(
                            role=msg.role,
                            content=truncated_content + "\n\n[... truncated for context budget ...]"
                        ))
                break

        # ── Post-trim cleanup: fix orphaned tool messages ──
        # Context trimming can cut in the middle of a tool-call sequence:
        #   tool(no tid) → assistant(tc) → tool(tid) → ...
        # Google/Gemini rejects tool messages without tool_call_id.
        # Strip leading messages until we reach a valid conversation start
        # (developer, system, user, or assistant without tool_call dependency).
        while trimmed:
            first = trimmed[0]
            if first.role == "tool":
                # Orphaned tool result — no matching assistant+tool_call before it
                trimmed.pop(0)
            elif (first.role == "assistant"
                  and getattr(first, 'tool_calls', None)
                  and not getattr(first, 'content', None)):
                # Assistant with only tool_calls but no content — the tool results
                # that follow may also be orphaned if they reference earlier context
                trimmed.pop(0)
            else:
                break

        return trimmed, max_output


# ─── Output Strategy ──────────────────────────────────────────────────────────

class OutputStrategy:
    """
    Decides whether to request diff or full-file output.
    Diffs save 70-90% on premium output tokens.
    """

    DIFF_THRESHOLD_LINES = 30
    DIFF_ELIGIBLE_TYPES = ["code_suggestion", "code_review", "command_execution"]
    FULL_OUTPUT_TYPES = ["explanation_generic", "explanation_contextual", "documentation"]

    def get_strategy(self, response_type: str, tier: str,
                     file_context: Optional[dict] = None) -> dict:
        """
        Returns output strategy for a request.
        """
        # Non-code → always full
        if response_type in self.FULL_OUTPUT_TYPES:
            return {
                "mode": "full",
                "inject_diff_instruction": False,
                "max_output_tokens": self._get_output_limit(tier),
            }

        # Cheap tier → full is fine (low cost)
        if tier in ("local", "cheap", "medium"):
            return {
                "mode": "full",
                "inject_diff_instruction": False,
                "max_output_tokens": self._get_output_limit(tier),
            }

        # Premium + code → evaluate diff
        if file_context:
            file_lines = file_context.get("total_lines", 0)
            if file_lines <= self.DIFF_THRESHOLD_LINES:
                return {
                    "mode": "full",
                    "inject_diff_instruction": False,
                    "max_output_tokens": self._get_output_limit(tier),
                }
            return {
                "mode": "diff",
                "inject_diff_instruction": True,
                "max_output_tokens": min(2000, self._get_output_limit(tier)),
            }

        # Default for premium code tasks: prefer diff
        if response_type in self.DIFF_ELIGIBLE_TYPES:
            return {
                "mode": "diff",
                "inject_diff_instruction": True,
                "max_output_tokens": min(2000, self._get_output_limit(tier)),
            }

        return {
            "mode": "full",
            "inject_diff_instruction": False,
            "max_output_tokens": self._get_output_limit(tier),
        }

    def _get_output_limit(self, tier: str) -> int:
        limits = {"local": 2000, "cheap": 2000, "medium": 4000, "premium": 8000}
        return limits.get(tier, 4000)


# ─── Log Compression ──────────────────────────────────────────────────────────

def compress_logs(logs: str, max_tokens: int = 2000) -> str:
    """
    Compress logs for context inclusion.
    1. Remove duplicates
    2. Normalize timestamps
    3. Shorten stack traces
    4. Trim to budget
    """
    if not logs:
        return ""

    lines = logs.split("\n")

    # 1. Remove duplicate lines
    seen = set()
    unique_lines = []
    for line in lines:
        normalized = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?',
            '<TIME>', line
        )
        normalized = re.sub(r'[a-f0-9]{8,}', '<ID>', normalized)
        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line)

    # 2. Shorten stack traces
    compressed = _compress_stacktraces(unique_lines)

    # 3. Trim to token budget
    result = "\n".join(compressed)
    while estimate_tokens(result) > max_tokens and compressed:
        compressed = compressed[1:]
        result = "\n".join(compressed)

    return result


def _compress_stacktraces(lines: list[str]) -> list[str]:
    """Shorten stack traces to first 3 + last 3 frames."""
    result = []
    in_stacktrace = False
    stack_buffer = []

    for line in lines:
        if re.match(r'\s*(at |File "|Traceback)', line):
            in_stacktrace = True
            stack_buffer.append(line)
            continue

        if in_stacktrace and not re.match(r'\s*(at |File "|\s+\^)', line):
            in_stacktrace = False
            if len(stack_buffer) > 6:
                result.extend(stack_buffer[:3])
                result.append(f"    ... ({len(stack_buffer) - 6} frames omitted)")
                result.extend(stack_buffer[-3:])
            else:
                result.extend(stack_buffer)
            stack_buffer = []

        if in_stacktrace:
            stack_buffer.append(line)
        else:
            result.append(line)

    # Flush remaining
    if stack_buffer:
        result.extend(stack_buffer[:3])
        if len(stack_buffer) > 6:
            result.append(f"    ... ({len(stack_buffer) - 6} frames omitted)")
            result.extend(stack_buffer[-3:])

    return result


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    words = text.split()
    # ~1.3 tokens per word
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# Global instances
context_budget = ContextBudget()
output_strategy = OutputStrategy()
