"""
LLM Gateway - Verification Layer
=================================
Implements "Draft & Verify" pattern:
  1. Cheap model generates a draft answer
  2. Expensive model verifies (not regenerates) — much fewer output tokens
  3. If verification fails → re-generate with bigger model

Key insight: Verification is cheaper than generation because
the verifier only outputs "correct/incorrect + reason" (< 200 tokens)
vs. regenerating the full answer (potentially thousands of tokens).

Also enforces evidence spans: answers must reference specific text locations.
"""

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from models import ChatMessage
from context import estimate_tokens
from metrics import metrics

log = logging.getLogger("gateway.verification")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class EvidenceSpan:
    """A reference to a specific location in the source text."""
    chunk_id: int
    text_snippet: str = ""
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class VerificationResult:
    """Result of the verification step."""
    verdict: str  # "correct", "partially_correct", "incorrect", "insufficient_evidence"
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    missing_context: list[int] = field(default_factory=list)  # chunk_ids needed
    should_regenerate: bool = False
    verification_tokens: int = 0
    verification_cost: float = 0.0


# ─── Verification Prompts ─────────────────────────────────────────────────────

VERIFY_SYSTEM_PROMPT = """You are a verification agent. Your ONLY job is to check whether a draft answer is correct and complete.

You will receive:
1. The original question
2. A draft answer (from a smaller model)
3. Relevant source text / context

Your task:
- Check if the draft answer is factually consistent with the source text
- Check if it missed critical information
- Check for hallucinations (claims not supported by the source)
- Check for contradictions with the source

IMPORTANT: You are NOT generating a new answer. You are ONLY verifying.

Respond ONLY with JSON:
{
  "verdict": "correct" | "partially_correct" | "incorrect" | "insufficient_evidence",
  "confidence": 0.0-1.0,
  "issues": ["list of specific problems found, if any"],
  "missing_chunks": [list of chunk_ids that would be needed for full verification],
  "should_regenerate": true/false
}

Rules:
- "correct" = answer is accurate and complete based on available evidence
- "partially_correct" = mostly right but has minor gaps or imprecisions
- "incorrect" = contains factual errors or significant hallucinations
- "insufficient_evidence" = cannot verify because needed context is missing
- When in doubt, set should_regenerate=true (safety over cost)
"""

EVIDENCE_INSTRUCTION = """
EVIDENCE REQUIREMENT: For every factual claim in your answer, you MUST include a reference.
Format: [ref:chunk_id] after each claim.

Example:
"The API rate limit is 1000 requests per minute [ref:3] and requires OAuth2 [ref:7]."

If you cannot find evidence for a claim in the provided context, state:
"[ref:NONE] — this claim needs verification against the full document."
"""


# ─── Verification Layer ──────────────────────────────────────────────────────

class VerificationLayer:
    """
    Implements the Draft & Verify pattern.

    Flow:
      1. draft = cheap_model.generate(query, partial_context)
      2. verification = expensive_model.verify(query, draft, targeted_context)
      3. if verification.should_regenerate → expensive_model.generate(query, full_context)

    Cost savings: verification uses ~100-200 output tokens vs. 500-5000 for generation.
    """

    def __init__(self, confidence_threshold: float = 0.8, enabled: bool = True):
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled

    async def verify_draft(
        self,
        query: str,
        draft_response: str,
        context_text: str,
        verifier_provider,
        verifier_model: str,
    ) -> VerificationResult:
        """
        Verify a draft answer against source context.

        Args:
            query: Original user question
            draft_response: Answer from the cheap model
            context_text: Relevant context (targeted chunks, not full document)
            verifier_provider: LLM provider to use for verification
            verifier_model: Model ID for verification

        Returns:
            VerificationResult with verdict and action recommendation
        """
        if not self.enabled:
            return VerificationResult(verdict="correct", confidence=1.0)

        start = time.time()

        # Build verification prompt
        user_prompt = self._build_verify_prompt(query, draft_response, context_text)

        try:
            result = await verifier_provider.chat(
                messages=[ChatMessage(role="user", content=user_prompt)],
                model=verifier_model,
                max_tokens=300,  # Verification is compact
                temperature=0,  # Deterministic for consistency
                system_prompt=VERIFY_SYSTEM_PROMPT,
                use_cache=False,
            )

            parsed = self._parse_verification(result.get("content", ""))
            parsed.verification_tokens = result["usage"].get("total_tokens", 0)
            parsed.verification_cost = result.get("cost_usd", 0)

            latency = (time.time() - start) * 1000
            metrics.histogram("verification_latency_ms", latency)
            metrics.increment("verification_verdicts", tags={"verdict": parsed.verdict})

            log.info(
                f"Verification: {parsed.verdict} (conf={parsed.confidence:.2f}, "
                f"{parsed.verification_tokens} tok, ${parsed.verification_cost:.4f}, "
                f"{latency:.0f}ms)"
            )

            return parsed

        except Exception as e:
            log.warning(f"Verification failed: {e} — defaulting to regenerate")
            metrics.increment("verification_errors")
            return VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.0,
                should_regenerate=True,
                issues=[f"Verification error: {str(e)}"],
            )

    def _build_verify_prompt(self, query: str, draft: str, context: str) -> str:
        """Build the verification prompt with targeted context."""
        # Truncate context to keep verification cheap
        max_context_tokens = 4000
        if estimate_tokens(context) > max_context_tokens:
            # Keep beginning and end (most relevant parts)
            words = context.split()
            max_words = int(max_context_tokens / 1.3)
            half = max_words // 2
            context = " ".join(words[:half]) + "\n\n[...]\n\n" + " ".join(words[-half:])

        return f"""ORIGINAL QUESTION:
{query}

DRAFT ANSWER (to verify):
{draft}

SOURCE CONTEXT:
{context}

Verify the draft answer against the source context. Respond with JSON only."""

    def _parse_verification(self, response_text: str) -> VerificationResult:
        """Parse the verifier's JSON response."""
        try:
            # Clean potential markdown wrapping
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(clean)

            verdict = parsed.get("verdict", "insufficient_evidence")
            confidence = float(parsed.get("confidence", 0.5))

            # Determine if regeneration is needed
            should_regenerate = parsed.get("should_regenerate", False)
            if not should_regenerate:
                # Apply our own threshold
                if verdict == "incorrect":
                    should_regenerate = True
                elif verdict == "insufficient_evidence":
                    should_regenerate = True
                elif verdict == "partially_correct" and confidence < self.confidence_threshold:
                    should_regenerate = True

            return VerificationResult(
                verdict=verdict,
                confidence=confidence,
                issues=parsed.get("issues", []),
                missing_context=parsed.get("missing_chunks", []),
                should_regenerate=should_regenerate,
                verification_tokens=0,
                verification_cost=0.0,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning(f"Failed to parse verification response: {e}")
            return VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.0,
                should_regenerate=True,
                issues=["Verification response could not be parsed"],
            )

    def should_verify(self, tier: str, total_context_tokens: int,
                      query_complexity: str = "medium") -> bool:
        """
        Decide whether to run verification on this request.

        Verification adds latency + cost, so only use it when:
        - Context is large (high hallucination risk)
        - Task is high-value
        - Draft came from a small model

        Skip verification when:
        - Simple queries (greetings, lookups)
        - Already using premium model
        - Context is small and well-scoped
        """
        if not self.enabled:
            return False

        # Don't verify premium (already using best model)
        if tier == "premium":
            return False

        # Don't verify trivial tasks
        if tier in ("local", "cache_only"):
            return False

        # Verify when context is large (high hallucination risk)
        if total_context_tokens > 5000:
            return True

        # Verify medium-tier responses for complex queries
        if tier == "medium" and query_complexity in ("complex", "premium"):
            return True

        # Default: verify cheap responses with moderate+ context
        if tier == "cheap" and total_context_tokens > 2000:
            return True

        return False


# ─── Evidence Span Extraction ────────────────────────────────────────────────

def extract_evidence_refs(response_text: str) -> list[dict]:
    """
    Extract [ref:N] references from a response.
    Returns list of {"chunk_id": N, "claim": "text before ref"}.
    """
    pattern = re.compile(r'([^.!?\n]+?)\s*\[ref:(\w+)\]')
    refs = []
    for match in pattern.finditer(response_text):
        claim = match.group(1).strip()
        ref_id = match.group(2)
        try:
            chunk_id = int(ref_id) if ref_id != "NONE" else -1
        except ValueError:
            chunk_id = -1
        refs.append({"chunk_id": chunk_id, "claim": claim})
    return refs


def count_unsupported_claims(response_text: str) -> int:
    """Count claims marked as [ref:NONE] — needing full context."""
    return len(re.findall(r'\[ref:NONE\]', response_text))


# ─── Global Instance ────────────────────────────────────────────────────────

verification_layer = VerificationLayer()
