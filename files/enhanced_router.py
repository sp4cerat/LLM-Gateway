"""
LLM Gateway - Enhanced Router
===============================
"Split-Brain" routing: 4 layers of decision-making instead of LLM-only.

Execution order:
  1. Deterministic hard rules (token count, document size)
  2. Task-class heuristics (keyword patterns)
  3. Utility function (risk × value → budget allocation)
  4. LLM-based routing (only for ambiguous cases)

The router also outputs WHICH CHUNKS are needed (not just which tier).
This enables targeted retrieval instead of sending the full document.
"""

import re
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from models import RouterAction, RouterResult, GatewayConfig
from context import estimate_tokens
from context_mapper import ContextMap
from metrics import metrics

log = logging.getLogger("gateway.enhanced_router")


# ─── Enhanced Router Result ──────────────────────────────────────────────────

@dataclass
class EnhancedRouterResult:
    """Extended routing decision including chunk selection."""
    action: RouterAction = RouterAction.CHEAP
    confidence: float = 0.5
    response_type: str = "explanation_generic"
    reason: str = ""
    needed_chunks: list[int] = field(default_factory=list)
    strategy: str = "direct"  # direct, retrieve_then_solve, map_reduce, verify
    routing_layer: str = "unknown"  # deterministic, heuristic, utility, llm
    utility_score: float = 0.0
    is_code_generation: bool = False

    def to_router_result(self) -> RouterResult:
        """Convert to standard RouterResult for backward compatibility."""
        return RouterResult(
            action=self.action,
            confidence=self.confidence,
            response_type=self.response_type,
            reason=f"{self.routing_layer}:{self.reason}",
            is_code_generation=self.is_code_generation,
        )


# ─── Utility Function ───────────────────────────────────────────────────────

@dataclass
class RequestContext:
    """Context for utility-based routing decisions."""
    # Task properties
    query_tokens: int = 0
    context_tokens: int = 0
    document_type: str = "unknown"

    # Value signals
    customer_tier: str = "standard"  # standard, premium, enterprise
    task_priority: str = "normal"    # low, normal, high, critical

    # Risk signals
    context_complexity: float = 0.5  # 0=simple, 1=complex
    hallucination_risk: float = 0.3  # based on context size & type

    # Budget state
    budget_utilization: float = 0.0  # 0-1, how much of daily budget is spent


class UtilityRouter:
    """
    Computes a utility score for routing decisions.

    utility = value_factor × risk_factor × (1 - budget_pressure)

    High utility → use expensive model
    Low utility → use cheap model
    """

    # Thresholds for tier assignment
    PREMIUM_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.35
    CHEAP_THRESHOLD = 0.15

    # Value multipliers per customer/task tier
    VALUE_MULTIPLIERS = {
        ("enterprise", "critical"): 1.0,
        ("enterprise", "high"): 0.9,
        ("premium", "critical"): 0.85,
        ("enterprise", "normal"): 0.8,
        ("premium", "high"): 0.75,
        ("standard", "critical"): 0.7,
        ("premium", "normal"): 0.6,
        ("standard", "high"): 0.5,
        ("standard", "normal"): 0.3,
        ("standard", "low"): 0.15,
    }

    def compute_utility(self, ctx: RequestContext) -> float:
        """Compute utility score (0.0 to 1.0) for a request."""
        # Value factor
        value_key = (ctx.customer_tier, ctx.task_priority)
        value_factor = self.VALUE_MULTIPLIERS.get(value_key, 0.3)

        # Risk factor: how likely is the cheap model to hallucinate?
        risk_factor = self._compute_risk(ctx)

        # Budget pressure: higher when budget is nearly exhausted
        budget_pressure = self._budget_pressure(ctx.budget_utilization)

        # Composite utility
        utility = value_factor * risk_factor * (1.0 - budget_pressure * 0.5)

        return round(min(1.0, max(0.0, utility)), 3)

    def _compute_risk(self, ctx: RequestContext) -> float:
        """Estimate hallucination risk based on context properties."""
        risk = 0.2  # baseline

        # More context → higher risk of missing information
        if ctx.context_tokens > 50000:
            risk += 0.4
        elif ctx.context_tokens > 10000:
            risk += 0.25
        elif ctx.context_tokens > 3000:
            risk += 0.1

        # Document type risk — based on v3.2 benchmark data
        # Higher risk = more likely to escalate to medium/premium
        type_risk = {
            "legal": 0.35,       # High stakes → premium
            "code": 0.25,        # Medium 93% vs Cheap 96%, but hard code needs premium (99%)
            "reasoning": 0.35,   # Cheap 70% vs Medium 85% — BIG gap, always escalate
            "creative": 0.30,    # Cheap 83% vs Medium 92% — escalate to medium
            "translation": 0.15, # All tiers 96% — cheap is fine for most
            "log": 0.1,          # Structured, pattern-based
            "prose": 0.2,        # Moderate
            "structured": 0.1,
            "vision": 0.15,      # All tiers 94% — medium handles well
        }
        risk += type_risk.get(ctx.document_type, 0.15)

        # Complexity adds risk
        risk += ctx.context_complexity * 0.2

        return min(1.0, risk)

    def _budget_pressure(self, utilization: float) -> float:
        """
        Budget pressure curve:
          0-50% spent → no pressure
          50-80% → linear ramp
          80-100% → strong pressure
        """
        if utilization < 0.5:
            return 0.0
        elif utilization < 0.8:
            return (utilization - 0.5) / 0.3 * 0.5  # 0→0.5
        else:
            return 0.5 + (utilization - 0.8) / 0.2 * 0.5  # 0.5→1.0

    def recommend_tier(self, utility: float) -> RouterAction:
        """Map utility score to tier."""
        if utility >= self.PREMIUM_THRESHOLD:
            return RouterAction.PREMIUM
        elif utility >= self.MEDIUM_THRESHOLD:
            return RouterAction.MEDIUM
        elif utility >= self.CHEAP_THRESHOLD:
            return RouterAction.CHEAP
        else:
            return RouterAction.LOCAL


# ─── Patterns for Global Reasoning Detection ────────────────────────────────

GLOBAL_REASONING_PATTERNS = [
    re.compile(r'\b(widersprüch|contradiction|inconsisten|discrepan)', re.I),
    re.compile(r'\b(gesamte[rsnm]?|entire|whole|all of|throughout|across all)\b', re.I),
    re.compile(r'\b(vergleich|compar|contrast|differ|versus)\b.*\b(alle|all|every|each)\b', re.I),
    re.compile(r'\b(zusammenfass|summar|overview|überblick)\b.*\b(gesamt|complete|full|entire)\b', re.I),
]

RETRIEVAL_PATTERNS = [
    re.compile(r'\b(klausel|clause|section|paragraph|abschnitt|§)\s*\d', re.I),
    re.compile(r'\b(wo steht|where does|find|locate|suche)\b.*\b(im text|in the|im dokument)\b', re.I),
    re.compile(r'\b(definition|define|bedeut|meaning)\s+(von|of)\b', re.I),
]

SIMPLE_PATTERNS = [
    re.compile(r'^(hi|hello|hey|hallo|moin|servus)\b', re.I),
    re.compile(r'^(was ist|what is|define|erkläre)\s+\w+\??$', re.I),
    re.compile(r'^(danke|thanks|thank you|ok|gut)\b', re.I),
]


# ─── Enhanced Router ─────────────────────────────────────────────────────────

class EnhancedRouter:
    """
    Multi-layer router with utility-based budget allocation.

    Layer 1: Deterministic hard rules
    Layer 2: Task-class heuristics  
    Layer 3: Utility function (risk × value)
    Layer 4: LLM classification (fallback for ambiguous cases)
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.utility_router = UtilityRouter()
        self.llm_router = None  # Set externally (IntentRouter)

    def set_llm_router(self, router):
        """Set the LLM-based router for Layer 4 fallback."""
        self.llm_router = router

    async def route(
        self,
        query: str,
        context_map: Optional[ContextMap] = None,
        request_context: Optional[RequestContext] = None,
    ) -> EnhancedRouterResult:
        """
        Multi-layer routing decision.

        Args:
            query: User's question
            context_map: Structural map of the document (from ContextMapper)
            request_context: Metadata about the request (customer tier, budget, etc.)
        """
        start = time.time()
        ctx = request_context or RequestContext()

        # Enrich context from map
        if context_map:
            ctx.context_tokens = context_map.total_tokens_est
            ctx.document_type = context_map.document_type
            ctx.context_complexity = self._estimate_complexity(context_map)

        # ─── Layer 1: Deterministic Hard Rules ────────────────────────
        result = self._deterministic_route(query, ctx, context_map)
        if result:
            self._log_decision(result, start)
            return result

        # ─── Layer 2: Task-Class Heuristics ───────────────────────────
        result = self._heuristic_route(query, ctx, context_map)
        if result:
            self._log_decision(result, start)
            return result

        # ─── Layer 3: Utility Function ────────────────────────────────
        utility = self.utility_router.compute_utility(ctx)
        tier = self.utility_router.recommend_tier(utility)

        # For medium utility, try LLM for better precision
        if 0.25 < utility < 0.65 and self.llm_router:
            # ─── Layer 4: LLM Classification ──────────────────────
            try:
                # Build enhanced query with context map info
                enhanced_query = query
                if context_map and context_map.chunk_count > 0:
                    enhanced_query = (
                        f"{context_map.to_router_prompt(max_lines=20)}\n\n"
                        f"QUERY: {query}"
                    )

                llm_result = await self.llm_router.route(enhanced_query)
                result = EnhancedRouterResult(
                    action=llm_result.action,
                    confidence=llm_result.confidence,
                    response_type=llm_result.response_type,
                    reason=llm_result.reason,
                    routing_layer="llm",
                    utility_score=utility,
                    strategy=self._pick_strategy(llm_result.action, ctx, context_map),
                    is_code_generation=llm_result.is_code_generation,
                )
                self._log_decision(result, start)
                return result

            except Exception as e:
                log.warning(f"LLM router failed in enhanced routing: {e}")
                # Fall through to utility-based decision

        # ─── Layer 4b: Code Detection for low-utility queries ─────
        # For cheap-tier queries, still ask the LLM if this is code generation.
        # Don't override tier — only set is_code_generation flag.
        # Groq calls are ~free (<0.001ct) and ~50ms.
        _llm_code_flag = False
        if utility <= 0.25 and self.llm_router and tier in (
            RouterAction.CHEAP, RouterAction.LOCAL
        ):
            try:
                llm_result = await self.llm_router.route(query)
                _llm_code_flag = llm_result.is_code_generation
                if _llm_code_flag:
                    log.info(f"EnhancedRouter: LLM detected code_generation for low-utility query")
            except Exception:
                pass  # Non-critical, fall through

        # Use utility-based decision
        result = EnhancedRouterResult(
            action=tier,
            confidence=0.7,
            response_type="explanation_generic",
            reason=f"utility={utility:.2f}",
            routing_layer="utility",
            utility_score=utility,
            strategy=self._pick_strategy(tier, ctx, context_map),
            needed_chunks=self._select_relevant_chunks(query, context_map),
            is_code_generation=_llm_code_flag,
        )
        self._log_decision(result, start)
        return result

    # ─── Layer 1: Deterministic ──────────────────────────────────────────────

    def _deterministic_route(
        self, query: str, ctx: RequestContext, context_map: Optional[ContextMap]
    ) -> Optional[EnhancedRouterResult]:
        """Hard rules that always apply. No LLM needed."""

        # Rule 1: Context exceeds 80% of any model's window → force premium
        if ctx.context_tokens > 100_000:
            return EnhancedRouterResult(
                action=RouterAction.PREMIUM,
                confidence=0.99,
                reason="context_exceeds_100k",
                routing_layer="deterministic",
                strategy="direct",
            )

        # Rule 2: Empty/trivial query
        q = query.strip()
        if len(q) < 3:
            return EnhancedRouterResult(
                action=RouterAction.CHEAP,
                confidence=0.95,
                reason="trivial_query",
                routing_layer="deterministic",
                response_type="explanation_generic",
            )

        # Rule 3: Global reasoning detected → premium + full context
        for pattern in GLOBAL_REASONING_PATTERNS:
            if pattern.search(query):
                return EnhancedRouterResult(
                    action=RouterAction.PREMIUM,
                    confidence=0.90,
                    reason=f"global_reasoning:{pattern.pattern[:30]}",
                    routing_layer="deterministic",
                    strategy="direct",  # needs full context
                    needed_chunks=[],  # empty = send everything
                )

        return None

    # ─── Layer 2: Heuristics ─────────────────────────────────────────────────

    def _heuristic_route(
        self, query: str, ctx: RequestContext, context_map: Optional[ContextMap]
    ) -> Optional[EnhancedRouterResult]:
        """Pattern-based classification for clear-cut cases."""

        # Simple queries → cheap, no context needed
        for pattern in SIMPLE_PATTERNS:
            if pattern.match(query):
                return EnhancedRouterResult(
                    action=RouterAction.CHEAP,
                    confidence=0.90,
                    reason="simple_pattern",
                    routing_layer="heuristic",
                    response_type="explanation_generic",
                    strategy="direct",
                )

        # Retrieval-style queries → medium + targeted chunks
        for pattern in RETRIEVAL_PATTERNS:
            if pattern.search(query):
                chunks = self._select_relevant_chunks(query, context_map)
                return EnhancedRouterResult(
                    action=RouterAction.MEDIUM,
                    confidence=0.80,
                    reason=f"retrieval_pattern:{pattern.pattern[:30]}",
                    routing_layer="heuristic",
                    strategy="retrieve_then_solve",
                    needed_chunks=chunks,
                )

        return None

    # ─── Strategy Selection ──────────────────────────────────────────────────

    def _pick_strategy(
        self, tier: RouterAction, ctx: RequestContext, context_map: Optional[ContextMap]
    ) -> str:
        """
        Pick execution strategy:
          - "direct":              Send query + full context to model
          - "retrieve_then_solve": Retrieve specific chunks, then solve
          - "map_reduce":          Process chunks independently, then aggregate
          - "verify":              Draft with cheap, verify with expensive
        """
        if not context_map or ctx.context_tokens < 3000:
            return "direct"

        # Large context with cheap/medium model → verify pattern
        if tier in (RouterAction.CHEAP, RouterAction.MEDIUM) and ctx.context_tokens > 8000:
            return "verify"

        # Very large context → map_reduce if task is decomposable
        if ctx.context_tokens > 50000:
            return "map_reduce"

        # Medium context → targeted retrieval
        if ctx.context_tokens > 5000:
            return "retrieve_then_solve"

        return "direct"

    # ─── Chunk Selection ─────────────────────────────────────────────────────

    def _select_relevant_chunks(
        self, query: str, context_map: Optional[ContextMap]
    ) -> list[int]:
        """
        Select chunks likely relevant to the query.
        Uses keyword overlap between query and chunk keywords/headings.
        """
        if not context_map or not context_map.chunks:
            return []

        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return []

        scored = []
        for chunk in context_map.chunks:
            chunk_words = set(
                w.lower() for w in chunk.keywords
            ) | set(
                w.lower() for h in chunk.headings
                for w in re.findall(r'\b\w{3,}\b', h)
            )

            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((chunk.chunk_id, overlap))

        # Sort by relevance, return top chunks
        scored.sort(key=lambda x: -x[1])
        max_chunks = 5  # Don't retrieve too many
        return [cid for cid, _ in scored[:max_chunks]]

    # ─── Complexity Estimation ───────────────────────────────────────────────

    def _estimate_complexity(self, context_map: ContextMap) -> float:
        """Estimate context complexity (0-1) from the map."""
        complexity = 0.0

        # More chunks = more complex
        if context_map.chunk_count > 20:
            complexity += 0.3
        elif context_map.chunk_count > 10:
            complexity += 0.2
        elif context_map.chunk_count > 5:
            complexity += 0.1

        # Diverse headings = more structured = more complex reasoning
        if len(context_map.global_headings) > 15:
            complexity += 0.2
        elif len(context_map.global_headings) > 5:
            complexity += 0.1

        # Document type
        if context_map.document_type == "legal":
            complexity += 0.3
        elif context_map.document_type == "code":
            complexity += 0.15

        return min(1.0, complexity)

    # ─── Logging ─────────────────────────────────────────────────────────────

    def _log_decision(self, result: EnhancedRouterResult, start_time: float):
        latency = (time.time() - start_time) * 1000
        metrics.histogram("enhanced_router_latency_ms", latency)
        metrics.increment(
            "enhanced_router_decisions",
            tags={"tier": result.action.value, "layer": result.routing_layer, "strategy": result.strategy}
        )
        log.info(
            f"EnhancedRouter: {result.action.value} | layer={result.routing_layer} | "
            f"strategy={result.strategy} | conf={result.confidence:.2f} | "
            f"utility={result.utility_score:.2f} | "
            f"code_gen={result.is_code_generation} | "
            f"chunks={result.needed_chunks} | "
            f"reason={result.reason} | {latency:.0f}ms"
        )
