"""
LLM Gateway - Intent Router
Groq-based fast intent classification with fallback chain.
"""

import json
import time
import logging
import httpx
from typing import Optional
from models import RouterAction, RouterResult, GatewayConfig
from metrics import metrics

log = logging.getLogger("gateway.router")


# ─── Router System Prompt ─────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for coding and general requests.

Classify the request into EXACTLY ONE category:

CACHE_ONLY - Request is too vague/unclear for a meaningful answer
  Examples: "help", "?", "code", "fix it"

LOCAL - Trivial questions that don't require analysis
  Examples: "What does HTTP 404 mean?", "What's the command for...", "Hello", "Hallo"

CHEAP - Simple lookups, greetings, short factual answers
  Examples: "What port does MySQL use?", "Hi!", "What version of Python?"

CHEAP_PLUS - Questions requiring CURRENT/REAL-TIME data (weather, stocks, news, live scores)
  Examples: "What's the weather in Berlin?", "Current Bitcoin price?", "Latest news about...", "How's the DAX doing today?"

MEDIUM - Explanations, documentation, moderate code, general best practices
  Examples: "Explain async/await to me", "What's the difference between...", "Write a short function that..."

PREMIUM - Complex code generation, large patches, architecture, project-specific tasks
  Examples: "Write a full REST API for...", "Refactor this entire module...", "Review this code...", "Design a system for..."

Note: Queries may be in any language. Greetings and small talk in any language are CHEAP.
Keywords like "aktuell", "currently", "today", "live", "jetzt", "weather", "Wetter", "Kurs", "stock" → CHEAP_PLUS.

is_code_generation: Set to true when the user asks to CREATE, GENERATE, WRITE, BUILD, or MODIFY code, 
  a website, HTML, CSS, JavaScript, a script, a program, an app, a component, or any software artifact.
  This includes requests in ANY language (German: "erstelle", "generiere", "programmiere", "baue";
  French: "créez", "générez"; etc.). Set to false for questions ABOUT code, explanations, or non-code tasks.

Reply ONLY with JSON:
{"action": "...", "confidence": 0.0-1.0, "response_type": "...", "is_code_generation": true/false, "reason": "..."}

response_type must be one of:
- explanation_generic
- explanation_contextual
- code_suggestion
- code_review
- command_execution
- documentation"""


# ─── Groq Router ──────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Multi-provider intent router with fallback chain.
    Primary: Groq/OpenRouter (fast, cheap) → Fallback: rule-based heuristics
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.api_key: Optional[str] = None
        self.router_model = config.providers.router_model
        self.router_provider = config.providers.router_provider
        self.timeout = 5.0

    def set_api_key(self, key: str):
        self.api_key = key

    async def route(self, query: str, context: str = "") -> RouterResult:
        """
        Classify intent with resilient fallback chain.
        Groq/OpenRouter → Heuristic → Default (Premium)
        """
        start = time.time()

        # Try LLM classification (Groq or OpenRouter)
        if self.api_key and not self.config.mock_mode:
            try:
                result = await self._llm_classify(query, context)
                latency = (time.time() - start) * 1000
                metrics.histogram("router_latency_ms", latency)
                metrics.increment("router_success", tags={"provider": self.router_provider})
                log.debug(f"Router: {result.action.value} ({result.confidence:.2f}) - {result.reason}")
                return result
            except Exception as e:
                log.warning(f"{self.router_provider} router failed: {e}, falling back to heuristics")
                metrics.increment("router_fallback", tags={"from": self.router_provider, "to": "heuristic"})

        # Fallback: Rule-based heuristics
        result = self._heuristic_classify(query)
        latency = (time.time() - start) * 1000
        metrics.histogram("router_latency_ms", latency)
        metrics.increment("router_success", tags={"provider": "heuristic"})
        log.debug(f"Router (heuristic): {result.action.value} - {result.reason}")
        return result

    async def _llm_classify(self, query: str, context: str = "") -> RouterResult:
        """Classify via Groq or OpenRouter (OpenAI-compatible API)."""
        user_content = f"Context: {context}\n\nQuery: {query}" if context else query

        # Select API endpoint based on router provider
        if self.router_provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            # Default to Groq
            base_url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.router_provider == "openrouter":
            headers["X-Title"] = "LLM Gateway Router"

        request_body = {
            "model": self.router_model,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 150,
            "temperature": 0,
        }
        # Groq supports response_format, OpenRouter may not for all models
        if self.router_provider == "groq":
            request_body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                base_url,
                headers=headers,
                json=request_body,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result_text = response.json()["choices"][0]["message"]["content"]
            # Parse JSON (handle potential markdown wrapping)
            clean_text = result_text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(clean_text)

            # Track router token usage
            usage = response.json().get("usage", {})
            router_tokens = usage.get("total_tokens", 0)
            router_cost = router_tokens * (0.06 / 1_000_000)  # approx cost
            metrics.increment("router_tokens", router_tokens)
            metrics.increment("router_cost_usd", router_cost)

            return RouterResult(
                action=RouterAction(parsed.get("action", "cheap").lower()),
                confidence=float(parsed.get("confidence", 0.8)),
                response_type=parsed.get("response_type", "explanation_generic"),
                reason=parsed.get("reason", f"{self.router_provider}_classified"),
                is_code_generation=bool(parsed.get("is_code_generation", False)),
            )

    def _heuristic_classify(self, query: str) -> RouterResult:
        """
        Rule-based fallback classifier.
        Fast, free, works offline. Covers ~80% of cases correctly.
        """
        q = query.lower().strip()

        # Greetings (multilingual)
        greetings = {"hi", "hello", "hey", "hallo", "moin", "servus", "grüezi",
                     "guten tag", "guten morgen", "guten abend", "na", "huhu",
                     "ciao", "salut", "hola", "bonjour"}
        if q in greetings or q.startswith(("wie geht", "was geht", "how are you",
                                           "whats up", "what's up")):
            return RouterResult(
                action=RouterAction.CHEAP,
                confidence=0.95,
                response_type="explanation_generic",
                reason="greeting",
            )

        # Too vague
        if len(q) < 5 or q in ("help", "?"):
            return RouterResult(
                action=RouterAction.CACHE_ONLY,
                confidence=0.9,
                response_type="explanation_generic",
                reason="too_vague",
            )

        # Real-time / current data queries → cheap_plus (Gemini 3 Flash)
        realtime_patterns = [
            "wetter", "weather", "temperatur", "temperature", "forecast",
            "aktienkurs", "stock price", "börsenkurs", "exchange rate",
            "wechselkurs", "bitcoin price", "crypto price", "kurs von",
            "aktuelle nachrichten", "latest news", "breaking news",
            "spielstand", "score of", "ergebnis von",
            "what time is it", "wie spät", "uhrzeit",
            "öffnungszeiten", "opening hours",
            # Research/analysis queries that need web data
            "recherchiere", "recherche zu", "recherche über",
            "aktuelle studien", "aktuelle quellen", "nutze quellen",
            "nutze aktuelle", "finde heraus", "finde informationen",
            "suche nach", "search for", "look up", "find out",
            "research", "investigate",
        ]
        realtime_keywords = [
            "aktuell", "currently", "right now", "gerade", "heute",
            "today", "live", "jetzt", "derzeit", "momentan",
        ]
        for pattern in realtime_patterns:
            if pattern in q:
                return RouterResult(
                    action=RouterAction.CHEAP_PLUS,
                    confidence=0.90,
                    response_type="explanation_generic",
                    reason=f"heuristic_realtime:{pattern}",
                )
        # Check for realtime keywords combined with question words
        has_question = any(w in q for w in ["was", "wie", "what", "how", "wieviel", "how much", "which", "welche"])
        for kw in realtime_keywords:
            if kw in q and has_question:
                return RouterResult(
                    action=RouterAction.CHEAP_PLUS,
                    confidence=0.75,
                    response_type="explanation_generic",
                    reason=f"heuristic_realtime_keyword:{kw}",
                )

        # Premium indicators (complex code tasks)
        premium_patterns = [
            "write a full", "write an entire", "implement a complete",
            "refactor the entire", "refactor this module",
            "design a system", "architect", "full rest api",
            "review this code", "review the code",
            "multi-file", "migration", "scaffold",
            "rewrite the entire", "rewrite this module",
        ]
        for pattern in premium_patterns:
            if pattern in q:
                return RouterResult(
                    action=RouterAction.PREMIUM,
                    confidence=0.85,
                    response_type="code_suggestion",
                    reason=f"heuristic_premium:{pattern}",
                    is_code_generation=True,
                )

        # Code generation indicators (multilingual)
        # Technology keywords that strongly signal code creation regardless of language
        _code_tech = ["html", "css", "javascript", "python", "react", "vue",
                      "typescript", "json", "yaml", "sql", "api", "rest",
                      "webseite", "website", "webpage", "app", "script",
                      "component", "funktion", "function", "klasse", "class"]
        _code_verbs = ["schreib", "erstell", "generier", "programmier", "bau",
                       "entwickl", "mach", "erzeug", "implementier",
                       "write", "create", "generate", "build", "make",
                       "code", "develop", "implement",
                       "créez", "générez", "écrivez",  # French
                       "crea", "genera", "scrivi",     # Italian/Spanish
                       ]
        _has_tech = any(t in q for t in _code_tech)
        _has_verb = any(v in q for v in _code_verbs)
        if _has_tech and _has_verb:
            return RouterResult(
                action=RouterAction.MEDIUM,
                confidence=0.85,
                response_type="code_suggestion",
                reason=f"heuristic_code_gen",
                is_code_generation=True,
            )

        # Medium indicators (moderate code + explanations)
        medium_patterns = [
            "write a function", "write a class", "write a script",
            "write code", "implement", "refactor", "fix the bug",
            "fix this", "debug", "patch", "modify the", "change the",
            "create a component", "build a", "optimize this",
            "rewrite", "convert this", "generate",
            "explain", "what is", "what are", "what does",
            "how does", "how do", "difference between",
            "tell me about", "describe", "tutorial", "guide",
            "why is", "why does", "when should",
        ]
        for pattern in medium_patterns:
            if pattern in q:
                _is_explanation = "explain" in pattern or "what" in pattern or "how" in pattern or "why" in pattern
                return RouterResult(
                    action=RouterAction.MEDIUM,
                    confidence=0.75,
                    response_type="explanation_generic" if _is_explanation else "code_suggestion",
                    reason=f"heuristic_medium:{pattern}",
                    is_code_generation=not _is_explanation,
                )

        # Cheap: simple lookups
        cheap_patterns = [
            "what command", "how to install", "shortcut for",
            "what port", "what version", "list of",
            "syntax for", "flag for", "define",
        ]
        for pattern in cheap_patterns:
            if pattern in q:
                return RouterResult(
                    action=RouterAction.CHEAP,
                    confidence=0.7,
                    response_type="explanation_generic",
                    reason=f"heuristic_cheap:{pattern}",
                )

        # Default: Medium (safe middle-ground)
        return RouterResult(
            action=RouterAction.MEDIUM,
            confidence=0.5,
            response_type="explanation_generic",
            reason="heuristic_default",
        )
