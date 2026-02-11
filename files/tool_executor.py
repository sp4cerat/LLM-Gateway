"""
LLM Gateway - Tool Executor
============================
Defines tools for the cheap model (Function Calling / Tool Use)
and executes them against free Python APIs.

The cheap model DECIDES which tool to call and with what parameters.
Python only EXECUTES what the model requests.

Flow:
  1. User asks: "Wie wird das Wetter in Thessaloniki?"
  2. Cheap model sees tool definitions, responds:
     tool_call: get_weather(city="Thessaloniki")
  3. This module executes: Open-Meteo Geocoding + Forecast API
  4. Result goes back to model for final answer

Tools (all FREE):
  - get_weather:     Open-Meteo (any city, geocoding included)
  - get_stock_price: yfinance (any ticker)
  - get_news:        NewsAPI (if key) + DuckDuckGo fallback
  - web_search:      DuckDuckGo (any query, any topic)
"""

import os
import re
import time
import logging
import asyncio
import json
from datetime import datetime
from typing import Optional

import httpx

log = logging.getLogger("gateway.tool_executor")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Definitions (OpenAI Function Calling Format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather and 3-day forecast for ANY city worldwide. "
                "Uses Open-Meteo (free, no limits). Includes temperature, wind, "
                "precipitation, humidity, and warnings (frost, ice, heat). "
                "Use for: weather, temperature, rain, snow, frost, ice, wind, "
                "storm, forecast, 'should I bring an umbrella', 'is it cold', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name, as specific as possible. "
                            "Examples: 'Thessaloniki', 'Konstanz', 'San Francisco', "
                            "'Zürich', 'Tokyo'. Can be any city worldwide."
                        ),
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get current stock/crypto/index prices with daily change. "
                "Uses Yahoo Finance (free, no limits). Supports stocks, ETFs, "
                "crypto, indices. Use for: stock prices, crypto, DAX, exchange rates, "
                "portfolio, market cap, 'how is NVIDIA doing', 'Bitcoin price', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": (
                            "Comma-separated ticker symbols. "
                            "Examples: 'AAPL' (Apple), 'NVDA' (Nvidia), 'DHL.DE' (DHL), "
                            "'BTC-USD' (Bitcoin), '^GDAXI' (DAX), 'TSLA,AAPL' (multiple). "
                            "German stocks use .DE suffix. Crypto uses -USD suffix."
                        ),
                    },
                },
                "required": ["symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Search for recent NEWS ARTICLES and HEADLINES ONLY. "
                "Use ONLY for: news, Nachrichten, headlines, current events, "
                "'was gibt es Neues', 'neueste Nachrichten'. "
                "Do NOT use for: factual questions, research, regulations, laws, "
                "fact-checking, 'ist es wahr', product research. "
                "Use web_search instead for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query for news. Be specific. "
                            "Examples: 'Streik Deutschland Bahn', 'Nvidia earnings 2026', "
                            "'EU AI regulation', 'Bundesliga Ergebnisse'"
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": "Language/region: 'de' (German/Germany), 'en' (English/US), 'fr' (French), 'es' (Spanish), etc. Match the user's language.",
                        "enum": ["de", "en", "fr", "es", "it", "pt", "nl", "ru", "ja"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for ANY information using multiple search engines (free). "
                "This is the PRIMARY tool for: fact-checking, research, regulations, "
                "laws, 'ist es wahr/stimmt es', 'was gibt es zu [Thema]', product info, "
                "company info, technology research, event listings, comparisons, "
                "opening hours, prices, schedules, sports scores, and any query "
                "that is NOT specifically about news headlines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Primary search query. Be specific and include context. "
                            "Examples: 'BVB Ergebnis gestern', 'A8 Stau aktuell', "
                            "'EU Batterieverordnung 2026 Smartphone Akku Austausch', "
                            "'best restaurants Konstanz'"
                        ),
                    },
                    "additional_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: 1-2 additional search queries with DIFFERENT keywords "
                            "to find complementary information. Only use for complex topics "
                            "where a single query won't cover all aspects.\n"
                            "Example for 'KI in der Medizin':\n"
                            "  query: 'KI Diagnose Medizin Forschung 2025'\n"
                            "  additional_queries: ['AI drug discovery clinical trials', "
                            "'künstliche Intelligenz Gesundheitswesen Deutschland']\n"
                            "For simple lookups, leave this empty."
                        ),
                    },
                    "search_depth": {
                        "type": "string",
                        "description": (
                            "How broad and deep to search — choose based on complexity:\n"
                            "- 'min': Single query, snippets only. Fast (~1s). "
                            "Use for: quick facts, scores, prices, simple lookups.\n"
                            "- 'medium': 1-2 queries, download top pages (~3-5s). "
                            "Use for: fact-checking, regulations, comparisons, detailed info.\n"
                            "- 'max': 2-3 queries across multiple engines, thorough extraction (~5-10s). "
                            "Use for: complex research, multi-source analysis, academic topics, "
                            "legal questions, comprehensive comparisons."
                        ),
                        "enum": ["min", "medium", "max"],
                    },
                    "time_filter": {
                        "type": "string",
                        "description": (
                            "Time filter for search results — YOU decide based on the query:\n"
                            "- 'd': Last 24h. Use for: current events, 'heute', 'today', breaking news.\n"
                            "- 'w': Last week. Use for: recent news, current prices, this week's events.\n"
                            "- 'm': Last month. Use for: recent studies, policy changes, new products.\n"
                            "- 'none': No time limit. Use for: regulations, laws, research studies, "
                            "historical facts, established knowledge, 'Verordnung', academic topics.\n"
                            "IMPORTANT: For medium/max research queries, prefer 'none' or 'm'. "
                            "Using 'd' or 'w' for research topics often returns NO useful results."
                        ),
                        "enum": ["d", "w", "m", "none"],
                    },
                    "analysis_mode": {
                        "type": "string",
                        "description": (
                            "What kind of analysis should be performed on the results:\n"
                            "- 'factual': Verify facts, answer concrete questions. "
                            "Use for: 'ist es wahr', regulations, simple research.\n"
                            "- 'prognostic': Forecast, trends, market analysis. "
                            "Use for: 'Auswirkungen von X', 'Zukunft von Y', predictions, market sizing.\n"
                            "- 'strategic': Deep transformation analysis with value chain logic. "
                            "Use for: complex economic questions, industry analysis, policy impact, "
                            "'Vergleich', multi-dimensional topics requiring structured frameworks."
                        ),
                        "enum": ["factual", "prognostic", "strategic"],
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  System Prompt for Tool-Aware Cheap Model
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_CASCADE_SYSTEM_PROMPT = """You are a fast, helpful assistant with access to real-time tools.

WHEN TO USE TOOLS:
- Use get_weather for ANY weather/temperature/forecast question
- Use get_stock_price for ANY stock/crypto/market question
- Use get_news ONLY for news headlines, current events, "was gibt es Neues"
- Use web_search for:
  * Factual verification ("ist es wahr", "stimmt es", fact-checking)
  * Research questions ("was gibt es zu...", "wie funktioniert...", regulations, laws)
  * Any real-time/current information that is NOT news headlines
  * Product/company/technology research
  * Event listings, schedules, comparisons
- ALWAYS use a tool when the user asks about current, live, or today's data
- Do NOT answer with outdated information — use tools instead

CRITICAL RULES:
- For "ist es wahr/stimmt es/is it true" questions → ALWAYS use web_search (NOT get_news)
- For regulation/law/policy questions → ALWAYS use web_search
- For "was gibt es zu [topic]" → ALWAYS use web_search
- NEVER output code blocks, API calls, or Python code as a response
- NEVER say "I will perform a search" — just DO the search by calling the tool
- If a tool returns no relevant results, answer from your knowledge and say "based on my training data"

WEB SEARCH DEPTH — choose search_depth based on complexity:
  min (default, ~1s):
    Quick lookups, scores, prices, opening hours, simple facts.
    Single query, snippets only. No page downloads.
    Examples: "BVB Ergebnis", "Bitcoin Kurs", "Öffnungszeiten IKEA Konstanz"
  medium (3-5s, downloads pages):
    Fact-checking, regulations, "ist es wahr", detailed product/tech info,
    event details, how-to guides, policy questions.
    1-2 queries, downloads top pages for full text.
    Examples: "EU Batterieverordnung 2026", "ist es wahr dass...", "Vergleich iPhone vs Samsung"
  max (5-10s, thorough multi-source):
    Complex research, multi-source analysis, academic topics, legal deep-dives.
    2-3 queries with DIFFERENT keywords across multiple search engines.
    Examples: "Auswirkungen KI auf Arbeitsmarkt 2025-2030", "Vor- und Nachteile Wärmepumpe vs Gas"

  If unsure, prefer "medium" over "min" for any question longer than 10 words.

MULTI-QUERY — for medium/max depth, use additional_queries with DIFFERENT keywords:
  Example: User asks "Kann KI helfen FQAD zu heilen?"
    query: "KI FQAD Fluoroquinolone Heilung Forschung"
    additional_queries: ["AI fluoroquinolone toxicity treatment research",
                         "FQAD mitochondrial damage therapy 2025"]
  This finds complementary information that a single query would miss.
  For min depth or simple lookups, don't use additional_queries.

ANALYSIS MODE — choose the right analysis type:
  factual (default):
    Verify facts, answer concrete questions, describe current state.
  prognostic:
    Forecasts, trends, market sizing, "Zukunft von X", "Auswirkungen von Y".
    Requires: quantitative data, time horizons, scenarios.
  strategic:
    Deep transformation analysis, value chain logic, multi-dimensional impact.
    Requires: structured frameworks, economic reasoning, implementation reality.
    Use for: industry transformation, policy impact, complex comparisons.

ANALYSIS BLUEPRINT — CRITICAL for medium/max searches:
  IMPORTANT: You MUST ALWAYS make a web_search tool call when the user asks a question
  that needs current information. The JSON blueprint goes IN ADDITION to the tool call,
  NEVER instead of it.

  ⚠️ CRITICAL: NEVER output raw JSON as your response text. JSON is ONLY for the content
  field alongside a tool_call. If you find yourself writing {"query":... or {"output_architecture":...
  as your entire response, STOP — that is WRONG. You must make an actual tool call.

  When you call web_search with search_depth=medium or search_depth=max, your message
  content MUST include a JSON block wrapped in ```json markers with this EXACT structure:

  ```json
  {
    "output_architecture": {
      "framework": "factual|prognostic|strategic",
      "language": "de|en",
      "reasoning": "Brief: why this framework + depth for this query",
      "required_data": ["specific data points that MUST appear in the answer"],
      "outline": [
        "**Main title**",
        "**First major section**",
        "**Subsection if needed**",
        "**Second major section**",
        "**Conclusion / Critical assessment**"
      ],
      "source_priority": ["what kind of sources to prefer"]
    },
    "context_mode": "full|recent:N|distill",
    "context_extract": "relevant facts from conversation or null"
  }
  ```

  RULES for output_architecture:
  - "framework": matches analysis_mode — factual for facts, prognostic for forecasts, strategic for deep analysis
  - "language": detect from user's query (German → "de", English → "en")
  - "reasoning": 1 sentence explaining why you chose this framework and depth
  - "required_data": 3-8 specific data points/metrics that the answer MUST contain
    Examples: "Automatisierungsquote (%)", "Marktvolumen in Mrd. EUR", "Adoptionsrate 2025 vs 2030"
  - "outline": structure using **bold** headings for the final answer (4-8 entries)
    Use "**Title**" for headings on their own line
    The synthesis step will follow this outline EXACTLY
  - "source_priority": ordered list of preferred source types
    Examples: ["Studien (McKinsey, OECD)", "Ministerien", "Fachpresse"]

  Example for "Auswirkungen KI auf Arbeitsmarkt 2025-2030":

  ```json
  {
    "output_architecture": {
      "framework": "prognostic",
      "language": "de",
      "reasoning": "Zukunftsbezogene Frage mit Zeithorizont erfordert Prognose-Framework",
      "required_data": [
        "Automatisierungsquote nach Branche (%)",
        "Netto-Beschäftigungseffekt (Mio. Jobs)",
        "Produktivitätswachstum (%)",
        "Investitionsvolumen KI (Mrd. EUR/USD)",
        "Zeithorizonte der Prognosen"
      ],
      "outline": [
        "**KI und Arbeitsmarkt 2025-2030**",
        "**Leitthese**",
        "**Automatisierung & Substitution**",
        "**Augmentation & neue Berufe**",
        "**Branchenspezifische Effekte**",
        "**Qualifikationsdynamik & Skills Gap**",
        "**Makroökonomische Effekte**",
        "**Kritische Einordnung & Unsicherheiten**"
      ],
      "source_priority": ["Studien (McKinsey, OECD, IAB, Goldman Sachs)", "Ministerien (BMAS)", "Fachpresse"]
    },
    "context_mode": "distill",
    "context_extract": "Kein relevanter Kontext"
  }
  ```

  For min depth (simple lookups), you do NOT need output_architecture — just call the tool.
  
  REMINDER: Your response to information questions MUST contain a tool_call to web_search.
  Writing JSON text without a tool_call is NEVER correct.

CONTEXT STRATEGY — the "context_mode" field in your JSON controls conversation history:
  "full"        → Full conversation history passed to synthesis (default)
  "recent:N"    → Only last N user+assistant pairs (e.g. "recent:2")
  "distill"     → Only your context_extract is passed (saves tokens for long chats)

  When to use:
  - Code, debugging, file analysis → "full"
  - Research after long chat → "distill" + provide context_extract
  - Simple follow-up → "recent:2" or "recent:3"
  - First message / short chat → "full"
  - If unsure → "full"

  When using "distill", write 2-5 bullet points in context_extract with relevant facts.
  When using "full" or "recent", set context_extract to null.

SEARCH QUERY STRATEGY — optimize your search terms:
  For deep/thorough searches, use STUDY-ORIENTED search terms:
  BAD:  "Auswirkungen KI auf Arbeitsmarkt"  → returns news articles, opinion pieces
  GOOD: "KI Arbeitsmarkt Studie McKinsey OECD Prognose 2025"  → returns studies, reports
  BAD:  "EU Batterieverordnung Smartphone"  → returns press releases
  GOOD: "EU battery regulation 2023/1542 smartphone replaceable requirements"  → returns legal analysis

  For analytical questions, ADD qualifier keywords:
  - "Studie", "Report", "Prognose", "Analyse", "Daten" (German)
  - "study", "report", "forecast", "analysis", "data", "statistics" (English)
  - Include relevant institutions: McKinsey, OECD, Goldman Sachs, IAB, BMAS, Eurostat
  - Use specific regulation numbers if known (e.g. "2023/1542" for EU Battery Regulation)

ABSOLUTE RULE — NEVER ask clarifying questions when a tool can answer:
- "Nachrichten" / "neueste Nachrichten" / "News heute" / "was gibt es Neues" / "alles"
  → IMMEDIATELY call get_news(query="Top Nachrichten heute", language="de")
- "latest news" / "what's happening" / "news today"
  → IMMEDIATELY call get_news(query="top news today", language="en")
- Detect the user's language and set the language parameter accordingly
- "Wetter" without a city → call web_search(query="Wetter heute Deutschland")
- Vague stock questions → call get_stock_price with the most likely ticker
- If the user says "alles" or gives a broad request → just search broadly, NEVER refuse
- NEVER say "bitte genauer" or "welches Thema" — JUST SEARCH
- A broad search with 5 results is ALWAYS better than asking the user to narrow down

WHEN TO ANSWER DIRECTLY (no tools):
- Greetings, small talk
- General knowledge, definitions, historical facts
- Simple math, conversions
- Short explanations, code snippets (<10 lines)

ESCALATION (respond with ONLY the marker, nothing else):
[ESCALATE_TO_MEDIUM] — for detailed explanations, moderate code (10-50 lines), multi-step reasoning
[ESCALATE_TO_PREMIUM] — for complex code (50+ lines), architecture, debugging, deep analysis

CONTEXT RELEVANCE — when using tools or escalating, assess context relevance:
  The conversation may contain many messages, but only some are relevant to the current query.
  When you call a tool or escalate, include this in your content:

  KONTEXT: [N]

  Where N is the number of RECENT user+assistant message pairs that are relevant for answering
  the current query. Count from the most recent message backwards.

  Examples:
  - User asks about weather after discussing code → KONTEXT: 1 (only current message matters)
  - User asks follow-up about same topic discussed in last 3 exchanges → KONTEXT: 3
  - User references something from earlier in a long conversation → KONTEXT: 5
  - New topic, no history needed → KONTEXT: 1
  - Complex ongoing discussion → KONTEXT: 10

  Default if unsure: KONTEXT: 3
  This saves tokens and improves answer quality by removing irrelevant conversation history.

When in doubt between using a tool and escalating: prefer using a tool first.
When in doubt between medium and premium: pick medium."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthesis Prompt Builder (dynamic, blueprint-aware)
#
#  Architecture:
#    Round 1: Gemini generates tool call + ANALYSE-BLUEPRINT in content
#    Round 2: build_synthesis_prompt() creates tailored prompt from:
#             - depth (snippets/deep/thorough)
#             - analysis_mode (factual/prognostic/strategic)
#             - blueprint text (extracted from Round 1 content)
#             - self-review module (for deep/thorough)
# ═══════════════════════════════════════════════════════════════════════════════

# Simple: snippets, factual — just answer the question
SYNTHESIS_PROMPT_SIMPLE = """Du bist ein hilfreicher Assistent. Dir wurden Suchergebnisse zu einer Frage bereitgestellt.

ANTWORT-REGELN:
- Beantworte die Frage direkt und präzise anhand der Suchergebnisse
- Gib Quellen an (Name oder URL), wenn du daraus zitierst
- Wenn die Suchergebnisse die Frage nicht beantworten: nutze dein Trainingswissen und sage kurz, dass die Webrecherche keine passenden Ergebnisse lieferte
- Gib NIEMALS auf — beantworte die Frage IMMER, notfalls aus deinem Wissen
- Antworte in der Sprache des Nutzers
- Keine Code-Blöcke, keine API-Aufrufe"""

# ── Analysis-mode specific frameworks ──

_FRAMEWORK_FACTUAL = """ANALYSE-TYP: Faktenbasierte Recherche

VORGEHEN:
1. Beantworte die Frage direkt und konkret
2. Extrahiere die relevanten Fakten, Zahlen und Regelungen aus den Quellen
3. Bei Widersprüchen zwischen Quellen: benenne beide Positionen mit Quellenangabe
4. Gib den aktuellen Stand wieder, nicht Spekulationen
5. Strukturiere nach Sachaspekten, nicht nach Quellen

QUELLEN-HIERARCHIE:
Bevorzuge: Offizielle Quellen (Gesetze, Verordnungen, Ministerien), Fachmedien, Studien
Ignoriere: SEO-Portale, reine Meinungsartikel, Einzelanekdoten ohne Daten"""

_FRAMEWORK_PROGNOSTIC = """ANALYSE-TYP: Prognostische Transformationsanalyse

INTERNES 2-PHASEN-VORGEHEN (beide Phasen in einer Antwort):

═══ PHASE 1 — DATENEXTRAKTION (intern, nicht ausgeben) ═══
Bevor du schreibst: scanne ALLE Quellen und extrahiere mental:
- Konkrete Zahlen (Marktvolumen, %, Wachstumsraten, CAGR)
- Zeitangaben und Prognose-Horizonte
- Anwendungsbeispiele mit Ergebnissen
- Regulatorische Änderungen mit Datum
- Quellenzuordnung (wer sagt was)
Erst wenn du diese Datenbasis hast → Phase 2.

═══ PHASE 2 — STRUKTURIERTE ANALYSE (das ist deine Antwort) ═══

1. LEITTHESE: Beginne mit EINER klaren Transformationshypothese in 1-2 Sätzen.
   Beispiel: "KI entwickelt sich bis 2030 in der Medizin vom Diagnose-Assistenten
   zum autonomen Behandlungsoptimierer — mit 15-25% Kostenreduktion bei
   gleichzeitiger Qualitätssteigerung."

2. TREIBER & KONSENS: Extrahiere den Prognose-Konsens und die Bandbreite der Schätzungen

3. SZENARIEN: Differenziere nach Zeithorizont:
   - Kurzfristig (1-2J): Was passiert bereits?
   - Mittelfristig (3-5J): Was ist wahrscheinlich?
   - Langfristig (5-10J): Was ist möglich unter welchen Bedingungen?

4. WIRTSCHAFTLICHE HEBEL (PFLICHT):
   - Wo entstehen Margenverschiebungen?
   - Wo sinken Kosten und um wieviel?
   - Welche neuen Geschäftsmodelle entstehen?
   - Wer gewinnt / wer verliert strukturell?

5. UNSICHERHEITEN: Benenne explizit was unklar ist und welche Annahmen die Prognosen stützen

6. QUANTIFIZIERUNG: Mindestens 3-5 konkrete Zahlen/Spannbreiten in der Antwort.
   Keine Aussage ohne Zahl wenn eine verfügbar ist.

QUELLEN-HIERARCHIE:
Bevorzuge: Beratungsstudien (McKinsey, BCG, Deloitte), Forschungsinstitute (Fraunhofer, MIT),
  internationale Organisationen (OECD, WHO, WEF), Marktanalysen (Gartner, IDC, Statista)
Ignoriere: SEO-Artikel, PR-Meldungen, reine Meinungsbeiträge, Einzelanekdoten"""

_FRAMEWORK_STRATEGIC = """ANALYSE-TYP: Strategische Transformationsanalyse

INTERNES 2-PHASEN-VORGEHEN (beide Phasen in einer Antwort):

═══ PHASE 1 — DATENEXTRAKTION (intern, nicht ausgeben) ═══
Bevor du schreibst: scanne ALLE Quellen und extrahiere mental:
- Marktvolumina, CAGR, Investitionsvolumen
- Adoptionsraten und Penetrationskurven
- Konkrete Anwendungsfälle mit messbaren Ergebnissen
- Regulatorische Meilensteine mit Datum
- Branchenspezifische Unterschiede
Erst wenn du diese Datenbasis hast → Phase 2.

═══ PHASE 2 — STRUKTURIERTE ANALYSE (das ist deine Antwort) ═══

1. LEITTHESE (PFLICHT — erster Absatz):
   Formuliere EINE klare Transformationshypothese in 1-2 Sätzen.
   Muster: "[Technologie] entwickelt sich bis [Jahr] von [Status Quo] zu
   [Zielzustand] — mit [quantifiziertem Effekt]."

2. WERTSCHÖPFUNGSKETTE: Analysiere entlang der Wertschöpfungskette.
   - Welche Stufen werden transformiert?
   - Wo verschieben sich Margen?
   - Wer gewinnt, wer verliert strukturell?

3. MARKTDIMENSIONIERUNG: Nenne Marktgrößen, Investitionsvolumina, Adoptionsraten

4. STRUKTURWANDEL: Differenziere nach Branchen/Segmenten — keine Pauschalaussagen

5. WIRTSCHAFTLICHE HEBEL (PFLICHT — eigener Abschnitt):
   Beantworte explizit:
   □ Wo entstehen Margenverschiebungen?
   □ Wo sinken Kosten und um wieviel (%)?
   □ Welche neuen Geschäftsmodelle / Revenue Streams entstehen?
   □ Wer sind die strukturellen Gewinner und Verlierer?

6. ZEITACHSE: Staffelung nach Phase:
   - Early Adoption (jetzt – 2026)
   - Mainstreaming (2027 – 2029)
   - Mature / Strukturwandel (2030+)

7. REALITÄTSCHECK (PFLICHT — eigener Abschnitt am Ende):
   □ Implementierungshürden (technisch, organisatorisch)
   □ Regulatorische Verzögerungen
   □ Daten- / Infrastrukturprobleme
   □ Kapitalbedarf
   □ Akzeptanz- / Change-Management-Fragen
   Trenne Hype von strukturellem Wandel mit Evidenz.

QUANTIFIZIERUNG: Mindestens 5 konkrete Zahlen/Spannbreiten in der Antwort.

QUELLEN-HIERARCHIE:
Bevorzuge: Beratungsstudien (McKinsey, BCG, Deloitte, Roland Berger), Think-Tanks (WEF, OECD),
  Marktanalysen (Gartner, IDC, Grand View Research), Forschungsinstitute (Fraunhofer, MIT, IAB),
  Ministerien/Regulierer (EU-Kommission, BMAS, BMWi)
Ignoriere: SEO-Portale, Pressemeldungen ohne Daten, reine Meinungsartikel, Einzelanekdoten"""

# ── Self-Review Module ──

_SELF_REVIEW = """

QUALITÄTS-SELBSTPRÜFUNG (führe am Ende intern durch, korrigiere VOR der Ausgabe):

Pflicht-Check (alle müssen erfüllt sein):
□ Klare Leitthese im ersten Absatz vorhanden?
□ Mindestens 3-5 konkrete Zahlen/Prozente integriert?
□ Zeitachse enthalten (kurzfristig/mittelfristig/langfristig)?
□ Wirtschaftliche Hebel benannt (Kosten, Margen, Geschäftsmodelle)?
□ Regulatorische Dimension berücksichtigt?
□ Realitätscheck / Implementierungshürden formuliert?

Qualitäts-Check:
□ Ist die Struktur thematisch (nicht quellenweise)?
□ Gibt es Quellenangaben bei konkreten Aussagen?
□ Sind Aussagen zu vage wo Quantifizierung möglich wäre? ("könnte", "potenziell", "erheblich")
□ Gibt es redundante Wiederholungen?
□ Fehlt eine Dimension die in den Quellen abgedeckt ist?

Falls Mängel bei Pflicht-Checks → überarbeite die Stellen BEVOR du antwortest.
Falls Mängel bei Qualitäts-Checks → verbessere gezielt."""

# ── The dynamic builder ──

def build_synthesis_prompt(
    depth: str = "snippets",
    analysis_mode: str = "factual",
    blueprint: str = "",
    output_architecture: dict = None,
) -> str:
    """
    Build a dynamic synthesis prompt from Round 1 parameters.

    Args:
        depth: snippets/deep/thorough
        analysis_mode: factual/prognostic/strategic
        blueprint: Legacy ANALYSE-BLUEPRINT text (may be empty)
        output_architecture: Structured JSON dict from Round 1 (preferred over blueprint)
    """
    # Snippets → simple prompt, no framework needed
    if depth == "snippets" and not blueprint and not output_architecture:
        return SYNTHESIS_PROMPT_SIMPLE

    # If we have structured architecture, use its framework setting
    if output_architecture and output_architecture.get("framework"):
        analysis_mode = output_architecture["framework"]

    # Select analysis framework
    frameworks = {
        "factual": _FRAMEWORK_FACTUAL,
        "prognostic": _FRAMEWORK_PROGNOSTIC,
        "strategic": _FRAMEWORK_STRATEGIC,
    }
    framework = frameworks.get(analysis_mode, _FRAMEWORK_FACTUAL)

    # Build the prompt
    parts = []

    # Role
    if depth == "thorough" or analysis_mode == "strategic":
        parts.append("Du bist ein Senior-Analyst auf Beratungsniveau. Dir wurden "
                      "umfangreiche Webinhalte zu einer komplexen Frage bereitgestellt.")
    elif depth == "deep":
        parts.append("Du bist ein analytischer Recherche-Assistent. Dir wurden "
                      "detaillierte Webinhalte zu einer Frage bereitgestellt.")
    else:
        parts.append("Du bist ein hilfreicher Assistent. Dir wurden Suchergebnisse bereitgestellt.")

    # Analysis framework
    parts.append(framework)

    # ── Structured output architecture (preferred) ──
    if output_architecture:
        arch_parts = []
        arch_parts.append("\nOUTPUT-ARCHITEKTUR (ZWINGEND — befolge diese Struktur exakt):")

        if output_architecture.get("reasoning"):
            arch_parts.append(f"Begründung: {output_architecture['reasoning']}")

        if output_architecture.get("required_data"):
            data_items = output_architecture["required_data"]
            arch_parts.append(f"\nPFLICHT-DATENPUNKTE (diese Daten MÜSSEN in der Antwort vorkommen):")
            for i, d in enumerate(data_items, 1):
                arch_parts.append(f"  {i}. {d}")

        if output_architecture.get("outline"):
            outline = output_architecture["outline"]
            arch_parts.append(f"\nGLIEDERUNG (folge dieser Struktur EXAKT — verwende **Fettschrift** für Überschriften):")
            for entry in outline:
                # Normalize any format to **bold**
                e = entry.strip()
                # Strip legacy H1:/H2:/H3: prefixes
                if e.startswith("H1:") or e.startswith("H1 "):
                    e = e.split(":", 1)[-1].strip()
                elif e.startswith("H2:") or e.startswith("H2 "):
                    e = e.split(":", 1)[-1].strip()
                elif e.startswith("H3:") or e.startswith("H3 "):
                    e = e.split(":", 1)[-1].strip()
                # Strip markdown heading markers
                e = e.lstrip("#").strip()
                # Ensure bold formatting
                if not e.startswith("**"):
                    e = f"**{e}**"
                arch_parts.append(f"  {e}")

        if output_architecture.get("source_priority"):
            prio = output_architecture["source_priority"]
            arch_parts.append(f"\nQUELLEN-PRIORITÄT: {' > '.join(prio)}")

        # Detect language from architecture
        lang = output_architecture.get("language", "de")
        if lang == "en":
            arch_parts.append("\nRESPOND IN ENGLISH.")

        parts.append("\n".join(arch_parts))

    # ── Legacy blueprint fallback ──
    elif blueprint:
        parts.append(f"""
ANALYSE-BLUEPRINT (aus der Recherche-Planung — befolge diese Vorgaben ZWINGEND):
{blueprint}

Die oben genannten Pflicht-Dimensionen, Quantitativen Pflichtangaben und die Struktur 
sind VERBINDLICH. Jeder genannte Punkt muss in der Antwort abgedeckt werden.""")

    # Universal quality rules
    parts.append("""
SYNTHESE-REGELN:
- SYNTHETISIERE die Quellen zu einer kohärenten Analyse — NICHT Quelle für Quelle zusammenfassen
- Strukturiere nach THEMATISCHEN ASPEKTEN, nicht nach Quellen
- Quantitativ > qualitativ ("25-30%" statt "erheblich")
- Studien/Reports > Zeitungsartikel > Meinungsbeiträge
- Antworte in der Sprache des Nutzers
- Keine Code-Blöcke, keine API-Aufrufe

INHALT — ABSOLUT ZWINGEND:
- Du MUSST konkrete Informationen aus den Quellen extrahieren und aufführen
- NIEMALS schreiben "besuchen Sie die Webseite für mehr Details" oder "müsste ich detaillierter durchsuchen"
- NIEMALS schreiben "die Webseite listet X Veranstaltungen" — stattdessen die Veranstaltungen SELBST auflisten
- NIEMALS nur Webseiten empfehlen statt Inhalte zu liefern
- Wenn eine Quelle "Konzert X am 15.02. um 20:00 Uhr" enthält, dann SCHREIBE das in deine Antwort
- Wenn du aus den Quellen keine konkreten Details extrahieren kannst, nutze dein Trainingswissen
- Jede Überschrift MUSS mindestens einen konkreten Inhaltspunkt darunter haben — leere Sektionen sind VERBOTEN
- Lieber weniger Überschriften mit echtem Inhalt als viele leere Sektionen

⚠️ QUELLENREFERENZEN — EXTREM WICHTIG:
- [1], [2] etc. dürfen NUR auf tatsächlich vorhandene Quellen oben verweisen
- ERFINDE NIEMALS [N]-Referenzen für Informationen die nicht in den Quellen stehen
- Informationen aus deinem Trainingswissen OHNE [N]-Referenz schreiben
- Am Ende KEIN eigenes "Quellen:" Verzeichnis erstellen — das wird automatisch angehängt
- Wenn du nur 3 Quellen hast, darfst du maximal [1], [2], [3] verwenden — NIEMALS [4], [5], [6]

⚠️ ANTI-HALLUZINATION:
- ERFINDE KEINE Events, URLs, Termine oder Fakten die nicht in den Quellen stehen
- Wenn die Quellen wenig hergeben: sage ehrlich was du weißt und was du nicht bestätigen kannst
- Lieber weniger aber korrekte Informationen als viele erfundene Details

ANTI-FAULHEITS-CHECK: Wenn deine Antwort die Wörter "empfiehlt es sich", "ratsam", "detaillierter durchsuchen",
"besuchen Sie", "für mehr Informationen", "genannten Webseiten" oder "Nicht gefunden" enthält, ist sie FALSCH.
Der Nutzer will ANTWORTEN, keine Links zum Selbstsuchen.

FORMATIERUNG — ZWINGEND:
- Überschriften als **Fettschrift** formatieren: "**Titel**" auf einer eigenen Zeile
- Unterüberschriften ebenfalls fett: "**Untertitel**"
- Nach jeder Überschrift eine Leerzeile
- Bullet Points mit "•" für Listen
- KEINE Markdown-Heading-Syntax (kein #, ##, ###)
- Beispiel einer GUTEN Antwort:

  **Veranstaltungen in Stuttgart nächste Woche**

  **Konzerte**

  • Tarja — Living the Dream Tour am 13.02. um 20:00 Uhr, Porsche Arena [3]
  • Mother Black Cat am 15.02. um 20:30 Uhr, Nürtingen [3]

  **Ausstellungen**

  • MARVEL: Universe of Super Heroes, täglich ab 10:00 Uhr in Ludwigsburg. Tickets ab 22€ [2]

- Beispiel einer SCHLECHTEN Antwort (VERMEIDE):

  Auf eventfinder.de finden sich zahlreiche Veranstaltungen für nächste Woche.
  Für detaillierte Informationen besuchen Sie visitberlin.de.

QUELLENREFERENZEN — ZWINGEND:
- Referenziere Quellen als Nummern in eckigen Klammern: [1], [2], [3]
- Setze die Nummer NACH dem Satz oder Fakt: "Konzert am 13.02. um 20:00 Uhr [1]"
- NIEMALS Quellennamen oder URLs inline schreiben
- NIEMALS "(Quelle: eventfinder.de)" oder "(Quelle: https://...)" — NUR [1], [2] etc.
- Die Nummern entsprechen der Reihenfolge der Quellen die du erhältst
- Referenziere NUR Quellen die du tatsächlich inhaltlich verwendest

VERMEIDE:
- Quellenweise Zusammenfassung ("Quelle 1 sagt X, Quelle 2 sagt Y")
- Vage Qualifier wenn Zahlen in den Quellen verfügbar sind
- Redundante Wiederholungen
- Plain-text Überschriften ohne Fettschrift
- Markdown-Heading-Syntax (#, ##, ###)
- Quellen als Text/URLs statt Nummern
- Leere Abschnitte ohne konkreten Inhalt
- Verweise auf Webseiten statt eigener Inhaltsextraktion
- EIGENES Quellenverzeichnis am Ende — KEIN "Quellen:" Abschnitt mit URLs generieren!
  Die Quellenliste wird AUTOMATISCH angehängt. Du schreibst NUR [1], [2] etc. inline.
- ERFUNDENE Informationen: Wenn du keine konkreten Daten aus den Quellen hast,
  nutze Trainingswissen, aber erfinde KEINE URLs, Termine oder Fakten.
  Schreibe lieber "basierend auf allgemeinen Informationen" statt eine fake URL zu zitieren.

WENN DIE SUCHERGEBNISSE IRRELEVANT ODER UNZUREICHEND SIND:
- Nutze dein Trainingswissen, um die Frage trotzdem fundiert zu beantworten
- Sage am Anfang kurz: "Die Webrecherche lieferte keine aktuellen Studien. Basierend auf meinem Wissensstand:" und beantworte dann die Frage vollständig
- Gib NIEMALS auf und sage nur "keine Ergebnisse gefunden" — beantworte die Frage IMMER
- Kombiniere Suchergebnisse (soweit relevant) mit Trainingswissen für die beste Antwort
- ERFINDE KEINE Quellen-Nummern für Trainingswissen — nur echte Suchergebnisse referenzieren""")

    # Self-review for deep/thorough
    if depth in ("deep", "thorough"):
        parts.append(_SELF_REVIEW)

    return "\n".join(parts)


def _extract_block(content: str, markers: list[str], end_markers: list[str]) -> str:
    """Generic block extractor: find start marker, trim at end markers."""
    if not content:
        return ""
    for marker in markers:
        idx = content.find(marker)
        if idx >= 0:
            text = content[idx + len(marker):]
            for em in end_markers:
                end = text.find(em)
                if end > 0:
                    text = text[:end]
                    break
            return text.strip()
    return ""


def extract_blueprint(content: str) -> str:
    """Extract ANALYSE-BLUEPRINT block from Round 1 assistant content.
    Legacy fallback — tries old free-text format first, then JSON."""
    # Try old format first (backward compat)
    old_bp = _extract_block(
        content,
        markers=["ANALYSE-BLUEPRINT:", "ANALYSIS-BLUEPRINT:", "BLUEPRINT:"],
        end_markers=["\n\n\n", "\nKONTEXT-MODUS:", "\nKONTEXT-EXTRAKT:",
                     "\nSEARCH", "\nANALYSIS"],
    )
    if old_bp:
        return old_bp

    # Try extracting from JSON output_architecture
    arch = extract_output_architecture(content)
    if arch:
        # Convert structured JSON to blueprint text for backward compat
        parts = []
        if arch.get("framework"):
            parts.append(f"Typ: {arch['framework']}")
        if arch.get("required_data"):
            parts.append(f"Quantitative Pflicht: {', '.join(arch['required_data'])}")
        if arch.get("outline"):
            # Convert H1/H2/H3 or markdown headings to plain section names
            sections = []
            for o in arch["outline"]:
                o = o.strip()
                # Strip markdown heading markers
                if o.startswith("### "):
                    o = o[4:]
                elif o.startswith("## "):
                    o = o[3:]
                elif o.startswith("# "):
                    o = o[2:]
                # Strip legacy H1:/H2:/H3: markers
                elif ": " in o and o[:3] in ("H1:", "H2:", "H3:"):
                    o = o.split(": ", 1)[-1]
                sections.append(o.strip())
            parts.append(f"Struktur: {', '.join(sections)}")
        if arch.get("source_priority"):
            parts.append(f"Quellen-Priorität: {' > '.join(arch['source_priority'])}")
        return "\n".join(parts)

    return ""


def extract_output_architecture(content: str) -> dict:
    """
    Extract structured output_architecture JSON from Round 1 assistant content.

    The model outputs a JSON block in ```json ... ``` markers containing:
    {
        "output_architecture": {
            "framework": "factual|prognostic|strategic",
            "language": "de|en",
            "reasoning": "...",
            "required_data": ["..."],
            "outline": ["H1: ...", "H2: ...", ...],
            "source_priority": ["..."]
        },
        "context_mode": "full|recent:N|distill",
        "context_extract": "..." or null
    }

    Returns the output_architecture dict, or {} if not found.
    """
    if not content:
        return {}

    import json as _json
    import re

    # Try to find JSON block in ```json ... ``` markers
    json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Fallback: try to find raw JSON with output_architecture key
        json_match = re.search(r'\{[^{}]*"output_architecture"[^{}]*\{.*?\}[^{}]*\}',
                               content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {}

    try:
        data = _json.loads(json_str)
        arch = data.get("output_architecture", {})
        if isinstance(arch, dict) and arch:
            return arch
    except (_json.JSONDecodeError, ValueError, TypeError) as e:
        log.debug(f"JSON parse failed for output_architecture: {e}")

    return {}


def extract_full_r1_json(content: str) -> dict:
    """
    Extract the full Round 1 JSON block (output_architecture + context_mode + context_extract).
    Returns the complete parsed dict, or {} if not found.
    """
    if not content:
        return {}

    import json as _json
    import re

    json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            return _json.loads(json_match.group(1).strip())
        except (_json.JSONDecodeError, ValueError):
            pass

    return {}


def extract_context_strategy(content: str) -> dict:
    """
    Extract context strategy from Round 1 content.
    Supports both new JSON format and legacy KONTEXT-MODUS text format.

    Returns dict with:
        mode: "full" | "recent" | "distill"
        recent_n: int (only if mode == "recent")
        distill_text: str (only if mode == "distill")
    """
    import re

    result = {"mode": "full", "recent_n": 0, "distill_text": ""}

    if not content:
        return result

    # ── Try new JSON format first ──
    r1_json = extract_full_r1_json(content)
    if r1_json:
        ctx_mode = r1_json.get("context_mode", "full")
        ctx_extract = r1_json.get("context_extract") or ""

        if ctx_mode == "full":
            result["mode"] = "full"
        elif ctx_mode.startswith("recent"):
            result["mode"] = "recent"
            # Parse "recent:3" → 3
            match = re.match(r'recent:?(\d+)?', ctx_mode)
            n = int(match.group(1)) if match and match.group(1) else 3
            result["recent_n"] = max(1, min(n, 50))
        elif ctx_mode == "distill":
            result["mode"] = "distill"
            skip_phrases = ["kein relevanter kontext", "no relevant context",
                            "nicht relevant", "standalone", "null"]
            if ctx_extract and not any(p in ctx_extract.lower() for p in skip_phrases):
                result["distill_text"] = ctx_extract

        return result

    # ── Fallback: legacy KONTEXT-MODUS text format ──
    match = re.search(
        r'KONTEXT-MODUS:\s*(full|recent(?::(\d+))?|distill)',
        content, re.IGNORECASE,
    )
    if match:
        mode_str = match.group(1).lower()
        if mode_str == "full":
            result["mode"] = "full"
        elif mode_str.startswith("recent"):
            result["mode"] = "recent"
            n = int(match.group(2)) if match.group(2) else 3
            result["recent_n"] = max(1, min(n, 50))
        elif mode_str == "distill":
            result["mode"] = "distill"
            distill_text = _extract_block(
                content,
                markers=["KONTEXT-EXTRAKT:", "CONTEXT-EXTRACT:"],
                end_markers=["\n\n\n", "\nANALYSE-BLUEPRINT:", "\nBLUEPRINT:",
                             "\nKONTEXT-MODUS:", "\nSEARCH", "\nANALYSIS",
                             "\n```"],
            )
            skip_phrases = ["kein relevanter kontext", "no relevant context",
                            "nicht relevant", "standalone", "nicht benötigt"]
            if any(p in distill_text.lower() for p in skip_phrases):
                distill_text = ""
            result["distill_text"] = distill_text

    return result


# Keep old name as alias for backward compatibility
def extract_context_distillation(content: str) -> str:
    """Legacy wrapper — returns distill_text or empty string."""
    strategy = extract_context_strategy(content)
    if strategy["mode"] == "distill":
        return strategy["distill_text"]
    return ""


def generate_auto_blueprint(query: str, analysis_mode: str = "factual") -> str:
    """
    Generate a reasonable default blueprint when Round 1 didn't produce one.
    This is a fallback for when Gemini generates content_len=0 with tool calls.
    Returns a text blueprint for backward compatibility.
    """
    arch = generate_auto_architecture(query, analysis_mode)
    # Convert to text for backward compat with build_synthesis_prompt blueprint param
    parts = []
    if arch.get("framework"):
        parts.append(f"Typ: {arch['framework']}")
    if arch.get("required_data"):
        parts.append(f"Quantitative Pflicht: {', '.join(arch['required_data'])}")
    if arch.get("outline"):
        sections = [o.split(": ", 1)[-1] if ": " in o else o for o in arch["outline"]]
        parts.append(f"Struktur: {', '.join(sections)}")
    if arch.get("source_priority"):
        parts.append(f"Quellen-Priorität: {' > '.join(arch['source_priority'])}")
    return "\n".join(parts)


def generate_auto_architecture(query: str, analysis_mode: str = "factual") -> dict:
    """
    Generate a reasonable default output_architecture when Round 1 didn't produce one.
    Returns a structured dict matching the output_architecture format.
    """
    if analysis_mode == "strategic":
        return {
            "framework": "strategic",
            "language": "de",
            "reasoning": "Auto-generated: complex analytical query requires strategic framework",
            "required_data": [
                "Marktvolumen/CAGR",
                "Produktivitäts-/Kosteneffekte (%)",
                "Investitionsvolumen",
                "Adoptionsraten",
                "Zeithorizonte",
            ],
            "outline": [
                "H1: Analyse",
                "H2: Leitthese",
                "H2: Wertschöpfungskette & Markt",
                "H2: Branchen-/Segmentanalyse",
                "H2: Wirtschaftliche Hebel",
                "H2: Zeitachse (Early/Main/Mature)",
                "H2: Regulierung & Implementierung",
                "H2: Kritische Einordnung",
            ],
            "source_priority": [
                "Beratungsstudien (McKinsey, BCG)",
                "Think-Tanks (WEF, OECD)",
                "Marktanalysen (Gartner, IDC)",
                "Forschungsinstitute",
                "Fachpresse",
            ],
        }
    elif analysis_mode == "prognostic":
        return {
            "framework": "prognostic",
            "language": "de",
            "reasoning": "Auto-generated: forecasting query requires prognostic framework",
            "required_data": [
                "Wachstumsraten (%)",
                "Marktprognosen",
                "Kosteneffekte",
                "Zeithorizonte",
            ],
            "outline": [
                "H1: Prognose",
                "H2: Leitthese",
                "H2: Status Quo & Treiber",
                "H2: Kurzfrist (1-2 Jahre)",
                "H2: Mittelfrist (3-5 Jahre)",
                "H2: Langfrist (5-10 Jahre)",
                "H2: Wirtschaftliche Hebel",
                "H2: Szenarien & Unsicherheiten",
            ],
            "source_priority": [
                "Beratungsstudien",
                "Think-Tanks",
                "Marktanalysen",
                "Forschungsinstitute",
                "Fachpresse",
            ],
        }
    else:  # factual
        return {
            "framework": "factual",
            "language": "de",
            "reasoning": "Auto-generated: factual query requires direct answer structure",
            "required_data": [
                "konkrete Zahlen/Daten",
                "Fristen/Termine",
                "Regelungen/Ausnahmen",
            ],
            "outline": [
                "H1: Antwort",
                "H2: Kernaussage",
                "H2: Details & Regelungen",
                "H2: Ausnahmen/Einschränkungen",
                "H2: Quellen",
            ],
            "source_priority": [
                "Offizielle Quellen (Gesetze, Verordnungen)",
                "Fachmedien",
                "Allgemeinpresse",
            ],
        }


# Legacy compatibility: static map (used if blueprint extraction fails)
SYNTHESIS_PROMPTS = {
    "snippets": SYNTHESIS_PROMPT_SIMPLE,
    "deep": build_synthesis_prompt("deep", "factual"),
    "thorough": build_synthesis_prompt("thorough", "strategic"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Execution Engine
# ═══════════════════════════════════════════════════════════════════════════════

# Geocoding cache
_geocode_cache = {}

# Known cities for instant lookup
_KNOWN_CITIES = {
    "berlin": (52.52, 13.405), "hamburg": (53.551, 9.994),
    "frankfurt": (50.110, 8.682), "stuttgart": (48.776, 9.183),
    "konstanz": (47.660, 9.175), "london": (51.507, -0.128),
    "paris": (48.857, 2.352), "new york": (40.713, -74.006),
    "tokyo": (35.682, 139.759), "sydney": (-33.869, 151.209),
    "amsterdam": (52.370, 4.895), "madrid": (40.417, -3.704),
    "barcelona": (41.389, 2.159), "istanbul": (41.009, 28.978),
    "dubai": (25.205, 55.271), "seoul": (37.567, 126.978),
    "mumbai": (19.076, 72.878), "bangkok": (13.756, 100.502),
    "bonn": (50.737, 7.099), "bremen": (53.079, 8.802),
    "dresden": (51.051, 13.738), "leipzig": (51.340, 12.375),
    "hannover": (52.376, 9.739), "freiburg": (47.999, 7.842),
    "mannheim": (49.489, 8.467), "heidelberg": (49.399, 8.673),
    "dortmund": (51.514, 7.468), "essen": (51.456, 7.012),
    "bern": (46.948, 7.448), "basel": (47.559, 7.589),
    "graz": (47.070, 15.439), "salzburg": (47.811, 13.055),
    "karlsruhe": (49.007, 8.404), "augsburg": (48.371, 10.898),
    "münchen": (48.137, 11.576), "munich": (48.137, 11.576),
    "köln": (50.938, 6.960), "cologne": (50.938, 6.960),
    "düsseldorf": (51.228, 6.774), "zürich": (47.377, 8.540),
    "zurich": (47.377, 8.540), "wien": (48.208, 16.374),
    "vienna": (48.208, 16.374), "nürnberg": (49.454, 11.078),
    "genf": (46.205, 6.144), "geneva": (46.205, 6.144),
    "rom": (41.903, 12.496), "rome": (41.903, 12.496),
    "athen": (37.984, 23.728), "athens": (37.984, 23.728),
    "mailand": (45.464, 9.190), "milan": (45.464, 9.190),
    "prag": (50.076, 14.438), "prague": (50.076, 14.438),
    "peking": (39.904, 116.407), "beijing": (39.904, 116.407),
    "moskau": (55.756, 37.617), "moscow": (55.756, 37.617),
    "singapur": (1.352, 103.820), "singapore": (1.352, 103.820),
    "shanghai": (31.231, 121.474), "chicago": (41.878, -87.630),
    "los angeles": (34.052, -118.244), "san francisco": (37.775, -122.419),
    "toronto": (43.653, -79.383), "vancouver": (49.283, -123.121),
    "kairo": (30.044, 31.236), "cairo": (30.044, 31.236),
    "kapstadt": (-33.925, 18.424), "cape town": (-33.925, 18.424),
    "lissabon": (38.722, -9.139), "lisbon": (38.722, -9.139),
    "warschau": (52.230, 21.012), "warsaw": (52.230, 21.012),
    "budapest": (47.498, 19.040), "kopenhagen": (55.676, 12.569),
    "stockholm": (59.329, 18.069), "oslo": (59.914, 10.752),
    "helsinki": (60.169, 24.938), "innsbruck": (47.260, 11.394),
}


async def _geocode(city: str) -> Optional[tuple[str, float, float]]:
    """Geocode a city name to (display_name, lat, lon). Free via Open-Meteo."""
    low = city.lower().strip()
    if low in _KNOWN_CITIES:
        return (city, _KNOWN_CITIES[low][0], _KNOWN_CITIES[low][1])
    if low in _geocode_cache:
        return _geocode_cache[low]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "de"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        if results:
            r = results[0]
            name = r.get("name", city)
            country = r.get("country", "")
            display = f"{name}, {country}" if country else name
            result = (display, r["latitude"], r["longitude"])
            _geocode_cache[low] = result
            log.debug(f"Geocoded '{city}' -> {display}")
            return result
    except Exception as e:
        log.warning(f"Geocoding failed for '{city}': {e}")
    return None


WEATHER_CODES = {
    0: "Klar", 1: "Meist klar", 2: "Teilw. bewölkt", 3: "Bewölkt",
    45: "Nebel", 48: "Raureif", 51: "Niesel", 53: "Nieselregen",
    56: "Gefr. Niesel", 61: "Leichter Regen", 63: "Regen", 65: "Starkregen",
    66: "Gefr. Regen", 71: "Leichter Schnee", 73: "Schnee", 75: "Starker Schnee",
    80: "Regenschauer", 81: "Starke Schauer", 85: "Schneeschauer",
    95: "Gewitter", 96: "Gewitter+Hagel",
}


async def tool_get_weather(city: str) -> str:
    """Execute get_weather tool: geocode + Open-Meteo forecast."""
    loc = await _geocode(city)
    if not loc:
        return f"Fehler: Stadt '{city}' konnte nicht gefunden werden."

    display, lat, lon = loc
    try:
        params = {
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,weather_code",
        }
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.open-meteo.com/v1/forecast",
                                 params=params, timeout=10.0)
            r.raise_for_status()
            data = r.json()

        c = data.get("current", {})
        d = data.get("daily", {})
        desc = WEATHER_CODES.get(c.get("weather_code", -1), "?")
        tz = data.get("timezone", "")

        lines = [
            f"WETTER {display.upper()} (Open-Meteo, {datetime.now():%d.%m.%Y %H:%M})",
            f"Koordinaten: {lat:.2f}, {lon:.2f} | Zeitzone: {tz}",
            f"Aktuell: {desc}",
            f"Temperatur: {c.get('temperature_2m','?')}°C (gefühlt {c.get('apparent_temperature','?')}°C)",
            f"Luftfeuchtigkeit: {c.get('relative_humidity_2m','?')}%",
            f"Wind: {c.get('wind_speed_10m','?')} km/h (Böen {c.get('wind_gusts_10m','?')})",
            f"Niederschlag: {c.get('precipitation',0)} mm",
        ]
        t = c.get("temperature_2m", 99)
        p = c.get("precipitation", 0)
        if isinstance(t,(int,float)) and t<=2 and isinstance(p,(int,float)) and p>0:
            lines.append("⚠️ GLATTEISRISIKO: Temperatur nahe 0°C bei Niederschlag!")
        elif isinstance(t,(int,float)) and t<=0:
            lines.append("⚠️ FROST: Temperaturen unter 0°C!")
        elif isinstance(t,(int,float)) and t>=35:
            lines.append("⚠️ HITZE: Über 35°C!")
        if d.get("time"):
            lines.append("\n3-Tage-Prognose:")
            for i, ds in enumerate(d["time"][:3]):
                dc = d.get("weather_code",[0])[i] if i<len(d.get("weather_code",[])) else 0
                tmax = d.get("temperature_2m_max",["?"])[i] if i<len(d.get("temperature_2m_max",[])) else "?"
                tmin = d.get("temperature_2m_min",["?"])[i] if i<len(d.get("temperature_2m_min",[])) else "?"
                rp = d.get("precipitation_probability_max",[0])[i] if i<len(d.get("precipitation_probability_max",[])) else 0
                lines.append(f"  {ds}: {WEATHER_CODES.get(dc,'?')} | {tmin}–{tmax}°C | Regen: {rp}%")
        return "\n".join(lines)
    except Exception as e:
        return f"Fehler beim Abruf der Wetterdaten für {display}: {e}"


async def tool_get_stock_price(symbols: str) -> str:
    """Execute get_stock_price tool: yfinance lookup."""
    tickers = [s.strip() for s in symbols.split(",") if s.strip()][:5]
    if not tickers:
        return "Fehler: Keine Ticker-Symbole angegeben."

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_stocks_sync, tickers)


def _fetch_stocks_sync(tickers):
    try:
        import yfinance as yf
    except ImportError:
        return "Fehler: yfinance nicht installiert."

    lines = [f"BÖRSENDATEN ({datetime.now():%d.%m.%Y %H:%M}) | Quelle: Yahoo Finance"]
    for ts in tickers:
        try:
            t = yf.Ticker(ts)
            fi = t.fast_info if hasattr(t, 'fast_info') else {}
            price = getattr(fi, 'last_price', None)
            prev = getattr(fi, 'previous_close', None)
            cur = getattr(fi, 'currency', 'USD')
            if price is None:
                h = t.history(period="2d")
                if not h.empty:
                    price = h['Close'].iloc[-1]
                    if len(h) > 1: prev = h['Close'].iloc[-2]
            if price is not None:
                l = f"{ts}: {price:.2f} {cur}"
                if prev and prev > 0:
                    ch = ((price - prev) / prev) * 100
                    arrow = "📈" if ch >= 0 else "📉"
                    l += f" | {arrow} {ch:+.2f}%"
                mcap = getattr(fi, 'market_cap', None)
                if mcap:
                    if mcap >= 1e12: l += f" | MktCap: {mcap/1e12:.1f}T"
                    elif mcap >= 1e9: l += f" | MktCap: {mcap/1e9:.1f}B"
                lines.append(l)
            else:
                lines.append(f"{ts}: Keine Daten verfügbar")
        except Exception as e:
            lines.append(f"{ts}: Fehler ({str(e)[:60]})")
    return "\n".join(lines)


async def tool_get_news(query: str, language: str = "de") -> str:
    """Execute get_news tool: NewsAPI (if key) + DuckDuckGo. Both run in parallel."""
    parts = []
    tasks = []

    # NewsAPI (if key configured)
    api_key = os.getenv("NEWSAPI_KEY", "")
    if api_key:
        tasks.append(_newsapi_fetch(query, language, api_key))

    # DuckDuckGo always runs (free, no key, catches broad queries)
    ddg_query = query if len(query.split()) > 2 else f"{query} Nachrichten heute"
    tasks.append(tool_web_search(ddg_query, time_filter="d"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, str) and r and not r.startswith("Fehler") and not r.startswith("Keine"):
            parts.append(r)

    if parts:
        return "\n\n".join(parts)
    return f"Keine Nachrichten gefunden für: {query}"


async def _newsapi_fetch(query: str, language: str, api_key: str) -> Optional[str]:
    """Fetch from NewsAPI. Uses top-headlines for broad queries, everything for specific."""
    try:
        # Broad query detection: short or generic terms
        broad_terms = {"nachrichten", "news", "schlagzeilen", "headlines", "top",
                       "neueste", "aktuell", "heute", "today", "alles", "all"}
        words = set(query.lower().split())
        is_broad = len(words) <= 3 or words.issubset(broad_terms | {"von", "die", "der", "das", "gib", "zeig", "mir"})

        # Map language to country for top-headlines
        LANG_TO_COUNTRY = {
            "de": "de", "en": "us", "fr": "fr", "es": "es",
            "it": "it", "pt": "pt", "nl": "nl", "pl": "pl",
            "ru": "ru", "ja": "jp", "zh": "cn", "ko": "kr",
            "ar": "ae", "sv": "se", "no": "no",
        }
        country = LANG_TO_COUNTRY.get(language, "us")

        async with httpx.AsyncClient() as client:
            if is_broad:
                # Top headlines endpoint (no query needed, just country)
                r = await client.get("https://newsapi.org/v2/top-headlines",
                    params={"country": country,
                            "pageSize": 5, "apiKey": api_key}, timeout=10.0)
            else:
                r = await client.get("https://newsapi.org/v2/everything",
                    params={"q": query, "language": language, "sortBy": "publishedAt",
                            "pageSize": 5, "apiKey": api_key}, timeout=10.0)
            r.raise_for_status()
            data = r.json()

        arts = data.get("articles", [])
        if not arts:
            return None

        endpoint = "Top-Headlines" if is_broad else "Suche"
        lines = [f"NEWS ({datetime.now():%d.%m.%Y %H:%M}) | NewsAPI {endpoint}\n"]
        for i, a in enumerate(arts[:5], 1):
            src = a.get("source", {}).get("name", "?")
            title = a.get("title", "?")
            published = a.get("publishedAt", "")[:10]
            lines.append(f"{i}. [{src}, {published}] {title}")
            d = (a.get("description") or "")[:200]
            if d:
                lines.append(f"   {d}")
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"NewsAPI: {e}")
        return None


async def tool_web_search(
    query: str, time_filter: str = "w",
    depth: str = "snippets", max_pages: Optional[int] = None,
    additional_queries: Optional[list[str]] = None,
    search_depth: Optional[str] = None,
) -> str:
    """Execute web_search tool with multi-engine, multi-query, and source tracking.

    The search_depth parameter (min/medium/max) maps to the old depth system:
      min    → snippets (0 pages)
      medium → deep (config.max_pages_deep pages)
      max    → thorough (config.max_pages_thorough pages, multi-engine)
    """
    # ── Load config ──
    try:
        from config import load_config
        ws_cfg = load_config().web_search
    except Exception:
        from models import WebSearchConfig
        ws_cfg = WebSearchConfig()

    # ── Map search_depth → internal depth ──
    if search_depth:
        depth_map = {"min": "snippets", "medium": "deep", "max": "thorough"}
        depth = depth_map.get(search_depth, depth)

    # ── Auto-upgrade: snippets → deep for complex queries ──
    # The model often picks "snippets" for questions that need real page content.
    # Upgrade if the query looks like it needs research (policy, analysis, comparison).
    if depth == "snippets":
        q_lower = query.lower()
        _complex_indicators = [
            # German research keywords
            "was sieht", "was plant", "wie ändert", "wie verändert",
            "auswirkungen", "konsequenzen", "reform", "gesetz", "verordnung",
            "vergleich", "unterschied", "analyse", "studie", "prognose",
            "vor- und nachteile", "stabilisieren", "strategie", "konzept",
            # German event/local queries (need page content, not just snippets)
            "veranstaltung", "event", "programm", "was gibt es",
            "was ist los", "was kann man", "interessant für",
            # English research keywords
            "what are the", "how does", "compare", "impact", "consequences",
            "analysis", "policy", "reform", "regulation", "strategy",
            "what events", "things to do", "interesting for tourists",
            # Question complexity: >8 words or contains analytical keywords
        ]
        _is_complex = (
            any(ind in q_lower for ind in _complex_indicators)
            or len(query.split()) > 10
        )
        if _is_complex:
            depth = "deep"
            log.info(f"web_search: auto-upgrade snippets→deep for complex query: "
                     f"'{query[:60]}'")

    # ── Resolve max_pages from config + depth ──
    DEPTH_DEFAULTS = {
        "snippets": ws_cfg.min_pages,   # Even snippets get min_pages
        "deep": ws_cfg.max_pages_deep,
        "thorough": ws_cfg.max_pages_thorough,
    }
    if max_pages is None:
        max_pages = DEPTH_DEFAULTS.get(depth, ws_cfg.min_pages)
    max_pages = max(ws_cfg.min_pages, min(ws_cfg.max_pages_thorough, max_pages))

    # Convert "none" string to actual None for search APIs
    effective_tf = None if time_filter == "none" else time_filter
    # For research: don't append current date to search query
    append_date = depth == "snippets"

    # ── Determine which engines to use ──
    # min/medium: primary engine only, max: all configured engines
    if depth == "thorough":
        engines = ws_cfg.engines
    else:
        engines = ws_cfg.engines[:1]  # Just primary

    log.info(f"web_search: depth={depth}, max_pages={max_pages}, "
             f"engines={engines}, tf={effective_tf}, date={append_date}, "
             f"multi_q={len(additional_queries or [])}, query='{query[:60]}'")

    # ── Collect all queries ──
    all_queries = [query]
    if additional_queries and ws_cfg.multi_query:
        all_queries.extend(additional_queries[:ws_cfg.max_queries - 1])

    # ── Step 1: Snippet-only mode (only if min_pages=0 explicitly) ──
    if max_pages == 0:
        loop = asyncio.get_event_loop()
        # If multi-query: run all queries and merge snippets
        if len(all_queries) > 1:
            tasks = [
                loop.run_in_executor(None, _ddg_search_sync, q, effective_tf, append_date)
                for q in all_queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            merged_parts = []
            for r in results:
                if isinstance(r, str) and not r.startswith("Fehler"):
                    merged_parts.append(r)
            snippets = "\n\n".join(merged_parts) if merged_parts else f"Keine Ergebnisse für: {query}"
        else:
            snippets = await loop.run_in_executor(
                None, _ddg_search_sync, query, effective_tf, append_date)

        # Build source footer from snippet URLs
        if ws_cfg.show_sources and ws_cfg.source_format == "footer":
            _snippet_footer = _extract_source_footer_from_snippets(snippets)
            if _snippet_footer:
                _set_source_footer(_snippet_footer)

        return snippets

    # ── Step 2: Deep/thorough — multi-engine, multi-query ──
    try:
        from web_enrichment import (
            multi_engine_search, fetch_pages, build_context, build_source_footer,
        )

        # Run all queries across all engines in parallel
        search_tasks = []
        for q in all_queries:
            search_tasks.append(
                multi_engine_search(
                    query=q,
                    engines=engines,
                    max_results=ws_cfg.max_snippets,
                    time_filter=effective_tf,
                    append_date=append_date,
                )
            )

        all_results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge + deduplicate across all queries
        seen_urls = set()
        merged_results = []
        for qr in all_results_per_query:
            if isinstance(qr, Exception):
                log.warning(f"Search query failed: {qr}")
                continue
            for r in qr:
                url_key = r.url.rstrip("/").lower()
                if url_key not in seen_urls:
                    seen_urls.add(url_key)
                    merged_results.append(r)

        if not merged_results:
            return f"Keine Ergebnisse für: {query}"

        log.info(f"Multi-query search: {len(all_queries)} queries × "
                 f"{len(engines)} engines → {len(merged_results)} unique results")

        # Fetch full-text from top results
        fetched = await fetch_pages(merged_results, max_pages=max_pages)

        if fetched > 0:
            # Detect language from query
            _lang = "de" if any(c in query.lower() for c in
                               ["ä","ö","ü","ß","und","der","die","das"]) else "en"
            # Use build_context for consistent analytical framing
            deep_context = build_context(
                query, merged_results, mode="synthesis",
                language=_lang, depth=depth,
            )

            # Build source footer if enabled
            source_footer = ""
            if ws_cfg.show_sources and ws_cfg.source_format == "footer":
                source_footer = build_source_footer(merged_results, language=_lang, depth=depth)

            log.info(f"Deep web search: {fetched} pages, depth={depth}, "
                     f"~{len(deep_context)//4} tok, "
                     f"sources={'yes' if source_footer else 'no'}")

            # Store source footer for later appending to LLM response
            _set_source_footer(source_footer)

            return deep_context

    except Exception as e:
        log.warning(f"Deep web search failed ({depth}): {e}")

    # Fallback: DDG snippets
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _ddg_search_sync, query, effective_tf, append_date)


# Module-level storage for source footer (set by web_search, read by main.py)
# Using a plain dict instead of contextvars because contextvars don't propagate
# back from asyncio.gather() tasks to the caller.
# This is safe because asyncio runs single-threaded.
_source_footer_store: dict[str, str] = {"footer": ""}

def get_last_source_footer() -> str:
    """Get the source footer from the most recent web search."""
    return _source_footer_store.get("footer", "")

def clear_source_footer():
    """Clear the stored source footer."""
    _source_footer_store["footer"] = ""

def _set_source_footer(footer: str):
    """Store source footer for later retrieval by main.py."""
    _source_footer_store["footer"] = footer


def _extract_source_footer_from_snippets(snippet_text: str) -> str:
    """Extract URLs from DDG snippet text and build a source footer.

    Snippet text contains lines like:
        → https://example.com/article
    Parse these into a numbered source list.
    Filters out obviously irrelevant domains.
    """
    import re
    from urllib.parse import urlparse

    _IRRELEVANT_TLDS = {".ru", ".pro", ".ua"}
    _IRRELEVANT_DOMAINS = {
        "hltv.org", "mail.ru", "tv.mail.ru", "kinodraiv.pro",
        "o-politico.ru", "pndexam.ru",
    }

    urls = re.findall(r'→\s*(https?://\S+)', snippet_text)
    if not urls:
        return ""

    seen = set()
    lines = ["\n---", "Quellen:"]
    idx = 1
    for url in urls:
        url_key = url.rstrip("/").lower()
        if url_key in seen:
            continue

        # Filter irrelevant domains
        try:
            domain = urlparse(url).hostname or ""
            tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
            if tld in _IRRELEVANT_TLDS or domain in _IRRELEVANT_DOMAINS:
                continue
        except Exception:
            pass

        seen.add(url_key)
        lines.append(f"[{idx}] {url}")
        idx += 1

    return "\n".join(lines) if idx > 1 else ""


def _ddg_search_sync(query, time_filter, append_date=True):
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        try:
            from ddgs import DDGS
        except ImportError:
            return "Fehler: ddgs nicht installiert. Run: pip install ddgs"
    try:
        if append_date:
            search_q = f"{query} {datetime.now().strftime('%B %Y')}"
        else:
            search_q = query
        with DDGS() as ddgs:
            results = list(ddgs.text(
                search_q, max_results=5,
                timelimit=time_filter,  # None = no restriction
            ))
        if not results:
            return f"Keine Ergebnisse für: {query}"

        tl_map = {"d": "24h", "w": "Woche", "m": "Monat"}
        tl_str = tl_map.get(time_filter, "unbegrenzt") if time_filter else "unbegrenzt"
        lines = [f"WEBSUCHE ({datetime.now():%d.%m.%Y %H:%M}) | DuckDuckGo | '{query}' (Zeitraum: {tl_str})\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '?')}")
            b = r.get("body", "")[:250]
            if b: lines.append(f"   {b}")
            h = r.get("href", "")
            if h: lines.append(f"   → {h}")
        return "\n".join(lines)
    except Exception as e:
        return f"Fehler bei Web-Suche: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Router: Parse response → Execute → Return result
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_MAP = {
    "get_weather": tool_get_weather,
    "get_stock_price": tool_get_stock_price,
    "get_news": tool_get_news,
    "web_search": tool_web_search,
}


async def execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """
    Execute one or more tool calls in parallel.
    Returns list of {tool_call_id, name, result} dicts.
    """
    start = time.time()
    results = []
    tasks = []
    meta = []

    for tc in tool_calls:
        func_name = tc.get("function", {}).get("name", "")
        args_str = tc.get("function", {}).get("arguments", "{}")
        tc_id = tc.get("id", "")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            results.append({
                "tool_call_id": tc_id,
                "name": func_name,
                "result": f"Fehler: Ungültige Argumente: {args_str}",
            })
            continue

        handler = TOOL_MAP.get(func_name)
        if not handler:
            results.append({
                "tool_call_id": tc_id,
                "name": func_name,
                "result": f"Fehler: Unbekanntes Tool '{func_name}'",
            })
            continue

        # Build async task
        if func_name == "get_weather":
            tasks.append(handler(args.get("city", "")))
        elif func_name == "get_stock_price":
            tasks.append(handler(args.get("symbols", "")))
        elif func_name == "get_news":
            tasks.append(handler(args.get("query", ""), args.get("language", "de")))
        elif func_name == "web_search":
            tasks.append(handler(
                args.get("query", ""),
                args.get("time_filter", "w"),
                args.get("depth", "snippets"),           # Legacy compat
                args.get("max_pages"),                    # None = auto from depth
                args.get("additional_queries"),            # Multi-query
                args.get("search_depth"),                  # min/medium/max
            ))
        meta.append((tc_id, func_name))

    if tasks:
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for (tc_id, func_name), task_result in zip(meta, task_results):
            if isinstance(task_result, Exception):
                results.append({
                    "tool_call_id": tc_id,
                    "name": func_name,
                    "result": f"Fehler: {task_result}",
                })
            else:
                results.append({
                    "tool_call_id": tc_id,
                    "name": func_name,
                    "result": str(task_result),
                })

    elapsed = (time.time() - start) * 1000
    tools_used = [m[1] for m in meta]
    log.info(f"Tool execution: {tools_used} | {elapsed:.0f}ms")
    return results
