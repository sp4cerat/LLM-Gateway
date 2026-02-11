#!/usr/bin/env python3
"""
LLM Gateway Benchmark v4 — Web Search + Synthesis + Code Stitching
═══════════════════════════════════════════════════════════════════
Tests specifically for the Session 20/21 improvements:
  1. Web Search Quality (DDG date fix, time_filter model-decision, garbage filter)
  2. Synthesis Quality (Leitthese, quantification, economic levers, reality check)
  3. Code Stitching (long code continuation instead of premium escalation)
  4. Context Strategy (full/recent:N/distill mode selection)

Uses model=auto (cascade mode) since that's where the tool-calling pipeline runs.

Usage:
    python benchmark_v4.py                          # All tests
    python benchmark_v4.py --category search        # Only web search tests
    python benchmark_v4.py --category synthesis      # Only synthesis quality
    python benchmark_v4.py --category stitch         # Only code stitching
    python benchmark_v4.py --quick                   # 5 key tests
    python benchmark_v4.py --report                  # Report from saved results
"""

import argparse
import asyncio
import httpx
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Auto-Config ────────────────────────────────────────────────────────────

def load_env_file() -> dict:
    env = {}
    for candidate in [Path.cwd() / ".env", Path(__file__).parent / ".env"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
            break
    return env


def auto_detect_config() -> tuple[str, str]:
    env = load_env_file()
    return (
        os.environ.get("GATEWAY_URL") or env.get("GATEWAY_URL") or "http://localhost:8000",
        os.environ.get("GATEWAY_SECRET") or env.get("GATEWAY_SECRET") or "",
    )


TIMEOUT = 300  # seconds — stitch+repair cascade can take 2-3min
MAX_RETRIES = 3
RETRY_BASE_DELAY = 8
KB_FILE = Path(os.path.abspath(__file__)).parent / "benchmark_v4_results.json"

# ─── Test Definitions ────────────────────────────────────────────────────────

TESTS = {
    # ═══ Web Search Quality ═══
    # These test that the tool-calling pipeline (Round 1 + Round 2) produces
    # useful results instead of "keine Ergebnisse"

    "search-research-1": {
        "category": "search",
        "difficulty": "hard",
        "prompt": "Was sind die Auswirkungen von KI auf die Medizinbranche 2025-2030?",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["Diagnos", "Radiolog", "klinisch", "Patien", "Gesundheit",
                                 "Medizin", "Pharma", "Klinik", "Therapie"],
            "must_not_contain": ["keine relevanten Ergebnisse", "keine Ergebnisse gefunden",
                                 "leider keine", "nicht gefunden"],
            "min_length": 500,
            "description": "Research query must produce substantial analysis, not 'no results'",
        },
    },
    "search-research-2": {
        "category": "search",
        "difficulty": "hard",
        "prompt": "EU Batterieverordnung 2023: Was sind die Anforderungen für austauschbare Smartphone-Akkus?",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["2023/1542", "Batterie", "austauschbar", "Akku", "EU",
                                 "Verordnung", "Regulation"],
            "must_not_contain": ["keine relevanten Ergebnisse", "keine Ergebnisse"],
            "min_length": 300,
            "description": "Regulation query with no time limit should find the actual regulation",
        },
    },
    "search-current-1": {
        "category": "search",
        "difficulty": "medium",
        "prompt": "Neueste Nachrichten heute",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["heute", "aktuell", "Nachrichten", "Meldung"],
            "must_not_contain": ["keine Nachrichten"],
            "min_length": 200,
            "description": "Current news should use time_filter=d and produce fresh results",
        },
    },
    "search-no-garbage-1": {
        "category": "search",
        "difficulty": "medium",
        "prompt": "Vergleich Wärmepumpe vs Gasheizung Kosten 2025",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["Wärmepumpe", "Kosten", "Euro", "€", "kWh", "Heiz"],
            "must_not_contain": ["bild.de", "google.com", "keine Ergebnisse"],
            "min_length": 400,
            "description": "Should not cite garbage homepage URLs",
        },
    },

    # ═══ Synthesis Quality (Leitthese, Quantifizierung, Hebel) ═══
    # These test the upgraded frameworks (2-phase, mandatory fields)

    "synth-prognostic-1": {
        "category": "synthesis",
        "difficulty": "hard",
        "prompt": "Recherchiere aktuelle Prognosen: Wie wird sich autonomes Fahren auf die deutsche Automobilindustrie bis 2030 auswirken? Marktvolumen, Timeline, Gewinner/Verlierer. Nutze aktuelle Studien und Quellen.",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["2030", "2025", "Milliard", "Prozent", "%", "Markt"],
            "min_numbers": 3,  # At least 3 quantitative claims
            "min_length": 800,
            "should_have_structure": True,  # Check for Leitthese, timeline, etc.
            "description": "Prognostic analysis must have thesis, numbers, timeline",
        },
    },
    "synth-strategic-1": {
        "category": "synthesis",
        "difficulty": "hard",
        "prompt": "Recherchiere und analysiere: Wie transformiert KI die Finanzdienstleistungsbranche? Aktuelle Studien, Wertschöpfungskette, Gewinner/Verlierer, Regulierung. Nutze aktuelle Quellen.",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["Wertschöpfung", "Regulier", "Bank", "Versicher", "FinTech",
                                 "Kosten", "Marge", "Effizienz"],
            "economic_levers": True,  # Must mention costs, margins, business models
            "reality_check": True,  # Must have implementation barriers
            "min_numbers": 4,
            "min_length": 1000,
            "description": "Strategic analysis must have economic levers and reality check",
        },
    },
    "synth-factual-1": {
        "category": "synthesis",
        "difficulty": "medium",
        "prompt": "Recherchiere den aktuellen Stand des Lieferkettengesetzes in Deutschland. Seit wann gilt es, für welche Unternehmen, welche Strafen drohen?",
        "check": "quality",
        "quality_criteria": {
            "must_contain_any": ["Lieferketten", "Sorgfaltspflichten", "Unternehmen",
                                 "2023", "Mitarbeiter", "Bußgeld"],
            "min_length": 300,
            "description": "Factual query about a regulation should return concrete facts",
        },
    },

    # ═══ Code Stitching ═══
    # Long code requests that would trigger truncation on cheap model

    "stitch-long-1": {
        "category": "stitch",
        "difficulty": "hard",
        "has_code": True,
        "prompt": (
            "Schreibe ein vollständiges Python-Programm: Inventarverwaltung mit Klassen "
            "(Product, Warehouse, Order), CRUD-Operationen, Bestandsprüfung, Berichterstellung, "
            "CSV-Import/Export, und Kommandozeilen-Interface mit argparse. "
            "Mindestens 200 Zeilen. Vollständig und lauffähig. if __name__. Nur Code."
        ),
        "check": "code_complete",
        "code_criteria": {
            "min_lines": 150,
            "must_contain": ["class Product", "class Warehouse", "argparse",
                             "if __name__"],
            "must_parse": True,  # Must be valid Python
            "description": "Long code should be complete (stitched if needed), not truncated",
        },
    },
    "stitch-long-2": {
        "category": "stitch",
        "difficulty": "hard",
        "has_code": True,
        "prompt": (
            "Vollständiges Python-Skript: REST API Server mit Flask/http.server. "
            "Endpoints: GET/POST/PUT/DELETE für /users und /products. "
            "In-Memory-Datenbank (dict). Input-Validierung. Error Handling. "
            "Logging. Unit Tests mit unittest am Ende. "
            "Mindestens 250 Zeilen. Nur Code, keine Erklärung."
        ),
        "check": "code_complete",
        "code_criteria": {
            "min_lines": 180,
            "must_contain": ["def ", "class ", "import ", "if __name__"],
            "must_parse": True,
            "description": "Very long code — tests whether stitching or escalation produces complete output",
        },
    },

    # ═══ Metadata Quality ═══
    # Tests that check gateway_metadata for correct routing decisions

    "meta-tool-route-1": {
        "category": "meta",
        "difficulty": "easy",
        "prompt": "Was ist die Hauptstadt von Frankreich?",
        "check": "metadata",
        "meta_criteria": {
            "expected_tier": "cheap",
            "should_not_escalate": True,
            "description": "Simple knowledge question should stay cheap, no tools",
        },
    },
    "meta-tool-route-2": {
        "category": "meta",
        "difficulty": "medium",
        "prompt": "Wie hat sich der DAX heute entwickelt?",
        "check": "metadata",
        "meta_criteria": {
            "expected_tool_any": ["web_search", "get_stock_price"],
            "description": "Current data question should trigger web_search or stock tool",
        },
    },
}

# ─── API Client ──────────────────────────────────────────────────────────────

async def call_gateway(client: httpx.AsyncClient, base_url: str, api_key: str,
                       prompt: str, model: str = "auto") -> dict:
    """Call gateway in cascade mode and return full response with metadata."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-No-Cache": "true",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.3,
    }

    for attempt in range(MAX_RETRIES + 1):
        start = time.time()
        try:
            r = await client.post(f"{base_url}/v1/chat/completions",
                                  headers=headers, json=payload, timeout=TIMEOUT)
            latency = int((time.time() - start) * 1000)

            if r.status_code == 429:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"⏳ 429, retry in {delay}s...", end=" ", flush=True)
                    await asyncio.sleep(delay)
                    continue
                return {"error": f"429 after {MAX_RETRIES} retries", "latency_ms": latency}

            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "latency_ms": latency}

            data = r.json()
            ch = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            meta = data.get("gateway_metadata", {})
            return {
                "response": ch.get("message", {}).get("content", ""),
                "model": data.get("model", "unknown"),
                "tier": meta.get("tier", "unknown"),
                "cost": usage.get("estimated_cost_usd", 0.0),
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "latency_ms": latency,
                "metadata": meta,
                "finish_reason": ch.get("finish_reason", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"⏳ {type(e).__name__}, retry in {delay}s...", end=" ", flush=True)
                await asyncio.sleep(delay)
                continue
            return {"error": str(e)[:200], "latency_ms": int((time.time() - start) * 1000)}

# ─── Scoring ─────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract code from markdown code blocks."""
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if blocks:
        return "\n".join(blocks)
    # No code blocks — maybe the whole response is code
    if "def " in text or "class " in text or "import " in text:
        return text
    return ""


def count_numbers(text: str) -> int:
    """Count distinct quantitative claims (numbers, percentages, currencies)."""
    patterns = [
        r'\d+[.,]\d+\s*%',  # percentages
        r'\d+\s*%',
        r'\d+[.,]\d+\s*(Mrd|Mio|Billion|Million|Milliard)',  # currency amounts
        r'(?:€|\$|EUR|USD)\s*\d+',
        r'\d+[.,]\d+\s*(?:€|\$)',
        r'CAGR\s*[\w\s]*\d+',
    ]
    found = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            found.add(m.group())
    return len(found)


def score_test(test_id: str, test_def: dict, result: dict) -> dict:
    """Score a single test result. Returns {score, max_score, details, pass}."""
    if "error" in result:
        return {"score": 0, "max_score": 10, "details": [f"ERROR: {result['error']}"], "pass": False}

    response = result.get("response", "")
    check = test_def.get("check", "quality")
    details = []
    score = 0
    max_score = 10

    if check == "quality":
        criteria = test_def.get("quality_criteria", {})

        # Min length
        min_len = criteria.get("min_length", 100)
        if len(response) >= min_len:
            score += 2
            details.append(f"✓ Length: {len(response)} chars (min {min_len})")
        else:
            details.append(f"✗ Too short: {len(response)} chars (min {min_len})")

        # Must contain any
        must_any = criteria.get("must_contain_any", [])
        if must_any:
            found = [kw for kw in must_any if kw.lower() in response.lower()]
            if len(found) >= 2:
                score += 3
                details.append(f"✓ Keywords: {', '.join(found[:5])}")
            elif len(found) >= 1:
                score += 1
                details.append(f"~ Partial keywords: {', '.join(found)}")
            else:
                details.append(f"✗ Missing keywords: {must_any[:5]}")

        # Must not contain
        must_not = criteria.get("must_not_contain", [])
        bad_found = [kw for kw in must_not if kw.lower() in response.lower()]
        if not bad_found:
            score += 2
            details.append("✓ No bad patterns")
        else:
            details.append(f"✗ Bad patterns found: {bad_found}")

        # Min numbers
        min_nums = criteria.get("min_numbers", 0)
        if min_nums > 0:
            num_count = count_numbers(response)
            if num_count >= min_nums:
                score += 2
                details.append(f"✓ Quantified: {num_count} numbers (min {min_nums})")
            elif num_count > 0:
                score += 1
                details.append(f"~ Partially quantified: {num_count}/{min_nums}")
            else:
                details.append(f"✗ No quantification (need {min_nums})")

        # Economic levers
        if criteria.get("economic_levers"):
            lever_terms = ["kosten", "marge", "geschäftsmodell", "revenue", "einspar",
                           "effizienz", "gewinn", "verlier", "verlierer"]
            found_levers = [t for t in lever_terms if t in response.lower()]
            if len(found_levers) >= 3:
                score += 1
                details.append(f"✓ Economic levers: {found_levers[:4]}")
            else:
                details.append(f"~ Weak economic levers: {found_levers}")

        # Reality check
        if criteria.get("reality_check"):
            barrier_terms = ["barriere", "hürde", "herausforderung", "risik",
                             "regulier", "implementierung", "kapital", "fachkräfte"]
            found_barriers = [t for t in barrier_terms if t in response.lower()]
            if len(found_barriers) >= 2:
                score += 1  # Bonus
                details.append(f"✓ Reality check: {found_barriers[:4]}")
            else:
                details.append(f"~ Weak reality check: {found_barriers}")

        # No points lost means remaining points
        score = min(score, max_score)

    elif check == "code_complete":
        criteria = test_def.get("code_criteria", {})
        code = extract_code(response)

        # Must parse as Python
        if criteria.get("must_parse"):
            try:
                compile(code, "<test>", "exec")
                score += 3
                details.append("✓ Valid Python")
            except SyntaxError as e:
                details.append(f"✗ Syntax error: {e.msg} (line {e.lineno})")

        # Min lines
        min_lines = criteria.get("min_lines", 50)
        code_lines = len([l for l in code.split("\n") if l.strip()])
        if code_lines >= min_lines:
            score += 3
            details.append(f"✓ Lines: {code_lines} (min {min_lines})")
        elif code_lines >= min_lines * 0.7:
            score += 1
            details.append(f"~ Partial: {code_lines}/{min_lines} lines")
        else:
            details.append(f"✗ Too short: {code_lines}/{min_lines} lines")

        # Must contain patterns
        must_have = criteria.get("must_contain", [])
        found_patterns = [p for p in must_have if p in code]
        if len(found_patterns) == len(must_have):
            score += 2
            details.append(f"✓ All patterns found")
        elif found_patterns:
            score += 1
            missing = [p for p in must_have if p not in code]
            details.append(f"~ Missing: {missing}")
        else:
            details.append(f"✗ Missing all: {must_have}")

        # Completeness: check for truncation signs
        truncated = (result.get("finish_reason") == "length" or
                     code.rstrip().endswith("...") or
                     code.count("```") % 2 != 0)
        if not truncated:
            score += 2
            details.append("✓ Not truncated")
        else:
            details.append("✗ Appears truncated")

        # Check metadata for code_stitched
        meta = result.get("metadata", {})
        if meta.get("code_stitched"):
            details.append("ℹ Code was stitched (continuation used) — saved premium cost!")
        if meta.get("code_repaired"):
            details.append("ℹ Code was repaired (cheap self-correction) — saved premium cost!")
        if meta.get("validation_escalated"):
            details.append(f"⚠ Escalated from {meta['validation_escalated']} (stitch+repair didn't help)")

    elif check == "metadata":
        meta_criteria = test_def.get("meta_criteria", {})
        meta = result.get("metadata", {})

        expected_tier = meta_criteria.get("expected_tier")
        if expected_tier:
            if result.get("tier") == expected_tier:
                score += 5
                details.append(f"✓ Tier: {result['tier']}")
            else:
                score += 2
                details.append(f"~ Tier: {result['tier']} (expected {expected_tier})")

        if meta_criteria.get("should_not_escalate"):
            if not meta.get("validation_escalated"):
                score += 3
                details.append("✓ No escalation")
            else:
                details.append(f"✗ Escalated: {meta.get('validation_escalated')}")

        expected_tool = meta_criteria.get("expected_tool")
        expected_tool_any = meta_criteria.get("expected_tool_any")
        if expected_tool:
            tools = meta.get("tool_calls_executed", [])
            if expected_tool in (tools or []):
                score += 5
                details.append(f"✓ Tool used: {expected_tool}")
            elif tools:
                score += 2
                details.append(f"~ Tools used: {tools} (expected {expected_tool})")
            else:
                details.append(f"✗ No tools used (expected {expected_tool})")
        elif expected_tool_any:
            tools = meta.get("tool_calls_executed", []) or []
            matched = [t for t in expected_tool_any if t in tools]
            if matched:
                score += 5
                details.append(f"✓ Tool used: {matched[0]}")
            elif tools:
                score += 2
                details.append(f"~ Tools used: {tools} (expected one of {expected_tool_any})")
            else:
                details.append(f"✗ No tools used (expected one of {expected_tool_any})")

        # Always give basic points for getting a response
        if response and len(response) > 20:
            score += 2
            details.append(f"✓ Got response ({len(response)} chars)")

        score = min(score, max_score)

    return {
        "score": score,
        "max_score": max_score,
        "details": details,
        "pass": score >= max_score * 0.6,
    }

# ─── Knowledge Base ──────────────────────────────────────────────────────────

def load_kb() -> dict:
    if KB_FILE.exists():
        return json.loads(KB_FILE.read_text())
    return {}

def save_kb(kb: dict):
    KB_FILE.write_text(json.dumps(kb, indent=2, ensure_ascii=False, default=str))

# ─── Runner ──────────────────────────────────────────────────────────────────

async def run_tests(tests_to_run: dict, base_url: str, api_key: str,
                    refresh: bool = False) -> dict:
    """Run all tests and return results."""
    kb = load_kb()
    results = {}

    async with httpx.AsyncClient() as client:
        for test_id, test_def in tests_to_run.items():
            # Check KB cache
            if not refresh and test_id in kb and "result" in kb[test_id]:
                print(f"  📦 {test_id}: cached")
                results[test_id] = kb[test_id]
                continue

            print(f"  🔄 {test_id}: ", end="", flush=True)
            result = await call_gateway(client, base_url, api_key, test_def["prompt"])

            if "error" in result:
                print(f"❌ {result['error'][:60]}")
            else:
                resp_len = len(result.get("response", ""))
                cost = result.get("cost", 0)
                tier = result.get("tier", "?")
                latency = result.get("latency_ms", 0)
                print(f"✓ {resp_len} chars | {tier} | {latency}ms | ${cost:.6f}")

            # Score
            scoring = score_test(test_id, test_def, result)

            entry = {
                "test_def": test_def,
                "result": result,
                "scoring": scoring,
                "timestamp": datetime.now().isoformat(),
            }
            results[test_id] = entry
            kb[test_id] = entry
            save_kb(kb)

            # Brief pause to avoid rate limits
            await asyncio.sleep(2)

    return results

# ─── Report ──────────────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print a formatted benchmark report."""
    print("\n" + "═" * 80)
    print("  LLM GATEWAY BENCHMARK v4 — RESULTS")
    print("═" * 80)

    # Group by category
    categories = {}
    for test_id, entry in results.items():
        cat = entry.get("test_def", {}).get("category", "unknown")
        categories.setdefault(cat, []).append((test_id, entry))

    total_score = 0
    total_max = 0
    total_pass = 0
    total_tests = 0
    total_cost = 0.0

    for cat, tests in sorted(categories.items()):
        print(f"\n  ── {cat.upper()} {'─' * (60 - len(cat))}")

        for test_id, entry in tests:
            scoring = entry.get("scoring", {})
            result = entry.get("result", {})
            test_def = entry.get("test_def", {})

            score = scoring.get("score", 0)
            max_s = scoring.get("max_score", 10)
            passed = scoring.get("pass", False)
            total_score += score
            total_max += max_s
            total_pass += 1 if passed else 0
            total_tests += 1
            cost = result.get("cost", 0)
            total_cost += cost

            icon = "✅" if passed else "❌"
            bar = "█" * int(score / max_s * 10) + "░" * (10 - int(score / max_s * 10))
            tier = result.get("tier", "?")
            latency = result.get("latency_ms", 0)
            meta = result.get("metadata", {})
            stitched = " 🧵" if meta.get("code_stitched") else ""
            escalated = f" ⬆{meta.get('validation_escalated')}" if meta.get("validation_escalated") else ""

            print(f"  {icon} {test_id:<25s} [{bar}] {score:2d}/{max_s:2d} "
                  f"| {tier:<8s} | {latency:5d}ms | ${cost:.5f}{stitched}{escalated}")

            for detail in scoring.get("details", []):
                print(f"       {detail}")

    # Summary
    pct = (total_score / total_max * 100) if total_max > 0 else 0
    print(f"\n{'═' * 80}")
    print(f"  TOTAL: {total_score}/{total_max} ({pct:.1f}%) "
          f"| {total_pass}/{total_tests} passed "
          f"| ${total_cost:.5f} total cost")

    # Feature-specific summary
    for cat, tests in sorted(categories.items()):
        cat_score = sum(t[1].get("scoring", {}).get("score", 0) for t in tests)
        cat_max = sum(t[1].get("scoring", {}).get("max_score", 10) for t in tests)
        cat_pct = (cat_score / cat_max * 100) if cat_max > 0 else 0
        cat_pass = sum(1 for t in tests if t[1].get("scoring", {}).get("pass", False))
        print(f"    {cat:>12s}: {cat_score}/{cat_max} ({cat_pct:.0f}%) — {cat_pass}/{len(tests)} passed")

    print(f"{'═' * 80}")

    # Actionable insights
    print("\n  📊 INSIGHTS:")
    search_tests = [t for t in results.values() if t.get("test_def", {}).get("category") == "search"]
    synth_tests = [t for t in results.values() if t.get("test_def", {}).get("category") == "synthesis"]
    stitch_tests = [t for t in results.values() if t.get("test_def", {}).get("category") == "stitch"]

    if search_tests:
        search_pass = sum(1 for t in search_tests if t.get("scoring", {}).get("pass", False))
        if search_pass == len(search_tests):
            print("  ✅ Web search: All queries return useful results (DDG fix working)")
        else:
            print(f"  ⚠️  Web search: {search_pass}/{len(search_tests)} — check DDG/time_filter")

    if synth_tests:
        # Check for quantification
        has_numbers = sum(1 for t in synth_tests
                         if any("Quantif" in d for d in t.get("scoring", {}).get("details", [])))
        print(f"  {'✅' if has_numbers > 0 else '⚠️'}  Synthesis: {has_numbers}/{len(synth_tests)} have quantification")

    if stitch_tests:
        stitched = sum(1 for t in stitch_tests
                       if t.get("result", {}).get("metadata", {}).get("code_stitched"))
        repaired = sum(1 for t in stitch_tests
                       if t.get("result", {}).get("metadata", {}).get("code_repaired"))
        escalated = sum(1 for t in stitch_tests
                        if t.get("result", {}).get("metadata", {}).get("validation_escalated"))
        print(f"  ℹ️  Code: {stitched} stitched, {repaired} repaired, {escalated} escalated to premium")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Gateway Benchmark v4")
    parser.add_argument("--category", "-c", choices=["search", "synthesis", "stitch", "meta"],
                        help="Run only tests in this category")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run 5 key tests only")
    parser.add_argument("--refresh", "-r", action="store_true",
                        help="Re-fetch all (ignore cache)")
    parser.add_argument("--report", action="store_true",
                        help="Just print report from saved results")
    parser.add_argument("--test", "-t", help="Run a specific test by ID")
    args = parser.parse_args()

    # Report mode
    if args.report:
        kb = load_kb()
        if not kb:
            print("No saved results. Run the benchmark first.")
            sys.exit(1)
        print_report(kb)
        return

    base_url, api_key = auto_detect_config()
    print(f"Gateway: {base_url}")

    # Select tests
    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(sorted(TESTS.keys()))}")
            sys.exit(1)
        tests_to_run = {args.test: TESTS[args.test]}
    elif args.category:
        tests_to_run = {k: v for k, v in TESTS.items() if v["category"] == args.category}
    elif args.quick:
        quick_ids = ["search-research-1", "synth-strategic-1", "stitch-long-1",
                     "search-current-1", "meta-tool-route-2"]
        tests_to_run = {k: TESTS[k] for k in quick_ids if k in TESTS}
    else:
        tests_to_run = TESTS

    print(f"Running {len(tests_to_run)} tests...\n")

    results = asyncio.run(run_tests(tests_to_run, base_url, api_key, refresh=args.refresh))
    print_report(results)


if __name__ == "__main__":
    main()
