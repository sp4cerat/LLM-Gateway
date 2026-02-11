#!/usr/bin/env python3
"""
LLM Gateway Benchmark v3.2 — Wissensdatenbank + Cache-Bypass + Retry
─────────────────────────────────────────────────────────────────────
Runs each test on ALL 3 tiers (cheap/medium/premium).
Stores results in persistent KB (JSON) — pay once, reuse forever.
Uses X-No-Cache header → each tier hits its real model.
Auto-retries on 429 with exponential backoff.

Usage:
    python benchmark.py                        # All tests (uses KB cache)
    python benchmark.py --quick                # 5 diverse tests
    python benchmark.py --category code        # Only code
    python benchmark.py --refresh              # Re-fetch all tiers
    python benchmark.py --refresh-tier cheap   # Re-fetch only cheap
    python benchmark.py --refresh-test X       # Re-fetch specific test
    python benchmark.py --refresh-scores       # Re-judge only
    python benchmark.py --report               # Report from KB only
    python benchmark.py --add                  # Add custom test
    python benchmark.py --export               # Markdown table
"""

import argparse
import asyncio
import httpx
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Auto-Config ───────────────────────────────────────────────────────────────

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


TIMEOUT = 120.0
TIERS = ["cheap", "medium", "premium"]
KB_FILE = Path("benchmark_kb.json")

# ─── Criteria ──────────────────────────────────────────────────────────────────

CRITERIA = ["correctness", "completeness", "language", "structure", "code_quality"]
CRITERIA_SHORT = {"correctness": "Korr", "completeness": "Voll", "language": "Spr",
                  "structure": "Str", "code_quality": "Code"}
WEIGHTS = {"correctness": 0.35, "completeness": 0.25, "language": 0.15,
           "structure": 0.10, "code_quality": 0.15}

# Code sub-criteria (reported separately, feed into code_quality)
CODE_SUB = ["syntax", "logic", "completeness_code", "style"]
CODE_SUB_SHORT = {"syntax": "Syntax", "logic": "Logik", "completeness_code": "Vollst", "style": "Stil"}


def weighted_pct(scores: dict) -> float:
    return sum(scores.get(c, 0) * WEIGHTS[c] for c in CRITERIA) * 10


# ─── KB ────────────────────────────────────────────────────────────────────────

def load_kb(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"tests": {}, "meta": {"created": datetime.now().isoformat(), "total_cost": 0.0}}


def save_kb(kb: dict, path: Path):
    kb["meta"]["last_updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(kb, indent=2, ensure_ascii=False))


# ─── Test Suite ────────────────────────────────────────────────────────────────

BUILTIN_TESTS = {
    # ═══ Knowledge ═══════════════════════════════════════════════════════
    "know-easy-1": {"category": "knowledge", "difficulty": "easy",
        "prompt": "Was ist die Hauptstadt von Frankreich?",
        "reference": "Paris", "check": "contains", "keywords": ["Paris"]},
    "know-easy-2": {"category": "knowledge", "difficulty": "easy",
        "prompt": "Wie viele Planeten hat unser Sonnensystem?",
        "reference": "8", "check": "contains", "keywords": ["8"]},
    "know-easy-3": {"category": "knowledge", "difficulty": "easy",
        "prompt": "In welcher Sprache ist der Linux-Kernel geschrieben?",
        "reference": "C", "check": "contains", "keywords": ["C"]},
    "know-med-1": {"category": "knowledge", "difficulty": "medium",
        "prompt": "Erkläre den Unterschied zwischen TCP und UDP in 2-3 Sätzen.",
        "reference": "TCP: verbindungsorientiert, Fehlerkorrektur. UDP: verbindungslos, schneller.",
        "check": "judge"},
    "know-med-2": {"category": "knowledge", "difficulty": "medium",
        "prompt": "Wichtigste Unterschiede zwischen GmbH und AG in Deutschland?",
        "reference": "GmbH: 25.000€, Geschäftsführer. AG: 50.000€, Vorstand+Aufsichtsrat, börsennotierbar.",
        "check": "judge"},
    "know-hard-1": {"category": "knowledge", "difficulty": "hard",
        "prompt": "Erkläre Flash Attention: Komplexitätsreduktion, SRAM vs HBM, Bedeutung für LLM-Training.",
        "reference": "O(N²)→O(N) durch Tiling in SRAM. IO-Aware: minimiert HBM-Zugriffe.",
        "check": "judge"},
    "know-hard-2": {"category": "knowledge", "difficulty": "hard",
        "prompt": "Vergleiche NVIDIA H100 vs A100: Architektur, Speicher, FP8, LLM-Training. Konkrete Zahlen.",
        "reference": "H100: Hopper, HBM3 3.35TB/s, FP8. A100: Ampere, HBM2e 2TB/s. H100 ~3x schneller.",
        "check": "judge"},
    "know-hard-3": {"category": "knowledge", "difficulty": "hard",
        "prompt": "Erkläre Mixture of Experts (MoE): Router, Expert-Selection, Load Balancing Loss. "
                  "Vergleiche Dense vs MoE Modell (z.B. Mixtral vs Llama). FLOPs, Speicher, Throughput.",
        "reference": "MoE: Router wählt Top-K Experts pro Token. Load Balancing Loss verhindert Expert-Collapse. "
                     "MoE: weniger FLOPs/Token, aber mehr Speicher. Mixtral 8x7B ≈ 12.9B aktiv vs 46.7B total.",
        "check": "judge"},
    # ═══ Math ════════════════════════════════════════════════════════════
    "math-easy-1": {"category": "math", "difficulty": "easy",
        "prompt": "Was ist 17 × 23?", "reference": "391",
        "check": "contains", "keywords": ["391"]},
    "math-easy-2": {"category": "math", "difficulty": "easy",
        "prompt": "3 Äpfel + 5 gekauft - 2 verschenkt = ?", "reference": "6",
        "check": "contains", "keywords": ["6"]},
    "math-med-1": {"category": "math", "difficulty": "medium",
        "prompt": "Ableitung von f(x) = x³ · ln(x). Zeige Rechenweg.",
        "reference": "f'(x) = x²(3ln(x)+1)", "check": "judge"},
    "math-med-2": {"category": "math", "difficulty": "medium",
        "prompt": "Integral von 1/(x²+1) dx. Zeige Herleitung und Ergebnis.",
        "reference": "arctan(x) + C", "check": "judge"},
    "math-hard-1": {"category": "math", "difficulty": "hard",
        "prompt": "100.000€: 60% ETFs (7%), 30% Anleihen (3%), 10% Tagesgeld (1.5%). Wert nach 10 Jahren Zinseszins?",
        "reference": "≈169.897€", "check": "judge"},
    "math-hard-2": {"category": "math", "difficulty": "hard",
        "prompt": "Beweise: Summe von 1 bis n = n(n+1)/2 mittels vollständiger Induktion. "
                  "Zeige Induktionsanfang, Induktionsannahme, und Induktionsschritt.",
        "reference": "IA: n=1: 1=1·2/2 ✓. IS: Summe(1..n+1) = n(n+1)/2 + (n+1) = (n+1)(n+2)/2 ✓",
        "check": "judge"},
    # ═══ Code ════════════════════════════════════════════════════════════
    "code-easy-1": {"category": "code", "difficulty": "easy", "has_code": True,
        "prompt": "Python: `is_prime(n)` — prüft Primzahl. Nur Code.",
        "reference": "def is_prime(n): ...", "check": "code_exec",
        "code_test": 'assert is_prime(2)==True\nassert is_prime(4)==False\nassert is_prime(17)==True\nassert is_prime(1)==False\nassert is_prime(97)==True\nprint("ALL TESTS PASSED")'},
    "code-easy-2": {"category": "code", "difficulty": "easy", "has_code": True,
        "prompt": "Python: `flatten(lst)` — verschachtelte Listen flachmachen, rekursiv. Nur Code.",
        "reference": "def flatten(lst): ...", "check": "code_exec",
        "code_test": 'assert flatten([1,[2,[3,4],5],6])==[1,2,3,4,5,6]\nassert flatten([])==[]\nassert flatten([1,2,3])==[1,2,3]\nassert flatten([[[[1]]]])==[1]\nprint("ALL TESTS PASSED")'},
    "code-med-1": {"category": "code", "difficulty": "medium", "has_code": True,
        "prompt": "Python: `merge_sorted(a, b)` — zwei sortierte Listen, O(n+m), kein sort(). Nur Code.",
        "reference": "def merge_sorted(a, b): ...", "check": "code_exec",
        "code_test": 'assert merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6]\nassert merge_sorted([],[1,2,3])==[1,2,3]\nassert merge_sorted([1],[])==[1]\nassert merge_sorted([1,5,9],[2,3,7,8,10])==[1,2,3,5,7,8,9,10]\nprint("ALL TESTS PASSED")'},
    "code-med-2": {"category": "code", "difficulty": "medium", "has_code": True,
        "prompt": "Python: `balanced_brackets(s)` — prüft ob Klammern balanciert sind: ()[]{}, "
                  "inkl. verschachtelung. True/False. Nur Code.",
        "reference": "def balanced_brackets(s): ...", "check": "code_exec",
        "code_test": 'assert balanced_brackets("([]){}")==True\nassert balanced_brackets("([)]")==False\n'
                     'assert balanced_brackets("")==True\nassert balanced_brackets("(((")==False\n'
                     'assert balanced_brackets("{[()]}")==True\nassert balanced_brackets("}")==False\n'
                     'print("ALL TESTS PASSED")'},
    "code-hard-1": {"category": "code", "difficulty": "hard", "has_code": True,
        "prompt": "Python: `LRUCache(capacity)` mit `get(key)→Wert/-1` und `put(key,val)`. O(1). Nur Code.",
        "reference": "class LRUCache: ...", "check": "code_exec", "min_tokens": 200,
        "code_test": 'c=LRUCache(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nassert c.get(3)==3\nprint("ALL TESTS PASSED")'},
    "code-hard-2": {"category": "code", "difficulty": "hard", "has_code": True,
        "prompt": "Python: `MinStack` mit push(x), pop(), top(), getMin(). Alle O(1). Nur Code.",
        "reference": "class MinStack: ...", "check": "code_exec", "min_tokens": 150,
        "code_test": 's=MinStack()\ns.push(-2);s.push(0);s.push(-3)\nassert s.getMin()==-3\ns.pop()\nassert s.top()==0\nassert s.getMin()==-2\ns.push(1)\nassert s.getMin()==-2\nprint("ALL TESTS PASSED")'},
    "code-trunc-1": {"category": "code", "difficulty": "hard", "has_code": True,
        "prompt": "Vollständiges Python-Skript: CSV 'name,age,city,salary' (20 Einträge) generieren, Statistiken berechnen, JSON speichern. if __name__. Keine ext. Deps.",
        "reference": "csv+json script", "check": "code_exec", "min_tokens": 400,
        "code_test": 'import os\nprint("SCRIPT EXECUTED")\nfor f in ["data.csv","stats.json","output.json","results.json"]:\n    if os.path.exists(f): os.remove(f)\nprint("ALL TESTS PASSED")'},
    # ═══ Reasoning ═══════════════════════════════════════════════════════
    "reason-easy-1": {"category": "reasoning", "difficulty": "easy",
        "prompt": "Wenn alle Äpfel Früchte sind und einige Früchte rot sind, "
                  "sind dann alle Äpfel rot? Begründe kurz.",
        "reference": "Nein. Nur 'einige' Früchte rot → kein Schluss auf alle Äpfel.",
        "check": "judge"},
    "reason-med-1": {"category": "reasoning", "difficulty": "medium",
        "prompt": "3 Lichtschalter, 1 Lampe im Nebenraum, 1x reingehen. Wie findest du den richtigen Schalter?",
        "reference": "Schalter 1 an→warten→aus. Schalter 2 an. an=2, warm=1, kalt=3.",
        "check": "judge"},
    "reason-med-2": {"category": "reasoning", "difficulty": "medium",
        "prompt": "Ein Bauer hat 17 Schafe. Alle außer 9 sterben. Wie viele hat er noch?",
        "reference": "9. 'Alle außer 9' = 9 überleben.",
        "check": "judge"},
    "reason-hard-1": {"category": "reasoning", "difficulty": "hard",
        "prompt": "12 Kugeln, eine schwerer/leichter. Balkenwaage, 3 Wiegungen. Strategie?",
        "reference": "3×4 Gruppen, systematische Elimination.", "check": "judge"},
    "reason-hard-2": {"category": "reasoning", "difficulty": "hard",
        "prompt": "Fünf Piraten teilen 100 Goldmünzen. Der älteste schlägt vor, Abstimmung (Mehrheit). "
                  "Bei Ablehnung wird er über Bord geworfen und nächster schlägt vor. "
                  "Alle rational und gierig. Wie verteilt der älteste? Zeige Rückwärts-Induktion.",
        "reference": "98-0-1-0-1. Rückwärts: 2 Piraten: 100-0. 3: 99-0-1. 4: 99-0-1-0. 5: 98-0-1-0-1.",
        "check": "judge"},
    # ═══ Translation ═════════════════════════════════════════════════════
    "trans-easy-1": {"category": "translation", "difficulty": "easy",
        "prompt": "Übersetze: 'Der frühe Vogel fängt den Wurm, aber die zweite Maus bekommt den Käse.'",
        "reference": "The early bird catches the worm, but the second mouse gets the cheese.",
        "check": "judge"},
    "trans-med-1": {"category": "translation", "difficulty": "medium",
        "prompt": "Übersetze ins Englische (idiomatisch, nicht wörtlich): "
                  "'Das ist nicht mein Bier. Da hast du wohl den Bock zum Gärtner gemacht.'",
        "reference": "That's not my problem/cup of tea. You've set the fox to guard the henhouse.",
        "check": "judge"},
    "trans-hard-1": {"category": "translation", "difficulty": "hard",
        "prompt": "Ins Deutsche (Fachbegriffe): 'The garbage collector uses generational approach. Young objects use copying collector, tenured use mark-and-sweep.'",
        "reference": "Generationenbasierter GC. Junge→Copying, langlebige→Mark-and-Sweep.",
        "check": "judge"},
    # ═══ Creative ════════════════════════════════════════════════════════
    "creative-easy-1": {"category": "creative", "difficulty": "easy",
        "prompt": "Schreibe einen kurzen Witz über einen Informatiker.",
        "reference": "Ein lustiger Witz mit IT-Bezug.", "check": "judge"},
    "creative-med-1": {"category": "creative", "difficulty": "medium",
        "prompt": "Limerick (AABBA) über einen Programmierer der einen Bug sucht.",
        "reference": "5 Zeilen, AABBA, Thema Bug.", "check": "judge"},
    "creative-hard-1": {"category": "creative", "difficulty": "hard",
        "prompt": "Schreibe ein Sonett (14 Zeilen, Reimschema ABAB CDCD EFEF GG) über künstliche Intelligenz. "
                  "Jede Zeile soll metrisch sein (5-hebiger Jambus bevorzugt).",
        "reference": "14 Zeilen, Reimschema ABAB CDCD EFEF GG, Thema KI, metrisch.",
        "check": "judge"},
    # ═══ Vision (multimodal) ═════════════════════════════════════════════
    "vision-easy-1": {"category": "vision", "difficulty": "easy",
        "prompt": "Was steht in diesem Bild? Gib den Text exakt wieder.",
        "reference": "Hello World 42", "check": "contains", "keywords": ["Hello", "World", "42"],
        "image_key": "text_simple"},
    "vision-easy-2": {"category": "vision", "difficulty": "easy",
        "prompt": "Beschreibe die Formen und Farben in diesem Bild.",
        "reference": "Roter Kreis und blaues Rechteck/Quadrat.",
        "check": "judge",
        "image_key": "shapes"},
    "vision-med-1": {"category": "vision", "difficulty": "medium",
        "prompt": "Analysiere dieses Balkendiagramm: Welche Kategorie hat den höchsten Wert? "
                  "Welche den niedrigsten? Schätze die ungefähren Werte.",
        "reference": "C ist am höchsten (~200), D am niedrigsten (~90). A ~180, B ~120.",
        "check": "judge",
        "image_key": "chart"},
    "vision-hard-1": {"category": "vision", "difficulty": "hard",
        "prompt": "Lies den Code in diesem Screenshot. Beschreibe was die Funktion tut. "
                  "Gibt es einen Bug? Wenn ja, erkläre ihn.",
        "reference": "fibonacci() — gibt n-te Fibonacci-Zahl. Kommentar suggeriert Bug, "
                     "aber return b ist korrekt für iterative Fibonacci.",
        "check": "judge",
        "image_key": "code_screenshot"},
}


# ─── API Client with Retry ────────────────────────────────────────────────────

MAX_RETRIES = 4
RETRY_BASE_DELAY = 15  # seconds — OpenRouter limits are per-minute
_adaptive_delay = 6.0  # Increases after 429s, decreases on success
_min_delay = 4.0

def _update_delay(hit_429: bool):
    """Adaptive delay: slow down after 429, speed up on success."""
    global _adaptive_delay
    if hit_429:
        _adaptive_delay = min(_adaptive_delay * 2.0, 30.0)  # Aggressive backoff, max 30s
    else:
        _adaptive_delay = max(_adaptive_delay * 0.85, _min_delay)  # Slow recovery

# ─── Load Vision Test Images ──────────────────────────────────────────────────

VISION_IMAGES = {}

def load_vision_images():
    """Load base64 test images from benchmark_images.json."""
    global VISION_IMAGES
    img_path = Path(__file__).parent / "benchmark_images.json"
    if img_path.exists():
        VISION_IMAGES = json.loads(img_path.read_text())
        print(f"   Bilder:   {len(VISION_IMAGES)} geladen ({', '.join(VISION_IMAGES.keys())})")
    else:
        print(f"   Bilder:   ⚠️  benchmark_images.json nicht gefunden — Vision-Tests übersprungen")


async def call_tier(client: httpx.AsyncClient, base_url: str, api_key: str,
                    prompt: str, tier: str, max_retries: int = MAX_RETRIES,
                    image_b64: str = None) -> dict:
    """Call a specific tier with X-No-Cache and retry on 429.
    If image_b64 is provided, send as multimodal message."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-No-Cache": "true",  # Bypass gateway cache → each tier hits its real model
    }

    # Build message content (text-only or multimodal)
    if image_b64:
        content = [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    payload = {
        "model": tier,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0.3,
    }

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            r = await client.post(f"{base_url}/v1/chat/completions",
                                  headers=headers, json=payload, timeout=TIMEOUT)
            latency = int((time.time() - start) * 1000)

            if r.status_code == 429:
                _update_delay(True)
                if attempt < max_retries:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), 60)  # 15, 30, 60, 60
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        delay = max(delay, int(retry_after))
                    print(f"⏳ 429, retry in {delay}s...", end=" ", flush=True)
                    await asyncio.sleep(delay)
                    continue
                return {"error": f"429 after {max_retries} retries", "latency_ms": latency}

            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:150]}", "latency_ms": latency}

            _update_delay(False)
            data = r.json()
            ch = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            meta = data.get("gateway_metadata", {})
            return {
                "response": ch.get("message", {}).get("content", ""),
                "model": data.get("model", "unknown"),
                "tier_actual": meta.get("tier", tier),
                "cost": usage.get("estimated_cost_usd", 0.0),
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "latency_ms": latency,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            if attempt < max_retries:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), 60)
                print(f"⏳ {type(e).__name__}, retry in {delay}s...", end=" ", flush=True)
                await asyncio.sleep(delay)
                continue
            return {"error": str(e)[:200], "latency_ms": int((time.time() - start) * 1000)}

    return {"error": "max retries exceeded", "latency_ms": 0}


# ─── Code Evaluation (4 Sub-Criteria) ─────────────────────────────────────────

def extract_code(response: str) -> str:
    for pat in [r'```python\n(.*?)```', r'```py\n(.*?)```', r'```\n(.*?)```']:
        m = re.search(pat, response, re.DOTALL)
        if m: return m.group(1).strip()
    lines, active = [], False
    for l in response.split('\n'):
        s = l.rstrip()
        if any(s.startswith(k) for k in ['def ','class ','import ','from ','if ','for ','while ','    ','\t']):
            active = True
        if active: lines.append(l)
    return '\n'.join(lines).strip() if lines else response.strip()


def check_truncation(response: str, min_tok: int = 0) -> bool:
    if response.count('```') % 2 != 0: return True
    for o, c in [('{','}'),('[',']'),('(',')') ]:
        if response.count(o) - response.count(c) > 1: return True
    if min_tok and len(response.split()) < min_tok * 0.5: return True
    return False


def eval_code(response: str, test_def: dict) -> dict:
    """Evaluate code with 4 sub-criteria: syntax, logic, completeness, style.
    Returns dict with sub-scores (0-10) and aggregated code_quality."""
    code = extract_code(response)
    result = {
        "code_extracted": bool(code.strip()),
        "syntax": 0.0, "logic": 0.0, "completeness_code": 0.0, "style": 0.0,
        "exec_output": "", "code_tests_pass": False, "code_executes": False,
    }

    if not code.strip():
        result["exec_output"] = "No code found in response"
        return result

    # ─── 1. SYNTAX: Does it parse? ─────────────────────────────
    try:
        compile(code, "<benchmark>", "exec")
        result["syntax"] = 10.0  # Parses without error
    except SyntaxError as e:
        result["syntax"] = 0.0
        result["exec_output"] = f"SyntaxError: {e}"
        # Can't test further if syntax is broken
        result["completeness_code"] = 2.0 if len(code.split('\n')) > 3 else 0.0
        return result

    # ─── 2. LOGIC: Does it run + pass tests? ───────────────────
    test_code = test_def.get("code_test", "")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n\n" + test_code)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True,
                           timeout=10, cwd=tempfile.gettempdir())
        output = (r.stdout + r.stderr)[:500]
        result["exec_output"] = output

        if r.returncode != 0 and "ALL TESTS PASSED" not in r.stdout:
            # Runs but crashes
            result["code_executes"] = False
            if "AssertionError" in output or "AssertionError" in r.stderr or "assert" in output.lower():
                result["logic"] = 3.0   # Logic error (assertions fail)
                result["syntax"] = 10.0  # Syntax was fine
            elif "NameError" in output or "AttributeError" in output:
                result["logic"] = 2.0   # Missing definitions
            elif "TypeError" in output or "ValueError" in output:
                result["logic"] = 2.0   # Type/value errors
            else:
                result["logic"] = 1.0   # Other runtime error
        elif "ALL TESTS PASSED" in r.stdout:
            result["code_executes"] = True
            result["code_tests_pass"] = True
            result["logic"] = 10.0
        else:
            # Runs without error but no test output (partial)
            result["code_executes"] = True
            result["logic"] = 6.0
    except subprocess.TimeoutExpired:
        result["exec_output"] = "TIMEOUT (>10s)"
        result["logic"] = 2.0  # Infinite loop = logic error
    except Exception as e:
        result["exec_output"] = str(e)[:200]
        result["logic"] = 1.0
    finally:
        try: os.unlink(tmp)
        except: pass

    # ─── 3. COMPLETENESS: All requested parts present? ─────────
    truncated = check_truncation(response, test_def.get("min_tokens", 0))
    if truncated:
        result["completeness_code"] = 3.0
    elif result["code_tests_pass"]:
        result["completeness_code"] = 10.0
    elif result["code_executes"]:
        result["completeness_code"] = 7.0
    else:
        # Check if the main constructs are present
        has_func = "def " in code or "class " in code
        has_return = "return " in code
        result["completeness_code"] = 5.0 if (has_func and has_return) else 2.0

    # ─── 4. STYLE: Code quality heuristics ─────────────────────
    style_score = 7.0  # baseline
    lines = code.split('\n')
    # Docstrings/comments
    if any('"""' in l or "'''" in l or l.strip().startswith('#') for l in lines):
        style_score += 1.0
    # Reasonable line length
    if all(len(l) < 120 for l in lines):
        style_score += 0.5
    # Type hints
    if re.search(r'def \w+\([^)]*:\s*\w+', code):
        style_score += 0.5
    # No global state / magic numbers (penalty)
    if re.search(r'^\w+\s*=\s*\d+', code, re.MULTILINE) and 'def ' not in code[:50]:
        style_score -= 1.0
    # Very short = probably not production quality
    if len(lines) < 3:
        style_score -= 2.0
    result["style"] = max(0.0, min(10.0, style_score))

    return result


# ─── Non-Code Evaluation ──────────────────────────────────────────────────────

def check_contains(response: str, keywords: list) -> bool:
    low = response.lower()
    return all(k.lower() in low for k in keywords)


# ─── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_TPL = """Bewerte diese LLM-Antwort auf 0-10 pro Kriterium.

═ FRAGE ═
{prompt}

═ ANTWORT ═
{response}

═ REFERENZ ═
{reference}

{extra}

WICHTIG bei Code-Aufgaben:
- Wenn der Prompt "Nur Code" verlangt, ist KEINE Erklärung nötig → LANGUAGE und STRUCTURE trotzdem hoch bewerten wenn der Code sauber ist
- CORRECTNESS bei Code = produziert korrektes Ergebnis? Tests bestanden?
- COMPLETENESS bei Code = alle geforderten Funktionen/Klassen vorhanden?
- LANGUAGE bei Code = saubere Syntax, gute Variablennamen, lesbar
- STRUCTURE bei Code = gut organisiert, logischer Aufbau, Fehlerbehandlung

CORRECTNESS: Sachlich korrekt / Ergebnis stimmt? 10=perfekt 0=falsch
COMPLETENESS: Vollständig beantwortet? 10=komplett 0=kaum
LANGUAGE: Sprache/Grammatik/Code-Lesbarkeit? 10=perfekt 0=unverständlich
STRUCTURE: Klar strukturiert/organisiert? 10=exzellent 0=chaotisch
CODE_QUALITY: Code-Qualität? (nicht-Code=7) 10=produktionsreif 0=kaputt

Antwort EXAKT:
CORRECTNESS: <0-10>
COMPLETENESS: <0-10>
LANGUAGE: <0-10>
STRUCTURE: <0-10>
CODE_QUALITY: <0-10>
COMMENT: <1-2 Sätze>"""


async def judge_response(client, base_url, api_key, test_def, response, reference, extra=""):
    prompt = JUDGE_TPL.format(
        prompt=test_def["prompt"], response=response[:3000],
        reference=reference[:2000], extra=extra or "",
    )
    # Judge uses MEDIUM tier (Gemini 3 Flash) — reliable, fast, no rate limit issues
    # Premium is reserved for test responses only
    r = await call_tier(client, base_url, api_key, prompt, "medium", max_retries=2)
    if "error" in r:
        return {"correctness": 5, "completeness": 5, "language": 5, "structure": 5,
                "code_quality": 7 if not test_def.get("has_code") else 5}, "Judge error"

    text = r.get("response", "")
    scores = {}
    for c in CRITERIA:
        m = re.search(rf'{c.upper()}:\s*(\d+(?:\.\d+)?)', text)
        scores[c] = max(0, min(10, float(m.group(1)))) if m else 5.0
    if not test_def.get("has_code") and scores.get("code_quality", 0) == 0:
        scores["code_quality"] = 7.0

    cm = re.search(r'COMMENT:\s*(.+)', text, re.DOTALL)
    return scores, (cm.group(1).strip()[:200] if cm else "")


# ─── Fetch All Tiers ──────────────────────────────────────────────────────────

async def fetch_test_tiers(client, base_url, api_key, test_id, test_def, kb,
                           refresh_tiers=None):
    """Fetch responses for all tiers, using KB cache where available."""
    if test_id not in kb["tests"]:
        kb["tests"][test_id] = {
            "prompt": test_def["prompt"], "category": test_def["category"],
            "difficulty": test_def["difficulty"], "reference": test_def.get("reference", ""),
            "check": test_def["check"], "tiers": {},
        }

    entry = kb["tests"][test_id]
    cost = 0.0

    for tier in TIERS:
        cached = entry["tiers"].get(tier)
        if cached and not cached.get("error") and tier not in (refresh_tiers or []):
            emoji = {"cheap": "⚡", "medium": "🔵", "premium": "🟣"}[tier]
            model = cached.get('model', '?').split('/')[-1][:20]
            print(f"    {emoji} {tier}: 💾 cache ({model})")
            continue

        emoji = {"cheap": "⚡", "medium": "🔵", "premium": "🟣"}[tier]
        print(f"    {emoji} {tier}: ", end="", flush=True)

        # Get image for vision tests
        image_b64 = None
        image_key = test_def.get("image_key")
        if image_key:
            image_b64 = VISION_IMAGES.get(image_key)
            if not image_b64:
                print(f"⏭️  Bild '{image_key}' nicht geladen")
                continue

        result = await call_tier(client, base_url, api_key, test_def["prompt"], tier,
                                 image_b64=image_b64)

        if "error" in result:
            print(f"❌ {result['error'][:60]}")
            entry["tiers"][tier] = {"error": result["error"], "timestamp": datetime.now().isoformat()}
        else:
            model = result['model'].split('/')[-1][:20]
            print(f"✅ {model} ({result['latency_ms']}ms, ${result['cost']:.5f})")
            entry["tiers"][tier] = result
            cost += result.get("cost", 0)

        await asyncio.sleep(_adaptive_delay)  # Adaptive delay between tier calls

    kb["meta"]["total_cost"] = kb["meta"].get("total_cost", 0) + cost
    return cost


# ─── Score All Tiers ──────────────────────────────────────────────────────────

async def score_test(client, base_url, api_key, test_id, test_def, kb,
                     refresh_scores=False):
    entry = kb["tests"].get(test_id)
    if not entry: return

    premium_resp = entry["tiers"].get("premium", {}).get("response", "")
    # Use premium response as reference, but fall back to builtin reference if premium failed
    reference = premium_resp if premium_resp else test_def.get("reference", "")

    for tier in TIERS:
        td = entry["tiers"].get(tier, {})
        if not td or td.get("error"): continue
        if td.get("scores") and not refresh_scores: continue

        response = td.get("response", "")
        emoji = {"cheap": "⚡", "medium": "🔵", "premium": "🟣"}[tier]

        # ─── Code evaluation (4 sub-criteria) ─────────────────
        if test_def["check"] == "code_exec":
            code_result = eval_code(response, test_def)
            td["code_sub"] = {k: code_result[k] for k in CODE_SUB}
            td["code_executes"] = code_result["code_executes"]
            td["code_tests_pass"] = code_result["code_tests_pass"]
            td["exec_output"] = code_result["exec_output"][:300]
            td["truncated"] = check_truncation(response, test_def.get("min_tokens", 0))

            # Build extra context for judge
            sub = code_result
            extra_lines = [
                f"Code-Syntax: {'✅ fehlerfrei' if sub['syntax'] >= 8 else '❌ SyntaxError'}",
                f"Code-Logik: {'✅ alle Tests bestanden' if sub['code_tests_pass'] else '❌ Tests fehlgeschlagen'}",
                f"Code-Vollständigkeit: {'✂️ abgeschnitten' if td['truncated'] else '✅ komplett'}",
            ]
            extra = "\n".join(extra_lines)

            # Judge for other criteria
            print(f"    {emoji} {tier}: 🧑‍⚖️ ...", end=" ", flush=True)
            scores, comment = await judge_response(
                client, base_url, api_key, test_def, response, reference, extra
            )

            # Override code_quality from actual sub-criteria (weighted)
            sub_avg = (sub["syntax"] * 0.25 + sub["logic"] * 0.40
                       + sub["completeness_code"] * 0.20 + sub["style"] * 0.15)
            scores["code_quality"] = sub_avg

            # Floor: if ALL tests pass, code is correct → minimum scores
            if code_result["code_tests_pass"]:
                scores["correctness"] = max(scores.get("correctness", 0), 9.0)
                scores["completeness"] = max(scores.get("completeness", 0), 8.0)
                scores["language"] = max(scores.get("language", 0), 7.0)
                scores["structure"] = max(scores.get("structure", 0), 7.0)

            # Also adjust completeness from judge if truncated
            if td["truncated"]:
                scores["completeness"] = min(scores.get("completeness", 5), 3.0)

            td["scores"] = scores
            td["comment"] = comment
            print(f"{weighted_pct(scores):.0f}% (Syn:{sub['syntax']:.0f} Log:{sub['logic']:.0f} "
                  f"Voll:{sub['completeness_code']:.0f} Stil:{sub['style']:.0f})")

        elif test_def["check"] == "contains":
            ok = check_contains(response, test_def.get("keywords", []))
            if ok:
                td["scores"] = {"correctness": 10, "completeness": 9, "language": 8,
                                "structure": 8, "code_quality": 7}
                td["comment"] = "Keyword-Check bestanden"
            else:
                td["scores"] = {"correctness": 0, "completeness": 2, "language": 5,
                                "structure": 5, "code_quality": 7}
                td["comment"] = f"Falsch. Erwartet: {test_def.get('keywords')}"
            td["truncated"] = False
            # No print needed for simple contains

        else:  # judge
            td["truncated"] = check_truncation(response)
            print(f"    {emoji} {tier}: 🧑‍⚖️ ...", end=" ", flush=True)
            scores, comment = await judge_response(
                client, base_url, api_key, test_def, response, reference
            )
            td["scores"] = scores
            td["comment"] = comment
            print(f"{weighted_pct(scores):.0f}%")

        await asyncio.sleep(_adaptive_delay)  # Adaptive delay between judge calls

# ─── Add Test ──────────────────────────────────────────────────────────────────

def add_test_interactive(kb):
    print("\n📝 Neuen Test hinzufügen\n" + "─" * 40)
    tid = input("Test-ID (z.B. know-med-3): ").strip()
    if not tid: print("Abgebrochen."); return
    if tid in BUILTIN_TESTS or tid in kb.get("custom_tests", {}):
        print(f"⚠️ {tid} existiert bereits!"); return
    cats = ["knowledge", "code", "math", "reasoning", "translation", "creative"]
    cat = input(f"Kategorie ({'/'.join(cats)}): ").strip()
    diff = input("Schwierigkeit (easy/medium/hard): ").strip()
    prompt = input("Prompt: ").strip()
    if not prompt: print("Abgebrochen."); return
    ref = input("Referenz-Antwort: ").strip()
    check = input("Check (contains/judge/code_exec) [judge]: ").strip() or "judge"
    td = {"category": cat, "difficulty": diff, "prompt": prompt, "reference": ref, "check": check}
    if check == "contains":
        td["keywords"] = [k.strip() for k in input("Keywords (komma): ").split(",")]
    elif check == "code_exec":
        td["has_code"] = True
        print("Test-Code (leere Zeile = Ende):")
        lines = []
        while True:
            l = input()
            if l == "": break
            lines.append(l)
        td["code_test"] = "\n".join(lines)
    if "custom_tests" not in kb: kb["custom_tests"] = {}
    kb["custom_tests"][tid] = td
    save_kb(kb, KB_FILE)
    print(f"\n✅ '{tid}' hinzugefügt! Teste mit: python benchmark.py --test-id {tid}")


# ─── Report ────────────────────────────────────────────────────────────────────

def bar100(val, w=20):
    f = int(max(0, min(val, 100)) / 100 * w)
    return "█" * f + "░" * (w - f) + f" {val:.0f}%"


def generate_report(kb):
    L = []
    ts = kb["meta"].get("last_updated", "?")
    total_cost = kb["meta"].get("total_cost", 0)

    L.append("")
    L.append("╔════════════════════════════════════════════════════════════════════════════╗")
    L.append("║             LLM GATEWAY BENCHMARK — WISSENSDATENBANK v3.2                ║")
    L.append(f"║  {ts:50s}Kosten: ${total_cost:.4f}   ║")
    L.append("╚════════════════════════════════════════════════════════════════════════════╝")
    L.append("")

    test_ids = sorted(kb["tests"].keys())
    if not test_ids:
        L.append("  (keine Testergebnisse vorhanden)")
        return "\n".join(L)

    # ─── 3-Tier Matrix ────────────────────────────────────────
    L.append("═══ 3-TIER VERGLEICHSMATRIX ═════════════════════════════════════════════════")
    L.append(f"  {'Test-ID':<16} │{'CHEAP':^20}│{'MEDIUM':^20}│{'PREMIUM':^20}│")
    L.append(f"  {'─'*16}─┼{'─'*20}┼{'─'*20}┼{'─'*20}┤")

    tier_scores = {t: [] for t in TIERS}
    for tid in test_ids:
        entry = kb["tests"][tid]
        parts = [f"  {tid:<16}"]
        for tier in TIERS:
            td = entry["tiers"].get(tier, {})
            if td.get("error"):
                parts.append(f"{'ERROR':^20}")
            elif td.get("scores"):
                pct = weighted_pct(td["scores"])
                tier_scores[tier].append(pct)
                tr = "✂" if td.get("truncated") else " "
                cx = "✓" if td.get("code_tests_pass") else ("✗" if td.get("code_tests_pass") is False else " ")
                st = "✅" if pct >= 60 else ("⚠️" if pct >= 40 else "❌")
                model = td.get("model", "?").split("/")[-1][:8]
                parts.append(f" {pct:4.0f}% {st}{tr}{cx} {model:<8}")
            else:
                parts.append(f"{'—':^20}")
        L.append(" │".join(parts) + "│")

    # Averages
    L.append(f"  {'─'*16}─┼{'─'*20}┼{'─'*20}┼{'─'*20}┤")
    avg_parts = [f"  {'DURCHSCHNITT':<16}"]
    tier_avg = {}
    for tier in TIERS:
        if tier_scores[tier]:
            avg = sum(tier_scores[tier]) / len(tier_scores[tier])
            tier_avg[tier] = avg
            g = "A" if avg >= 80 else "B" if avg >= 70 else "C" if avg >= 60 else "D" if avg >= 50 else "F"
            avg_parts.append(f" {avg:4.0f}%  = {g}           ")
        else:
            tier_avg[tier] = 0
            avg_parts.append(f"{'—':^20}")
    L.append(" │".join(avg_parts) + "│")
    L.append("")

    # ─── Criteria per Tier ────────────────────────────────────
    L.append("═══ KRITERIEN PRO TIER (⌀) ═════════════════════════════════════════════════")
    L.append(f"  {'Kriterium':<12} {'Gew':>4} │{'CHEAP':>8} │{'MEDIUM':>8} │{'PREMIUM':>8} │ Δ c→p")
    L.append(f"  {'─'*12}─{'─'*4}─┼{'─'*8}─┼{'─'*8}─┼{'─'*8}─┤{'─'*6}")

    for c in CRITERIA:
        tavg = {}
        for tier in TIERS:
            vals = [kb["tests"][t]["tiers"].get(tier, {}).get("scores", {}).get(c, 0)
                    for t in test_ids if kb["tests"][t]["tiers"].get(tier, {}).get("scores")]
            tavg[tier] = sum(vals) / len(vals) if vals else 0
        d = tavg["premium"] - tavg["cheap"]
        L.append(f"  {CRITERIA_SHORT[c]:<12} {WEIGHTS[c]*100:3.0f}% │{tavg['cheap']:7.1f}  │{tavg['medium']:7.1f}  │"
                 f"{tavg['premium']:7.1f}  │{d:+5.1f}")

    d_w = tier_avg.get("premium", 0) - tier_avg.get("cheap", 0)
    L.append(f"  {'GEWICHTET':<12}      │{tier_avg.get('cheap',0):6.0f}%  │{tier_avg.get('medium',0):6.0f}%  │"
             f"{tier_avg.get('premium',0):6.0f}%  │{d_w:+4.0f}%")
    L.append("")

    # ─── Code Sub-Criteria ────────────────────────────────────
    code_tests = [t for t in test_ids if kb["tests"][t].get("check") == "code_exec"
                  or BUILTIN_TESTS.get(t, {}).get("check") == "code_exec"]
    if code_tests:
        L.append("═══ CODE-BEWERTUNG (4 Sub-Kriterien) ═════════════════════════════════════")
        L.append(f"  {'Test-ID':<16} │{'Tier':<8} {'Syntax':>6} {'Logik':>6} {'Vollst':>6} {'Stil':>6} │ {'Tests':>5} {'Ges%':>5}")
        L.append(f"  {'─'*16}─┼{'─'*8}{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}─┼{'─'*5}─{'─'*5}")

        for tid in code_tests:
            for tier in TIERS:
                td = kb["tests"][tid]["tiers"].get(tier, {})
                sub = td.get("code_sub", {})
                if not sub: continue
                emoji = {"cheap": "⚡", "medium": "🔵", "premium": "🟣"}[tier]
                tp = "✅" if td.get("code_tests_pass") else "❌"
                pct = weighted_pct(td["scores"]) if td.get("scores") else 0
                L.append(f"  {tid:<16} │{emoji}{tier:<7} {sub.get('syntax',0):5.0f}  {sub.get('logic',0):5.0f}  "
                         f"{sub.get('completeness_code',0):5.0f}  {sub.get('style',0):5.0f} │  {tp}   {pct:4.0f}%")
            L.append(f"  {'─'*16}─┼{'─'*8}{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}─┼{'─'*5}─{'─'*5}")
        L.append("")

    # ─── By Category ──────────────────────────────────────────
    L.append("═══ NACH KATEGORIE ════════════════════════════════════════════════════════")
    L.append(f"  {'Kategorie':<14} {'N':>3} │{'CHEAP':>7} │{'MEDIUM':>8} │{'PREMIUM':>8} │ Δ c→p")
    L.append(f"  {'─'*14}─{'─'*3}─┼{'─'*7}─┼{'─'*8}─┼{'─'*8}─┤{'─'*6}")
    categories = sorted(set(kb["tests"][t].get("category", "?") for t in test_ids))
    for cat in categories:
        cids = [t for t in test_ids if kb["tests"][t].get("category") == cat]
        n = len(cids)
        tavg = {}
        for tier in TIERS:
            vals = [weighted_pct(kb["tests"][t]["tiers"].get(tier, {}).get("scores", {}))
                    for t in cids if kb["tests"][t]["tiers"].get(tier, {}).get("scores")]
            tavg[tier] = sum(vals) / len(vals) if vals else 0
        loss = tavg["premium"] - tavg["cheap"]
        flag = " ⚠️" if loss > 20 else ""
        L.append(f"  {cat:<14} {n:3d} │{tavg['cheap']:6.0f}% │{tavg['medium']:7.0f}% │"
                 f"{tavg['premium']:7.0f}% │{loss:+4.0f}%{flag}")
    L.append("")

    # ─── By Difficulty ────────────────────────────────────────
    L.append("═══ NACH SCHWIERIGKEIT ═════════════════════════════════════════════════════")
    L.append(f"  {'Level':<8} {'N':>3} │{'CHEAP':>7} │{'MEDIUM':>8} │{'PREMIUM':>8} │ Δ c→p")
    L.append(f"  {'─'*8}─{'─'*3}─┼{'─'*7}─┼{'─'*8}─┼{'─'*8}─┤{'─'*6}")
    for diff in ["easy", "medium", "hard"]:
        dids = [t for t in test_ids if kb["tests"][t].get("difficulty") == diff]
        if not dids: continue
        tavg = {}
        for tier in TIERS:
            vals = [weighted_pct(kb["tests"][t]["tiers"].get(tier, {}).get("scores", {}))
                    for t in dids if kb["tests"][t]["tiers"].get(tier, {}).get("scores")]
            tavg[tier] = sum(vals) / len(vals) if vals else 0
        loss = tavg["premium"] - tavg["cheap"]
        flag = " ⚠️" if loss > 20 else ""
        L.append(f"  {diff:<8} {len(dids):3d} │{tavg['cheap']:6.0f}% │{tavg['medium']:7.0f}% │"
                 f"{tavg['premium']:7.0f}% │{loss:+4.0f}%{flag}")
    L.append("")

    # ─── Cost ─────────────────────────────────────────────────
    L.append("═══ KOSTEN-ANALYSE ════════════════════════════════════════════════════════")
    for tier in TIERS:
        tc = sum(kb["tests"][t]["tiers"].get(tier, {}).get("cost", 0) for t in test_ids)
        lats = [kb["tests"][t]["tiers"].get(tier, {}).get("latency_ms", 0) for t in test_ids
                if kb["tests"][t]["tiers"].get(tier, {}).get("latency_ms")]
        al = sum(lats) / len(lats) if lats else 0
        avg_q = tier_avg.get(tier, 0)
        eff = f"{avg_q / (tc / len(test_ids) * 1000):.1f} pts/m$" if tc > 0 else "∞"
        L.append(f"  {tier:<10} ∑${tc:.5f}  ⌀{al:.0f}ms  ⌀{avg_q:.0f}%  Effizienz: {eff}")
    L.append("")

    # ─── P/L Scorecard ────────────────────────────────────────
    L.append("═══ PREIS/LEISTUNGS-SCORECARD ═══════════════════════════════════════════════")
    L.append("")

    prem_cost = sum(kb["tests"][t]["tiers"].get("premium", {}).get("cost", 0) for t in test_ids)
    cheap_cost = sum(kb["tests"][t]["tiers"].get("cheap", {}).get("cost", 0) for t in test_ids)
    med_cost = sum(kb["tests"][t]["tiers"].get("medium", {}).get("cost", 0) for t in test_ids)

    # ── Strategy A: Classic (easy→cheap, medium→medium, hard→premium) ──
    sim_a_scores, sim_a_cost = [], 0
    for tid in test_ids:
        diff = kb["tests"][tid].get("difficulty", "medium")
        sim_tier = {"easy": "cheap", "medium": "medium", "hard": "premium"}.get(diff, "medium")
        td = kb["tests"][tid]["tiers"].get(sim_tier, {})
        if td.get("scores"):
            sim_a_scores.append(weighted_pct(td["scores"]))
        sim_a_cost += td.get("cost", 0)
    sim_a_avg = sum(sim_a_scores) / len(sim_a_scores) if sim_a_scores else 0

    # ── Strategy B: Verify-then-Escalate (all→medium, escalate on low completeness) ──
    sim_b_scores, sim_b_cost, escalated_count = [], 0, 0
    for tid in test_ids:
        md = kb["tests"][tid]["tiers"].get("medium", {})
        pd = kb["tests"][tid]["tiers"].get("premium", {})
        m_scores = md.get("scores", {})
        p_scores = pd.get("scores", {})

        # Simulate: send to medium first (always pay medium cost)
        sim_b_cost += md.get("cost", 0)

        # Check if medium response would trigger escalation:
        # - Code completeness < 7 (out of 10) → truncated code
        # - Overall score < 80% when premium is >90% → clear quality gap
        m_pct = weighted_pct(m_scores) if m_scores else 0
        p_pct = weighted_pct(p_scores) if p_scores else 0
        code_incomplete = m_scores.get("code_completeness", 10) < 7
        quality_gap = (p_pct - m_pct > 10) and m_pct < 85

        if code_incomplete or quality_gap:
            # Escalate: also pay premium cost, use premium score
            sim_b_cost += pd.get("cost", 0)
            if p_scores:
                sim_b_scores.append(weighted_pct(p_scores))
            escalated_count += 1
        else:
            # Medium was good enough
            if m_scores:
                sim_b_scores.append(weighted_pct(m_scores))

    sim_b_avg = sum(sim_b_scores) / len(sim_b_scores) if sim_b_scores else 0

    sav_a = (1 - sim_a_cost / prem_cost) * 100 if prem_cost > 0 else 0
    sav_b = (1 - sim_b_cost / prem_cost) * 100 if prem_cost > 0 else 0

    L.append(f"  Alles Premium:         {tier_avg.get('premium',0):.0f}% Qualität, ${prem_cost:.5f}")
    L.append(f"  Alles Medium:          {tier_avg.get('medium',0):.0f}% Qualität, ${med_cost:.5f}")
    L.append(f"  Alles Cheap:           {tier_avg.get('cheap',0):.0f}% Qualität, ${cheap_cost:.5f}")
    L.append("")
    L.append(f"  ▸ Strategie A (Klassisch: easy→cheap, hard→premium):")
    L.append(f"    {sim_a_avg:.0f}% Qualität, ${sim_a_cost:.5f} ({sav_a:.0f}% gespart)")
    L.append(f"  ▸ Strategie B (Verify-then-Escalate: medium→check→premium):")
    L.append(f"    {sim_b_avg:.0f}% Qualität, ${sim_b_cost:.5f} ({sav_b:.0f}% gespart, {escalated_count}/{len(test_ids)} eskaliert)")
    L.append("")

    # Use the better strategy for the final score
    best_strat = "B" if sim_b_avg >= sim_a_avg and sav_b >= sav_a else "A"
    sim_avg = sim_b_avg if best_strat == "B" else sim_a_avg
    sim_cost = sim_b_cost if best_strat == "B" else sim_a_cost
    sav = sav_b if best_strat == "B" else sav_a

    ql = tier_avg.get("premium", 0) - sim_avg
    L.append(f"  Beste Strategie: {best_strat}")
    L.append(f"  Qualität:         {bar100(sim_avg)}")
    L.append(f"  Kostenersparnis:  {bar100(max(0, sav))}")
    L.append(f"  Qualitätsverlust: {bar100(max(0, 100 - ql * 5))}  ({ql:.1f}% weniger)")
    overall = sim_avg * 0.5 + min(100, max(0, sav)) * 0.3 + max(0, 100 - ql * 5) * 0.2
    grade = ("A+" if overall >= 90 else "A" if overall >= 80 else "B" if overall >= 70
             else "C" if overall >= 60 else "D" if overall >= 50 else "F")
    L.append(f"  ═════════════════════════════════════════════════")
    L.append(f"  GESAMT P/L:       {bar100(overall)}  → Note: {grade}")
    L.append("")

    # ─── Where Cheap Fails ────────────────────────────────────
    L.append("═══ WO CHEAP VERSAGT (→ Router-Optimierung) ═════════════════════════════════")
    problems = []
    for tid in test_ids:
        cd = kb["tests"][tid]["tiers"].get("cheap", {})
        pd = kb["tests"][tid]["tiers"].get("premium", {})
        if cd.get("scores") and pd.get("scores"):
            cp = weighted_pct(cd["scores"])
            pp = weighted_pct(pd["scores"])
            if pp - cp > 15:
                problems.append((tid, cp, pp, pp - cp, kb["tests"][tid]))

    if problems:
        problems.sort(key=lambda x: -x[3])
        L.append(f"  {'Test-ID':<16} {'Cheap':>6} {'Prem':>6} {'Δ':>5}  Empfehlung")
        L.append(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*5}  {'─'*25}")
        for tid, cp, pp, loss, entry in problems:
            rec = "→ PREMIUM" if loss > 30 else "→ MEDIUM"
            L.append(f"  {tid:<16} {cp:5.0f}% {pp:5.0f}% {loss:4.0f}%  {rec} ({entry['category']}/{entry['difficulty']})")
        L.append(f"\n  💡 {len(problems)} Tests mit >15% Qualitätsverlust → Router-Schwellenwerte anpassen")
    else:
        L.append("  ✅ Cheap liefert überall akzeptable Qualität!")
    L.append("")

    # ─── KB Stats ─────────────────────────────────────────────
    nc = len(kb.get("custom_tests", {}))
    n_resp = sum(1 for t in test_ids for tier in TIERS
                 if kb["tests"][t]["tiers"].get(tier, {}).get("response"))
    n_scored = sum(1 for t in test_ids for tier in TIERS
                   if kb["tests"][t]["tiers"].get(tier, {}).get("scores"))
    L.append("═══ WISSENSDATENBANK ════════════════════════════════════════════════════════")
    L.append(f"  Tests:       {len(test_ids)} ({len(test_ids)-nc} builtin + {nc} custom)")
    L.append(f"  Antworten:   {n_resp}/{len(test_ids)*3}")
    L.append(f"  Bewertungen: {n_scored}/{len(test_ids)*3}")
    L.append(f"  Kosten:      ${total_cost:.4f}")
    L.append(f"  Datei:       {KB_FILE}")
    L.append("")
    L.append("════════════════════════════════════════════════════════════════════════════════")

    return "\n".join(L)


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    p = argparse.ArgumentParser(description="LLM Gateway Benchmark v3.2")
    p.add_argument("--base-url", help="Gateway URL (auto aus .env)")
    p.add_argument("--api-key", help="API Key (auto aus .env)")
    p.add_argument("--category")
    p.add_argument("--difficulty")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--test-id")
    p.add_argument("--refresh", action="store_true", help="Alle Tiers neu holen")
    p.add_argument("--refresh-tier", help="Nur diesen Tier neu")
    p.add_argument("--refresh-test", help="Nur diesen Test neu")
    p.add_argument("--refresh-scores", action="store_true", help="Nur neu bewerten")
    p.add_argument("--report", action="store_true", help="Nur Report aus KB")
    p.add_argument("--add", action="store_true", help="Test hinzufügen")
    p.add_argument("--export", action="store_true", help="Markdown-Export")
    p.add_argument("--clean", action="store_true", help="Vergiftete 53%%-Scores entfernen")
    p.add_argument("--delay", type=float, default=6.0, help="Sekunden zwischen API-Calls (default: 6)")
    p.add_argument("--kb", default="benchmark_kb.json")
    args = p.parse_args()

    global KB_FILE, _adaptive_delay, _min_delay
    KB_FILE = Path(args.kb)
    _adaptive_delay = args.delay
    _min_delay = args.delay
    kb = load_kb(KB_FILE)

    if args.add:
        add_test_interactive(kb); return

    if args.report:
        if not kb["tests"]: print("KB leer!"); return
        print(generate_report(kb)); return

    if args.export:
        print("| Test-ID | Kat. | Diff. | Cheap | Medium | Premium |")
        print("|---------|------|-------|-------|--------|---------|")
        for tid in sorted(kb["tests"]):
            e = kb["tests"][tid]
            row = [tid, e.get("category",""), e.get("difficulty","")]
            for tier in TIERS:
                td = e["tiers"].get(tier, {})
                row.append(f"{weighted_pct(td['scores']):.0f}%" if td.get("scores") else "—")
            print("| " + " | ".join(row) + " |")
        return

    if args.clean:
        fixed_scores = 0
        fixed_responses = 0
        for tid, entry in kb["tests"].items():
            for tier_name, td in entry.get("tiers", {}).items():
                # Remove 53% ghost scores (judge failure defaults)
                scores = td.get("scores")
                if scores:
                    pct = weighted_pct(scores)
                    if abs(pct - 53.0) < 1.5:
                        print(f"  🗑️  {tid}/{tier_name}: {pct:.0f}% → Score entfernt (Judge-Failure)")
                        del td["scores"]
                        td.pop("comment", None)
                        fixed_scores += 1
                # Remove error responses so they get re-fetched
                if td.get("error"):
                    print(f"  🗑️  {tid}/{tier_name}: Error → Response entfernt (wird neu geholt)")
                    entry["tiers"].pop(tier_name)
                    fixed_responses += 1
        save_kb(kb, KB_FILE)
        print(f"\n✅ {fixed_scores} vergiftete Scores + {fixed_responses} Error-Responses entfernt")
        print(f"   → Nächster Run holt fehlende Antworten + bewertet neu")
        return

    base_url, api_key = auto_detect_config()
    base_url = args.base_url or base_url
    api_key = args.api_key or api_key

    if not api_key:
        print("❌ Kein API-Key! .env mit GATEWAY_SECRET=... oder --api-key"); return

    all_tests = dict(BUILTIN_TESTS)
    all_tests.update(kb.get("custom_tests", {}))

    tests = all_tests
    if args.category: tests = {k: v for k, v in tests.items() if v["category"] == args.category}
    if args.difficulty: tests = {k: v for k, v in tests.items() if v["difficulty"] == args.difficulty}
    if args.test_id: tests = {k: v for k, v in tests.items() if k == args.test_id}
    if args.quick:
        tests = {k: v for k, v in tests.items()
                 if k in {"know-easy-1","know-hard-3","math-hard-2","code-easy-1","code-hard-2",
                          "reason-hard-2","vision-easy-1","vision-hard-1"}}

    if not tests: print("Keine Tests!"); return

    refresh_tiers = None
    if args.refresh: refresh_tiers = TIERS
    elif args.refresh_tier: refresh_tiers = [args.refresh_tier]
    elif args.refresh_test: refresh_tiers = TIERS

    # Count work
    to_fetch = 0
    for tid in tests:
        for tier in TIERS:
            cached = kb.get("tests", {}).get(tid, {}).get("tiers", {}).get(tier, {})
            if not cached or cached.get("error") or (refresh_tiers and tier in refresh_tiers):
                if not args.refresh_test or args.refresh_test == tid:
                    to_fetch += 1

    print(f"\n🧪 LLM Gateway Benchmark v3.2")
    print(f"   URL:      {base_url}")
    print(f"   Key:      {api_key[:8]}...{api_key[-4:]}")
    print(f"   Tests:    {len(tests)}")
    print(f"   KB:       {KB_FILE} ({'existiert' if KB_FILE.exists() else 'NEU'})")

    # Load vision test images
    load_vision_images()
    # Filter out vision tests if images not loaded
    vision_tests = {k for k, v in tests.items() if v.get("image_key")}
    if vision_tests and not VISION_IMAGES:
        print(f"   ⚠️  {len(vision_tests)} Vision-Tests übersprungen (keine Bilder)")
        tests = {k: v for k, v in tests.items() if not v.get("image_key")}

    print(f"   Zu holen: {to_fetch} Tier-Antworten (Rest aus KB-Cache)")
    print(f"   Header:   X-No-Cache: true (Cache-Bypass)")
    print(f"   Delay:    {_adaptive_delay:.0f}s zwischen Calls (adaptive, --delay N)")

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{base_url}/health", timeout=5)
            print(f"   Status:   ✅")
        except Exception as e:
            print(f"   Status:   ❌ {e}\n   Abbruch."); return

    print("═" * 60)

    async with httpx.AsyncClient() as client:
        for i, (tid, tdef) in enumerate(tests.items(), 1):
            print(f"\n[{i}/{len(tests)}] {tid} ({tdef['category']}/{tdef['difficulty']})")

            rt = refresh_tiers if (not args.refresh_test or args.refresh_test == tid) else None
            await fetch_test_tiers(client, base_url, api_key, tid, tdef, kb, rt)
            await score_test(client, base_url, api_key, tid, tdef, kb, args.refresh_scores)

            save_kb(kb, KB_FILE)

            if i < len(tests):
                await asyncio.sleep(max(_adaptive_delay, _min_delay * 1.5))  # Extra gap between tests

    print("\n")
    report = generate_report(kb)
    print(report)

    rdir = Path("benchmark_results")
    rdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    (rdir / f"report_{ts}.txt").write_text(report)
    print(f"\n📁 Report:  benchmark_results/report_{ts}.txt")
    print(f"📁 KB:      {KB_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
