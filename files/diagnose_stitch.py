#!/usr/bin/env python3
"""
Stitch/Repair Diagnostic Tool
==============================
Investigates WHY code stitching fails by replaying each stage
and showing exactly what happens.

Usage:
    python diagnose_stitch.py              # Full diagnosis
    python diagnose_stitch.py --test 1     # Only stitch-long-1
    python diagnose_stitch.py --test 2     # Only stitch-long-2
"""

import ast
import re
import os
import sys
import json
import time
import textwrap
import httpx
from pathlib import Path
from dotenv import load_dotenv

# ─── Config ────────────────────────────────────────────────────
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

BASE_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.getenv("GATEWAY_SECRET", "")

TESTS = {
    1: {
        "name": "stitch-long-1 (Inventory System)",
        "prompt": (
            "Schreibe ein vollständiges Python-Programm: Lagerverwaltungssystem mit "
            "Klassen Product, Warehouse, Order. CRUD-Operationen, CSV-Import/Export, "
            "Bestandswarnungen, argparse CLI. Mindestens 200 Zeilen. Nur Code, keine Erklärung."
        ),
    },
    2: {
        "name": "stitch-long-2 (REST API)",
        "prompt": (
            "Schreibe einen vollständigen Python REST API Server (mit http.server oder Flask): "
            "CRUD-Endpoints für eine Aufgabenverwaltung, JSON-Validierung, Logging, "
            "Fehlerbehandlung, und mindestens 5 Unit-Tests. Mindestens 250 Zeilen. Nur Code."
        ),
    },
}

TIMEOUT = 300


def call_gateway(prompt, model="auto", max_tokens=8192):
    """Call gateway and return full response with metadata."""
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-No-Cache": "true",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
        return r.json()


def call_gateway_cheap_only(prompt, max_tokens=4096):
    """Call gateway with explicit cheap model (no cascade)."""
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-No-Cache": "true",
            },
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
        return r.json()


def call_gateway_continuation(original_prompt, assistant_content, continuation_prompt, max_tokens=4096):
    """Call gateway with conversation history for continuation."""
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-No-Cache": "true",
            },
            json={
                "model": "cheap",
                "messages": [
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": continuation_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
        return r.json()


# ─── Validator Logic (copied from response_validator.py) ───────
_CODE_BLOCK_RE = re.compile(r'```(\w*)\n(.*?)(?:```|$)', re.DOTALL)


def extract_code_blocks(text):
    blocks = []
    for m in _CODE_BLOCK_RE.finditer(text):
        lang = m.group(1).lower() or "unknown"
        code = m.group(2)
        blocks.append((lang, code))
    return blocks


def check_fence_balance(text):
    fence_pattern = re.compile(r'^```', re.MULTILINE)
    fences = fence_pattern.findall(text)
    if len(fences) % 2 != 0:
        return False, f"Odd number of code fences ({len(fences)})"
    return True, ""


def check_bracket_balance(code):
    in_string = None
    depth = {"(": 0, "[": 0, "{": 0}
    close_map = {")": "(", "]": "[", "}": "{"}
    i = 0
    while i < len(code):
        c = code[i]
        if c in ('"', "'"):
            if code[i:i+3] in ('"""', "'''"):
                if in_string == code[i:i+3]:
                    in_string = None
                    i += 3; continue
                elif in_string is None:
                    in_string = code[i:i+3]
                    i += 3; continue
            elif in_string is None:
                in_string = c
            elif in_string == c:
                in_string = None
            i += 1; continue
        if c == '#' and in_string is None:
            while i < len(code) and code[i] != '\n':
                i += 1
            continue
        if in_string is None:
            if c in depth: depth[c] += 1
            elif c in close_map: depth[close_map[c]] -= 1
        i += 1
    unbalanced = {k: v for k, v in depth.items() if v > 0}
    if unbalanced:
        return False, f"Unbalanced: {unbalanced}"
    return True, ""


def check_python_syntax(code):
    try:
        ast.parse(textwrap.dedent(code))
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"


def check_truncation_patterns(text):
    stripped = text.rstrip()
    if stripped.endswith(("...", "…", "# ...", "// ...")):
        return False, "Ends with ellipsis"
    lines = stripped.split('\n')
    if len(lines) > 20:
        last = lines[-1].strip()
        natural_endings = ('.', ')', ']', '}', ';', ':', '```', '`', '"""', "'''")
        if (0 < len(last) < 5
            and not any(last.endswith(e) for e in natural_endings)
            and last not in ('```',)):
            return False, f"Short last line: '{last}'"
    return True, ""


def full_validate(text, finish_reason="stop"):
    """Run all validator checks, return (is_valid, details)."""
    details = []

    if finish_reason == "length":
        details.append("❌ finish_reason=length (hard truncation)")
        return False, details

    # Fence balance
    ok, msg = check_fence_balance(text)
    if not ok:
        details.append(f"❌ {msg}")

    # Code blocks
    blocks = extract_code_blocks(text)
    for lang, code in blocks:
        ok, msg = check_bracket_balance(code)
        if not ok:
            details.append(f"❌ [{lang}] {msg}")

        if lang in ("python", "py", "python3"):
            ok, msg = check_python_syntax(code)
            if not ok:
                details.append(f"❌ [{lang}] Syntax: {msg}")

    # Truncation patterns
    ok, msg = check_truncation_patterns(text)
    if not ok:
        details.append(f"❌ {msg}")

    if not details:
        details.append("✅ All checks passed")
        return True, details
    return False, details


# ─── Trim Logic (matches main.py) ─────────────────────────────
def extract_code_from_response(content):
    pattern = r'(```(?:python|py|python3)?\n)(.*?)(\n```)?$'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        start = match.start()
        prefix = content[:start] + match.group(1)
        code = match.group(2)
        suffix = match.group(3) or ""
        return prefix, code, suffix
    if any(kw in content for kw in ["def ", "class ", "import "]):
        return "", content, ""
    return content, "", ""


def trim_broken_tail(content):
    """Find last valid Python parse point."""
    prefix, code, suffix = extract_code_from_response(content)
    if not code:
        return content, 0, "no code found"

    lines = code.split("\n")
    if len(lines) < 5:
        return content, 0, "too short to trim"

    try:
        ast.parse(textwrap.dedent(code))
        return content, 0, "already parses OK"
    except SyntaxError as e:
        original_error = f"{e.msg} at line {e.lineno}"

    for trim in range(1, min(30, len(lines) - 5)):
        candidate = "\n".join(lines[:-trim])
        try:
            ast.parse(textwrap.dedent(candidate))
            trimmed_code = candidate
            result = prefix + trimmed_code + "\n" + suffix if suffix else prefix + trimmed_code + "\n"
            return result, trim, f"trimmed {trim} lines → parses OK"
        except SyntaxError:
            continue

    return content, 0, f"no valid parse point found (original: {original_error})"


# ─── Diff Cleaning (matches main.py _clean_code_response) ─────
def clean_diff_markers(content):
    """Strip +/- diff prefixes from code blocks."""
    if not content or "```" not in content:
        return content, False
    modified = False
    result_parts = []
    last_end = 0
    for m in re.finditer(r'(```\w*\n)(.*?)(\n```)', content, re.DOTALL):
        result_parts.append(content[last_end:m.start()])
        fence_open = m.group(1)
        code = m.group(2)
        fence_close = m.group(3)
        code_lines = code.split("\n")
        diff_lines = sum(1 for l in code_lines if l.startswith("+") or l.startswith("-"))
        total_nonblank = sum(1 for l in code_lines if l.strip())
        if total_nonblank > 5 and diff_lines / max(total_nonblank, 1) > 0.5:
            cleaned_lines = []
            for line in code_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    cleaned_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    continue
                else:
                    cleaned_lines.append(line)
            code = "\n".join(cleaned_lines)
            modified = True
        result_parts.append(fence_open + code + fence_close)
        last_end = m.end()
    if modified:
        result_parts.append(content[last_end:])
        return "".join(result_parts), True
    # Also check unclosed blocks
    unclosed = re.search(r'(```\w*\n)(.*?)$', content[last_end:], re.DOTALL)
    if unclosed:
        code = unclosed.group(2)
        code_lines = code.split("\n")
        diff_lines = sum(1 for l in code_lines if l.startswith("+") or l.startswith("-"))
        total_nonblank = sum(1 for l in code_lines if l.strip())
        if total_nonblank > 5 and diff_lines / max(total_nonblank, 1) > 0.5:
            cleaned_lines = []
            for line in code_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    cleaned_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    continue
                else:
                    cleaned_lines.append(line)
            result_parts.append(content[last_end:unclosed.start() + last_end])
            result_parts.append(unclosed.group(1) + "\n".join(cleaned_lines) + "\n```")
            return "".join(result_parts), True
    return content, False


# ─── Main Diagnosis ───────────────────────────────────────────
def diagnose(test_id):
    test = TESTS[test_id]
    print(f"\n{'='*70}")
    print(f"  DIAGNOSING: {test['name']}")
    print(f"{'='*70}")

    # ── Stage 1: Get raw cheap output ──
    print(f"\n── Stage 1: Raw cheap model output ──")
    t0 = time.time()
    try:
        resp = call_gateway_cheap_only(test["prompt"])
    except Exception as e:
        print(f"  ❌ Cheap call failed: {e}")
        return
    elapsed = time.time() - t0

    content = resp["choices"][0]["message"]["content"]
    finish = resp["choices"][0].get("finish_reason", "stop")
    tier = resp.get("gateway_metadata", {}).get("tier_used", "?")
    lines = content.strip().split("\n")

    print(f"  Response: {len(content)} chars, {len(lines)} lines")
    print(f"  finish_reason: {finish}")
    print(f"  Tier: {tier}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Last 5 lines:")
    for l in lines[-5:]:
        print(f"    |{l[:80]}{'...' if len(l) > 80 else ''}|")

    # ── Stage 1b: Clean diff markers ──
    cleaned_content, was_cleaned = clean_diff_markers(content)
    if was_cleaned:
        print(f"\n── Stage 1b: Diff-marker cleaning ──")
        cleaned_lines = cleaned_content.strip().split("\n")
        print(f"  ✅ Stripped diff markers: {len(content)} → {len(cleaned_content)} chars")
        print(f"  Lines: {len(lines)} → {len(cleaned_lines)}")
        content = cleaned_content  # Use cleaned version for remaining stages
        lines = cleaned_lines

    # ── Stage 2: Validate ──
    print(f"\n── Stage 2: Validator checks ──")
    is_valid, details = full_validate(content, finish)
    for d in details:
        print(f"  {d}")
    print(f"  → {'VALID ✅' if is_valid else 'INVALID → would trigger stitch/repair'}")

    if is_valid:
        print(f"\n  ✅ Code passes validation — no stitch needed!")
        # Still check if it has the required patterns
        check_completeness(content, test_id)
        return

    # ── Stage 3: Smart Trim ──
    print(f"\n── Stage 3: Smart trim (remove broken tail) ──")
    trimmed, removed, reason = trim_broken_tail(content)
    print(f"  Result: {reason}")
    if removed > 0:
        trimmed_lines = trimmed.strip().split("\n")
        print(f"  Lines: {len(lines)} → {len(trimmed_lines)} (-{removed})")
        print(f"  New last 3 lines:")
        for l in trimmed_lines[-3:]:
            print(f"    |{l[:80]}|")

        # Re-validate trimmed version
        is_valid2, details2 = full_validate(trimmed, "stop")
        print(f"  Trimmed validation: {'✅' if is_valid2 else '❌'}")
        for d in details2:
            print(f"    {d}")
    else:
        trimmed = content  # Use original for continuation
        print(f"  ⚠ No trimming possible — stitch will use original broken code")

    # ── Stage 4: Stitch (continuation) ──
    print(f"\n── Stage 4: Stitch (continuation call) ──")
    continuation_prompt = (
        "Dein Code oben ist unvollständig — es fehlen noch Teile "
        "(fehlende Klassen, Funktionen, if __name__ Block, etc.). "
        "Schreibe NUR den fehlenden Rest des Codes. "
        "Wiederhole NICHT was du bereits geschrieben hast. "
        "Falls der Code doch bereits vollständig ist, schreibe nur: COMPLETE"
    )

    t0 = time.time()
    try:
        stitch_resp = call_gateway_continuation(
            test["prompt"], trimmed, continuation_prompt
        )
    except Exception as e:
        print(f"  ❌ Stitch call failed: {e}")
        try_repair(test, content)
        return
    elapsed = time.time() - t0

    stitch_content = stitch_resp["choices"][0]["message"]["content"]
    stitch_finish = stitch_resp["choices"][0].get("finish_reason", "stop")
    stitch_lines = stitch_content.strip().split("\n")

    print(f"  Continuation: {len(stitch_content)} chars, {len(stitch_lines)} lines")
    print(f"  finish_reason: {stitch_finish}")
    print(f"  Time: {elapsed:.1f}s")

    if "COMPLETE" in stitch_content and len(stitch_content) < 50:
        print(f"  ℹ Model says COMPLETE — thinks code is done")
    else:
        print(f"  First 3 lines:")
        for l in stitch_lines[:3]:
            print(f"    |{l[:80]}|")
        print(f"  Last 3 lines:")
        for l in stitch_lines[-3:]:
            print(f"    |{l[:80]}|")

    # Stitch together
    combined = trimmed.rstrip() + "\n" + stitch_content.lstrip()
    combined_lines = combined.strip().split("\n")
    print(f"\n  Combined: {len(combined)} chars, {len(combined_lines)} lines")

    # Validate combined
    is_valid3, details3 = full_validate(combined, "stop")
    print(f"  Combined validation: {'✅' if is_valid3 else '❌'}")
    for d in details3:
        print(f"    {d}")

    if is_valid3:
        print(f"\n  ✅ STITCH WORKS! Combined code passes validation.")
        check_completeness(combined, test_id)
        return

    # ── Stage 5: Repair ──
    try_repair(test, combined)


def try_repair(test, broken_content):
    """Stage 5: Ask model to fix its own broken code."""
    print(f"\n── Stage 5: Repair (self-correction) ──")

    _, repair_details = full_validate(broken_content, "stop")
    error_summary = "; ".join(d for d in repair_details if d.startswith("❌"))

    repair_prompt = (
        f"Dein Code hat Fehler die automatisch erkannt wurden:\n"
        f"→ {error_summary}\n\n"
        f"Bitte schreibe den KOMPLETTEN korrigierten Code. "
        f"Achte besonders auf: alle Klammern schließen, "
        f"alle Code-Blöcke mit ``` schließen, "
        f"vollständige Klassen/Funktionen, if __name__ Block. "
        f"Schreibe den gesamten Code nochmal korrekt."
    )

    print(f"  Sending errors: {error_summary[:100]}...")

    t0 = time.time()
    try:
        repair_resp = call_gateway_continuation(
            test["prompt"], broken_content, repair_prompt
        )
    except Exception as e:
        print(f"  ❌ Repair call failed: {e}")
        return
    elapsed = time.time() - t0

    repair_content = repair_resp["choices"][0]["message"]["content"]
    repair_lines = repair_content.strip().split("\n")

    print(f"  Repair output: {len(repair_content)} chars, {len(repair_lines)} lines")
    print(f"  Time: {elapsed:.1f}s")

    # Validate repair
    is_valid, details = full_validate(repair_content, "stop")
    print(f"  Repair validation: {'✅' if is_valid else '❌'}")
    for d in details:
        print(f"    {d}")

    if is_valid:
        print(f"\n  ✅ REPAIR WORKS! Self-corrected code passes validation.")
        check_completeness(repair_content, test["name"])
    else:
        print(f"\n  ❌ REPAIR ALSO FAILS — premium escalation is the only option")
        print(f"\n  🔎 ROOT CAUSE ANALYSIS:")
        print(f"  The cheap model (Gemini Flash) cannot produce this much")
        print(f"  structurally valid code. Options:")
        print(f"  1. Lower the bar: reduce min_lines requirement")
        print(f"  2. Simpler prompts: fewer classes/features per test")
        print(f"  3. Accept: some code tasks need premium")
        print(f"  4. max_tokens: try higher limit for cheap model")


def check_completeness(content, test_id):
    """Check if the code has all required patterns."""
    patterns_1 = {
        "class Product": "class Product" in content,
        "class Warehouse": "class Warehouse" in content,
        "class Order": "class Order" in content,
        "argparse/CLI": "argparse" in content or "ArgumentParser" in content,
        "csv": "csv" in content,
    }
    patterns_2 = {
        "Flask/http.server": "Flask" in content or "http.server" in content or "HTTPServer" in content,
        "CRUD routes": any(kw in content for kw in ["GET", "POST", "PUT", "DELETE",
                                                      "@app.route", "do_GET", "do_POST"]),
        "unittest/test": "unittest" in content or "def test_" in content or "assert" in content,
        "logging": "logging" in content or "log" in content,
    }

    patterns = patterns_1 if (isinstance(test_id, int) and test_id == 1) else patterns_2
    print(f"\n  Pattern check:")
    for name, found in patterns.items():
        print(f"    {'✅' if found else '❌'} {name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=[1, 2], help="Test to diagnose")
    args = parser.parse_args()

    print(f"Gateway: {BASE_URL}")

    if args.test:
        diagnose(args.test)
    else:
        diagnose(1)
        diagnose(2)

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")
