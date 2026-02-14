"""
LLM Gateway - Response Validator
=================================
Post-generation quality check for cheap/medium tier responses.
Detects incomplete or truncated code without using an LLM call (= free).

Checks:
  1. Truncation: finish_reason == "length" → definitely truncated
  2. Bracket balance: unclosed {, (, [, ``` → likely truncated
  3. Syntax check: ast.parse for Python, bracket-only for others
  4. Common truncation patterns: ends mid-line, trailing ellipsis

Language support:
  - Python: full (ast.parse + brackets + fences)
  - C/C++/Java/JS/HTML: brackets + fences (no syntax parse)
  - All others: fences + truncation patterns

If any check fails → escalate to premium for re-generation.
Cost: $0 (all checks are local, no API calls).
"""

import ast
import re
import logging
from dataclasses import dataclass

log = logging.getLogger("gateway.validator")


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool = True
    should_escalate: bool = False
    reason: str = ""
    checks_run: int = 0
    checks_failed: int = 0
    details: list[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []


# ─── Code Extraction ─────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r'```(\w*)\n(.*?)(?:```|$)', re.DOTALL)

def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (language, code) pairs from markdown code blocks.
    
    Returns all code blocks found, including ones where the closing ``` 
    is missing (which itself is a truncation signal).
    """
    blocks = []
    for m in _CODE_BLOCK_RE.finditer(text):
        lang = m.group(1).lower() or "unknown"
        code = m.group(2)
        blocks.append((lang, code))
    return blocks


def _has_code_content(text: str) -> bool:
    """Heuristic: does this response contain code?"""
    if _CODE_BLOCK_RE.search(text):
        return True
    # Check for code-like patterns even without fences
    code_signals = [
        "def ", "class ", "import ", "function ", "const ", "let ", "var ",
        "for ", "while ", "if __name__", "return ", "async def",
        "SELECT ", "CREATE TABLE", "INSERT INTO",
        "#include", "public class", "package ", "void main",
    ]
    return any(sig in text for sig in code_signals)


# ─── Individual Checks ───────────────────────────────────────────

def _check_finish_reason(finish_reason: str) -> tuple[bool, str]:
    """Check if response was cut off by token limit."""
    if finish_reason == "length":
        return False, "finish_reason='length' — response hit token limit (truncated)"
    return True, ""


def _check_unclosed_code_fences(text: str) -> tuple[bool, str]:
    """Check if code blocks are properly closed with ```."""
    # Count opening and closing fences
    fence_pattern = re.compile(r'^```', re.MULTILINE)
    fences = fence_pattern.findall(text)
    if len(fences) % 2 != 0:
        return False, f"Odd number of code fences ({len(fences)}) — unclosed code block"
    return True, ""


def _check_bracket_balance(code: str, lang: str = "unknown") -> tuple[bool, str]:
    """Check if brackets/parens/braces are balanced in code.
    
    Works for ALL languages that use {}, [], ().
    """
    in_string = None
    depth = {"(": 0, "[": 0, "{": 0}
    close_map = {")": "(", "]": "[", "}": "{"}
    
    # Comment markers vary by language
    line_comment = "#" if lang in ("python", "py", "python3", "ruby", "bash", "sh") else "//"
    
    i = 0
    while i < len(code):
        c = code[i]
        
        # Handle escape sequences inside strings
        if in_string and c == '\\':
            i += 2  # Skip escaped character
            continue
        
        # Handle strings (single and double quotes, universal)
        if c in ('"', "'"):
            # Triple quotes (Python/JS)
            if code[i:i+3] in ('"""', "'''"):
                if in_string == code[i:i+3]:
                    in_string = None
                    i += 3
                    continue
                elif in_string is None:
                    in_string = code[i:i+3]
                    i += 3
                    continue
            elif in_string is None:
                in_string = c
            elif in_string == c:
                in_string = None
            i += 1
            continue
        
        # Handle template literals (JS backticks)
        if c == '`' and in_string in (None, '`'):
            if in_string == '`':
                in_string = None
            else:
                in_string = '`'
            i += 1
            continue
        
        # Handle line comments
        if in_string is None:
            if line_comment == "#" and c == '#':
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue
            elif line_comment == "//" and code[i:i+2] == "//":
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue
            # Block comments (C/C++/Java/JS)
            elif code[i:i+2] == "/*":
                end = code.find("*/", i + 2)
                if end == -1:
                    i = len(code)  # Unclosed block comment
                else:
                    i = end + 2
                continue
        
        if in_string is None:
            if c in depth:
                depth[c] += 1
            elif c in close_map:
                depth[close_map[c]] -= 1
        
        i += 1
    
    unbalanced = {k: v for k, v in depth.items() if v > 0}
    if unbalanced:
        total_imbalance = sum(unbalanced.values())
        code_lines = code.count('\n') + 1

        # Tolerance: for long code (100+ lines), a small bracket imbalance
        # (1-2) is likely a false positive from f-strings, regex, or
        # template literals — not actual truncation.
        # True truncation usually leaves 3+ unclosed brackets.
        if code_lines >= 100 and total_imbalance <= 2:
            log.info(f"Bracket imbalance {unbalanced} in {code_lines} lines "
                     f"— within tolerance (likely f-string/regex)")
            return True, ""

        details = ", ".join(f"'{k}' +{v}" for k, v in unbalanced.items())
        return False, f"Unbalanced brackets: {details} — code likely truncated"
    return True, ""


def _check_python_syntax(code: str) -> tuple[bool, str]:
    """Try to parse Python code. Syntax errors may indicate truncation."""
    try:
        # Dedent first: Flash sometimes wraps entire code blocks with
        # uniform indentation (e.g. "    import csv\n    class Foo:...")
        # which causes "unexpected indent at line 1" but isn't truncation.
        import textwrap
        ast.parse(textwrap.dedent(code))
        return True, ""
    except SyntaxError as e:
        # Common truncation-induced syntax errors
        if e.msg and any(kw in e.msg.lower() for kw in [
            "unexpected eof", "expected an indented block",
            "unterminated string", "unmatched",
        ]):
            return False, f"Python syntax error (likely truncation): {e.msg} at line {e.lineno}"
        # Other syntax errors might be intentional (pseudo-code, snippets)
        return True, ""


def _check_html_balance(code: str) -> tuple[bool, str]:
    """Check if HTML tags are roughly balanced (major tags only)."""
    # Only check major structural tags — not self-closing ones
    major_tags = ["div", "section", "main", "header", "footer", "nav",
                  "table", "form", "ul", "ol", "body", "html", "head"]
    
    for tag in major_tags:
        opens = len(re.findall(f'<{tag}[\\s>]', code, re.IGNORECASE))
        closes = len(re.findall(f'</{tag}>', code, re.IGNORECASE))
        if opens > closes:
            return False, f"Unclosed <{tag}> tags ({opens} opens, {closes} closes)"
    return True, ""


def _check_truncation_patterns(text: str) -> tuple[bool, str]:
    """Check for common truncation patterns at end of response."""
    stripped = text.rstrip()
    if not stripped:
        return True, ""
    
    # Ends with ... or … (explicitly truncated)
    if stripped.endswith(("...", "…", "# ...", "// ...")):
        return False, "Response ends with ellipsis — explicitly truncated"
    
    # Ends mid-word or mid-line (very short last line after long response)
    lines = stripped.split('\n')
    if len(lines) > 20:
        last_line = lines[-1].strip()
        # Very short last line that doesn't look like a natural ending
        # EXCLUDE: code fences (```), markdown separators (---), common endings
        natural_short_endings = ('.', ')', ']', '}', ';', ':', '```', '`',
                                  '"""', "'''", '*/', '-->', '?>', '!')
        natural_short_lines = {'```', '---', '***', '___', '|', '> ', ''}
        if (0 < len(last_line) < 5 
            and not any(last_line.endswith(e) for e in natural_short_endings)
            and last_line not in natural_short_lines):
            return False, f"Suspiciously short last line: '{last_line}' — possible truncation"
    
    # ── Mid-sentence truncation: long response ends without sentence punctuation ──
    # Gemini Flash sometimes returns finish_reason="stop" but stops mid-sentence.
    # Detect: response > 500 chars, last char is not sentence-ending punctuation.
    if len(stripped) > 500:
        # Get last non-markdown, non-whitespace character
        _tail = stripped.rstrip('*_ \t\n')  # Strip trailing bold/italic/whitespace
        if _tail:
            # Check if last line is a markdown structural element (separator, fence, etc.)
            _last_line_raw = _tail.split('\n')[-1].strip()
            _structural_lines = {'---', '***', '___', '```', '|', '> '}
            if _last_line_raw not in _structural_lines:
                _last_char = _tail[-1]
                _sentence_endings = {'.', '!', '?', ':', ')', ']', '}', '"', "'",
                                      '`', '>', ';', '|', '–', '—', '-'}
                if _last_char not in _sentence_endings:
                    _is_list_heading = (_last_line_raw.startswith(('#', '-', '*', '1.', '2.', '3.'))
                                        and len(_last_line_raw) < 80
                                        and _last_line_raw.rstrip().endswith(':'))
                    if not _is_list_heading:
                        return False, (f"Mid-sentence truncation: response ends with "
                                       f"'{_tail[-30:]}' — no sentence-ending punctuation")
    
    return True, ""


# ─── Main Validator ──────────────────────────────────────────────

# Languages where we can do deeper syntax analysis
_PYTHON_LANGS = {"python", "py", "python3"}
_BRACKET_LANGS = {"python", "py", "python3", "javascript", "js", "typescript", "ts",
                  "java", "c", "cpp", "c++", "csharp", "cs", "go", "rust", "swift",
                  "kotlin", "php", "ruby", "bash", "sh", "unknown"}
_HTML_LANGS = {"html", "htm", "xml", "svg", "jsx", "tsx", "vue"}


def validate_response(
    response_text: str,
    finish_reason: str = "stop",
    tier: str = "medium",
    response_type: str = "general",
) -> ValidationResult:
    """
    Validate a response from cheap/medium tier.
    
    Returns ValidationResult with should_escalate=True if the response
    appears incomplete and should be re-generated with premium.
    
    This function is FREE — no API calls, pure local analysis.
    """
    result = ValidationResult()
    
    # Only validate cheap/medium responses (premium is already the best we have)
    if tier == "premium":
        return result
    
    # ── Check 1: finish_reason ──
    result.checks_run += 1
    ok, reason = _check_finish_reason(finish_reason)
    if not ok:
        result.checks_failed += 1
        result.details.append(reason)
        result.should_escalate = True
        result.is_valid = False
        result.reason = reason
        log.warning(f"Validation FAIL: {reason}")
        # finish_reason=length is definitive — no need for other checks
        return result
    
    # Only run code-specific checks if response contains code
    has_code = _has_code_content(response_text)
    
    if has_code:
        # ── Check 2: Code fence balance ──
        result.checks_run += 1
        ok, reason = _check_unclosed_code_fences(response_text)
        if not ok:
            result.checks_failed += 1
            result.details.append(reason)
        
        # ── Check 3+4: Per-language code block checks ──
        code_blocks = _extract_code_blocks(response_text)
        for lang, code in code_blocks:
            # Check 3: Bracket balance (works for all C-style and Python)
            if lang in _BRACKET_LANGS:
                result.checks_run += 1
                ok, reason = _check_bracket_balance(code, lang)
                if not ok:
                    result.checks_failed += 1
                    result.details.append(f"[{lang}] {reason}")
            
            # Check 4a: Python syntax check
            if lang in _PYTHON_LANGS:
                result.checks_run += 1
                ok, reason = _check_python_syntax(code)
                if not ok:
                    result.checks_failed += 1
                    result.details.append(f"[{lang}] {reason}")
            
            # Check 4b: HTML tag balance
            if lang in _HTML_LANGS:
                result.checks_run += 1
                ok, reason = _check_html_balance(code)
                if not ok:
                    result.checks_failed += 1
                    result.details.append(f"[{lang}] {reason}")
    
    # ── Check 5: Truncation patterns ──
    result.checks_run += 1
    ok, reason = _check_truncation_patterns(response_text)
    if not ok:
        result.checks_failed += 1
        result.details.append(reason)
    
    # ── Decision ──
    if result.checks_failed > 0:
        result.is_valid = False
        result.should_escalate = True
        result.reason = f"{result.checks_failed}/{result.checks_run} checks failed: {'; '.join(result.details)}"
        log.warning(f"Validation FAIL ({tier}): {result.reason}")
    else:
        log.debug(f"Validation OK ({tier}): {result.checks_run} checks passed")
    
    return result
