#!/usr/bin/env python3
"""
LLM Gateway v1.5 — Interactive Test Client
───────────────────────────────────────────
Chat with your gateway from the terminal.
Shows model used, token counts, cost, and tool calls.

Usage:
    python test_interactive.py                  # localhost:8000
    python test_interactive.py --url http://my-vps:8000
    python test_interactive.py --key my-secret  # if auth enabled
"""

import argparse
import httpx
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Colors ──────────────────────────────────────────────────────────────────
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"


def print_banner():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════╗
║   LLM Gateway v1.5 — Interactive Test Client      ║
║   Tool Calling · News · i18n · Cascade             ║
╚══════════════════════════════════════════════════╝{NC}
""")


def check_health(base_url: str, headers: dict) -> bool:
    """Check gateway health and print version info."""
    try:
        r = httpx.get(f"{base_url}/health", headers=headers, timeout=5)
        data = r.json()
        version = data.get("version", "?")
        status = data.get("status", "?")
        print(f"  {GREEN}✓ Connected{NC} — v{version} ({status})")
        
        # Show model config
        models = data.get("models", data.get("config", {}))
        if isinstance(models, dict):
            for tier in ["cheap", "medium", "premium"]:
                model = models.get(f"{tier}_model", models.get(tier, ""))
                if model:
                    print(f"    {DIM}{tier:>8}: {model}{NC}")
        print()
        return True
    except Exception as e:
        print(f"  {RED}✗ Cannot reach {base_url}: {e}{NC}")
        return False


def send_message(base_url: str, headers: dict, message: str, conversation: list) -> dict:
    """Send a message and return the response."""
    conversation.append({"role": "user", "content": message})
    
    payload = {
        "messages": conversation,
        "stream": False,
    }
    
    start = time.time()
    try:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start
        
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text}", "elapsed": elapsed}
        
        return {**r.json(), "elapsed": elapsed}
    except httpx.TimeoutException:
        return {"error": "Request timed out (60s)", "elapsed": time.time() - start}
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}


def format_response(data: dict, conversation: list) -> str:
    """Format the gateway response for display."""
    if "error" in data:
        return f"{RED}Error: {data['error']}{NC}"
    
    # Extract response text
    choices = data.get("choices", [])
    if not choices:
        return f"{RED}No response choices{NC}"
    
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    
    # Add to conversation history
    conversation.append({"role": "assistant", "content": content})
    
    # Extract metadata
    usage = data.get("usage", {})
    meta = data.get("metadata", data.get("gateway_metadata", {}))
    
    model = meta.get("model_used", data.get("model", "?"))
    tier = meta.get("tier", "?")
    input_tok = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tok = usage.get("completion_tokens", usage.get("output_tokens", 0))
    cost = meta.get("cost_usd", meta.get("estimated_cost", 0))
    elapsed = data.get("elapsed", 0)
    cached = meta.get("cache_hit", False)
    tools_used = meta.get("tool_calls_executed", meta.get("tools_used", []))
    
    # Format output
    lines = []
    lines.append(f"\n{GREEN}{content}{NC}")
    lines.append("")
    
    # Metadata line
    meta_parts = []
    if tier and tier != "?":
        meta_parts.append(f"{tier}")
    meta_parts.append(f"{model}")
    meta_parts.append(f"{input_tok}+{output_tok} tok")
    if cost:
        if isinstance(cost, (int, float)):
            meta_parts.append(f"${cost:.6f}")
        else:
            meta_parts.append(f"${cost}")
    meta_parts.append(f"{elapsed*1000:.0f}ms")
    if cached:
        meta_parts.append(f"{YELLOW}CACHED{NC}")
    
    lines.append(f"{DIM}{' | '.join(meta_parts)}{NC}")
    
    # Tool calls
    if tools_used:
        if isinstance(tools_used, list):
            tool_str = ", ".join(str(t) for t in tools_used)
        else:
            tool_str = str(tools_used)
        lines.append(f"{CYAN}  🔧 Tools: {tool_str}{NC}")
    
    # Multimodal info
    if meta.get("multimodal"):
        mtypes = meta.get("media_types", [])
        vstrat = meta.get("vision_strategy", "")
        vskip = meta.get("vision_image_skipped", False)
        parts = []
        if mtypes:
            parts.append(f"media:{','.join(mtypes)}")
        if vstrat:
            parts.append(f"strategy:{vstrat}")
        if vskip:
            parts.append("image_skipped ✓")
        lines.append(f"{YELLOW}  📎 {' | '.join(parts)}{NC}")
    
    return "\n".join(lines)


def run_quick_tests(base_url: str, headers: dict):
    """Run a set of quick validation tests."""
    tests = [
        ("Greeting (cheap)",         "Hi!"),
        ("News/Tools (de)",          "Was sind die Nachrichten heute?"),
        ("News/Tools (en)",          "What's in the news today?"),
        ("Time tool",                "What time is it?"),
        ("Calculator",               "What is 47 * 389?"),
        ("Medium escalation",        "Explain how transformer attention works in detail"),
    ]
    
    print(f"\n{CYAN}Running {len(tests)} quick tests...{NC}\n")
    
    for name, prompt in tests:
        print(f"  {BOLD}Test: {name}{NC}")
        print(f"  {DIM}> {prompt}{NC}")
        
        result = send_message(base_url, headers, prompt, [])
        
        if "error" in result:
            print(f"  {RED}✗ {result['error']}{NC}")
        else:
            choices = result.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            meta = result.get("metadata", result.get("gateway_metadata", {}))
            usage = result.get("usage", {})
            
            model = meta.get("model_used", result.get("model", "?"))
            tier = meta.get("tier", "?")
            input_tok = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tok = usage.get("completion_tokens", usage.get("output_tokens", 0))
            tools = meta.get("tool_calls_executed", [])
            elapsed = result.get("elapsed", 0)
            
            # Truncate content for display
            preview = content[:120].replace("\n", " ")
            if len(content) > 120:
                preview += "..."
            
            print(f"  {GREEN}✓{NC} {preview}")
            print(f"    {DIM}{tier} | {model} | {input_tok}+{output_tok} tok | {elapsed*1000:.0f}ms{NC}")
            if tools:
                print(f"    {CYAN}🔧 {tools}{NC}")
        
        print()
    
    print(f"{GREEN}Tests complete!{NC}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM Gateway v1.5 Interactive Test Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--key", default=None, help="Gateway API key (if auth enabled)")
    parser.add_argument("--test", action="store_true", help="Run quick validation tests")
    parser.add_argument("--no-history", action="store_true", help="Don't keep conversation history")
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    headers = {}
    api_key = args.key
    
    # Auto-read from .env if no key provided
    if not api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GATEWAY_SECRET=") and not line.startswith("#"):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        print(f"  🔑 Using API key: {api_key[:10]}...")
    else:
        print(f"  ⚠️  No API key (use --key or set GATEWAY_SECRET in .env)")
    
    print_banner()
    print(f"  Connecting to {base_url}...")
    
    if not check_health(base_url, headers):
        sys.exit(1)
    
    # Quick test mode
    if args.test:
        run_quick_tests(base_url, headers)
        return
    
    # Interactive chat
    conversation = []
    total_cost = 0.0
    msg_count = 0
    
    print(f"  Type your message and press Enter. Commands:")
    print(f"    {DIM}/quit    — exit")
    print(f"    /clear   — reset conversation")
    print(f"    /test    — run quick tests")
    print(f"    /stats   — show session stats")
    print(f"    /health  — check gateway status{NC}")
    print()
    
    while True:
        try:
            user_input = input(f"{BOLD}You:{NC} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM}Goodbye!{NC}")
            break
        
        if not user_input:
            continue
        
        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(f"\n{DIM}Session: {msg_count} messages, ${total_cost:.6f} total cost{NC}")
            break
        
        if user_input.lower() == "/clear":
            conversation = []
            print(f"  {YELLOW}Conversation cleared{NC}\n")
            continue
        
        if user_input.lower() == "/test":
            run_quick_tests(base_url, headers)
            continue
        
        if user_input.lower() == "/stats":
            print(f"\n  {CYAN}Session Stats{NC}")
            print(f"    Messages: {msg_count}")
            print(f"    Total cost: ${total_cost:.6f}")
            print(f"    History length: {len(conversation)} turns")
            print()
            continue
        
        if user_input.lower() == "/health":
            check_health(base_url, headers)
            continue
        
        # Send message
        conv = [] if args.no_history else conversation
        data = send_message(base_url, headers, user_input, conv)
        output = format_response(data, conv)
        print(output)
        
        # Track stats
        msg_count += 1
        meta = data.get("metadata", data.get("gateway_metadata", {}))
        cost = meta.get("cost_usd", meta.get("estimated_cost", 0))
        if isinstance(cost, (int, float)):
            total_cost += cost
        
        print()


if __name__ == "__main__":
    main()
