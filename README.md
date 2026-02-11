# ⚡ LLM Gateway

**Cost-optimized AI routing proxy with OpenAI-compatible API.**

Route requests across multiple LLM tiers — from free local tools to premium models — based on query complexity. Drop-in replacement for OpenAI's API that cuts costs by 90%+ without meaningful quality loss.

## Why?

| Approach | Quality | Cost per Query | Monthly (1K req/day) |
|---|---|---|---|
| Always Gemini 2.5 Pro | 93% | $0.02+ | ~$600 |
| Always Gemini 3 Flash | 92% | $0.02 | ~$600 |
| **LLM Gateway (cascade)** | **90–92%** | **$0.001** | **~$30** |

*Benchmarked across 36 tests (code, math, reasoning, knowledge, translation, creative, vision) scored by LLM-as-judge.*

The gateway handles 80% of requests with free tools and cheap models. Web search, stock prices, and weather run through free Python APIs — no paid subscriptions needed. Code stitching and web enrichment let the cheapest model match expensive model quality. Only genuinely complex tasks escalate to premium.

## Features

**Intelligent Routing**
- 4-tier cascade: `cheap` → `cheap_plus` → `medium` → `premium`
- Token-count routing with query intent detection
- Web search detection (DE/EN): "recherchiere", "aktuell", "find out about"
- Code complexity routing: short code → fast model, long programs → reliable model
- Finance/stock queries → free tool APIs instead of paid web search
- Budget guard with soft/medium/hard limits and kill switch

**Free Tool Calling**
- Weather via [Open-Meteo](https://open-meteo.com) — no API key needed
- Stock prices via [yfinance](https://github.com/ranaroussi/yfinance) — no API key needed
- News/web search via [DuckDuckGo](https://duckduckgo.com) — no API key needed
- Deep web extraction via [trafilatura](https://github.com/adbar/trafilatura)
- The LLM *decides* which tool to call — the gateway only executes

**Web Enrichment Pipeline (20x cost savings)**
- DuckDuckGo search → trafilatura full-text extraction → structured context injection
- Cheap model + web context achieves expensive model quality
- A web-enriched query costs $0.001 instead of $0.02 with Gemini 3 Flash directly

**Code Generation Quality**
- Post-generation validator: bracket balance, `ast.parse`, truncation detection
- Code stitching: continues truncated output via follow-up LLM call
- Code repair: self-correction cascade before premium escalation
- Diff-marker cleaning for models that output patch-style `+`/`-` lines
- Multi-language: Python (full syntax), C/C++/Java/JS (brackets), HTML (tags)

**Vision Pipeline**
- Tiered OCR: PaddleOCR → Tesseract → Cloud Vision
- Intent detection: skip cloud API when local OCR is sufficient
- Image preprocessing: resize, grayscale, format optimization

**Speech-to-Text (free)**
- 3-tier transcription: Browser Speech API → Groq Whisper (free) → local faster-whisper
- Auto-transcription: audio in chat messages is automatically transcribed before the LLM sees it
- Supports all common formats: mp3, m4a, wav, ogg, webm, flac, opus
- Zero cost for most use cases (Groq free tier or local CPU)

**Document Extraction**
- PDF text extraction with 3-way fallback: PyMuPDF → pdfplumber → pypdf
- DOCX/DOC text extraction via python-docx
- XLSX/XLS/CSV spreadsheet reading via openpyxl
- ZIP/TAR archive extraction (recursive)
- Audio files auto-redirected to transcription pipeline

**Infrastructure**
- Two-stage caching: exact (SHA-256) + semantic (embeddings)
- Context budgeting with per-tier token limits
- Streaming support (SSE)
- Built-in chat UI at `/chat`
- OpenAI-compatible API — works with any client
- Security: API key auth, policy gate, sensitive data blocking, CORS

## Architecture

```
Client (any OpenAI-compatible client or SDK)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Gateway (FastAPI)                                  │
│                                                         │
│  Request → Auth → Cache → Context Budget → Routing      │
│                                              │          │
│    ┌──────────┬────────────┬──────────┐      │          │
│    │  cheap    │ cheap_plus │  medium  │      │          │
│    │  +tools   │ +web search│          │      │          │
│    │  $0.10/M  │ $0.10/M    │ $0.10/M  │      │          │
│    └─────┬─────┴──────┬─────┴─────┬────┘      │          │
│          │   validate │ validate  │           │          │
│          ▼            ▼           ▼           │          │
│    ┌────────────────────────────────────┐     │          │
│    │     premium ($1.25/$10/M)          │◄────┘          │
│    │  (only if cheaper tiers fail)      │  escalation    │
│    └────────────────────────────────────┘                │
│                                                         │
│  Response ← Validator ← Cache Store ← Stream            │
└─────────────────────────────────────────────────────────┘
```

**Routing decisions:**

| Query type | Tier | Why |
|---|---|---|
| Simple questions, greetings | `cheap` | Fast, ~$0.001 |
| Finance: "DAX heute?" | `cheap` + tool call | Free yfinance API |
| Research: "Recherchiere..." | `cheap_plus` | Needs web grounding |
| Long code (200+ lines) | `medium` | More reliable output |
| Complex reasoning, large context | `premium` | Quality matters |

## Quick Start

### Prerequisites
- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (provides access to all models with a single key)

### Automated Setup

```bash
git clone https://github.com/sp4cerat/LLM-Gateway.git
cd LLM-Gateway/files
bash setup.sh
```

The setup script creates a venv, installs dependencies, generates a gateway API key, prompts for your OpenRouter key, and optionally installs a systemd service.

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: better OCR for vision pipeline
sudo apt install -y tesseract-ocr tesseract-ocr-deu
pip install paddlepaddle paddleocr

# Configure
cp .env.example .env
# Edit .env: add your OPENROUTER_API_KEY

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GATEWAY_SECRET" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

## Configuration

### config.yaml

The main configuration file controls models, budgets, and routing strategy.

```yaml
routing_strategy: "cascade"     # cascade | cost_optimized | quality_first

providers:
  cheap_model: "google/gemini-2.0-flash-001"
  cheap_plus_model: "google/gemini-3-flash-preview"
  medium_model: "google/gemini-3-flash-preview"
  premium_model: "google/gemini-2.5-pro"

budget:
  daily_soft_limit: 5.0         # Log warnings
  daily_medium_limit: 15.0      # Throttle requests
  daily_hard_limit: 50.0        # Kill switch
```

All providers go through [OpenRouter](https://openrouter.ai) by default, so you only need one API key. You can also configure individual providers (Anthropic, OpenAI, Groq) — see the comments in `config.yaml`.

### .env

```bash
OPENROUTER_API_KEY=sk-or-v1-...    # Required
GATEWAY_SECRET=gw-...               # Auto-generated by setup.sh

# Optional
NEWSAPI_KEY=...                     # NewsAPI.org (100 free req/day)
GROQ_API_KEY=...                    # Groq Whisper (free speech-to-text)
```

### Model Selection

The `model` parameter in API requests controls routing:

| Value | Behavior |
|---|---|
| `auto` | Cascade routing based on query complexity (recommended) |
| `cheap` | Force cheapest model (with tool calling) |
| `cheap_plus` | Force cheap model with web search grounding |
| `medium` | Force medium model (no tools, no web) |
| `premium` | Force premium model |

## Project Structure

```
llm-gateway/
├── main.py                 # FastAPI app, routing, cascade orchestration
├── router.py               # Heuristic intent classification
├── enhanced_router.py      # 4-layer router for large context requests
├── providers.py            # LLM provider integrations (OpenRouter, Anthropic, etc.)
├── models.py               # Pydantic data models
├── config.py               # YAML + .env configuration loader
├── cache.py                # Two-stage cache (exact + semantic)
├── context.py              # Context budgeting per tier
├── context_mapper.py       # Document chunking + targeted retrieval
├── rate_limiter.py         # Per-tier rate limiting
├── security.py             # Auth, policy gate, CORS, data blocking
├── metrics.py              # In-memory metrics
├── verification.py         # LLM-as-judge response scoring
├── response_validator.py   # Code syntax/truncation detection
├── tool_executor.py        # Free API tool execution
├── web_enrichment.py       # Deep search + trafilatura extraction
├── vision_processor.py     # OCR pipeline (PaddleOCR → Tesseract)
├── data_collector.py       # Benchmark data collection
├── config.yaml             # Model & budget configuration
├── requirements.txt        # Python dependencies
├── setup.sh                # Automated setup script
├── benchmark.py            # Full 3-tier benchmark (36 tests)
├── benchmark_v4.py         # Fast operational benchmark (11 tests)
├── diagnose_stitch.py      # Code stitch/repair diagnostics
├── test_interactive.py     # Interactive testing CLI
└── templates/
    └── chat.html           # Built-in chat UI
```

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible. Supports `messages`, `model`, `temperature`, `stream`, `max_tokens`.

### Management

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + version |
| `/v1/models` | GET | Available models/tiers |
| `/gateway/stats` | GET | Usage statistics & metrics |
| `/gateway/budget` | GET | Current budget status |
| `/gateway/budget/kill` | POST | Emergency stop |
| `/gateway/budget/reset` | POST | Reset daily budget |
| `/gateway/cache/invalidate` | POST | Clear all caches |
| `/v1/audio/transcribe` | POST | Speech-to-text (Whisper) |
| `/v1/files/extract` | POST | Document text extraction |
| `/chat` | GET | Built-in chat UI |

## Benchmark Results

### Full Quality Benchmark (36 tests, LLM-as-judge)

Tested across 7 categories: code, math, reasoning, knowledge, translation, creative, vision.

| Tier | Model | Quality | Cost (36 tests) | Efficiency |
|---|---|---|---|---|
| Cheap | Gemini 2.0 Flash | 90% | $0.007 | 480 pts/m$ |
| Medium | Gemini 3 Flash | 92% | $0.016 | 202 pts/m$ |
| Premium | Gemini 2.5 Pro | 93% | $1.122 | 3 pts/m$ |

**Key finding:** The cheap tier achieves 90% of premium quality at **0.6% of the cost**. The medium tier reaches 92% at 1.5% of the cost.

#### Quality by Category

| Category | Cheap | Medium | Premium |
|---|---|---|---|
| Code (7 tests) | 96% | 96% | 99% |
| Math (6 tests) | 91% | 94% | 94% |
| Reasoning (5 tests) | 87% | 92% | 89% |
| Knowledge (8 tests) | 85% | 89% | 86% |
| Translation (3 tests) | 96% | 94% | 96% |
| Creative (3 tests) | 83% | 92% | 94% |
| Vision (4 tests) | 94% | 92% | 94% |

#### Quality by Difficulty

| Difficulty | Cheap | Medium | Premium |
|---|---|---|---|
| Easy (12 tests) | 92% | 92% | 92% |
| Medium (11 tests) | 93% | 94% | 96% |
| Hard (13 tests) | 86% | 92% | 90% |

#### Best Strategy: Verify-then-Escalate

| Strategy | Quality | Cost | Savings |
|---|---|---|---|
| Always Premium | 93% | $1.122 | — |
| Classic (easy→cheap, hard→premium) | 92% | $0.678 | 40% |
| **Verify-then-Escalate (cascade)** | **92%** | **$0.016** | **99%** |

### Operational Benchmark (v4)

11 targeted tests validating cascade routing, code stitching, web search, and tool calling.

| Metric | Result |
|---|---|
| Tests passed | **11/11** |
| Total score | 91/110 (83%) |
| Total cost | **$0.013** |
| Premium escalations | **0** |
| Code stitching needed | 0 |

### Running Benchmarks

```bash
python benchmark.py --refresh              # Full 3-tier comparison (36 tests)
python benchmark.py --refresh-tier medium  # Single tier
python benchmark_v4.py --refresh           # Quick operational check (11 tests)
python diagnose_stitch.py                  # Code pipeline diagnostics
```

## How It Compares

LLM Gateway runs on a €5/month VPS (6 cores, 8GB RAM). Most requests are handled by Gemini 2.0 Flash with free tool calling. Web-enriched queries that would normally require Gemini 3 Flash ($0.02/query) cost $0.001 through the gateway's search + synthesis pipeline.

### Cost per Query

| Approach | Model | Cost/Query | Quality |
|---|---|---|---|
| Direct API call | Gemini 2.5 Pro | $0.02+ | 93% |
| Direct API call | Gemini 3 Flash (web) | $0.02 | ~92% |
| **LLM Gateway (cascade)** | **Gemini 2.0 Flash + enrichment** | **$0.001** | **90–92%** |

### vs. Commercial Gateways

| Feature | LLM Gateway | LiteLLM | Portkey | Helicone |
|---|---|---|---|---|
| **Intelligent routing** | ✅ 4-tier cascade | ⚠️ Basic | ✅ | ⚠️ |
| **Code stitching** | ✅ | ❌ | ❌ | ❌ |
| **Web search + enrichment** | ✅ Built-in (free) | ❌ | ❌ | ❌ |
| **Vision pipeline (local OCR)** | ✅ 3-stage | ❌ | ❌ | ❌ |
| **Free tool calling** | ✅ Weather/stocks/news | ❌ | ❌ | ❌ |
| Budget guard (kill switch) | ✅ 3-tier | ✅ | ✅ | ✅ |
| Security (auth, injection detection) | ✅ | ⚠️ | ✅ | ⚠️ |
| SOC 2 / compliance | ❌ | ⏳ | ✅ | ⚠️ |
| Load balancing | ❌ | ✅ | ✅ | ✅ |
| Multi-tenant RBAC | ❌ | ⚠️ | ✅ | ⚠️ |
| **Hosting cost** | **€5/month** | $100–300/mo | $499+/mo | SaaS |
| **Cost per query** | **$0.001** | $0.02 | $0.02 | $0.02 |

### Where It Excels

- **Research & fact-checking** — web enrichment produces more current answers than training data alone
- **OCR-heavy workloads** — 10K images/month: ~$0.40 vs $2–150 with cloud vision APIs
- **High volume on a budget** — 100K queries/month for ~$100 total
- **EU/GDPR compliance** — self-hosted on European infrastructure
- **Solo developers & small teams** — full control, no vendor lock-in

### Trade-offs

- Higher latency (~50–200ms routing overhead) vs. commercial gateways (<20ms in Rust/Go)
- Single server — no built-in load balancing or horizontal scaling
- No compliance certifications (SOC 2, ISO 27001)
- Not suited for multi-tenant SaaS or >1000 RPS sustained load

## Client Integration

### OpenClaw (recommended)

[OpenClaw](https://github.com/nicokempe/openclaw) is a terminal-based chat client that works with any OpenAI-compatible API.

**Setup:**

1. Install OpenClaw and create its config:
   ```bash
   # Config location: ~/.config/openclaw/config.toml
   mkdir -p ~/.config/openclaw
   nano ~/.config/openclaw/config.toml
   ```

2. Add this configuration:
   ```toml
   [providers.openai]
   api_key = "gw-YOUR-GATEWAY-SECRET"       # From your .env (GATEWAY_SECRET=...)
   base_url = "http://127.0.0.1:8000/v1"    # Gateway runs locally
   default_model = "auto"                    # Cascade routing (recommended)

   # If gateway runs on a remote server:
   # base_url = "http://YOUR-SERVER-IP:8000/v1"
   # Or via SSH tunnel:
   #   ssh -L 8000:127.0.0.1:8000 user@server
   #   base_url = "http://127.0.0.1:8000/v1"

   [conversation]
   max_context_tokens = 8000
   system_prompt = """You are a helpful assistant. Answer precisely and provide working code when asked."""
   ```

3. Available model values for `default_model`:

   | Value | Behavior |
   |---|---|
   | `auto` | Cascade routing — cheapest tier that works (recommended) |
   | `cheap` | Always Gemini 2.0 Flash (fastest, with tools) |
   | `medium` | Always Gemini 3 Flash (reliable for long code) |
   | `premium` | Always Gemini 2.5 Pro (highest quality) |

**Optional: Custom skills** (run local commands without LLM calls):

```toml
[skills]
enabled = true

[[skills.custom]]
name = "gateway_stats"
description = "Show gateway statistics and budget"
trigger = ["gateway status", "gateway stats", "budget"]
type = "bash"
command = """
curl -s -H "Authorization: Bearer $(grep GATEWAY_SECRET ~/gg/.env | cut -d= -f2)" \
  http://127.0.0.1:8000/gateway/stats | python3 -m json.tool
"""
```

### Python (OpenAI SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="gw-...")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer gw-..." \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

### Any OpenAI-compatible client

The gateway is a drop-in replacement for the OpenAI API. Point any client that supports custom base URLs to `http://YOUR-SERVER:8000/v1` with your gateway secret as API key.

## License

MIT
