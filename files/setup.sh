#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  LLM Gateway v1.5 — Setup & Deployment Script                  ║
# ║  Tool Calling · News · i18n · 3-Tier Cascade                   ║
# ╚══════════════════════════════════════════════════════════════════╝
set -euo pipefail

GATEWAY_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$GATEWAY_DIR/venv"
SERVICE_NAME="llm-gateway"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PORT=8000

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   LLM Gateway v1.5 — Setup                      ║${NC}"
echo -e "${CYAN}║   Tool Calling · News · i18n · Cascade           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ─── 1. Python Check ────────────────────────────────────────────────────────
info "Checking Python..."
if command -v python3 &>/dev/null; then
    PY=$(python3 --version 2>&1)
    ok "Found $PY"
else
    err "Python 3 not found. Install python3 first."
    exit 1
fi

# ─── 2. Virtual Environment ─────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
ok "Activated venv"

# ─── 3. Install Dependencies ────────────────────────────────────────────────
info "Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$GATEWAY_DIR/requirements.txt" -q

# Optional: local Whisper for free audio transcription (no API key needed)
pip install faster-whisper --break-system-packages -q 2>/dev/null || echo "⚠ faster-whisper install failed (optional — needs ~500MB disk)"
ok "All dependencies installed"

# ─── 3b. System Dependencies (Tesseract OCR) ────────────────────────────────
if command -v tesseract &>/dev/null; then
    ok "Tesseract OCR already installed"
else
    info "Installing Tesseract OCR (for vision pipeline)..."
    if command -v apt &>/dev/null; then
        sudo apt install -y tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng 2>/dev/null && \
            ok "Tesseract OCR installed (DE + EN)" || \
            warn "Tesseract install failed — vision OCR will be disabled"
    else
        warn "Non-Debian system. Install tesseract-ocr manually for vision OCR."
    fi
fi

# ─── 4. Environment File ────────────────────────────────────────────────────
ENV_FILE="$GATEWAY_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    ok ".env file exists"
    # Check if GATEWAY_SECRET is set
    if grep -q "^GATEWAY_SECRET=" "$ENV_FILE" && ! grep -q "^GATEWAY_SECRET=$\|^#GATEWAY_SECRET" "$ENV_FILE"; then
        ok "GATEWAY_SECRET is configured"
    else
        # Auto-generate a secret
        NEW_SECRET=$(python3 -c "import secrets; print(f'gw-{secrets.token_hex(32)}')")
        if grep -q "GATEWAY_SECRET" "$ENV_FILE"; then
            sed -i "s|^#\?GATEWAY_SECRET=.*|GATEWAY_SECRET=$NEW_SECRET|" "$ENV_FILE"
        else
            echo "" >> "$ENV_FILE"
            echo "# Gateway API key (auto-generated)" >> "$ENV_FILE"
            echo "GATEWAY_SECRET=$NEW_SECRET" >> "$ENV_FILE"
        fi
        warn "GATEWAY_SECRET auto-generated: $NEW_SECRET"
        warn "Use this key in your API requests: Authorization: Bearer $NEW_SECRET"
    fi
else
    # Generate fresh secret
    NEW_SECRET=$(python3 -c "import secrets; print(f'gw-{secrets.token_hex(32)}')")
    warn "No .env file found — creating with auto-generated secret..."
    cat > "$ENV_FILE" << ENVEOF
# LLM Gateway v1.5 — Environment Variables

# ─── REQUIRED: At least one provider API key ────────────────────
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# ─── REQUIRED: Gateway authentication (protects your VPS!) ─────
GATEWAY_SECRET=$NEW_SECRET

# ─── CORS: Allowed origins (comma-separated) ───────────────────
# Leave empty for localhost-only (safe default)
# Set to your domain if using a frontend: https://chat.yourdomain.com
#CORS_ORIGINS=https://yourdomain.com

# ─── Optional: Ollama for local vision (if installed) ──────────
#OLLAMA_URL=http://localhost:11434
#OLLAMA_VISION_MODEL=moondream

# ─── Optional: Direct provider keys ────────────────────────────
#ANTHROPIC_API_KEY=sk-ant-...
#GROQ_API_KEY=gsk_...

# ─── Optional: News API (free tier: 100 req/day) ───────────────
#NEWSAPI_KEY=your-newsapi-key
#TAVILY_API_KEY=tvly-...
ENVEOF
    warn "Edit .env and add your OpenRouter API key: nano $ENV_FILE"
    info "Your auto-generated GATEWAY_SECRET: $NEW_SECRET"
    info "Use in requests: curl -H 'Authorization: Bearer $NEW_SECRET' ..."
fi

# ─── 5. Config Check ────────────────────────────────────────────────────────
if [ -f "$GATEWAY_DIR/config.yaml" ]; then
    ok "config.yaml exists"
    # Check mock mode
    if grep -q "mock_mode: true" "$GATEWAY_DIR/config.yaml"; then
        warn "mock_mode is TRUE — set to false for production"
    fi
else
    err "config.yaml not found!"
    exit 1
fi

# ─── 6. Systemd Service ─────────────────────────────────────────────────────
echo ""
read -p "$(echo -e "${CYAN}Install as systemd service? [y/N]:${NC} ")" INSTALL_SERVICE

if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    info "Creating systemd service..."
    sudo tee "$SERVICE_FILE" > /dev/null << SVCEOF
[Unit]
Description=LLM Gateway v1.5 - Cost-Optimized AI Routing with Tool Calling
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$GATEWAY_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$GATEWAY_DIR/.env
ExecStart=$VENV_DIR/bin/python -m uvicorn main:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    ok "Service installed and enabled"

    read -p "$(echo -e "${CYAN}Start service now? [y/N]:${NC} ")" START_NOW
    if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
        sudo systemctl restart "$SERVICE_NAME"
        sleep 2
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            ok "Gateway is running!"
        else
            err "Service failed to start. Check: journalctl -u $SERVICE_NAME -n 50"
            exit 1
        fi
    fi
else
    info "Skipping systemd setup."
    echo ""
    info "Manual start:"
    echo "  cd $GATEWAY_DIR"
    echo "  source venv/bin/activate"
    echo "  python -m uvicorn main:app --host 0.0.0.0 --port $PORT"
fi

# ─── 7. Verify ───────────────────────────────────────────────────────────────
echo ""
info "Verifying gateway..."
sleep 1

if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    VERSION=$(curl -sf "http://localhost:$PORT/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null || echo "?")
    AUTH_STATUS=$(curl -sf "http://localhost:$PORT/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print('🔒 AUTH ON' if d.get('security',{}).get('api_key_required') else '⚠️  AUTH OFF')" 2>/dev/null || echo "?")
    ok "Gateway v${VERSION} is live at http://localhost:$PORT ($AUTH_STATUS)"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ Setup Complete!                              ║${NC}"
    echo -e "${GREEN}║                                                  ║${NC}"
    echo -e "${GREEN}║   API:  http://localhost:$PORT/v1/chat/completions ║${NC}"
    echo -e "${GREEN}║   Web:  http://localhost:$PORT                    ║${NC}"
    echo -e "${GREEN}║   Test: python test_interactive.py               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
else
    warn "Gateway not responding yet on port $PORT"
    echo "  Start manually or check logs: journalctl -u $SERVICE_NAME -f"
fi

# ─── 8. Nginx Reverse Proxy (HTTPS) ────────────────────────────────────────
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  ⚠️  IMPORTANT: Secure your public VPS!${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Port 8000 should NOT be exposed directly to the internet."
echo "  Use nginx as reverse proxy with HTTPS (Let's Encrypt):"
echo ""
echo "  Step 1: Block port 8000 from outside"
echo "    sudo ufw deny 8000"
echo "    sudo ufw allow 443"
echo "    sudo ufw allow 80"
echo "    sudo ufw enable"
echo ""
echo "  Step 2: Install nginx + certbot"
echo "    sudo apt install -y nginx certbot python3-certbot-nginx"
echo ""
echo "  Step 3: Create nginx config"

# Generate nginx config
NGINX_CONF="$GATEWAY_DIR/nginx-gateway.conf"
cat > "$NGINX_CONF" << 'NGINXEOF'
# /etc/nginx/sites-available/llm-gateway
# Copy: sudo cp ~/gw/nginx-gateway.conf /etc/nginx/sites-available/llm-gateway
# Enable: sudo ln -sf /etc/nginx/sites-available/llm-gateway /etc/nginx/sites-enabled/
# Test: sudo nginx -t && sudo systemctl reload nginx

server {
    listen 80;
    server_name YOUR_DOMAIN;  # ← Replace with your domain or IP

    # Redirect HTTP → HTTPS (uncomment after certbot)
    # return 301 https://$host$request_uri;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for LLM responses (can be slow)
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        proxy_connect_timeout 10s;

        # Body size limit (images!)
        client_max_body_size 20m;
    }

    # Block direct access to sensitive paths
    location ~ ^/(security|metrics|stats) {
        # Only allow with valid auth header (nginx doesn't check, gateway does)
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

# HTTPS server block (created by certbot, or manually):
# server {
#     listen 443 ssl http2;
#     server_name YOUR_DOMAIN;
#     ssl_certificate /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem;
#     ... same location blocks as above ...
# }
NGINXEOF

echo "    Generated: $NGINX_CONF"
echo ""
echo "  Step 4: Enable + get HTTPS certificate"
echo "    sudo cp $NGINX_CONF /etc/nginx/sites-available/llm-gateway"
echo "    # Edit server_name in the config first!"
echo "    sudo ln -sf /etc/nginx/sites-available/llm-gateway /etc/nginx/sites-enabled/"
echo "    sudo nginx -t && sudo systemctl reload nginx"
echo "    sudo certbot --nginx -d YOUR_DOMAIN"
echo ""

echo ""
