#!/bin/bash

# ─────────────────────────────────────────────────────────────
# RC Church Translation Launcher
# ─────────────────────────────────────────────────────────────
# Intended for:
# - Automator Application
# - macOS launcher app
#
# This script:
# - Starts the control server
# - Opens the control panel in Chrome
# - Cleans up processes when closed
# ─────────────────────────────────────────────────────────────

# Project location (same across volunteer installs)
PROJECT_DIR="$HOME/Documents/church-translation"

CONTROL_PORT="${CONTROL_PORT:-9090}"
CAPTION_PORT="${CAPTION_PORT:-8080}"

CONTROL_URL="http://localhost:${CONTROL_PORT}"

# ─────────────────────────────────────────────────────────────
# Cleanup on exit
# ─────────────────────────────────────────────────────────────

cleanup() {
    echo "[Launcher] Shutting down servers…"

    lsof -ti :"$CONTROL_PORT" | xargs kill -9 2>/dev/null
    lsof -ti :"$CAPTION_PORT" | xargs kill -9 2>/dev/null

    echo "[Launcher] Done."
}

trap cleanup EXIT TERM INT

# ─────────────────────────────────────────────────────────────
# Enter project directory
# ─────────────────────────────────────────────────────────────

cd "$PROJECT_DIR" || {
    echo "[Launcher] Could not enter project directory."
    exit 1
}

# ─────────────────────────────────────────────────────────────
# Activate virtual environment
# ─────────────────────────────────────────────────────────────

if [ ! -d "venv" ]; then
    echo "[Launcher] Missing venv directory."
    exit 1
fi

source venv/bin/activate

# ─────────────────────────────────────────────────────────────
# Start control server
# ─────────────────────────────────────────────────────────────

if lsof -i :"$CONTROL_PORT" >/dev/null 2>&1; then
    echo "[Launcher] Control server already running."
else
    python3 control_server.py \
        > /tmp/rc_translation.log 2>&1 &

    SERVER_PID=$!

    echo "[Launcher] Control server started (PID $SERVER_PID)"
fi

# Wait for server startup
sleep 4

# ─────────────────────────────────────────────────────────────
# Open control panel
# ─────────────────────────────────────────────────────────────

open -a "Google Chrome" "$CONTROL_URL"

# Bring Chrome forward
osascript <<'EOF'
tell application "Google Chrome" to activate
EOF

# ─────────────────────────────────────────────────────────────
# Wait for control server exit
# ─────────────────────────────────────────────────────────────

if [ -n "$SERVER_PID" ]; then
    wait "$SERVER_PID"

    echo "[Launcher] Control server exited."

    lsof -ti :"$CAPTION_PORT" \
        | xargs kill -9 2>/dev/null
fi
