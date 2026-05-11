#!/bin/bash
# ─────────────────────────────────────────────────────────────
# RC Church Translation Launcher
# Automator → Application → Run Shell Script
# Shell: /bin/bash   |   Pass input: as arguments
# ─────────────────────────────────────────────────────────────

PROJECT_DIR="/Users/mariakim/Development/ChurchTranslation/church-translation"
CONTROL_URL="http://localhost:9090"

# ── Option 4: trap ensures servers die when Automator app is quit ──────────
cleanup() {
    echo "[Launcher] Shutting down servers…"
    lsof -ti :9090 | xargs kill -9 2>/dev/null
    lsof -ti :8080 | xargs kill -9 2>/dev/null
    echo "[Launcher] Done."
}
trap cleanup EXIT TERM INT

# ─────────────────────────────────────────────────────────────
# Launch LadioCast
# ─────────────────────────────────────────────────────────────
open -a "LadioCast"
sleep 5

# ─────────────────────────────────────────────────────────────
# Start translation control server
# ─────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || exit 1
source venv/bin/activate

if lsof -i :9090 >/dev/null 2>&1; then
    echo "[Launcher] Control server already running."
else
    python3 control_server.py > /tmp/rc_translation.log 2>&1 &
    SERVER_PID=$!
    echo "[Launcher] Control server started (PID $SERVER_PID)"
fi

sleep 4

# ─────────────────────────────────────────────────────────────
# Open control panel only
# ─────────────────────────────────────────────────────────────
open -a "Google Chrome" "$CONTROL_URL"

# ─────────────────────────────────────────────────────────────
# Bring Chrome forward
# ─────────────────────────────────────────────────────────────
osascript <<'EOF'
tell application "Google Chrome" to activate
EOF

# ─────────────────────────────────────────────────────────────
# Wait for the control server process to exit.
# control_server.py kills itself when the browser tab closes
# (heartbeat timeout) or when the volunteer clicks Stop Server.
# When it exits, this wait returns, the script ends, and
# Automator's gear stops spinning.
# ─────────────────────────────────────────────────────────────
if [ -n "$SERVER_PID" ]; then
    wait "$SERVER_PID"
    echo "[Launcher] Control server exited. Cleaning up."
    lsof -ti :8080 | xargs kill -9 2>/dev/null
fi
