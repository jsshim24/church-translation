#!/usr/bin/env python3
"""
control_server.py — Volunteer control panel server for RC Church Translation.

Serves the control UI at http://localhost:9090 and manages the main.py session.

Usage:
    python control_server.py
    python control_server.py --port 9090

Then open http://localhost:9090 in a browser.
"""

import json
import os
import sys
import signal
import subprocess
import threading
import tempfile
import argparse
import time
import http.server
from pathlib import Path
from typing import Optional, List, Dict

# ── State ──────────────────────────────────────────────────────────────────────

_session_proc: Optional[subprocess.Popen] = None
_session_lock = threading.Lock()
_outline_temp: Optional[tempfile.NamedTemporaryFile] = None

# Heartbeat — set to current time whenever /api/heartbeat is called.
_last_heartbeat: float = 0.0
_HEARTBEAT_TIMEOUT = 12  # seconds; browser pings every 4s

# Reference to the HTTPServer so handlers can call server.shutdown()
_http_server: Optional[http.server.HTTPServer] = None

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_audio_devices() -> List[Dict]:
    """Return list of audio input devices via sounddevice."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append({"index": i, "name": dev["name"]})
        return result
    except Exception as e:
        return [{"index": -1, "name": f"Error loading devices: {e}"}]


def build_command(payload: dict, outline_path: Optional[str]) -> list[str]:
    # Use the venv Python explicitly so dependencies (sounddevice, etc.) are available
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python, "main.py"]
    cmd += ["--source", payload["source"]]
    cmd += ["--target", ",".join(payload["targets"])]
    if payload.get("device") is not None:
        cmd += ["--device", str(payload["device"])]
    cmd += ["--port", str(payload.get("port", 8080))]
    if not payload.get("tunnel", True):
        cmd.append("--no-tunnel")
    if outline_path:
        cmd += ["--outline", outline_path]
    return cmd


def stream_output(proc: subprocess.Popen):
    """Stream subprocess stdout/stderr to this process's terminal."""
    def _stream(pipe):
        for line in iter(pipe.readline, b""):
            try:
                sys.stdout.write(line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            except Exception:
                pass

    t1 = threading.Thread(target=_stream, args=(proc.stdout,), daemon=True)
    t2 = threading.Thread(target=_stream, args=(proc.stderr,), daemon=True)
    t1.start()
    t2.start()


# ── HTTP Handler ───────────────────────────────────────────────────────────────

class ControlHandler(http.server.BaseHTTPRequestHandler):

    # ── Static UI ──────────────────────────────────────────────────────────────
    def _serve_ui(self):
        ui_path = Path(__file__).parent / "control.html"
        if not ui_path.exists():
            self.send_error(404, "control.html not found next to control_server.py")
            return
        html = ui_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html)

    # ── /api/devices ───────────────────────────────────────────────────────────
    def _serve_devices(self):
        devices = get_audio_devices()
        self._json(200, devices)

    # ── /api/start ─────────────────────────────────────────────────────────────
    def _handle_start(self):
        global _session_proc, _outline_temp

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "Invalid JSON"})
            return

        with _session_lock:
            if _session_proc and _session_proc.poll() is None:
                self._json(409, {"error": "Session already running"})
                return

            # Write outline to temp file if provided
            outline_path = None
            if payload.get("outline"):
                try:
                    _outline_temp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", encoding="utf-8", delete=False
                    )
                    _outline_temp.write(payload["outline"])
                    _outline_temp.flush()
                    _outline_temp.close()
                    outline_path = _outline_temp.name
                except Exception as e:
                    self._json(500, {"error": f"Could not write outline: {e}"})
                    return

            cmd = build_command(payload, outline_path)
            print(f"\n[ControlServer] Starting: {' '.join(cmd)}\n")

            try:
                _session_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(__file__).parent,
                )
                stream_output(_session_proc)
                self._json(200, {"ok": True, "pid": _session_proc.pid})
            except Exception as e:
                self._json(500, {"error": str(e)})

    # ── /api/stop ──────────────────────────────────────────────────────────────
    def _handle_stop(self):
        global _session_proc, _outline_temp

        with _session_lock:
            if _session_proc:
                try:
                    _session_proc.send_signal(signal.SIGINT)
                    _session_proc.wait(timeout=5)
                except Exception:
                    try:
                        _session_proc.terminate()
                    except Exception:
                        pass
                _session_proc = None
                print("[ControlServer] Session stopped.")

            # Clean up outline temp file
            if _outline_temp:
                try:
                    os.unlink(_outline_temp.name)
                except Exception:
                    pass
                _outline_temp = None

        self._json(200, {"ok": True})

    # ── /api/status ────────────────────────────────────────────────────────────
    def _serve_status(self):
        with _session_lock:
            running = _session_proc is not None and _session_proc.poll() is None
        self._json(200, {"running": running})

    # ── /api/heartbeat ─────────────────────────────────────────────────────────
    def _handle_heartbeat(self):
        global _last_heartbeat
        _last_heartbeat = time.monotonic()
        self._json(200, {"ok": True})
    def _proxy_latest(self):
        """Proxy /api/latest to the caption server port."""
        import urllib.request
        import urllib.error

        caption_port = 8080  # default; could be stored from last start payload
        try:
            with urllib.request.urlopen(
                f"http://localhost:{caption_port}/api/latest", timeout=1
            ) as r:
                data = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
        except Exception:
            self._json(503, {"lines": [], "updated": 0})

    # ── JSON helper ────────────────────────────────────────────────────────────
    def _json(self, status: int, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ── Routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/" or path == "/index.html":
                self._serve_ui()
            elif path == "/api/devices":
                self._serve_devices()
            elif path == "/api/status":
                self._serve_status()
            elif path == "/api/latest":
                self._proxy_latest()
            elif path == "/api/heartbeat":
                self._handle_heartbeat()
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass

    # ── /api/shutdown ─────────────────────────────────────────────────────────
    def _handle_shutdown(self):
        """Stop translation session and shut down the control server cleanly."""
        self._handle_stop()
        self._json(200, {"ok": True})
        def _do_shutdown():
            time.sleep(0.5)
            print("[ControlServer] Shutdown requested via UI.")
            if _http_server:
                _http_server.shutdown()
        threading.Thread(target=_do_shutdown, daemon=True).start()

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/start":
                self._handle_start()
            elif path == "/api/stop":
                self._handle_stop()
            elif path == "/api/shutdown":
                self._handle_shutdown()
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass

    def log_message(self, fmt, *args):
        # Suppress noisy GET /api/latest and /api/heartbeat polling logs
        msg = str(args[0] if args else "")
        if "/api/latest" not in msg and "/api/heartbeat" not in msg:
            super().log_message(fmt, *args)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Volunteer control panel server")
    parser.add_argument("--port", type=int, default=9090,
                        help="Port for the control panel (default: 9090)")
    args = parser.parse_args()

    global _http_server
    _http_server = http.server.HTTPServer(("", args.port), ControlHandler)
    server = _http_server

    # Start heartbeat watcher — shuts down if browser tab closes
    global _last_heartbeat
    _last_heartbeat = time.monotonic()  # grace period from startup

    def _heartbeat_watcher(srv):
        # Give the browser 20s to open before we start checking
        time.sleep(20)
        while True:
            time.sleep(3)
            if time.monotonic() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
                print("[ControlServer] Heartbeat timeout — browser tab closed. Shutting down.")
                srv.shutdown()  # unblocks serve_forever(); finally block cleans up
                break

    threading.Thread(target=_heartbeat_watcher, args=(server,), daemon=True).start()

    print(f"""
╔══════════════════════════════════════════════╗
║   RC Church · Live Translation               ║
║   Volunteer control panel                    ║
╚══════════════════════════════════════════════╝

  Open this in a browser:
  → http://localhost:{args.port}

  Press Ctrl+C to quit.
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[ControlServer] Shutting down…")
        with _session_lock:
            if _session_proc and _session_proc.poll() is None:
                try:
                    _session_proc.send_signal(signal.SIGINT)
                    _session_proc.wait(timeout=3)
                except Exception:
                    try:
                        _session_proc.terminate()
                    except Exception:
                        pass
        server.server_close()


if __name__ == "__main__":
    main()
