#!/usr/bin/env python3
"""Korean-to-English live sermon translation pipeline.

Captures live audio from a USB audio interface connected to a church
soundboard, streams to Soniox real-time STT via WebSocket,
translates each final phrase via Claude, and displays bilingual output.

Usage:
    python translate.py
    python translate.py --language "ko en" --endpoint-delay 1500
"""

import os
import sys
import json
import time
import queue
import argparse
import threading
import http.server
from collections import deque
from urllib.parse import urlparse

import sounddevice as sd
from websockets import ConnectionClosedOK
from websockets.sync.client import connect as ws_connect
from dotenv import load_dotenv
import anthropic

# ── Constants ──────────────────────────────────────────────────────────────────

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
CHUNK_SIZE = 3200           # 100ms of 16-bit mono @ 16kHz
DEFAULT_ENDPOINT_DELAY_MS = 1500
PHRASE_PUNCT = frozenset(".!?")

CONTEXT_WINDOW = 5  # number of past (Korean, English) pairs for Claude context

# Audio format constants (mono 16-bit 16kHz PCM)
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 32,000 bytes/sec

# Translation model
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a translation assistant for a live Korean Christian church sermon. "
    "Translate each phrase from Korean to English. "
    "Rules: Drop Korean hesitation sounds (아, 어) but preserve everything else "
    "including rhetorical questions (그죠?, 그렇죠?), affirmations (예, 네), "
    "stage directions (e.g. please put up the slide), and any instructions the speaker gives. "
    "Preferred theological terms: "
    "은혜 → grace, 말씀 → the Word, 성령 → the Holy Spirit, 구원 → salvation, "
    "화목 → reconciliation, 긍휼 → mercy, 성도 → congregation or saints, "
    "여러분 → everyone. "
    "Maintain the speaker's conversational tone and rhetorical style. "
    "If the input is already in English, output it as-is. "
    "Output ONLY the translated text — no commentary, notes, parenthetical remarks, "
    "or explanations. If the phrase is incomplete, translate what is there and stop — "
    "do not add trailing periods, dashes, ellipses, or other punctuation to signal "
    "incompleteness. Only use dashes or ellipses when they reflect the speaker's "
    "actual rhetorical intent. "
    "If the input is garbled, unintelligible, or not recognizable as a real phrase, "
    "output exactly: [SKIP] "
    "Do not summarize, paraphrase, or make content decisions — "
    "translate the full meaning of what was said."
)

# ── Configuration ──────────────────────────────────────────────────────────────


def load_config():
    """Load .env and extract API keys."""
    load_dotenv(override=True)

    soniox_api_key = os.getenv("SONIOX_API_KEY")
    if not soniox_api_key:
        sys.exit("Error: SONIOX_API_KEY not set in .env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        sys.exit("Error: ANTHROPIC_API_KEY not set in .env — please add your key")

    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    return {
        "soniox_api_key": soniox_api_key,
        "api_key": api_key,
        "model": model,
    }


# ── Live Audio Capture ────────────────────────────────────────────────────────


def select_audio_device():
    """List available input devices and prompt user to select one."""
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((i, dev))

    if not input_devices:
        sys.exit("Error: No audio input devices found")

    print("Available audio input devices:")
    print("\u2500" * 60)
    for idx, dev in input_devices:
        sr = dev["default_samplerate"]
        ch = dev["max_input_channels"]
        print(f"  [{idx}]  {dev['name']}  ({ch}ch, {sr:.0f}Hz)")
    print()

    while True:
        try:
            choice = input("Enter device index to use: ").strip()
            idx = int(choice)
            dev = sd.query_devices(idx)
            if dev["max_input_channels"] > 0:
                return idx, dev["name"]
            print("  That device has no input channels. Try again.")
        except (ValueError, sd.PortAudioError):
            print("  Invalid device index. Try again.")


class AudioCapture:
    """Captures live audio from an input device into a thread-safe buffer.

    Audio chunks are placed into a queue for real-time STT streaming.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.audio_queue = queue.Queue()
        self.stream = None
        self.capturing = False
        self.total_bytes = 0
        self.start_time = None

    def _callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs on audio thread."""
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        raw = bytes(indata)
        self.audio_queue.put(raw)
        self.total_bytes += len(raw)

    def start(self):
        """Start audio capture."""
        self.capturing = True
        self.start_time = time.monotonic()
        # blocksize in frames; each frame = 2 bytes (16-bit mono)
        blocksize = CHUNK_SIZE // 2
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=blocksize,
            device=self.device_index,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        """Stop audio capture."""
        self.capturing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def elapsed(self) -> float:
        """Seconds since capture started."""
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time


# ── Soniox Real-Time STT ─────────────────────────────────────────────────────


def _stream_audio(audio_capture: AudioCapture, ws) -> None:
    """Send audio chunks from the capture queue to the WebSocket."""
    try:
        while audio_capture.capturing:
            try:
                chunk = audio_capture.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            ws.send(chunk)

        # Empty string signals end-of-audio to the server
        ws.send("")
    except Exception:
        # WebSocket already closed (e.g. Ctrl+C teardown)
        pass


def soniox_transcribe(soniox_api_key: str, audio_capture: AudioCapture,
                      language_hints: list, endpoint_delay_ms: int):
    """Sync generator yielding (transcript, elapsed_secs, dbg) from Soniox STT.

    Connects to Soniox via sync WebSocket, streams audio in a background
    thread, and receives transcription results in the calling thread.
    """
    config = {
        "api_key": soniox_api_key,
        "model": "stt-rt-v4",
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "language_hints": language_hints,
        "enable_endpoint_detection": True,
        "enable_language_identification": True,
        "max_endpoint_delay_ms": endpoint_delay_ms,
    }

    with ws_connect(SONIOX_WS_URL) as ws:
        # Send config as first message
        ws.send(json.dumps(config))
        print("  [STT] Connected to Soniox", file=sys.stderr)

        # Stream audio in background thread
        threading.Thread(
            target=_stream_audio,
            args=(audio_capture, ws),
            daemon=True,
        ).start()

        final_tokens = []
        # Track the last speech token's timing for latency measurement.
        # end_ms = audio position of the token, recv_wall = when we got it.
        # This excludes endpoint delay (which only affects <end> arrival).
        last_speech_end_ms = 0
        last_speech_recv_wall = 0.0

        try:
            while True:
                message = ws.recv()
                recv_wall = time.monotonic()
                data = json.loads(message)

                if data.get("error_code") is not None:
                    print(f"  [STT] Error: {data['error_code']} - "
                          f"{data['error_message']}", file=sys.stderr)
                    break

                tokens = data.get("tokens", [])
                end_seen = False
                punct_seen = False
                for token in tokens:
                    if not token.get("is_final"):
                        continue
                    text = token.get("text", "")
                    if text == "<end>":
                        end_seen = True
                        continue
                    if text:
                        final_tokens.append(text)
                        end_ms = token.get("end_ms", 0)
                        if end_ms:
                            last_speech_end_ms = end_ms
                            last_speech_recv_wall = recv_wall
                        if text.rstrip()[-1:] in PHRASE_PUNCT:
                            punct_seen = True

                # Yield on punctuation (fast) or <end> (fallback flush)
                if (punct_seen or end_seen) and final_tokens:
                    full_text = "".join(final_tokens).strip()

                    if last_speech_end_ms and audio_capture.start_time:
                        spoken_wall = audio_capture.start_time + last_speech_end_ms / 1000
                        latency_ms = int((last_speech_recv_wall - spoken_wall) * 1000)
                    else:
                        latency_ms = 0

                    if full_text:
                        yield full_text, audio_capture.elapsed(), {
                            "stt_latency_ms": latency_ms,
                            "recv_wall": recv_wall,
                        }

                    final_tokens.clear()
                    last_speech_end_ms = 0
                    last_speech_recv_wall = 0.0
                elif end_seen:
                    final_tokens.clear()
                    last_speech_end_ms = 0
                    last_speech_recv_wall = 0.0

                if data.get("finished"):
                    break

        except ConnectionClosedOK:
            pass
        except Exception as e:
            print(f"  [STT] Error: {e}", file=sys.stderr)


# ── Claude Translation ────────────────────────────────────────────────────────


def translate_phrase(client, korean_text: str, context: deque,
                     model: str = DEFAULT_MODEL) -> str:
    """Translate a Korean phrase to English using Claude with rolling context."""
    messages = []

    # Add previous pairs as conversation context
    for ko, en in context:
        messages.append({"role": "user", "content": ko})
        messages.append({"role": "assistant", "content": en})

    # Add the current phrase to translate
    messages.append({"role": "user", "content": korean_text})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text.strip()


# ── Web Caption Server ─────────────────────────────────────────────────────────

# Shared state for web clients
_web_state = {"lines": [], "updated": 0}
_web_lock = threading.Lock()


def _update_web_state(english: str, korean: str, timestamp: str):
    """Thread-safe update of translation state for web clients."""
    with _web_lock:
        _web_state["lines"].append({
            "english": english,
            "korean": korean,
            "timestamp": timestamp,
        })
        _web_state["updated"] = time.time()


def _get_web_state_json() -> bytes:
    """Thread-safe read of translation state as JSON bytes."""
    with _web_lock:
        return json.dumps(_web_state).encode()


CAPTION_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    width: 100%; height: 100%;
    background: transparent;
    overflow: hidden;
  }
  #container {
    width: 100%; height: 100%;
    overflow-y: auto;
    scroll-behavior: smooth;
    scrollbar-width: none;
    -ms-overflow-style: none;
  }
  #container::-webkit-scrollbar { display: none; }
  .phrase {
    animation: fadeIn 0.25s ease-out;
  }
  .phrase-korean {
    opacity: 0.6;
    font-size: 0.65em;
    display: block;
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
</style>
</head><body>
<div id="container">
  <div id="lines"></div>
</div>
<script>
(function() {
  const params = new URLSearchParams(window.location.search);

  // Typography
  const fontSize   = params.get('fontSize')   || '48';
  const fontFamily = params.get('fontFamily') || 'system-ui, sans-serif';
  const googleFont = params.get('googleFont');
  const fontWeight = params.get('fontWeight') || 'normal';
  const color      = params.get('color')      || 'white';
  const lineSpacing = params.get('lineSpacing') || '1.4';
  const textAlign  = params.get('textAlign')  || 'left';
  const textShadow = params.get('textShadow') || 'none';

  // Layout
  const bgColor = params.get('bgColor') || 'transparent';
  const padding = params.get('padding') || '20';
  const maxLines = params.get('maxLines') ? parseInt(params.get('maxLines')) : 0;

  // Content
  const showKorean = params.get('showKorean') === 'true';

  // Load Google Font if specified
  if (googleFont) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family='
              + encodeURIComponent(googleFont) + '&display=swap';
    document.head.appendChild(link);
  }

  const container = document.getElementById('container');
  const linesDiv  = document.getElementById('lines');

  // Apply styles
  document.body.style.background = bgColor;
  container.style.padding = padding + 'px';

  const resolvedFamily = googleFont
    ? '"' + googleFont.replace(/\+/g, ' ') + '", ' + fontFamily
    : fontFamily;
  linesDiv.style.cssText = [
    'font-size:'    + fontSize + 'px',
    'font-family:'  + resolvedFamily,
    'font-weight:'  + fontWeight,
    'color:'        + color,
    'line-height:'  + lineSpacing,
    'text-align:'   + textAlign,
    'text-shadow:'  + textShadow,
  ].join(';');

  let lastCount = 0;
  let lastUpdated = 0;
  const DOM_CAP = 200;  // hard cap to prevent infinite DOM growth

  async function poll() {
    try {
      const resp = await fetch('/api/latest');
      const data = await resp.json();
      if (data.updated === lastUpdated) return;
      lastUpdated = data.updated;

      // Append only new phrases (inline, paragraph-style)
      const newLines = data.lines.slice(lastCount);
      for (const line of newLines) {
        if (showKorean && line.korean) {
          const ko = document.createElement('span');
          ko.className = 'phrase-korean';
          ko.textContent = line.korean;
          linesDiv.appendChild(ko);
        }

        const span = document.createElement('span');
        span.className = 'phrase';
        span.textContent = line.english + ' ';
        linesDiv.appendChild(span);
      }
      lastCount = data.lines.length;

      // Trim DOM: respect maxLines (count phrase spans), and hard cap
      const phrases = linesDiv.querySelectorAll('.phrase');
      const limit = maxLines > 0 ? maxLines : DOM_CAP;
      let toRemove = phrases.length - limit;
      while (toRemove > 0) {
        // Remove the phrase span and its preceding korean span if present
        const first = phrases[phrases.length - toRemove];
        if (first.previousElementSibling
            && first.previousElementSibling.classList.contains('phrase-korean')) {
          first.previousElementSibling.remove();
        }
        first.remove();
        toRemove--;
      }

      // Smooth scroll to bottom
      container.scrollTop = container.scrollHeight;
    } catch (e) {
      // Server not ready or network hiccup — silently retry
    }
  }

  setInterval(poll, 150);
})();
</script>
</body></html>
"""


class _CaptionHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the caption web server."""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/latest":
            data = _get_web_state_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path == "/":
            html = CAPTION_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html)

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default stderr request logging."""
        pass


def start_caption_server(port: int):
    """Start the caption web server in a daemon thread.

    Returns the HTTPServer instance (can be used to shut down later).
    """
    server = http.server.HTTPServer(("", port), _CaptionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ── Main ──────────────────────────────────────────────────────────────────────


def format_timestamp(elapsed: float) -> str:
    """Format elapsed seconds as HH:MM:SS or MM:SS."""
    total_secs = int(elapsed)
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Korean-to-English live sermon translator")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (skip interactive selection)")
    parser.add_argument("--model", type=str, default=None,
                        help="Claude model for translation (overrides CLAUDE_MODEL env var)")
    parser.add_argument("--haiku", action="store_true",
                        help="Use claude-haiku-4-5 for faster (but lighter) translation")
    parser.add_argument("--language", type=str, default="ko en",
                        help='Language hints, e.g. "ko" or "ko en" (space-separated)')
    parser.add_argument("--endpoint-delay", type=int, default=DEFAULT_ENDPOINT_DELAY_MS,
                        help=f"Max endpoint delay in ms (500-3000, default {DEFAULT_ENDPOINT_DELAY_MS})")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web caption server port (default: 8080, 0 to disable)")
    args = parser.parse_args()

    config = load_config()
    if args.haiku:
        model = "claude-haiku-4-5"
    else:
        model = args.model or config["model"]

    language_hints = args.language.split()

    # Select audio input device
    if args.device is not None:
        dev = sd.query_devices(args.device)
        device_index, device_name = args.device, dev["name"]
    else:
        device_index, device_name = select_audio_device()
    print(f"\n  Using: [{device_index}] {device_name}\n")

    # Initialize translator
    translator = anthropic.Anthropic(api_key=config["api_key"])

    # Rolling context for Claude
    context = deque(maxlen=CONTEXT_WINDOW)

    # Start web caption server
    if args.port > 0:
        start_caption_server(args.port)
        web_url = f"http://localhost:{args.port}"
    else:
        web_url = None

    # Start audio capture
    capture = AudioCapture(device_index)
    phrase_count = 0

    print("=" * 50)
    print("  Korean \u2192 English Sermon Translation")
    print(f"  Model:   {model}")
    print(f"  STT:     Soniox stt-rt-v4 ({', '.join(language_hints)})")
    if web_url:
        print(f"  Web:     {web_url}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()

    try:
        capture.start()

        for korean_text, elapsed_time, dbg in soniox_transcribe(
            config["soniox_api_key"], capture, language_hints, args.endpoint_delay
        ):
            ts = format_timestamp(elapsed_time)

            tl_start = time.monotonic()
            try:
                english_text = translate_phrase(translator, korean_text, context, model)
            except Exception as e:
                english_text = f"[Translation error: {e}]"
            tl_secs = time.monotonic() - tl_start

            # Skip garbled/unintelligible transcriptions
            if english_text.strip() == "[SKIP]":
                print(f"  [SKIP] \"{korean_text}\"", file=sys.stderr)
                continue

            context.append((korean_text, english_text))
            phrase_count += 1

            # Push to web caption server
            if web_url:
                _update_web_state(english_text, korean_text, ts)

            stt_ms = dbg["stt_latency_ms"]
            total_secs = stt_ms / 1000 + tl_secs
            print(f"[{ts}] \U0001f1f0\U0001f1f7 {korean_text}")
            print(f"       \U0001f1fa\U0001f1f8 {english_text}")
            print(f"       \u23f1 stt={stt_ms}ms + tl={tl_secs:.1f}s = {total_secs:.1f}s")
            print("\u2500" * 40)

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        duration = capture.elapsed()
        print()
        print("=" * 50)
        print(f"  Session ended")
        print(f"  Duration:  {format_timestamp(duration)}")
        print(f"  Phrases:   {phrase_count}")
        print("=" * 50)


if __name__ == "__main__":
    main()
