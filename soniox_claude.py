import json
import os
import re
import sys
import queue
import time
import threading
import argparse
import subprocess
import http.server
from typing import Optional
from urllib.parse import urlparse

import anthropic

# Suppress noisy thread exception tracebacks on Ctrl+C.
threading.excepthook = lambda args: None

import sounddevice as sd
from dotenv import load_dotenv
from websockets import ConnectionClosedOK
from websockets.sync.client import connect

SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

SAMPLE_RATE = 16000
CHUNK_FRAMES = 1600  # 100ms at 16kHz

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a live translation assistant for a Korean church sermon. "
    "You receive a rolling context window of recent phrases; prior translations are provided as context. "
    "Translate ONLY the latest phrase from Korean to English. "
    "Drop Korean hesitation fillers (아, 어). "
    "Preferred terms: 여러분 → everyone; 정목사 → Pastor Chung. "
    "If input contains both English and Korean, keep the English as-is and translate the Korean portions into English, "
    "even if the Korean repeats or paraphrases the English — always include both. "
    "If input is entirely in English, output it unchanged. "
    "Prefix output with the ISO 639-1 language code in brackets, e.g. [en]. "
    "Output ONLY the translation — no commentary or notes. "
    "Phrases may arrive as incomplete clauses. Translate only the words present — "
    "never infer or complete missing verbs or conclusions. "
    "If the fragment is too incomplete or garbled, output exactly: [SKIP] "
    "Short fragments that lack a verb or predicate and cannot stand alone as a meaningful sentence "
    "should be [SKIP]ped — they will be prepended to the next phrase automatically. "
    "When quoting or referencing Bible passages, use the English Standard Version (ESV) for English."
)

SYSTEM_PROMPT_EN_TO_KO = (
    "You are a live translation assistant for an English church sermon. "
    "You receive a rolling context window of recent phrases; prior translations are provided as context. "
    "Translate ONLY the latest phrase from English to Korean. "
    "Drop English hesitation fillers (uh, um, like, you know, so, I mean). "
    "Prefix output with the ISO 639-1 language code in brackets, e.g. [ko]. "
    "Output ONLY the translation — no commentary or notes. "
    "Phrases may arrive as incomplete clauses. Translate only the words present — "
    "never infer or complete missing verbs or conclusions. "
    "If the fragment is too incomplete or garbled, output exactly: [SKIP] "
    "Short fragments that lack a verb or predicate and cannot stand alone as a meaningful sentence "
    "should be [SKIP]ped — they will be prepended to the next phrase automatically. "
    "Use natural, fluent Korean appropriate for a church setting. "
    "Prefer formal polite speech (합쇼체/해요체) as is standard for sermon translation. "
    "When quoting or referencing Bible passages, use the New Korean Revised Version (개역개정) for Korean."
)

# ── Web State ─────────────────────────────────────────────────────────────────

_web_state = {"lines": [], "updated": 0}
_web_lock = threading.Lock()


def _update_web_state(kind: str, lang: str, text: str):
    """kind='transcription' or 'translation', lang='en'/'ko'/etc."""
    with _web_lock:
        _web_state["lines"].append({"kind": kind, "lang": lang, "text": text})
        _web_state["updated"] = time.time()


def _get_web_state_json() -> bytes:
    with _web_lock:
        return json.dumps(_web_state).encode()


# ── HTML Template ─────────────────────────────────────────────────────────────

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
  .line-item {
    animation: fadeIn 0.25s ease-out;
  }
  .span-item {
    /* no animation — instant append for paragraph mode */
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

  // Content filtering
  const mode    = params.get('mode')    || 'transcription';
  const lang    = params.get('lang')    || 'en';
  const display = params.get('display') || 'line';

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
  const bgColor  = params.get('bgColor') || 'transparent';
  const padding  = params.get('padding') || '20';
  const maxLines = Math.min(
    params.get('maxLines') ? parseInt(params.get('maxLines')) : 0,
    200
  );

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
  const DOM_CAP = 200;

  async function poll() {
    try {
      const resp = await fetch('/api/latest');
      const data = await resp.json();
      if (data.updated === lastUpdated) return;
      lastUpdated = data.updated;

      // Filter and append only new lines
      const allLines = data.lines;
      const newLines = allLines.slice(lastCount);
      lastCount = allLines.length;

      for (const line of newLines) {
        if (line.kind !== mode) continue;
        if (line.lang !== lang) continue;

        if (display === 'paragraph') {
          const span = document.createElement('span');
          span.className = 'span-item';
          span.textContent = line.text + ' ';
          linesDiv.appendChild(span);
        } else {
          const div = document.createElement('div');
          div.className = 'line-item';
          div.textContent = line.text;
          linesDiv.appendChild(div);
        }
      }

      // Trim DOM
      const selector = display === 'paragraph' ? '.span-item' : '.line-item';
      const items = linesDiv.querySelectorAll(selector);
      const limit = maxLines > 0 ? maxLines : DOM_CAP;
      const toRemove = items.length - limit;
      for (let i = 0; i < toRemove; i++) {
        items[i].remove();
      }

      container.scrollTop = container.scrollHeight;
    } catch (e) {}
  }

  setInterval(poll, 150);
})();
</script>
</body></html>
"""

# ── HTTP Server ───────────────────────────────────────────────────────────────


class _CaptionHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
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
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        pass


def start_caption_server(port: int):
    server = http.server.HTTPServer(("", port), _CaptionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ── Cloudflare Tunnel ─────────────────────────────────────────────────────────


def start_cloudflare_tunnel(tunnel_name: str, port: int):
    """Launch cloudflared as a subprocess for a named tunnel."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "run", "--url", f"http://localhost:{port}", tunnel_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# Get Soniox STT config.
def get_config(api_key: str) -> dict:
    return {
        "api_key": api_key,
        "model": "stt-rt-v4",
        "language_hints": ["ko", "en"],
        "language_hints_strict": True,
        "enable_language_identification": True,
        "enable_endpoint_detection": True,
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "translation": {
            "type": "one_way",
            "target_language": "en",
        },
        "context": {
            "general": [
                {"key": "domain", "value": "Religion"},
                {"key": "topic", "value": "Korean church sermon"},
            ],
            "text": "Live Korean church sermon with a pastor preaching to the congregation.",
            "terms": [
                "하나님", "예수님", "성령", "아멘",
            ],
            "translation_terms": [
                {"source": "하나님", "target": "God"},
                {"source": "예수님", "target": "Jesus"},
                {"source": "성령", "target": "the Holy Spirit"},
            ],
        },
    }


def get_config_en_to_ko(api_key: str) -> dict:
    return {
        "api_key": api_key,
        "model": "stt-rt-v4",
        "language_hints": ["en"],
        "language_hints_strict": True,
        "enable_language_identification": True,
        "enable_endpoint_detection": True,
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "translation": {
            "type": "one_way",
            "target_language": "ko",
        },
        "context": {
            "general": [
                {"key": "domain", "value": "Religion"},
                {"key": "topic", "value": "English church sermon"},
            ],
            "text": "Live English church sermon with a pastor preaching to the congregation.",
            "terms": [
                "God", "Jesus", "Holy Spirit", "amen",
            ],
            "translation_terms": [
                {"source": "God", "target": "하나님"},
                {"source": "Jesus", "target": "예수님"},
                {"source": "the Holy Spirit", "target": "성령"},
            ],
        },
    }


# List available input devices and prompt user to select one.
def select_audio_device():
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((i, dev))

    if not input_devices:
        sys.exit("Error: No audio input devices found")

    print("Available audio input devices:")
    print("─" * 60)
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


# Stream microphone audio to the websocket.
def stream_audio(device_index: int, ws, stop_event: threading.Event) -> None:
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_FRAMES,
        device=device_index,
        dtype="int16",
        channels=1,
        callback=callback,
    )

    try:
        with stream:
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                ws.send(chunk)
    except Exception:
        pass

    # Empty string signals end-of-audio to the server.
    try:
        ws.send("")
    except Exception:
        pass


# Convert tokens into a readable transcript.
def render_tokens(final_tokens: list[dict]) -> str:
    text_parts: list[str] = []
    current_speaker: Optional[str] = None
    current_language: Optional[str] = None

    # Process all tokens in order.
    for token in final_tokens:
        text = token["text"]
        if text == "<end>":
            continue
        speaker = token.get("speaker")
        language = token.get("language")
        is_translation = token.get("translation_status") == "translation"

        # Speaker changed -> add a speaker tag.
        #if speaker is not None and speaker != current_speaker:
        #    if current_speaker is not None:
        #        text_parts.append("\n\n")
        #    current_speaker = speaker
        #    current_language = None  # Reset language on speaker changes.
        #    text_parts.append(f"Speaker {current_speaker}:")

        # Language changed -> add a language or translation tag.
        if language is not None and language != current_language:
            # Add a space before the language tag if there's preceding text.
            if text_parts and not text_parts[-1].endswith(" "):
                text_parts.append(" ")
            current_language = language
            prefix = "[Translation] " if is_translation else ""
            text_parts.append(f"{prefix}[{current_language}] ")
            text = text.lstrip()

        text_parts.append(text)

    return "".join(text_parts)


def translate_phrase(client: anthropic.Anthropic, korean_text: str,
                     context: list[tuple[str, str]], model: str = DEFAULT_MODEL,
                     system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = []
    for speaker, translation in context:
        messages.append({"role": "user", "content": speaker})
        messages.append({"role": "assistant", "content": translation})
    messages.append({"role": "user", "content": korean_text})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text.strip()


def _push_to_web(kind: str, text: str, fallback_lang: str = "en"):
    """Parse [lang] prefix from text and push to web state."""
    m = re.match(r"\[([a-z]{2})\]\s*", text)
    if m:
        lang = m.group(1)
        raw_text = text[m.end():]
    else:
        lang = fallback_lang
        raw_text = text
    if raw_text.strip():
        _update_web_state(kind, lang, raw_text.strip())


def run_session(api_key: str, device_index: int, anthropic_api_key: str,
                lang: str = "ko") -> None:
    if lang == "en":
        config = get_config_en_to_ko(api_key)
        system_prompt = SYSTEM_PROMPT_EN_TO_KO
        fallback_lang = "ko"
    else:
        config = get_config(api_key)
        system_prompt = SYSTEM_PROMPT
        fallback_lang = "en"
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    context: list[tuple[str, str]] = []

    print("Connecting to Soniox...")
    with connect(SONIOX_WEBSOCKET_URL) as ws:
        # Send first request with config.
        ws.send(json.dumps(config))

        # Start streaming audio in the background.
        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=stream_audio,
            args=(device_index, ws, stop_event),
            daemon=True,
        )
        audio_thread.start()

        print("Session started. Speak into your microphone. Press Ctrl+C to stop.")

        final_tokens: list[dict] = []
        final_translation_tokens: list[dict] = []
        prev_final_count = 0
        prev_translation_count = 0
        recent_phrases: list[str] = []
        pending_text = ""

        try:
            while True:
                message = ws.recv()
                res = json.loads(message)

                # Error from server.
                if res.get("error_code") is not None:
                    print(f"Error: {res['error_code']} - {res['error_message']}")
                    break

                # Parse tokens from current response.
                for token in res.get("tokens", []):
                    if token.get("text"):
                        # Track translation tokens separately (not printed,
                        # but available as a signal for external translation).
                        if token.get("translation_status") == "translation":
                            if token.get("is_final"):
                                final_translation_tokens.append(token)
                            continue
                        if token.get("is_final"):
                            final_tokens.append(token)

                # Print buffered transcription tokens when a translation arrives.
                # (Change to `len(final_tokens) == prev_final_count` to print immediately.)
                if len(final_translation_tokens) == prev_translation_count:
                    continue

                new_tokens = final_tokens[prev_final_count:]
                prev_final_count = len(final_tokens)
                prev_translation_count = len(final_translation_tokens)
                text = render_tokens(new_tokens)
                recent_phrases.append(text)
                if len(recent_phrases) > 5:
                    recent_phrases.pop(0)

                # Accumulate with any previously skipped text
                combined = (pending_text + " " + text).strip() if pending_text else text

                text = f"[Transcription] {text}"
                print(text)

                # Push transcription to web state
                _push_to_web("transcription", text.removeprefix("[Transcription] "), fallback_lang)

                try:
                    translation = translate_phrase(client, combined, context,
                                                   system_prompt=system_prompt)
                except Exception as e:
                    print(f"[Translation error: {e}]")
                    continue

                if "[SKIP]" in translation:
                    pending_text = combined
                    continue

                pending_text = ""
                context.append((combined, translation))
                if len(context) > 5:
                    context.pop(0)

                translation = f"[Translation] {translation}"
                print(translation)

                # Push translation to web state
                _push_to_web("translation", translation.removeprefix("[Translation] "), fallback_lang)

                # Session finished.
                if res.get("finished"):
                    print("Session finished.")

        except ConnectionClosedOK:
            pass
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            stop_event.set()
            audio_thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Soniox real-time sermon translation from microphone")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko",
                        help="Source language: ko = Korean→English (default), en = English→Korean")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (skip interactive selection)")
    parser.add_argument("--port", type=int, default=8080, help="Web caption server port (default: 8080, 0 to disable)")
    parser.add_argument("--tunnel", type=str, default=None, help="Cloudflare tunnel name (e.g. church-live)")
    args = parser.parse_args()

    load_dotenv(override=True)
    api_key = os.environ.get("SONIOX_API_KEY")
    if api_key is None:
        raise RuntimeError("Missing SONIOX_API_KEY. Set it in .env or environment.")

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY. Set it in .env or environment.")

    if args.device is not None:
        device_index = args.device
        dev = sd.query_devices(device_index)
        print(f"Using device [{device_index}]: {dev['name']}")
    else:
        device_index, device_name = select_audio_device()
        print(f"Using device [{device_index}]: {device_name}")

    # Start web caption server
    if args.port > 0:
        start_caption_server(args.port)
        print(f"Web captions: http://localhost:{args.port}")

    # Start Cloudflare tunnel
    tunnel_proc = None
    if args.tunnel:
        tunnel_proc = start_cloudflare_tunnel(args.tunnel, args.port)
        print(f"Cloudflare tunnel '{args.tunnel}' started → https://live.rctranslation.org")

    try:
        direction = "Korean→English" if args.lang == "ko" else "English→Korean"
        print(f"Translation mode: {direction}")
        run_session(api_key, device_index, anthropic_api_key, lang=args.lang)
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
