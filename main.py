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
import importlib
from typing import Callable, Optional
from urllib.parse import urlparse

# Suppress noisy thread exception tracebacks on Ctrl+C.
threading.excepthook = lambda args: None

import sounddevice as sd
from dotenv import load_dotenv


# ── Audio constants ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK_FRAMES = 1600  # 100ms at 16kHz


# ── Shared language constants ─────────────────────────────────────────────────

LANG_NAMES = {"ko": "Korean", "en": "English", "es": "Spanish"}

# Languages Soniox should hint for each --source. Strict hints.
SOURCE_LANGS = {
    "ko":    ["ko", "en"],
    "en":    ["en"],
    "es":    ["es", "en"],
    "multi": ["ko", "en", "es"],
}

# Fallback tag when render_tokens emits text with no [xx] language prefix
# (edge case: language_identification is enabled, so this rarely fires).
PRIMARY_SRC = {"ko": "ko", "en": "en", "es": "es", "multi": "en"}

# Matches [xx] tags render_tokens emits; stripped before Claude sees the text
# so embedded tags can't masquerade as the desired output prefix.
_LANG_TAG_RE = re.compile(r"\[[a-z]{2}\]\s*")


# ── Prompt pieces (Claude zone) ───────────────────────────────────────────────

FILLER_CLAUSE_BY_LANG = {
    "ko": "Korean hesitation fillers (아, 어)",
    "en": "English hesitation fillers (uh, um, like, you know, so, I mean)",
    "es": "Spanish hesitation fillers (eh, este, pues, o sea, bueno)",
}

BIBLE_BY_TARGET = {
    "en": "English Standard Version (ESV)",
    "ko": "New Korean Revised Version (개역개정)",
    "es": "Reina-Valera 1960 (RVR1960)",
}

REGISTER_BY_TARGET = {
    "ko": " Use natural, formal polite speech (합쇼체/해요체) as is standard for sermon translation.",
    "en": "",
    "es": "",
}

# Proper-noun / address preferences, keyed by (source, target).
# Religious nouns (하나님, Dios, etc.) live in the Soniox terms list; Claude
# translates them naturally without hints. This table is for proper nouns and
# address-form overrides specific to a direction pair.
TERM_PREFS_BY_PAIR = {
    ("ko", "en"):    "여러분 → everyone; 정목사 → Pastor Chung.",
    ("ko", "es"):    "여러분 → todos; 정목사 → Pastor Chung.",
    ("en", "ko"):    "",
    ("en", "es"):    "",
    ("es", "en"):    "",
    ("es", "ko"):    "",
    # multi → any: use ko-specific prefs since 정목사 only appears in Korean speech.
    ("multi", "en"): "여러분 → everyone; 정목사 → Pastor Chung.",
    ("multi", "es"): "여러분 → todos; 정목사 → Pastor Chung.",
    ("multi", "ko"): "",
}

SOURCE_COMPOSITION = {
    "ko":    "Korean (with occasional English)",
    "en":    "English",
    "es":    "Spanish (with occasional English)",
    "multi": "mixed Korean, English, and Spanish",
}

OUTLINE_WRAPPER = (
    "\n\n--- SERMON OUTLINE (CONTEXT ONLY) ---\n"
    "The following outline is provided for logical flow and topical context only. "
    "Do NOT use it to infer, complete, or reshape what the speaker actually says. "
    "If the spoken phrase contradicts, diverges from, or rhetorically opposes the "
    "outline, translate what is said literally. The outline is background knowledge, "
    "not a script.\n"
    "--- OUTLINE BEGINS ---\n"
    "{outline}\n"
    "--- OUTLINE ENDS ---"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_outline(path: str) -> str:
    """Read a UTF-8 sermon outline file. Fail loudly on any issue."""
    if not os.path.exists(path):
        raise RuntimeError(f"Outline file not found: {path}")
    if os.path.isdir(path):
        raise RuntimeError(f"Outline path is a directory, expected a file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Outline file is not valid UTF-8: {path} ({e})")
    except OSError as e:
        raise RuntimeError(f"Could not read outline file: {path} ({e})")
    if not text.strip():
        raise RuntimeError(f"Outline file is empty: {path}")
    return text.strip()


def build_prompt(source: str, target: str) -> str:
    """Assemble the live-translation system prompt for a (source, target) pair.

    Composed from piece dicts above — no per-combo hardcoded strings.
    """
    langs_present = SOURCE_LANGS[source]
    fillers = " and ".join(FILLER_CLAUSE_BY_LANG[l] for l in langs_present)
    tname = LANG_NAMES[target]
    same_lang_clause = (
        f"For segments already in {tname}, keep them unchanged. "
        f"Translate segments in other languages into {tname}, even if they repeat "
        "or paraphrase already-translated content — always include both."
    )
    prefs = TERM_PREFS_BY_PAIR[(source, target)]
    prefs_clause = f"Preferred terms: {prefs} " if prefs else ""
    return (
        f"You are a live translation assistant for a {SOURCE_COMPOSITION[source]} church sermon. "
        "You receive a rolling context window of recent phrases; prior translations are provided as context. "
        f"Translate ONLY the latest phrase into {tname}. "
        f"Drop hesitation fillers like {fillers}. "
        f"{same_lang_clause} "
        f"{prefs_clause}"
        "Output ONLY the translation — no commentary, notes, or language code prefix. "
        "Phrases may arrive as incomplete clauses. Translate only the words present — "
        "never infer or complete missing verbs or conclusions. "
        "If the fragment is too incomplete or garbled, output exactly: [SKIP] "
        "Short fragments that lack a verb or predicate and cannot stand alone as a meaningful sentence "
        "should be [SKIP]ped — they will be prepended to the next phrase automatically. "
        f"When quoting or referencing Bible passages, use the {BIBLE_BY_TARGET[target]} for {tname}."
        f"{REGISTER_BY_TARGET[target]}"
    )


# ── Web State ─────────────────────────────────────────────────────────────────

_web_state = {"lines": [], "updated": 0}
_web_lock = threading.Lock()
_default_target_lang = "en"  # set in main() from the first --target; injected into HTML


def _update_web_state(kind: str, lang: str, text: str):
    """kind='transcription' or 'translation', lang='en'/'ko'/'es'/…"""
    with _web_lock:
        _web_state["lines"].append({"kind": kind, "lang": lang, "text": text})
        _web_state["updated"] = time.time()


def _get_web_state_json() -> bytes:
    with _web_lock:
        return json.dumps(_web_state).encode()


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


# ── HTML Template ─────────────────────────────────────────────────────────────

CAPTION_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  html, body {
    width: 100%;
    height: 100%;
    background: transparent;
    overflow: hidden;
  }

  #container {
    width: 100%;
    height: 100%;
    overflow-y: auto;
    scroll-behavior: smooth;
    scrollbar-width: none;
    -ms-overflow-style: none;
  }

  #container::-webkit-scrollbar {
    display: none;
  }

  .multi-line-block {
    animation: fadeIn 0.25s ease-out;
    margin-bottom: 0.45em;
    opacity: 0.45;
    transition: opacity 0.4s ease;
  }

  .multi-line-block:last-child {
    opacity: 1;
  }

  .lang-line {
    display: block;
    width: 100%;
  }

  .line-item {
    animation: fadeIn 0.25s ease-out;
    opacity: 0.45;
    transition: opacity 0.4s ease;
  }

  .line-item:last-child {
    opacity: 1;
  }

  .span-item {
    /* paragraph mode */
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  #waiting-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.82);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 18px;
    z-index: 999;
    transition: opacity 0.4s ease;
  }

  #waiting-overlay.hidden {
    opacity: 0;
    pointer-events: none;
  }

  #waiting-overlay .spinner {
    width: 36px;
    height: 36px;
    border: 3px solid rgba(255,255,255,0.15);
    border-top-color: rgba(255,255,255,0.8);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  #waiting-overlay .msg {
    color: rgba(255,255,255,0.85);
    font-family: system-ui, sans-serif;
    font-size: 18px;
  }

  #waiting-overlay .dismiss {
    color: rgba(255,255,255,0.35);
    font-family: system-ui, sans-serif;
    font-size: 13px;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.15);
    padding: 6px 18px;
    border-radius: 20px;
    background: none;
    transition: all 0.2s;
  }

  #waiting-overlay .dismiss:hover {
    color: rgba(255,255,255,0.7);
    border-color: rgba(255,255,255,0.35);
  }
</style>
</head><body>

<div id="waiting-overlay">
  <div class="spinner"></div>
  <div class="msg">Waiting for transcription…</div>
  <button class="dismiss" onclick="dismissOverlay()">Dismiss</button>
</div>

<div id="container">
  <div id="lines"></div>
</div>

<script>
(function() {
  const params = new URLSearchParams(window.location.search);

  // Modes
  const DEFAULT_TARGET_LANG = "__DEFAULT_TARGET_LANG__";
  const mode = params.get('mode') || 'translation';
  const display = params.get('display') || 'line';

  // Multi-language support:
  // ?langs=en,ko,es
  // Each language renders on its own line inside the same caption block.
  //
  // Backwards compatibility:
  // ?lang=en still works.
  const langsParam = params.get('langs');
  const singleLang = params.get('lang');

  let langs = [];

  if (langsParam) {
    langs = langsParam
      .split(',')
      .map(x => x.trim())
      .filter(Boolean);
  } else if (singleLang) {
    langs = [singleLang];
  } else if (mode === 'translation') {
    langs = [DEFAULT_TARGET_LANG];
  }

  const multiLangMode = langs.length > 1;

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
  const bgColor  = params.get('bgColor') || '#000';
  const padding  = params.get('padding') || '20';

  const maxLines = Math.min(
    params.get('maxLines') ? parseInt(params.get('maxLines')) : 0,
    200
  );

  // Load Google Font if specified
  if (googleFont) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href =
      'https://fonts.googleapis.com/css2?family='
      + encodeURIComponent(googleFont)
      + '&display=swap';

    document.head.appendChild(link);
  }

  const container = document.getElementById('container');
  const linesDiv  = document.getElementById('lines');
  const overlay   = document.getElementById('waiting-overlay');

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
    'white-space:pre-wrap'
  ].join(';');

  let lastCount = 0;
  let lastUpdated = 0;

  let overlayDismissed = false;
  let hasReceivedData = false;

  const DOM_CAP = 200;

  const FAST_MS = 150;
  const MAX_MS  = 1000;
  const GROWTH  = 1.5;

  let pollDelay = FAST_MS;

  // Multi-language grouping state
  // Groups buffer lines until all expected langs arrive or a timeout fires,
  // then render them in LANG_ORDER sequence.
  const LANG_ORDER = ['ko', 'en', 'es'];  // fixed display order
  const GROUP_WINDOW_MS = 6000;  // wait up to 6s for slow translations
  const recentGroups = [];

  function getExpectedLangs() {
    // langs array holds the active target languages from URL params
    return langs.length > 0
      ? LANG_ORDER.filter(l => langs.includes(l))
      : LANG_ORDER;
  }

  function flushGroup(group) {
    if (group.flushed) return;
    group.flushed = true;
    clearTimeout(group.timer);
    const ordered = getExpectedLangs();
    ordered.forEach(lang => {
      if (group.buffer[lang] !== undefined) {
        const langLine = makeLangLine(lang, group.buffer[lang]);
        group.el.appendChild(langLine);
      }
    });
  }

  function appendMultiLanguageLine(line) {
    const now = Date.now();
    const expected = getExpectedLangs();

    // Find a non-flushed recent group that doesn't already have this lang
    let group = null;
    for (let i = recentGroups.length - 1; i >= 0; i--) {
      const g = recentGroups[i];
      if (!g.flushed && (now - g.ts) <= GROUP_WINDOW_MS && g.buffer[line.lang] === undefined) {
        group = g;
        break;
      }
    }

    // Create new group if needed
    if (!group) {
      const wrapper = document.createElement('div');
      wrapper.className = 'multi-line-block';
      linesDiv.appendChild(wrapper);
      group = { ts: now, el: wrapper, buffer: {}, flushed: false, timer: null };
      recentGroups.push(group);
      while (recentGroups.length > 50) recentGroups.shift();
    }

    group.buffer[line.lang] = line.text;

    // Flush immediately if all expected langs are present
    const have = expected.filter(l => group.buffer[l] !== undefined);
    if (have.length >= expected.length) {
      flushGroup(group);
      return;
    }

    // Otherwise set/reset a deadline timer so we don't wait forever
    clearTimeout(group.timer);
    group.timer = setTimeout(() => flushGroup(group), GROUP_WINDOW_MS);
  }

  window.dismissOverlay = function() {
    overlayDismissed = true;
    overlay.classList.add('hidden');
  };

  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) {
      window.dismissOverlay();
    }
  });

  function scrollToBottom() {
    requestAnimationFrame(function() {
      container.scrollTop = container.scrollHeight;
    });
  }

  function langAllowed(lang) {
    if (langs.length === 0) return true;
    return langs.includes(lang);
  }

  function makeLangLine(lang, text) {
    const div = document.createElement('div');
    div.className = 'lang-line';
    div.dataset.lang = lang;
    div.textContent = text;
    return div;
  }

  function appendSingleLine(text) {
    if (display === 'paragraph') {
      const span = document.createElement('span');
      span.className = 'span-item';
      span.textContent = text + ' ';
      linesDiv.appendChild(span);
    } else {
      const div = document.createElement('div');
      div.className = 'line-item';
      div.textContent = text;
      linesDiv.appendChild(div);
    }
  }

  function trimDom() {
    const selector = multiLangMode
      ? '.multi-line-block'
      : (display === 'paragraph' ? '.span-item' : '.line-item');

    const items = linesDiv.querySelectorAll(selector);

    const limit = maxLines > 0 ? maxLines : DOM_CAP;
    const toRemove = items.length - limit;

    for (let i = 0; i < toRemove; i++) {
      items[i].remove();
    }
  }

  async function poll() {
    try {
      const resp = await fetch('/api/latest');

      if (!resp.ok) {
        throw new Error('HTTP ' + resp.status);
      }

      const data = await resp.json();

      pollDelay = FAST_MS;

      if (data.updated === lastUpdated) {
        return;
      }

      lastUpdated = data.updated;

      const allLines = data.lines;
      const newLines = allLines.slice(lastCount);

      lastCount = allLines.length;

      let appended = false;

      for (const line of newLines) {
        if (line.kind !== mode) continue;
        if (!langAllowed(line.lang)) continue;

        if (!overlayDismissed && !hasReceivedData) {
          hasReceivedData = true;
          window.dismissOverlay();
        }

        if (multiLangMode) {
          appendMultiLanguageLine(line);
        } else {
          appendSingleLine(line.text);
        }

        appended = true;
      }

      trimDom();

      if (appended) {
        scrollToBottom();
      }

    } catch (e) {
      pollDelay = Math.min(pollDelay * GROWTH, MAX_MS);

    } finally {
      setTimeout(poll, pollDelay);
    }
  }

  poll();
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
                safe_lang = _default_target_lang if _default_target_lang in ("ko", "en", "es") else "en"
                html = CAPTION_HTML.replace("__DEFAULT_TARGET_LANG__", safe_lang).encode()
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


# ── Audio ─────────────────────────────────────────────────────────────────────


def select_audio_device():
    """List available input devices and prompt the user to select one."""
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


def iter_audio_chunks(device_index: int, sample_rate: int, chunk_frames: int,
                      stop_event: threading.Event):
    """Yield raw int16 PCM chunks from the mic until stop_event fires.

    Pure mic-capture; transport is the caller's responsibility.
    """
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=chunk_frames,
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
                yield chunk
    except Exception:
        pass


# ── Translation Worker ────────────────────────────────────────────────────────


class TranslationWorker:
    """LLM-agnostic queue/[SKIP]/rolling-context shell for one target language.

    State per worker: rolling context window (own), `[SKIP]` pending-text
    buffer (own), input queue (own), and a backend (Claude/Gemini/etc.) that
    owns the actual translation API call, cache, and keepalive. The only
    external seam is the `on_translation(target, text)` callback passed at
    construction — the callee decides how to surface the output (e.g. push
    to web state).
    """

    def __init__(self, backend, source: str, stop_event: threading.Event,
                 on_translation: Callable[[str, str], None]):
        self.backend = backend
        self.source = source
        self.stop_event = stop_event
        self.on_translation = on_translation
        self.inbox: queue.Queue[str] = queue.Queue()
        self.context: list[tuple[str, str]] = []   # last 5 (source, translation)
        self.pending_text: str = ""
        self._run_thread: Optional[threading.Thread] = None

    def warm(self) -> None:
        self.backend.warmup()

    def start(self) -> None:
        self._run_thread = threading.Thread(target=self._run, daemon=True)
        self._run_thread.start()
        self.backend.start_keepalive(self.stop_event)

    def enqueue(self, source_text: str) -> None:
        self.inbox.put(source_text)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                src = self.inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            clean_src = _LANG_TAG_RE.sub("", src).strip()
            if not clean_src:
                continue
            combined = (self.pending_text + " " + clean_src).strip() if self.pending_text else clean_src
            try:
                out = self.backend.translate(self.context, combined)
            except Exception as e:
                print(f"[{self.backend.target} translation error: {e}]", file=sys.stderr)
                self.backend.mark_activity()
                continue
            if "[SKIP]" in out:
                self.pending_text = combined
                continue
            self.pending_text = ""
            self.context.append((combined, out))
            if len(self.context) > 5:
                self.context.pop(0)
            prefixed = f"[{self.backend.target}] {out}"
            print(f"[Translation:{self.backend.target}] {prefixed}")
            self.on_translation(self.backend.target, prefixed)


# ── Orchestration ─────────────────────────────────────────────────────────────


def _build_workers(client, source: str, targets: list[str],
                   outline: Optional[str], stop_event: threading.Event,
                   model: str, backend_cls) -> list[TranslationWorker]:
    """Construct one TranslationWorker per target. Per-worker cache eligibility
    is decided independently inside backend_cls.from_outline."""
    workers: list[TranslationWorker] = []
    for t in targets:
        backend = backend_cls.from_outline(client, source, t, outline, model)
        w = TranslationWorker(
            backend=backend,
            source=source,
            stop_event=stop_event,
            on_translation=lambda tgt, txt: _push_to_web(
                "translation", txt, fallback_lang=tgt
            ),
        )
        workers.append(w)
    return workers


def run_session(api_key: str, device_index: int, anthropic_api_key: str,
                source: str, targets: list[str], outline: Optional[str],
                transcriber_cls, backend_cls, make_client_fn, model: str) -> None:
    client = make_client_fn(anthropic_api_key)
    stop_event = threading.Event()

    workers = _build_workers(client, source, targets, outline, stop_event,
                             model, backend_cls)

    # Warm each cached worker's ephemeral cache before opening the mic.
    for w in workers:
        w.warm()
    for w in workers:
        w.start()

    transcription_fallback = PRIMARY_SRC[source]

    def on_phrase(text: str) -> None:
        _push_to_web("transcription", text, fallback_lang=transcription_fallback)
        # Fan-out: enqueue the raw source phrase to every target worker.
        # Each worker applies its own [SKIP] logic and rolling context.
        for w in workers:
            w.enqueue(text)

    transcriber = transcriber_cls(source=source, api_key=api_key)
    try:
        transcriber.run(device_index, on_phrase, stop_event)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_event.set()


def _parse_and_validate_targets(source: str, target_arg: Optional[str]) -> list[str]:
    """Resolve --target against --source. Returns the validated target list in
    the order specified by the user (first target becomes the web default)."""
    ALL = {"ko", "en", "es"}
    DEFAULTS = {"ko": "en", "multi": "ko,en,es"}

    if target_arg is None:
        if source not in DEFAULTS:
            sys.exit(f"--target is required for --source {source}")
        target_arg = DEFAULTS[source]

    targets = [t.strip() for t in target_arg.split(",") if t.strip()]

    if source == "multi":
        if set(targets) != ALL or len(targets) != 3:
            sys.exit("--source multi requires --target ko,en,es (all three)")
        return targets

    allowed = ALL - {source}
    if source in targets:
        sys.exit(f"--target cannot include --source ({source})")
    if not targets or not set(targets).issubset(allowed):
        sys.exit(f"--target must be a non-empty subset of {sorted(allowed)}")
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Soniox real-time sermon translation from microphone"
    )
    parser.add_argument(
        "--source", choices=["ko", "en", "es", "multi"], default="ko",
        help="Source language: ko (Korean + English), en (English only), "
             "es (Spanish + English), multi (Korean + English + Spanish). Default: ko.",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Comma-separated translation targets (e.g. 'en' or 'ko,es'). "
             "Defaults to 'en' when --source ko, and 'ko,en,es' when --source multi. "
             "Required for --source en or --source es.",
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (skip interactive selection)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web caption server port (default: 8080, 0 to disable)")
    parser.add_argument("--tunnel", type=str, default="church-live",
                        help="Cloudflare tunnel name (default: church-live). "
                             "Use --no-tunnel to skip.")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="Skip starting the Cloudflare tunnel.")
    parser.add_argument("--outline", type=str, default=None,
                        help="Path to a UTF-8 .txt sermon outline for context. "
                             "Enables per-target prompt caching when the combined "
                             "system prompt exceeds 1024 tokens.")
    parser.add_argument("--transcriber", choices=["soniox"], default="soniox",
                        help="Transcription backend (default: soniox). "
                             "Loads transcribe_<name>.py at startup.")
    parser.add_argument("--translator", choices=["claude"], default="claude",
                        help="Translation backend (default: claude). "
                             "Loads translate_<name>.py at startup.")
    args = parser.parse_args()

    try:
        tx_mod = importlib.import_module(f"transcribe_{args.transcriber}")
        tl_mod = importlib.import_module(f"translate_{args.translator}")
    except ModuleNotFoundError as e:
        sys.exit(f"Backend module not found: {e.name}")

    targets = _parse_and_validate_targets(args.source, args.target)

    outline_text: Optional[str] = None
    if args.outline is not None:
        outline_text = load_outline(args.outline)
        print(f"Loaded outline: {args.outline} ({len(outline_text)} chars)")

    global _default_target_lang
    _default_target_lang = targets[0]

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
        device_index = None
        print("Using system default audio input device.")

    if args.port > 0:
        start_caption_server(args.port)
        print(f"Web captions: http://localhost:{args.port}")

    tunnel_proc = None
    if args.tunnel and not args.no_tunnel:
        tunnel_proc = start_cloudflare_tunnel(args.tunnel, args.port)
        print(f"Cloudflare tunnel '{args.tunnel}' started → https://live.rctranslation.org")

    try:
        print(f"Translation mode: {args.source} → {', '.join(targets)}")
        run_session(api_key, device_index, anthropic_api_key,
                    source=args.source, targets=targets, outline=outline_text,
                    transcriber_cls=tx_mod.Transcriber, backend_cls=tl_mod.Backend,
                    make_client_fn=tl_mod.make_client, model=tl_mod.DEFAULT_MODEL)
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
