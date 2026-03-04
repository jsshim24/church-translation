#!/usr/bin/env python3
"""Korean-to-English live sermon translation pipeline.

Captures live audio from a USB audio interface connected to a church
soundboard, streams to Google Cloud Speech-to-Text V2 (Chirp 3),
translates each final phrase via Claude, and displays bilingual output.

Automatically chains STT sessions every 4 minutes at phrase boundaries
to support sermons of any length.

Usage:
    python translate.py
"""

import os
import sys
import json
import time
import queue
import argparse
import threading
import http.server
from pathlib import Path
from collections import deque
from urllib.parse import urlparse

import sounddevice as sd
from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
import anthropic

# ── Constants ──────────────────────────────────────────────────────────────────

REGION = "us"
CHUNK_SIZE = 12288  # ~12KB per streaming request (API limit is 15KB)
CONTEXT_WINDOW = 5  # number of past (Korean, English) pairs for Claude context

# Audio format constants (mono 16-bit 16kHz PCM)
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 32,000 bytes/sec

# Endless streaming constants
STREAM_DURATION_SECS = 240  # 4 min per session (safely under 5-min API limit)
STREAM_MAX_SECS = 280  # 4:40 safety hard limit (under 5-min API cap)

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
    """Load .env and extract project_id from credentials JSON."""
    load_dotenv(override=True)

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).exists():
        sys.exit("Error: GOOGLE_APPLICATION_CREDENTIALS not set or file not found")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        sys.exit("Error: ANTHROPIC_API_KEY not set in .env — please add your key")

    with open(creds_path) as f:
        creds_data = json.load(f)
    project_id = creds_data.get("project_id")
    if not project_id:
        sys.exit("Error: project_id not found in credentials JSON")

    model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    return {
        "creds_path": creds_path,
        "project_id": project_id,
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
    Full session audio is stored for precise bridging across session restarts,
    following Google's recommended endless streaming approach.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.audio_queue = queue.Queue()
        self.stream = None
        self.capturing = False
        self.total_bytes = 0
        self.start_time = None

        # Session audio storage for precise bridging
        self.audio_input = []           # all chunks for the current session
        self.last_audio_input = []      # chunks from the previous session
        self.new_stream = True          # flag: fresh session needs bridging

        # Timing state for bridging calculation (all in ms)
        self.result_end_time_ms = 0         # end time of the most recent result
        self.is_final_end_time_ms = 0       # end time of the most recent final result
        self.accepted_end_time_ms = 0       # end time of last non-[SKIP] result (for bridging)
        self.final_request_end_time_ms = 0  # carried across sessions for offset calc
        self.bridging_offset_ms = 0         # total time of replayed bridging audio
        self.last_queue_lag = 0.0           # seconds between callback enqueue and generator read
        self.live_start_wall = 0.0          # wall time when live audio starts (after bridging)

    def _callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs on audio thread."""
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        raw = bytes(indata)
        self.audio_queue.put((raw, time.monotonic()))  # timestamped for lag measurement
        self.audio_input.append(raw)
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

    def prepare_new_session(self):
        """Prepare state for a new STT session (called between sessions).

        Transfers current session audio to last_audio_input for bridging,
        carries forward timing state, and resets for the new session.

        Uses accepted_end_time_ms (last non-[SKIP] result) for bridging so
        that garbled end-of-session content gets replayed and re-recognized
        in the next session, rather than being permanently lost.
        """
        if self.accepted_end_time_ms > 0:
            self.final_request_end_time_ms = self.accepted_end_time_ms
        elif self.result_end_time_ms > 0:
            self.final_request_end_time_ms = self.is_final_end_time_ms
        self.result_end_time_ms = 0
        self.accepted_end_time_ms = 0
        self.last_audio_input = self.audio_input
        self.audio_input = []
        self.new_stream = True

    def elapsed(self) -> float:
        """Seconds since capture started."""
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time


# ── Google Cloud Speech-to-Text V2 ────────────────────────────────────────────


def create_stt_client():
    """Create Speech V2 client with regional endpoint."""
    return SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{REGION}-speech.googleapis.com",
        )
    )


def build_streaming_config(project_id: str, fast_endpointing: bool = True):
    """Build the first streaming request with recognition config for LINEAR16 PCM."""
    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=1,
        ),
        language_codes=["ko-KR", "en-US"],
        model="chirp_3",
    )

    streaming_kwargs = {"interim_results": False}
    if fast_endpointing:
        streaming_kwargs["endpointing_sensitivity"] = (
            cloud_speech.StreamingRecognitionFeatures.EndpointingSensitivity
            .ENDPOINTING_SENSITIVITY_SHORT
        )
    streaming_features = cloud_speech.StreamingRecognitionFeatures(**streaming_kwargs)

    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=streaming_features,
    )

    return cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{REGION}/recognizers/_",
        streaming_config=streaming_config,
    )


def live_request_generator(config_request, audio_capture: AudioCapture,
                           stop_event: threading.Event = None):
    """Yield config request, then bridging audio, then live mic chunks.

    On session restarts, replays audio from the previous session starting
    from the precise point after the last finalized result (Google's
    recommended endless streaming approach).

    Stops when stop_event is set (phrase-boundary restart) or when
    STREAM_MAX_SECS is reached (safety hard limit).
    """
    yield config_request

    # Replay bridging audio from previous session with moderate pacing.
    # This gives the STT context from the last finalized result onward,
    # so it can seamlessly continue transcription into the new live audio.
    # Paced at ~7.7x real-time (50ms per 384ms chunk) — fast enough to
    # avoid significant queue buildup, but slow enough to let live audio
    # accumulate in the queue so there's no gap at the bridging→live
    # transition (which would trigger premature STT finalization).
    if audio_capture.new_stream and audio_capture.last_audio_input:
        chunk_time_ms = CHUNK_SIZE / BYTES_PER_SECOND * 1000  # 384ms, always exact

        if chunk_time_ms != 0:
            # Clamp bridging offset to valid range
            if audio_capture.bridging_offset_ms < 0:
                audio_capture.bridging_offset_ms = 0
            if audio_capture.bridging_offset_ms > audio_capture.final_request_end_time_ms:
                audio_capture.bridging_offset_ms = audio_capture.final_request_end_time_ms

            # Calculate how many chunks from the start to skip (replay the rest).
            # Clamp to valid range to prevent negative bridging.
            chunks_from_ms = round(
                (audio_capture.final_request_end_time_ms - audio_capture.bridging_offset_ms)
                / chunk_time_ms
            )
            chunks_from_ms = max(0, min(chunks_from_ms, len(audio_capture.last_audio_input)))

            # Update bridging offset for next session
            audio_capture.bridging_offset_ms = round(
                (len(audio_capture.last_audio_input) - chunks_from_ms) * chunk_time_ms
            )

            # Log bridging details for diagnostics
            num_bridging = len(audio_capture.last_audio_input) - chunks_from_ms
            print(f"  [STT] bridging: {num_bridging} chunks "
                  f"({audio_capture.bridging_offset_ms}ms) "
                  f"| final_end={audio_capture.final_request_end_time_ms}ms",
                  file=sys.stderr)

            # Send bridging chunks with moderate pacing to maintain
            # continuous audio flow. Pure burst causes a gap before the
            # first live chunk arrives (~384ms), which triggers premature
            # STT finalization with fast endpointing. Pacing at ~7.7x
            # real-time (50ms vs 384ms per chunk) lets live audio
            # accumulate in the queue so there's no gap at the transition.
            for i in range(chunks_from_ms, len(audio_capture.last_audio_input)):
                yield cloud_speech.StreamingRecognizeRequest(
                    audio=audio_capture.last_audio_input[i]
                )
                time.sleep(0.05)  # 50ms pace → ~500ms for 10 chunks

        audio_capture.new_stream = False

    # Mark the moment live audio begins (after bridging).
    # Used for accurate STT latency measurement — sess_start is too
    # early because it includes gRPC setup + bridging pacing overhead.
    audio_capture.live_start_wall = time.monotonic()

    # Stream live audio until signaled to stop or safety limit
    session_start = time.monotonic()
    grace_start = None
    while audio_capture.capturing:
        # Graceful stop with grace period: after stop_event fires,
        # continue sending audio briefly so STT can properly finalize
        # the current utterance instead of truncating mid-word.
        if stop_event and stop_event.is_set():
            if grace_start is None:
                grace_start = time.monotonic()
            elif time.monotonic() - grace_start >= 1.5:
                break
        # Safety hard limit: force stop under API's 5-min cap
        if time.monotonic() - session_start >= STREAM_MAX_SECS:
            break
        try:
            chunk, enqueued_at = audio_capture.audio_queue.get(timeout=0.5)
            audio_capture.last_queue_lag = time.monotonic() - enqueued_at
            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
        except queue.Empty:
            continue


def live_stream_transcribe(client, config_request, audio_capture: AudioCapture,
                           stop_event: threading.Event = None):
    """Run one live STT session.

    Yields (transcript, result_end_ms, received_at) for each final result.
    Uses a background thread to timestamp results the instant gRPC delivers
    them, so the timestamp isn't delayed by downstream translation work.
    Also updates audio_capture timing state for precise session bridging.
    """
    result_q = queue.Queue()
    stream_error = [None]  # mutable container to pass error from thread

    responses = client.streaming_recognize(
        requests=live_request_generator(
            config_request, audio_capture, stop_event
        )
    )

    def _consume_responses():
        """Background thread: read gRPC responses and timestamp them."""
        try:
            for response in responses:
                if response.speech_event_type:
                    event_name = cloud_speech.StreamingRecognizeResponse.SpeechEventType(
                        response.speech_event_type
                    ).name
                    print(f"  [STT] speech_event: {event_name}", file=sys.stderr)

                if not response.results:
                    continue
                for result in response.results:
                    if result.is_final and result.alternatives:
                        transcript = result.alternatives[0].transcript.strip()
                        if transcript:
                            end_ms = 0
                            if result.result_end_offset:
                                end_ms = int(
                                    result.result_end_offset.total_seconds() * 1000
                                )

                            # Update timing state for bridging
                            audio_capture.result_end_time_ms = end_ms
                            audio_capture.is_final_end_time_ms = end_ms

                            # Timestamp the moment gRPC delivers the result
                            result_q.put((transcript, end_ms, time.monotonic()))
        except Exception as e:
            stream_error[0] = e
        finally:
            result_q.put(None)  # sentinel: stream ended

    thread = threading.Thread(target=_consume_responses, daemon=True)
    thread.start()

    while True:
        item = result_q.get()
        if item is None:
            break
        yield item  # (transcript, end_ms, received_at)

    thread.join(timeout=5)
    if stream_error[0]:
        raise stream_error[0]


def endless_live_transcribe(client, project_id: str, audio_capture: AudioCapture,
                            fast_endpointing: bool = True):
    """Chain live STT sessions indefinitely.

    Uses Google's recommended endless streaming approach: stores all session
    audio and tracks result_end_time for precise bridging. Sessions restart
    at phrase boundaries past STREAM_DURATION_SECS.

    Yields (transcript, absolute_elapsed_secs) tuples.
    """
    session_num = 0
    prev_session_transcripts = set()   # transcripts yielded in previous session
    current_session_transcripts = set()  # transcripts yielded in current session

    while audio_capture.capturing:
        session_num += 1
        stop_event = threading.Event()
        session_wall_start = time.monotonic()
        intentional_restart = False

        # Prepare bridging state (except for the first session)
        if session_num > 1:
            audio_capture.prepare_new_session()
            prev_session_transcripts = current_session_transcripts
            current_session_transcripts = set()

        config_request = build_streaming_config(project_id, fast_endpointing)

        try:
            for transcript, result_end_ms, stt_received in live_stream_transcribe(
                client, config_request, audio_capture, stop_event
            ):
                # Smart dedup for bridging results: use TEXT matching rather
                # than time-based filtering. Time-based filtering loses new
                # content (e.g. "리" completing "아무리"), but without any
                # dedup we get exact duplicates from re-recognition. Solution:
                # only skip bridging results whose text exactly matches a
                # transcript already yielded in the previous session.
                if result_end_ms <= audio_capture.bridging_offset_ms:
                    if transcript in prev_session_transcripts:
                        print(f"  [STT] dedup bridging: end={result_end_ms}ms "
                              f'"{transcript}"', file=sys.stderr)
                        continue  # exact duplicate of previously yielded text
                    print(f"  [STT] new bridging: end={result_end_ms}ms "
                          f"(offset={audio_capture.bridging_offset_ms}ms) "
                          f'"{transcript}"', file=sys.stderr)

                current_session_transcripts.add(transcript)
                yield transcript, audio_capture.elapsed(), {
                    'end_ms': result_end_ms,
                    'off_ms': audio_capture.bridging_offset_ms,
                    'session': session_num,
                    'stt_wall': stt_received,
                    'live_start': audio_capture.live_start_wall,
                    'q_lag': audio_capture.last_queue_lag,
                }

                # After yielding a complete phrase, check if session is due for restart.
                # Uses STT stream time (result_end_ms) rather than wall clock so that
                # translation delays don't trigger premature restarts. The generator's
                # STREAM_MAX_SECS safety limit prevents exceeding the API's 5-min cap.
                # Don't break — keep draining remaining STT responses so we don't
                # lose phrases that were finalized while the consumer was translating.
                if not intentional_restart:
                    stream_time_ms = result_end_ms - audio_capture.bridging_offset_ms
                    if stream_time_ms >= STREAM_DURATION_SECS * 1000:
                        stop_event.set()  # signal generator to stop sending audio
                        intentional_restart = True
                        print(f"  [STT] session {session_num} restart triggered "
                              f"| stream={stream_time_ms}ms", file=sys.stderr)

        except Exception as e:
            if not audio_capture.capturing:
                break
            print(f"  [STT] Error: {e} — reconnecting...", file=sys.stderr)

        # Log reason for session end
        session_secs = time.monotonic() - session_wall_start
        if intentional_restart:
            print(f"  [STT] session {session_num} ended at {format_timestamp(session_secs)} "
                  f"(scheduled restart)", file=sys.stderr)
        else:
            print(f"  [STT] session {session_num} ended at {format_timestamp(session_secs)} "
                  f"(API closed stream) — reconnecting seamlessly", file=sys.stderr)


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
    parser.add_argument("--standard-endpointing", action="store_true",
                        help="Use standard endpointing instead of fast (default: fast)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web caption server port (default: 8080, 0 to disable)")
    args = parser.parse_args()

    config = load_config()
    if args.haiku:
        model = "claude-haiku-4-5"
    else:
        model = args.model or config["model"]

    # Select audio input device
    if args.device is not None:
        dev = sd.query_devices(args.device)
        device_index, device_name = args.device, dev["name"]
    else:
        device_index, device_name = select_audio_device()
    print(f"\n  Using: [{device_index}] {device_name}\n")

    # Initialize clients
    stt_client = create_stt_client()
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
    if web_url:
        print(f"  Web:     {web_url}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()

    try:
        capture.start()

        for korean_text, elapsed_time, dbg in endless_live_transcribe(
            stt_client, config["project_id"], capture, not args.standard_endpointing
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
                print(f"  [SKIP] \"{korean_text}\" "
                      f"(end={dbg['end_ms']} off={dbg['off_ms']} "
                      f"s{dbg['session']})", file=sys.stderr)
                continue

            context.append((korean_text, english_text))
            phrase_count += 1

            # Mark this result as accepted for bridging. If the session ends
            # with a garbled force-finalized result ([SKIP]), bridging will
            # replay from this point, giving the next session a chance to
            # correctly recognize the truncated content.
            capture.accepted_end_time_ms = dbg['end_ms']

            # Push to web caption server (no debug info — clean output only)
            if web_url:
                _update_web_state(english_text, korean_text, ts)

            # STT processing delay: time from when the audio was spoken to
            # when the result was received. Uses live_start (set after bridging
            # completes) as the reference — sess_start was too early, inflating
            # the measurement by gRPC setup + bridging pacing overhead.
            stream_secs = max(0, (dbg['end_ms'] - dbg['off_ms']) / 1000)
            stt_secs = max(0, dbg['stt_wall'] - dbg['live_start'] - stream_secs)
            total_secs = stt_secs + tl_secs
            stream_ms = dbg['end_ms'] - dbg['off_ms']

            print(f"[{ts}] \U0001f1f0\U0001f1f7 {korean_text}")
            print(f"       \U0001f1fa\U0001f1f8 {english_text}")
            q_lag = dbg.get('q_lag', 0)
            print(f"       \u23f1 stt {stt_secs:.1f}s + tl {tl_secs:.1f}s = {total_secs:.1f}s"
                  f"  | q={q_lag:.1f}s stream={stream_ms}ms s{dbg['session']}")
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
