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
from pathlib import Path
from collections import deque

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
OVERLAP_SECS = 2  # overlap to recapture any audio buffered between phrase end and stream close
OVERLAP_BYTES = OVERLAP_SECS * BYTES_PER_SECOND

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
    A rolling overlap buffer retains the last few seconds for session bridging.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.audio_queue = queue.Queue()
        overlap_chunks = (OVERLAP_BYTES + CHUNK_SIZE - 1) // CHUNK_SIZE + 1
        self.overlap_buffer = deque(maxlen=overlap_chunks)
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
        self.overlap_buffer.append(raw)
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

    def get_overlap_audio(self) -> bytes:
        """Return the last OVERLAP_SECS of captured audio for session bridging."""
        data = b"".join(self.overlap_buffer)
        if len(data) > OVERLAP_BYTES:
            data = data[-OVERLAP_BYTES:]
        return data

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


def build_streaming_config(project_id: str, fast_endpointing: bool = False):
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
                           overlap_audio: bytes = b"",
                           stop_event: threading.Event = None):
    """Yield config request, then overlap audio, then live mic chunks.

    Stops when stop_event is set (phrase-boundary restart) or when
    STREAM_MAX_SECS is reached (safety hard limit).
    """
    yield config_request

    # Send overlap audio from previous session at real-time speed.
    # Pacing lets the STT build up acoustic model state naturally,
    # avoiding the garbled transcriptions that happen with burst delivery.
    if overlap_audio:
        chunk_duration = CHUNK_SIZE / BYTES_PER_SECOND  # ~0.384s per chunk
        offset = 0
        while offset < len(overlap_audio):
            chunk_end = min(offset + CHUNK_SIZE, len(overlap_audio))
            yield cloud_speech.StreamingRecognizeRequest(
                audio=overlap_audio[offset:chunk_end]
            )
            offset = chunk_end
            if offset < len(overlap_audio):
                time.sleep(chunk_duration)

    # Stream live audio until signaled to stop or safety limit
    session_start = time.monotonic()
    while audio_capture.capturing:
        # Graceful stop: signaled by consumer after a phrase boundary
        if stop_event and stop_event.is_set():
            break
        # Safety hard limit: force stop under API's 5-min cap
        if time.monotonic() - session_start >= STREAM_MAX_SECS:
            break
        try:
            chunk = audio_capture.audio_queue.get(timeout=0.5)
            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
        except queue.Empty:
            continue


def live_stream_transcribe(client, config_request, audio_capture: AudioCapture,
                           overlap_audio: bytes = b"",
                           stop_event: threading.Event = None):
    """Run one live STT session.

    Yields (transcript, result_end_offset_secs) for each final result.
    """
    responses = client.streaming_recognize(
        requests=live_request_generator(
            config_request, audio_capture, overlap_audio, stop_event
        )
    )

    for response in responses:
        # Log speech events (e.g. SPEECH_ACTIVITY_TIMEOUT) for diagnostics
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
                    end_offset_secs = 0.0
                    if result.result_end_offset:
                        end_offset_secs = result.result_end_offset.total_seconds()
                    yield transcript, end_offset_secs


def dedup_transcript(transcript, recent_transcripts, min_overlap_chars=1):
    """Check a new transcript against recently yielded ones for overlap.

    Returns:
        The transcript to yield (possibly trimmed), or None to skip entirely.
    """
    for prev in recent_transcripts:
        # Exact match — skip
        if transcript == prev:
            return None
        # New is fully contained within a recent phrase — skip fragment
        if transcript in prev:
            return None

    # Check for suffix→prefix overlap at session boundary:
    # e.g. prev="...하나님이 인도해 주시는 그 과정 가운데"
    #      new= "그 과정 가운데 얼마나 많이들"
    # → yield only "얼마나 많이들"
    for prev in recent_transcripts:
        max_check = min(len(prev), len(transcript))
        for overlap_len in range(max_check, min_overlap_chars - 1, -1):
            if transcript[:overlap_len] == prev[-overlap_len:]:
                remainder = transcript[overlap_len:].strip()
                return remainder if len(remainder) > 3 else None

    return transcript


def endless_live_transcribe(client, project_id: str, audio_capture: AudioCapture,
                            fast_endpointing: bool = False):
    """Chain live STT sessions indefinitely.

    Sessions restart at phrase boundaries: after yielding a final result
    past STREAM_DURATION_SECS, the consumer signals the generator to stop
    via a threading.Event, so audio is never cut mid-utterance.

    Yields (transcript, absolute_elapsed_secs) tuples.
    """
    session_num = 0
    recent_transcripts = deque(maxlen=3)

    while audio_capture.capturing:
        session_num += 1
        stop_event = threading.Event()
        session_wall_start = time.monotonic()
        intentional_restart = False

        # Get overlap audio for bridging (empty for first session)
        overlap_audio = audio_capture.get_overlap_audio() if session_num > 1 else b""

        config_request = build_streaming_config(project_id, fast_endpointing)

        try:
            for transcript, result_end_offset_secs in live_stream_transcribe(
                client, config_request, audio_capture, overlap_audio, stop_event
            ):
                # Dedup: skip or trim overlapping content from session boundary
                cleaned = dedup_transcript(transcript, recent_transcripts)
                if cleaned is None:
                    continue

                absolute_time = audio_capture.elapsed()
                recent_transcripts.append(transcript)  # store original for future dedup
                yield cleaned, absolute_time

                # After yielding a complete phrase, check if session is due for restart
                session_elapsed = time.monotonic() - session_wall_start
                if session_elapsed >= STREAM_DURATION_SECS:
                    stop_event.set()  # signal generator to stop sending audio
                    intentional_restart = True
                    break  # start new session; overlap will recapture any lost tail

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
    parser.add_argument("--fast-endpointing", action="store_true",
                        help="Use SHORT endpointing for faster phrase finalization")
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

    # Start audio capture
    capture = AudioCapture(device_index)
    phrase_count = 0

    print("=" * 50)
    print("  Korean \u2192 English Sermon Translation")
    print(f"  Model:   {model}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()

    try:
        capture.start()

        for korean_text, elapsed_time in endless_live_transcribe(
            stt_client, config["project_id"], capture, args.fast_endpointing
        ):
            ts = format_timestamp(elapsed_time)

            try:
                english_text = translate_phrase(translator, korean_text, context, model)
            except Exception as e:
                english_text = f"[Translation error: {e}]"

            # Skip garbled/unintelligible transcriptions
            if english_text.strip() == "[SKIP]":
                continue

            context.append((korean_text, english_text))
            phrase_count += 1

            print(f"[{ts}] \U0001f1f0\U0001f1f7 {korean_text}")
            print(f"       \U0001f1fa\U0001f1f8 {english_text}")
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
