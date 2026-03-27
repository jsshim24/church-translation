import json
import os
import sys
import queue
import threading
import argparse
from typing import Optional

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
    "Preferred terms: 여러분 → everyone. "
    "If input contains both English and Korean, keep the English as-is and translate the Korean portions into English, "
    "even if the Korean repeats or paraphrases the English — always include both. "
    "If input is entirely in English, output it unchanged. "
    "Prefix output with the ISO 639-1 language code in brackets, e.g. [en]. "
    "Output ONLY the translation — no commentary or notes. "
    "Phrases may arrive as incomplete clauses. Translate only the words present — "
    "never infer or complete missing verbs or conclusions. "
    "If the fragment is too incomplete or garbled, output exactly: [SKIP]"
)


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
                     context: list[tuple[str, str]], model: str = DEFAULT_MODEL) -> str:
    messages = []
    for speaker, translation in context:
        messages.append({"role": "user", "content": speaker})
        messages.append({"role": "assistant", "content": translation})
    messages.append({"role": "user", "content": korean_text})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()


def run_session(api_key: str, device_index: int, anthropic_api_key: str) -> None:
    config = get_config(api_key)
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

                try:
                    translation = translate_phrase(client, combined, context)
                except Exception as e:
                    print(f"[Translation error: {e}]")
                    continue

                if translation == "[SKIP]":
                    pending_text = combined
                    continue

                pending_text = ""
                context.append((combined, translation))
                if len(context) > 5:
                    context.pop(0)

                translation = f"[Translation] {translation}"                
                print(translation)

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
    parser = argparse.ArgumentParser(description="Soniox real-time Korean→English translation from microphone")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (skip interactive selection)")
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

    run_session(api_key, device_index, anthropic_api_key)


if __name__ == "__main__":
    main()
