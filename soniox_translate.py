#!/usr/bin/env python3
"""Native Soniox transcription + translation pipeline for church sermons.

Uses Soniox's built-in real-time translation (Korean → English) with
church-specific context, eliminating the need for a separate Claude
translation step.

Usage:
    python soniox_translate.py
    python soniox_translate.py --device 2
"""

import json
import os
import queue
import sys
import threading
import argparse

import sounddevice as sd
from dotenv import load_dotenv
from websockets import ConnectionClosedOK
from websockets.sync.client import connect

SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SAMPLE_RATE = 16000
CHUNK_SIZE = 3200  # 100ms of 16-bit mono @ 16kHz


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


def get_config(api_key: str) -> dict:
    """Build Soniox config with church sermon context and one-way translation."""
    return {
        "api_key": api_key,
        "model": "stt-rt-v4",
        # Korean and English only — strict mode prevents hallucinating other languages
        "language_hints": ["ko", "en"],
        "language_hints_strict": True,
        "enable_language_identification": True,
        "enable_speaker_diarization": True,
        "enable_endpoint_detection": True,
        # Raw PCM from the microphone
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        # One-way translation: all speech → English
        "translation": {
            "type": "one_way",
            "target_language": "en",
        },
        # Church sermon domain context
        "context": {
            "general": [
                {"key": "domain", "value": "Religion"},
                {"key": "topic", "value": "Korean Christian church sermon, live translation"},
                {"key": "speaker", "value": "Korean pastor preaching to congregation"},
            ],
            "text": (
                "Live Korean church sermon being translated to English for the congregation. "
                "The pastor speaks primarily in Korean with occasional English words or phrases mixed in. "
                "Common sermon topics include Bible stories, faith, prayer, salvation, and Christian living. "
                "The speaker uses a conversational, rhetorical preaching style with rhetorical questions, "
                "repetition, and direct address to the congregation."
            ),
            "terms": [
                "은혜", "말씀", "성령", "구원", "화목", "긍휼", "성도",
                "하나님", "예수님", "십자가", "부활", "회개", "찬양",
                "예배", "기도", "아멘", "할렐루야", "복음", "천국",
                "요셉", "모세", "다윗", "아브라함", "바울",
                "정목사", "그루터기 교회",
            ],
            "translation_terms": [
                {"source": "은혜", "target": "grace"},
                {"source": "말씀", "target": "the Word"},
                {"source": "성령", "target": "the Holy Spirit"},
                {"source": "구원", "target": "salvation"},
                {"source": "화목", "target": "reconciliation"},
                {"source": "긍휼", "target": "mercy"},
                {"source": "성도", "target": "saint"},
                {"source": "정목사", "target": "Pastor Chung"},
                {"source": "그루터기 교회", "target": "Remnant Church"},
                {"source": "하나님", "target": "God"},
                {"source": "예수님", "target": "Jesus"},
                {"source": "십자가", "target": "the cross"},
                {"source": "부활", "target": "resurrection"},
                {"source": "회개", "target": "repentance"},
                {"source": "찬양", "target": "praise"},
                {"source": "예배", "target": "worship"},
                {"source": "복음", "target": "the gospel"},
                {"source": "천국", "target": "the kingdom of heaven"},
            ],
        },
    }


def stream_audio(audio_queue: queue.Queue, ws, running: list) -> None:
    """Drain audio queue and send chunks to the websocket."""
    try:
        while running[0]:
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            ws.send(chunk)
        # Empty string signals end-of-audio to the server
        ws.send("")
    except Exception:
        pass


def run_session(api_key: str, device_index: int) -> None:
    """Connect to Soniox, stream audio, and display transcript."""
    config = get_config(api_key)

    # Set up audio capture
    audio_queue: queue.Queue = queue.Queue()
    running = [True]

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    blocksize = CHUNK_SIZE // 2
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=blocksize,
        device=device_index,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    )
    stream.start()

    print("Connecting to Soniox...")
    with connect(SONIOX_WEBSOCKET_URL) as ws:
        ws.send(json.dumps(config))

        threading.Thread(
            target=stream_audio,
            args=(audio_queue, ws, running),
            daemon=True,
        ).start()

        print("Session started. Speak into your microphone. Ctrl+C to stop.\n")

        final_tokens: list[dict] = []

        try:
            while True:
                message = ws.recv()
                res = json.loads(message)

                if res.get("error_code") is not None:
                    print(f"Error: {res['error_code']} - {res['error_message']}")
                    break

                non_final_tokens: list[dict] = []
                for token in res.get("tokens", []):
                    if token.get("text"):
                        if token.get("is_final"):
                            final_tokens.append(token)
                        else:
                            non_final_tokens.append(token)

                # # Print only final translation tokens
                # for token in res.get("tokens", []):
                #     text = token.get("text", "")
                #     if not text or text == "<end>":
                #         continue
                #     if token.get("is_final") and token.get("translation_status") == "translation":
                #         print(text, end="", flush=True)

                # Print final transcription tokens, one line per burst
                burst = []
                for token in res.get("tokens", []):
                    text = token.get("text", "")
                    if not text or text == "<end>":
                        continue
                    if token.get("is_final") and token.get("translation_status") != "translation":
                        burst.append(text)
                if burst:
                    print("".join(burst), flush=True)

                if res.get("finished"):
                    print("Session finished.")

        except ConnectionClosedOK:
            pass
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            running[0] = False
            stream.stop()
            stream.close()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Soniox native Korean→English sermon translation"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (skip interactive selection)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SONIOX_API_KEY")
    if not api_key:
        sys.exit("Error: SONIOX_API_KEY not set. Add it to .env or export it.")

    if args.device is not None:
        device_index = args.device
        dev = sd.query_devices(device_index)
        print(f"Using audio device [{device_index}]: {dev['name']}")
    else:
        device_index, _ = select_audio_device()

    run_session(api_key, device_index)


if __name__ == "__main__":
    main()
