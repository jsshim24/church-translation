import json
import threading
from typing import Callable, Optional

from websockets import ConnectionClosedOK
from websockets.sync.client import connect

from main import SAMPLE_RATE, CHUNK_FRAMES, SOURCE_LANGS, iter_audio_chunks


# ── Soniox constants ──────────────────────────────────────────────────────────

SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


# ── Soniox Config ─────────────────────────────────────────────────────────────

TERMS_KO = ["하나님", "예수님", "성령", "아멘", "목사님", "집사님", "장로님", "권사님", "전도사님"]
TERMS_EN = ["God", "Jesus", "Holy Spirit", "amen", "Pastor"]
TERMS_ES = ["Dios", "Jesús", "Cristo", "Espíritu Santo", "amén", "Pastor", "hermano", "hermana", "iglesia"]

SOURCE_TERMS = {
    "ko":    TERMS_KO + TERMS_EN,
    "en":    TERMS_EN,
    "es":    TERMS_ES + TERMS_EN,
    "multi": TERMS_KO + TERMS_EN + TERMS_ES,
}

SOURCE_CONTEXT = {
    "ko":    ("Korean church sermon",
              "Live Korean church sermon with occasional English, with a pastor preaching to the congregation."),
    "en":    ("English church sermon",
              "Live English church sermon with a pastor preaching to the congregation."),
    "es":    ("Spanish church sermon",
              "Live Spanish church sermon with occasional English, with a pastor preaching to the congregation."),
    "multi": ("Multilingual church sermon",
              "Live multilingual church sermon in Korean, English, and Spanish, with a pastor preaching to the congregation."),
}


def build_soniox_config(source: str, api_key: str) -> dict:
    """Build the initial-frame JSON for the Soniox STT websocket.

    `translation.target_language` is fixed at `zh` across all sources — Soniox
    translation tokens are used only as a phrase-boundary gating signal; the
    translated text is discarded. Pivoting to an unused language keeps this
    config source-agnostic.
    """
    topic, text = SOURCE_CONTEXT[source]
    return {
        "api_key": api_key,
        "model": "stt-rt-v4",
        "language_hints": SOURCE_LANGS[source],
        "language_hints_strict": True,
        "enable_language_identification": True,
        "enable_endpoint_detection": True,
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "translation": {
            "type": "one_way",
            "target_language": "zh",
        },
        "context": {
            "general": [
                {"key": "domain", "value": "Religion"},
                {"key": "topic", "value": topic},
            ],
            "text": text,
            "terms": SOURCE_TERMS[source],
        },
    }


# ── Token Rendering ───────────────────────────────────────────────────────────


def render_tokens(final_tokens: list[dict]) -> str:
    """Convert Soniox tokens into a readable transcript, interleaving [xx] tags
    on language changes."""
    text_parts: list[str] = []
    current_language: Optional[str] = None

    for token in final_tokens:
        text = token["text"]
        if text == "<end>":
            continue
        language = token.get("language")
        is_translation = token.get("translation_status") == "translation"

        if language is not None and language != current_language:
            if text_parts and not text_parts[-1].endswith(" "):
                text_parts.append(" ")
            current_language = language
            prefix = "[Translation] " if is_translation else ""
            text_parts.append(f"{prefix}[{current_language}] ")
            text = text.lstrip()

        text_parts.append(text)

    return "".join(text_parts)


# ── Transcriber ───────────────────────────────────────────────────────────────


class Transcriber:
    """Soniox real-time transcription backend.

    Owns the websocket session, audio pump, and recv/gating loop. Each
    finalized phrase is emitted via the on_phrase(text) callback; the
    [lang] prefix from render_tokens is preserved so the orchestrator can
    fan out the source text to every translation worker unchanged.
    """

    def __init__(self, source: str, api_key: str):
        self.source = source
        self.api_key = api_key

    def run(self, device_index: int, on_phrase: Callable[[str], None],
            stop_event: threading.Event) -> None:
        config = build_soniox_config(self.source, self.api_key)
        print("Connecting to Soniox...")
        with connect(SONIOX_WEBSOCKET_URL) as ws:
            ws.send(json.dumps(config))

            def audio_pump():
                try:
                    for chunk in iter_audio_chunks(device_index, SAMPLE_RATE,
                                                   CHUNK_FRAMES, stop_event):
                        ws.send(chunk)
                except Exception:
                    pass
                # Empty string signals end-of-audio to the server.
                try:
                    ws.send("")
                except Exception:
                    pass

            audio_thread = threading.Thread(target=audio_pump, daemon=True)
            audio_thread.start()

            print("Session started. Speak into your microphone. Press Ctrl+C to stop.")

            final_tokens: list[dict] = []
            final_translation_tokens: list[dict] = []
            prev_final_count = 0
            prev_translation_count = 0

            try:
                while True:
                    message = ws.recv()
                    res = json.loads(message)

                    if res.get("error_code") is not None:
                        print(f"Error: {res['error_code']} - {res['error_message']}")
                        break

                    for token in res.get("tokens", []):
                        if token.get("text"):
                            # Soniox translation tokens are the phrase-boundary gate.
                            # Their content is discarded (target is zh, a dummy pivot).
                            if token.get("translation_status") == "translation":
                                if token.get("is_final"):
                                    final_translation_tokens.append(token)
                                continue
                            if token.get("is_final"):
                                final_tokens.append(token)

                    # Flush the buffered transcription when a gating translation
                    # token arrives — that marks the end of the latest phrase.
                    if len(final_translation_tokens) == prev_translation_count:
                        continue

                    new_tokens = final_tokens[prev_final_count:]
                    prev_final_count = len(final_tokens)
                    prev_translation_count = len(final_translation_tokens)
                    text = render_tokens(new_tokens)

                    print(f"[Transcription] {text}")
                    on_phrase(text)

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
