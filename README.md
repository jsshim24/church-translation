# Live Church Sermon Translation

Real-time sermon translation using [Soniox](https://soniox.com/) real-time STT and [Claude](https://anthropic.com/) for translation, with a built-in web display for ProPresenter or any browser. Supports Korean, English, and Spanish — in any source/target combination, including multilingual (ko+en+es) sermons. Each translation target runs on its own parallel worker, so one Korean phrase can be translated into English and Spanish simultaneously on separate URLs.

## Prerequisites

- macOS with [Homebrew](https://brew.sh/)
- A [Soniox API key](https://soniox.com/) (real-time speech-to-text)
- An [Anthropic API key](https://console.anthropic.com/) (Claude translation)
- An audio input device (e.g. USB interface from church soundboard)

## Setup

```bash
# Install dependencies (skip any you already have)
brew install python git portaudio

# Clone the repo
git clone https://github.com/junseobshim/church-translation.git
cd church-translation

# Create a virtual environment and install Python packages
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env   # then edit .env and fill in SONIOX_API_KEY and ANTHROPIC_API_KEY
```

## Running

```bash
# Korean (mixed with English) → English  (default)
python soniox_claude.py

# English → Korean
python soniox_claude.py --source en --target ko

# Korean → English AND Spanish (parallel translation streams on separate URLs)
python soniox_claude.py --source ko --target en,es

# Spanish (mixed with English) → English
python soniox_claude.py --source es --target en

# Multilingual (ko + en + es speech) → all three translation streams
python soniox_claude.py --source multi
```

You'll be prompted to select an audio input device, then transcription and translation begin immediately. A web caption server starts on port 8080 by default.

`--source` picks Soniox's strict language hints: `ko` → `[ko, en]`, `en` → `[en]`, `es` → `[es, en]`, `multi` → `[ko, en, es]`. `--target` accepts a comma-separated subset of `{ko, en, es}` minus the source; `--source multi` is fixed at `--target ko,en,es`. `--target` is required for `--source en` and `--source es`; defaults exist only for `ko` (→ `en`) and `multi` (→ `ko,en,es`).

### Sermon Outline (optional)

If you have the sermon outline ahead of time, pass it with `--outline` to give Claude topical and structural context. This also activates Anthropic prompt caching, making every subsequent translation call cheaper and slightly faster.

```bash
python soniox_claude.py --outline path/to/sermon.txt
```

- The file must be UTF-8 plain text. Any `.txt` with bullet points, verse references, or prose works. For a multilingual sermon, use a single multilingual outline; it is attached verbatim to every target worker's system prompt.
- Caching activates only when the combined system prompt + outline exceeds 1024 tokens (roughly 700–800 words). Below that, the script warns on stderr and runs without caching.
- With multiple `--target` languages, each target worker caches its own system-prompt + outline independently and has its own keep-alive ping. Expect one `Cache warmed` message per cached worker at startup.
- The cache has a 5-minute lifetime between calls. A keep-alive ping fires every 4m30s of silence so the cache survives long pauses.
- The outline is used as **context only** — Claude is instructed to translate what is actually said, even when the speaker rhetorically diverges from the outline.

### Application
Use the below script (audio device 4, default CLI flags otherwise) with Automator (Application, Run Shell Script) for one click execution (.app file, pinnable to dock)

`osascript -e "tell application \"Terminal\" to do script \"cd $HOME/Documents/church-translation && source venv/bin/activate && python soniox_claude.py --device 4\""`

Duplicate and edit the `.app` for other source/target combinations (e.g. `--source es --target en` for a Spanish service).

## Web Display

Open in any browser or ProPresenter Web Fill:

| URL | What it shows |
|-----|---------------|
| `http://localhost:8080/` | All transcription lines, regardless of detected language (Korean, English, Spanish as spoken). No query params needed. |
| `http://localhost:8080/?mode=translation` | Translations in the default target (the first `--target`). |
| `http://localhost:8080/?mode=translation&lang=en` | English translations only |
| `http://localhost:8080/?mode=translation&lang=ko` | Korean translations only |
| `http://localhost:8080/?mode=translation&lang=es` | Spanish translations only |
| `http://localhost:8080/?display=paragraph` | Paragraph style (for ProPresenter) |
| `http://localhost:8080/?mode=translation&lang=en&display=paragraph` | English translations, paragraph style |
| `http://localhost:8080/?mode=translation&lang=en&display=paragraph&fontSize=96&fontWeight=500&lineSpacing=1.3&bgColor=transparent&hideStatus=1` | English translations default for RCC Sanctuary TV display (ProPresenter web fill — transparent overlay, no status indicator) |
| `http://localhost:8080/?mode=transcription&lang=ko` | Only Korean transcription segments (explicit filter on the transcription stream) |

### Query Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `mode` | `transcription` | `transcription` or `translation` |
| `lang` | first `--target` for translation mode; no filter for transcription mode | ISO 639-1 language filter. In transcription mode, omitting `lang` shows all languages as spoken; in translation mode it defaults to the first `--target`. Explicit value always wins. |
| `display` | `line` | `line` (block divs) or `paragraph` (inline spans) |
| `fontSize` | `48` | Font size in px |
| `fontFamily` | `system-ui, sans-serif` | CSS font stack |
| `googleFont` | — | Google Fonts name (auto-loaded) |
| `fontWeight` | `normal` | CSS font weight |
| `color` | `white` | Text color |
| `lineSpacing` | `1.4` | CSS line-height |
| `textAlign` | `left` | CSS text-align |
| `textShadow` | `none` | CSS text-shadow |
| `bgColor` | `#000` | Background color. Pass `bgColor=transparent` for ProPresenter web fill. |
| `hideStatus` | `0` | Set to `1` to suppress the bottom-right "Waiting for transcription…" connection indicator. Use for ProPresenter web fill so the indicator never paints on the projection. |
| `padding` | `20` | Container padding in px |
| `maxLines` | `0` (unlimited) | Max lines displayed (hard cap 200) |

## Cloudflare Tunnel (Internet Access)

To make the web display accessible over the internet (e.g. at `live.rctranslation.org`):

```bash
# One-time setup (see below for steps regarding tunnel credentials)
brew install cloudflared
cloudflared tunnel login
```

Copy the .json file (`~/.cloudflared/` on an existing installation) associated with the church-live tunnel onto the new device.

Alternatively, regenerate the credentials through the Cloudflare dashboard (Networking -> Tunnels -> Rotate token).

The tunnel starts automatically when you run the script — `--tunnel church-live` is the default. Pass `--no-tunnel` to skip it for local-only work:

```bash
python soniox_claude.py              # tunnel runs automatically
python soniox_claude.py --no-tunnel  # localhost only
```

Viewers can access:
- `https://live.rctranslation.org/` — all transcription lines, regardless of language, with a solid black background (default)
- `https://live.rctranslation.org/?mode=translation&lang=en` — English translations
- `https://live.rctranslation.org/?mode=translation&lang=ko` — Korean translations
- `https://live.rctranslation.org/?mode=translation&lang=es` — Spanish translations

### Waiting page

When the tunnel has no origin (i.e. no device is running `soniox_claude.py`), visitors to `live.rctranslation.org` see Cloudflare's default 530 error. To replace that with a branded "Waiting for transcription…" page that auto-refreshes into captions when the tunnel comes back online, deploy the Cloudflare Worker in `worker/`. See [`worker/README.md`](worker/README.md) for the one-time deploy.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source {ko,en,es,multi}` | `ko` | Source language. `ko` = Korean + English, `en` = English only, `es` = Spanish + English, `multi` = Korean + English + Spanish. Sets Soniox's strict language hints. |
| `--target CSV` | `en` when `--source ko`, `ko,en,es` when `--source multi`; required otherwise | Comma-separated translation targets. Must be a non-empty subset of `{ko,en,es}` excluding `--source`. For `--source multi`, must be exactly `ko,en,es`. Each target runs as its own parallel Claude worker. |
| `--device N` | (interactive) | Audio input device index (skip selection prompt) |
| `--port PORT` | `8080` | Web caption server port (`0` to disable) |
| `--tunnel NAME` | `church-live` | Cloudflare tunnel name to start |
| `--no-tunnel` | — | Skip starting the Cloudflare tunnel |
| `--outline PATH` | — | Path to a UTF-8 `.txt` sermon outline. Enables per-target prompt caching when the combined system prompt exceeds 1024 tokens. |
| `--transcriber {soniox}` | `soniox` | Transcription backend. Loads `transcribe_<name>.py` at startup. More options will be added as alternative backends land (e.g. `mlx-whisper`, `azure`). |
| `--translator {claude}` | `claude` | Translation backend. Loads `translate_<name>.py` at startup. More options will be added as alternative backends land (e.g. `qwen-mlx`, `gemini`). |

## Architecture

The codebase splits into a shared shell plus per-backend modules:

- **`main.py`** — shared infrastructure: audio capture, web caption server, Cloudflare tunnel, prompt-building scaffolding, the LLM-agnostic `TranslationWorker` (queue/`[SKIP]`/rolling context), orchestration, and CLI. It loads the requested transcription and translation modules lazily via `importlib`, so a deployment using only e.g. `azure` + `gemini` backends would not pull in `websockets` or `anthropic`.
- **`transcribe_soniox.py`** — Soniox transcription backend: WebSocket session, audio pump, recv/gating loop, term lists, and the `[Transcription]` print. Imports `websockets`.
- **`translate_claude.py`** — Claude translation backend: per-target system prompt, ephemeral cache eligibility check, cache warmup, keepalive thread, and the `messages.create` translation call. Imports `anthropic`.

Alternative backends drop in alongside without modifying the main file beyond extending the `--transcriber` / `--translator` choice lists. They must implement these contracts:

- `Transcriber(source, api_key)` with `run(device_index, on_phrase, stop_event)` — blocking; calls `on_phrase(text)` once per finalized phrase and prints `[Transcription] {text}` itself.
- `Backend` with `from_outline(client, source, target, outline, model)` classmethod plus `warmup()`, `translate(context, latest)`, `mark_activity()`, and `start_keepalive(stop_event)` instance methods. Module-level `make_client(api_key)` factory and `DEFAULT_MODEL` constant.

`websockets` is Soniox-only and `anthropic` is Claude-only at the import level — both are still required for the default Soniox + Claude path. Once optional backends ship, `requirements.txt` may split into extras keyed by backend.

## License

Unlicense
