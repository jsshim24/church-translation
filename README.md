# Korean → English Live Church Translation

Real-time Korean-to-English translation pipeline for church sermons using [Soniox](https://soniox.com/) real-time STT and [Claude](https://anthropic.com/) for translation.

## Files

| File | Description |
|------|-------------|
| `translate.py` | Full pipeline — Soniox STT + Claude translation + web caption server for ProPresenter |
| `soniox_translate.py` | Soniox STT with native translation (WIP — migrating `translate.py` features here) |

## How It Works

1. **Audio capture** — `sounddevice` streams raw PCM from your audio input device (e.g. USB interface from church soundboard)
2. **Speech-to-text** — Soniox real-time STT via WebSocket transcribes Korean (with automatic language identification for English pass-through)
3. **Translation** — Claude translates each finalized phrase, maintaining conversational tone and theological terminology
4. **Web captions** — A built-in HTTP server serves a live caption page for display on screens via ProPresenter Web Fill or any browser

## Setup

### Prerequisites

- Python 3.10+
- A [Soniox](https://soniox.com/) API key (for real-time speech-to-text)
- An [Anthropic API key](https://console.anthropic.com/) (for Claude translation)

### Install

```bash
git clone https://github.com/jsshim24/church-translation.git
cd church-translation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
SONIOX_API_KEY=your-soniox-api-key
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### translate.py (full pipeline)

```bash
python3 translate.py
```

You'll be prompted to select an audio input device, then translation begins immediately. A web caption server starts on port 8080 by default.

#### Options

| Flag | Description |
|------|-------------|
| `--device N` | Skip device selection, use device index N |
| `--model MODEL` | Claude model to use (default: `claude-sonnet-4-6`) |
| `--haiku` | Use `claude-haiku-4-5` for faster, lighter translation |
| `--language LANGS` | Language hints for STT, space-separated (default: `"ko en"`) |
| `--endpoint-delay MS` | Max endpoint delay in ms, 500–3000 (default: `1500`) |
| `--port PORT` | Web caption server port (default: `8080`, set to `0` to disable) |

### soniox_translate.py (WIP)

```bash
python3 soniox_translate.py
python3 soniox_translate.py --device 2
```

Currently uses Soniox native translation. Claude Haiku translation and web caption support are being migrated here.

Press **Ctrl+C** to stop either script.

## License

Unlicense
