# Korean → English Live Sermon Translator

Real-time Korean-to-English translation for live church sermons. Captures audio from a soundboard (or any input device), transcribes with Google Cloud Speech-to-Text V2 (Chirp 3), and translates each phrase with Claude.

## How It Works

1. **Audio capture** — `sounddevice` streams raw PCM from your audio input device
2. **Speech-to-text** — Google Cloud STT V2 with the Chirp 3 model transcribes Korean (and passes through English) in real time
3. **Translation** — Claude translates each finalized phrase, maintaining conversational tone and theological terminology
4. **Endless streaming** — STT sessions automatically chain every 4 minutes at phrase boundaries, with overlap audio to prevent content loss

```
[03:42] 🇰🇷 하나님의 은혜로 우리가 구원을 받았습니다
       🇺🇸 By God's grace, we have received salvation
────────────────────────────────────
[03:47] 🇰🇷 이것은 우리의 행위로 된 것이 아닙니다
       🇺🇸 This was not done by our own works
────────────────────────────────────
```

## Setup

### Prerequisites

- Python 3.10+
- A [Google Cloud](https://console.cloud.google.com/) project with the Speech-to-Text V2 API enabled
- A Google Cloud service account key (JSON)
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/sermon-translator.git
cd sermon-translator
python -m venv venv
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
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
python translate.py
```

You'll be prompted to select an audio input device, then translation begins immediately.

### Options

| Flag | Description |
|------|-------------|
| `--device N` | Skip device selection, use device index N |
| `--model MODEL` | Claude model to use (default: `claude-sonnet-4-6`) |
| `--haiku` | Use `claude-haiku-4-5` for faster, lighter translation |
| `--fast-endpointing` | Shorter pauses trigger phrase finalization (faster but may split mid-sentence) |

### Examples

```bash
# Use a specific audio device and faster model
python translate.py --device 2 --haiku

# Fast endpointing for rapid-fire speakers
python translate.py --fast-endpointing
```

Press **Ctrl+C** to stop.

## Audio Setup

This is designed for a Mac Mini connected to a church soundboard via a USB audio interface. Any audio input device that shows up in your system works — USB interfaces, built-in mic, virtual audio devices, etc.

## License

MIT
