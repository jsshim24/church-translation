# Korean → English Live Church Translation

Real-time Korean-to-English translation for a live church context. Captures audio from any input device, transcribes with Google Cloud Speech-to-Text V2 (Chirp 3), and translates each phrase with Claude.

## How It Works

1. **Audio capture** — `sounddevice` streams raw PCM from your audio input device
2. **Speech-to-text** — Google Cloud STT V2 with the Chirp 3 model transcribes Korean (and passes through English) in real time
3. **Translation** — Claude translates each finalized phrase, maintaining conversational tone and theological terminology
4. **Endless streaming** — STT sessions automatically chain every 4 minutes at phrase boundaries, with overlap audio to prevent content loss
5. **Web captions** — A built-in HTTP server serves a live caption page for display on screens via ProPresenter Web Fill or any browser

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
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
python3 translate.py
```

You'll be prompted to select an audio input device, then translation begins immediately. A web caption server starts on port 8080 by default.

### Options

| Flag | Description |
|------|-------------|
| `--device N` | Skip device selection, use device index N |
| `--model MODEL` | Claude model to use (default: `claude-sonnet-4-6`) |
| `--haiku` | Use `claude-haiku-4-5` for faster, lighter translation |
| `--standard-endpointing` | Use standard endpointing instead of fast (default is fast) |
| `--port PORT` | Web caption server port (default: `8080`, set to `0` to disable) |

### Examples

```bash
# Use a specific audio device and faster model
python3 translate.py --device 2 --haiku

# Disable web caption server
python3 translate.py --port 0

# Standard (slower) endpointing for more complete phrases
python3 translate.py --standard-endpointing
```

Press **Ctrl+C** to stop.

### Web Captions (ProPresenter / Browser)

The built-in web server serves a live caption page at `http://localhost:8080`. Use it as a ProPresenter Web Fill layer or open it directly in a browser.

Styling is controlled via URL query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fontSize` | `48` | Font size in pixels |
| `fontFamily` | `system-ui, sans-serif` | CSS font stack (system fonts) |
| `googleFont` | — | Google Font name (e.g. `Noto+Sans+KR`) |
| `fontWeight` | `normal` | Font weight (`normal`, `bold`, `700`, etc.) |
| `color` | `white` | Text color |
| `bgColor` | `transparent` | Background color (transparent for ProPresenter overlay) |
| `lineSpacing` | `1.4` | CSS line-height |
| `textAlign` | `left` | Text alignment (`left`, `center`, `right`) |
| `textShadow` | `none` | CSS text-shadow (e.g. `2px 2px 4px black`) |
| `padding` | `20` | Container padding in pixels |
| `maxLines` | `0` | Max phrases to keep (0 = unlimited, capped at 200) |
| `showKorean` | `false` | Show Korean source text above each translation |

Example URL:

```
http://localhost:8080/?fontSize=64&color=yellow&bgColor=black&googleFont=Noto+Sans+KR&showKorean=true
```

## License

Unlicense
