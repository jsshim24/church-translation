# Korean → English Live Church Translation

Real-time Korean-to-English translation for church sermons using [Soniox](https://soniox.com/) real-time STT and [Claude](https://anthropic.com/) for translation, with a built-in web display for ProPresenter or any browser.

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
git clone https://github.com/jsshim24/church-translation.git
cd church-translation

# Create a virtual environment and install Python packages
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env   # then edit .env and fill in SONIOX_API_KEY and ANTHROPIC_API_KEY
```

## Running

```bash
python3 soniox_claude.py
```

You'll be prompted to select an audio input device, then transcription and translation begin immediately. A web caption server starts on port 8080 by default.

## Web Display

Open in any browser or ProPresenter Web Fill:

| URL | What it shows |
|-----|---------------|
| `http://localhost:8080/` | Transcriptions, line-by-line (default) |
| `http://localhost:8080/?mode=translation&lang=en` | English translations only |
| `http://localhost:8080/?display=paragraph` | Paragraph style (for ProPresenter) |
| `http://localhost:8080/?mode=translation&lang=en&display=paragraph` | English translations, paragraph style |

### Query Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `mode` | `transcription` | `transcription` or `translation` |
| `lang` | `en` | ISO 639-1 language filter (`en`, `ko`, etc.) |
| `display` | `line` | `line` (block divs) or `paragraph` (inline spans) |
| `fontSize` | `48` | Font size in px |
| `fontFamily` | `system-ui, sans-serif` | CSS font stack |
| `googleFont` | — | Google Fonts name (auto-loaded) |
| `fontWeight` | `normal` | CSS font weight |
| `color` | `white` | Text color |
| `lineSpacing` | `1.4` | CSS line-height |
| `textAlign` | `left` | CSS text-align |
| `textShadow` | `none` | CSS text-shadow |
| `bgColor` | `transparent` | Background color |
| `padding` | `20` | Container padding in px |
| `maxLines` | `0` (unlimited) | Max lines displayed (hard cap 200) |

## Cloudflare Tunnel (Internet Access)

To make the web display accessible over the internet (e.g. at `live.rctranslation.org`):

```bash
# One-time setup
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create church-live
cloudflared tunnel route dns church-live live.rctranslation.org
```

Then run with the `--tunnel` flag:

```bash
python3 soniox_translate.py --tunnel church-live
```

Viewers can access:
- `https://live.rctranslation.org/` — transcriptions (default)
- `https://live.rctranslation.org/?mode=translation&lang=en` — English translations

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device N` | (interactive) | Audio input device index (skip selection prompt) |
| `--port PORT` | `8080` | Web caption server port (`0` to disable) |
| `--tunnel NAME` | — | Cloudflare tunnel name to start |

## License

Unlicense
