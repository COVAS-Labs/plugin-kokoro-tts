# COVAS:NEXT Plugin Kokoro TTS

Run high-quality Kokoro speech synthesis locally in COVAS:NEXT using ONNX Runtime.

## Features

- Fully local CPU inference after installation.
- 54 preset voices across American and British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese.
- Voice language is inferred automatically from the selected preset.
- 24 kHz PCM audio streamed in short chunks for COVAS:NEXT.

## Installation

Download the latest release and unpack it into the COVAS:NEXT `plugins` directory. Select **Kokoro (Local)** as the TTS provider in **Advanced** -> **TTS Settings**.

The plugin archive includes the model, voices, and Python dependencies. It does not require internet access at runtime.

## Settings

- **Default voice**: Used when COVAS:NEXT does not request a Kokoro preset directly. Common provider-agnostic names such as `nova`, `alloy`, `echo`, `fable`, `onyx`, and `shimmer` map to suitable Kokoro voices.
- **Speed**: Controls speech rate from `0.5` to `2.0`; `1.0` is the default.

Kokoro performs best with passages around 100 to 200 tokens. The plugin splits long input at sentence boundaries before synthesis.

## Development

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

Download the model assets explicitly when needed:

```bash
./scripts/download_kokoro_assets.sh
```

```powershell
.\scripts\download_kokoro_assets.ps1
```

Build a distributable archive with `./pack.sh` on Linux/macOS or `./pack.ps1` on Windows. The pack scripts download missing model assets automatically.

## Acknowledgements

- [COVAS:NEXT](https://github.com/RatherRude/Elite-Dangerous-AI-Integration)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), licensed under Apache 2.0
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), licensed under MIT
