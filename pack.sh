#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for candidate in "$HOME/.pyenv/versions/3.12.13/bin/python" "python3.12" "python3" "python"; do
        if [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "Could not find a Python interpreter for packaging. Set PYTHON_BIN to continue." >&2
    exit 1
fi

rm -rf dist deps
mkdir dist

"$PYTHON_BIN" -m pip install --target ./deps -r requirements.txt

if [ ! -f "model/kokoro-v1.0.fp16.onnx" ] || [ ! -f "model/voices-v1.0.bin" ]; then
    chmod +x scripts/download_kokoro_assets.sh
    ./scripts/download_kokoro_assets.sh
fi

zip -r -9 "dist/cn-plugin-kokoro-tts.zip" \
    cn-plugin-kokoro-tts.py requirements.txt manifest.json __init__.py THIRD_PARTY_NOTICES.md \
    deps model scripts
