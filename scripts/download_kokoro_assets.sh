#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PLUGIN_DIR/model"
BASE_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

mkdir -p "$MODEL_DIR"

download() {
    local filename="$1"
    local target="$MODEL_DIR/$filename"
    local temporary="$target.tmp"

    if [ -f "$target" ]; then
        return
    fi

    echo "Downloading $filename ..."
    curl --fail --location --retry 3 --output "$temporary" "$BASE_URL/$filename"
    mv "$temporary" "$target"
}

download "kokoro-v1.0.fp16.onnx"
download "voices-v1.0.bin"

echo "Kokoro assets are ready in $MODEL_DIR"
