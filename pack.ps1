if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

$PythonCommand = $env:PYTHON_BIN
$PythonArgs = @()
if (-not $PythonCommand) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCommand = "python"
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonCommand = "py"
        $PythonArgs = @("-3.12")
    }
    else {
        throw "Could not find a Python interpreter for packaging. Set PYTHON_BIN to continue."
    }
}

New-Item "dist" -ItemType Directory | Out-Null
if (Test-Path "deps") {
    Remove-Item -Recurse -Force "deps"
}

& $PythonCommand @PythonArgs -m pip install --target ./deps -r requirements.txt

if ((-not (Test-Path "model\kokoro-v1.0.fp16.onnx")) -or (-not (Test-Path "model\voices-v1.0.bin"))) {
    .\scripts\download_kokoro_assets.ps1
}

$artifacts = "cn-plugin-kokoro-tts.py", "requirements.txt", "manifest.json", "__init__.py", "THIRD_PARTY_NOTICES.md", "deps", "model", "scripts"
Compress-Archive -LiteralPath $artifacts -CompressionLevel Fastest -DestinationPath "dist\cn-plugin-kokoro-tts.zip"
