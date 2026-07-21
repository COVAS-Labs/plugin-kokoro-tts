$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginDir = Split-Path -Parent $ScriptDir
$ModelDir = Join-Path $PluginDir "model"
$BaseUrl = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

function Get-KokoroAsset([string]$Filename) {
    $Target = Join-Path $ModelDir $Filename
    $Temporary = "$Target.tmp"

    if (Test-Path $Target) {
        return
    }

    Write-Host "Downloading $Filename ..."
    Invoke-WebRequest -Uri "$BaseUrl/$Filename" -OutFile $Temporary
    Move-Item -Force $Temporary $Target
}

Get-KokoroAsset "kokoro-v1.0.fp16.onnx"
Get-KokoroAsset "voices-v1.0.bin"

Write-Host "Kokoro assets are ready in $ModelDir"
