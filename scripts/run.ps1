param([Parameter(ValueFromRemainingArguments = $true)][string[]]$FerArgs)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $PSScriptRoot '..\src'
$env:NO_ALBUMENTATIONS_UPDATE = '1'
$pythonPath = $env:FER_PYTHON
$localConfigPath = Join-Path $PSScriptRoot '..\.fer-local.json'
if (Test-Path -LiteralPath $localConfigPath) {
    $ferSettings = Get-Content -LiteralPath $localConfigPath -Raw | ConvertFrom-Json
    if (-not $pythonPath) { $pythonPath = $ferSettings.python }
    if (-not $env:FER_TRACKING_DIR) { $env:FER_TRACKING_DIR = $ferSettings.tracking_dir }
}
if (-not $pythonPath) { $pythonPath = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe' }
Push-Location (Join-Path $PSScriptRoot '..')
try {
    & $pythonPath -m cli @FerArgs
    $ferExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $ferExitCode
