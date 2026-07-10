[CmdletBinding()]
param(
    [string]$Model = 'llama3:latest',
    [int]$Iterations = 3,
    [double]$TimeoutSeconds = 120,
    [string]$OutputRoot = 'C:\AGI_Runtime\benchmark-results\live-reliability-v1'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Iterations -lt 1) {
    throw 'Iterations must be at least 1.'
}
if ($TimeoutSeconds -le 0) {
    throw 'TimeoutSeconds must be greater than 0.'
}

$WorkspaceRoot = 'C:\AGI'
$RuntimeRoot = 'C:\AGI_Runtime'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $WorkspaceRoot -PathType Container)) {
    throw "Workspace root is unavailable: $WorkspaceRoot"
}
if (-not (Test-Path -LiteralPath $RuntimeRoot -PathType Container)) {
    throw "Runtime root is unavailable: $RuntimeRoot"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Development environment is missing. Run Setup-Development.ps1 first.'
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$Probe = Join-Path $OutputRoot ('.write-probe-' + [guid]::NewGuid().ToString('N') + '.tmp')
try {
    Set-Content -LiteralPath $Probe -Value 'probe' -Encoding UTF8 -NoNewline
    Remove-Item -LiteralPath $Probe -Force
}
catch {
    throw "Benchmark output root is not writable: $OutputRoot. $($_.Exception.Message)"
}

$Ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($null -eq $Ollama) {
    throw 'Ollama CLI is not available on PATH.'
}

$ListOutput = & ollama list
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to list installed Ollama models.'
}
if (-not ($ListOutput -match [regex]::Escape($Model))) {
    throw "Selected Ollama model is not installed: $Model"
}

try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -Method Get -TimeoutSec 5 | Out-Null
}
catch {
    throw "Ollama is not reachable at http://127.0.0.1:11434. Leave the Ollama terminal running before starting this benchmark. $($_.Exception.Message)"
}

& $Python -m operator_research_engine.live_benchmark `
    --model $Model `
    --iterations $Iterations `
    --timeout-seconds $TimeoutSeconds `
    --output-root $OutputRoot
exit $LASTEXITCODE
