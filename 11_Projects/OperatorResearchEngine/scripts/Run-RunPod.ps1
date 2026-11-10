[CmdletBinding()]
param([Parameter(Mandatory, Position = 0)][string]$Goal)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
    throw 'Set RUNPOD_API_KEY in this PowerShell session. It is never read from a repository file.'
}
if ([string]::IsNullOrWhiteSpace($env:RUNPOD_ENDPOINT_ID)) {
    throw 'Set RUNPOD_ENDPOINT_ID to the endpoint ID only, not a URL.'
}
if ($env:RUNPOD_ENDPOINT_ID -match '\s|/|:') {
    throw 'RUNPOD_ENDPOINT_ID must be an endpoint ID only, without whitespace or URL characters.'
}
$env:MODEL_PROVIDER = 'runpod'
if ([string]::IsNullOrWhiteSpace($env:RUNPOD_MODEL)) {
    $env:RUNPOD_MODEL = 'qwen3-32b-awq'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Development environment is missing. Run Setup-Development.ps1 first.'
}
& $Python -m operator_research_engine.app --goal $Goal
exit $LASTEXITCODE
