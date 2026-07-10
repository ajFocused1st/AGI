[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Goal
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Development environment is missing. Run Setup-Development.ps1 first.'
}
if ([string]::IsNullOrWhiteSpace($env:OLLAMA_MODEL)) {
    throw 'Set OLLAMA_MODEL to an already-installed local model. No model will be pulled automatically.'
}

& $Python -m operator_research_engine.app --goal $Goal
exit $LASTEXITCODE
