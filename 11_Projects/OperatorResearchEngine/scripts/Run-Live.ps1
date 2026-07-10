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

$RuntimeRoot = 'C:\AGI_Runtime'
if (-not [string]::IsNullOrWhiteSpace($env:AGI_RUNTIME_ROOT)) {
    $RuntimeRoot = $env:AGI_RUNTIME_ROOT
}
$LogRoot = Join-Path $RuntimeRoot 'logs'
try {
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $Probe = Join-Path $LogRoot ('.write-probe-' + [guid]::NewGuid().ToString('N') + '.tmp')
    Set-Content -LiteralPath $Probe -Value 'probe' -Encoding UTF8 -NoNewline
    Remove-Item -LiteralPath $Probe -Force
}
catch {
    throw "Runtime log root is not writable: $LogRoot. Fix permissions or set AGI_RUNTIME_ROOT to a writable runtime directory outside C:\AGI. $($_.Exception.Message)"
}

& $Python -m operator_research_engine.app --goal $Goal
exit $LASTEXITCODE
