[CmdletBinding()]
param(
    [string]$Model = 'qwen3-32b-awq',
    [int]$Iterations = 1,
    [double]$TimeoutSeconds = 300,
    [string]$OutputRoot = 'C:\AGI_Runtime\benchmark-results\runpod-live-reliability-v1'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) { throw 'Set RUNPOD_API_KEY in this PowerShell session.' }
if ([string]::IsNullOrWhiteSpace($env:RUNPOD_ENDPOINT_ID)) { throw 'Set RUNPOD_ENDPOINT_ID to the endpoint ID only.' }
$env:MODEL_PROVIDER = 'runpod'
$env:RUNPOD_MODEL = $Model

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Development environment is missing. Run Setup-Development.ps1 first.' }
& $Python -m operator_research_engine.live_benchmark --model $Model --iterations $Iterations --timeout-seconds $TimeoutSeconds --output-root $OutputRoot
exit $LASTEXITCODE
