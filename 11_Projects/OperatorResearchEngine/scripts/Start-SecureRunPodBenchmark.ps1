[CmdletBinding()]
param(
    [int]$Iterations = 1,
    [double]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$secureKey = $null
$plainKey = $null
$keyPointer = [IntPtr]::Zero

try {
    Write-Host "RunPod credentials will exist only during this $Iterations-iteration benchmark." -ForegroundColor Cyan
    $secureKey = Read-Host 'Paste the temporary RunPod API key' -AsSecureString
    $endpointId = (Read-Host 'Paste the RunPod endpoint ID only').Trim()
    if ($endpointId -notmatch '^[A-Za-z0-9_-]{1,128}$') { throw 'Invalid endpoint ID.' }

    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'The API key was blank.' }

    $env:RUNPOD_API_KEY = $plainKey
    $env:RUNPOD_ENDPOINT_ID = $endpointId
    $env:RUNPOD_MODEL = 'qwen3-32b-awq'
    $env:MODEL_PROVIDER = 'runpod'

    & (Join-Path $PSScriptRoot 'Run-RunPodBenchmark.ps1') `
        -Model 'qwen3-32b-awq' `
        -Iterations $Iterations `
        -TimeoutSeconds $TimeoutSeconds
    if ($LASTEXITCODE -notin @(0, 2)) {
        throw "The RunPod benchmark exited with environment error code $LASTEXITCODE."
    }
    if ($LASTEXITCODE -eq 2) {
        Write-Host 'Benchmark completed; one or more quality gates failed.' -ForegroundColor Yellow
    }
    else {
        Write-Host 'Benchmark completed and all provisional quality gates passed.' -ForegroundColor Green
    }
}
catch {
    Write-Host "RunPod benchmark failed: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:RUNPOD_ENDPOINT_ID -ErrorAction SilentlyContinue
    Remove-Item Env:RUNPOD_MODEL -ErrorAction SilentlyContinue
    Remove-Item Env:MODEL_PROVIDER -ErrorAction SilentlyContinue
    $plainKey = $null
    $secureKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Write-Host 'Credentials cleared from this PowerShell process.' -ForegroundColor Cyan
    Read-Host 'Press Enter to close this window'
}
