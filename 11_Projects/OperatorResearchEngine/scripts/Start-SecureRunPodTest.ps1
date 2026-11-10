[CmdletBinding()]
param(
    [string]$Goal = 'Create a three-step plan for researching whether a local courier company should target medical offices.'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$secureKey = $null
$plainKey = $null
$keyPointer = [IntPtr]::Zero

try {
    Write-Host 'RunPod credentials will exist only in this PowerShell process.' -ForegroundColor Cyan
    $secureKey = Read-Host 'Paste the temporary RunPod API key' -AsSecureString
    $endpointId = (Read-Host 'Paste the RunPod endpoint ID only (not the URL)').Trim()
    if ($endpointId -notmatch '^[A-Za-z0-9_-]{1,128}$') {
        throw 'The endpoint ID contains invalid characters. Paste only the ID, not a URL.'
    }

    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw 'The API key was blank.'
    }

    $env:RUNPOD_API_KEY = $plainKey
    $env:RUNPOD_ENDPOINT_ID = $endpointId
    $env:RUNPOD_MODEL = 'qwen3-32b-awq'
    $env:MODEL_PROVIDER = 'runpod'

    & (Join-Path $PSScriptRoot 'Run-RunPod.ps1') -Goal $Goal
    if ($LASTEXITCODE -ne 0) {
        throw "The controlled RunPod test exited with code $LASTEXITCODE."
    }
    Write-Host 'Controlled RunPod test completed successfully.' -ForegroundColor Green
}
catch {
    Write-Host "RunPod test failed: $($_.Exception.Message)" -ForegroundColor Red
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
