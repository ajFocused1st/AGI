[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$secureKey = $null
$plainKey = $null
$keyPointer = [IntPtr]::Zero

try {
    Write-Host 'This diagnostic lists served model IDs and sends one plain chat request.' -ForegroundColor Cyan
    $secureKey = Read-Host 'Paste the temporary RunPod API key' -AsSecureString
    $endpointId = (Read-Host 'Paste the RunPod endpoint ID only').Trim()
    if ($endpointId -notmatch '^[A-Za-z0-9_-]{1,128}$') {
        throw 'Invalid endpoint ID.'
    }
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'The API key was blank.' }

    $headers = @{ Authorization = "Bearer $plainKey"; Accept = 'application/json' }
    $baseUrl = "https://api.runpod.ai/v2/$endpointId/openai/v1"
    Write-Host 'Requesting the provider model list...'
    $models = Invoke-RestMethod -Uri "$baseUrl/models" -Headers $headers -Method Get -TimeoutSec 300
    $modelIds = @($models.data | ForEach-Object { [string]$_.id } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($modelIds.Count -eq 0) { throw 'RunPod returned no served model IDs.' }
    Write-Host ('Served model IDs: ' + ($modelIds -join ', ')) -ForegroundColor Green

    $payload = @{
        model = $modelIds[0]
        messages = @(@{ role = 'user'; content = 'Reply with exactly: connection ok' })
        temperature = 0
        max_tokens = 32
        stream = $false
    } | ConvertTo-Json -Depth 6 -Compress
    Write-Host 'Sending one plain chat request...'
    $response = Invoke-RestMethod -Uri "$baseUrl/chat/completions" -Headers $headers -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 300
    $content = [string]$response.choices[0].message.content
    Write-Host ('Plain chat response: ' + $content) -ForegroundColor Green
    Write-Host 'Provider-only diagnostic succeeded.' -ForegroundColor Green
}
catch {
    $status = $null
    if ($null -ne $_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
    }
    if ($null -ne $status) {
        Write-Host "Provider-only diagnostic failed with HTTP $status." -ForegroundColor Red
    }
    else {
        Write-Host "Provider-only diagnostic failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}
finally {
    $headers = $null
    $plainKey = $null
    $secureKey = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Write-Host 'Credentials cleared from this PowerShell process.' -ForegroundColor Cyan
    Read-Host 'Press Enter to close this window'
}
