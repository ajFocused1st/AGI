[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot '..\..')).Path
$VirtualEnvironment = Join-Path $ProjectRoot '.venv'
$Python = Join-Path $VirtualEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    python -m venv --system-site-packages $VirtualEnvironment
}

$SourceRoots = @(
    (Join-Path $WorkspaceRoot '01_Core\Contracts\src'),
    (Join-Path $WorkspaceRoot '01_Core\Evaluation\src'),
    (Join-Path $WorkspaceRoot '06_Tools\src'),
    (Join-Path $WorkspaceRoot '07_Models\Adapters\Ollama\src'),
    (Join-Path $WorkspaceRoot '02_Agents\Runtime\src'),
    (Join-Path $ProjectRoot 'src')
)

foreach ($SourceRoot in $SourceRoots) {
    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        throw "Missing source root: $SourceRoot"
    }
}

$SitePackages = (& $Python -c "import site; print(site.getsitepackages()[0])").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($SitePackages)) {
    throw 'Could not locate the virtual environment site-packages directory.'
}
$PathFile = Join-Path $SitePackages 'agi_workspace_sources.pth'
[System.IO.File]::WriteAllLines(
    $PathFile,
    [string[]]$SourceRoots,
    [System.Text.UTF8Encoding]::new($false)
)

& $Python -c "import agi_agent_runtime, agi_contracts, agi_evaluation, agi_ollama, agi_tools, operator_research_engine; print('Development environment ready.')"
if ($LASTEXITCODE -ne 0) {
    throw 'Development import validation failed.'
}
