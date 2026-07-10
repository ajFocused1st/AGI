[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Development environment is missing. Run Setup-Development.ps1 first.'
}

$TestRoots = @(
    (Join-Path $WorkspaceRoot '01_Core\Contracts\tests'),
    (Join-Path $WorkspaceRoot '01_Core\Evaluation\tests'),
    (Join-Path $WorkspaceRoot '06_Tools\tests'),
    (Join-Path $WorkspaceRoot '07_Models\Adapters\Ollama\tests'),
    (Join-Path $WorkspaceRoot '02_Agents\Runtime\tests'),
    (Join-Path $ProjectRoot 'tests')
)

foreach ($TestRoot in $TestRoots) {
    if (Test-Path -LiteralPath $TestRoot -PathType Container) {
        & $Python -m unittest discover -v -s $TestRoot -p 'test_*.py'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
