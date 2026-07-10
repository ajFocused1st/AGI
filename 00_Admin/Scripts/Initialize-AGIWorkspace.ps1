[CmdletBinding()]
param(
    [string]$Root = 'C:\AGI',
    [string]$RuntimeRoot = 'C:\AGI_Runtime',
    [switch]$SkipRuntime
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExpectedRoot = 'C:\AGI'
$ExpectedRuntimeRoot = 'C:\AGI_Runtime'
$Created = [System.Collections.Generic.List[string]]::new()
$Skipped = [System.Collections.Generic.List[string]]::new()
$Failed = [System.Collections.Generic.List[string]]::new()

function Get-NormalizedPath {
    param([Parameter(Mandatory)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Assert-ExactRoot {
    param(
        [Parameter(Mandatory)][string]$Actual,
        [Parameter(Mandatory)][string]$Expected,
        [Parameter(Mandatory)][string]$Label
    )
    if ((Get-NormalizedPath $Actual) -ine (Get-NormalizedPath $Expected)) {
        throw "$Label must be exactly '$Expected'; received '$Actual'."
    }
}

function New-SafeDirectory {
    param([Parameter(Mandatory)][string]$Path)
    try {
        if (Test-Path -LiteralPath $Path) {
            if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
                throw "A non-directory item already exists at this path."
            }
            $Skipped.Add("FOLDER $Path")
            return
        }
        $null = New-Item -ItemType Directory -Path $Path
        $Created.Add("FOLDER $Path")
    }
    catch {
        $Failed.Add("FOLDER $Path :: $($_.Exception.Message)")
    }
}

function New-SafeFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [AllowEmptyString()][string]$Content = ''
    )
    try {
        if (Test-Path -LiteralPath $Path) {
            $Skipped.Add("FILE   $Path")
            return
        }
        $parent = Split-Path -Parent $Path
        if ($parent) { New-SafeDirectory -Path $parent }
        $null = New-Item -ItemType File -Path $Path
        if ($Content.Length -gt 0) {
            [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
        }
        $Created.Add("FILE   $Path")
    }
    catch {
        $Failed.Add("FILE   $Path :: $($_.Exception.Message)")
    }
}

function Add-RelativeDirectories {
    param(
        [Parameter(Mandatory)][string]$Base,
        [Parameter(Mandatory)][string[]]$Paths
    )
    foreach ($relativePath in $Paths) {
        New-SafeDirectory -Path (Join-Path $Base $relativePath)
    }
}

function Add-MarkdownFile {
    param(
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string]$Purpose
    )
    New-SafeFile -Path (Join-Path $Root $RelativePath) -Content "# $Title`r`n`r`n$Purpose`r`n"
}

Assert-ExactRoot -Actual $Root -Expected $ExpectedRoot -Label 'Source root'
if (-not $SkipRuntime) {
    Assert-ExactRoot -Actual $RuntimeRoot -Expected $ExpectedRuntimeRoot -Label 'Runtime root'
}

Write-Host "Source root:  $(Get-NormalizedPath $Root)"
if (-not $SkipRuntime) { Write-Host "Runtime root: $(Get-NormalizedPath $RuntimeRoot)" }

New-SafeDirectory -Path $Root

$TopLevelFolders = @(
    '00_Admin', '01_Core', '02_Agents', '03_Cognition', '04_Memory',
    '05_Knowledge', '06_Tools', '07_Models', '08_Data', '09_Experiments',
    '10_Benchmarks', '11_Projects', '12_Research', '13_APIs',
    '14_Deployments', '15_Monitoring', '16_Security', '17_UI',
    '18_Documentation', '19_Backups', 'Sandbox'
)
Add-RelativeDirectories -Base $Root -Paths $TopLevelFolders

$Directories = @(
    '00_Admin\Scripts', '00_Admin\Reports', '00_Admin\Decisions',
    '01_Core\Contracts\schemas', '01_Core\Contracts\events',
    '01_Core\Contracts\interfaces', '01_Core\Contracts\errors',
    '01_Core\Runtime', '01_Core\Orchestration', '01_Core\Planning',
    '01_Core\Evaluation', '01_Core\Permissions', '01_Core\Config',
    '02_Agents\Runtime', '02_Agents\Profiles', '02_Agents\Prompts',
    '02_Agents\Policies', '02_Agents\Tests',
    '03_Cognition\Planning', '03_Cognition\Reasoning',
    '03_Cognition\Reflection', '03_Cognition\Verification',
    '03_Cognition\DecisionMaking', '03_Cognition\Uncertainty',
    '03_Cognition\SelfCorrection',
    '04_Memory\Contracts', '04_Memory\Working', '04_Memory\LongTerm',
    '04_Memory\Retrieval', '04_Memory\Policies', '04_Memory\Tests',
    '05_Knowledge\Sources', '05_Knowledge\Ingestion', '05_Knowledge\Indexing',
    '05_Knowledge\Retrieval', '05_Knowledge\Citations', '05_Knowledge\Evaluation',
    '06_Tools\Registry', '06_Tools\Browser', '06_Tools\Search',
    '06_Tools\Python', '06_Tools\Terminal', '06_Tools\FileSystem',
    '06_Tools\Email', '06_Tools\Database', '06_Tools\Permissions',
    '06_Tools\Sandboxes', '06_Tools\Audit',
    '07_Models\Adapters', '07_Models\Registry', '07_Models\Routing',
    '07_Models\Metadata', '07_Models\Local', '07_Models\Remote',
    '07_Models\Embeddings', '07_Models\Evaluation',
    '08_Data\Samples', '08_Data\Schemas', '08_Data\Fixtures',
    '09_Experiments\Templates', '09_Experiments\Active',
    '09_Experiments\Completed', '09_Experiments\Archived',
    '10_Benchmarks\Definitions', '10_Benchmarks\Runners',
    '10_Benchmarks\Datasets', '10_Benchmarks\Reports',
    '11_Projects\AGI_Core', '11_Projects\SharedLibraries',
    '12_Research\Papers', '12_Research\Reviews', '12_Research\Hypotheses',
    '12_Research\Replications', '12_Research\Drafts',
    '13_APIs\Contracts', '13_APIs\Providers\OpenAI',
    '13_APIs\Providers\Anthropic', '13_APIs\Providers\Google',
    '13_APIs\Providers\RunPod', '13_APIs\Providers\Ollama',
    '13_APIs\Auth', '13_APIs\RateLimits', '13_APIs\Retries', '13_APIs\Tests',
    '14_Deployments\Local', '14_Deployments\Containers',
    '14_Deployments\Cloud', '14_Deployments\Scripts', '14_Deployments\Configs',
    '15_Monitoring\Configs', '15_Monitoring\Dashboards', '15_Monitoring\Alerts',
    '16_Security\Policies', '16_Security\RiskAssessments',
    '16_Security\IncidentResponse', '16_Security\Audit',
    '17_UI\Dashboard', '17_UI\Components', '17_UI\Assets', '17_UI\Prototypes',
    '18_Documentation\Architecture', '18_Documentation\APIs',
    '18_Documentation\SetupGuides', '18_Documentation\UserGuides',
    '18_Documentation\DeveloperGuides', '18_Documentation\Decisions',
    '19_Backups\scripts',
    'Sandbox\Scratch', 'Sandbox\Tests', 'Sandbox\Broken',
    'Sandbox\Temporary', 'Sandbox\Prototypes', 'Sandbox\Downloads'
)
Add-RelativeDirectories -Base $Root -Paths $Directories

$MarkdownFiles = @(
    @('00_Admin\Vision.md', 'Vision', 'Defines the intended outcomes and guiding principles for the AGI workspace.'),
    @('00_Admin\Roadmap.md', 'Roadmap', 'Tracks staged capabilities, dependencies, and review gates.'),
    @('00_Admin\Architecture.md', 'Architecture', 'Records the current system boundaries and their ownership.'),
    @('00_Admin\NamingConvention.md', 'Naming Convention', 'Defines naming rules for code, configuration, data, and documentation.'),
    @('01_Core\README.md', 'Core', 'Owns shared contracts, runtime behavior, orchestration, evaluation, permissions, and configuration.'),
    @('02_Agents\README.md', 'Agents', 'Defines agents as profiles and policies running on a shared agent runtime.'),
    @('03_Cognition\README.md', 'Cognition', 'Contains research and reusable implementations for cognitive processes.'),
    @('04_Memory\README.md', 'Memory', 'Owns memory contracts, retrieval behavior, retention policies, and tests.'),
    @('05_Knowledge\README.md', 'Knowledge', 'Owns source ingestion, indexing, retrieval, citations, and knowledge evaluation.'),
    @('06_Tools\README.md', 'Tools', 'Owns executable capabilities, common tool contracts, permission checks, sandboxing, and auditing.'),
    @('07_Models\README.md', 'Models', 'Owns model abstractions, adapters, metadata, registration, routing, and evaluation.'),
    @('08_Data\README.md', 'Data', 'Contains only small, versionable schemas, fixtures, and samples; runtime data belongs in C:\AGI_Runtime.'),
    @('09_Experiments\README.md', 'Experiments', 'Provides controlled templates and records for hypothesis-driven experimental work.'),
    @('10_Benchmarks\README.md', 'Benchmarks', 'Defines reproducible evaluations while storing generated results outside source control.'),
    @('11_Projects\README.md', 'Projects', 'Contains deliverable projects that consume shared platform capabilities.'),
    @('12_Research\README.md', 'Research', 'Tracks literature, hypotheses, replications, reviews, and draft research.'),
    @('13_APIs\README.md', 'API Integrations', 'Owns external-service contracts, provider adapters, authentication hooks, rate limits, and retries.'),
    @('14_Deployments\README.md', 'Deployments', 'Contains reviewed deployment definitions and scripts; it does not deploy during initialization.'),
    @('15_Monitoring\README.md', 'Monitoring', 'Defines observability configuration; generated telemetry belongs in C:\AGI_Runtime.'),
    @('16_Security\README.md', 'Security', 'Defines enforceable security governance, reviews, permissions, and incident practices.'),
    @('16_Security\ThreatModel.md', 'Threat Model', 'Documents assets, trust boundaries, threats, mitigations, and residual risk.'),
    @('16_Security\DataClassification.md', 'Data Classification', 'Defines data sensitivity levels and required handling controls.'),
    @('16_Security\RetentionPolicy.md', 'Retention Policy', 'Defines retention and deletion requirements by data class.'),
    @('16_Security\ToolPermissions.md', 'Tool Permissions', 'Defines tool risk levels, required permissions, approval gates, and audit requirements.'),
    @('16_Security\HumanApprovalPolicy.md', 'Human Approval Policy', 'Defines actions that require explicit human authorization.'),
    @('16_Security\IncidentResponse.md', 'Incident Response', 'Defines detection, containment, recovery, notification, and review procedures.'),
    @('16_Security\AuditPolicy.md', 'Audit Policy', 'Defines auditable events, integrity requirements, access, and retention.'),
    @('16_Security\SecretsManagement.md', 'Secrets Management', 'Defines approved secret storage, rotation, access, and incident procedures.'),
    @('17_UI\README.md', 'User Interfaces', 'Contains interface applications and shared presentation components.'),
    @('18_Documentation\README.md', 'Documentation', 'Provides maintained system, API, setup, user, and developer documentation.'),
    @('18_Documentation\Architecture\SystemOverview.md', 'System Overview', 'Describes the system context, stable boundaries, and major data flows.'),
    @('19_Backups\README.md', 'Backup Definitions', 'Contains backup policy and scripts only; actual backups must be stored on independent media.'),
    @('19_Backups\backup_policy.md', 'Backup Policy', 'Defines independent backup destinations, schedules, encryption, retention, and ownership.'),
    @('19_Backups\restore_procedure.md', 'Restore Procedure', 'Defines tested restoration and validation steps.'),
    @('Sandbox\README.md', 'Sandbox', 'Contains disposable exploratory work that has not yet earned a stable architectural home.')
)
foreach ($file in $MarkdownFiles) {
    Add-MarkdownFile -RelativePath $file[0] -Title $file[1] -Purpose $file[2]
}

$AgentProfiles = @{
    'researcher.yaml' = "id: researcher`r`ndescription: Researches with source attribution.`r`npermissions: []`r`napproval_required: true`r`n"
    'coder.yaml'      = "id: coder`r`ndescription: Proposes and validates code changes.`r`npermissions: []`r`napproval_required: true`r`n"
    'critic.yaml'     = "id: critic`r`ndescription: Evaluates evidence, assumptions, and failure modes.`r`npermissions: []`r`napproval_required: false`r`n"
    'business.yaml'   = "id: business`r`ndescription: Analyzes business questions without autonomous external action.`r`npermissions: []`r`napproval_required: true`r`n"
}
foreach ($entry in $AgentProfiles.GetEnumerator()) {
    New-SafeFile -Path (Join-Path $Root "02_Agents\Profiles\$($entry.Key)") -Content $entry.Value
}

$RootGitIgnore = @'
__pycache__/
*.py[cod]
.venv/
venv/
env/
node_modules/
.env
.env.*
!.env.example
*.key
*.pem
*.pfx
credentials.json
tokens.json
*.gguf
*.safetensors
*.bin
*.pt
*.pth
*.onnx
*.db
*.sqlite
*.sqlite3
*.log
.DS_Store
Thumbs.db
.vscode/
.idea/
08_Data/**
!08_Data/README.md
!08_Data/Schemas/**
!08_Data/Fixtures/**
!08_Data/Samples/**
09_Experiments/Active/**/data/
09_Experiments/Active/**/logs/
09_Experiments/Active/**/artifacts/
10_Benchmarks/Reports/**
Sandbox/*
!Sandbox/README.md
!Sandbox/.gitignore
'@
New-SafeFile -Path (Join-Path $Root '.gitignore') -Content ($RootGitIgnore.TrimStart() + "`r`n")

$SandboxGitIgnore = @'
*
!.gitignore
!README.md
'@
New-SafeFile -Path (Join-Path $Root 'Sandbox\.gitignore') -Content ($SandboxGitIgnore.TrimStart() + "`r`n")

$SecurityGitIgnore = @'
Secrets/
*.key
*.pem
*.pfx
*.env
.env
.env.*
credentials.json
tokens.json
'@
New-SafeFile -Path (Join-Path $Root '16_Security\.gitignore') -Content ($SecurityGitIgnore.TrimStart() + "`r`n")

$RootReadme = @'
# AGI Workspace

`C:\AGI` is a source-controlled engineering and research workspace for building governed AI systems.

The numbered folders separate administration, core runtime contracts, agent profiles, cognition, memory, knowledge, tools, models, versionable data, experiments, benchmarks, projects, research, API integrations, deployments, monitoring, security, user interfaces, documentation, and backup definitions. `Sandbox` is for disposable exploration.

Shared capabilities belong in their owning platform area rather than being copied into individual projects. New experimental work begins in `Sandbox` or `09_Experiments` and moves only after its boundary is proven. Runtime logs, databases, caches, downloaded models, uploads, and generated artifacts belong in `C:\AGI_Runtime`, not in this repository. Secrets must never be committed. Large model files must not be stored in Git. `19_Backups` holds policy and scripts only; real backups must use independent storage.
'@
New-SafeFile -Path (Join-Path $Root 'README.md') -Content ($RootReadme.TrimStart() + "`r`n")

$WorkspaceJson = @'
{
  "folders": [
    { "path": "C:\\AGI" },
    { "path": "C:\\AGI\\01_Core" },
    { "path": "C:\\AGI\\02_Agents" },
    { "path": "C:\\AGI\\09_Experiments" },
    { "path": "C:\\AGI\\11_Projects" },
    { "path": "C:\\AGI\\12_Research" }
  ],
  "settings": {}
}
'@
New-SafeFile -Path (Join-Path $Root 'AGI.code-workspace') -Content ($WorkspaceJson.TrimStart() + "`r`n")

$Requirements = @'
# Minimal shared development dependencies; install only within a project environment.
fastapi
uvicorn
pydantic
python-dotenv
httpx
requests
pytest
ruff
'@
New-SafeFile -Path (Join-Path $Root 'requirements-base.txt') -Content ($Requirements.TrimStart() + "`r`n")

$ComposeExample = @'
services:
  postgres:
    image: postgres:REPLACE_WITH_PINNED_VERSION
    environment:
      POSTGRES_USER: REPLACE_ME
      POSTGRES_PASSWORD: REPLACE_ME
      POSTGRES_DB: REPLACE_ME
    profiles: ["example"]
  mongodb:
    image: mongo:REPLACE_WITH_PINNED_VERSION
    profiles: ["example"]
  qdrant:
    image: qdrant/qdrant:REPLACE_WITH_PINNED_VERSION
    profiles: ["example"]
  redis:
    image: redis:REPLACE_WITH_PINNED_VERSION
    profiles: ["example"]
'@
New-SafeFile -Path (Join-Path $Root 'docker-compose.example.yml') -Content ($ComposeExample.TrimStart() + "`r`n")

New-SafeFile -Path (Join-Path $Root '09_Experiments\experiment_index.csv') -Content "experiment_id,title,status,start_date,end_date,owner,model,dataset,result_summary`r`n"
New-SafeFile -Path (Join-Path $Root '10_Benchmarks\benchmark_registry.csv') -Content "benchmark_id,name,category,version,metric,dataset,status,last_run`r`n"
New-SafeFile -Path (Join-Path $Root '12_Research\research_index.csv') -Content "research_id,title,topic,status,date_started,date_completed,related_project,publication_url`r`n"
New-SafeFile -Path (Join-Path $Root '14_Deployments\deployment_registry.csv') -Content "deployment_id,project,environment,platform,status,version,last_updated`r`n"

if (-not $SkipRuntime) {
    New-SafeDirectory -Path $RuntimeRoot
    Add-RelativeDirectories -Base $RuntimeRoot -Paths @(
        'logs', 'databases', 'caches', 'models', 'uploads', 'temp',
        'benchmark-results', 'experiment-artifacts'
    )
    New-SafeFile -Path (Join-Path $RuntimeRoot 'README.txt') -Content "Runtime state for C:\AGI. Keep this directory outside source control. Do not store credentials here in plaintext.`r`n"
}

$ScriptDestination = Join-Path $Root '00_Admin\Scripts\Initialize-AGIWorkspace.ps1'
if ($PSCommandPath -and ((Get-NormalizedPath $PSCommandPath) -ine (Get-NormalizedPath $ScriptDestination))) {
    New-SafeFile -Path $ScriptDestination -Content ([System.IO.File]::ReadAllText($PSCommandPath))
}

$ReportDirectory = Join-Path $Root '00_Admin\Reports'
New-SafeDirectory -Path $ReportDirectory
$PrimaryReport = Join-Path $ReportDirectory 'create_structure_report.txt'
if (Test-Path -LiteralPath $PrimaryReport) {
    $timestamp = Get-Date -Format 'yyyy-MM-dd_HHmmss_fff'
    $ReportPath = Join-Path $ReportDirectory "create_structure_report_$timestamp.txt"
}
else {
    $ReportPath = $PrimaryReport
}

$Report = @(
    'AGI workspace initialization report'
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')"
    "Source root: $(Get-NormalizedPath $Root)"
    $(if (-not $SkipRuntime) { "Runtime root: $(Get-NormalizedPath $RuntimeRoot)" } else { 'Runtime root: skipped by request' })
    ''
    "CREATED ($($Created.Count))"
    $Created
    ''
    "SKIPPED ($($Skipped.Count))"
    $Skipped
    ''
    "FAILED ($($Failed.Count))"
    $Failed
)
[System.IO.File]::WriteAllLines($ReportPath, [string[]]$Report, [System.Text.UTF8Encoding]::new($false))
Write-Host "Report: $ReportPath"

if ($Failed.Count -gt 0) {
    throw "Initialization completed with $($Failed.Count) failure(s). Review the report."
}
