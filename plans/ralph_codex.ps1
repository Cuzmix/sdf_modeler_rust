param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Interactive", "Headless")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [string]$WorkingDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($WorkingDirectory) {
    Set-Location -LiteralPath $WorkingDirectory
}

function Resolve-CodexPath {
    if ($env:RALPH_CODEX_EXE) {
        if (Test-Path -LiteralPath $env:RALPH_CODEX_EXE) {
            return (Resolve-Path -LiteralPath $env:RALPH_CODEX_EXE).Path
        }

        throw "RALPH_CODEX_EXE is set but does not exist: $env:RALPH_CODEX_EXE"
    }

    $npmShimCandidates = @(
        (Join-Path $env:APPDATA "npm\codex.cmd"),
        (Join-Path $env:APPDATA "npm\codex.ps1"),
        (Join-Path $env:APPDATA "npm\codex.exe")
    )

    foreach ($candidatePath in $npmShimCandidates) {
        if (Test-Path -LiteralPath $candidatePath) {
            return (Resolve-Path -LiteralPath $candidatePath).Path
        }
    }

    foreach ($commandName in @("codex", "codex.cmd", "codex.ps1", "codex.exe")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $whereResults = @(
        where.exe codex.cmd 2>$null
        where.exe codex.ps1 2>$null
        where.exe codex.exe 2>$null
    ) | Where-Object { $_ }
    if ($whereResults.Count -gt 0) {
        return $whereResults[0]
    }

    $localAliasPath = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\codex.exe"
    if (Test-Path -LiteralPath $localAliasPath) {
        return $localAliasPath
    }

    $windowsAppsRoot = Join-Path $env:ProgramFiles "WindowsApps"
    $codexInstallDirectories = Get-ChildItem `
        -Path $windowsAppsRoot `
        -Directory `
        -Filter "OpenAI.Codex_*" `
        -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending

    foreach ($installDirectory in $codexInstallDirectories) {
        $candidatePath = Join-Path $installDirectory.FullName "app\resources\codex.exe"
        if (Test-Path -LiteralPath $candidatePath) {
            return $candidatePath
        }
    }

    throw "Unable to locate Codex CLI. Install Codex or set RALPH_CODEX_EXE to the full codex.exe path."
}

$promptText = Get-Content -LiteralPath $PromptFile -Raw

$codexPath = Resolve-CodexPath

switch ($Mode) {
    "Interactive" {
        & $codexPath $promptText
        exit $LASTEXITCODE
    }
    "Headless" {
        & $codexPath exec $promptText
        exit $LASTEXITCODE
    }
}
