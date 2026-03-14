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

function Resolve-ExistingFilePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf -ErrorAction Stop) {
            return (Resolve-Path -LiteralPath $LiteralPath -ErrorAction Stop).Path
        }
    } catch {
        return $null
    }

    return $null
}

function Get-WhereMatches {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    try {
        $whereOutput = & cmd.exe /d /c "where $CommandName 2>nul"
        if ($LASTEXITCODE -eq 0) {
            return @($whereOutput | Where-Object { $_ })
        }
    } catch {
        return @()
    }

    return @()
}

function Resolve-CodexPath {
    if ($env:RALPH_CODEX_EXE) {
        $resolvedConfiguredPath = Resolve-ExistingFilePath -LiteralPath $env:RALPH_CODEX_EXE
        if ($resolvedConfiguredPath) {
            return $resolvedConfiguredPath
        }

        throw "RALPH_CODEX_EXE is set but does not exist: $env:RALPH_CODEX_EXE"
    }

    $npmShimCandidates = @(
        (Join-Path $env:APPDATA "npm\codex.cmd"),
        (Join-Path $env:APPDATA "npm\codex.ps1"),
        (Join-Path $env:APPDATA "npm\codex.exe")
    )

    foreach ($candidatePath in $npmShimCandidates) {
        $resolvedCandidatePath = Resolve-ExistingFilePath -LiteralPath $candidatePath
        if ($resolvedCandidatePath) {
            return $resolvedCandidatePath
        }
    }

    foreach ($commandName in @("codex", "codex.cmd", "codex.ps1", "codex.exe")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $whereResults = @(
        Get-WhereMatches -CommandName "codex.cmd"
        Get-WhereMatches -CommandName "codex.ps1"
        Get-WhereMatches -CommandName "codex.exe"
    ) | Where-Object { $_ }

    foreach ($whereResult in @($whereResults)) {
        $resolvedWherePath = Resolve-ExistingFilePath -LiteralPath $whereResult
        if ($resolvedWherePath) {
            return $resolvedWherePath
        }
    }

    $localAliasPath = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\codex.exe"
    $resolvedAliasPath = Resolve-ExistingFilePath -LiteralPath $localAliasPath
    if ($resolvedAliasPath) {
        return $resolvedAliasPath
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
        $resolvedInstallPath = Resolve-ExistingFilePath -LiteralPath $candidatePath
        if ($resolvedInstallPath) {
            return $resolvedInstallPath
        }
    }

    throw "Unable to locate Codex CLI. Install Codex or set RALPH_CODEX_EXE to the full codex.exe path."
}

$promptText = Get-Content -LiteralPath $PromptFile -Raw

$codexPath = Resolve-CodexPath
$executionDirectory = (Get-Location).Path

switch ($Mode) {
    "Interactive" {
        & $codexPath -C $executionDirectory $promptText
        exit $LASTEXITCODE
    }
    "Headless" {
        $lastMessageFile = New-TemporaryFile
        try {
            $headlessArguments = @(
                "exec",
                "--full-auto",
                "--color", "never",
                "-C", $executionDirectory,
                "--output-last-message", $lastMessageFile.FullName,
                "-"
            )

            $promptText | & $codexPath @headlessArguments
            $exitCode = $LASTEXITCODE

            if (Test-Path -LiteralPath $lastMessageFile.FullName -PathType Leaf) {
                Get-Content -LiteralPath $lastMessageFile.FullName -Raw
            }

            exit $exitCode
        } finally {
            Remove-Item -LiteralPath $lastMessageFile.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}
