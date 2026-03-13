param(
    [switch]$Watch
)

$ErrorActionPreference = 'Stop'
$expectedCodegenVersion = '2.11.1'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$fvmFlutterBin = Join-Path $repoRoot '.fvm\flutter_sdk\bin'
$env:Path = "$fvmFlutterBin;$env:Path"

$codegenCommand = Get-Command flutter_rust_bridge_codegen -ErrorAction SilentlyContinue
if ($null -eq $codegenCommand) {
    throw "flutter_rust_bridge_codegen is not installed. Install it with: cargo install flutter_rust_bridge_codegen --version $expectedCodegenVersion --locked"
}

$installedVersion = & $codegenCommand.Path --version
if ($installedVersion -notmatch [regex]::Escape($expectedCodegenVersion)) {
    throw "Expected flutter_rust_bridge_codegen $expectedCodegenVersion but found '$installedVersion'. Install the matching version with: cargo install flutter_rust_bridge_codegen --version $expectedCodegenVersion --locked"
}

$arguments = @(
    'generate',
    '--rust-input', 'crate::api',
    '--rust-root', 'rust',
    '--dart-output', 'lib/src/rust',
    '--dart-root', '.',
    '--dart-entrypoint-class-name', 'RustLib',
    '--stop-on-error'
)
if ($Watch) {
    $arguments += '--watch'
}

Push-Location $projectRoot
try {
    & $codegenCommand.Path @arguments
} finally {
    Pop-Location
}

