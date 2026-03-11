param(
    [switch]$Watch
)

$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$fvmFlutterBin = Join-Path $repoRoot '.fvm\flutter_sdk\bin'
$env:Path = "$fvmFlutterBin;$env:Path"

$arguments = @('generate', '--config-file', 'flutter_rust_bridge.yaml')
if ($Watch) {
    $arguments += '--watch'
}

Push-Location $projectRoot
try {
    & flutter_rust_bridge_codegen @arguments
} finally {
    Pop-Location
}
