param(
    [switch]$Install,
    [string]$DeviceId,
    [switch]$RefreshSdkCopy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $scriptDir = Split-Path -Parent $PSCommandPath
    return Split-Path -Parent $scriptDir
}

function Resolve-JavaHome {
    if ($env:JAVA_HOME) {
        $javaExe = Join-Path $env:JAVA_HOME "bin\\java.exe"
        if (Test-Path $javaExe) {
            return $env:JAVA_HOME
        }
    }

    $androidStudioJbr = "C:\\Program Files\\Android\\Android Studio\\jbr"
    if (Test-Path (Join-Path $androidStudioJbr "bin\\java.exe")) {
        return $androidStudioJbr
    }

    $jetbrainsRoot = "C:\\Program Files\\JetBrains"
    if (Test-Path $jetbrainsRoot) {
        $rustRoverDirs = Get-ChildItem $jetbrainsRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "RustRover*" } |
            Sort-Object Name -Descending

        foreach ($rustRoverDir in $rustRoverDirs) {
            $jbrPath = Join-Path $rustRoverDir.FullName "jbr"
            if (Test-Path (Join-Path $jbrPath "bin\\java.exe")) {
                return $jbrPath
            }
        }
    }

    throw "JAVA_HOME is not set and no Android Studio or RustRover JBR was found."
}

function Resolve-NdkPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SdkPath
    )

    $preferredNdkPath = Join-Path $SdkPath "ndk\\27.2.12479018"
    if (Test-Path $preferredNdkPath) {
        return $preferredNdkPath
    }

    $ndkRoot = Join-Path $SdkPath "ndk"
    if (-not (Test-Path $ndkRoot)) {
        throw "No Android NDK directory found under '$SdkPath'."
    }

    $ndkCandidates = Get-ChildItem $ndkRoot -Directory | Sort-Object Name -Descending
    if ($ndkCandidates.Count -eq 0) {
        throw "No Android NDK versions found under '$ndkRoot'."
    }

    return $ndkCandidates[0].FullName
}

function Ensure-SdkCopy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$UserHome,
        [Parameter(Mandatory = $true)]
        [bool]$RefreshCopy
    )

    $repoSdkPath = Join-Path $RepoRoot ".android-sdk"
    if (-not (Test-Path $repoSdkPath)) {
        throw "Repo-local Android SDK not found at '$repoSdkPath'."
    }

    $pathHasSpaces = $RepoRoot.Contains(" ") -or $repoSdkPath.Contains(" ")
    if (-not $pathHasSpaces) {
        return $repoSdkPath
    }

    $sdkCopyPath = Join-Path $UserHome "android-sdk-sdfmodeler"
    if ($RefreshCopy -or -not (Test-Path $sdkCopyPath)) {
        New-Item -ItemType Directory -Path $sdkCopyPath -Force | Out-Null
        Write-Host "Syncing Android SDK to '$sdkCopyPath'..."
        & robocopy $repoSdkPath $sdkCopyPath /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -gt 7) {
            throw "robocopy failed while preparing the Android SDK copy. Exit code: $LASTEXITCODE"
        }
    }

    return $sdkCopyPath
}

function Ensure-AndroidPlatform35 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SdkPath
    )

    $platform35 = Join-Path $SdkPath "platforms\\android-35"
    if (-not (Test-Path $platform35)) {
        throw "Android platform 35 is missing at '$platform35'. Install it before building."
    }
}

function Resolve-AdbPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SdkPath
    )

    $adbPath = Join-Path $SdkPath "platform-tools\\adb.exe"
    if (-not (Test-Path $adbPath)) {
        throw "adb.exe not found at '$adbPath'."
    }

    return $adbPath
}

function Resolve-InstallTarget {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AdbPath,
        [string]$ExplicitDeviceId
    )

    if ($ExplicitDeviceId) {
        return $ExplicitDeviceId
    }

    $deviceOutput = & $AdbPath devices
    $connectedDevices = @(
        $deviceOutput |
        Select-Object -Skip 1 |
        Where-Object { $_ -match "^\S+\s+device$" } |
        ForEach-Object { ($_ -split "\s+")[0] }
    )

    if ($connectedDevices.Count -eq 0) {
        throw "No connected Android device was found. Connect a device or pass -DeviceId."
    }

    if ($connectedDevices.Count -gt 1) {
        throw "Multiple Android devices are connected. Re-run with -DeviceId <serial>."
    }

    return $connectedDevices[0]
}

$repoRoot = Get-RepoRoot
$userHome = [Environment]::GetFolderPath("UserProfile")
$javaHome = Resolve-JavaHome
$sdkPath = Ensure-SdkCopy -RepoRoot $repoRoot -UserHome $userHome -RefreshCopy:$RefreshSdkCopy
$ndkPath = Resolve-NdkPath -SdkPath $sdkPath
$androidUserHome = Join-Path $userHome ".android-sdfmodeler"
$targetDir = Join-Path $userHome "target-sdfmodeler-android"

Ensure-AndroidPlatform35 -SdkPath $sdkPath
New-Item -ItemType Directory -Path $androidUserHome -Force | Out-Null
New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

$env:JAVA_HOME = $javaHome
$env:ANDROID_HOME = $sdkPath
$env:ANDROID_SDK_ROOT = $sdkPath
$env:ANDROID_NDK_HOME = $ndkPath
$env:ANDROID_USER_HOME = $androidUserHome

Push-Location $repoRoot
try {
    & cargo apk2 build --lib --target-dir $targetDir
    if ($LASTEXITCODE -ne 0) {
        throw "Android debug build failed."
    }

    $apkPath = Join-Path $targetDir "debug\\apk\\sdf-modeler.apk"
    if (-not (Test-Path $apkPath)) {
        throw "Expected APK was not created at '$apkPath'."
    }

    Write-Host "Debug APK ready: $apkPath"

    if ($Install) {
        $adbPath = Resolve-AdbPath -SdkPath $sdkPath
        $installTarget = Resolve-InstallTarget -AdbPath $adbPath -ExplicitDeviceId $DeviceId
        Write-Host "Installing to device '$installTarget'..."
        & $adbPath -s $installTarget install -r $apkPath
        if ($LASTEXITCODE -ne 0) {
            throw "adb install failed."
        }
    }
}
finally {
    Pop-Location
}
