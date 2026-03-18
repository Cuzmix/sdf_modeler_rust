# Android APK Build

This repo now has the minimum Android packaging hooks needed for `cargo-apk2`:

- `cdylib` output from the library target
- `eframe` Android NativeActivity feature enabled
- a `#[no_mangle] fn android_main(app: AndroidApp)` entrypoint
- `cargo-apk2` manifest metadata in [`Cargo.toml`](../Cargo.toml)

## Required local tooling

Install and configure these before building:

```powershell
rustup target add aarch64-linux-android
cargo install cargo-apk2
```

Set environment variables so `cargo-apk2` can find the Android toolchain:

```powershell
$env:ANDROID_SDK_ROOT="C:\Android\Sdk"
$env:ANDROID_NDK_HOME="C:\Android\Sdk\ndk\<ndk-version>"
$env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
```

For the current repo config, make sure the SDK has an API 35 platform installed. The
default Android target in [`Cargo.toml`](../Cargo.toml) is pinned to API 35 because the
NDK version used in local testing only supports up to API 35.

## Build

Debug APK:

```powershell
cargo apk2 build --lib
```

Repo helper script:

```powershell
.\scripts\build_android_debug.ps1
```

Build and install to a connected device:

```powershell
.\scripts\build_android_debug.ps1 -Install
```

Release APK:

```powershell
cargo apk2 build --lib --release
```

The generated APK is written under the target output directory used by `cargo-apk2`.

## Windows path caveat

On Windows, `cargo-apk2` can fail if the Android SDK path or APK target directory contains
spaces. If that happens:

1. Use an SDK path without spaces.
2. Use a `--target-dir` path without spaces.
3. Keep using `--lib`, since this app packages the Android `cdylib`, not the desktop bin.

Example debug build:

```powershell
$env:ANDROID_HOME="C:\Users\<you>\android-sdk-sdfmodeler"
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:ANDROID_NDK_HOME="$env:ANDROID_HOME\ndk\27.2.12479018"
$env:ANDROID_USER_HOME="C:\Users\<you>\.android-sdfmodeler"
cargo apk2 build --lib --target-dir "C:\Users\<you>\target-sdfmodeler-android"
```

## Current Android behavior

This first pass makes the crate Android-buildable, but it does not yet provide Android-native file pickers or export destinations.

Current Android limitations:

- project open/save dialogs are disabled
- node preset import/export dialogs are disabled
- reference image picker is disabled
- mesh import/export dialogs are disabled
- screenshot save dialog is disabled
- settings/keybinding import/export dialogs are disabled

What does work:

- settings, autosave metadata, and material presets now write to an app-private storage location instead of next to the executable
- the Android shell can launch through `NativeActivity`

## Next Android-specific work

If you want full device usability instead of just APK packaging, the next slices should be:

1. Add Android share/open/save flows or SAF document pickers.
2. Add an Android-native destination for screenshots and mesh exports.
3. Review touch-first layout and viewport navigation on a tablet.
4. Run `cargo check` for `aarch64-linux-android` and a real `cargo apk2 build` on a machine with the SDK/NDK installed.
