# Android Build Prep

This repo is no longer using the old `eframe` Android path. The current Android prep is based on the Slint NativeActivity flow.

For the broader cross-target picture, see [platform_targets.md](./platform_targets.md).

## What Exists Now

The crate is prepared for Android at the bootstrap level:

- `Cargo.toml` enables `cdylib` output for the library target
- `slint` enables `backend-android-activity-06`
- `src/lib.rs` exports `android_main`
- `src/platform.rs` owns the Android startup handoff
- `src/app/slint_frontend/mod.rs` can initialize the Slint host from `android_main`

The desktop-only file dialog layer remains intentionally disabled on Android.

## Required Local Tooling

Before Android validation/builds can succeed locally, the machine needs:

- Rust Android target:

```powershell
rustup target add aarch64-linux-android
```

- Android SDK and NDK installed locally
- environment variables that let Cargo/Skia find the Android toolchain, especially:

```powershell
$env:ANDROID_NDK="C:\Android\Sdk\ndk\<version>"
$env:ANDROID_SDK_ROOT="C:\Android\Sdk"
```

Depending on the build flow you use later, `ANDROID_HOME`, `JAVA_HOME`, and related Android Studio tooling may also be required.

## Current Validation Result

The repo now gets past source-level Android setup, but local validation is still blocked by machine configuration:

```powershell
cargo check --lib --target aarch64-linux-android
```

Current blocker on this machine:

- `ANDROID_NDK` is not configured, so Skia's Android build step aborts before the target check can complete

## Product Gaps Still To Solve

Even after the NDK is configured, Android is not yet a finished product target.

Still needed:

- Android-native open/save/import/export flows
- Android-native destinations for screenshots and mesh exports
- reference-image and HDRI picking on device
- touch-first overlay controls and safe-area-aware layout
- a real install/package/test path on hardware or emulator

## Recommended Next Android Slice

1. Configure SDK/NDK locally and rerun `cargo check --lib --target aarch64-linux-android`.
2. Add Android-native host services for document and media flows.
3. Continue the tablet-first Slint shell work so the UI is usable on touch devices.
