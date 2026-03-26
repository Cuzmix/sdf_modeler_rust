# Platform Targets

This document tracks the current Slint host status across desktop, mobile, and web targets.

## Current Direction

The app is now structured as a shared Rust/Slint library with target-specific startup paths:

- shared app/core logic stays in the existing toolkit-neutral modules
- shared Slint host runtime stays in `src/app/slint_frontend/`
- target bootstrap lives in `src/platform.rs`
- desktop-only file dialog behavior stays isolated in `src/desktop_dialogs.rs`

The goal is to keep the editor core shared while letting the outer host vary by target.

## Desktop

Desktop is the active supported target today.

Current status:

- `src/main.rs` launches the desktop Slint host through `sdf_modeler_lib::run_native()`
- `src/bin/slint_host.rs` remains available as a desktop-only helper binary
- Windows release builds now use the `windows` subsystem to avoid a stray console window
- the native WGPU bootstrap is target-aware instead of hardcoded to one backend path

Backend selection:

- Windows: DX12
- macOS / iOS: Metal
- Android: Vulkan
- other native targets: `wgpu::Backends::PRIMARY`

## Android

Android is now prepared at the crate/bootstrap layer, but not fully product-ready.

What is in place:

- the library crate now exports both `rlib` and `cdylib`
- `slint` enables `backend-android-activity-06`
- `src/lib.rs` exports `android_main`
- `android_main` delegates to `src/platform.rs`
- `src/app/slint_frontend/mod.rs` has an Android-capable Slint host entrypoint

What is still missing locally:

- a configured Android SDK/NDK environment on this machine
- `ANDROID_NDK` was not set during validation, so the Android target check stopped in Skia's Android build step

Current Android product limitations:

- desktop file dialogs are intentionally unsupported on Android
- project open/save, reference-image pickers, HDRI pickers, screenshot destinations, and mesh import/export still need Android-native replacements
- the UI is only beginning its tablet-first adaptation and is not yet tuned for touch-first device workflows

## iOS

iOS is not wired yet, but the new structure is deliberately moving in that direction.

What already helps:

- the shared host/bootstrap split avoids burying startup logic in `main.rs`
- the renderer bootstrap already chooses Metal for Apple targets
- the crate now supports `cdylib`, which is also needed for mobile packaging flows

What is still missing:

- an iOS-specific entry/build flow
- macOS/Xcode-side project generation and signing setup
- device/simulator validation on macOS

## Web

Web is not a supported product target yet.

Important constraints from Slint:

- Rust is the only supported language for Slint WebAssembly
- Slint-on-web uses the Winit backend with FemtoVG and renders into a `<canvas>`
- Slint explicitly frames web as better for demos or secondary targets than as a primary general-purpose web app surface

Current repo status:

- the crate now supports `cdylib`, which is one prerequisite for Wasm packaging
- the actual host is still native-first and pulls in desktop/native renderer dependencies
- a quick `wasm32-unknown-unknown` check still fails in desktop/native dependency paths, so web support needs a dedicated target-specific host/config split rather than a small patch

## Validation Status

Platform-prep validation completed for the desktop path:

1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. desktop smoke launch

Cross-target status:

- `cargo check --lib --target aarch64-linux-android`
  blocked by local environment: `ANDROID_NDK` not configured
- `cargo check --lib --target wasm32-unknown-unknown`
  not ready yet due to native/desktop renderer dependencies in the current host stack

## Next Steps

Recommended follow-up order:

1. Continue the tablet-first Slint UI work so desktop and Android converge on the same touch-capable shell.
2. Add Android-native host services for open/save/import/export/reference-image/HDRI flows.
3. Add safe-area-aware overlays and touch-friendly controls.
4. Introduce a real Android build path once the local SDK/NDK is configured.
5. Treat web as a separate target track with its own Slint feature/config split.
