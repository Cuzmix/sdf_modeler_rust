# Repository Memory

## Purpose

`sdf_modeler_rust` is a native desktop Signed Distance Function 3D modeler. The core product areas are parametric SDF scene editing, voxel sculpting, GPU raymarched rendering with WGSL, and mesh import/export.

## Active Architecture

- The active app is Slint-based native desktop UI plus a shared `wgpu` 27 rendering backend.
- Legacy `egui`/`eframe` paths are not part of the active architecture and should not be reintroduced.
- `src/main.rs` is the native entry point. `src/lib.rs` declares modules and routes startup through `platform::run_desktop()`.

## Layer Map

1. `src/app/`
   Shared app/core layer. Owns frame lifecycle, state composition, structural actions, reducers, presenter models, viewport interaction, sculpt orchestration, and Slint host integration.
2. `src/graph/`
   Scene graph, presented-object layer, undo/redo history, and CPU voxel data.
3. `src/gpu/`
   Buffer encoding, camera uniforms, runtime WGSL generation, and GPU picking.
4. `src/viewport/`
   Shared viewport rendering, compositing, environment resources, and overlays.
5. `src/gizmo/`
   Viewport gizmo interaction, transform math, and multi-selection helpers.
6. Top-level support modules
   `export.rs`, `io.rs`, `mesh_import.rs`, `desktop_dialogs.rs`, `settings.rs`, `platform.rs`, `sculpt.rs`.

## UI And Mutation Boundary

- Keep toolkit-agnostic frame logic in `src/app/backend_frame.rs`.
- Keep shared app construction/state in `src/app/mod.rs` and `src/app/state.rs`.
- Keep structural mutations behind `Action` values and `process_actions()` in `src/app/action_handler.rs`.
- Keep high-frequency but toolkit-neutral edit logic in `src/app/controllers.rs` and `src/app/viewport_interaction.rs`.
- Keep Slint event decoding in `src/app/slint_bridge.rs`.
- Keep Slint runtime, bindings, callbacks, and viewport texture handling in `src/app/slint_frontend/`.
- Keep declarative UI and typed UI contracts in `src/app/slint_ui/`.
- `.slint` files should stay declarative and emit domain intents, not business logic.

## Runtime Flow

1. Slint emits UI and viewport events.
2. `src/app/slint_bridge.rs` converts them into backend-friendly input snapshots.
3. `src/app/backend_frame.rs` runs pre-UI backend work.
4. `src/app/viewport_interaction.rs` handles camera, picking, sculpt, and gizmo interaction.
5. `src/app/frontend_models.rs` builds a `ShellSnapshot`.
6. `src/app/slint_frontend/bindings/` applies grouped state to the Slint window.
7. Slint renders the shell and viewport image.
8. `src/app/slint_frontend/callbacks/` converts UI intent into `Action` values or shared controller calls.
9. `src/app/action_handler.rs` applies structural mutations.
10. `src/app/backend_frame.rs` runs post-UI backend work.

## Performance And Sculpt Guardrails

- Performance, memory use, and responsiveness are top-priority constraints.
- Keep export writers dependency-light and `std::io`-oriented where practical.
- For sculpt work, preserve:
  - non-blocking async pick behavior
  - predictive sculpt fallback while pick is pending
  - off-mesh Grab continuation until release
  - voxel-aware interpolation density
  - per-sample delta clamping for Add/Carve/Flatten/Inflate
  - Taubin smoothing for Smooth
- Manual checks are required for sculpt-related changes: continuous fast strokes, Grab continuity off-mesh, no stepping when toggling Shadows/AO, no latency spikes at stroke start or during drag.

## Known Regression Hotspot

- Multi-select transform gizmo rotation snap-back can regress if per-frame incremental drag values are mixed with baseline re-application flows.
- First places to inspect:
  - `src/gizmo/viewport.rs`
  - `src/viewport/draw.rs`
  - `src/app/state.rs`
- Guardrail:
  - incremental flows use `ScreenRotationDragState::consume_applied_delta`
  - baseline re-apply flows use `ScreenRotationDragState::consume_applied_total`

## Windows Skia Toolchain Bring-Up (2026-03-28)

- `cargo run` can fail early in `third_party/skia-bindings` if `.cargo/config.toml` points to missing repo-local tools:
  - `LLVM_HOME=.tools/llvm`
  - `LIBCLANG_PATH=.tools/llvm/bin`
  - `SKIA_NINJA_COMMAND=.tools/ninja/ninja.exe`
  - `SKIA_SOURCE_DIR=.tools/skia`
- Working bootstrap sequence:
  1. Install LLVM and Ninja with `winget`.
  2. Create `.tools/llvm` as a junction to `C:\Program Files\LLVM`.
  3. Copy `ninja.exe` into `.tools/ninja/ninja.exe`.
  4. Clone Skia source into `.tools/skia`:
     `git clone --depth 1 --branch m142-0.89.1 https://github.com/rust-skia/skia .tools/skia`
- Why clone was needed:
  - The tarball download/extract path hit Windows symlink privilege error `os error 1314` during unpack.
  - A direct git clone into `.tools/skia` bypassed that extraction failure.
- Current status after bootstrap:
  - `cargo run` compiles and launches `target\debug\sdf_modeler.exe`.
  - One observed follow-up in this environment: a shutdown-time TLS panic (`AccessError`) with `STATUS_STACK_BUFFER_OVERRUN` after process exit path. Treat as a separate runtime issue from toolchain setup.

## Expected Validation Order

1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. Manual visual or behavioral verification when applicable

## Good First Reads For Future Work

- `AGENTS.md`
- `CLAUDE.md`
- `docs/architecture.md`
- `docs/ui_backend_boundary.md`
- `docs/slint_frontend.md`
- `docs/sculpt_responsiveness_findings.md`
