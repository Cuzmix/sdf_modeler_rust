# SDF Modeler - Claude Code Instructions

## Coding Standards (MANDATORY)

### Architecture And File Organization
- No monolithic files. Keep modules small, discoverable, and responsibility-focused.
- Group related functionality into folders or submodules.
- Follow the current native desktop architecture instead of re-introducing old `egui` patterns.

### Readability And Naming
- Write code that is readable to junior Rust developers.
- Prefer explicit names over short names.
- Do not use single-letter variables except for standard math or loop cases.

### Programming Practices
- Follow Rust idioms and keep control flow direct.
- Implement only what is required.
- Search the codebase before adding new code to avoid duplicates.
- Do not add placeholders, TODO stubs, or partial implementations.

### Pushback And Decision Quality
- Push back on technically unsound or maintainability-hostile requests.
- Prefer the smallest correct fix over the broadest possible change.
- Keep engine/debug complexity out of product-facing UX unless it is clearly needed.

### Performance (Top Priority)
- Do not regress runtime performance, memory use, or sculpt responsiveness.
- Prefer focused in-repo implementations over unnecessary dependencies.
- Keep export writers dependency-free (`std::io` only where practical).

### Verification Loops (MANDATORY Before Every Commit)
Every commit must pass:
1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. Manual verification for visual or behavioral changes

### Git Discipline
- One logical change per commit.
- Use descriptive commit messages that explain why.
- Keep commits focused to reduce merge conflicts.

---

## Project Overview

SDF Modeler is a native desktop Signed Distance Function (SDF) 3D modeling application. It renders SDF scenes on the GPU via raymarching, supports voxel sculpting, and exports meshes via marching cubes.

### Current Tech Stack

| Component | Library |
|-----------|---------|
| UI | Slint 1.15.1 |
| GPU | wgpu 27.0.1 |
| Math | glam 0.29 |
| Serialization | serde + serde_json + bytemuck |
| File dialogs | rfd |
| Parallelism | rayon |

### Native Desktop Entry Point

- `src/main.rs`
  Native launcher.
- `src/lib.rs`
  Module declarations and `run_native()`, which starts the Slint host.

### Current Source Tree

```text
src/
├── app/
│   ├── mod.rs                  # Shared app construction and state composition
│   ├── actions.rs              # Structural actions
│   ├── action_handler.rs       # process_actions() structural mutation gate
│   ├── backend_frame.rs        # Toolkit-neutral frame lifecycle
│   ├── controllers.rs          # Shared non-structural edit controllers
│   ├── frontend_models.rs      # Presenter models for the Slint shell
│   ├── slint_bridge.rs         # Slint event/input decoding
│   ├── slint_frontend/         # Slint runtime host, bindings, callbacks
│   └── slint_ui/               # Declarative .slint components and view models
├── gizmo/                      # Viewport gizmo math, selection, and overlays
├── gpu/                        # GPU buffers, camera, codegen, picking
├── graph/                      # Scene graph, presented objects, history, voxel data
├── viewport/                   # Shared viewport rendering and overlays
├── shaders/                    # WGSL shaders
├── desktop_dialogs.rs          # Native dialog/service helpers
├── export.rs                   # Marching cubes and mesh writers
├── io.rs                       # Project file persistence
├── native_wgpu.rs              # Native WGPU device descriptor helpers
└── sculpt.rs                   # Sculpt tool definitions and brush state
```

For a file-by-file Slint frontend map, see `docs/slint_frontend.md`.

---

## UI/Backend Separation (MANDATORY)

The native desktop UI is Slint-based, but the app still keeps a strict boundary between toolkit code and shared logic.

- Keep toolkit-agnostic frame logic in `src/app/backend_frame.rs`.
- Keep shared app construction and state in `src/app/mod.rs` and `src/app/state.rs`.
- Keep structural mutations in `src/app/action_handler.rs` through `process_actions()`.
- Keep Slint input decoding in `src/app/slint_bridge.rs`.
- Keep Slint runtime/binding/callback wiring in `src/app/slint_frontend/`.
- Keep declarative Slint UI and typed UI contracts in `src/app/slint_ui/`.
- Keep `src/app/frontend_models.rs` as the presenter layer between app/core state and Slint.

When extending the UI, add domain-level Slint callbacks and grouped state models instead of putting business logic into `.slint` files.

Reference: `docs/ui_backend_boundary.md`

---

## Sculpt Responsiveness Non-Regression (MANDATORY)

Sculpt smoothness is a hard quality bar.

Preserve:
- non-blocking async pick behavior during active sculpt
- predictive sculpt fallback while async pick is pending
- off-mesh Grab continuation until mouse release
- voxel-aware interpolation density
- per-sample delta clamping for Add/Carve/Flatten/Inflate
- Taubin smoothing behavior for Smooth

Manual verification for sculpt input or render changes:
1. Fast circular strokes remain continuous.
2. Grab drag continues when leaving geometry until release.
3. Toggling Shadows or AO during sculpt does not introduce stepping.
4. No latency spike appears at stroke start or during continuous drag.

Reference: `docs/sculpt_responsiveness_findings.md`

---

## Build And Run

```bash
# Native desktop
cargo run

# Release build
cargo run --release

# Validation
cargo check
cargo clippy -- -D warnings
cargo test
cargo build
```

### Build Notes

- `build.rs` compiles `src/app/slint_ui/slint_host_window.slint` with `slint-build`.
- The native host uses Slint with the shared WGPU 27 backend.
- The project is native-desktop-first; old `egui` and `eframe` paths are no longer part of the active app.

---

## Additional References

- `docs/architecture.md`
- `docs/slint_frontend.md`
- `docs/ui_backend_boundary.md`
- `docs/sculpt_responsiveness_findings.md`
