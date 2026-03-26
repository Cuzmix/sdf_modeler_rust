# SDF Modeler Architecture

This document is the current architecture reference for the native Slint-based desktop app.

## 1. High-Level Overview

SDF Modeler is a real-time Signed Distance Function (SDF) 3D modeling application.

Core capabilities:

- parametric SDF scene editing
- voxel sculpting
- GPU raymarching with WGSL shaders
- mesh import and export
- native desktop editing through Slint

The current application is organized into five layers:

```text
┌─────────────────────────────────────────────────────────┐
│ Slint UI (`src/app/slint_ui/`)                         │
│ Declarative components, panel state structs, actions   │
├─────────────────────────────────────────────────────────┤
│ Slint Host (`src/app/slint_frontend/`, bridge)         │
│ Runtime host, bindings, callbacks, viewport transport  │
├─────────────────────────────────────────────────────────┤
│ App Core (`src/app/`)                                  │
│ Frame lifecycle, actions, controllers, state           │
├─────────────────────────────────────────────────────────┤
│ Scene Graph (`src/graph/`)                             │
│ Nodes, history, presented objects, voxel data          │
├─────────────────────────────────────────────────────────┤
│ GPU + Viewport (`src/gpu/`, `src/viewport/`, shaders)  │
│ WGSL codegen, buffers, picking, raymarching, overlays  │
└─────────────────────────────────────────────────────────┘
```

## 2. Current Tech Stack

| Component | Library |
|-----------|---------|
| UI | Slint 1.15.1 |
| GPU | wgpu 27.0.1 |
| Math | glam 0.29 |
| Serialization | serde + serde_json + bytemuck |
| File dialogs | rfd |
| Parallelism | rayon |
| Logging | log + env_logger |

`build.rs` compiles `src/app/slint_ui/slint_host_window.slint` with `slint-build`, and `src/app/slint_frontend/mod.rs` loads the generated modules with `slint::include_modules!()`.

## 3. Entry Points

- `src/main.rs`
  Native binary entry point.
- `src/lib.rs`
  Declares modules and calls `app::slint_frontend::run_slint_host(settings)`.

The desktop app is Slint-only. `egui`, `eframe`, `egui_dock`, and `egui_node_graph2` are no longer part of the active codebase.

## 4. Core Module Map

### `src/app/`

- `mod.rs`
  Shared app construction, `SdfApp`, async status types, and top-level state composition.
- `state.rs`
  Shared app state, including selection, expert-panel registry, viewport interaction state, and multi-transform session state.
- `actions.rs`
  Structural `Action` definitions.
- `action_handler.rs`
  `process_actions()`, the structural mutation gate.
- `backend_frame.rs`
  Toolkit-neutral frame lifecycle.
- `controllers.rs`
  Shared non-structural edit controllers used by the frontend.
- `frontend_models.rs`
  Presenter models for the Slint shell.
- `slint_bridge.rs`
  Slint event and viewport input decoding.
- `slint_frontend/`
  Native Slint runtime host, bindings, callbacks, and viewport texture management.
- `slint_ui/`
  Declarative `.slint` components and typed Slint view-model definitions.

### `src/graph/`

- `scene.rs`
  Core scene graph.
- `presented_object.rs`
  Presented-object layer used by the Slint shell and selection logic.
- `history.rs`
  Undo/redo history.
- `voxel.rs`
  CPU voxel grid data and sculpt math.

### `src/gpu/`

- `buffers.rs`
  Scene and voxel buffer encoding.
- `camera.rs`
  Camera model and uniforms.
- `codegen.rs`
  Runtime WGSL generation from the scene graph.
- `picking.rs`
  GPU pick pipeline and readback structures.

### `src/viewport/`

- `draw.rs`
  Shared viewport draw orchestration and selection baseline sync.
- `gpu_ops.rs`
  GPU render passes and compositing.
- `reference_overlay.rs`
  Reference image viewport overlay.
- supporting modules for textures and pipelines.

### `src/gizmo/`

- `viewport.rs`
  Viewport gizmo interaction, overlay generation, and transform math.
- `selection.rs`
  Multi-selection transform baseline and readout helpers.

## 5. Frame Flow

The current frame flow is split between toolkit-neutral backend logic and the Slint host.

1. Slint collects input and viewport events.
2. `src/app/slint_bridge.rs`
   Converts those events into `FrameInputSnapshot` and `ViewportInputSnapshot`.
3. `src/app/backend_frame.rs`
   Runs `run_backend_pre_ui(...)`.
4. `src/app/viewport_interaction.rs`
   Evaluates camera, pick, sculpt, and gizmo interaction feedback.
5. `src/app/frontend_models.rs`
   Builds a `ShellSnapshot`.
6. `src/app/slint_frontend/bindings/`
   Applies grouped panel state to the Slint window.
7. Slint renders the desktop shell and viewport image.
8. `src/app/slint_frontend/callbacks/`
   Converts UI intents back into `Action` values or shared controller calls.
9. `src/app/action_handler.rs`
   Applies structural mutations through `process_actions()`.
10. `src/app/backend_frame.rs`
   Runs `run_backend_post_ui(...)`.

## 6. UI Boundary Rules

- Toolkit-neutral frame logic stays in `src/app/backend_frame.rs`.
- Structural mutations stay behind `Action` plus `process_actions()`.
- Shared presenter models stay in `src/app/frontend_models.rs`.
- Slint input decoding stays in `src/app/slint_bridge.rs`.
- Slint runtime and callback wiring stay in `src/app/slint_frontend/`.
- Declarative UI stays in `src/app/slint_ui/`.

For the detailed frontend contract, see `docs/ui_backend_boundary.md` and `docs/slint_frontend.md`.

## 7. Slint Frontend Map

The frontend is split into:

- runtime host
  `src/app/slint_frontend/mod.rs`
- host state
  `src/app/slint_frontend/host_state/`
- bindings
  `src/app/slint_frontend/bindings/`
- callbacks
  `src/app/slint_frontend/callbacks/`
- declarative UI
  `src/app/slint_ui/*.slint`
- Slint view-model definitions
  `src/app/slint_ui/view_models/`

See `docs/slint_frontend.md` for the component-by-component breakdown.

## 8. Performance And Sculpt Guardrails

Sculpt responsiveness remains a hard quality bar.

Preserve:

- async non-blocking sculpt pick behavior
- predictive sculpt fallback while pick is pending
- off-mesh Grab continuation until release
- voxel-aware interpolation density
- per-sample delta clamping for Add/Carve/Flatten/Inflate
- Taubin smoothing for Smooth

Any change touching sculpt input, viewport interaction, or render scheduling needs manual verification.

Reference: `docs/sculpt_responsiveness_findings.md`

## 9. Validation

Every significant change must pass:

1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. manual visual/behavioral verification when applicable
