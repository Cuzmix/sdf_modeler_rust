# UI/Backend Boundary Contract

This document defines the current boundary between shared app logic and the Slint frontend.

## Goals

- Keep frame logic independent from any one widget toolkit.
- Keep structural mutations explicit and reviewable.
- Keep the Slint layer declarative and free of engine logic.
- Preserve performance while the UI continues to evolve.

## Current Native Boundary

### Shared App/Core Layer

- `src/app/backend_frame.rs`
  Owns the toolkit-neutral frame lifecycle:
  - `run_backend_pre_ui(...)`
  - `run_backend_post_ui(...)`
  - `FrameInputSnapshot`
  - `UiFrameFeedback`
  - `FrameCommands`

- `src/app/actions.rs`
  Defines structural `Action` values.

- `src/app/action_handler.rs`
  Owns `process_actions()`, the structural mutation gate.

- `src/app/controllers.rs`
  Owns non-structural edit controllers used by the frontend.

- `src/app/frontend_models.rs`
  Builds `ShellSnapshot` and related presenter models from shared app state.

- `src/app/viewport_interaction.rs`
  Owns toolkit-neutral viewport interaction logic.

These modules must not depend on Slint widget types.

### Slint Bridge Layer

- `src/app/slint_bridge.rs`
  Converts Slint events into:
  - `FrameInputSnapshot`
  - `ViewportInputSnapshot`
  - coarse UI events such as undo/redo/frame-all

This file is the input boundary between Slint and shared frame logic.

### Slint Runtime Host Layer

- `src/app/slint_frontend/mod.rs`
  Bootstraps the native Slint host and shared WGPU backend.

- `src/app/slint_frontend/host_state/`
  Owns tick flow, viewport texture lifetime, and host runtime state.

- `src/app/slint_frontend/bindings/`
  Applies presenter models to grouped Slint state structs.

- `src/app/slint_frontend/callbacks/`
  Converts Slint domain callbacks into `Action` values or shared controller calls.

These modules may depend on Slint runtime APIs, but should not become a second reducer.

### Declarative Slint UI Layer

- `src/app/slint_ui/*.slint`
  Owns layout, visual composition, and domain-level callback emission.

- `src/app/slint_ui/view_models/`
  Owns the Slint-side state structs and action enums used by the UI.

The `.slint` files should stay declarative. They should not embed business rules or mutate shared state directly.

## Mutation Rules

- Structural scene changes must go through `Action` plus `process_actions()`.
- Frontend adapters should emit `Action` values for structural edits.
- High-frequency value edits may use shared controllers when latency matters, but they must stay toolkit-neutral.
- Do not move Slint types into `backend_frame.rs`, reducers, scene graph modules, or GPU modules.

## Frontend Design Rules

- Use grouped Slint panel state instead of large flat property bags.
- Use domain-level callbacks instead of one callback per widget.
- Split bindings, callbacks, and host-state code by responsibility; avoid monolithic frontend files.
- Keep viewport event transport separate from low-frequency UI commands.

## Review Checklist

1. `src/app/backend_frame.rs` remains toolkit-neutral.
2. `process_actions()` remains the structural mutation gate.
3. `src/app/slint_bridge.rs` is only input decoding, not business logic.
4. `src/app/slint_frontend/` is runtime/binding/callback wiring, not scene mutation policy.
5. `.slint` files stay declarative and emit domain intents only.
6. Validation passes in order:
   - `cargo check`
   - `cargo clippy -- -D warnings`
   - `cargo test`
   - `cargo build`
7. Manual visual verification is completed for visual or behavioral changes.

## Related Docs

- `docs/architecture.md`
- `docs/slint_frontend.md`
- `docs/sculpt_responsiveness_findings.md`
