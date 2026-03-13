# UI/Backend Boundary Contract

This document defines the required separation between application core logic and UI toolkit code.

## Goals

- Keep core frame logic independent from `egui`, so migration to Slint/Iced is incremental.
- Preserve behavior and performance while swapping UI adapters.
- Keep ownership and mutation flow explicit.

## Current Boundary

- `src/app/backend_frame.rs`
  - Owns toolkit-agnostic frame lifecycle (`run_backend_pre_ui`, `run_backend_post_ui`).
  - Defines boundary DTOs: `FrameInputSnapshot`, `UiFrameFeedback`, `FrameCommands`.
  - Must not import `eframe::egui` or toolkit-specific UI types.

- `src/app/egui_frontend.rs`
  - Owns egui-only drawing and egui-specific feedback capture.

- `src/app/frontend_bridge.rs`
  - Converts `egui::Context` to `FrameInputSnapshot`.
  - Applies `FrameCommands` back into egui viewport/repaint commands.

- `src/app/mod.rs`
  - Must remain orchestration-only.
  - Frame flow: capture input -> backend pre-ui -> frontend draw -> process actions -> backend post-ui -> apply commands.

## Mutation Rules

- Structural scene changes must go through `Action` + `process_actions()`.
- Frontend adapters should emit `Action` values via `ActionSink` for structural edits.
- Data-level live edits may stay direct where required for latency, but do not introduce toolkit types into backend modules.
- Viewport camera and selection commands that must survive toolkit swaps belong in toolkit-neutral facades such as `src/app_bridge/`; do not leave preset views, projection toggles, or framing behavior owned only by egui panels or egui action glue.
- Native viewport hosts may own frame cadence and texture transport, but Rust backend modules own scene state, camera behavior, picking, and command semantics.

## Adding Another UI Toolkit

- Add toolkit-specific adapter modules parallel to egui (for example `slint_frontend.rs`, `slint_bridge.rs`).
- Reuse `backend_frame.rs` unchanged where possible.
- Keep boundary DTOs generic and toolkit-neutral.
- Update `CLAUDE.md` and `docs/architecture.md` if the contract changes.

## Review Checklist

1. `backend_frame.rs` has no toolkit imports.
2. Toolkit calls stay in frontend adapter/bridge modules.
3. `mod.rs` remains orchestration-only.
4. `process_actions()` remains the structural mutation gate.
5. Validation passes in order: `cargo check`, `cargo clippy -- -D warnings`, `cargo test`, `cargo build`.
6. Manual visual verification is done for visual/behavioral changes.