# AGENTS.md

Operational guide for coding agents working in this repository.

## Source Of Truth
Read these before large or architectural changes:
- `CLAUDE.md` (full project policy and workflow)
- `docs/architecture.md` (engine architecture)
- `docs/ui_backend_boundary.md` (UI/backend separation boundary)
- `docs/sculpt_responsiveness_findings.md` (sculpt responsiveness guardrails)

If this file conflicts with `CLAUDE.md`, follow `CLAUDE.md`.

## Non-Negotiable Coding Standards
- Keep architecture modular and discoverable; avoid monolithic files.
- Use explicit, readable names; avoid single-letter names except standard math/index cases.
- Follow Rust idioms and keep code junior-friendly.
- Do not over-engineer; implement only what is required.
- Do not leave placeholders, TODO stubs, or partial implementations.
- Before adding code, search the codebase first to avoid duplicate implementations.

## Pushback And Decision Quality
- Push back on technically unsound requests or changes likely to hurt maintainability/performance.
- Prefer the smallest correct fix that addresses the root cause.
- Avoid exposing engine-internal/debug complexity in user-facing UX unless clearly required.

## Performance Rules (Top Priority)
- Do not regress runtime performance, memory usage, or responsiveness.
- Prefer focused hand-rolled implementations over unnecessary dependencies.
- Keep export writers dependency-free (`std::io`-only where applicable).

## UI/Backend Boundary (Mandatory)
- Keep toolkit-agnostic frame logic in `src/app/backend_frame.rs`.
- Keep egui drawing in `src/app/egui_frontend.rs`.
- Keep egui input/command mapping in `src/app/frontend_bridge.rs`.
- Keep `src/app/mod.rs` orchestration-only.
- Keep `process_actions()` as the structural mutation gate; UI should emit `Action` values.

## Sculpt Responsiveness Non-Regression (Mandatory)
For sculpt-related changes, preserve:
- Non-blocking async pick behavior in active sculpt paths.
- Predictive sculpt fallback while async pick is pending.
- Off-mesh Grab continuation until mouse release.
- Voxel-aware interpolation density.
- Per-sample delta clamping behavior for Add/Carve/Flatten/Inflate.
- Taubin smoothing behavior for Smooth brush.

Manual checks required for sculpt input/render changes:
1. Fast circular strokes remain continuous.
2. Grab drag continues even when leaving geometry, until release.
3. Toggling Shadows/AO during sculpt does not introduce stepping.
4. No latency spikes at stroke start or during continuous drag.

## Mandatory Validation Before Commit
Every commit must pass, in this order:
1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. Manual verification for visual/behavioral changes

Do not commit if any step fails.

## Known Regression Notes
### Transform Gizmo Rotation Snap-Back (Multi-Select)
- Symptom: during multi-object rotate drag, objects appear to rotate then snap back toward the start pose on subsequent frames.
- Root cause pattern: mixing per-frame incremental drag values with a baseline re-application path.
- In this codebase/egui setup, `response.drag_delta()` is per-frame; baseline re-apply flows must consume total applied rotation, not incremental-only deltas.
- Guardrail:
  - Incremental apply flows should use `ScreenRotationDragState::consume_applied_delta`.
  - Baseline re-apply flows should use `ScreenRotationDragState::consume_applied_total`.
- First places to verify when this regresses:
  - `src/ui/gizmo.rs` (`handle_multi_rotate_drag`, `apply_world_rotation_to_targets`, `ScreenRotationDragState`)
  - `src/ui/properties.rs` and `src/ui/viewport/draw.rs` (multi-transform baseline/session sync paths)
- Regression tests to keep passing:
  - `ui::gizmo::tests::snapped_rotation_consumes_incremental_applied_delta`
  - `ui::gizmo::tests::snapped_rotation_consumes_total_applied_angle_for_baseline_flows`

## Git Discipline
- One logical change per commit.
- Use descriptive commit messages that explain why.
- Keep commits focused to reduce merge conflicts.

## Ralph Workflow Rules
When using PRD/progress loop mode:
- Work one PRD task per iteration.
- Run validation loop before committing.
- Append to `plans/progress.txt` (do not overwrite).
- Commit code + `plans/prd.json` + `plans/progress.txt` together.
- If a new bug is discovered, log it as a PRD item instead of batching unrelated fixes.
