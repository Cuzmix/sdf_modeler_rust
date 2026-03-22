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
