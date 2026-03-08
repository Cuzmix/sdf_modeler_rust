# Slint Migration Plan (Cross-Platform Program)

## Goal
Migrate UI from egui/eframe to Slint while preserving the current Rust scene, GPU, and action architecture. End state should ship on Windows/macOS/Linux from one codebase, with professional-grade node workflow and stable rendering.

## Scope
- In scope: UI framework migration, app shell, panel system, node graph UI, viewport integration, input routing, dialogs/settings.
- Out of scope (phase 1): mobile, web target, full plugin API rewrite.

## Current Reality (Main Branch)
- Core model + rendering stack are already modular enough to preserve:
  - Scene/graph/history: `src/graph/*`
  - GPU/codegen/shaders: `src/gpu/*`, `src/shaders/*`
  - Actions/reducer pattern: `src/app/actions.rs`, `src/app/action_handler.rs`
- Main migration burden is UI coupling in:
  - `src/app/mod.rs`
  - `src/ui/*`
  - egui-dock + egui-node-graph2 dependence

---

## Recommended Strategy
Use a **strangler migration**:
1. Extract UI-agnostic application core first.
2. Build Slint shell around the extracted core.
3. Port panels incrementally behind adapter interfaces.
4. Replace node graph UI with a custom Slint graph widget (not a temporary clone of egui behavior).

This avoids a high-risk "big bang" rewrite.

---

## Phases

## Phase 0: Technical Spike (1 week)
### Deliverables
- Validate Slint + wgpu integration approach for viewport rendering.
- Decide final window/render ownership model.
- Document performance baseline and expected regression budget.

### Decision Gate
Pick one architecture:
- A) Slint owns app window, viewport renderer binds to Slint window handle.
- B) Winit/wgpu host owns loop and embeds Slint UI layer.

### Exit Criteria
- Minimal prototype renders a colored triangle in viewport area + receives mouse input.
- No frame hitching at idle.

---

## Phase 1: Core/UI Separation (2 weeks)
### Objective
Remove egui types from business logic path.

### Tasks
- Introduce `AppCore` (UI-agnostic) containing:
  - scene, history, camera state, tool state, async task state, action reducer
- Define UI-facing interfaces:
  - `UiCommand` (from UI to core)
  - `UiSnapshot` (from core to UI)
  - `ViewportInput` / `ViewportOutput`
- Move all non-visual mutations to reducer boundaries.
- Keep current egui UI running via adapter while separating core.

### Exit Criteria
- `src/app` can compile/run with minimal egui knowledge.
- Core has deterministic tests independent of UI framework.

---

## Phase 2: Slint App Shell + Panel Host (2 weeks)
### Objective
Replace eframe shell first, while keeping functionality parity for non-node-graph panels.

### Tasks
- Create Slint shell (menu bar, status bar, panel tabs, toast host).
- Port these panels first (lowest risk):
  - scene tree
  - history
  - render settings
  - lights panel
  - scene stats
  - properties (read-only first, then editable)
- Wire `UiCommand` dispatch and `UiSnapshot` pull.

### Exit Criteria
- App opens in Slint shell and updates scene state through reducer.
- Existing save/load, undo/redo, and tool switching work.

---

## Phase 3: Viewport Integration (2 weeks)
### Objective
Preserve render parity under Slint host.

### Tasks
- Integrate current wgpu viewport backend into Slint surface region.
- Route camera input (orbit/pan/zoom), picking, gizmo hits.
- Keep render settings and dynamic pipeline rebuild behavior intact.
- Add frame pacing policy (interaction/full quality).

### Exit Criteria
- Visual parity with current main branch on representative scenes.
- No major perf regression (>10% frame-time increase at same settings).

---

## Phase 4: Node Graph 2.0 in Slint (4 weeks)
### Objective
Build professional, scalable graph UX (Houdini/Blender direction).

### Tasks
- Implement custom Slint node graph widget with explicit graph instance IDs.
- Keep graph model in Rust (Scene graph remains source of truth).
- Features to ship in this phase:
  - robust spawn placement (cursor/selection offset + collision avoidance + grid snap)
  - organize/layout (layered DAG flow, crossing reduction heuristics)
  - minimap with correct viewport mapping + drag nav
  - ALT+LMB bypass disconnect
  - auto-insert rewiring with visual wire-hover hint
  - reroute nodes, comments/groups scaffolding
- Ensure strict separation between SDF graph and Light graph contexts.

### Exit Criteria
- No widget-ID collisions.
- SDF graph actions do not leak into Light graph.
- Graph usability passes internal UX checklist on large scenes.

---

## Phase 5: Dialogs, Polish, and Cross-Platform Hardening (2 weeks)
### Tasks
- Port remaining dialogs (import/export/settings/recovery).
- Keyboard shortcut parity and focus/IME fixes.
- HiDPI scaling behavior verification.
- Platform-specific QA:
  - Windows (DX12/Vulkan)
  - macOS (Metal)
  - Linux (Vulkan)

### Exit Criteria
- Full feature parity with current egui app for desktop scope.
- CI green on all supported desktop targets.

---

## Phase 6: Cutover + Cleanup (1 week)
### Tasks
- Remove old egui UI stack from default build.
- Keep temporary compatibility feature flag for one release cycle:
  - `--features legacy-egui-ui`
- Update docs, architecture notes, contributor guide.

### Exit Criteria
- Slint UI is default and stable.
- Legacy flag optional, not required for release.

---

## Architecture Rules for Migration
1. Scene graph remains the single source of truth.
2. UI does not mutate scene directly; all structural edits go through actions/reducer.
3. Node graph behavior is graph-instance scoped (SDF vs Light vs future graphs).
4. No UI toolkit types in core domain modules.
5. All async operations report progress/events through core state, not direct UI handles.

---

## Risk Register
- Docking parity risk: Slint does not provide egui-dock equivalent out-of-the-box.
  - Mitigation: implement constrained tab/split host first; avoid over-engineered docking in v1.
- Node graph complexity risk.
  - Mitigation: deliver graph as dedicated workstream with tests from day 1.
- Input routing conflicts (viewport vs graph vs shortcuts).
  - Mitigation: explicit focus manager in core.
- GPU surface lifecycle differences by platform.
  - Mitigation: phase-0 prototype + platform CI smoke tests early.

---

## Validation Loop (every migration PR)
- `cargo check`
- `cargo clippy -- -D warnings`
- `cargo test`
- `cargo build`
- Manual smoke:
  - create/edit/delete nodes
  - node graph connect/disconnect/organize
  - viewport orbit/pick/gizmo
  - save/load/undo/redo

---

## Suggested Execution Order (Ralph backlog)
1. Phase 0 spike + architecture decision record
2. Phase 1 core extraction
3. Phase 2 shell + low-risk panels
4. Phase 3 viewport
5. Phase 4 node graph
6. Phase 5 polish/cross-platform QA
7. Phase 6 cutover

## Target Timeline
~14 weeks total (parallelized):
- Core/Shell/Viewport track: ~6 weeks
- Node graph track: ~4 weeks
- QA/cutover: ~4 weeks

