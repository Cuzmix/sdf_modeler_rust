# Slint Migration Status and Full Cutover Plan

Last updated: 2026-03-08

## Last Completed Work (What just landed)

### Embedded viewport stabilization (safe path)
- Removed unstable Slint `wgpu_28` coupling from runtime selection.
- Replaced the experimental texture-import path with a stable in-process software image bridge using `SharedPixelBuffer`.
- Wired periodic embedded viewport refresh in the Slint shell center panel (`33ms` cadence).
- Kept external viewport host controls available (`Open Viewport 3D` / `Close Viewport 3D`) so migration can continue without blocking users.

### Why this path
- The unstable Slint WGPU feature path introduced dependency/version coupling risk on Windows.
- The software bridge keeps migration momentum while preserving a single-process embedded viewport area in the shell UI.
- This is a migration bridge, not the final renderer architecture.

## What is Next (Immediate plan)

1. **Slice 5A: Host renderer bridge API**
   - Define a backend-neutral render target contract between app core and Slint shell.
   - Keep renderer ownership in Rust core, not in Slint UI modules.

2. **Slice 5B: Real viewport embedding path**
   - Replace software image bridge with the real renderer frame path inside the shell center region.
   - Preserve current command/callback behavior and status telemetry.

3. **Slice 5C: Input routing**
   - Route orbit/pan/zoom/pick/gizmo events from Slint viewport item to core commands.
   - Maintain parity with current egui interaction semantics.

4. **Slice 5D: Manual parity pass**
   - Validate camera controls, selection, gizmo drag, and sculpt entry/exit in Slint shell mode.

## Full Migration List (End-to-end)

## Phase 0: Foundations
- [x] Add framework-neutral core command/read-model layer (`AppCore`, `CoreCommand`, snapshots).
- [x] Keep egui runtime behavior unchanged while migrating command handlers.

## Phase 1: Slint shell scaffold
- [x] Boot Slint shell window and route basic toolbar callbacks.
- [x] Add scene read-model panels (Scene Tree, History, Lights, Properties, Render Settings, Scene Stats).
- [x] Make Slint shell default for `--frontend slint`.

## Phase 2: Slint action parity (non-viewport)
- [x] Wire create/select/delete/undo/redo/toggle/focus/rename flows.
- [x] Add light creation and light solo controls from shell.

## Phase 3: Viewport transition bridge
- [x] Add external viewport host lifecycle controls from shell.
- [x] Add embedded viewport region in shell center panel.
- [x] Stabilize with software image bridge while WGPU major alignment is deferred.

## Phase 4: True embedded viewport (in progress)
- [ ] Introduce renderer bridge API for shell-owned viewport surface.
- [ ] Pipe real renderer output into shell center region.
- [ ] Remove software bridge once parity is confirmed.

## Phase 5: Interaction parity for embedded viewport
- [ ] Orbit / pan / zoom parity.
- [ ] GPU pick parity (node + gizmo handles).
- [ ] Transform gizmo drag parity.
- [ ] Sculpt interaction parity and brush preview parity.

## Phase 6: Panel and workflow parity
- [ ] Node graph parity (SDF graph + Light graph panels with clear isolation).
- [ ] Reference image workflow parity (dedicated panel + no distortion regressions).
- [ ] Import/export dialogs parity.
- [ ] Settings/help/profiler parity.

## Phase 7: Legacy egui retirement
- [ ] Keep `legacy-egui-ui` only as temporary fallback.
- [ ] Remove dead egui docking/panel paths when parity gates pass.
- [ ] Remove third-party egui node graph dependency after Slint-native replacement is complete.

## Phase 8: Release hardening
- [ ] Performance baseline vs legacy (frame time, input latency, startup time).
- [ ] Cross-platform smoke (Windows/macOS/Linux build + run).
- [ ] Regression pass on project load/save, sculpt, lights, export.

## Validation Gate (required per migration commit)
1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`
5. Manual shell smoke for changed UX paths
