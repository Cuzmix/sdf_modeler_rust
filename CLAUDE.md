# SDF Modeler — Claude Code Instructions

## Coding Standards (MANDATORY)

### Architecture & File Organization
- **No monolithic files.** New functions, methods, and systems must be isolated in their own modules.
- **Group related functionality** into the same folder or subfolder for a discoverable, modular structure.
- **Maintain intentional architecture** that follows strong Rust design principles.

### Readability & Naming
- Code must be **readable and understandable even to non-developers**.
- Variable and function names **prioritize clarity over brevity**. Length is not a concern if it improves comprehension.
- **No single-letter variable names** except for well-established mathematical contexts (x, y, z, t, i, n, etc.).

### Programming Practices
- Follow **proper Rust idioms, patterns, and best practices**.
- Keep the codebase **junior-friendly, explicit, and easy to navigate**.
- Avoid over-engineering — only make changes that are directly requested or clearly necessary.

### Performance (Top Priority)
- **Performance must never be degraded.**
- Every change must **preserve or improve** runtime efficiency, memory usage, and architectural stability.
- Performance is the **#1 priority** under all circumstances.

### Dependencies
- Prefer hand-rolled implementations when well-suited to the project's specific needs.
- Only add dependencies when they provide substantial value beyond a focused implementation.
- Keep export format writers dependency-free (pure Rust with std::io only).

### Verification Loops (MANDATORY before every commit)
Every commit MUST pass these checks. Do NOT commit if any fail — fix the issue first.
1. `cargo check` — type checking (the most important feedback loop)
2. `cargo clippy -- -D warnings` — lint errors are build failures
3. `cargo test` — all tests must pass
4. `cargo build` — full compilation must succeed
5. Manual verification — if the change is visual/behavioral, run the app and confirm

### Search Before Implementing
Before writing new code, **search the codebase first** to verify it doesn't already exist.
Use ripgrep/grep to check for existing implementations. Duplicate code is a common failure
mode — always look before you write.

### No Placeholders or Stubs
Do NOT write placeholder implementations, TODO stubs, or minimal mock code.
Every function must be a **full, working implementation**. If a feature is too complex
for a single pass, break it into smaller complete pieces — never leave half-finished code.

### Git Discipline
- **One logical change per commit.** Don't bundle unrelated changes.
- Write descriptive commit messages that explain the "why", not just the "what".
- Keep changes small and focused to minimize context rot and merge conflicts.
- Commit the PRD and progress files alongside code changes when using Ralph workflow.

---

## Project Overview

**SDF Modeler** is a real-time Signed Distance Function (SDF) 3D modeling application. It renders SDF scenes on the GPU via raymarching, supports sculpting via voxel grids, and exports meshes via marching cubes.

### Tech Stack
- **Language**: Rust (edition 2021)
- **GPU**: wgpu 22.1.0 (via eframe 0.29) — WGSL shaders, sphere tracing
- **UI**: egui (via eframe 0.29) + egui_dock (dockable panels) + egui_node_graph2
- **Math**: glam 0.29
- **Serialization**: serde + serde_json, bytemuck for GPU data
- **Parallelism**: rayon (native only)
- **File dialogs**: rfd (native only)
- **Platform**: Windows 11 (primary), WASM support shelved

---

## Source Tree

```
src/
├── main.rs              # Entry point → lib::run_native()
├── lib.rs               # Module declarations, native + WASM entry points
├── compat.rs            # Cross-platform abstractions (Instant, Duration, etc.)
├── settings.rs          # Persistent user settings (vsync, camera, export, etc.)
├── io.rs                # File I/O (save/load project as JSON)
├── export.rs            # Mesh export: marching cubes + OBJ/STL/PLY/glTF/USDA writers
├── sculpt.rs            # Sculpt tool definitions (ActiveTool, SculptState, brush ops)
│
├── app/                 # Application core (update loop, state, actions)
│   ├── mod.rs           # SdfApp struct, FrameTimings, eframe::App impl
│   ├── state.rs         # State decomposition: DocumentState, GpuSyncState, UiState, etc.
│   ├── actions.rs       # Action enum (all structural state-mutating intents)
│   ├── action_handler.rs# process_actions() — single mutation point (Redux reducer)
│   ├── input.rs         # Keyboard shortcut handling → actions
│   ├── gpu_sync.rs      # Pipeline rebuild, shader compilation, dirty tracking
│   ├── async_tasks.rs   # Async task polling (export progress, bake results)
│   ├── sculpting.rs     # Per-frame sculpt brush application
│   └── ui_panels.rs     # Menu bar, legacy UI panel drawing
│
├── gpu/                 # GPU abstraction layer
│   ├── mod.rs           # Module declarations
│   ├── buffers.rs       # GPU buffer management (uniform, storage, staging)
│   ├── camera.rs        # CameraUniform, orbit/pan/zoom, projection matrices
│   ├── codegen.rs       # Runtime WGSL generation from scene tree (expression builder)
│   ├── picking.rs       # GPU pick pass (1x1 texture, node/gizmo identification)
│   └── shader_templates.rs # Static shader template strings
│
├── graph/               # Scene graph data model
│   ├── mod.rs           # Module declarations
│   ├── scene.rs         # Scene, SceneNode, NodeId, SdfPrimitive, CsgOp, transforms
│   ├── voxel.rs         # VoxelGrid (CPU-side SDF evaluation for sculpting)
│   └── history.rs       # Undo/redo history (clone-based scene snapshots)
│
├── ui/                  # UI components (egui panels and widgets)
│   ├── mod.rs           # Module declarations
│   ├── dock.rs          # Dockable panel layout (egui_dock), tab viewer
│   ├── viewport/        # 3D viewport rendering
│   │   ├── mod.rs       # ViewportResources, viewport orchestration
│   │   ├── pipelines.rs # Render pipeline creation and caching
│   │   ├── textures.rs  # Viewport texture management (render targets, voxel textures)
│   │   ├── gpu_ops.rs   # Per-frame GPU operations (dispatch, upload)
│   │   ├── draw.rs      # Viewport drawing (egui integration, input handling)
│   │   └── composite.rs # Composite pass (final output to screen)
│   ├── scene_tree.rs    # Scene hierarchy tree panel
│   ├── properties.rs    # Node property inspector panel
│   ├── node_graph.rs    # Visual node graph editor (egui_node_graph2)
│   ├── gizmo.rs         # Transform gizmo (translate/rotate/scale)
│   ├── export_dialog.rs # Export settings dialog
│   ├── render_settings.rs # Render quality settings panel
│   ├── settings_window.rs # Application settings window
│   ├── profiler.rs      # Performance profiler overlay
│   ├── toasts.rs        # Toast notification system
│   └── help.rs          # Help/about dialog
│
└── shaders/             # WGSL shader files
    ├── bindings.wgsl    # Bind group declarations
    ├── vertex.wgsl      # Fullscreen triangle vertex shader
    ├── primitives.wgsl  # SDF primitive functions (sphere, box, cylinder, etc.)
    ├── operations.wgsl  # CSG operations (union, subtract, intersect + smooth variants)
    ├── transforms.wgsl  # Transform utilities for SDF space
    ├── modifiers.wgsl   # SDF modifiers (round, elongate, etc.)
    ├── rendering.wgsl   # Raymarching, lighting, shadows, AO
    ├── voxel_sampling.wgsl # Voxel grid sampling (storage buffer + texture paths)
    ├── pick.wgsl        # GPU pick pass shader
    ├── brush.wgsl       # Sculpt brush compute shader
    ├── blit.wgsl        # Blit/copy shader
    └── composite_entry.wgsl # Composite pass entry point
```

---

## Key Architecture Patterns

> **Detailed reference**: Architecture, GPU pipeline, raymarching, and shader conventions are documented in the `sdf-architecture` Claude skill (auto-loaded when working on GPU/shader/codegen code) and in `docs/architecture.md`.

- **Action system**: Redux-inspired. `Action` enum → `process_actions()` reducer. Data-level edits use `&mut Scene` directly; structural changes go through actions.
- **Two-speed GPU sync**: `structure_key()` changes → shader rebuild (~10ms). `data_fingerprint()` changes → buffer upload only (~1ms).
- **State decomposition**: `SdfApp` has 7 sub-structs (doc, gizmo, gpu, async_state, ui, persistence, perf).
- **Scene graph**: Binary tree with `HashMap<NodeId, SceneNode>`. Codegen flattens to WGSL via post-order traversal.

---

## Sculpt Responsiveness Non-Regression (MANDATORY)

Sculpt smoothness is a hard quality bar. "60 FPS" is not sufficient if brush motion looks delayed or stepped.

Required guardrails for sculpt changes:
- Do not block active sculpt strokes on GPU pick readback.
- Keep async pick non-blocking; never add `Maintain::Wait` in active sculpt drag paths.
- Preserve predictive sculpt fallback while async pick is pending.
- Preserve off-mesh Grab continuation until mouse release.
- Preserve voxel-aware stroke interpolation density.
- Preserve per-sample delta clamping for Add/Carve/Flatten/Inflate.
- Preserve Taubin smoothing for Smooth brush (volume-friendly behavior).

Manual verification for any sculpt input/render change:
1. Hold LMB and sculpt fast circles on a low-resolution Sculpt node. Stroke must look continuous.
2. Use Grab and drag quickly, including leaving geometry. Motion must continue until release.
3. Toggle Shadows/AO while sculpting. Brush motion must stay smooth without visible stepping.
4. Confirm no obvious latency spikes at stroke start or during continuous drag.

Reference implementation files:
- `src/app/sculpting.rs`
- `src/app/mod.rs`
- `src/sculpt.rs`
- `src/shaders/brush.wgsl`
- `docs/sculpt_responsiveness_findings.md` (detailed findings and test matrix)

---

## Build & Run

```bash
# Native build (default)
cargo run

# Release build with debug symbols (for profiling)
cargo run --release

# Check compilation
cargo check
```

### Profile Configuration
- `[profile.dev]` opt-level = 2 — fast debug builds with ~90% release perf
- `[profile.release]` debug = true — debug symbols for flamegraph/samply profiling

---

## WASM (Shelved)
- Code compiles for `wasm32-unknown-unknown` on `feature_refactored_code` branch
- Blocked by wgpu sending deprecated `maxInterStageShaderComponents` to Chrome 135+
- Will work once wgpu v29+ releases

## Flutter Migration (Shelved)
- Branch: `Slint_migration`
- Feature gates: `egui_ui` (default) | `flutter_ui`
- flutter_rust_bridge v2.11.1

---

## Ralph Workflow (Autonomous Agent Loop)

This project supports the Ralph technique for long-running autonomous coding sessions.

### Files
- `plans/prd.json` — Product requirements document. JSON array of tasks with `passes` flags.
- `plans/progress.txt` — Append-only log of learnings, decisions, and completed work per sprint.
- `plans/ralph.sh` — AFK loop: runs Claude Code headlessly for N iterations.
- `plans/ralph-once.sh` — HITL mode: single interactive iteration for steering.

### How It Works
1. The PRD defines **what** needs to be done (not how). Each item has a `passes` boolean.
2. Each loop iteration: pick highest-priority incomplete task, implement it, run verification
   loops, update PRD (`passes: true`), append to progress.txt, git commit.
3. Loop exits when all PRD items pass or max iterations reached.

### Rules for Ralph Mode
- Work on **exactly one task** per iteration — never batch multiple features.
- Keep tasks small. If a PRD item is too large, split it before starting.
- Always run verification loops (cargo check/clippy/test) before committing.
- Append to progress.txt (don't overwrite) — this is memory for the next iteration.
- Commit PRD + progress.txt + code together so git history reflects the sprint.
- If you discover a bug while working, add it to the PRD — don't fix it in the same iteration.
