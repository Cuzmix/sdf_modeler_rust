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
4. Manual verification — if the change is visual/behavioral, run the app and confirm

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

### Action System (React/Redux-inspired)
- `Action` enum in `src/app/actions.rs` — all structural state-mutating intents
- `ActionSink = Vec<Action>` — UI pushes actions, never mutates state directly
- `process_actions()` in `src/app/action_handler.rs` — single mutation point (Redux reducer)
- **Data-level edits** (sliders, colors, gizmo transforms) use `&mut Scene` for zero-latency
- **Structural changes** (delete, create, insert modifier, bake, tool switch) go through actions

### Update Loop Phases (per frame)
1. Frame setup (timing, input)
2. Async polling (export progress, bake results)
3. GPU pipeline sync (shader recompilation if topology changed)
4. Collect actions from keyboard shortcuts + UI drawing
5. `process_actions()` — apply all queued actions
6. Post-action cleanup
7. GPU upload + dirty tracking
8. Finalize (history snapshot, request repaint)

### State Decomposition
`SdfApp` is decomposed into sub-structs in `src/app/state.rs`:
- `DocumentState` — scene, camera, history, active_tool, sculpt_state, clipboard_node
- `GizmoContext` — gizmo mode, space, state
- `GpuSyncState` — buffer dirty flags, pipeline state
- `AsyncState` — background task handles
- `UiState` — node_graph_state, dock state, UI toggles
- `PersistenceState` — save/load state
- `PerfState` — frame timings, profiler data

Access pattern: `self.doc.scene`, `self.gpu.buffer_dirty`, `self.ui.node_graph_state`

### Scene Graph
- Binary tree with `HashMap<NodeId, SceneNode>` in `src/graph/scene.rs`
- Node types: primitives, CSG operations, transforms, modifiers
- Codegen flattens tree to WGSL via post-order traversal (`src/gpu/codegen.rs`)
- Fast/slow path: `structure_key()` hashes topology — slider drags only update storage buffer (fast), topology changes regenerate shader (slow)

### GPU Pipeline
- Bind groups: @group(0)=camera, @group(1)=scene storage, @group(2)=voxel textures
- Render shader: `texture_3d<f32>` + `textureSampleLevel` for hardware trilinear
- Pick shader: 1x1 Rgba8Unorm, encoding 0=bg, 1=floor, 2+=node, 253-255=gizmo axes
- Brush shader: compute shader for sculpt voxel modification
- Custom device limits: `max_storage_buffers_per_shader_stage = 4`, 128MB storage buffers

### Mesh Export
- Hand-rolled marching cubes in `src/export.rs`
- rayon-parallelized by z-slice, two-phase vertex deduplication
- Formats: OBJ, STL (binary), PLY (ASCII), glTF Binary (.glb), USD ASCII (.usda)
- Async: `start_export` spawns thread, `poll_export` checks mpsc channel

---

## Raymarching & SDF Notes (iq-inspired)

- Conservative step: `t += d * 0.9` + epsilon 0.0005 + 96 max steps
- Enhanced sphere tracing: Keinert over-relaxation (omega=1.2), ~20% fewer steps
- Tetrahedron normals (4 evals vs 6)
- Distance-adaptive normal epsilon: `clamp(0.001*t, 0.0005, 0.05)`
- Improved soft shadows (Aaltonen variant with `ph` tracking)
- Scene-level ray-AABB intersection for early-out
- Interactive quality mode: half steps, skip AO+shadows during camera drag
- Per-subtree bounding skip: expensive subtrees (with sculpt) wrapped in AABB conditional
- Binary search refinement is BAD (50% slower per iq)
- GPU ternaries don't create real branches — don't "optimize" with step/mix
- sqrt() is nearly free on GPU

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
