# SDF Modeler — Architecture Documentation

> Comprehensive technical reference for the SDF Modeler Rust codebase.
> Intended for competent Rust developers who are new to this project, egui, and/or wgpu.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Module-by-Module Documentation](#2-module-by-module-documentation)
3. [Code Style, Patterns, and Conventions](#3-code-style-patterns-and-conventions)
4. [Design Decisions and Rationale](#4-design-decisions-and-rationale)
5. [Workarounds and Technical Debt](#5-workarounds-and-technical-debt)
6. [Maintenance and Onboarding Guide](#6-maintenance-and-onboarding-guide)

---

## 1. High-Level Overview

### 1.1 Application Purpose

SDF Modeler is a real-time Signed Distance Function (SDF) 3D modeling application. It renders SDF scenes on the GPU via sphere-tracing (raymarching), supports voxel-based sculpting, and exports meshes via marching cubes.

**Core capabilities:**
- Build 3D scenes from parametric SDF primitives (sphere, box, cylinder, torus, etc.)
- Combine shapes with CSG operations (union, subtract, intersect — with smooth, chamfer, stairs, and columns variants)
- Apply domain modifiers (twist, bend, taper, repeat, mirror, noise, etc.)
- Sculpt voxel-based geometry with 6 brush modes and 5 brush shapes
- Real-time PBR rendering with shadows, AO, SSS, fog, and bloom
- Export meshes to OBJ, STL, PLY, glTF (.glb), and USD (.usda)
- Import external meshes and voxelize them

**Target users:** 3D artists, technical artists, and developers interested in SDF-based modeling workflows.

### 1.2 Architecture Summary

The application is organized into four layers:

```
┌──────────────────────────────────────────────────────┐
│  UI Layer (src/ui/)                                  │
│  egui panels, viewport, gizmos, dialogs              │
├──────────────────────────────────────────────────────┤
│  App Core (src/app/)                                 │
│  Frame loop, action system, state management         │
├──────────────────────────────────────────────────────┤
│  Scene Graph (src/graph/)                            │
│  Nodes, SDF primitives, voxel grids, undo/redo       │
├──────────────────────────────────────────────────────┤
│  GPU Pipeline (src/gpu/ + src/shaders/)              │
│  WGSL codegen, buffers, raymarching, picking         │
└──────────────────────────────────────────────────────┘
```

**Key crates:**
| Crate | Version | Purpose |
|-------|---------|---------|
| eframe | 0.29 | Window management, bundles wgpu 22.1.0 + egui |
| egui_dock | 0.14 | Dockable panel layout |
| egui_node_graph2 | 0.7 | Visual node graph editor |
| glam | 0.29 | Math (Vec3, Mat4, Quat) with serde |
| bytemuck | 1 | Safe GPU struct casting (Pod/Zeroable) |
| serde + serde_json | 1 | Project serialization |
| rayon | 1.11 | CPU parallelism (native only) |
| rfd | 0.15 | Native file dialogs |
| image | 0.25 | Screenshot PNG encoding |
| naga | 22.1 | Shader validation (dev-dependency, tests only) |

### 1.3 Data and Render Flow

```
User Input (mouse/keyboard)
    │
    ▼
Keyboard Shortcuts ──► Action Enum (ActionSink = Vec<Action>)
UI Panel Interactions ─┘
    │
    ▼
process_actions()  ◄── Single mutation point (Redux reducer)
    │
    ▼
Scene Graph (HashMap<NodeId, SceneNode>)
    │
    ├─► structure_key() hash ──► Shader Rebuild (slow path, ~10ms)
    │                              └─► WGSL codegen ──► Pipeline creation
    │
    └─► data_fingerprint() hash ──► Buffer Upload (fast path, ~1ms)
                                      └─► SdfNodeGpu array ──► GPU storage buffer
    │
    ▼
GPU Raymarching (sphere tracing per pixel)
    │
    ▼
egui_wgpu Paint Callback ──► Offscreen texture ──► Blit to screen
```

**Two-speed GPU synchronization:**
- **Slow path**: When topology changes (add/delete node, reconnect graph), `structure_key()` changes → full shader recompilation + pipeline rebuild
- **Fast path**: When only data changes (slider drag, color pick, gizmo transform), only `data_fingerprint()` changes → buffer upload only (reuses existing shader)

### 1.4 Frame Update Loop

The `eframe::App::update()` method in `src/app/mod.rs` runs every frame in 8 phases:

1. **Frame setup** — Calculate delta time, update frame timings (EMA), tick camera animations, update turntable rotation, call `history.begin_frame()` to capture "before" snapshot

2. **Async polling** — Check mpsc receivers for completed background tasks: voxel bake, mesh export, mesh import, sculpt GPU pick. Detect sculpt drag end (LMB release).

3. **GPU pipeline sync** — Hash scene topology via `structure_key()`. If changed: regenerate WGSL shaders, rebuild render/pick/composite pipelines. Mark buffer dirty.

4. **Collect actions from UI** — `collect_keyboard_actions()` from keymap bindings. Draw menu bar, status bar, overlays, dialogs, dock panels (which draws viewport, scene tree, properties, etc.). Each UI component pushes `Action` variants to the `ActionSink`.

5. **Process all actions** — `process_actions()` iterates the collected `Vec<Action>` and applies mutations to scene, history, selection, tools, settings. This is the **single mutation point** for structural changes.

6. **Post-action cleanup** — Validate selection (clear if deleted). Copy viewport output (pending_pick, modifier keys) to async state. Submit async sculpt pick. Reset stroke interpolation if idle.

7. **GPU upload + dirty tracking** — Fingerprint scene data. If changed: build voxel buffer + node buffer, upload to GPU. Track unsaved changes. Auto-save if enabled.

8. **Finalize** — Call `history.end_frame()` to commit undo snapshot if scene changed. Determine if repaint is needed (dragging, animating, sculpting, async tasks). Request repaint only when necessary (avoids idle GPU spin).

---

### 1.5 Sculpt Responsiveness Pipeline (Critical)

Sculpt interaction quality depends on latency, not only frame rate. The app now uses a hybrid path to keep strokes smooth:

- Async GPU pick for accurate surface hits (`poll_sculpt_pick` and `submit_sculpt_pick`).
- Predictive drag fallback while pick readback is still pending (projects the current cursor ray to the drag depth plane and applies a temporary hit).
- Off-mesh Grab continuation so Grab keeps moving until mouse release.

Recent brush behavior upgrades:
- Grab uses Kelvinlet backward warp from a stroke snapshot for stable large moves.
- Smooth uses Taubin-style smoothing (lambda/mu passes) for better volume preservation.
- Add/Carve/Flatten/Inflate clamp per-sample SDF delta to avoid stepping artifacts.
- Stroke interpolation spacing adapts to voxel size to reduce rubber-banding during fast drags.
- GPU brush compute (`brush.wgsl`) mirrors CPU delta clamp logic to keep behavior consistent.

Non-regression requirement:
- Any change touching sculpt input, picking, brush math, or per-frame update order must be manually checked for visual smoothness under active drag, including shadows/AO enabled.
- Detailed notes and regression checklist: docs/sculpt_responsiveness_findings.md.

---

## 2. Module-by-Module Documentation

### 2.1 Entry Points

**`src/main.rs`** — Single line: `sdf_modeler::run_native()`.

**`src/lib.rs`** — Module declarations + platform entry points:
- **Native** (`run_native()`): Initializes env_logger, loads Settings, configures eframe with custom wgpu device descriptor (FLOAT32_FILTERABLE feature, 4 storage buffers, 128MB storage limit, 8192 max texture). Spawns `SdfApp::new()`.
- **WASM** (`WebHandle`): Async start on canvas element. Reduced limits (32MB storage, 4096 max texture, no storage textures).

### 2.2 App Core (`src/app/`)

#### `src/app/mod.rs` — Application Core

**`SdfApp`** — Main application struct, decomposed into 7 sub-structs for clear ownership:

```rust
pub struct SdfApp {
    pub doc: DocumentState,        // Scene, camera, history, tools
    pub gizmo: GizmoContext,       // Gizmo mode, space, interaction state
    pub gpu: GpuSyncState,         // Pipeline state, dirty flags, buffer offsets
    pub async_state: AsyncState,   // Background task handles and sculpt state
    pub ui: UiState,               // Dock layout, dialogs, search, selection
    pub persistence: PersistenceState, // File path, dirty flag, auto-save timer
    pub perf: PerfState,           // Frame timings, profiler data
    pub settings: Settings,        // Render config, keybinds, bookmarks
}
```

Implements `eframe::App` with the 8-phase update loop described in Section 1.4.

**`FrameTimings`** — Per-phase CPU timings with EMA smoothing and 120-frame ring buffer for sparkline display.

#### `src/app/state.rs` — State Decomposition

**`DocumentState`** — Core model state:
- `scene: Scene` — The SDF scene graph
- `camera: Camera` — Viewport camera (orbit/pan/zoom)
- `history: History` — Undo/redo stacks (50 snapshots each)
- `active_tool: ActiveTool` — Select or Sculpt
- `sculpt_state: SculptState` — Active brush parameters
- `sculpt_history: SculptHistory` — Per-stroke sculpt undo
- `clipboard_node: Option<NodeId>` — Copied node for paste

**`GpuSyncState`** — GPU resource tracking:
- `render_state: RenderState` — Arc'd wgpu Device/Queue from eframe
- `current_structure_key: u64` — Hash of scene topology (triggers shader rebuild when changed)
- `buffer_dirty: bool` — Needs scene buffer upload
- `last_data_fingerprint: u64` — Hash of node property data
- `voxel_gpu_offsets: HashMap<NodeId, u32>` — Per-sculpt offset in storage buffer
- `sculpt_tex_indices: HashMap<NodeId, usize>` — Per-sculpt texture3D index

**`AsyncState`** — Background tasks and sculpt interaction:
- `bake_status`, `export_status`, `import_status` — Thread join handles + progress
- `pick_state` — GPU async pick buffer receiver
- `last_sculpt_hit`, `lazy_brush_pos` — Brush stroke interpolation
- `sculpt_ctrl_held`, `sculpt_shift_held` — Modifier keys during drag
- `sculpt_pressure` — Pen pressure (0–1)
- `hover_world_pos` — 3D brush preview position

**`UiState`** — UI layout and interaction:
- `dock_state: DockState<Tab>` — egui_dock panel layout
- `node_graph_state: NodeGraphState` — Graph editor selection, pan/zoom
- Show flags: `show_debug`, `show_help`, `show_export_dialog`, `show_settings`
- `renaming_node`, `rename_buf` — In-place rename state
- `scene_tree_search` — Filter text for scene tree
- `isolation_state` — Isolation mode (show selected subtree only)
- `toasts: Vec<Toast>` — Active notifications
- `command_palette_open`, `command_palette_query` — Command palette state
- `sculpt_convert_dialog`, `import_dialog` — Modal dialog state
- `active_light_ids: HashSet<NodeId>` — Nearest MAX_SCENE_LIGHTS to camera
- `property_clipboard` — Copied material properties

**`PersistenceState`** — Save/load tracking:
- `current_file_path`, `scene_dirty`, `saved_fingerprint`, `last_auto_save`

**`PerfState`** — Performance monitoring:
- `timings: FrameTimings` — CPU phase breakdown
- `resolution_upgrade_pending` — Full-res render pending after interaction

#### `src/app/actions.rs` — Action System

**`Action`** enum — All structural state-mutating intents (~60 variants):

| Category | Variants |
|----------|----------|
| Scene | `NewScene`, `OpenProject`, `OpenRecentProject`, `SaveProject` |
| Selection | `Select(Option<NodeId>)`, `DeleteSelected`, `DeleteNode` |
| Clipboard | `Copy`, `Paste`, `Duplicate` |
| History | `Undo`, `Redo`, `SculptUndo`, `SculptRedo` |
| Camera | `FocusSelected`, `FrameAll`, `CameraFront`/`Top`/`Right`/`Back`/`Left`/`Bottom`, `ToggleOrtho` |
| Tools | `SetTool`, `SetGizmoMode`, `ToggleGizmoSpace`, `ResetPivot` |
| Sculpt | `EnterSculptMode`, `ShowSculptConvertDialog`, `CommitSculptConvert` |
| Scene Mutations | `CreatePrimitive`, `CreateOperation`, `CreateTransform`, `CreateModifier`, `CreateLight`, `InsertModifierAbove`, `InsertTransformAbove`, `ReparentNode`, `RenameNode`, `ToggleVisibility`, `ToggleLock`, `SwapChildren` |
| Graph | `SetLeftChild`, `SetRightChild`, `SetSculptInput` |
| Bake/Export | `RequestBake`, `ShowExportDialog`, `ImportMesh`, `CommitImport`, `TakeScreenshot` |
| Viewport | `ToggleIsolation`, `CycleShadingMode`, `ToggleTurntable` |
| Workspace | `SetWorkspace(WorkspacePreset)` |
| Light Linking | `SetLightMask`, `ToggleLightMaskBit` |
| Lighting Presets | `ApplyLightingPreset(LightingPreset)` |
| Settings | `SettingsChanged`, `MarkBufferDirty` |

**`ActionSink`** = `Vec<Action>` — UI components push actions during drawing; never mutate state directly.

**`LightingPreset`** enum — Studio, Outdoor, Dramatic, Flat.

#### `src/app/action_handler.rs` — Redux Reducer

**`process_actions()`** — Iterates all collected actions and applies mutations. This is the **single mutation point** for structural changes (analogous to a Redux reducer). Each action variant has a match arm that:
- Mutates `self.doc.scene` (add/remove/reparent nodes)
- Updates `self.doc.history` (push undo snapshots)
- Changes `self.ui` state (open dialogs, update selection)
- Marks `self.gpu.buffer_dirty` when scene data changes
- Resets `self.gpu.current_structure_key` when topology changes

Helper functions:
- `apply_lighting_preset_to_scene()` — Finds Key/Fill/Ambient light nodes by name, updates properties
- `find_parent_transform()` — Locates the Transform node parenting a given node
- `validate_sculpt_resolution()` — Checks if resolution fits in GPU storage buffer limit (320³ max)
- `duplicate_and_offset()` — Clone subtree with position offset

#### `src/app/input.rs` — Keyboard Shortcuts

**`collect_keyboard_actions()`** — Two-phase processing:
1. **Immutable detection**: Iterate keymap bindings, check context (sculpt mode, text focus, etc.), detect key presses → build `triggered` list
2. **Mutable dispatch**: Apply triggered bindings. Direct state mutation for brush modes (zero-latency). Push actions for structural changes.

Context-sensitive behaviors:
- Undo/Redo → SculptUndo/SculptRedo when in sculpt mode
- Brush mode keys (Add, Carve, Smooth, etc.) → direct SculptState mutation
- Bracket keys `[`/`]` → brush resize (0.05 step, clamp 0.05–2.0)
- Symmetry toggles (X/Y/Z) → toggle symmetry_axis
- Camera bookmarks (Ctrl+1-9) → SaveBookmark actions

#### `src/app/gpu_sync.rs` — GPU Synchronization

**`sync_gpu_pipeline()`**:
- Hash scene topology via `scene.structure_key()`
- If changed: regenerate WGSL shaders (`codegen::generate_shader()`, `generate_pick_shader()`), rebuild pipelines, mark buffer dirty
- If composite enabled: rebuild composite pipelines

**`upload_scene_buffer()`**:
- Build voxel buffer (concatenated sculpt grids)
- Build node buffer (SdfNodeGpu per visible node)
- Collect sculpt texture info
- Upload to GPU via ViewportResources

**Incremental updates**:
- `try_incremental_voxel_upload()` — Upload only brush-affected z-slices
- `upload_voxel_texture_region()` — Update texture3D region after brush stroke

#### `src/app/async_tasks.rs` — Background Tasks

**Bake (voxelization)**:
- Native: `std::thread::spawn` + mpsc channel
- WASM: synchronous (no threading)
- `poll_async_bake()` → check receiver → `apply_bake_result()` creates/updates sculpt node

**Export**:
- Spawn thread → marching cubes → write mesh file
- Progress tracked via `Arc<AtomicU32>`
- `poll_export()` checks receiver each frame

**Import**:
- Spawn thread → load mesh → voxelize to grid
- `poll_import()` checks receiver

**Screenshot**:
- Render to offscreen texture → encode PNG → save file

#### `src/app/sculpting.rs` — Per-Frame Brush Application

**`poll_sculpt_pick()`**:
- Poll GPU async pick buffer (1-frame delay, no stall)
- Route to `handle_sculpt_hit()` (brush stroke) or `handle_hover_pick()` (preview only)

**`handle_sculpt_hit()`**:
- Apply pen pressure
- Check if hit is child of active sculpt (differential mode)
- Interpolate brush stroke between frames
- Apply brush to CPU voxel grid
- Dispatch GPU compute brush shader
- Upload modified z-slab to GPU

#### `src/app/ui_panels.rs` — Menu Bar & Overlays

**`show_menu_bar()`** — File, Edit, View, Scene, Render menus. All items push actions.

**`show_status_bar()`** — Frame time, FPS, resolution scale, export progress.

### 2.3 Scene Graph (`src/graph/`)

#### `src/graph/scene.rs` — Core Data Model

**`NodeId`** = `u64` — Unique opaque identifier, monotonically incrementing.

**`SceneNode`** struct:
- `id: NodeId`, `name: String`, `data: NodeData`, `locked: bool`

**`NodeData`** enum (6 variants):

| Variant | Children | Key Fields |
|---------|----------|------------|
| `Primitive` | None | `kind: SdfPrimitive`, position, rotation, scale, color, metallic, roughness, emissive, fresnel |
| `Operation` | `left`, `right` | `op: CsgOp`, `smooth_k`, `steps`, `color_blend` |
| `Sculpt` | `input` (optional) | `voxel_grid: VoxelGrid`, position, rotation, material properties, `desired_resolution` |
| `Transform` | `input` | `translation`, `rotation`, `scale` |
| `Modifier` | `input` | `kind: ModifierKind`, `value: Vec3`, `extra: Vec3` |
| `Light` | None | `light_type: LightType`, `color`, `intensity`, `range`, `spot_angle` |

**`SdfPrimitive`** enum (10 variants): Sphere, Box, Cylinder, Torus, Plane, Cone, Capsule, Ellipsoid, HexPrism, Pyramid.

**`CsgOp`** enum (13 variants): Union, SmoothUnion, Subtract, SmoothSubtract, Intersect, SmoothIntersect, ChamferUnion/Subtract/Intersect, StairsUnion/Subtract, ColumnsUnion/Subtract.

**`ModifierKind`** enum (12 variants): Twist, Bend, Taper, Round, Onion, Elongate, Mirror, Repeat, FiniteRepeat, RadialRepeat, Offset, Noise.

**`LightType`** enum (4 variants): Point, Spot, Directional, Ambient.

**`Scene`** struct:
```rust
pub struct Scene {
    pub nodes: HashMap<NodeId, SceneNode>,
    pub next_id: u64,
    pub name_counters: HashMap<String, u32>,  // Auto-naming: Sphere, Sphere_1, ...
    pub hidden_nodes: HashSet<NodeId>,
    pub light_masks: HashMap<NodeId, u8>,     // Per-node light linking (default 0xFF)
}
```

**Key methods:**
- `structure_key()` → `u64` hash of node topology (connections, types, visibility). Used for shader rebuild detection.
- `data_fingerprint()` → `u64` hash of all node property data. Used for buffer dirty detection.
- `compute_bounds()` → `([f32; 3], [f32; 3])` world-space AABB, walks tree applying transforms.
- `visible_topo_order()` → `Vec<NodeId>` post-order traversal of visible nodes.
- `content_eq()` → Deep equality (used by undo/redo).
- `build_parent_map()` → `HashMap<NodeId, NodeId>` child→parent mapping.
- `is_descendant()` → Check parent-child relationship in tree.
- `create_primitive()`, `create_operation()`, `create_transform()`, `create_modifier()`, `create_light()` → Node factory methods.
- `create_default_lights()` → Creates Key Light (Directional), Fill Light (Directional), Ambient Light.
- `remove_node()` → Recursive deletion with parent rewiring and light mask cleanup.

**Constants:**
- `MAX_SCENE_LIGHTS = 8` — Maximum lights rendered per frame (nearest to camera).

#### `src/graph/voxel.rs` — Voxel Grid & CPU SDF Evaluation

**`VoxelGrid`** struct:
```rust
pub struct VoxelGrid {
    pub resolution: u32,       // Typically 96–256
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub is_displacement: bool, // false=total SDF, true=displacement-only
    pub data: Vec<f32>,        // Flat: z*res*res + y*res + x
}
```

**Two modes:**
- **Total SDF** (`is_displacement=false`): Data stores complete distances. Unfilled = 999.0 (FAR_DISTANCE).
- **Differential** (`is_displacement=true`): Data stores displacement from analytical base. Unfilled = 0.0.

**Sparse serialization**: Custom serde module encodes only non-fill entries as `(index, value)` pairs. 10-100x file size reduction for typical grids.

**CPU SDF evaluation:**
- `evaluate_sdf_tree()` — Recursive evaluation from scene tree (primitives, CSG, transforms, modifiers, voxel sampling)
- `evaluate_sdf_tree_with_blend()` — Extended version returning (distance, material_id, blend_factor)

**Brush application:**
- `apply_brush()` — Main dispatcher (Add, Carve, Smooth, Flatten, Inflate)
- `apply_smooth_to_grid()` — Laplacian smoothing with falloff
- `apply_grab_to_grid()` — Trilinear-interpolated grab brush
- Returns `(z0, z1)` dirty z-slab range for incremental GPU upload

**Noise functions** (matching WGSL for GPU/CPU consistency):
- `hash33()`, `noise3d()`, `fbm_noise()` — Perlin-style 3D gradient noise + FBM

#### `src/graph/history.rs` — Undo/Redo

**Clone-based snapshot system:**

```rust
struct Snapshot {
    scene: Scene,              // Full deep clone
    selected: Option<NodeId>,
    label: String,             // "Add Sphere", "Delete Box", "Edit", etc.
}
```

- `MAX_UNDO_DEPTH = 50` — Circular buffer, oldest evicted
- **Drag coalescing**: Holds snapshot during property drags, commits on mouse release → single undo entry
- `begin_frame()` captures "before" state; `end_frame()` commits if changed
- `detect_change_label()` auto-detects action name from scene diff

### 2.4 GPU Pipeline (`src/gpu/`)

#### `src/gpu/buffers.rs` — GPU Resource Management

**`SdfNodeGpu`** (128 bytes = 8 × vec4f, `#[repr(C)]`, `Pod`, `Zeroable`):

| Field | Contents |
|-------|----------|
| `type_op` | [type_id, smooth_k/intensity, metallic/range, roughness/spot_angle] |
| `position` | [x, y, z, layer_intensity] |
| `rotation` | [rx, ry, rz, 0] |
| `scale` | [sx, sy, sz, **light_mask**] |
| `color` | [r, g, b, is_selected] |
| `extra0` | [voxel_offset, resolution, emissive.x, emissive.y] |
| `extra1` | [bounds_min.xyz, emissive.z] |
| `extra2` | [bounds_max.xyz, fresnel] |

**GPU Type IDs:**

| Range | Node Type |
|-------|-----------|
| 0–9 | Primitives (Sphere=0, Box=1, Cylinder=2, Torus=3, Plane=4, Cone=5, Capsule=6, Ellipsoid=7, HexPrism=8, Pyramid=9) |
| 10–22 | CSG Operations |
| 20 | Sculpt |
| 21 | Transform |
| 30–41 | Modifiers |
| 50–53 | Lights (Point=50, Spot=51, Directional=52, Ambient=53) |

**`SceneLightGpu`** (64 bytes):
- `position_type`: [pos.xyz, type (0=point, 1=spot, 2=directional)]
- `direction_intensity`: [dir.xyz, intensity]
- `color_range`: [color.rgb, range]
- `params`: [cos_half_spot_angle, 0, 0, 0]

**`SceneAmbient`**: Accumulated color from visible Ambient light nodes.

**Key functions:**
- `build_node_buffer()` → `Vec<SdfNodeGpu>` from `visible_topo_order()`
- `build_voxel_buffer()` → `(Vec<f32>, HashMap<NodeId, u32>)` concatenated voxel data + offset map
- `collect_scene_lights()` → `(count, Vec<SceneLightGpu>, SceneAmbient)` sorted by distance to camera
- `identify_active_lights()` → `(HashSet<NodeId>, total_count)` for UI indicators

#### `src/gpu/camera.rs` — Camera & Projection

**`Camera`** struct (serializable):
- `yaw`, `pitch`, `roll` (radians), `distance` (0.1–100.0), `target` (Vec3), `fov` (45° default), `orthographic`
- `transition: Option<ViewTransition>` — Smooth 0.3s view preset animation with ease_out_cubic

**Methods:**
- `orbit(dx, dy)`: yaw += dx × 0.005, pitch += dy × 0.005 (clamped ±89°)
- `pan(dx, dy)`: target += right × dx + up × dy (scaled by distance)
- `zoom(scroll)`: distance *= (1 - scroll × 0.001)
- `focus_on(center, radius)`: Auto-frame with adaptive distance
- `start_transition()` / `tick_transition()`: Smooth animated view changes
- `projection_matrix(aspect)`: Perspective or orthographic (half_height = distance × tan(fov/2))

**`CameraUniform`** (`#[repr(C)]`, `Pod`, `Zeroable`, 368 bytes):
- `inv_view_proj`: mat4×4 (64 bytes)
- `eye`, `viewport`: vec4f each
- `time`, `quality_mode`, `grid_enabled`, `selected_idx`: 4 × f32
- `scene_min`, `scene_max`: vec4f (shading_mode packed in scene_min.w)
- `brush_pos`: [x, y, z, radius]
- `cross_section`: [axis, position, 0, 0]
- `ambient_info`: [ambient_intensity, 0, 0, 0]
- `scene_light_info`: [count, 0, 0, 0]
- `scene_lights`: 8 lights × 4 vec4f = 128 vec4f (512 bytes)

#### `src/gpu/codegen.rs` — Runtime WGSL Generation

Converts the scene graph into WGSL shader source code at runtime via post-order tree traversal.

**Key functions:**
- `generate_shader()` → Main render shader with scene_sdf() function
- `generate_pick_shader()` → 1×1 pick pass shader
- `generate_composite_shader()` → 3D volume compute shader
- `generate_composite_render_shader()` → Volume visualization

**Two-phase optimization for expensive subtrees:**
1. Cheap subtrees (no sculpt) evaluated first → accumulated into `result`
2. Expensive subtrees (with sculpt) wrapped in `if _bd < result.x { ... }` using precomputed bounding sphere → skips entire subtree evaluation when out of range

**Transform chain helpers:**
- `get_transform_chain()` — Walks ancestors collecting Transform + point-modifying Modifier nodes
- Returns chain from innermost to outermost for proper composition

**Voxel texture declarations:**
- `generate_voxel_texture_decls()` — Emits per-sculpt sampling functions
- Standalone: `sdf_voxel_tex_N()` (box_dist + texture sample)
- Differential: `disp_voxel_tex_N()` (displacement only within bounds)

#### `src/gpu/picking.rs` — GPU Pick Pass

**Encoding scheme** (1×1 Rgba8Unorm texture):
- 0 = background (miss)
- 1 = ground plane
- 2+ = scene node (node = material_id - 2, mapped via topo_order index)
- 253/254/255 = gizmo axes (X/Y/Z)

**`PickResult`**: material_id, world_pos, normal.

**Flow:**
1. Viewport click → `PendingPick` stored in async_state
2. Submit GPU pick render (1×1 texture + staging buffer readback)
3. Next frame: poll staging buffer → decode PickResult → route to selection or sculpt

#### `src/gpu/shader_templates.rs` — Shader Module Assembly

Assembles WGSL from individual `.wgsl` files via `include_str!()`:
- **Render prelude**: bindings + voxel_sampling + vertex + transforms + primitives + modifiers + noise + operations
- **Compute prelude**: Same but without vertex shader (compute shaders would error with @vertex)

`apply_march_placeholders()` replaces `/*MARCH_MAX_STEPS*/`, `/*MARCH_EPSILON*/`, etc. with values from RenderConfig before shader compilation.

### 2.5 Shaders (`src/shaders/`)

13 WGSL shader files:

| File | Purpose |
|------|---------|
| `bindings.wgsl` | Struct definitions (Camera, SdfNode) + bind group declarations |
| `vertex.wgsl` | Fullscreen triangle (no vertex input, generates 3 vertices procedurally) |
| `primitives.wgsl` | SDF distance functions for 10 primitive shapes (iq-inspired) |
| `operations.wgsl` | CSG operations (union, subtract, intersect + smooth/chamfer/stairs/columns variants) |
| `transforms.wgsl` | Transform matrix utilities, point/normal transforms |
| `modifiers.wgsl` | Domain deformations (twist, bend, taper, repeat, mirror, etc.) |
| `noise.wgsl` | Perlin-style 3D gradient noise + FBM (matches CPU implementation) |
| `rendering.wgsl` | Raymarching loop, PBR lighting, shadows, AO, SSS, fog |
| `voxel_sampling.wgsl` | Voxel grid sampling (storage buffer path) |
| `pick.wgsl` | GPU pick pass (1×1 render, material ID encoding) |
| `brush.wgsl` | Sculpt brush compute shader |
| `blit.wgsl` | Composite/outline post-process (edge detection for selection outline) |
| `composite_entry.wgsl` | Composite pass entry point (3D volume bake) |

**Bind groups:**

| Group | Binding | Resource |
|-------|---------|----------|
| @group(0) | @binding(0) | `var<uniform> camera: Camera` (CameraUniform) |
| @group(1) | @binding(0) | `var<storage, read> nodes: array<SdfNode>` |
| @group(1) | @binding(1) | `var<storage, read> voxel_data: array<f32>` |
| @group(2) | @binding(0) | `var voxel_sampler: sampler` |
| @group(2) | @binding(1+) | `var voxel_tex_N: texture_3d<f32>` (per sculpt) |

**Raymarching configuration (compile-time placeholders):**
- `MARCH_MAX_STEPS` = 96 (interactive: 48)
- `MARCH_EPSILON` = 0.0005
- `MARCH_STEP_MULT` = 0.9 (conservative step)
- `MARCH_MAX_DIST` = 100.0

**Enhanced sphere tracing (Keinert over-relaxation):**
- omega = 1.2 in full quality, 1.0 in interactive mode
- Overshoot detection: if combined radii exceed step, fallback to omega = 1.0
- ~20% fewer steps than basic sphere tracing

**Lighting pipeline:**
- Tetrahedron normals (4 SDF evals vs 6 central differences)
- Distance-adaptive normal epsilon: `clamp(0.001*t, 0.0005, 0.05)`
- Cook-Torrance PBR: GGX NDF, Smith GGX geometric attenuation, Schlick Fresnel
- Improved soft shadows with ph tracking (iq/Aaltonen variant)
- 5-sample AO with exponential decay
- Scene lights loop with bitmask filtering for light linking
- Shading modes: Full PBR, Solid (flat diffuse), Clay (uniform gray), Normals, Matcap, StepHeatmap, CrossSection

### 2.6 UI Layer (`src/ui/`)

22 modules providing all user interface components.

#### `src/ui/dock.rs` — Panel Layout

**`Tab`** enum (9 variants): Viewport, NodeGraph, Properties, SceneTree, RenderSettings, History, BrushSettings, Lights, LightLinking.

**`SdfTabViewer`** — Implements `egui_dock::TabViewer`. Holds references to scene, camera, settings, actions, and context bundles. Dispatches to module-specific `draw()` functions per tab.

**Preset layouts:**
- `create_dock_state()` — Default modeling (80/20 split)
- `create_dock_sculpting()` — Brush-focused (82/18 split)
- `create_dock_rendering()` — Render settings side-by-side

#### `src/ui/viewport/` — 3D Viewport (6 files)

**`viewport/mod.rs`** — `ViewportResources`: manages render pipeline, pick pipeline, brush pipeline, blit pipeline, storage buffers, voxel textures, offscreen render targets.

**`viewport/draw.rs`** — Main viewport drawing:
- `ViewportCallback` implements `egui_wgpu::CallbackTrait` for custom GPU rendering
- CPU voxel raycast (`cursor_in_sculpt_bounds()`) for zero-latency orbit-vs-sculpt detection
- Brush preview, symmetry plane overlay, node labels, light gizmos
- `ViewportOutput` carries results (pending_pick, modifier keys, brush deltas) back to dock

**`viewport/pipelines.rs`** — Pipeline creation for render, pick, brush compute, and blit passes.

**`viewport/textures.rs`** — Creates Texture3D per sculpt node (R32Float, trilinear filtering).

**`viewport/gpu_ops.rs`** — Buffer uploads, pick dispatch, brush compute dispatch.

**`viewport/composite.rs`** — Optional cached SDF volume for performance. Dual-path rendering: direct sphere-trace vs cached volume.

#### `src/ui/scene_tree.rs` — Hierarchy Panel

- Recursive tree with type-colored dots (Primitive=blue, Operation=green, Sculpt=orange, Transform=purple, Modifier=yellow, Light=warm yellow)
- Drag-and-drop reparenting with drop zone validation
- Search/filter with flat results display
- In-place rename, visibility toggle (eye icon), context menu deletion

#### `src/ui/properties.rs` — Node Inspector

- Material presets (Gold, Silver, Chrome, Plastic, Ceramic, Rubber, Glow)
- Color presets (R, G, B, Y, O, W, Gray, Tan)
- Per-node light linking collapsing section
- Property clipboard (copy/paste material properties)
- Direct `&mut Scene` mutation for data-level edits (zero-latency sliders)

#### `src/ui/node_graph.rs` — Visual Node Graph

- `egui_node_graph2` integration with custom types
- Port colors, node colors per type
- Multi-select with primary + set tracking
- Node finder for creating new nodes via search
- Bidirectional sync: graph ↔ scene (graph is derived view, scene is source of truth)

#### `src/ui/gizmo.rs` — Transform Gizmo

- Modes: Translate, Rotate, Scale
- Space: Local, World
- Depth-sorted axis rendering with hover highlight
- Snap quantization during drag (configurable in SnapConfig)
- Math utilities: `rotate_euler()`, `euler_to_quat()`, `quat_to_euler_stable()`, `world_to_screen()`

#### `src/ui/light_gizmo.rs` — Light Billboards

- Distance-adaptive sizing (12–48 px) and fading
- Type-specific icons: Point (radiating rays), Spot (cone), Directional (sun), Ambient (concentric rings)
- Wireframe gizmos when selected: range sphere, spot cone, direction arrows
- Hit testing for selection (16px radius)

#### `src/ui/lights_panel.rs` — Light Management

- Quick-add buttons (+Point, +Spot, +Directional, +Ambient)
- Per-light row: eye toggle, active indicator, color swatch, type badge, name, intensity slider
- Active/inactive indicators (green dot for GPU-active, gray for over-limit)
- Click-to-select routing

#### `src/ui/light_linking.rs` — Light Linking Matrix

- Maya/Houdini-style matrix: rows = geometry nodes, columns = active lights
- Per-cell checkboxes for light-geometry linking
- Bulk actions: Link All, Unlink All, Reset All (per row, per column, global)
- Pushes `ToggleLightMaskBit` and `SetLightMask` actions

#### Other UI Modules

| Module | Purpose |
|--------|---------|
| `render_settings.rs` | Quality presets (Fast/Balanced/Quality), lighting presets, per-category controls |
| `export_dialog.rs` | Resolution, format, bounds, progress display |
| `import_dialog.rs` | Mesh stats, resolution auto-calculation, voxelization options |
| `sculpt_convert_dialog.rs` | Bake mode selection (whole scene, active node, flatten) |
| `settings_window.rs` | Application settings (vsync, autosave, keybinds) |
| `brush_settings.rs` | Sculpt brush parameters (mode, radius, strength, falloff, shape) |
| `command_palette.rs` | Ctrl+P searchable command palette |
| `quick_toolbar.rs` | Shift+A quick-add overlay |
| `profiler.rs` | FPS sparkline, CPU phase breakdown, scene stats |
| `toasts.rs` | Bottom-right notifications (green=success 4s, red=error 6s) |
| `help.rs` | Help/about dialog |
| `history_panel.rs` | Undo/redo history list |
| `export_progress.rs` | Export progress modal |

### 2.7 Other Modules

#### `src/settings.rs` — Configuration

**`RenderConfig`** — Extensive render quality settings:
- Shadows (enabled, steps, penumbra_k, bias, min_t, max_t)
- AO (enabled, samples, step, decay, intensity)
- Raymarching (max_steps, epsilon, step_multiplier, max_distance)
- Lighting (ambient)
- Sky/Background (mode, horizon/zenith colors, solid color)
- Effects (SSS, fog, bloom with threshold/intensity/radius)
- Gamma/tonemapping (gamma value, ACES toggle)
- Outline (color, thickness)
- Viewport (grid, node labels, bounding box, light gizmos)
- Performance (sculpt_fast_mode, auto_reduce_steps, interaction/rest render scales)

**`ShadingMode`** enum: Full, Solid, Clay, Normals, Matcap, StepHeatmap, CrossSection. Each has `gpu_value()` (0–6), `label()`, `cycle()`.

**`SnapConfig`**: translate=0.25, rotate=15°, scale=0.1 defaults.

**Persistence**: JSON file next to executable (native) or localStorage (WASM).

#### `src/io.rs` — File I/O

**`ProjectFile`** (v5): `{ version, scene, camera }` serialized as JSON.

- `project_to_json()` / `json_to_project()` — Pure serialization
- `save_project()` / `load_project()` — Native file I/O (rfd dialogs)
- `web_save_project()` / `web_download()` — WASM download
- **v4→v5 migration**: Transform nodes changed from {kind, value} to {translation, rotation, scale}

#### `src/sculpt.rs` — Sculpt Tool Definitions

**`ActiveTool`**: Select | Sculpt

**`BrushMode`** (6 modes): Add, Carve, Smooth, Flatten, Inflate, Grab.

**`BrushShape`** (5 shapes): Sphere, Cube, Diamond, Ring, Cylinder.

**`FalloffMode`** (4 curves): Smooth (Hermite), Linear, Sharp (quadratic), Flat.

**`SculptState`**: Inactive | Active { node_id, brush parameters, symmetry, grab snapshot }.

#### `src/sculpt_history.rs` — Sculpt Undo/Redo

Per-stroke voxel grid snapshots (separate from scene history). Captures full grid clone before each brush stroke.

#### `src/export.rs` — Mesh Export

Hand-rolled marching cubes with Paul Bourke lookup tables:
- rayon-parallelized by z-slice
- Two-phase vertex deduplication (local per-slice, global merge)
- 5 format writers (OBJ, STL binary, PLY ASCII, glTF binary, USD ASCII) — all pure Rust, std::io only
- Async: thread + mpsc + `Arc<AtomicU32>` progress counter
- Vertex colors sampled from scene SDF materials

#### `src/mesh_import.rs` — Mesh Import

Loads external meshes (OBJ format) and voxelizes them into SDF grids. Resolution auto-calculated from mesh bounds.

#### `src/keymap.rs` — Configurable Keybindings

**`KeyCombo`**: key + modifiers (Ctrl, Shift, Alt). Serializable.

**`ActionBinding`** enum: Named actions (Undo, Redo, Delete, etc.) mappable to key combos.

- `matches_egui()` — Runtime matching against egui input
- Conflict detection, reset to defaults, serialization roundtrip

#### `src/compat.rs` — Platform Abstractions

- `Instant` / `Duration`: `std::time` (native) or `web_time` (WASM)
- `maybe_par_iter!` macro: `into_par_iter()` (native/rayon) or `into_iter()` (WASM)

---

## 3. Code Style, Patterns, and Conventions

### 3.1 Ownership and Borrowing

- **No `unsafe` blocks** anywhere in the codebase (0 occurrences)
- **No `Rc`, `RefCell`** — all state is owned by `SdfApp` sub-structs with clear borrowing
- **wgpu Device/Queue** are internally `Arc`'d by eframe; obtained via `cc.wgpu_render_state().device.clone()`
- **Clone-based undo** — `Scene` is `Clone` + `Serialize`; snapshots are full deep clones
- **`bytemuck::Pod` + `Zeroable`** for GPU structs (safe casting, no unsafe)
- **`Arc<AtomicU32>`** for cross-thread progress counters (export, bake, import)

### 3.2 Error Handling

- **Minimal error propagation** — file I/O returns `Result`, most internal code uses `unwrap_or()` / `Option`
- **No `thiserror`, `anyhow`**, or custom error types
- **Toast notifications** for user-facing errors (export failure, GPU buffer limit exceeded, etc.)
- **Defensive cleanup** — selection validated after every action pass; isolation state reset on undo/redo

### 3.3 Async Strategy

- **Not tokio** — uses `std::thread::spawn` + `std::sync::mpsc` channels for native
- **Synchronous fallback** on WASM (no threads available)
- **1-frame delay** for GPU pick reads (async staging buffer readback, no CPU stall)
- **`poll_*()` functions** checked each frame in phase 2 of update loop

### 3.4 Action System (Redux-Inspired)

Two categories of state mutation:

| Category | Mechanism | Latency | Examples |
|----------|-----------|---------|----------|
| **Structural** | Action → process_actions() | 1 frame | Add/delete node, reparent, bake, tool switch |
| **Data-level** | Direct `&mut Scene` | 0 frames | Slider drag, color pick, gizmo transform |

This split ensures zero-latency feedback for interactive edits while maintaining a single mutation point for structural changes.

### 3.5 Two-Speed GPU Synchronization

| Change Type | Detection | Action | Cost |
|-------------|-----------|--------|------|
| Topology (add/delete/reconnect) | `structure_key()` hash | Full shader recompile + pipeline rebuild | ~10ms |
| Data (slider/color/position) | `data_fingerprint()` hash | Buffer upload only | ~1ms |

### 3.6 Naming Conventions

- **All Rust** `snake_case` for functions, methods, variables, modules
- **PascalCase** for types, traits, enum variants
- **Descriptive names** prioritized over brevity (e.g., `sculpt_convert_dialog`, `apply_lighting_preset_to_scene`)
- **No single-letter variables** except established math contexts (x, y, z, t, i, n, dt, vp)
- **File names match module names** (e.g., `action_handler.rs` contains `process_actions()`)

### 3.7 Module Layout

- **One concern per file** — no monolithic files
- **Related files grouped** in folders: `app/`, `gpu/`, `graph/`, `ui/`, `ui/viewport/`
- **22 UI modules**, each with a `draw()` function taking `&mut egui::Ui` + references + `&mut ActionSink`
- **13 WGSL shaders**, each for a single responsibility

### 3.8 Testing

**380 tests** — all unit tests in `#[cfg(test)]` blocks within source files.

| Module | Test Count | What's Tested |
|--------|-----------|---------------|
| `gpu/codegen.rs` | ~80 | WGSL generation correctness + naga shader validation |
| `graph/scene.rs` | ~70 | Node creation, deletion, topology, serialization, light masks |
| `graph/voxel.rs` | ~60 | SDF evaluation, brush application, noise, grid math |
| `export.rs` | ~50 | Marching cubes, format writers, vertex dedup |
| `graph/history.rs` | ~20 | Undo/redo, drag coalescing, label detection |
| `gpu/buffers.rs` | ~15 | Buffer layout, light collection, type encoding |
| `ui/node_graph.rs` | ~10 | Multi-select state machine |
| `keymap.rs` | ~10 | Key combo matching, serialization, conflict detection |
| `app/state.rs` | ~5 | Import dialog auto-resolution |

**Not tested**: UI rendering, GPU operations (no wgpu in tests), keyboard input handling, viewport drawing.

**Run with:** `cargo test` (all), `cargo test -- codegen` (filtered)

---

## 4. Design Decisions and Rationale

### 4.1 Runtime WGSL Codegen (vs Compile-Time)

**Decision**: Generate WGSL shader source code at runtime from the scene tree.

**Why**: SDF scenes have dynamic topology — users add, remove, and reconnect nodes at will. Compile-time shader generation would require predefined scene structures. Runtime codegen allows arbitrary scene complexity.

**Trade-off**: ~10ms shader recompile on topology change. Mitigated by structure_key caching — property-only changes skip recompilation entirely.

### 4.2 Clone-Based Undo (vs Command Pattern)

**Decision**: Full scene snapshots (deep clone) for undo/redo.

**Why**: Scene has complex internal references (NodeId-based parent/child). Command-pattern undo would need inverse operations for every possible mutation — error-prone and hard to maintain. Cloning is simpler and guarantees correctness.

**Trade-off**: Memory cost (~50 snapshots × scene size). Acceptable for typical scenes. Large sculpt grids (256³ × 4 bytes = 67MB each) could be expensive. Sculpt-specific undo uses separate per-stroke snapshots.

### 4.3 Hand-Rolled Marching Cubes (vs Library)

**Decision**: Custom implementation with Paul Bourke tables.

**Why**: Project needs tight integration with its SDF evaluation pipeline, rayon parallelization by z-slice, and custom vertex color sampling. Library crates would wrap the same algorithm in a generic API without these integrations.

**Trade-off**: ~300 lines of lookup tables. Offset by zero external dependencies for export (pure Rust + std::io).

### 4.4 Immediate-Mode UI via egui (vs Retained)

**Decision**: egui (immediate-mode) instead of retained-mode GUI toolkit.

**Why**: Matches Rust's ownership model naturally — no widget lifetime management, no callback closures holding references. Rapid iteration on UI. Excellent wgpu integration via eframe.

**Trade-off**: Re-draws entire UI every frame. Mitigated by eframe's repaint-on-demand (only repaints when requested). No complex state synchronization needed.

### 4.5 egui_wgpu Paint Callback (vs Raw wgpu)

**Decision**: Render SDF scene inside egui's paint callback system.

**Why**: Leverages eframe's device/queue/surface management. No need to manage window, swapchain, or resize handling manually. Viewport integrates naturally as an egui widget.

**Trade-off**: Constrained to egui's render pass lifecycle. Offscreen rendering + blit required for resolution scaling and post-processing (outline edge detection, bloom).

### 4.6 Per-Subtree Bounding Skip (vs BVH)

**Decision**: Wrap expensive subtrees (with sculpt/voxel reads) in AABB distance checks in generated WGSL.

**Why**: Exploits SDF tree structure directly. No need to build and maintain a separate BVH. Codegen naturally identifies expensive subtrees and inserts conditional skips.

**Trade-off**: Only applies to top-level unions. Deeply nested expensive subtrees may not benefit. Effective for typical scenes with 1-3 sculpt nodes.

### 4.7 Light Linking Bitmask (vs Linking Table)

**Decision**: 8-bit bitmask per geometry node (`u8` packed into `scale.w` on GPU).

**Why**: MAX_SCENE_LIGHTS = 8 → `u8` is optimal. Maps directly to GPU bit operations. Trivially serializable. Per-pixel cost is a single bitwise AND.

**Trade-off**: Hard limit of 8 simultaneous lights. Beyond 8, only nearest lights are active (sorted by distance to camera).

### 4.8 CPU Voxel Raycast for Sculpt Detection

**Decision**: CPU-side voxel ray-AABB intersection + sphere tracing for orbit-vs-sculpt mode detection.

**Why**: GPU pick has 1-frame latency (async staging buffer readback). For sculpt mode, instant detection is critical — user needs to know immediately whether they're sculpting or orbiting.

**Trade-off**: CPU ray-march through voxel grid (~10 steps, <0.1ms). Only runs when cursor is over viewport during sculpt mode.

---

## 5. Workarounds and Technical Debt

### 5.1 Grab Brush Temporary Swap

**Location**: `src/sculpt.rs:621-636`

**What**: The grab brush needs to read from a snapshot while writing to the live grid. Due to borrow checker constraints, the VoxelGrid is temporarily swapped out of the scene node, operated on, then swapped back.

**Risk**: Low — swap is atomic within a single function call. No concurrent access possible.

**Ideal fix**: Refactor VoxelGrid storage to allow simultaneous snapshot + live grid access (e.g., separate snapshot field in SculptState).

### 5.2 Node Data Snapshot in Node Graph

**Location**: `src/ui/node_graph.rs`

**What**: Clones node data into `SdfGraphUserState.node_data_snapshot` before egui drawing to avoid nested borrow issues. Writes back dirty entries after UI pass.

**Risk**: Low — bounded by number of graph nodes. Cloning is O(nodes).

**Ideal fix**: Inherent limitation of egui's immediate-mode approach with mutable state. Would require architectural change to graph library.

### 5.3 WASM Build Blocked

**What**: wgpu sends deprecated `maxInterStageShaderComponents` to Chrome 135+, causing validation failure.

**Status**: Waiting on wgpu v29+ release.

**Impact**: WASM support is shelved. All WASM code paths exist and compile but cannot run in modern browsers.

### 5.4 128MB Storage Buffer Limit

**Location**: `src/lib.rs` (device descriptor), `src/app/action_handler.rs` (validation)

**What**: Custom `max_storage_buffer_binding_size = 1 << 27` (128MB). Maximum sculpt resolution is 320³ (320³ × 4 bytes ≈ 131MB).

**Risk**: Exceeding this limit causes GPU buffer creation failure. Validated before bake with user-visible error toast.

### 5.5 Hardcoded Light Limit

**Location**: `src/graph/scene.rs` (`MAX_SCENE_LIGHTS = 8`), `src/shaders/rendering.wgsl`

**What**: Only 8 lights can be active simultaneously. Nearest to camera are selected.

**Risk**: Scenes with >8 lights silently drop distant ones. Toast warning shown in lights panel.

**Ideal fix**: Clustered lighting or tiled deferred shading (significant rendering architecture change).

### 5.6 No Incremental Voxel Resampling

**What**: Changing sculpt resolution requires full rebake of the subtree. No interpolation between resolutions.

**Impact**: Resolution change is a destructive operation — detail may be lost at lower resolutions.

### 5.7 History Memory Pressure

**What**: 50 full scene clones can be expensive for scenes with large sculpt grids (256³ × 4 bytes = 67MB per snapshot).

**Impact**: Theoretical 50 × 67MB = 3.3GB for sculpt undo. In practice, sculpt-specific undo is separate (per-stroke, not full scene).

### 5.8 Viewport draw() Parameter Count

**Location**: `src/ui/viewport/draw.rs`

**What**: The main `draw()` function takes ~30 parameters. Intentionally verbose for clarity per project conventions, but creates a large function signature.

---

## 6. Maintenance and Onboarding Guide

### 6.1 Project Setup

**Prerequisites:**
- Rust stable toolchain (edition 2021) — install via [rustup](https://rustup.rs/)
- GPU with Vulkan, Metal, or DX12 support
- No additional system dependencies

**Build:**
```bash
cargo run              # Debug build (opt-level=2, ~90% release perf)
cargo run --release    # Release build with debug symbols
cargo check            # Type check only (fastest feedback loop)
```

**Profile configuration** (`Cargo.toml`):
- `[profile.dev]` opt-level = 2 — fast debug builds with near-release performance
- `[profile.release]` debug = true — includes debug symbols for profiling tools (flamegraph, samply)

### 6.2 Verification Commands

**Mandatory before every commit** (per CLAUDE.md):
```bash
cargo check                    # Type checking
cargo clippy -- -D warnings    # Lint errors are build failures
cargo test                     # All 380 tests must pass
cargo build                    # Full compilation
```

### 6.3 Adding a New Feature

#### Adding a New SDF Primitive

1. **`src/graph/scene.rs`**: Add variant to `SdfPrimitive` enum + `ALL` array + `base_name()` + `default_position()` + `default_scale()` + `gpu_type_id()` + `scale_params()` + `sdf_function_name()` + `badge()`
2. **`src/shaders/primitives.wgsl`**: Add `sdf_<name>(p, params) -> f32` function
3. **`src/gpu/codegen.rs`**: Add case in node codegen that calls the new WGSL function
4. **`src/graph/voxel.rs`**: Add case in `evaluate_sdf_tree()` for CPU-side evaluation
5. **Tests**: Add naga validation test in `codegen.rs`, SDF eval test in `voxel.rs`

#### Adding a New Modifier

1. **`src/graph/scene.rs`**: Add variant to `ModifierKind` enum + `ALL` + `base_name()` + `gpu_type_id()` + `is_point_modifier()` + `badge()` + `default_value()` + `default_extra()`
2. **`src/shaders/modifiers.wgsl`**: Add WGSL function
3. **`src/gpu/codegen.rs`**: Add case in modifier codegen
4. **`src/graph/voxel.rs`**: Add case in CPU evaluation
5. **Tests**: naga validation + CPU eval tests

#### Adding a New Action

1. **`src/app/actions.rs`**: Add variant to `Action` enum
2. **`src/app/action_handler.rs`**: Add match arm in `process_actions()`
3. **UI source**: Push the action in the appropriate UI component

#### Adding a New Panel (Tab)

1. **Create `src/ui/<panel_name>.rs`** with `pub fn draw(ui: &mut egui::Ui, ...)`
2. **`src/ui/mod.rs`**: Add `pub mod <panel_name>;`
3. **`src/ui/dock.rs`**: Add variant to `Tab` enum, update `ALL`, `label()`, `title()`, `ui()` dispatcher
4. **Optional**: Add to default dock layout

### 6.4 Safe Refactoring Areas

| Area | Safety Level | Why |
|------|-------------|-----|
| `src/export.rs` | High | Well-tested (50+ tests), isolated from GPU/UI |
| `src/graph/scene.rs` | High | Well-tested (70+ tests), clear ownership |
| `src/settings.rs` | High | Serialized with serde defaults, backward-compatible |
| `src/ui/` panels | Medium | No tests, but isolated draw functions with no shared state |
| `src/sculpt.rs` | Medium | Tested but coupled to voxel grid internals |

### 6.5 Brittle Areas

| Area | Risk | Why |
|------|------|-----|
| `src/gpu/codegen.rs` | High | WGSL string generation — typos cause shader compilation failure |
| `src/gpu/shader_templates.rs` | High | Placeholder replacement — must match shader comments exactly |
| `src/shaders/*.wgsl` | High | Must stay in sync with codegen and CameraUniform struct layout |
| `src/ui/viewport/draw.rs` | Medium | 30+ parameter function, complex input routing |
| `src/gpu/camera.rs` (CameraUniform) | High | `#[repr(C)]` struct must exactly match WGSL Camera struct layout |

### 6.6 Performance Profiling

**Built-in profiler** — Toggle with Debug key (or via View menu):
- FPS + frame time with color coding (green ≥55fps, yellow ≥30fps, red <30fps)
- 120-frame sparkline history with 60fps target line
- CPU phase breakdown (Pipeline sync, Buffer upload, Composite dispatch, UI draw)
- Scene stats (node count, voxel memory, bounds)
- Camera position/rotation
- GPU buffer capacities

**Known bottlenecks:**
- Shader recompilation on topology change (~10ms)
- Voxel buffer upload for large grids (256³ × 4 bytes = 67MB)
- History clone for scenes with large sculpt grids
- Raymarching step count (96 steps per pixel in full quality)

**Profiling with external tools:**
```bash
cargo run --release    # debug=true included for symbol resolution
# Then use samply, flamegraph, or platform profiler
```

### 6.7 Source File Inventory

**57 Rust source files:**

```
src/
├── main.rs, lib.rs, compat.rs, settings.rs, io.rs
├── export.rs, sculpt.rs, sculpt_history.rs, mesh_import.rs, keymap.rs
├── app/
│   ├── mod.rs, state.rs, actions.rs, action_handler.rs
│   ├── input.rs, gpu_sync.rs, async_tasks.rs, sculpting.rs, ui_panels.rs
├── gpu/
│   ├── mod.rs, buffers.rs, camera.rs, codegen.rs, picking.rs, shader_templates.rs
├── graph/
│   ├── mod.rs, scene.rs, voxel.rs, history.rs
└── ui/
    ├── mod.rs, dock.rs, gizmo.rs, properties.rs, scene_tree.rs
    ├── node_graph.rs, render_settings.rs, settings_window.rs
    ├── light_gizmo.rs, lights_panel.rs, light_linking.rs
    ├── export_dialog.rs, export_progress.rs, import_dialog.rs
    ├── sculpt_convert_dialog.rs, brush_settings.rs, quick_toolbar.rs
    ├── command_palette.rs, profiler.rs, toasts.rs, help.rs, history_panel.rs
    └── viewport/
        ├── mod.rs, draw.rs, pipelines.rs, textures.rs, gpu_ops.rs, composite.rs
```

**13 WGSL shader files:**

```
src/shaders/
├── bindings.wgsl, vertex.wgsl, primitives.wgsl, operations.wgsl
├── transforms.wgsl, modifiers.wgsl, noise.wgsl, rendering.wgsl
├── voxel_sampling.wgsl, pick.wgsl, brush.wgsl, blit.wgsl
└── composite_entry.wgsl
```

---

## 7. UI/Backend Boundary Contract

For ongoing UI migration safety rules, see `docs/ui_backend_boundary.md`.

This contract defines which app modules stay toolkit-agnostic (`src/app/backend_frame.rs`) and which are toolkit adapters (`src/app/egui_frontend.rs`, `src/app/frontend_bridge.rs`).

