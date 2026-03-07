---
name: sdf-architecture
description: Background knowledge about SDF Modeler architecture, GPU pipeline, raymarching, shader conventions, scene graph, codegen, action system, state decomposition, and rendering pipeline. Auto-triggered when working on GPU, shader, codegen, or rendering code.
user-invocable: false
---

# SDF Modeler Architecture Reference

## Scene Graph (`src/graph/scene.rs`)

Binary tree stored as `HashMap<NodeId, SceneNode>`. NodeId = u64 (monotonically incrementing).

**NodeData variants**: Primitive (leaf), Operation (binary CSG, has left+right), Sculpt (voxel grid, optional input), Transform (unary), Modifier (unary), Light (leaf).

**Topology hashing**:
- `structure_key()` → u64 hash of node connections, types, visibility. Changes trigger shader rebuild.
- `data_fingerprint()` → u64 hash of all property data. Changes trigger buffer upload only.

## Action System (Redux-inspired)

- `Action` enum (~60 variants) in `src/app/actions.rs`
- `ActionSink = Vec<Action>` — UI pushes actions, never mutates state directly
- `process_actions()` in `src/app/action_handler.rs` — single mutation point
- **Data-level edits** (sliders, colors, gizmo) → direct `&mut Scene` for zero-latency
- **Structural changes** (delete, create, reparent, bake) → push Action

## State Decomposition (`src/app/state.rs`)

`SdfApp` has 7 sub-structs:
- `doc` (DocumentState) — scene, camera, history, tools
- `gizmo` (GizmoContext) — mode, space, interaction
- `gpu` (GpuSyncState) — pipeline state, dirty flags, buffer offsets
- `async_state` (AsyncState) — background tasks, sculpt interaction
- `ui` (UiState) — dock layout, dialogs, selection
- `persistence` (PersistenceState) — file path, dirty flag
- `perf` (PerfState) — frame timings, profiler

Access: `self.doc.scene`, `self.gpu.buffer_dirty`, `self.ui.dock_state`

## Frame Update Loop (8 phases)

1. Frame setup (timing, camera animation, turntable)
2. Async polling (bake, export, import, sculpt pick)
3. GPU pipeline sync (structure_key → shader rebuild if changed)
4. Collect actions (keyboard + UI drawing)
5. `process_actions()` — apply all queued actions
6. Post-action cleanup (validate selection, submit sculpt pick)
7. GPU upload + dirty tracking (data_fingerprint → buffer upload)
8. Finalize (history snapshot, conditional repaint)

## GPU Pipeline

See [gpu-pipeline.md](gpu-pipeline.md) for detailed bind group layouts and struct definitions.

### Codegen (`src/gpu/codegen.rs`)

Runtime WGSL generation via post-order tree traversal:
- Each visible node emits a `let n{i} = vec4f(distance, material_id, blend, extra);` line
- Transform chain: ancestors' Transform + point-Modifier nodes composed innermost-out
- Two-phase optimization: cheap subtrees first, expensive (sculpt) subtrees wrapped in AABB conditional skip

**Key functions**: `generate_shader()`, `generate_pick_shader()`, `generate_composite_shader()`

### Shader Templates (`src/gpu/shader_templates.rs`)

Assembles WGSL from `include_str!()` of individual `.wgsl` files. Compile-time placeholders (`/*MARCH_MAX_STEPS*/`, etc.) replaced with `RenderConfig` values before shader compilation.

### Pipeline Rebuild Triggers

| Change | Detection | Action |
|--------|-----------|--------|
| Node add/delete/reconnect | `structure_key()` changed | Full shader recompile + pipeline rebuild (~10ms) |
| Property edit (slider/color) | `data_fingerprint()` changed | Buffer upload only (~1ms) |
| Render settings change | `settings_changed` flag | Placeholder re-substitution + rebuild |

## Raymarching & SDF (iq-inspired)

- Conservative step: `t += d * 0.9`, epsilon 0.0005, 96 max steps
- Enhanced sphere tracing: Keinert over-relaxation (omega=1.2), ~20% fewer steps
- Tetrahedron normals (4 SDF evals vs 6 central differences)
- Distance-adaptive normal epsilon: `clamp(0.001*t, 0.0005, 0.05)`
- Improved soft shadows (Aaltonen variant with `ph` tracking)
- Scene-level ray-AABB intersection for early-out
- Interactive quality mode: half steps, skip AO+shadows during camera drag
- Per-subtree bounding skip: expensive subtrees wrapped in AABB conditional

### Key iq Findings

- Binary search refinement is BAD (50% slower)
- GPU ternaries don't create real branches — don't "optimize" with step/mix
- sqrt() is nearly free on GPU — don't avoid it
- For SDF bounding volumes: smooth_k is the BVH expansion amount

## Voxel System

- `VoxelGrid` in `src/graph/voxel.rs`: resolution (96-320), bounds, flat f32 array
- Two modes: Total SDF (data = complete distances) vs Differential (data = displacement from base)
- Render shader: `texture_3d<f32>` + `textureSampleLevel` (hardware trilinear)
- Pick/brush shaders: `sdf_voxel_grid` storage buffer path
- Bind groups: @group(2) for voxel textures (R32Float + FLOAT32_FILTERABLE)
- Sparse serialization for file I/O (10-100x compression)

## Lighting

- Cook-Torrance PBR: GGX NDF, Smith geometric, Schlick Fresnel
- MAX_SCENE_LIGHTS = 8, nearest to camera active, u8 bitmask light linking
- LightType: Point, Spot, Directional, Ambient
- Ambient lights accumulate into SceneAmbient (not light array)
- Shading modes: Full, Solid, Clay, Normals, Matcap, StepHeatmap, CrossSection

## wgpu 22 API Notes

- eframe 0.29 bundles wgpu 22.1.0
- Device/queue from `cc.wgpu_render_state().device.clone()` (internally Arc'd)
- API: `ImageCopyTexture`/`ImageDataLayout` (NOT `TexelCopyTextureInfo`)
- `push_constant_ranges: &[]`, `multiview: None` in pipeline descriptors
- Custom device_descriptor: `max_storage_buffers_per_shader_stage = 4`, 128MB storage limit
- FLOAT32_FILTERABLE device feature required for R32Float texture sampling
