# CLAUDE.md — SDF Modeler

## Project Overview

SDF Modeler is a real-time Signed Distance Field (SDF) 3D modeler written in Rust. It uses GPU ray-marching via WGSL shaders to render SDF scenes interactively, with support for CSG operations, sculpting, transforms, modifiers, and mesh export. It targets both native desktop (Linux/Windows/macOS) and WebAssembly (WebGPU).

## Build & Run

### Native

```bash
cargo build                # debug (opt-level=2 for usable perf)
cargo build --release      # release with debug symbols
cargo run                  # run native app
```

### WASM

```bash
# Requires wasm32-unknown-unknown target and wasm-bindgen-cli
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --out-dir web/pkg --target web --no-typescript \
    target/wasm32-unknown-unknown/release/sdf_modeler.wasm

# Serve with any HTTP server:
python -m http.server -d web 8080
# Open http://localhost:8080 in a WebGPU-capable browser (Chrome 113+)
```

The `web/build.sh` script automates the WASM build. It also runs `wasm-opt` if available.

### Checks

```bash
cargo check                # type-check without building
cargo clippy               # lint
```

There are no unit tests in this project currently.

## Architecture

### Module Structure

```
src/
├── main.rs                 # Native entry point (calls lib::run_native)
├── lib.rs                  # Module root, native + WASM entry points
├── compat.rs               # Platform abstractions (time, parallel iteration)
├── settings.rs             # App settings + render config (persisted as JSON)
├── io.rs                   # Project file save/load (.sdf JSON format)
├── export.rs               # Marching cubes mesh export (OBJ/STL/PLY/GLB/USDA)
├── sculpt.rs               # Brush system (add/carve/smooth/flatten/inflate/grab)
├── app/                    # Application state and main loop
│   ├── mod.rs              # SdfApp struct, eframe::App::update() main loop
│   ├── input.rs            # Keyboard shortcuts / input handling
│   ├── gpu_sync.rs         # Pipeline rebuild + GPU buffer sync
│   ├── sculpting.rs        # Sculpt mode interaction (pick, stroke, lazy brush)
│   ├── async_tasks.rs      # Async bake + export task management
│   └── ui_panels.rs        # Menu bar, status bar, help window, toasts
├── gpu/                    # GPU-side code
│   ├── mod.rs              # Module exports
│   ├── camera.rs           # Orbit camera + CameraUniform (GPU struct)
│   ├── codegen.rs          # Dynamic WGSL shader generation from scene graph
│   ├── buffers.rs          # GPU buffer building (scene nodes, voxel data)
│   ├── picking.rs          # GPU object picking via compute shader
│   └── shader_templates.rs # WGSL template strings and placeholder substitution
├── graph/                  # Scene graph data model
│   ├── mod.rs              # Module exports
│   ├── scene.rs            # Scene, SceneNode, NodeData, SdfPrimitive, CsgOp, etc.
│   ├── history.rs          # Undo/redo system
│   └── voxel.rs            # VoxelGrid (3D SDF storage) + CPU SDF evaluation
├── ui/                     # UI panels (egui-based)
│   ├── mod.rs              # Module exports
│   ├── dock.rs             # Dockable tab layout (egui_dock)
│   ├── viewport/           # 3D viewport rendering
│   │   ├── mod.rs          # ViewportResources, BrushGpuParams structs
│   │   ├── draw.rs         # Viewport draw callback
│   │   ├── pipelines.rs    # Render/compute pipeline creation
│   │   ├── textures.rs     # Voxel 3D texture management
│   │   ├── gpu_ops.rs      # GPU dispatch operations
│   │   └── composite.rs    # Composite volume cache (compute prepass)
│   ├── scene_tree.rs       # Scene hierarchy panel
│   ├── properties.rs       # Node properties inspector
│   ├── node_graph.rs       # Visual node graph editor
│   ├── gizmo.rs            # 3D transform gizmo (translate/rotate/scale)
│   ├── render_settings.rs  # Render settings UI panel
│   └── dev_settings.rs     # Developer/debug settings panel
└── shaders/                # WGSL shader files
    ├── bindings.wgsl       # Shared GPU buffer bindings
    ├── vertex.wgsl         # Fullscreen quad vertex shader
    ├── rendering.wgsl      # Ray-marching, lighting, shadows, AO
    ├── primitives.wgsl     # SDF primitive functions (sphere, box, etc.)
    ├── operations.wgsl     # CSG operations (union, subtraction, intersection)
    ├── transforms.wgsl     # Euler rotation, transform utilities
    ├── modifiers.wgsl      # Point-space modifiers (twist, bend, mirror, etc.)
    ├── brush.wgsl          # Sculpt brush compute shader
    ├── pick.wgsl           # GPU object picking compute shader
    ├── blit.wgsl           # Texture blit shader
    ├── composite_entry.wgsl# Composite volume compute entry point
    └── voxel_sampling.wgsl # Voxel grid SDF sampling functions
```

### Key Abstractions

- **`Scene`** (`graph/scene.rs`): HashMap-based scene graph. Nodes are identified by `NodeId` (u64). Contains `SceneNode`s with `NodeData` variants: `Primitive`, `Operation`, `Transform`, `Modifier`, `Sculpt`.
- **`SdfApp`** (`app/mod.rs`): Main application struct implementing `eframe::App`. Owns the scene, camera, GPU state, sculpt state, undo history, and all UI state.
- **`ViewportResources`** (`ui/viewport/mod.rs`): GPU resources stored in eframe's `callback_resources`. Holds render/compute pipelines, buffers, textures.
- **`VoxelGrid`** (`graph/voxel.rs`): 3D signed distance field stored as flat `Vec<f32>`. Two modes: total SDF (fill=999.0) or displacement (fill=0.0). Layout: `data[z * res * res + y * res + x]`.
- **Dynamic shader codegen** (`gpu/codegen.rs`): WGSL shaders are generated at runtime from the scene graph. Each node emits inline WGSL code. Shader is regenerated when the scene structure changes (tracked via `structure_key`).

### Data Flow

1. User edits scene graph via UI panels
2. Scene changes trigger `structure_key` or `data_fingerprint` change
3. `sync_gpu_pipeline()` detects structure changes → regenerates WGSL shaders → rebuilds GPU pipelines
4. `upload_scene_buffer()` uploads node data and voxel data to GPU
5. Viewport draws via ray-marching fullscreen quad + optional composite volume cache
6. Undo/redo tracked per-frame via `History`

### Platform Differences

The `compat.rs` module provides cross-platform abstractions:
- **Time**: `std::time::Instant` on native, `web_time::Instant` on WASM
- **Parallel iteration**: `rayon::into_par_iter()` on native, `into_iter()` on WASM (via `maybe_par_iter!` macro)
- **File I/O**: Native uses `rfd` file dialogs + `std::fs`; WASM uses browser download/localStorage
- **Settings**: Native persists to `settings.json` next to executable; WASM uses localStorage

Conditional compilation uses `#[cfg(not(target_arch = "wasm32"))]` and `#[cfg(target_arch = "wasm32")]` throughout.

## Key Conventions

### Code Style

- Rust 2021 edition
- Section separators use `// ---------------------------------------------------------------------------` comment blocks
- Module-level visibility: `pub(super)` for app-internal fields, `pub` for cross-module APIs
- `pub(crate)` for macro re-exports (e.g., `maybe_par_iter`)
- Enums use `Self::Variant` syntax in impl blocks
- Error handling: `Result<T, String>` for user-facing operations; `unwrap()`/`expect()` for invariants that should never fail
- No `#[test]` modules exist — the project relies on `cargo check` and `cargo clippy`

### Naming

- Types: PascalCase (`SdfApp`, `VoxelGrid`, `BrushMode`)
- Functions: snake_case (`generate_shader`, `apply_brush`)
- Constants: SCREAMING_SNAKE_CASE (`DEFAULT_BRUSH_RADIUS`, `TIMING_HISTORY_LEN`)
- GPU structs use `#[repr(C)]` with `bytemuck::Pod + Zeroable`
- Node indices in generated WGSL use `n{i}` for SDF results, `lp{i}` for local points, `tp{node}_{step}` for transform chain variables

### Serialization

- Project files: JSON via `serde` + `serde_json`, saved as `.sdf` or `.json`
- Settings: JSON persisted separately from project files
- VoxelGrid uses custom sparse serialization to reduce file sizes (only non-fill values stored)
- Version migration in `io.rs` (current version: 4, handles v2→v3 migration)

### GPU/Shader Conventions

- Shaders are WGSL (WebGPU Shading Language)
- Bind groups: group(0) = camera, group(1) = scene data + voxel buffer, group(2) = voxel textures or composite volume
- `scene_sdf(p: vec3f) -> vec2f`: returns `(distance, material_id)` — this is the core SDF function generated dynamically
- Scene nodes are uploaded as `SdfNodeGpu` structs in a storage buffer
- Sculpt voxel data goes through either storage buffers or 3D textures depending on the rendering path

### Performance Patterns

- Resolution scaling: lower render resolution during interaction, full resolution at rest (configurable via `interaction_render_scale` / `rest_render_scale`)
- Composite volume cache: optional compute prepass that evaluates SDF into a 3D texture for O(1) lookups during ray-marching
- Bounding sphere skip: expensive subtrees (those containing sculpt nodes) are wrapped in bounding sphere checks in generated WGSL
- Dirty flags: `buffer_dirty`, `structure_key`, `data_fingerprint` prevent redundant GPU work
- Repaint control: only requests repaints when interaction is active (sculpting, dragging, etc.)

## Dependencies

| Crate | Purpose |
|-------|---------|
| `eframe` | Application framework (egui + wgpu integration) |
| `egui_dock` | Dockable panel layout |
| `egui_node_graph2` | Visual node graph editor widget |
| `glam` | Math library (Vec3, Mat4, etc.) |
| `bytemuck` | Safe transmutes for GPU buffer data |
| `serde` / `serde_json` | Serialization for project files and settings |
| `log` | Logging facade |
| `rfd` | Native file dialogs (native only) |
| `env_logger` | Log output (native only) |
| `rayon` | Parallel iteration (native only) |
| `image` | Image processing (native only) |
| `web-time` | `Instant`/`Duration` polyfill (WASM only) |
| `wasm-bindgen` | Rust↔JS interop (WASM only) |
| `web-sys` / `js-sys` | Browser APIs (WASM only) |

## File Formats

- **Project files** (`.sdf` / `.json`): JSON with version field, scene graph, and camera state
- **Export formats**: OBJ, STL (binary), PLY (ASCII with vertex colors), GLB (binary glTF), USDA (USD ASCII)
- **Settings**: `settings.json` next to executable (native) or localStorage (WASM)
