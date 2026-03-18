# Source System Map

This rewrite should copy the product behavior of the current Rust project, not its implementation language.

## Current System, By Responsibility

| Current area | Role in the existing project | Rewrite target |
| --- | --- | --- |
| `src/graph/scene.rs`, `src/graph/history.rs`, `src/graph/voxel.rs` | Scene graph, semantic model state, undo/redo, voxel logic | `sdf_modeler_domain` plus `sdf_modeler_application` |
| `src/app/state.rs`, `src/app/mod.rs`, `src/keymap.rs`, `src/settings.rs` | App state, workflows, key bindings, settings | `sdf_modeler_application` and `sdf_modeler_infrastructure` |
| `src/app/backend_frame.rs`, `src/app_bridge/*` | UI/backend boundary, commands, snapshots, semantic workspace DTOs | `sdf_modeler_application` controller layer and DTO packages |
| `src/gpu/codegen.rs`, `src/gpu/*` | Shader/material code generation, GPU data preparation, render support | `sdf_modeler_rendering` |
| `src/sculpt.rs` | Sculpt interactions and brush behavior | `sdf_modeler_application` sessions plus domain tools |
| `src/export.rs`, `src/mesh_import.rs`, `src/io.rs` | Persistence, export, import | `sdf_modeler_infrastructure` with isolates for heavy jobs |
| `src/ui/*` | Desktop-specific widgets and editor presentation | `sdf_modeler_widgets` and the app shell |

## Behaviors That Must Survive The Rewrite

### 1. Semantic document model

The current app is not just a viewport with ad hoc tool state. It has a real scene graph, deterministic identifiers, undoable changes, and serializable document concepts. The pure Flutter rewrite must keep that as the source of truth.

### 2. Command-driven edits

The strongest reusable architectural idea in the current codebase is the command/snapshot boundary exposed by `app_bridge`. Recreate that boundary in Dart even though the whole app is now in one runtime.

### 3. Two-speed rendering invalidation

The current project distinguishes between lightweight view updates and heavier scene recomputation. Preserve that model. A Flutter renderer that blindly rebuilds everything each frame will regress performance.

### 4. Interaction sessions

Sculpting, camera movement, transform drags, and other editor gestures should be modeled as explicit sessions with begin/update/commit or cancel semantics. This keeps undo, preview rendering, and input routing coherent.

### 5. Workspace state as data

Panel visibility, selection context, tool mode, camera preset, and viewport toggles should live in explicit application state or DTOs. Do not bury this in widget-local state.

## Areas That Become Harder In Pure Flutter

- GPU-facing code generation and renderer resource lifetime now live in Dart instead of Rust.
- Large imports, export meshes, and heavy brush batches need isolate boundaries to avoid UI jank.
- Deterministic numeric behavior must be guarded with tests because the port changes language and numeric libraries.
- Desktop parity is riskier because the official Impeller story is less mature there than on mobile.

## Recommended Package Ownership

- `sdf_modeler_domain`
  Scene graph, value objects, math types, domain validation, history primitives.
- `sdf_modeler_application`
  Commands, reducers/controllers, interaction sessions, derived workspace state.
- `sdf_modeler_rendering`
  `flutter_gpu` renderer, scene extraction, shader/material mapping, draw scheduling.
- `sdf_modeler_infrastructure`
  File IO, serialization, import/export, settings persistence, isolates.
- `sdf_modeler_widgets`
  Editor shell, panels, inspectors, shortcuts, viewport widget wrappers.

## Port Order Guidance

Do not port by file count. Port by product backbone:

1. Document model and history.
2. Command layer and interaction sessions.
3. Viewport renderer spike.
4. Core modeling workflows.
5. Persistence and import/export.
6. Sculpt and advanced tools.
