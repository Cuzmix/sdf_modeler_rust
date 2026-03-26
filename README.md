# SDF Modeler

A real-time Signed Distance Function (SDF) 3D modeling application built in Rust. It renders SDF scenes on the GPU via raymarching, supports voxel sculpting, and exports meshes via marching cubes.

## Features

- **SDF Modeling** - Combine primitives with CSG operations and modifiers
- **GPU Raymarching** - Real-time rendering with lighting, shadows, AO, and post-processing
- **Voxel Sculpting** - Convert SDF subtrees into voxel layers and sculpt them interactively
- **Mesh Export** - Export OBJ, STL, PLY, glTF (`.glb`), and USD ASCII (`.usda`)
- **Scene Graph** - Presented-object hierarchy with undo/redo, isolation, and light linking
- **Native Slint UI** - Slint desktop shell with scene panel, viewport, inspector, render settings, import/export flows, and reference images

## Tech Stack

| Component | Library |
|-----------|---------|
| Language | Rust 2021 edition |
| GPU | wgpu 27.0.1 |
| UI | Slint 1.15.1 |
| Math | glam 0.29 |
| Serialization | serde + serde_json + bytemuck |
| Parallelism | rayon |
| File dialogs | rfd |

## Quick Start

### Prerequisites

- Rust stable toolchain (edition 2021)
- GPU drivers with DX12-capable hardware on Windows

### Build And Run

```bash
# Development build
cargo run

# Release build with debug symbols
cargo run --release
```

### Validation

```bash
cargo check
cargo clippy -- -D warnings
cargo test
cargo build
```

## Architecture

The application is organized into five layers:

```text
┌──────────────────────────────────────────────┐
│ Slint UI (`src/app/slint_ui/`)              │
├──────────────────────────────────────────────┤
│ Slint Host (`src/app/slint_frontend/`)      │
├──────────────────────────────────────────────┤
│ App Core (`src/app/`)                       │
├──────────────────────────────────────────────┤
│ Scene Graph (`src/graph/`)                  │
├──────────────────────────────────────────────┤
│ GPU + Viewport (`src/gpu/`, `src/viewport/`)│
└──────────────────────────────────────────────┘
```

Key patterns:

- **Redux-style structural actions** - UI emits `Action` values and `process_actions()` remains the structural mutation gate
- **Toolkit-neutral frame lifecycle** - `src/app/backend_frame.rs` stays free of Slint widget types
- **Presenter layer** - `src/app/frontend_models.rs` builds the shell snapshot consumed by the Slint host
- **Two-speed GPU sync** - topology changes trigger shader rebuilds, property changes reuse pipelines and upload new data
- **Runtime WGSL codegen** - the scene tree is compiled to WGSL at runtime

For current technical documentation, see:

- [docs/architecture.md](docs/architecture.md)
- [docs/ui_backend_boundary.md](docs/ui_backend_boundary.md)
- [docs/slint_frontend.md](docs/slint_frontend.md)

## Project Structure

```text
src/
├── app/                 # Shared app state, backend frame flow, Slint host and UI
├── gpu/                 # Buffers, camera, codegen, picking
├── graph/               # Scene graph, presented objects, voxel history
├── viewport/            # Viewport rendering and overlays
├── gizmo/               # Viewport gizmo interaction and overlay math
├── shaders/             # WGSL shader files
├── settings.rs          # Persistent render/app settings
├── export.rs            # Marching cubes and mesh format writers
├── sculpt.rs            # Brush tools and sculpt state
├── io.rs                # Project file save/load
├── native_wgpu.rs       # Native WGPU setup helpers
└── lib.rs               # Native entry point
```

The desktop app is Slint-native and no longer depends on `egui` or `eframe`.

## Contributing

- Follow the coding standards in [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md)
- One logical change per commit with descriptive messages
- All commits must pass:
  - `cargo check`
  - `cargo clippy -- -D warnings`
  - `cargo test`
  - `cargo build`
- Do not leave placeholders or partial implementations
- Performance and sculpt responsiveness are non-negotiable

## License

All rights reserved.
