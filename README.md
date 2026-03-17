# SDF Modeler

A real-time Signed Distance Function (SDF) 3D modeling application built in Rust. Render SDF scenes on the GPU via raymarching, sculpt geometry with voxel grids, and export meshes via marching cubes.

## Features

- **SDF Modeling** — Combine primitives (sphere, box, cylinder, torus, etc.) with CSG operations (union, subtract, intersect) and modifiers (round, shell, elongate, twist, bend, etc.)
- **GPU Raymarching** — Real-time rendering with enhanced sphere tracing, PBR lighting, soft shadows, ambient occlusion, subsurface scattering, fog, and bloom
- **Sculpting** — Convert any SDF subtree to a voxel grid and sculpt with Add, Subtract, Smooth, Flatten, Pinch, and Grab brushes
- **Mesh Export** — Marching cubes extraction to OBJ, STL, PLY, glTF (.glb), and USD ASCII (.usda)
- **Scene Graph** — Binary tree with undo/redo, drag-drop reparenting, node isolation, and light linking
- **Dockable UI** — Configurable workspace with viewport, scene tree, properties, node graph, lights panel, render settings, and more

## Tech Stack

| Component | Library |
|-----------|---------|
| Language | Rust 2021 edition |
| GPU | wgpu 22.1.0 (via eframe 0.29) — WGSL shaders |
| UI | egui + egui_dock + egui_node_graph2 |
| Math | glam 0.29 |
| Serialization | serde + serde_json, bytemuck (GPU) |
| Parallelism | rayon (native) |
| File Dialogs | rfd (native) |

## Development Setup

### Primary Environment

- Windows 11 is the primary development platform for this repo.
- The native Rust app lives at the repo root.
- Flutter host work lives in `apps/flutter`.
- The repo pins Flutter `3.41.4` in `.fvmrc`.

### Native Toolchain

Install these before working on the native app or the Flutter Windows host:

- Rust stable via `rustup`
- Visual Studio 2022 with the `Desktop development with C++` workload
- Current GPU drivers with Vulkan, Metal, or DX12 support

Verify the Rust toolchain:

```powershell
rustup default stable
cargo --version
rustc --version
```

### Flutter Toolchain

The repo uses a repo-local Flutter SDK under `.fvm/flutter_sdk`. In a new shell, add it to `PATH` and verify the install:

```powershell
$env:Path = "$(Resolve-Path .\.fvm\flutter_sdk\bin);$env:Path"
flutter --version
flutter doctor
flutter config --enable-windows-desktop
```

If `.fvm/flutter_sdk` is missing on your machine, install the pinned SDK with `fvm` instead:

```powershell
# Requires Dart on PATH
dart pub global activate fvm
$env:Path = "$env:USERPROFILE\AppData\Local\Pub\Cache\bin;$env:Path"
fvm install 3.41.4
fvm use 3.41.4
fvm flutter doctor
fvm flutter config --enable-windows-desktop
```

After `fvm install`, you can keep using the repo-local `flutter` binary from `.fvm/flutter_sdk\bin`, or replace `flutter` with `fvm flutter` in the commands below.

### Build and Run

Native Rust app from the repo root:

```powershell
cargo run
cargo run --release
```

Flutter host from `apps/flutter`:

```powershell
Set-Location apps\flutter
flutter pub get
flutter run -d windows
```

If you edit the Flutter <-> Rust bridge API, install the pinned generator once and regenerate from `apps/flutter`:

```powershell
cargo install flutter_rust_bridge_codegen --version 2.11.1 --locked
.\scripts\frb_codegen.ps1
```

### Verification

Native validation from the repo root:

```powershell
cargo check
cargo clippy -- -D warnings
cargo test
cargo build
```

Flutter validation from `apps/flutter`:

```powershell
flutter analyze
flutter test
```

## Architecture

The application is organized into four layers:

```
┌─────────────────────────────────────┐
│  UI Layer (src/ui/)                 │  egui panels, viewport, gizmos
├─────────────────────────────────────┤
│  App Core (src/app/)               │  Update loop, actions, state
├─────────────────────────────────────┤
│  Scene Graph (src/graph/)          │  Nodes, voxels, history
├─────────────────────────────────────┤
│  GPU Pipeline (src/gpu/)           │  Codegen, buffers, camera, picking
└─────────────────────────────────────┘
```

**Data flow:** User input → Actions → `process_actions()` → Scene mutation → GPU sync → Shader codegen → Raymarching → Screen

Key patterns:
- **Redux-style actions** — UI pushes `Action` variants into an `ActionSink`; `process_actions()` is the single mutation point
- **Two-speed GPU sync** — Topology changes (`structure_key`) trigger shader rebuild; property changes (`data_fingerprint`) only upload buffers
- **Runtime WGSL codegen** — Scene tree is compiled to WGSL at runtime via post-order traversal
- **Clone-based undo** — Full scene snapshots (50 levels), simple and correct

For comprehensive technical documentation, see [docs/architecture.md](docs/architecture.md).

## Project Structure

```
src/
├── app/          # Update loop, state, actions, action handler
├── gpu/          # Buffers, camera, codegen, picking, shader templates
├── graph/        # Scene graph, voxel grid, undo/redo history
├── ui/           # Panels, viewport, gizmos, dialogs
│   └── viewport/ # 3D viewport rendering pipeline
├── shaders/      # WGSL shader files (13 files)
├── settings.rs   # Persistent render/app settings
├── export.rs     # Marching cubes + mesh format writers
├── sculpt.rs     # Brush tools and sculpt state
├── io.rs         # Project file save/load (JSON)
└── lib.rs        # Module declarations, entry points
```

57 Rust source files, 13 WGSL shaders, 380+ tests, zero `unsafe` blocks.

## Build Profiles

| Profile | Config | Use Case |
|---------|--------|----------|
| `dev` | `opt-level = 2` | Fast iteration with near-release performance |
| `release` | `debug = true` | Full optimization with symbols for profiling |

## Flutter Host Workflow

Flutter host work lives in `apps/flutter` and uses the repo-pinned Flutter SDK from `.fvmrc`.

Migration guardrails:
- Keep toolkit-neutral frame logic in `src/app/backend_frame.rs`.
- Keep toolkit adapters in frontend-specific modules.
- Do not introduce new egui features during migration (critical bug fixes only).
- Keep structural mutations gated through `Action` + `process_actions()`.

## Contributing

- Follow the coding standards in [CLAUDE.md](CLAUDE.md)
- One logical change per commit with descriptive messages
- All commits must pass: `cargo check`, `cargo clippy -- -D warnings`, `cargo test`, `cargo build`
- No placeholder implementations or TODO stubs — every function must be complete
- Performance is the #1 priority — never degrade runtime efficiency

## License

All rights reserved.

