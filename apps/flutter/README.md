# SDF Modeler Flutter Host

This app is the alternate UI host for the Rust backend. Flutter owns layout and input. Rust owns scene state, viewport rendering, and command execution through `src/app_bridge/`.

## Run

From `apps/flutter`:

```powershell
fvm flutter run -d windows
```

## Regenerate flutter_rust_bridge Code

Do not hand-edit generated bridge files.

1. Install the exact FRB code generator version once:

```powershell
cargo install flutter_rust_bridge_codegen --version 2.11.1 --locked
```

2. Regenerate the bridge from `apps/flutter`:

```powershell
.\scripts\frb_codegen.ps1
```

3. Keep the generator in watch mode while editing the Rust API if needed:

```powershell
.\scripts\frb_codegen.ps1 -Watch
```

The script verifies the installed codegen version before running and uses the repo-pinned Flutter SDK on `PATH`.

## Source Layout

- `rust/src/api/simple.rs`: Flutter-facing Rust API surface.
- `rust/src/frb_generated.rs`: generated Rust FRB glue.
- `lib/src/rust/`: generated Dart FRB glue.
- `lib/src/viewport/`: Flutter viewport host widgets and input plumbing.
- `../../src/app_bridge/`: toolkit-neutral Rust session, snapshot, render, and viewport command facade.

## Input Boundary

Viewport interaction stays command-based:

- orbit
- pan
- zoom
- select by viewport pick

Flutter sends viewport intents through FRB. Rust updates camera/selection state and renders the next frame.