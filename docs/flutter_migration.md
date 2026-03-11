# Flutter Migration Guide

This document tracks migration from egui to Flutter while preserving the Rust backend architecture.

## Current Baseline

- Flutter host app path: `apps/flutter`
- Flutter SDK pinning: `.fvmrc`
- Flutter SDK manager: FVM
- FRB bridge crate path: `apps/flutter/rust`
- FRB codegen config: `apps/flutter/flutter_rust_bridge.yaml`
- Rust UI status: egui remains active, but new egui features are frozen (critical bug fixes only)

## Local Setup

```bash
# one-time
dart pub global activate fvm

# verify pinned SDK
fvm flutter --version

# install Flutter app deps
cd apps/flutter
fvm flutter pub get
```

## FRB Workflow

Regenerate bridge code after any API change in `apps/flutter/rust/src/api/*`:

```powershell
./apps/flutter/scripts/frb_codegen.ps1
```

Watch mode:

```powershell
./apps/flutter/scripts/frb_codegen.ps1 -Watch
```

## Rules During Migration

- Keep toolkit-neutral frame lifecycle logic in `src/app/backend_frame.rs`.
- Keep toolkit-specific code in frontend adapter modules.
- Keep structural scene mutations gated through `Action` + `process_actions()`.
- Do not introduce toolkit types into backend modules.

Reference: `docs/ui_backend_boundary.md`

## Next Milestones

1. Expand FRB API beyond smoke checks to scene/session actions.
2. Add Flutter-side action dispatch and state snapshot rendering.
3. Introduce viewport texture interop while keeping Rust renderer ownership.
