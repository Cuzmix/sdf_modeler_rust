# Flutter Migration Guide

This document tracks the migration from egui to Flutter while preserving the Rust backend architecture.

## Current Baseline

- Flutter host app path: `apps/flutter`
- Flutter SDK pinning: `.fvmrc`
- Flutter SDK manager: FVM
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

## Rules During Migration

- Keep toolkit-neutral frame lifecycle logic in `src/app/backend_frame.rs`.
- Keep toolkit-specific code in frontend adapter modules.
- Keep structural scene mutations gated through `Action` + `process_actions()`.
- Do not introduce toolkit types into backend modules.

Reference: `docs/ui_backend_boundary.md`

## Next Milestones

1. Integrate `flutter_rust_bridge` into `apps/flutter`.
2. Add FRB smoke API (`ping/version`) from Rust to Flutter.
3. Expose first document/action API slice.
