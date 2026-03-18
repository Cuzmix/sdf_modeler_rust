# Flutter Rewrite Package

This folder documents a full rewrite of the current SDF Modeler codebase as a pure Flutter application using `flutter_gpu` for rendering.

## Planning Assumptions

- The new project is Dart and Flutter only. No separate native core and no host/backend split.
- Rendering is implemented in Flutter with `flutter_gpu`, but the document model, command system, and interaction logic remain separate from widgets.
- Heavy CPU work such as import, export, mesh generation, and long sculpt batches moves to Dart isolates.
- The safest v1 scope is mobile-first. This is an inference from official Flutter docs checked on 2026-03-17: `flutter_gpu` is available as a low-level Flutter API, while official Impeller availability is clearest on iOS and Android API 29+, with web still on Skia and macOS still behind a flag.

References:
- https://api.flutter.dev/flutter/flutter_gpu/
- https://docs.flutter.dev/perf/impeller

## Document Index

- `source_system_map.md`
  Maps the current Rust project into rewrite responsibilities and parity targets.
- `architecture_blueprint.md`
  Defines the recommended pure-Flutter package layout, renderer boundary, state flow, and platform scope.
- `implementation_plan.md`
  Staged plan to recreate the app without losing modeling velocity or rendering correctness.
- `testing_and_quality.md`
  Validation gates, benchmarks, manual checks, and Dart/Flutter engineering rules.
- `improvements_and_decisions.md`
  Preserve-first behavior, rewrite improvements, and explicit deferrals.
- `new_repo_templates/AGENTS.md`
  Drop-in agent guardrails for the new repo.
- `new_repo_templates/CLAUDE.md`
  Drop-in project guidance for future agent work.
- `new_repo_templates/SKILLS.md`
  Suggested skill inventory for specialized future tasks.

## Rewrite Goal

Recreate the current product behavior while improving package boundaries, testability, and renderer isolation.

Parity means preserving these outcomes first:

- Scene graph editing remains stable and undoable.
- Viewport interaction remains responsive under normal modeling and sculpt workloads.
- Document persistence, export, and import remain deterministic.
- Workspace state is semantic and restorable, not just widget-local.

## Non-Goals For V1

- Desktop-first parity.
- Web parity.
- Reproducing every legacy panel, mode, or docking behavior before the core document and viewport are stable.
- Recreating the old architecture where UI and renderer-specific state are tightly coupled.
