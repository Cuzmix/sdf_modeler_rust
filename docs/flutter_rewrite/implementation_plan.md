# Implementation Plan

## Stage 0: Lock Scope And Budgets

Deliverables:

- Feature parity list for v1
- Primary platform decision
- Frame-time and interaction latency budgets
- Minimal file format commitments

Exit criteria:

- The team agrees what is in parity and what is deferred.
- One primary target platform is declared for local verification and CI.

## Stage 1: Bootstrap The Repo

Deliverables:

- Flutter monorepo or workspace
- The package split from `architecture_blueprint.md`
- Lint, formatting, and test pipelines
- App shell with editor frame and placeholder panels

Exit criteria:

- The repo builds cleanly.
- Package boundaries compile.
- No domain logic lives in widget files.

## Stage 2: Port The Domain Core

Deliverables:

- Scene graph
- Stable ids
- Selection model
- Undo/redo core
- Document open/save DTOs

Exit criteria:

- Domain tests cover scene edits and history invariants.
- A document can be created, mutated, serialized, and restored deterministically.

## Stage 3: Build The Application Layer

Deliverables:

- Command dispatcher
- Derived workspace state
- Tool mode state
- Input mapping and shortcut routing
- Session lifecycle primitives

Exit criteria:

- A widget tree can drive commands without owning business logic.
- Undo/redo works through the application layer only.

## Stage 4: Prove The `flutter_gpu` Viewport

Deliverables:

- Minimal renderer package
- Camera controls
- Stable frame loop
- Dirty-region handling
- Basic lit primitive rendering

Exit criteria:

- The viewport can render a stable test scene.
- Camera-only changes avoid full scene rebuilds.
- The app remains interactive under repeated viewport movement.

## Stage 5: Recreate Core Modeling Workflows

Deliverables:

- Create, delete, reorder, duplicate, and rename operations
- Selection workflows
- Transform workflows
- Basic material/look-dev controls
- Outliner and inspector integration

Exit criteria:

- Core editing matches the old product behavior at the command level.
- Widget tests and integration tests cover the main editor loop.

## Stage 6: Persistence, Import, And Export

Deliverables:

- Save/load
- Mesh import
- Mesh/export pipeline
- Settings persistence
- Recent files and recovery behavior

Exit criteria:

- File round-trips pass fixture tests.
- Long-running import/export jobs do not block the UI isolate.

## Stage 7: Sculpt And Advanced Interaction

Deliverables:

- Sculpt session model
- Brush preview rendering
- Batching or background preprocessing where profiling proves necessary
- Input smoothing and pressure rules if supported

Exit criteria:

- Sculpt feels responsive on the primary target hardware.
- Preview and commit behavior stay deterministic and undoable.

## Stage 8: Hardening And Performance

Deliverables:

- Benchmark suite
- Regression thresholds
- Crash recovery policy
- Accessibility and keyboard audits
- Memory and GPU lifecycle checks

Exit criteria:

- The app meets declared performance budgets on the primary device class.
- Manual verification scripts exist for every high-risk workflow.

## Stage 9: Additional Platform Support

Deliverables:

- Desktop feasibility report
- Web feasibility report
- Platform-specific input and packaging adaptations

Exit criteria:

- Extra platforms are added only after the renderer and interaction model are proven on the primary scope.

## Sequencing Rules

- Do not start by porting advanced panels or visual polish.
- Do not route rendering through `ui.Image` copies as a permanent architecture.
- Do not merge renderer internals into widget code for convenience.
- Do not attempt desktop and web parity in parallel with the first stable viewport.
