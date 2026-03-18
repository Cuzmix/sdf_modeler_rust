# Testing And Quality

## Validation Gate

Every commit in the new repo should pass this sequence:

1. `dart format --output=none --set-exit-if-changed .`
2. `dart analyze`
3. `flutter test`
4. `flutter build <primary-target> --debug`
5. Manual verification for viewport, input, and file workflows affected by the change

`<primary-target>` should be fixed for the project and documented in the new repo. For a Windows-based mobile-first team, `apk` is the simplest default.

## Test Pyramid

### Domain tests

Highest volume. These should cover:

- scene graph mutations
- undo/redo invariants
- selection semantics
- serialization round-trips
- deterministic ids and ordering

### Application tests

Cover:

- command handling
- session lifecycles
- shortcut routing
- dirty-state calculation
- workspace snapshot derivation

### Rendering tests

Use a mix of:

- extraction tests from document state to render data
- shader or pipeline contract tests where possible
- screenshot or golden smoke tests on supported devices

Prefer deterministic CPU-side extraction assertions over brittle pixel goldens when a rendering change is not visual by nature.

### Widget tests

Use for:

- panel state
- toolbars
- inspector editors
- keyboard and focus routing
- viewport host interactions that do not require real GPU output

### Integration tests

Use for high-value end-to-end workflows:

- create scene, edit object, save, reload
- camera navigation plus selection
- import/export flows
- sculpt interaction smoke tests

### Benchmark tests

Track:

- cold start
- command latency
- viewport frame pacing under camera movement
- renderer extraction time
- large document load
- sculpt update latency

## Flutter And Dart Best Practices

- Keep domain logic in pure Dart packages with minimal Flutter dependencies.
- Keep widgets thin and compositional; avoid placing document mutation logic inside widget callbacks.
- Prefer immutable state snapshots or clearly versioned mutable state over wide shared mutation.
- Use sealed types and small explicit DTOs for commands, session events, and file messages.
- Push heavy CPU work to isolates only after defining stable message boundaries.
- Minimize allocation and object churn on per-frame rendering paths.
- Measure before introducing caching complexity.

## Rendering Quality Rules

- Separate CPU extraction from GPU submission.
- Do not rebuild pipelines or large immutable buffers every frame.
- Keep camera-only changes on the cheapest possible path.
- Add profiling hooks around extraction, upload, and draw phases early.
- Treat any frame-jank regression in sculpt or camera movement as a release blocker.

## Manual Verification Checklist

Run these after any meaningful renderer, input, or state-management change:

- open an existing document and verify layout restoration
- orbit, pan, and zoom continuously for at least 30 seconds
- select, transform, undo, and redo repeatedly
- save, close, reopen, and compare document state
- export and re-import a representative asset
- run one dense-scene interaction and one sculpt smoke test

## CI Recommendations

- Run formatting, analyze, and tests on every PR.
- Run at least one build target on every PR.
- Run integration tests on a nightly or protected-branch pipeline if device setup is expensive.
- Run benchmarks on a fixed hardware class before release candidates.

## Known Risk Areas To Test Hard

- numeric drift after the Rust-to-Dart port
- isolate message overhead on large payloads
- renderer resource leaks
- gesture edge cases when focus changes
- serialization compatibility across app versions
