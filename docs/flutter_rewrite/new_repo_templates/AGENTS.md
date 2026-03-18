# Flutter SDF Modeler Agent Rules

This repository enforces the workflow and quality rules in `CLAUDE.md`.

## Mandatory Validation Before Commit

Every commit must pass, in this order:

1. `dart format --output=none --set-exit-if-changed .`
2. `dart analyze`
3. `flutter test`
4. `flutter build <primary-target> --debug`
5. Manual verification for viewport, input, or file-workflow changes

Set `<primary-target>` in the repo README and CI configuration.

## Scope And Architecture

- Keep domain logic out of widget files.
- Keep `flutter_gpu` code isolated from general application state.
- Prefer explicit names and small focused types over broad utility classes.
- Do not add placeholder implementations.
- Preserve clear package ownership.

## Performance

- Performance must not regress.
- Avoid per-frame allocation churn in renderer code.
- Preserve cheap paths for camera-only and preview-only updates.

## Workflow

- One logical change per commit.
- Keep commits focused and descriptive.
- Update docs and validation assets with code changes when behavior shifts.

## Source Of Truth

For architecture and workflow context, read:

- `CLAUDE.md`
- `docs/architecture.md`
- `docs/testing_and_quality.md`
