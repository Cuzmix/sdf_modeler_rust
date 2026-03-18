# Suggested Skills

## `scene-graph-change`

Use for:

- document model updates
- selection rules
- undo/redo changes
- serialization-affecting domain work

Checklist:

- history invariants covered
- deterministic ids preserved
- round-trip tests updated

## `renderer-change`

Use for:

- `flutter_gpu` pipeline work
- draw extraction changes
- camera and viewport updates

Checklist:

- dirty-path rules preserved
- no obvious per-frame churn added
- manual viewport verification recorded

## `sculpt-change`

Use for:

- brush behavior
- live preview
- dense-scene interaction updates

Checklist:

- responsiveness tested
- commit/cancel path verified
- undo/redo covered

## `workflow-change`

Use for:

- commands
- sessions
- shortcut routing
- outliner and inspector integration

Checklist:

- widget logic stays thin
- application-layer tests updated
- manual workflow script rerun

## `io-change`

Use for:

- save/load
- import/export
- settings
- autosave and recovery

Checklist:

- round-trip fixtures updated
- versioning impact reviewed
- isolate boundary checked for heavy work

## `verification`

Use for:

- release hardening
- regression sweeps
- manual verification scripts

Checklist:

- formatting and analysis clean
- tests and build clean
- manual high-risk flows executed

## `benchmark-and-regression`

Use for:

- performance investigations
- frame pacing issues
- memory and GPU lifecycle regressions

Checklist:

- before/after measurements captured
- primary target device class identified
- regression threshold documented
