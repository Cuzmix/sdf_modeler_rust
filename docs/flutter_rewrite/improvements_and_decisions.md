# Improvements And Decisions

## Preserve First

- Scene graph and history semantics.
- Deterministic command-driven editing.
- Two-speed invalidation between cheap viewport changes and heavy scene updates.
- Responsive sculpt and interaction feedback.
- Semantic workspace state that can be restored from data.

## Improve During The Rewrite

### Cleaner internal boundaries

The new project should make package boundaries harder to violate than the current app. Widgets, renderer internals, file IO, and domain logic should not bleed into each other.

### Typed command model

Define small explicit command types instead of allowing broad mutable state access from many directions.

### Session-oriented tools

All interactive editor gestures should use explicit session objects so previews, commits, cancellation, and undo boundaries are obvious.

### Better background job model

Use isolates for heavy import/export and preprocessing work instead of forcing the main UI isolate to absorb everything.

### Stronger persistence contracts

Keep file DTOs versioned and independently testable. Avoid coupling serialized form to widget trees or renderer structs.

### Performance instrumentation from day one

Add extraction, upload, and frame-time measurements before the renderer becomes complex.

## Explicit Decisions

- Pure Flutter only. No native core unless a later profiling pass proves it is necessary.
- `flutter_gpu` is the rendering target, but the renderer remains isolated behind a package boundary.
- Mobile-first v1 is the default planning scope.
- Desktop and web support are follow-on tracks, not day-one commitments.
- The rewrite should follow product workflows, not line-by-line source translation.

## Good Candidates For Deferral

- Old docking behavior that does not affect modeling correctness
- Rarely used panels
- Web support
- Desktop-specific polish
- Exotic import/export formats
- Non-essential visual chrome

## Improvements Worth Adding Even If They Are Not In The Old App

- Formal benchmark baselines
- Crash-safe autosave and recovery policy
- Better accessibility and keyboard audit coverage
- Document compatibility fixtures
- Explicit package-level ownership and review rules

## Decision Trigger Points

Re-evaluate architecture only if one of these happens:

- `flutter_gpu` cannot provide acceptable viewport performance on the primary target
- isolate overhead dominates import/export workflows
- desktop-first support becomes a hard requirement
- the renderer package becomes impossible to test in isolation
