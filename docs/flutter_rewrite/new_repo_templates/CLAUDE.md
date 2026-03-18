# Project Guidance

## Mission

Build a pure Flutter SDF modeler with a `flutter_gpu` viewport, strong editor workflows, and strict separation between domain state, application logic, rendering, infrastructure, and widgets.

## Architecture Rules

- `sdf_modeler_domain` owns scene graph, history primitives, math helpers, and validation.
- `sdf_modeler_application` owns commands, sessions, derived workspace state, and invalidation policy.
- `sdf_modeler_rendering` owns render extraction, GPU resources, shaders, and draw scheduling.
- `sdf_modeler_infrastructure` owns persistence, import/export, settings, and isolates.
- `sdf_modeler_widgets` owns presentation only.

## Non-Negotiable Engineering Rules

- Do not let widgets mutate the document directly.
- Do not let renderer types leak into file DTOs.
- Do not serialize widget state.
- Do not put long-running CPU work on the UI isolate.
- Do not collapse package boundaries for convenience.

## State And Interaction Rules

- All meaningful edits must flow through commands or explicit session updates.
- Every interactive tool should have clear start, update, commit, and cancel behavior.
- Undo and redo must stay deterministic.
- Renderer recreation must not threaten document integrity.

## Performance Rules

- Protect camera motion and sculpt responsiveness first.
- Track extraction time, upload time, and frame pacing.
- Avoid rebuilding immutable GPU resources on every frame.
- Introduce caching only after measurement.

## Quality Rules

Before merging, expect:

1. formatting clean
2. analysis clean
3. tests passing
4. primary-target debug build passing
5. manual verification for affected workflows

## Review Priorities

When reviewing changes, look for:

- state ownership violations
- regressions in undo/redo or session boundaries
- renderer invalidation mistakes
- hidden isolate or serialization costs
- missing tests around high-risk workflows
