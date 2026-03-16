# Egui Parity Audit

This audit captures the remaining egui-owned surface after PRD #25. It is the decommission checklist for PRD #27.

## Scope

- Goal: make the remaining egui-only surface explicit before staged deletion.
- Baseline: Flutter plus `src/app_bridge/` is now the active cross-toolkit path for document editing, scene hierarchy, basic properties, transform editing, viewport navigation, manipulation tooling, and document lifecycle.
- Constraint: do not delete egui files simply because Flutter has a matching button. Remove duplicate ownership first, then remove the egui adapter slice that only forwards to already-migrated backend behavior.

## Decommission-Ready Duplicate Ownership

These areas already have backend-neutral ownership through `src/app_bridge/` and active Flutter consumers. Their egui layers are now duplicate adapters, not the source of truth.

| Area | Backend-neutral owner | Flutter authority | egui duplicate surface | PRD #27 note |
| --- | --- | --- | --- | --- |
| Document lifecycle: new/open/save/save-as/recent/recovery | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | `apps/flutter/lib/src/session/document_session_panel.dart`, `apps/flutter/lib/app.dart` | `src/app/ui_panels.rs` File menu, `src/ui/recovery_dialog.rs` | Remove egui-only command wiring after confirming no remaining egui dialogs depend on it. |
| Export workflow and progress | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | `apps/flutter/lib/src/export/export_panel.dart`, `apps/flutter/lib/app.dart` | `src/ui/export_dialog.rs`, `src/ui/export_progress.rs`, `src/app/egui_frontend.rs` | Flutter now owns export settings, start/cancel, progress, and completion/error state through the bridge snapshot. |
| Mesh import and sculpt-convert entry flows | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs`, `src/app_bridge/workflows.rs` | `apps/flutter/lib/src/import/import_panel.dart`, `apps/flutter/lib/src/sculpt/sculpt_convert_panel.dart`, `apps/flutter/lib/app.dart` | `src/ui/import_dialog.rs`, `src/ui/sculpt_convert_dialog.rs`, `src/app/egui_frontend.rs` | Flutter now owns import/sculpt-convert dialog state, configuration, progress, and result messaging through the bridge snapshot. |
| Sculpt session controls and selected-sculpt workflow | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | `apps/flutter/lib/src/sculpt/sculpt_session_panel.dart`, `apps/flutter/lib/app.dart` | `src/ui/brush_settings.rs`, sculpt sections in `src/ui/properties.rs` | Flutter now owns sculpt session lifecycle, brush settings, symmetry, and resolution controls through backend snapshots and commands. Keep native viewport stroke handling isolated from panel ownership cleanup. |
| Undo/redo/duplicate/rename/delete/select/visibility/lock | `src/app_bridge/session.rs` | `apps/flutter/lib/app.dart`, `apps/flutter/lib/src/scene/scene_tree_panel.dart` | `src/app/ui_panels.rs`, `src/ui/scene_tree.rs`, parts of `src/ui/properties.rs` | Delete duplicate menu and panel handlers in small slices. |
| Scene tree snapshots and selection state | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | `apps/flutter/lib/src/scene/scene_tree_panel.dart` | `src/ui/scene_tree.rs` | Safe to decommission once any egui-only drag/drop or search behavior is either dropped or separately migrated. |
| Selected-node basics, primitive/material basics, transform inspector | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | Flutter property sections in `apps/flutter/lib/app.dart` | `src/ui/properties.rs` | Keep only egui-only fields that do not yet exist in backend-neutral snapshots. |
| Camera presets, frame-all, focus-selected, projection toggle | `src/app_bridge/session.rs` | Flutter command surfaces in `apps/flutter/lib/app.dart` | `src/app/ui_panels.rs`, viewport toolbar shortcuts | Remove duplicate egui command buttons without touching backend camera logic. |
| Application settings, bookmarks, and keymap editing | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs`, `src/keymap.rs` | `apps/flutter/lib/src/settings/settings_panel.dart`, `apps/flutter/lib/app.dart` | `src/ui/settings_window.rs`, menu hooks in `src/app/ui_panels.rs` | Flutter now owns persisted host settings, camera bookmarks, settings import/export/reset, and safe keymap editing through backend snapshots and commands. Keep profiler/history/statistics decisions isolated under PRD `#40`. |
| Viewport navigation, selection, manipulation mode/space/pivot, manipulation nudges | `src/app_bridge/session.rs`, native host event path | `apps/flutter/lib/src/viewport/viewport_surface.dart`, `apps/flutter/lib/src/viewport/viewport_tool_overlay.dart`, Windows texture host | egui input glue in `src/ui/viewport/draw.rs`, `src/ui/gizmo.rs`, `src/ui/quick_toolbar.rs` | Do not delete egui viewport rendering wholesale yet; first remove duplicate command ownership and keep any renderer-specific code that still serves the native egui app. |
| Viewport performance stats and interaction diagnostics | native Flutter host plus `src/app_bridge/renderer.rs` feedback | `apps/flutter/lib/src/viewport/viewport_feedback_overlay.dart` | `src/ui/profiler.rs`, viewport overlays in egui | The Flutter host now owns the migration-facing diagnostics surface. Keep egui profiler only if the native egui app still needs it locally. |

## Remaining Egui-Only Surface That Still Needs Migration

No additional egui-owned workflows remain in the active Flutter parity target.
Reference images were the last unresolved product decision and are now handled
explicitly in the PRD `#41` decision below.

## PRD #40 Diagnostics Decision

The Flutter parity target keeps only the diagnostics that already support the active touch-first host and migration workflow. The rest are explicitly dropped from Flutter parity scope.

| Decision | Surface | Current owner | Outcome |
| --- | --- | --- | --- |
| Retain | Undo/redo availability and history command state | `src/app_bridge/session.rs`, `src/app_bridge/dto.rs` | Already exposed through `history.can_undo` / `history.can_redo` and used by Flutter command surfaces. No separate Flutter history list is required. |
| Retain | Migration-facing viewport diagnostics: FPS, frame time, dropped frames, interaction phase, host error | Native Flutter host, `src/app_bridge/renderer.rs`, `apps/flutter/lib/src/viewport/viewport_feedback_overlay.dart` | Already implemented in the active Flutter host. This is the retained diagnostics subset for parity. |
| Drop | Labeled history inspection panel | `src/ui/history_panel.rs` | Keep as egui-local tooling only if the native egui app still needs it. It does not block Flutter parity or PRD `#30`. |
| Drop | Scene statistics window | `src/ui/scene_stats.rs` | Existing backend stats snapshots may remain available for debugging or future use, but no dedicated Flutter stats panel is part of the parity target. |
| Drop | Deep profiler views and debug-toggle-driven diagnostics windows | `src/ui/profiler.rs`, debug/profile hooks in egui menus | Treat as native egui developer tooling, not a Flutter product requirement. They do not block decommission of the Flutter parity queue. |

## PRD #41 Reference Images Decision

Reference images are explicitly dropped from the current Flutter parity target.
The existing implementation is an egui-local modeling aid built around
`ReferenceImageManager`, `egui::TextureHandle`, and egui-side file-dialog action
handling in `src/app/action_handler.rs`, so it does not yet behave like
backend-neutral document or session state.

| Decision | Surface | Current owner | Outcome |
| --- | --- | --- | --- |
| Drop | Reference image management and viewport overlays | `src/ui/reference_image.rs`, `src/app/action_handler.rs`, `src/ui/dock.rs`, `src/ui/viewport/draw.rs` | Keep as native egui-local tooling only. It is not a Flutter parity blocker and should be revisited only if a future PRD funds backend-owned viewport overlay state and cross-toolkit texture/file-picker plumbing. |

## Egui-Only Surface That Is Out Of The Current Flutter Parity Target

These features are not covered by the current migration PRD and should be treated as intentionally out of scope until a new PRD item says otherwise.

| Area | Current egui files | Current disposition |
| --- | --- | --- |
| History inspection panel | `src/ui/history_panel.rs` | Flutter keeps undo/redo availability only. The labeled history list is explicitly out of the parity target. |
| Scene statistics and profiler windows | `src/ui/scene_stats.rs`, `src/ui/profiler.rs` | Flutter keeps only the migration-facing viewport overlay diagnostics. Dedicated stats/profiler windows are explicitly out of scope. |
| Reference image management | `src/ui/reference_image.rs`, `src/ui/dock.rs`, `src/ui/viewport/draw.rs` | Keep as native egui-local tooling. The current implementation depends on egui textures and egui-owned file dialog flow, so it is explicitly out of the Flutter parity target. |
| Dock layout, tab management, and workspace presets | `src/ui/dock.rs`, `src/app/ui_panels.rs` | Flutter intentionally uses a touch-first shell instead of reproducing egui docking semantics. Keep only if the desktop native app continues to need them. |
| Command palette | `src/ui/command_palette.rs`, `src/app/egui_frontend.rs` | No Flutter parity target exists today. Replace with Flutter-specific shell affordances only if later needed. |
| Quick toolbar | `src/ui/quick_toolbar.rs` | The Flutter shell already owns its command presentation model. Treat egui quick access as toolkit-specific sugar. |
| Help window and keyboard shortcut cheat sheet | `src/ui/help.rs` | Useful, but not on the active migration path. Re-add later only if product value justifies it. |
| Node graph editor | `src/ui/node_graph.rs` | Not represented in the migration PRD. Keep out of PRD #27 deletions unless the entire egui app is being retired. |
| Light graph editor | `src/ui/light_graph.rs` | Same as node graph: not yet on the Flutter parity path and should not silently disappear without a product decision. |

## Duplicate Ownership To Remove First

These are the high-signal duplicate ownership points to cut before touching larger egui files:

- `src/app/ui_panels.rs`
  - File/Edit/View menu commands now overlap heavily with Flutter document, history, camera, and selection command surfaces.
- `src/ui/scene_tree.rs`
  - Scene hierarchy selection, rename, visibility, lock, and delete flows are duplicated by Flutter against backend snapshots.
- `src/ui/properties.rs`
  - Basic node, transform, primitive, material, sculpt session, advanced light, and render controls are duplicated. Settings sections are not.
- `src/ui/quick_toolbar.rs`
  - Manipulation and quick command affordances overlap with Flutter command strips and viewport tool overlay.

## PRD #27 Deletion Order

1. Remove duplicate egui menu and panel command wiring for document, history, camera, scene-tree, and basic property edits.
2. Keep intentionally retained egui-local tooling explicit instead of mixing it into duplicate-ownership deletions.
3. Decide whether out-of-scope egui-only features are intentionally retained for the native egui app or formally dropped.
4. Only after the remaining required workflows move behind backend-neutral modules should the egui adapter modules themselves be deleted.

## Decision Summary

- Flutter plus `src/app_bridge/` is already authoritative for the core editing loop.
- Render settings now round-trip through backend-neutral snapshots and Flutter surfaces instead of depending on egui ownership.
- Application settings, bookmarks, and keymap editing now round-trip through backend-neutral snapshots and Flutter surfaces instead of depending on the egui settings window.
- The retained Flutter diagnostics subset is now explicit: command availability plus the existing viewport overlay. The labeled history panel, scene stats window, and deep profiler views are out of parity scope.
- Reference image management is explicitly out of the Flutter parity target and remains native egui-local tooling until a future PRD says otherwise.
- There are no remaining unresolved parity decisions in the active migration queue.
- The biggest duplicate-ownership cleanup targets are `src/app/ui_panels.rs`, `src/ui/scene_tree.rs`, `src/ui/properties.rs`, and `src/ui/quick_toolbar.rs`.
