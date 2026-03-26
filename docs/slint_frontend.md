# Slint Frontend Structure

This document describes the native desktop frontend after the `egui` to Slint migration.

## Overview

The native app now has one UI host:

- Slint for windowing, widgets, and declarative layout
- shared `wgpu` 27 resources for the viewport renderer
- toolkit-neutral frame logic in `src/app/backend_frame.rs`

The frontend is intentionally split into four layers:

1. `src/app/slint_ui/`
   Declarative `.slint` components and typed UI contracts.
2. `src/app/slint_frontend/`
   Runtime host, snapshot binding, viewport texture lifetime, and callback installation.
3. `src/app/slint_bridge.rs`
   Input decoding between Slint events and toolkit-neutral frame/viewports DTOs.
4. `src/app/frontend_models.rs`
   Presenter models built from app/core state for the Slint shell.

## Runtime Host Modules

### `src/app/slint_frontend/mod.rs`

Desktop bootstrap for the Slint app:

- creates the shared native `wgpu` 27 device and queue
- selects the Slint backend with `BackendSelector::require_wgpu_27(...)`
- creates `SlintHostWindow`
- installs callbacks
- installs the rendering notifier
- drives the redraw/tick loop

This file should stay orchestration-only.

### `src/app/slint_frontend/host_state/`

Owns the mutable host runtime state.

- `mod.rs`
  Defines `NativeWgpuContext`, `TickOutcome`, and `SlintHostState`
- `tick.rs`
  Runs the host tick, applies shell snapshots, syncs runtime UI state, and updates redraw decisions
- `viewport_texture.rs`
  Owns the shared viewport texture import path for Slint and recreates it only when size/format changes

### `src/app/slint_frontend/bindings/`

Maps shared presenter models into Slint window properties.

- `mod.rs`
  Small façade used by the host tick
- `top_bar.rs`
  Builds `TopBarState`
- `scene_panel.rs`
  Builds `ScenePanelState`
- `inspector_panel.rs`
  Builds `InspectorPanelState`
- `utility_panel.rs`
  Builds `UtilityPanelState`
- `runtime_state.rs`
  Applies runtime-only state such as task/progress/import-dialog status
- `gizmo_overlay.rs`
  Converts viewport gizmo overlay paths into Slint path models

### `src/app/slint_frontend/callbacks/`

Decodes Slint domain callbacks into app actions or shared controller calls.

- `mod.rs`
  Installs callback groups
- `context.rs`
  Shared callback installation context
- `mutation.rs`
  Shared mutation helpers for queueing actions and forcing refreshes
- `scene_lookup.rs`
  Shared scene row lookup helpers
- `vector_axes.rs`
  Shared axis helpers for inspector edits
- `viewport.rs`
  High-frequency viewport event decoding
- `scene/`
  Top bar, scene toolbar, row actions, and scene text actions
- `inspector/`
  Transform, material, operation, sculpt, and light edit dispatch
- `utilities/`
  Render settings, reference images, and import dialog actions

### `src/app/slint_bridge.rs`

Converts Slint-side events into:

- `FrameInputSnapshot`
- `ViewportInputSnapshot`
- coarse app events such as undo/redo/frame-all

This file is the boundary between Slint pointer/scroll/button events and backend-friendly input snapshots.

## Slint Component Files

### Root Composition

- `src/app/slint_ui/slint_host_window.slint`
  Root window. Owns the top-level properties and the coarse callback boundary.
- `src/app/slint_ui/theme.slint`
  Shared colors, spacing, and visual tokens.

### Shell Components

- `src/app/slint_ui/top_bar.slint`
  Main desktop command bar.
- `src/app/slint_ui/scene_panel.slint`
  Scene list, selection summary, rename/filter text fields, create buttons, and history readout.
- `src/app/slint_ui/viewport_panel.slint`
  Displays the viewport image, emits pointer/scroll/double-click actions, and renders gizmo overlay paths.
- `src/app/slint_ui/right_sidebar.slint`
  Composes the inspector and utility panels.

### Inspector And Utility Components

- `src/app/slint_ui/inspector_panel.slint`
  Typed inspector editors for transform, material, operation, sculpt, and light properties.
- `src/app/slint_ui/reference_images_panel.slint`
  Reference image visibility/lock/plane/opacity/scale controls.
- `src/app/slint_ui/render_settings_panel.slint`
  Render toggles and export settings controls.
- `src/app/slint_ui/import_dialog_panel.slint`
  Mesh import confirmation and resolution controls.
- `src/app/slint_ui/nudge_row.slint`
  Reusable plus/minus row used by inspector and dialog editors.

## Slint View Model Files

The root file is now a barrel:

- `src/app/slint_ui/view_models.slint`

It re-exports the split contracts below.

### `src/app/slint_ui/view_models/common_views.slint`

Shared row/value structs used by multiple panels:

- `SceneRowView`
- `HistoryRowView`
- `ReferenceRowView`
- `ViewportPathView`

### `src/app/slint_ui/view_models/panel_states.slint`

Grouped UI state structs:

- `TopBarState`
- `ScenePanelState`
- `InspectorPanelState`
- `RenderSettingsState`
- `ImportDialogState`
- `UtilityPanelState`

### `src/app/slint_ui/view_models/ui_actions.slint`

Coarse-grained UI command enums:

- top bar actions
- scene toolbar, row, and text actions
- inspector edit kinds
- render settings actions
- reference image actions
- import dialog actions

### `src/app/slint_ui/view_models/viewport_actions.slint`

Typed high-frequency viewport actions:

- modifiers
- pointer button and phase
- pointer event payload
- scroll payload
- double-click payload

## Presenter Flow

The normal frame flow is:

1. `src/app/backend_frame.rs`
   Runs toolkit-neutral backend work.
2. `src/app/frontend_models.rs`
   Builds a `ShellSnapshot` from app/core state.
3. `src/app/slint_frontend/bindings/`
   Converts the snapshot into grouped Slint panel state.
4. `src/app/slint_ui/*.slint`
   Renders the current desktop shell.
5. `src/app/slint_frontend/callbacks/`
   Converts user intent back into `Action` values or shared controller calls.
6. `src/app/action_handler.rs`
   Applies structural changes through `process_actions()`.

## Extension Rules

When adding new Slint UI:

1. Add or extend presenter data in `src/app/frontend_models.rs`.
2. Add typed Slint state or action definitions in `src/app/slint_ui/view_models/`.
3. Add or update the `.slint` component under `src/app/slint_ui/`.
4. Bind new state in `src/app/slint_frontend/bindings/`.
5. Route new callbacks through `src/app/slint_frontend/callbacks/`.

Keep the callback surface domain-level. Avoid one callback per button or widget when a small action enum will do.
