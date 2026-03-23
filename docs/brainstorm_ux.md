# Sculpt-First Touch UI

## Rev 3 Direction

Rev 2 keeps the viewport-first shell and tightens the interaction model:

- pen edits
- touch navigates
- the 3D view stays dominant
- selection, measurement, and sculpt brushes live in one unified interaction rail

This remains a shell and workflow refactor, not a document-management redesign.

## Product Goals

- Keep the viewport as the persistent primary surface.
- Make modeling and sculpting mode switches feel immediate instead of panel-driven.
- Remove routine dependence on node-graph wiring for common workflows.
- Preserve expert graph workflows through `egui_dock` without exposing them as the default path.
- Use default `egui::Window` behavior for floating shell panels and keep `egui_dock` as the explicit expert docking path.

## Interaction Model

The main shell uses one exclusive interaction rail:

- `Select`
- `Measure`
- `Add`
- `Carve`
- `Smooth`
- `Flatten`
- `Inflate`
- `Grab`

Implementation rule:

- `Select` and `Measure` are top-level interaction modes.
- Sculpt brushes remain the `BrushMode` domain.
- `Distance` is not a brush and not an interaction mode. It stays a separate overlay toggle.

Expected behavior:

- `Select`
  - selection active
  - sculpt inactive
- `Measure`
  - selection active
  - measurement enabled
  - sculpt inactive
- `Sculpt(brush)`
  - enter sculpt immediately when a sculpt node is available
  - otherwise open the existing convert flow
  - preserve the chosen brush as the interaction preference

Keyboard shortcuts remain accelerators, but they should route through the same semantic interaction actions as the shell UI.

## Task Deck

The primary action surface is a floating `Task Deck`.

For continuity the user-facing label may still remain `Tool Panel`, but its role is task-oriented rather than generic tooling.

Deck states:

- `Select / Compose`
- `Measure`
- `Sculpt`

Top rail:

- `Select`
- `Measure`
- sculpt brushes
- `Distance` toggle

`Sculpt` deck layout:

- brush rail
- hot controls for `Radius`, `Strength`, `Falloff`
- `Brush Advanced` foldout for lower-frequency brush tuning
- `Modeling Commands` foldout for create/boolean/wrap/scene access

`Select / Compose` keeps the current create/wrap/scene actions in the same surface.

`Measure` keeps measurement actions together and avoids showing sculpt-only tuning.

Below the rail or foldouts, the deck contains collapsible command sections:

- `Create`
- `Guided Boolean`
- `Wrap Selection`
- `Scene`

This replaces the old split between choosing a sculpt action in one panel and adjusting it in another.

## Subject Inspector

The secondary floating panel becomes a context-only `Subject Inspector`.

Tabs:

- `Selection`
- `Material`
- `Light`
- `Node`

Default tab behavior:

- select or measure interaction -> `Selection`
- light selection -> `Light`
- material-capable selection -> `Material`

Inspector responsibilities:

- edit selection and transform properties
- show measurement status and clear/reset actions while `Measure` is active
- expose material, light, and node-specific properties as needed
- stay context-only during sculpt instead of owning brush tuning

## Launcher And Drawer

The bottom launcher strip stays anchored even though the content windows float.

Launcher entries:

- `Items`
- `History`
- `Reference`
- `Advanced`
- `Tool`
- `Inspector`
- `Reset Layout`

The selected launcher entry opens a floating content window near the bottom of the viewport.

Launcher behavior is dock-aware:

- if `Tool`, `Inspector`, or the bottom utility panel is floating, its launcher entry reflects that state
- if that panel is docked into `egui_dock`, the same launcher entry still reflects it as active
- toggling an active launcher entry removes the panel from whichever presentation is currently active, floating or docked
- switching `Items`, `History`, `Reference`, or `Advanced` while the drawer is docked retargets the docked drawer instead of forcing it back to floating

`Advanced` remains the expert path and continues to expose:

- node graph
- expert docking controls
- advanced render and scene controls

## Floating Shell Windows

Rev 3 uses plain `egui::Window` behavior for the shell:

- draggable
- resizable
- collapsible
- closable
- placement persisted by stable egui window IDs

Shell state persists:

- `open`
- active inspector tab
- active bottom utility tab
- `layout_revision`

`Reset Layout` must:

- restore default shell visibility
- restore default launcher and inspector tabs
- bump `layout_revision` so egui forgets old floating window placement
- reset the dock workspace to the primary shell layout

Pushback:

- do not build a second custom docking system
- do not pretend default egui windows dock to each other

`egui_dock` remains the only real docking system. Floating shell windows do not dock to each other directly.

All shell panels should be dockable through explicit handoff:

- `Tool Panel`
- `Inspector Panel`
- bottom utility `Drawer Panel`

Current handoff behavior:

- clicking `Dock` on a floating shell panel creates a detached `egui_dock` window at the same screen rect
- that detached window becomes the dock target for other tabs
- launcher toggles stay tab-scoped, so removing `Tool` only removes the `Tool` tab even if the detached dock window also contains `Inspector`
- docked shell tabs expose `Undock` and `Reset Layout`

## Node Graph Positioning

The node graph remains important for power users, but it is not the default workflow surface.

Phase 1 expectation:

- users can create primitives, wrap selections, enter sculpt, and add boolean operands without wiring graph edges by hand
- low-level graph connect and disconnect actions stay available for expert workflows only

If users still need frequent manual graph edits for common work, the scene-intent layer is insufficient and should be improved instead of being re-exposed in the main shell.

## Scene-Intent Rules

Phase 1 auto-build behavior:

- `Create Primitive`
  - with no selection, create a new top-level node and select it
- `Insert Transform`
  - wrap the current target
  - never leave a loose helper node
- `Insert Modifier`
  - wrap the current target
  - never leave a loose helper node
- `Enter Sculpt`
  - on a sculpt node, activate sculpt immediately
  - on a child under a sculpt node, activate the sculpt parent
  - on a non-sculpt node, open the convert flow
- `Boolean Add`
  - use the current selection as the base
  - auto-create the second operand
  - auto-build the operation node
  - keep the graph valid without manual wiring

## Input Constraints

- stylus is the editing device
- touch is the navigation and shell device
- existing multi-touch camera support remains
- existing pressure support remains
- sculpt responsiveness internals must not regress

Tablet-visible replacements are required for desktop-only affordances such as:

- right-click-only paths
- middle-click-only paths
- modifier-key-only brush adjustments
- keyboard-dependent modal sizing flows

## Implementation Notes

- Keep toolkit-agnostic frame logic in `src/app/backend_frame.rs`.
- Keep egui-specific shell drawing in egui UI modules.
- Keep `process_actions()` as the structural mutation gate.
- UI should emit semantic actions instead of directly wiring graph edges.
- Keep low-level graph wiring actions for advanced workflows only.
- Default startup layout should open directly into the viewport-first shell.
- The shell should prefer intent-driven commands and default egui floating windows over permanent dense panel chrome.

## Validation

Required automated validation order:

1. `cargo check`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. `cargo build`

Required manual checks:

1. The viewport remains dominant in portrait and landscape tablet layouts.
2. A user can switch between `Select`, `Measure`, and sculpt brushes from one panel.
3. `Distance` can be toggled without changing the active interaction mode.
4. A user can sculpt, navigate, add a primitive, add a boolean, and insert a modifier without opening the node graph.
5. Floating `Tool` and `Inspector` panels behave like standard egui windows.
6. Touch navigation does not accidentally begin sculpting.
7. Stylus pressure sculpting still works.
8. The advanced path still exposes expert graph workflows and docking.
