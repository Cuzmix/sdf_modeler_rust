# egui_tiles Spike Notes

Date: 2026-03-22
Status: shelved

## Summary

We tested replacing the expert dock workspace backend with `egui_tiles`.

Conclusion:

- `egui_tiles` works for docked workspace layouts.
- `egui_tiles` does not map cleanly to the app's floating shell panel goals.
- `egui::Window` is still the correct tool for true floating shell panels.
- A hybrid model is the right fit if this is revisited later:
  - `egui::Window` for floating `Tool`, `Inspector`, and `Drawer`
  - `egui_tiles` for expert docked workspaces
  - explicit dock/undock handoff between the two

## What egui_tiles is good at

- tabbed dock layouts
- horizontal and vertical tiling
- grid layouts
- drag-and-drop docking
- behavior customization through `Behavior<T>`

## Main limitation we hit

`egui_tiles` is a tiling system, not a floating panel system.

That means:

- it can replace `egui_dock`
- it cannot replace `egui::Window`
- making it feel like detachable floating desktop panels requires custom glue code

## Migration findings

The dock backend swap itself is localized and feasible.

Main integration points:

- `src/ui/dock.rs`
- `src/app/egui_frontend.rs`
- `src/app/state.rs`
- `src/app/action_handler.rs`
- `src/app/ui_panels.rs`
- `src/ui/primary_shell.rs`

The app's public dock surface can stay stable by keeping the existing `Tab` enum and routing open/close/show behavior through dock helper functions.

## Key behavioral gap

The spike could toggle docked panels on and off, but reopening a panel restored it to a preferred anchor group, not necessarily the exact previous dock location.

If we revisit this, the correct direction is:

- preserve hidden tiles instead of removing and reinserting panes
- restore exact prior tile position from hidden/inactive tile state

## Effort estimate if revisited

Small: 0.5 to 1 day

- exact dock restore for hidden tabs in `egui_tiles`

Medium: 2 to 4 days

- production-ready hybrid model
- floating `egui::Window` shell panels
- explicit dock/undock actions
- launcher awareness
- layout reset and persistence polish

High: 1 to 2 weeks

- custom drag-out / drag-back experience that feels like one unified floating+docking system
- not recommended unless panel management becomes a core product differentiator

## Recommendation

Do not use `egui_tiles` as the default shell panel system.

If revisited later:

1. keep floating shell panels on `egui::Window`
2. use `egui_tiles` only for the expert workspace layer
3. store hidden docked tabs instead of deleting them
4. do not build a custom full docking framework unless the product genuinely needs it

## Validation from the spike

The spike compiled and passed Rust validation, but it was shelved because the interaction model did not match the expected floating panel behavior.
