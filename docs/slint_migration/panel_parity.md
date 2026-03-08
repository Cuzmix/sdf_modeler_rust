# Slint Panel Parity Checklist

This checklist tracks non-graph panel migration parity for PRD #75.

## Scene Tree
- [x] Read-only hierarchical list from AppCore scene snapshot data
- [x] Search/filter interaction
- [x] Rename workflow (selected-node rename command)
- [x] Visibility toggle workflow (selected-node toggle)
- [x] Delete/focus selected workflow

## History
- [x] Read-only undo/redo stack summary
- [ ] Undo/redo controls in Slint panel
- [ ] Entry selection/jump workflow (if retained)

## Lights
- [x] Read-only active/total light summary with per-light rows
- [x] Create light actions
- [ ] Light property editing (type/intensity/range/color)
- [ ] Solo/active state controls

## Render Settings
- [x] Read-only render config summary (shading, scale, shadows/AO/fog/bloom)
- [ ] Editable controls with command dispatch
- [ ] Presets and reset actions

## Scene Stats
- [x] Read-only live stats (counts, complexity, voxel memory)
- [ ] Performance indicators with thresholds and color coding

## Properties
- [x] Read-only selected-node summary per node type
- [ ] Editable node properties via commands
- [ ] Multi-select property workflows

## Notes
- Current implementation lives under `src/ui_slint/panels`.
- Panel rendering is currently wired to Slint shell mode (`SDF_SLINT_HOST_MODE=shell`).
- `winit_host` remains viewport-first and will consume the same panel read-model in follow-up slices.

