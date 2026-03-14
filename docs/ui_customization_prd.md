# UI Customization PRD

## Purpose

This PRD starts after the functional Flutter migration foundation is stable.
It is intentionally separate from the migration execution PRD so visual polish,
branding, theming, and presentation work do not change backend ownership or
parity sequencing.

## Preconditions

This PRD assumes the following migration foundations are already in place:

- Touch-first sizing and gesture contracts from PRD `#9` and `#10`
- Backend-owned document editing, scene-tree, and property flows
- Backend-owned viewport navigation and manipulation flows
- Stable Flutter shell composition for tablet-first and desktop-adaptive use

## Goals

- Define a coherent visual language for the Flutter shell.
- Establish tablet-first presentation polish that still scales cleanly to desktop.
- Specify theming, typography, color, spacing, motion, and surface treatments.
- Improve perceived product quality without changing backend ownership boundaries.
- Keep customization work modular so presentation changes can ship incrementally.

## Non-Goals

- Reopening backend ownership decisions already settled in `src/app_bridge/`
- Folding remaining egui parity gaps into this PRD
- Replacing the touch-first shell contract with desktop-only affordances
- Mixing styling work with export/import/sculpt/light parity migrations

## Design Principles

- Tablet first: controls must remain touch-usable before desktop density is considered.
- Intentional hierarchy: scene structure, properties, and viewport controls must read clearly at a glance.
- Motion with purpose: animations should reinforce state changes, not decorate them.
- Consistent semantics: colors, badges, warnings, and affordances should keep stable meanings across the shell.
- Progressive enhancement: desktop can add hover and keyboard efficiency, but not at the expense of tablet clarity.

## Workstreams

### 1. Visual language foundation

- Define a color system for surfaces, accents, warnings, selection, and disabled states.
- Define a type scale and font pairing for command surfaces, inspector content, and viewport overlays.
- Define spacing, radius, border, and elevation tokens for reusable shell components.

### 2. Shell presentation

- Refine the inspector, command sheets, desktop side panel, and bottom-sheet presentation.
- Improve section headers, grouping, and density transitions between tablet and desktop widths.
- Standardize empty states, loading states, and backend error presentation.

### 3. Viewport-adjacent polish

- Refine viewport overlays, tool strips, badges, diagnostics, and selection feedback.
- Specify motion rules for overlay appearance, command confirmation, and panel transitions.
- Keep performance-sensitive overlays lightweight enough for the native host path.

### 4. Brand and product identity

- Define icon direction and illustration style, if any.
- Define product-level visual signatures that distinguish the tool without harming clarity.
- Decide which visual choices are core defaults versus future theme customization points.

## Constraints

- The Flutter shell remains the customization target; do not reintroduce egui as a visual source of truth.
- Backend commands, validation, and scene semantics remain in Rust.
- Touch target minimums and gesture arbitration from the migration foundation remain mandatory.
- Presentation work must not materially regress viewport responsiveness.

## Deliverables

- A shared design token set for Flutter shell components
- Updated shell component styling guidance
- Motion guidance for panel and overlay transitions
- A rollout plan for applying the visual language without bundling unrelated functionality

## Rollout Order

1. Tokenize color, type, spacing, and surface primitives.
2. Restyle shell containers and inspector sections.
3. Restyle viewport overlays and command surfaces.
4. Apply brand-level polish once functional readability remains strong.

## Acceptance Criteria

- The customization plan stays separate from `plans/prd.json`.
- The plan explicitly builds on the touch-first shell contract instead of replacing it.
- Styling tasks are grouped so they can be implemented incrementally without mixing in functional migration scope.
- The resulting UI direction is clear enough to guide future implementation slices.
