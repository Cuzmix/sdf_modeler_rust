# Professional UX Overhaul Batch 1 PRD

## Purpose

This document turns the broader UX vision in
`docs/professional_ux_overhaul_prd.md` into a concrete first execution batch.

Batch 1 is the point where SDF Modeler should stop feeling like a general app
shell around a viewport and start feeling like a deliberate professional tool.

This batch is intentionally focused on:

- shell structure
- contextual workflow
- viewport-first interaction
- live SDF blockout ergonomics
- sculpt workspace identity
- the app's unique live-to-sculpt hybrid story

It is not the full overhaul. It is the first buildable wave.

## Relationship To Other Planning Docs

- `plans/prd.json`: active migration/stabilization execution
- `docs/ui_customization_prd.md`: visual language and polish
- `docs/product_enhancements_prd.md`: broader enhancement roadmap
- `docs/professional_ux_overhaul_prd.md`: long-range UX vision

This batch PRD should stay separate from `plans/prd.json` unless and until the
team explicitly promotes one or more slices into the active execution queue.

## Preconditions

Before Batch 1 starts in earnest:

- The Flutter shell remains the primary UI direction.
- Backend-owned selection, scene, viewport, sculpt, and workflow commands remain in Rust.
- The unresolved Windows AXTree/accessibility issue is either fixed or sufficiently narrowed
  that it will not destabilize shell work.
- The existing Flutter shell token and theme foundation remains available as the base styling layer.

## Batch 1 Objective

By the end of Batch 1, the app should feel:

- viewport-first
- selection-driven
- professional under touch, pen, mouse, and keyboard
- clearly split into blockout, sculpt, lookdev, and review mental modes
- visibly unique in its live-SDF-to-sculpt workflow

Users should be able to:

- start shapes faster
- discover the next likely action without hunting through panels
- stay in the canvas longer
- understand whether an object is live, hybrid, or sculpt-baked
- move from blockout into sculpt with much less friction

## Batch 1 Non-Goals

This batch should not try to finish every future enhancement.

Out of scope for Batch 1:

- full reference image workflow
- full measurement and cross-section toolkit
- full material preset library
- full export preset and batch export system
- complete workspace preset system
- advanced node graph parity decisions
- major backend format or renderer rewrites
- broad stylistic polish beyond what is needed to support the new structure

Those should remain later slices unless a dependency forces a small enabling change.

## Product Risks Batch 1 Must Address

### 1. Panel-first feel

The shell currently risks feeling like a desktop app with a viewport inside it.
Batch 1 must reverse that.

### 2. Weak selection-to-action mapping

Too much thinking is still pushed onto the user.
Batch 1 must make selection drive the next actions.

### 3. Sculpt mode does not yet feel distinct enough

Sculpting needs a stronger identity and command surface.

### 4. The unique hybrid workflow is not yet legible

The difference between live SDF, hybrid, and sculpt-baked states should become
obvious and useful.

### 5. Too much routine interaction still depends on inspector navigation

Common operations should move closer to the viewport and current selection.

## Guiding Principles For Batch 1

- Do not hide the viewport behind more chrome.
- Do not recreate a generic CAD layout or a generic tablet art app layout.
- Prefer contextual UI over permanent UI.
- Keep backend ownership boundaries intact.
- Make routine actions easier before adding more total features.
- Favor professional clarity over novelty.

## Batch 1 User-Facing Outcomes

When Batch 1 is done, a first-time user should immediately see:

- a clear primary workspace
- a strong current mode
- a visible selected object state
- obvious next actions
- a clear path from "make shape" to "sculpt it"

A returning expert should feel:

- fewer taps to common actions
- less inspector scrolling
- stronger direct manipulation
- better canvas focus
- a more intentional tool identity

## Core Surfaces Touched In Batch 1

- top command bar
- left tool rail
- adaptive context shelf
- scene/layers drawer
- inspector role and grouping
- viewport HUD and confirmation surfaces
- sculpt workspace shell
- command search and quick access surfaces

## Ordered Execution Slices

## Slice B1-0. Shell Stability Gate

Priority: P0

Purpose:

- ensure shell work is not layered on top of unresolved viewport accessibility churn

Scope:

- close or narrow the AXTree issue
- verify decorative overlays and modal layers behave safely in semantics
- define a manual shell regression checklist for viewport, overlays, sheets, and modals

Required outcome:

- shell work can proceed without known unstable semantics churn being ignored

Why it belongs in Batch 1:

- a professional-feeling app cannot rest on an unstable shell foundation

## Slice B1-1. Viewport-First Shell Frame

Priority: P0

Purpose:

- establish the new top-level information architecture

Scope:

- top command bar
- left tool rail
- viewport-first central layout
- clear home for scene drawer and inspector
- workspace switcher for Blockout, Sculpt, Lookdev, and Review
- command/search entry point

Must deliver:

- the viewport is visually and behaviorally the center of the app
- the app no longer reads like a stack of unrelated panels
- mode/workspace is always legible

Out of scope:

- full visual polish pass
- detailed contextual logic
- advanced per-workspace custom behavior

## Slice B1-2. Selection Context Engine

Priority: P0

Purpose:

- make selection drive the UI

Scope:

- define a backend-neutral selection context model that tells Flutter:
  - selected item type
  - active workspace
  - recommended next actions
  - tool-specific quick parameters
  - state flags such as live/hybrid/sculpt
- render that model into the adaptive context shelf

Must deliver:

- a selected primitive, boolean, sculpt, transform, or light each produce visibly different next actions
- the user can reach common next steps without opening the deep inspector

Implementation constraints:

- keep business rules in Rust where possible
- Flutter should render recommendations, not invent product semantics ad hoc

## Slice B1-3. Scene / Layers Drawer Refresh

Priority: P1

Purpose:

- make scene management feel compact, visual, and fast

Scope:

- redesign the current scene tree surface into a scene/layers drawer
- inline controls for visibility, lock, solo, isolate
- clearer node-type badges
- active branch emphasis
- stronger drag/reparent affordances
- search and filter

Should deliver:

- object management feels closer to a modern sculpt/modeling app than a raw tree widget

Stretch if low-risk:

- saved selection sets
- simple color tags

Not yet required:

- thumbnails for every node type
- fully generalized layer system for every workflow

## Slice B1-4. Live Blockout Workflow

Priority: P0

Purpose:

- make SDF blockout immediate and signature-worthy

Scope:

- drag-out primitive creation in the viewport
- ghost preview before commit
- transient dimension / size feedback
- lightweight accept/cancel / keep-live strip
- fast boolean intent during placement where feasible
- quick duplicate/mirror/repeat/subtract actions near selection

Must deliver:

- creating a primitive no longer feels like a form-fill action
- blockout feels fast enough to invite experimentation

Uniqueness requirement:

- the live SDF nature of the created object should remain obvious after placement

## Slice B1-5. Hybrid Status And Branch-To-Sculpt Flow

Priority: P0

Purpose:

- surface the app's unique live-to-sculpt hybrid identity

Scope:

- visible status for live SDF, hybrid, and sculpt-baked objects
- branch-to-sculpt action exposed in the contextual workflow
- preserve-source behavior made explicit where applicable
- clearer convert/validate language

Must deliver:

- users understand whether they are editing a live procedural object or a sculpt result
- moving from blockout to sculpt feels like a designed workflow, not a technical conversion

Professional outcome:

- this becomes one of the app's clearest differentiators

## Slice B1-6. Sculpt Workspace Shell

Priority: P0

Purpose:

- make sculpting feel like a first-class professional mode

Scope:

- sculpt workspace state in the shell
- sculpt-focused tool rail behavior
- persistent brush HUD
- top utility strip for remesh/resolution/symmetry
- reduced unrelated chrome while sculpting
- stronger mode-specific viewport feedback

Should deliver:

- users can stay in sculpt mode for long sessions without interface fatigue
- the interface reads differently in Sculpt than in Blockout without feeling like a new app

Stretch if low-risk:

- recent brush row
- favorite brush placeholder

Not yet required:

- complete brush preset library
- final sculpt layer feature depth

## Slice B1-7. Command Search And Quick Access

Priority: P1

Purpose:

- make the app feel expert-friendly and professional

Scope:

- searchable command surface
- recent commands
- favorites
- workspace-aware results
- quick access entry from both touch and desktop paths

Must deliver:

- the app has a serious command surface rather than scattered action entry points

Stretch if low-risk:

- begin radial menu infrastructure in the same slice if implementation overlap is high

## Slice B1-8. Radial Menu And Canvas Quick Actions

Priority: P1

Purpose:

- support touch and pen speed without cluttering the shell

Scope:

- long-press or alternate gesture quick menu in viewport
- contextual radial actions for transform, sculpt, symmetry, boolean mode, and viewport actions
- recent/favorite actions entry

Must deliver:

- common touch/pen actions become faster without adding permanent chrome

Constraint:

- if this slice threatens Batch 1 pacing, it can ship after B1-7 as the first follow-on slice

## Cross-Slice Dependencies

- B1-0 should be complete or sufficiently narrowed before B1-1 and B1-2 harden.
- B1-1 is the structural base for every later slice.
- B1-2 depends on B1-1 because the adaptive shelf needs a permanent home.
- B1-4 and B1-5 depend on B1-2 because the new workflow should be selection-driven.
- B1-6 depends on B1-1 and benefits from B1-2 for contextual sculpt actions.
- B1-7 can start after B1-1.
- B1-8 should reuse the same action model exposed by B1-2 and B1-7.

## Recommended Build Order

1. B1-0 Shell Stability Gate
2. B1-1 Viewport-First Shell Frame
3. B1-2 Selection Context Engine
4. B1-3 Scene / Layers Drawer Refresh
5. B1-4 Live Blockout Workflow
6. B1-5 Hybrid Status And Branch-To-Sculpt Flow
7. B1-6 Sculpt Workspace Shell
8. B1-7 Command Search And Quick Access
9. B1-8 Radial Menu And Canvas Quick Actions

## Suggested Parallelization

Safe parallelization after B1-1:

- B1-2 selection context engine
- B1-3 scene/layers drawer refresh
- B1-7 command search and quick access

Safe parallelization after B1-2:

- B1-4 live blockout workflow
- B1-5 hybrid status and branch-to-sculpt flow
- B1-6 sculpt workspace shell

## Minimum Professional Standard For Batch 1

Batch 1 should not be called complete unless all of these are true:

- the viewport clearly dominates the app shell
- the selected object's likely next actions are visible without deep panel navigation
- blockout creation feels direct and lightweight
- sculpt mode has a visibly stronger professional identity
- live/hybrid/sculpt object status is understandable
- command access is credible for both touch and desktop users

## Batch 1 Verification

Every slice should use a combination of:

- Flutter analyze
- Flutter test
- Windows debug build
- manual viewport interaction checks
- manual touch-first and desktop-adaptive shell checks

Batch-level manual scenarios should include:

1. Create primitives rapidly without opening the inspector first.
2. Select representative node types and confirm the adaptive context shelf changes meaningfully.
3. Move from a live SDF blockout into sculpt without losing track of source/live state.
4. Stay in sculpt mode for a longer session and confirm the shell feels purpose-built.
5. Use command search to reach important actions without menu hunting.
6. Confirm the scene drawer remains navigable on both tablet-sized and desktop-sized windows.

## Batch 1 Acceptance Criteria

- Batch 1 stays separate from `plans/prd.json` until explicitly promoted into active execution.
- The first batch is concrete enough to split into individual implementation slices.
- The slices preserve Rust backend ownership and Flutter shell rendering responsibilities.
- The batch makes the product feel meaningfully more professional even before later enhancement waves land.
- The batch visibly amplifies what is unique about SDF Modeler instead of only imitating benchmark apps.

## Candidate Promotion Path Into Active Execution

If the team wants to start implementing immediately after migration stabilization,
the most sensible first promotion path is:

1. B1-0 Shell Stability Gate
2. B1-1 Viewport-First Shell Frame
3. B1-2 Selection Context Engine
4. B1-4 Live Blockout Workflow
5. B1-6 Sculpt Workspace Shell

That path creates a visible product transformation quickly while leaving lower-risk
supporting slices to follow.
