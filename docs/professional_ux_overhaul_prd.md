# Professional UX Overhaul PRD

## Purpose

This PRD defines a full UX and workflow overhaul for SDF Modeler.

The goal is not to make the app cosmetically resemble any one existing tool.
The goal is to make it feel as immediate, intentional, and professional as the
best tablet-first and desktop-grade 3D tools, while leaning hard into what is
already unique about this application:

- live non-destructive SDF modeling
- fast boolean and modifier iteration
- conversion into voxel sculpting
- direct GPU viewport feedback
- a hybrid workflow that sits between clay sculpting, procedural modeling, and
  fast concept blockout

This document is the structural and workflow companion to
`docs/ui_customization_prd.md`.

- `docs/ui_customization_prd.md` covers visual language and presentation polish.
- This PRD covers product behavior, information architecture, interaction
  design, workflow structure, and the signature experience the app should aim
  for.
- The first execution-ready slice plan now lives in
  `docs/professional_ux_overhaul_batch_1_prd.md`.

## Product Thesis

SDF Modeler should become:

**the fastest professional app for going from live procedural blockout to tactile sculpt to export-ready result without feeling like three different products glued together**

That means:

- the **clarity and selection-driven focus** of Shapr3D
- the **touch immediacy and customizable sculpt workflow** of Nomad Sculpt
- the **depth, seriousness, and professional confidence** of ZBrush for iPad
- plus a **distinct hybrid SDF workflow** that those tools do not own

The app should feel less like a panel-heavy experiment and more like a
purpose-built digital sculpt/modeling workstation.

## Benchmark Synthesis

### What to learn from Nomad Sculpt

- Touch-first speed and low-friction sculpting
- Deep interface customization, floating toolbars, and shortcut surfaces
- Sculpt layers, topology tools, and quick validate/remesh workflows
- A feeling that the tool is built for staying in the canvas, not browsing menus

### What to learn from ZBrush for iPad

- Professional depth without making the interface feel toy-like
- Strong brush-centric workflow and sculpt-specific command surfaces
- Pencil-first customization, quick access, and pro-grade roundtrip thinking
- Confidence that the app can handle serious work, not just sketching

### What to learn from Shapr3D

- Selection-based adaptive UI
- Canvas-first interaction with minimal chrome
- Contextual tool recommendation instead of tool hunting
- Clear transitions between direct manipulation and parameter refinement

### What to lean into that is unique to this app

- Live SDF trees as editable design intent
- Boolean and modifier operations that remain fluid later than in polygon workflows
- One-tap progression from procedural blockout into sculpt
- Hybrid hard-surface plus organic ideation in the same document
- Lighting and rendering as part of design iteration, not just a final pass

## Benchmark References

These references informed the UX direction and workflow assumptions in this PRD:

- Nomad Sculpt interface customization and floating shortcuts:
  https://nomadsculpt.com/manual/interface
- Nomad Sculpt overall tablet-first interface and stats/nav cube orientation:
  https://nomadsculpt.com/manual/gettingstarted
- Nomad Sculpt layers and topology workflows:
  https://nomadsculpt.com/manual/layers
  https://nomadsculpt.com/manual/topology
- Nomad Sculpt workflow tips and validate-to-sculpt style behavior:
  https://nomadsculpt.com/manual/tips
- ZBrush for iPad touch-first UI, customization, and pro feature direction:
  https://www.maxon.net/en/zbrush-for-ipad
- Shapr3D adaptive UI and selection-driven workflow:
  https://support.shapr3d.com/hc/en-us/articles/7873882619548-Adaptive-user-interface
  https://support.shapr3d.com/hc/en-us/articles/7378907587484-Accessing-tools

## What "Professional" Means Here

Professional does not just mean dark theme, dense panels, or more buttons.

For this app, professional means:

- fast for experts
- teachable for new users
- stable under long sessions
- predictable under touch, pen, mouse, and keyboard
- able to expose depth without drowning the user in permanent UI
- trustworthy during destructive or expensive operations
- visually calm while still communicating state clearly

The app should feel like a tool that was designed around serious workflows, not
around showing every capability at the same time.

## Target Users

### 1. Concept sculptor

Needs:

- rapid blockout
- fast primitive combination
- quick transition into sculpting
- strong reference workflow
- dependable brush and viewport feedback

### 2. Hybrid modeler

Needs:

- live booleans and modifiers for ideation
- selective conversion into sculpt or baked output
- scene organization that does not break flow
- direct dimensions and alignment where useful

### 3. Hard-surface artist and product-style blockout user

Needs:

- precise transforms
- repeat/mirror/array style workflows
- adaptive tool suggestions from selection
- cleaner object management than a traditional DCC

### 4. Technical artist / lookdev-minded user

Needs:

- material and light experimentation
- scene statistics and diagnostics
- export repeatability
- a workflow that surfaces structure, not just final mesh data

## Core Jobs To Be Done

- Start a shape fast
- Change that shape without committing too early
- Understand what is selected and what can happen next
- Move between blockout, refinement, sculpting, and review without mental reset
- Keep complex scenes navigable
- Reuse work instead of rebuilding it
- Trust the app during long sessions and expensive operations

## Current Experience Gaps To Address

- Too much capability is hidden behind panel browsing rather than context.
- The viewport does not yet feel like the primary workspace.
- Sculpting does not yet own a distinct enough workspace identity.
- The app's SDF strengths are powerful but not presented as a signature workflow.
- The shell still risks feeling like a generic app frame around a graphics viewport.
- High-value workflows such as presets, reference, inspection, and quick contextual actions
  are not yet elevated enough.

## Experience Principles

### 1. Viewport First

The viewport is the product.
Panels support it; they do not define the experience.

### 2. Selection Drives The Interface

The user should rarely ask, "Where is the tool?"
The app should answer, "Given what you selected, here is the next likely action."

### 3. Modeful But Not Rigid

The app needs stronger workspaces and mode identity, especially for sculpting,
without becoming a maze of disconnected sub-apps.

### 4. Progressive Disclosure

The default experience should stay clean and fast.
Depth should appear when the current tool, selection, or mode demands it.

### 5. Direct Manipulation Over Panel Digging

Whenever possible:

- drag in viewport
- adjust inline
- confirm with lightweight controls
- use inspector only for deeper refinement

### 6. Keep Design Intent Alive

Do not force early destructive conversion.
SDF operations, modifiers, and procedural structure should stay editable as long
as possible.

### 7. Sculpt Like Clay, Model Like Design

Organic and hard-surface flows should both feel native.
The app's identity is in letting those flows meet cleanly.

### 8. Professional Trust

Every expensive, destructive, or long-running action needs:

- clear state
- preview where possible
- cancel when appropriate
- undo confidence
- visible progress

## North-Star Product Positioning

If Nomad feels like the best mobile sculpt sketchbook, and Shapr3D feels like
the cleanest adaptive modeling workspace, SDF Modeler should feel like:

**a hybrid design clay workstation**

Key differentiator:

- You can stay live and procedural longer than traditional sculpt apps.
- You can become tactile and sculptural faster than traditional CAD tools.
- You can move from blockout to sculpt to lookdev with less app-switching.

## Overhaul Outcome

The overhaul should produce these high-level shifts:

- from panel-first to viewport-first
- from static menus to adaptive contextual actions
- from one-size-fits-all workspace to mode-aware workspace
- from generic controls to touch-and-pen-native command surfaces
- from hidden SDF power to signature hybrid workflow

## Information Architecture Overhaul

## Primary Shell Structure

The shell should be reorganized around five persistent zones:

### 1. Top Command Bar

Purpose:

- document actions
- undo/redo
- mode indicator
- workspace switcher
- search/command
- save/sync/status

Must feel:

- slim
- high-signal
- not overloaded with per-tool controls

### 2. Left Tool Rail

Purpose:

- major tool families and workspace entry points

Recommended core entries:

- Select
- Add
- Transform
- Sculpt
- Material
- Light
- Inspect
- Export/Share

Behavior:

- icon-led with labels on larger widths
- mode-aware highlight
- long-press or right-click for favorite subtools

### 3. Adaptive Context Shelf

Purpose:

- selection-driven recommendations
- the next 4-8 likely actions based on current context

This is the Shapr3D-style engine translated to SDF/sculpt workflows.

Examples:

- selected primitive -> Move, Scale, Round, Mirror, Repeat, Convert to Sculpt
- selected boolean op -> Change op, Smooth, Reparent, Duplicate branch
- selected sculpt -> Brush, Layers, Remesh, Symmetry, Mask, Resolution
- selected light -> Intensity, Color, Solo, Shadow, Cookie, Link
- selected material-bearing node -> Material preset, Roughness, Metallic, Color

### 4. Scene / Layers Drawer

Purpose:

- scene graph
- grouping
- visibility
- lock
- solo/isolate
- saved sets and layers

This should behave more like a high-quality object manager than a raw tree widget.

### 5. Inspector / Property Sheet

Purpose:

- deeper refinement
- numeric control
- advanced options
- metadata

The inspector should be secondary for common actions and primary for precise edits.

## Workspace Model

The app should explicitly support four primary workspaces:

### Workspace A. Blockout

Focus:

- primitives
- booleans
- modifiers
- transforms
- fast spatial ideation

Feel:

- clean
- quick
- low friction
- direct

### Workspace B. Sculpt

Focus:

- brushes
- layers
- masks
- topology
- remesh
- symmetry

Feel:

- immersive
- brush-led
- sticky presets
- minimal unrelated chrome

### Workspace C. Lookdev

Focus:

- materials
- lights
- shading modes
- environment
- render quality toggles

Feel:

- visual
- feedback-oriented
- still viewport-first

### Workspace D. Review / Output

Focus:

- scene statistics
- cross-section
- measurement
- export presets
- final checks

Feel:

- analytical
- controlled
- trustworthy

## Interaction Model Overhaul

## 1. Selection-Based Adaptive UI

This is the single highest-value structural change.

Selection should determine:

- recommended tools
- default gizmo
- bottom parameter chips
- contextual HUD labels
- available quick actions in radial menu

Examples:

- tap primitive -> transform gizmo + primitive dimensions + modifier suggestions
- tap sculpt -> brush cursor tools + sculpt shelf + layer/mask actions
- tap light -> light gizmo + intensity/color quick chips + solo
- multi-select nodes -> group, align, duplicate, hide, material apply

## 2. Radial Menus And Quick Menus

The app should have a first-class quick access surface for touch and pen.

Recommended invocation:

- long-press in viewport
- Apple Pencil secondary gesture where supported
- keyboard shortcut on desktop

Suggested radial menu categories:

- transform
- brush switch
- symmetry
- mask/hide/solo
- add primitive
- boolean mode
- viewport modes
- recent/favorite actions

This is required if the app is to feel fast rather than menu-driven.

## 3. Validate / Commit / Keep Live Workflow

The current and future SDF experience should lean into a clear hybrid lifecycle:

- create live primitive or modifier stack
- iterate directly in viewport
- keep it live as long as useful
- either:
  - keep procedural
  - branch to sculpt
  - bake/validate to sculpt
  - duplicate branch and continue experimentation

This must become a visible UX concept, not just an implementation detail.

## 4. Direct Manipulation First

The default expectation for common actions should be:

- manipulate in viewport first
- refine in contextual shelf second
- use inspector for deep control

This includes:

- primitive drag-out creation
- live sizing handles
- transient dimension labels
- modifier handle manipulation where appropriate
- transform gizmo with larger hit zones and better pen ergonomics

## 5. Lightweight Confirmation Model

Avoid full modal interruption for common edits.

Prefer:

- floating accept/cancel strip
- tap-empty-space to commit simple adaptive actions when safe
- bottom confirmation chip for live tools
- full dialogs only for expensive, destructive, or file-related work

## Major Feature And Workflow Overhauls

## A. Viewport-First Modeling

### Goals

- Make shape creation feel sketchable and immediate
- Keep design intent visible
- Expose SDF power as something elegant rather than technical

### Required capabilities

- Drag-out primitive creation directly in viewport
- Live boolean preview while placing new shapes
- Modifier stack cards with visible effect labels
- Inline dimension and parameter chips near selection
- One-tap duplicate, mirror, repeat, and subtract

### Recommended additions

- ghost preview before placing
- placement snapping and smart alignment guides
- temporary "keep live / validate / branch to sculpt" strip
- quick branch duplication for concept variants

## B. Sculpt Workspace Reinvention

### Goals

- Make sculpting feel like a real mode, not just another inspector panel
- Preserve immediacy and depth
- Keep responsiveness sacred

### Required capabilities

- Brush-centric tool rail
- Persistent brush HUD with radius, strength, falloff, symmetry, and brush alpha
- Sculpt layers, masks, and visibility grouped together
- Topology controls as a dedicated sculpt utility strip
- Clear remesh/resolution status

### Recommended additions

- Favorite brush palette
- recent brushes row
- brush preset save/load
- sticky per-brush settings
- sculpt-only viewport shading presets
- "panic free" autosave and crash recovery confidence

### Professional differentiator

The app should make switching from live SDF blockout into tactile sculpt feel
like a strength, not a mode change penalty.

## C. Scene Graph And Layer System

### Goals

- Make scene management feel compact, visual, and fast
- Merge the best parts of object list and sculpt layer thinking

### Required capabilities

- Compact scene list with clear node-type badges
- visibility, lock, solo, and isolate inline
- drag-reparent and reorder
- search and filter
- active branch emphasis

### Recommended additions

- user color tags
- saved selection sets
- branch folders / groups
- "recently edited" highlight
- thumbnails or lightweight previews for key node types

### Important distinction

The app should preserve the difference between:

- scene nodes
- sculpt layers
- workflow presets

Do not collapse these into one ambiguous system.

## D. Adaptive Context Shelf

### Goals

- Reduce tool hunting
- Raise discoverability
- make expert flow faster without hurting clarity

### Behaviors

- Always show the next likely actions for the current selection
- Prioritize the single best action but expose a small "More" path
- Show parameter chips inline for the active action
- Allow tap-empty-space completion for safe actions

### Example recommendations

- selected face/body equivalent in SDF space -> Move, Scale, Offset, Mirror
- selected subtree -> Duplicate, Group, Isolate, Convert to Sculpt
- selected sculpt -> Smooth, Clay, Crease, Mask, Layer, Remesh
- selected light + node -> Link, Mask, Solo, Cookie

## E. Inspection And Review Toolkit

### Goals

- help users understand form, scale, complexity, and export-readiness
- make the app feel serious and production-aware

### Required capabilities

- measurement mode
- cross-section / clipping mode
- scene statistics
- topology or resolution indicators where relevant
- export/readiness checks

### Recommended additions

- screenshot / turntable presets
- compare modes for shading variants
- complexity hotspots for SDF depth or voxel density

## F. Presets And Reuse Everywhere

### Goals

- make repeated work cheap
- make professional workflows cumulative

### Required preset systems

- brush presets
- node presets
- material presets
- export presets
- workspace presets

### Recommended additions

- quick favorite bar
- recent presets
- project-local vs global presets
- portable preset packages

## G. Reference And Ideation Support

### Goals

- support real-world professional concept workflows
- reduce app switching during design and sculpt sessions

### Required capabilities

- reference image import
- viewport image planes or pinned overlays
- opacity/visibility/lock controls
- orthographic alignment support

### Recommended additions

- reference collections per project
- side-by-side moodboard/reference sheet
- sketch-over / markup ready capture views later

## Unique Signature Features To Double Down On

These are the features that should make SDF Modeler feel like itself, not like a copy.

## 1. Live Hybrid Stack

An object should clearly communicate whether it is:

- live SDF
- mixed/hybrid
- sculpt-baked

The UI should make this status obvious and useful.

## 2. Branch To Sculpt, Don't Just Collapse

Instead of a single destructive "convert" mindset, promote:

- validate in place
- branch and sculpt
- preserve live source

This is one of the clearest ways to exploit the app's unique workflow.

## 3. Modifier Cards As Design Language

Twist, bend, taper, round, shell, repeat, noise, and similar operations should
feel like a native language of the app.

Make them visible, reorderable where valid, understandable, and enjoyable to use.

## 4. Light As Part Of Form Exploration

Because the app already has richer-than-average lighting and render feedback,
the UX should treat lighting as part of the authoring loop.

Features to emphasize:

- solo
- link
- cookie
- proximity modulation
- arrays

This can become a unique creative differentiator for concept work.

## 5. Scene Intelligence

Use the SDF and sculpt knowledge in the backend to surface better guidance:

- complexity cues
- resolution warnings
- export risk hints
- expensive branch visibility

Professional apps feel smart when they help the user avoid bad states before
failure.

## Professional Fit-And-Finish Systems

## 1. Search And Command

The app should include a serious command search surface.

Requirements:

- every major action searchable
- recent commands
- favorites
- keyboard-first on desktop
- touch-friendly on tablet

## 2. Workspace Memory

The app should remember:

- workspace
- tool state
- recent brushes
- panel/drawer state
- viewport mode preferences

Users should feel that the tool remembers how they work.

## 3. Error, Loading, And Recovery Quality

Every background or risky workflow needs:

- explicit progress
- understandable errors
- recovery path
- autosave confidence

This includes:

- import
- export
- sculpt conversion
- texture/native host errors

## 4. Input Professionalism

The app should treat pen, touch, mouse, and keyboard as first-class, not bolted-on.

Requirements:

- larger gizmo hit zones
- tuned drag thresholds
- hover feedback where supported
- Pencil shortcuts where supported
- context menus on desktop
- keyboard efficiency without requiring it

## 5. Accessibility And Semantics

Professional quality includes accessibility.

Requirements:

- decorative overlays excluded from semantics where appropriate
- modal layering that does not confuse assistive technologies
- readable focus order
- keyboard-operable command surfaces

## 6. Motion With Purpose

Motion should communicate:

- tool activation
- mode transition
- confirmation
- sheet state
- hierarchy change

Avoid decorative motion that competes with modeling work.

## What To Avoid

- panel soup
- making every feature permanently visible
- generic Material/desktop app chrome around a viewport
- mixing sculpt, modeling, and lookdev controls in the same surface all the time
- hiding critical actions behind tiny icons
- forcing destructive conversion too early
- over-copying the exact UI of Nomad, ZBrush, or Shapr3D

## Rollout Strategy

## Phase 1. UX Foundation

- adaptive context shelf
- stronger workspace model
- viewport-first shell cleanup
- command search
- radial menu infrastructure
- accessibility and shell hardening

## Phase 2. SDF Workflow Signature

- drag-out primitives
- live boolean placement
- modifier cards
- branch-to-sculpt workflow
- better scene/layers UX

## Phase 3. Sculpt Identity

- sculpt workspace redesign
- brush HUD
- brush presets
- sculpt layers/masks ergonomics
- topology and remesh strip

## Phase 4. Professional Workflow Depth

- measurement and cross-section
- reference image workflow
- node/material/export/workspace presets
- scene intelligence and diagnostics

## Phase 5. Platform Confidence

- desktop accessibility polish
- integration coverage for shell workflows
- diagnostic bundle
- long-session performance and reliability work

## Success Metrics

The overhaul is successful when the app feels:

- faster to start with
- easier to understand mid-session
- more powerful without looking more cluttered
- more trustworthy in long or expensive workflows

Operationally, target outcomes should include:

- common actions available within one or two taps after selection
- a visibly reduced need to browse inspector sections for routine operations
- sculpt mode that users can stay in for long sessions without interface fatigue
- a clear user mental model for when something is live SDF vs sculpt-baked
- strong manual and automated confidence around viewport, modal, and accessibility behavior

## Acceptance Criteria

- The app has a clear hybrid product identity instead of feeling like a generic 3D editor.
- The shell is viewport-first and selection-driven.
- Sculpting has a distinct professional workspace identity.
- The SDF stack is surfaced as a signature workflow, not hidden complexity.
- The overhaul plan remains compatible with Rust backend ownership and Flutter shell architecture.
- The document is concrete enough to be split into execution slices without rethinking the product vision from scratch.
