# Product Enhancements PRD

## Purpose

This PRD captures product and workflow enhancements that sit above the current
Flutter migration and the separate UI customization plan.

It exists so future feature work can be prioritized without mixing:

- active migration execution in `plans/prd.json`
- visual polish work in `docs/ui_customization_prd.md`
- exploratory product ideas that need a coherent roadmap

## Relationship To Other Planning Docs

- `plans/prd.json`: active migration and platform stabilization work
- `plans/progress.txt`: execution log for the active migration slices
- `docs/ui_customization_prd.md`: visual language, theming, and presentation polish
- `docs/professional_ux_overhaul_prd.md`: detailed product-UX overhaul direction
- `docs/professional_ux_overhaul_batch_1_prd.md`: first execution-ready overhaul batch
- `docs/egui_parity_audit.md`: current retained-vs-dropped scope for migration

This document is intentionally not part of the Ralph execution loop by default.
It is a backlog and sequencing artifact for future funded or prioritized work.

## Preconditions

This PRD assumes the following are true before any large enhancement slice starts:

- The Flutter host is the primary shell direction.
- Backend-owned scene, viewport, property, and workflow commands remain in Rust.
- The active migration PRD is either complete or reduced to isolated stabilization work.
- Desktop accessibility and viewport host stability are at least understood well enough
  that new feature work does not stack on top of unresolved shell churn.

## Goals

- Identify the highest-value product enhancements after migration stabilizes.
- Prefer enhancements that deepen authoring workflows instead of adding novelty for its own sake.
- Keep feature ownership aligned with the existing Rust backend and Flutter shell boundary.
- Group enhancement work into slices that can be shipped incrementally.
- Make future prioritization explicit enough that feature work does not become ad hoc.

## Non-Goals

- Reopening core architecture decisions already settled in `src/app_bridge/`
- Folding cosmetic restyling work into this document
- Treating every historical egui-only feature as automatically in scope
- Committing to a plugin system or scripting layer before core authoring workflows are stable

## Prioritization Principles

- Workflow depth beats surface-area breadth.
- Features that reduce repeated manual work rank higher than features that add isolated controls.
- Cross-toolkit/backend-owned features rank higher than new Flutter-only logic.
- Features that improve debugability, reliability, or accessibility rank high when they unblock future work.
- Large capability bets should be split into narrow vertical slices with manual verification gates.

## Product Opportunity Areas

### 1. Authoring Workflow Depth

- Reintroduce or expand authoring tools that materially improve scene construction speed.
- Prioritize reference imagery, measurement, cross-section inspection, and reusable presets.
- Keep these workflows backend-owned where possible so they survive future shell changes.

### 2. Reusable Content And Templates

- Turn one-off scene setup work into reusable assets.
- Prioritize node presets, material preset libraries, and repeatable export profiles.
- Keep preset serialization explicit and version-tolerant.

### 3. Viewport Intelligence

- Improve the viewport as a decision-making tool, not just a render target.
- Prioritize scene statistics, complexity diagnostics, clipping/cross-section tools, and selection feedback.
- Keep viewport overlays lightweight enough for the native texture path.

### 4. Sculpt And Modeling Acceleration

- Deepen workflows that reduce friction during sculpt and iterative modeling.
- Prioritize resolution management, brush falloff control, symmetry/mirroring workflows, and sculpt conversion ergonomics.
- Preserve the sculpt responsiveness guardrails already documented in the repo.

### 5. Pipeline And Interop

- Improve how scenes move into and out of the tool.
- Prioritize better import feedback, batch export flows, export presets, and project packaging.
- Avoid feature work that bloats file formats before core workflows stabilize.

### 6. Reliability, Accessibility, And Diagnostics

- Treat stability features as product enhancements when they prevent user trust erosion.
- Prioritize accessibility hardening, crash diagnostics, reproducible bug bundles, and higher-confidence integration coverage.
- Use these improvements to reduce regressions as the Flutter host grows.

## Recommended Rollout Order

1. Reliability and accessibility hardening that de-risks further shell work.
2. High-value authoring workflow tools that unblock day-to-day modeling speed.
3. Reusable presets and template workflows that compound productivity gains.
4. Viewport intelligence and diagnostics that help users understand complex scenes.
5. Deeper sculpt and pipeline enhancements once the shell and authoring basics are stable.
6. Larger bets such as graph-parity expansion or automation only after the earlier slices land cleanly.

## Candidate Execution Slices

### Slice A. Desktop Accessibility And Shell Hardening

Priority: P0

- Close the AXTree/semantics instability path in the Flutter Windows host.
- Add focused integration coverage for viewport overlays, modal layering, and accessibility-sensitive shell structure.
- Establish a baseline manual desktop accessibility checklist for future Flutter slices.

Success criteria:

- The repeated desktop AXTree failure is either fixed or narrowed to a known residual limitation.
- Viewport controls remain usable while decorative overlays no longer churn semantics.
- Future shell changes have a concrete regression checklist.

### Slice B. Reference Image Workflow Reintroduction

Priority: P1

- Revisit the current decision that reference images are out of Flutter parity scope.
- If product value is still high, move reference image state and commands behind backend-neutral ownership.
- Support import, placement, visibility, opacity, and simple transform controls sized for Flutter shell use.

Success criteria:

- Reference images can be loaded and managed from the Flutter host without egui-only ownership.
- Viewport overlays remain performant and predictable with reference content active.

### Slice C. Measurement And Cross-Section Toolkit

Priority: P1

- Add a dedicated inspection toolset for scene measurement and cross-section analysis.
- Keep measurements and cross-section state explicit rather than mixed into ad hoc overlays.
- Make the tools useful for blockout, scale validation, and mesh-export inspection.

Success criteria:

- Users can measure scene distances and inspect cross-sections from the primary host.
- The tools integrate cleanly with selection and viewport feedback.

### Slice D. Node Presets And Reusable Subtree Templates

Priority: P1

- Promote reusable scene fragments into a first-class workflow.
- Allow saving selected subtrees as templates and restoring them into future scenes.
- Keep naming, versioning, and placement behavior predictable.

Success criteria:

- A saved subtree can be reused without manual reconstruction.
- The preset workflow remains stable across ordinary scene/schema evolution.

### Slice E. Material Preset Library

Priority: P1

- Add a reusable material preset workflow for common looks.
- Support preset apply, overwrite, import/export, and clear ownership of preset storage.
- Keep parameter validation and material semantics in Rust.

Success criteria:

- Common looks can be applied quickly to multiple nodes without repeated manual tuning.
- Presets do not create duplicate material semantics in Dart.

### Slice F. Scene Statistics And Complexity Diagnostics

Priority: P2

- Expose scene complexity, memory, and structural diagnostics in a user-facing way.
- Help users understand why a scene is slow, heavy, or hard to export.
- Keep the diagnostics actionable rather than purely informational.

Success criteria:

- Users can inspect scene complexity and identify obvious hotspots.
- Diagnostics are aligned with actual backend structures, not guessed in the UI.

### Slice G. Workspace And Workflow Presets

Priority: P2

- Add saved shell/workspace layouts and higher-level workflow presets.
- Let users switch between modeling, sculpting, lighting, and export-focused workspace arrangements.
- Keep the touch-first shell contract intact while allowing desktop density improvements.

Success criteria:

- Users can switch between repeatable workspace configurations quickly.
- Workspace persistence does not reintroduce layout fragility.

### Slice H. Export Presets And Batch Pipeline Flow

Priority: P2

- Add reusable export profiles and multi-target export workflows.
- Support repeatable output settings for OBJ, glTF, USD, and other existing formats.
- Keep long-running export orchestration observable and cancellable.

Success criteria:

- Repeated export tasks no longer require full manual reconfiguration.
- Output profiles are explicit and testable.

### Slice I. Sculpt Resolution And Brush Workflow Expansion

Priority: P2

- Improve how sculpt resolution is managed over the life of a model.
- Add safer resample/upgrade/downgrade flows and richer brush shaping controls.
- Keep responsiveness and undo behavior non-negotiable.

Success criteria:

- Users can adjust sculpt fidelity without destructive guesswork.
- Active sculpt interaction remains smooth and predictable.

### Slice J. Advanced Viewport Authoring Surface Strategy

Priority: P3

- Decide which advanced viewport or graph-heavy tools deserve true Flutter parity.
- This includes the node graph editor and any advanced visual authoring surfaces still living only in egui.
- Prefer a deliberate product decision over silent scope creep.

Success criteria:

- Each advanced surface is explicitly retained, redesigned, or dropped.
- Future implementation work starts from a product decision instead of assumption.

### Slice K. Diagnostic Bundle And Support Workflow

Priority: P3

- Add a user-facing way to capture the information needed to debug project problems.
- Include environment, renderer, scene complexity, and recent workflow status where appropriate.
- Keep private or heavy data opt-in and explicit.

Success criteria:

- A reproducible bug bundle can be generated without asking users to manually gather raw logs.
- Support/debug flows become faster without coupling the app to external services.

## Suggested First Wave

If only three enhancement slices are funded after migration stabilization, start with:

1. Slice A: Desktop Accessibility And Shell Hardening
2. Slice C: Measurement And Cross-Section Toolkit
3. Slice D: Node Presets And Reusable Subtree Templates

This order improves trust first, then adds two workflow-deep capabilities with strong day-to-day value.

## Acceptance Criteria

- The enhancement plan stays separate from `plans/prd.json`.
- The enhancement plan stays separate from `docs/ui_customization_prd.md`.
- Proposed slices align with the current Rust-backend/Flutter-shell ownership model.
- The document is concrete enough to drive future slice selection without rediscovering the same roadmap.
