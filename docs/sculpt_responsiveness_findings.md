# Sculpt Responsiveness Findings

Last updated: 2026-03-09

## Summary

The user-reported issue was "laggy" sculpt and grab behavior even at a stable 60 FPS. The root problem was input-to-stroke latency and stepped updates, not rendering throughput.

## Observed Symptoms

- Grab felt delayed and moved in visible steps.
- Standard sculpt brushes looked rubber-banded during fast drags.
- Behavior was poor even at low voxel resolution.
- Shadow/AO settings could amplify the perception of stepping artifacts.

## Root Causes Identified

1. Async pick readback can arrive later than input updates, creating gaps between visible stroke updates.
2. Stroke interpolation spacing was brush-radius based only, not voxel-resolution aware.
3. Per-sample SDF deltas could be too aggressive, causing visible stepping.
4. Smooth brush used pure Laplacian smoothing, which can collapse volume and feel unstable.

## Implemented Fixes

### 1) Predictive sculpt while async pick is pending

- Added predictive drag application from pending cursor ray.
- Keeps brush motion continuous while waiting for exact GPU pick result.
- Files:
  - `src/app/sculpting.rs` (`predict_sculpt_from_pending_pick`, `ray_inputs_from_pending`)
  - `src/app/mod.rs` (predictive call in frame update before submitting next pick)

### 2) Grab continuity when cursor leaves geometry

- Grab can continue off-surface until release.
- Prevents abrupt drop-out when cursor exits mesh silhouette.
- File:
  - `src/app/sculpting.rs`

### 3) Kelvinlet Grab behavior

- Grab uses Kelvinlet backward warp from a stroke snapshot.
- Improves move/drag feel and avoids hard support cutoffs.
- File:
  - `src/sculpt.rs`

### 4) Voxel-aware interpolation density

- Stroke sampling step is constrained by voxel size, not only brush radius.
- Reduces rubber-banding during fast drag motion.
- File:
  - `src/app/sculpting.rs`

### 5) Per-sample delta clamp (CPU and GPU)

- Add/Carve/Flatten/Inflate deltas are clamped to a voxel-scale limit.
- Prevents sudden SDF jumps that look like stepping artifacts.
- Files:
  - `src/sculpt.rs`
  - `src/shaders/brush.wgsl`

### 6) Taubin smoothing for Smooth brush

- Replaced pure Laplacian-only behavior with Taubin-style two-pass smoothing.
- Better volume preservation and more stable feel.
- File:
  - `src/sculpt.rs`

## Non-Regression Test Matrix

Run these checks for any changes touching sculpt input, picking, brushes, or viewport update order:

1. Fast circular Add/Carve stroke on low-res sculpt node: no visible stepping.
2. Fast Grab motion across silhouette and off mesh: movement continues until release.
3. Smooth brush repeated passes: no obvious collapse spikes.
4. Shadows/AO enabled while sculpting: no strong stutter relative to disabled state.
5. Quick camera moves then immediate sculpt stroke: stroke starts without a visible hitch.

## Settings-Level Findings

- Raymarch and shadow step tuning can strongly affect perceived sculpt smoothness.
- Very conservative step multipliers can make results look cleaner but may raise GPU cost.
- Keep responsiveness testing separate from visual-quality tuning.

## Next Efficiency Targets (Likely)

1. Batch GPU brush dispatches per frame instead of submitting one command buffer per sample hit.
2. Reduce full-grid cloning inside smooth passes (use a local brush-region scratch buffer).
3. Cache expensive per-stroke differential snapshot work for Grab start on very large grids.
4. Add timing counters for: pick latency, stroke samples/frame, brush dispatch count/frame, texture region uploads/frame.
5. Evaluate optional direct CPU raycast mode during active sculpt drag (bypass pick latency entirely while dragging).

## Validation Commands

- `cargo check`
- `cargo clippy -- -D warnings`
- `cargo test`
- `cargo build`
