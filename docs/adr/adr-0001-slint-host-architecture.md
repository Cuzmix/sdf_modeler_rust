# ADR-0001: Slint Host Architecture for Migration

- Status: Accepted
- Date: 2026-03-08
- PRD: #71
- Decision owner: UI/Rendering architecture

## Context

We are migrating from `eframe/egui` to Slint while preserving the existing Rust scene graph, reducer/action flow, and GPU viewport behavior.

Phase 0 required a host spike that proves:
1. a viewport region can be reserved,
2. a trivial `wgpu` pass renders in that region,
3. pointer coordinates map correctly into viewport-local coordinates,
4. host loop models are benchmarked for frame pacing and idle CPU behavior on Windows.

## Decision

Adopt **Model B** as migration baseline:

- **Model B (chosen):** `winit/wgpu` host owns frame loop and viewport rendering lifecycle; Slint is integrated around that host model.

Reject Model A as the primary architecture:

- **Model A (rejected as primary):** Slint owns window lifecycle and rendering notifier drives viewport image updates.

## Alternatives Considered

### Option A: Slint-owned loop

Implementation in spike:
- `experiments/slint_host_spike/src/bin/model_a_slint_owned.rs`

Pros:
- Fastest path to first Slint window + viewport image.
- Minimal loop plumbing.

Cons:
- Weak frame pacing for continuous high-frequency viewport rendering in this spike.
- Lower control over render cadence and synchronization compared with renderer-owned loop.
- Higher risk for parity work in camera/pick/gizmo interactions that currently assume explicit host timing control.

### Option B: Winit/WGPU-owned loop

Implementation in spike:
- `experiments/slint_host_spike/src/bin/model_b_winit_owned.rs`

Pros:
- Deterministic control of frame pacing and input routing.
- Better alignment with current architecture (GPU-centric viewport with explicit lifecycle).
- Cleaner path for preserving existing render/pick/gizmo behavior in later PRDs.

Cons:
- Slightly more integration complexity up-front.
- Requires explicit bridge layers for Slint shell/panels.

## Benchmark Evidence (Windows, 3s sample)

- Model A (`model_a_slint_owned`):
  - `frames=29`
  - `fps_avg=9.48`
  - `frame_ms_avg=86.39`
  - `frame_ms_p95=99.55`
  - `frame_ms_p99=191.66`
  - `idle_cpu_estimate_pct=99.64`

- Model B (`model_b_winit_owned`):
  - `frames=184`
  - `fps_avg=61.01`
  - `frame_ms_avg=16.34`
  - `frame_ms_p95=17.03`
  - `frame_ms_p99=17.15`
  - `idle_cpu_estimate_pct=1.57`

Interpretation:
- Model B shows materially better frame pacing and sustained throughput for viewport-style continuous rendering.
- Model A can render correctly but does not provide competitive pacing in this spike configuration.

## Consequences

- PRD #72+ proceeds assuming **renderer-owned host control**.
- Slint migration should prioritize explicit host-to-UI bridging instead of UI-owned render cadence.
- `legacy-egui-ui` remains available during transition as planned.

## Implementation Artifacts

- Spike crate: `experiments/slint_host_spike/`
- Run targets:
  - `plans/run-slint-spike.ps1`
  - `plans/run-slint-spike.sh`
- Perf baseline: `docs/slint_migration/perf_baseline.md`

## Notes

- Spike remains isolated from main runtime path.
- This ADR only selects host architecture; it does not yet migrate production UI.
