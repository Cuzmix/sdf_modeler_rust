# Slint Host Spike (PRD #71)

This isolated crate evaluates two migration host models without touching the production runtime path.

## Models

- `model_a_slint_owned`: Slint owns window/event lifecycle; WGPU rendering is injected via Slint rendering notifier into a dedicated viewport image region.
- `model_b_winit_owned`: Winit/WGPU owns the frame loop and viewport rendering. This is the baseline host for a future Slint embedding bridge.

Both binaries:
- reserve a viewport region,
- run a trivial triangle render pass,
- map pointer coordinates into normalized viewport coordinates,
- print a frame pacing + idle CPU estimate benchmark line, then auto-exit.

## Run

```powershell
cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_a_slint_owned -- --benchmark-seconds 12
cargo run --manifest-path experiments/slint_host_spike/Cargo.toml --bin model_b_winit_owned -- --benchmark-seconds 12
```

## Benchmark Output Format

Each run prints one summary line:

```text
model=<name> frames=<n> elapsed_s=<s> fps_avg=<fps> frame_ms_avg=<ms> frame_ms_p95=<ms> frame_ms_p99=<ms> idle_cpu_estimate_pct=<pct>
```

## Notes

- The benchmark is interactive-window based by design; run on a desktop session (Windows primary target for this phase).
- Keep this crate isolated until the architecture decision in `docs/adr/adr-0001-slint-host-architecture.md` is approved.