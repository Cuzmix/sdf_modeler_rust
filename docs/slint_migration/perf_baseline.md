# Slint Host Spike Performance Baseline

Date: 2026-03-08
Scope: PRD #71 host architecture spike
Platform: Windows (primary dev machine)

## Commands

```powershell
experiments/slint_host_spike/target/debug/model_a_slint_owned.exe --benchmark-seconds 3
experiments/slint_host_spike/target/debug/model_b_winit_owned.exe --benchmark-seconds 3
```

## Raw Results

### Model A (Slint-owned lifecycle)

```text
model=model_a_slint_owned frames=29 elapsed_s=3.06 fps_avg=9.48 frame_ms_avg=86.39 frame_ms_p95=99.55 frame_ms_p99=191.66 idle_cpu_estimate_pct=99.64
```

### Model B (Winit/WGPU-owned lifecycle)

```text
model=model_b_winit_owned frames=184 elapsed_s=3.02 fps_avg=61.01 frame_ms_avg=16.34 frame_ms_p95=17.03 frame_ms_p99=17.15 idle_cpu_estimate_pct=1.57
```

## Summary

- Both models successfully reserve a viewport region, render a trivial `wgpu` pass, and map pointer coordinates into normalized viewport space.
- Model B has significantly better frame pacing and throughput in this spike.
- Architecture decision follows Model B (see ADR-0001).

---

## PRD #74 Integration Snapshot (Main Runtime, Slint Host)

Date: 2026-03-08
Scope: Integrated viewport path in src/ui_slint/winit_host.rs (full app renderer)

### Command

`powershell
cargo run -- --frontend slint --benchmark-seconds 2
`

### Raw Result

`	ext
model=main_slint_winit_host frames=122 elapsed_s=2.01 fps_avg=60.68 frame_ms_avg=15.85 frame_ms_p95=25.83 frame_ms_p99=29.37 idle_cpu_estimate_pct=14.09
`

### Comparison vs Phase-0 Model B Spike

- FPS average: 61.01 -> 60.68 (stable, -0.33 fps)
- Frame time average: 16.34 ms -> 15.85 ms (slightly better)
- Tail latency (p95/p99): 17.03/17.15 ms -> 25.83/29.37 ms (higher under full renderer workload)
- Idle CPU estimate: 1.57% -> 14.09% (expected increase because this run exercises real pipeline sync + scene rendering, not a trivial triangle pass)

### Notes

- This confirms the migrated Slint host keeps ~60 FPS in the integrated runtime path.
- Remaining perf work is tail-latency reduction during heavy interaction/pipeline changes (next PRD #74 follow-up).
