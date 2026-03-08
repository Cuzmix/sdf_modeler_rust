use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn contains(self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px <= self.x + self.width && py <= self.y + self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPointer {
    pub u: f32,
    pub v: f32,
}

pub fn map_pointer_to_viewport(px: f32, py: f32, viewport: ViewportRect) -> Option<NormalizedPointer> {
    if !viewport.contains(px, py) || viewport.width <= 0.0 || viewport.height <= 0.0 {
        return None;
    }

    let u = ((px - viewport.x) / viewport.width).clamp(0.0, 1.0);
    let v = ((py - viewport.y) / viewport.height).clamp(0.0, 1.0);

    Some(NormalizedPointer { u, v })
}

#[derive(Debug, Clone)]
pub struct FramePacingTracker {
    start_time: Instant,
    last_frame_time: Option<Instant>,
    busy_time: Duration,
    frame_intervals_ms: Vec<f32>,
    frame_count: u64,
}

impl Default for FramePacingTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FramePacingTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            last_frame_time: None,
            busy_time: Duration::ZERO,
            frame_intervals_ms: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn record_frame(&mut self, work_time: Duration) {
        let now = Instant::now();
        if let Some(previous) = self.last_frame_time {
            self.frame_intervals_ms.push((now - previous).as_secs_f32() * 1000.0);
        }
        self.last_frame_time = Some(now);
        self.busy_time += work_time;
        self.frame_count += 1;
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now() - self.start_time
    }

    pub fn build_report(&self, model_name: &str) -> String {
        let elapsed = self.elapsed().as_secs_f32().max(0.0001);
        let fps = self.frame_count as f32 / elapsed;
        let idle_ratio = (1.0 - (self.busy_time.as_secs_f32() / elapsed)).clamp(0.0, 1.0) * 100.0;

        let mut sorted = self.frame_intervals_ms.clone();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let avg_ms = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().sum::<f32>() / sorted.len() as f32
        };

        let p95_ms = percentile(&sorted, 0.95);
        let p99_ms = percentile(&sorted, 0.99);

        format!(
            "model={model_name} frames={} elapsed_s={:.2} fps_avg={:.2} frame_ms_avg={:.2} frame_ms_p95={:.2} frame_ms_p99={:.2} idle_cpu_estimate_pct={:.2}",
            self.frame_count, elapsed, fps, avg_ms, p95_ms, p99_ms, idle_ratio
        )
    }
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let clamped = pct.clamp(0.0, 1.0);
    let max_index = (sorted.len() - 1) as f32;
    let idx = (max_index * clamped).round() as usize;
    sorted[idx]
}

pub fn benchmark_seconds_from_args(default_seconds: u64) -> u64 {
    let args: Vec<String> = std::env::args().collect();

    let mut idx = 0usize;
    while idx < args.len() {
        let current = &args[idx];
        if let Some(value) = current.strip_prefix("--benchmark-seconds=") {
            if let Ok(seconds) = value.parse::<u64>() {
                return seconds.max(1);
            }
        }

        if current == "--benchmark-seconds" {
            if let Some(next) = args.get(idx + 1) {
                if let Ok(seconds) = next.parse::<u64>() {
                    return seconds.max(1);
                }
            }
        }

        idx += 1;
    }

    default_seconds
}

#[cfg(test)]
mod tests {
    use super::{map_pointer_to_viewport, benchmark_seconds_from_args, ViewportRect};

    #[test]
    fn pointer_map_inside_returns_normalized_point() {
        let viewport = ViewportRect { x: 100.0, y: 50.0, width: 400.0, height: 200.0 };
        let mapped = map_pointer_to_viewport(300.0, 150.0, viewport).unwrap();

        assert!((mapped.u - 0.5).abs() < 0.0001);
        assert!((mapped.v - 0.5).abs() < 0.0001);
    }

    #[test]
    fn pointer_map_outside_returns_none() {
        let viewport = ViewportRect { x: 100.0, y: 50.0, width: 400.0, height: 200.0 };
        assert!(map_pointer_to_viewport(10.0, 10.0, viewport).is_none());
    }

    #[test]
    fn benchmark_arg_parser_defaults_when_missing() {
        let parsed = benchmark_seconds_from_args(7);
        // We cannot control process args in this unit test without additional harness,
        // so this simply verifies no panic and a non-zero return.
        assert!(parsed >= 1);
    }
}