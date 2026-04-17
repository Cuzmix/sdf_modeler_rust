use eframe::egui;

use crate::app::state::GpuSyncState;
use crate::app::FrameTimings;
use crate::gpu::camera::Camera;
use crate::graph::scene::Scene;
use crate::settings::Settings;
use crate::ui::viewport::ViewportResources;

const TIMING_HISTORY_LEN: usize = 120;

/// Draw the profiler/debug window.
pub fn draw(
    ctx: &egui::Context,
    show: bool,
    timings: &FrameTimings,
    scene: &Scene,
    gpu: &GpuSyncState,
    settings: &Settings,
    camera: &Camera,
) {
    let motion = crate::ui::motion::settings(ctx);
    let window_id = egui::Id::new("profiler_window");
    let surface_t = crate::ui::motion::surface_open_t(ctx, window_id, show, motion);
    if !crate::ui::motion::should_draw_surface(show, surface_t) {
        crate::ui::motion::clear_surface_layers(ctx, window_id);
        return;
    }
    let alpha = crate::ui::motion::fade_alpha(surface_t, motion.reduced_motion);
    let t = timings;
    if let Some(window_response) = egui::Window::new("Profiler")
        .id(window_id)
        .fade_in(false)
        .default_pos([10.0, 10.0])
        .default_size([280.0, 320.0])
        .frame(crate::ui::motion::frame_with_alpha(
            egui::Frame::window(&ctx.style()),
            surface_t,
            motion,
        ))
        .show(ctx, |ui| {
            ui.multiply_opacity(alpha);
            // --- FPS / Frame time ---
            let color = if t.avg_fps >= 55.0 {
                egui::Color32::from_rgb(100, 255, 100)
            } else if t.avg_fps >= 30.0 {
                egui::Color32::from_rgb(255, 255, 100)
            } else {
                egui::Color32::from_rgb(255, 100, 100)
            };
            ui.colored_label(
                color,
                format!("FPS: {:.0}  ({:.2} ms)", t.avg_fps, t.avg_frame_ms),
            );

            // --- Frame time sparkline ---
            let history_ordered: Vec<f32> = {
                let idx = t.history_idx;
                let mut v = Vec::with_capacity(TIMING_HISTORY_LEN);
                v.extend_from_slice(&t.history[idx..]);
                v.extend_from_slice(&t.history[..idx]);
                v
            };
            let max_ms = history_ordered.iter().cloned().fold(1.0_f32, f32::max);
            let target_ms = 16.67_f32; // 60 FPS target line

            let (rect, _) = ui
                .allocate_exact_size(egui::vec2(ui.available_width(), 50.0), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));

            // Draw 60fps target line
            let target_y = rect.bottom() - (target_ms / max_ms) * rect.height();
            if target_y > rect.top() {
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), target_y),
                        egui::pos2(rect.right(), target_y),
                    ],
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_premultiplied(100, 100, 255, 80),
                    ),
                );
            }

            // Draw bars
            let bar_w = rect.width() / TIMING_HISTORY_LEN as f32;
            for (i, &ms) in history_ordered.iter().enumerate() {
                let h = (ms / max_ms) * rect.height();
                let x = rect.left() + i as f32 * bar_w;
                let bar_color = if ms <= 16.67 {
                    egui::Color32::from_rgb(80, 200, 80)
                } else if ms <= 33.33 {
                    egui::Color32::from_rgb(200, 200, 80)
                } else {
                    egui::Color32::from_rgb(200, 80, 80)
                };
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(x, rect.bottom() - h),
                        egui::vec2(bar_w.max(1.0), h),
                    ),
                    0.0,
                    bar_color,
                );
            }

            ui.add_space(2.0);

            // --- CPU phase breakdown ---
            egui::CollapsingHeader::new("CPU Phases")
                .default_open(true)
                .show(ui, |ui| {
                    ui.monospace(format!(
                        "Pipeline sync:  {:6.2} ms",
                        t.pipeline_sync_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "  structure key: {:6.2} ms",
                        t.structure_key_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "  shader codegen:{:6.2} ms",
                        t.shader_codegen_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "  rebuild pipes: {:6.2} ms",
                        t.pipeline_rebuild_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "Buffer upload:  {:6.2} ms",
                        t.buffer_upload_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "  build buffers: {:6.2} ms",
                        t.scene_buffer_build_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "  GPU writes:    {:6.2} ms",
                        t.scene_buffer_write_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "Comp dispatch:  {:6.2} ms",
                        t.composite_dispatch_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "Data hash:      {:6.2} ms",
                        t.data_fingerprint_s * 1000.0
                    ));
                    ui.monospace(format!(
                        "History commit: {:6.2} ms",
                        t.history_finalize_s * 1000.0
                    ));
                    ui.monospace(format!("UI draw:        {:6.2} ms", t.ui_draw_s * 1000.0));
                    ui.monospace(format!("Total CPU:      {:6.2} ms", t.total_cpu_s * 1000.0));
                });

            egui::CollapsingHeader::new("Sculpt Telemetry")
                .default_open(true)
                .show(ui, |ui| {
                    ui.monospace(format!("Brush samples/frame: {}", t.sculpt_brush_samples));
                    ui.monospace(format!("GPU dispatches/frame: {}", t.sculpt_gpu_dispatches));
                    ui.monospace(format!("GPU submits/frame: {}", t.sculpt_gpu_submits));
                    ui.monospace(format!(
                        "Pick latency:   {:6.2} ms (EMA {:6.2} ms)",
                        t.sculpt_pick_latency_ms, t.sculpt_pick_latency_avg_ms,
                    ));
                });
            ui.separator();

            // --- Scene stats ---
            egui::CollapsingHeader::new("Scene")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Nodes: {}", scene.nodes.len()));
                    ui.label(format!("Top-level: {}", scene.top_level_nodes().len()));
                    ui.label(format!("Sculpt textures: {}", gpu.sculpt_tex_indices.len()));
                    ui.label(format!(
                        "Composite: {}",
                        if settings.render.composite_volume_enabled {
                            "ON"
                        } else {
                            "OFF"
                        }
                    ));
                });

            // --- Render state ---
            egui::CollapsingHeader::new("Render State")
                .default_open(true)
                .show(ui, |ui| {
                    let renderer = gpu.render_state.renderer.read();
                    if let Some(res) = renderer.callback_resources.get::<ViewportResources>() {
                        ui.label(format!(
                            "Render size: {}x{}",
                            res.render_width, res.render_height
                        ));
                        ui.label(format!("Composite active: {}", res.use_composite));
                    }
                });

            // --- Camera ---
            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(ui, |ui| {
                    let eye = camera.eye();
                    ui.label(format!("Eye: ({:.2}, {:.2}, {:.2})", eye.x, eye.y, eye.z));
                    ui.label(format!("Distance: {:.2}", camera.distance));
                    ui.label(format!(
                        "Yaw: {:.1} Pitch: {:.1}",
                        camera.yaw.to_degrees(),
                        camera.pitch.to_degrees(),
                    ));
                });
        })
    {
        crate::ui::motion::apply_surface_transform(
            ctx,
            &window_response.response,
            surface_t,
            motion,
        );
    }
}
