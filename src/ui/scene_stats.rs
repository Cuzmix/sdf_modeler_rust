use eframe::egui;

use crate::app::FrameTimings;
use crate::graph::scene::Scene;

/// Draw the Scene Statistics panel showing real-time metrics about scene complexity.
pub fn draw(ui: &mut egui::Ui, scene: &Scene, timings: &FrameTimings) {
    let counts = scene.node_type_counts();

    // --- Node counts ---
    egui::CollapsingHeader::new("Node Counts")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.monospace(format!("{:>4}", counts.total));
                ui.label("Total");
            });
            ui.horizontal(|ui| {
                ui.monospace(format!("{:>4}", counts.visible));
                ui.label("Visible");
            });
            ui.separator();

            let type_rows = [
                ("Primitives", counts.primitives, egui::Color32::from_rgb(100, 149, 237)),
                ("Operations", counts.operations, egui::Color32::from_rgb(100, 200, 100)),
                ("Transforms", counts.transforms, egui::Color32::from_rgb(180, 130, 255)),
                ("Modifiers", counts.modifiers, egui::Color32::from_rgb(255, 200, 80)),
                ("Sculpts", counts.sculpts, egui::Color32::from_rgb(255, 160, 80)),
                ("Lights", counts.lights, egui::Color32::from_rgb(255, 220, 50)),
            ];

            for (label, count, color) in &type_rows {
                if *count > 0 {
                    ui.horizontal(|ui| {
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(8.0, 8.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().circle_filled(rect.center(), 4.0, *color);
                        ui.monospace(format!("{count:>4}"));
                        ui.label(*label);
                    });
                }
            }
        });

    ui.add_space(4.0);

    // --- SDF Complexity ---
    let sdf_complexity = scene.sdf_eval_complexity();
    let shader_ops_estimate = scene.node_type_counts();
    let estimated_ops = shader_ops_estimate.primitives * 10
        + shader_ops_estimate.operations * 20
        + shader_ops_estimate.modifiers * 15
        + shader_ops_estimate.transforms * 5
        + shader_ops_estimate.sculpts * 30;

    egui::CollapsingHeader::new("SDF Complexity")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.monospace(format!("{sdf_complexity:>4}"));
                ui.label("Eval nodes");
            });
            ui.horizontal(|ui| {
                let complexity_color = if estimated_ops < 200 {
                    egui::Color32::from_rgb(100, 255, 100)
                } else if estimated_ops < 500 {
                    egui::Color32::from_rgb(255, 255, 100)
                } else {
                    egui::Color32::from_rgb(255, 100, 100)
                };
                ui.monospace(format!("~{estimated_ops:>3}"));
                ui.colored_label(complexity_color, "shader ops (est.)");
            });
        });

    ui.add_space(4.0);

    // --- Voxel Memory ---
    let voxel_bytes = scene.voxel_memory_bytes();
    if voxel_bytes > 0 || counts.sculpts > 0 {
        egui::CollapsingHeader::new("Voxel Memory")
            .default_open(true)
            .show(ui, |ui| {
                let memory_color = if voxel_bytes < 10 * 1024 * 1024 {
                    egui::Color32::from_rgb(100, 255, 100)
                } else if voxel_bytes < 50 * 1024 * 1024 {
                    egui::Color32::from_rgb(255, 255, 100)
                } else {
                    egui::Color32::from_rgb(255, 100, 100)
                };
                ui.horizontal(|ui| {
                    ui.monospace(format_bytes(voxel_bytes));
                    ui.colored_label(memory_color, "allocated");
                });
            });
    }

    ui.add_space(4.0);

    // --- Frame Timing ---
    egui::CollapsingHeader::new("Frame Timing")
        .default_open(true)
        .show(ui, |ui| {
            let fps_color = if timings.avg_fps >= 55.0 {
                egui::Color32::from_rgb(100, 255, 100)
            } else if timings.avg_fps >= 30.0 {
                egui::Color32::from_rgb(255, 255, 100)
            } else {
                egui::Color32::from_rgb(255, 100, 100)
            };
            ui.horizontal(|ui| {
                ui.monospace(format!("{:5.1}", timings.avg_fps));
                ui.colored_label(fps_color, "FPS");
            });
            ui.horizontal(|ui| {
                ui.monospace(format!("{:5.2}", timings.avg_frame_ms));
                ui.label("ms/frame");
            });
        });
}

/// Format byte count into a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
