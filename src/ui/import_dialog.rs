use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::app::state::ImportDialog;

/// Draw the "Import Mesh" settings dialog. Allows the user to review mesh stats
/// and choose a voxel resolution (auto-calculated or manual) before voxelization.
pub fn draw(
    ctx: &egui::Context,
    dialog: &mut Option<ImportDialog>,
    actions: &mut ActionSink,
    max_sculpt_resolution: u32,
) {
    let Some(state) = dialog.as_mut() else {
        return;
    };

    let mut do_import = false;
    let mut do_cancel = false;

    egui::Window::new("Import Mesh")
        .resizable(false)
        .collapsible(false)
        .default_width(360.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.add_space(4.0);

            // File info
            ui.label(format!("File: {}", state.filename));
            ui.add_space(8.0);

            // Mesh statistics
            ui.group(|ui| {
                ui.label("Mesh Statistics");
                ui.add_space(4.0);
                egui::Grid::new("import_mesh_stats")
                    .num_columns(2)
                    .spacing([16.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Vertices:");
                        ui.label(format_count(state.vertex_count));
                        ui.end_row();

                        ui.label("Triangles:");
                        ui.label(format_count(state.triangle_count));
                        ui.end_row();

                        ui.label("Dimensions:");
                        ui.label(format!(
                            "{:.2} x {:.2} x {:.2}",
                            state.bounds_size.x, state.bounds_size.y, state.bounds_size.z
                        ));
                        ui.end_row();
                    });
            });

            ui.add_space(12.0);

            // Resolution mode toggle
            ui.label("Voxel Resolution:");
            ui.add_space(4.0);

            ui.radio_value(&mut state.use_auto, true, format!(
                "Auto-calculate ({}^3)",
                state.auto_resolution
            ));
            ui.radio_value(&mut state.use_auto, false, "Manual");

            if state.use_auto {
                state.resolution = state.auto_resolution;
            }

            ui.add_space(8.0);

            // Resolution presets + custom input (only when manual)
            ui.add_enabled_ui(!state.use_auto, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Presets:");
                    for &(label, res) in &[
                        ("Low", 32u32),
                        ("Medium", 64),
                        ("High", 96),
                        ("Ultra", 128),
                        ("Extreme", 256),
                    ] {
                        if ui
                            .selectable_label(state.resolution == res, label)
                            .clicked()
                        {
                            state.resolution = res;
                        }
                    }
                });

                let max_res = max_sculpt_resolution.max(16);
                ui.horizontal(|ui| {
                    ui.label("Custom:");
                    let mut res_i32 = state.resolution as i32;
                    let response = ui.add(
                        egui::DragValue::new(&mut res_i32)
                            .speed(1)
                            .range(16..=max_res as i32)
                            .suffix("^3"),
                    );
                    if response.changed() {
                        state.resolution = (res_i32 as u32).clamp(16, max_res);
                    }
                });
            });

            ui.add_space(8.0);

            // Stats and warnings
            let voxels = (state.resolution as u64).pow(3);
            let mem_mb = (voxels as f64 * 4.0) / (1024.0 * 1024.0);
            ui.weak(format!(
                "{} voxels ({:.1} MB)",
                format_voxel_count(voxels),
                mem_mb
            ));

            if state.resolution > 256 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 100, 100),
                    format!(
                        "Warning: {:.0} MB RAM — may cause slowdowns or crashes",
                        mem_mb
                    ),
                );
            } else if state.resolution > 128 {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "High resolution — voxelization will be slower",
                );
            }

            ui.add_space(12.0);

            // Buttons
            ui.horizontal(|ui| {
                if ui.button("Import").clicked() {
                    do_import = true;
                }
                if ui.button("Cancel").clicked() {
                    do_cancel = true;
                }
            });
        });

    if do_import {
        let resolution = state.resolution;
        actions.push(Action::CommitImport { resolution });
        // Don't clear dialog here — start_import_voxelize takes it via .take()
    } else if do_cancel {
        *dialog = None;
    }
}

/// Format large counts with K/M suffixes.
fn format_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

/// Format large voxel counts with K/M suffixes.
fn format_voxel_count(voxels: u64) -> String {
    if voxels >= 1_000_000 {
        format!("{:.1}M", voxels as f64 / 1_000_000.0)
    } else if voxels >= 1_000 {
        format!("{:.0}K", voxels as f64 / 1_000.0)
    } else {
        format!("{}", voxels)
    }
}
