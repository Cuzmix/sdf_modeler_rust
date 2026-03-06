use eframe::egui;

use crate::app::ExportStatus;
use crate::settings::Settings;

/// Result of drawing the export dialog this frame.
pub enum ExportDialogResult {
    /// No action — dialog still open or not visible.
    None,
    /// User clicked Export.
    Export,
    /// User closed or cancelled the dialog.
    Closed,
}

/// Format large voxel counts with K/M suffixes for readability.
fn format_voxel_count(voxels: u64) -> String {
    if voxels >= 1_000_000 {
        format!("{:.1}M", voxels as f64 / 1_000_000.0)
    } else if voxels >= 1_000 {
        format!("{:.0}K", voxels as f64 / 1_000.0)
    } else {
        format!("{}", voxels)
    }
}

/// Format vertex counts with K/M suffixes.
fn format_vertex_count(verts: u64) -> String {
    if verts >= 1_000_000 {
        format!("{:.1}M", verts as f64 / 1_000_000.0)
    } else if verts >= 1_000 {
        format!("{:.0}K", verts as f64 / 1_000.0)
    } else {
        format!("{}", verts)
    }
}

/// Draw the export mesh dialog. Returns the user's action.
pub fn draw(
    ctx: &egui::Context,
    show: &mut bool,
    settings: &mut Settings,
    export_status: &ExportStatus,
) -> ExportDialogResult {
    if !*show {
        return ExportDialogResult::None;
    }
    let mut open = *show;
    let mut do_export = false;
    let mut do_cancel = false;

    egui::Window::new("Export Mesh")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .default_width(300.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.heading("Export Settings");
            ui.add_space(4.0);

            // Preset buttons
            if !settings.export_presets.is_empty() {
                ui.label("Presets:");
                ui.horizontal(|ui| {
                    let presets = settings.export_presets.clone();
                    for preset in &presets {
                        if ui.button(&preset.name).clicked() {
                            settings.export_resolution = preset.resolution;
                        }
                    }
                });
                ui.add_space(4.0);
            }

            // Resolution input
            let mut res_i32 = settings.export_resolution as i32;
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                ui.add(
                    egui::DragValue::new(&mut res_i32)
                        .speed(1)
                        .range(16..=2048)
                        .suffix("^3"),
                );
            });
            settings.export_resolution = (res_i32 as u32).clamp(16, 2048);

            // Marching cubes estimate: surface-crossing cells ≈ 6*N^2 (sphere-like),
            // each generates ~2 triangles and ~3 unique vertices on average.
            let res = settings.export_resolution as u64;
            let voxels = res.pow(3);
            let mem_mb = (voxels as f64 * 4.0) / (1024.0 * 1024.0);
            let estimated_tris = 6 * res * res * 2;
            let estimated_verts = estimated_tris * 3 / 2; // ~1.5 verts per tri after dedup
            ui.weak(format!(
                "{} voxels ({:.1} MB)",
                format_voxel_count(voxels), mem_mb
            ));
            ui.weak(format!(
                "~{} triangles, ~{} vertices (estimate for typical geometry)",
                format_vertex_count(estimated_tris), format_vertex_count(estimated_verts)
            ));

            // Warnings
            if settings.export_resolution > 512 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 100, 100),
                    format!("Warning: {:.0} MB — export may take very long or run out of memory", mem_mb),
                );
            } else if settings.export_resolution > 256 {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "High resolution — export may take longer",
                );
            }
            ui.add_space(4.0);

            // Adaptive toggle
            ui.checkbox(&mut settings.adaptive_export, "Adaptive sampling")
                .on_hover_text("Skip empty regions for faster export at high resolutions");
            ui.add_space(8.0);

            // Info text
            ui.label("Supported formats: OBJ, STL, PLY, glTF (.glb), USD (.usda)");
            ui.weak("PLY, glTF, and USD include vertex colors.");
            ui.add_space(8.0);

            // Export button
            let export_idle = matches!(export_status, ExportStatus::Idle);
            ui.horizontal(|ui| {
                if ui.add_enabled(export_idle, egui::Button::new("Export...")).clicked() {
                    do_export = true;
                }
                if ui.button("Cancel").clicked() {
                    do_cancel = true;
                }
            });
        });

    if !open || do_cancel {
        *show = false;
        return ExportDialogResult::Closed;
    }
    if do_export {
        *show = false;
        return ExportDialogResult::Export;
    }
    ExportDialogResult::None
}
