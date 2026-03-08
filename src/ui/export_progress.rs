use std::sync::atomic::Ordering;

use eframe::egui;

use crate::app::{ExportStatus, ImportStatus};

/// Draw a dimmed overlay behind modals.
fn draw_overlay(ctx: &egui::Context) {
    let screen_rect = ctx.screen_rect();
    egui::Area::new(egui::Id::new("progress_modal_overlay"))
        .fixed_pos(screen_rect.min)
        .show(ctx, |ui| {
            ui.painter()
                .rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(100));
        });
}

/// Draw the export progress modal. Returns true if a modal was drawn.
pub fn draw_export(ctx: &egui::Context, export_status: &ExportStatus) -> bool {
    let ExportStatus::InProgress {
        progress,
        total,
        resolution,
        cancelled,
        ref path,
        ..
    } = export_status
    else {
        return false;
    };

    let done = progress.load(Ordering::Relaxed);
    let frac = done as f32 / (*total).max(1) as f32;

    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("mesh");
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("obj")
        .to_uppercase();

    let mut do_cancel = false;

    draw_overlay(ctx);

    egui::Window::new("Exporting Mesh")
        .resizable(false)
        .collapsible(false)
        .title_bar(true)
        .default_width(350.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.label(format!("Exporting to: {}", filename));
            ui.label(format!("Format: {} | Resolution: {}^3", ext, resolution));
            ui.add_space(8.0);

            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("{:.1}%", frac * 100.0))
                    .animate(true),
            );
            ui.add_space(4.0);

            // Phase description
            let sample_slices = *resolution + 1;
            let phase = if done < sample_slices {
                format!("Phase 1/3: Sampling SDF field ({}/{})", done, sample_slices)
            } else if done < *total {
                let cell_done = done - sample_slices;
                let cell_total = *total - sample_slices;
                format!(
                    "Phase 2/3: Extracting triangles ({}/{})",
                    cell_done, cell_total
                )
            } else {
                "Phase 3/3: Merging vertices and sampling colors...".to_string()
            };
            ui.weak(phase);
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel Export").clicked() {
                        do_cancel = true;
                    }
                });
            });
        });

    if do_cancel {
        cancelled.store(true, Ordering::Relaxed);
    }

    ctx.request_repaint();
    true
}

/// Draw the import progress modal. Returns true if a modal was drawn.
pub fn draw_import(ctx: &egui::Context, import_status: &ImportStatus) -> bool {
    let ImportStatus::InProgress {
        progress,
        total,
        ref filename,
        cancelled,
        ..
    } = import_status
    else {
        return false;
    };

    let done = progress.load(Ordering::Relaxed);
    let frac = done as f32 / (*total).max(1) as f32;

    let mut do_cancel = false;

    draw_overlay(ctx);

    egui::Window::new("Importing Mesh")
        .resizable(false)
        .collapsible(false)
        .title_bar(true)
        .default_width(350.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.label(format!("Importing: {}", filename));
            ui.label("Converting mesh to SDF voxel grid...");
            ui.add_space(8.0);

            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("{:.1}%", frac * 100.0))
                    .animate(true),
            );
            ui.add_space(4.0);

            ui.weak(format!("Voxelizing slice {}/{}", done, total));
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel Import").clicked() {
                        do_cancel = true;
                    }
                });
            });
        });

    if do_cancel {
        cancelled.store(true, Ordering::Relaxed);
    }

    ctx.request_repaint();
    true
}
