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

            // Resolution slider
            let mut res_i32 = settings.export_resolution as i32;
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                ui.add(egui::Slider::new(&mut res_i32, 32..=512).suffix("^3"));
            });
            settings.export_resolution = res_i32 as u32;

            let voxels = (settings.export_resolution as u64).pow(3);
            ui.weak(format!("{} voxels", voxels));
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
