use eframe::egui;

use crate::settings::Settings;

/// Draw the Render Settings panel. Returns `true` if a shader-affecting setting changed.
pub fn draw(ui: &mut egui::Ui, settings: &mut Settings) -> bool {
    let mut shader_dirty = false;

    ui.heading("Render Settings");
    ui.separator();

    ui.label("Lighting");
    let prev = settings.shadows_enabled;
    ui.checkbox(&mut settings.shadows_enabled, "Raymarched Shadows");
    if settings.shadows_enabled != prev {
        shader_dirty = true;
        settings.save();
    }

    shader_dirty
}
