use crate::ui::egui_compat::{corner_radius, margin_symmetric};

pub fn draw_chip(ui: &mut egui::Ui, label: &str, fill: egui::Color32, text_color: egui::Color32) {
    egui::Frame::new()
        .fill(fill)
        .corner_radius(corner_radius(6.0))
        .inner_margin(margin_symmetric(6.0, 2.0))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(label)
                    .size(10.0)
                    .color(text_color)
                    .strong(),
            );
        });
}
