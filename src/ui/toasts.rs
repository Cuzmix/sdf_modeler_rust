use eframe::egui;

use crate::app::Toast;

/// Draw active toast notifications (bottom-right, auto-dismissing with fade-out).
pub fn draw(ctx: &egui::Context, toasts: &mut Vec<Toast>) {
    let now = crate::compat::Instant::now();
    toasts.retain(|t| now.duration_since(t.created) < t.duration);

    if toasts.is_empty() {
        return;
    }

    egui::Area::new(egui::Id::new("toasts"))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -36.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            for toast in toasts.iter() {
                let elapsed = now.duration_since(toast.created).as_secs_f32();
                let remaining = toast.duration.as_secs_f32() - elapsed;
                // Fade out over last 0.5s
                let alpha = (remaining / 0.5).clamp(0.0, 1.0);

                let (bg, text_color) = if toast.is_error {
                    (
                        egui::Color32::from_rgba_unmultiplied(120, 30, 30, (220.0 * alpha) as u8),
                        egui::Color32::from_rgba_unmultiplied(255, 180, 180, (255.0 * alpha) as u8),
                    )
                } else {
                    (
                        egui::Color32::from_rgba_unmultiplied(30, 90, 50, (220.0 * alpha) as u8),
                        egui::Color32::from_rgba_unmultiplied(180, 255, 200, (255.0 * alpha) as u8),
                    )
                };

                egui::Frame::none()
                    .fill(bg)
                    .rounding(6.0)
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(&toast.message).color(text_color));
                    });
                ui.add_space(4.0);
            }
        });

    // Keep repainting while toasts are visible
    ctx.request_repaint();
}
