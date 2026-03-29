use eframe::egui;

use crate::app::Toast;

/// Draw active toast notifications (bottom-right, auto-dismissing with fade-out).
pub fn draw(ctx: &egui::Context, toasts: &mut Vec<Toast>) {
    let now = crate::compat::Instant::now();
    toasts.retain(|t| now.duration_since(t.created) < t.duration);

    if toasts.is_empty() {
        return;
    }

    let motion = crate::ui::motion::settings(ctx);
    let toast_duration = motion.toast_duration().max(0.000_1);
    let mut stack_y = 0.0_f32;

    for toast in toasts.iter() {
        let toast_id = egui::Id::new(("toast", toast.message.as_str(), toast.is_error));
        let elapsed = now.duration_since(toast.created).as_secs_f32();
        let remaining = (toast.duration.as_secs_f32() - elapsed).max(0.0);
        let entry_t = if toast_duration == 0.0 {
            1.0
        } else {
            (elapsed / toast_duration).clamp(0.0, 1.0)
        };
        let exit_t = if toast_duration == 0.0 {
            1.0
        } else {
            (remaining / toast_duration).clamp(0.0, 1.0)
        };
        let surface_t = entry_t.min(exit_t);
        let alpha = crate::ui::motion::fade_alpha(surface_t, motion.reduced_motion);
        let target_y = stack_y;
        let animated_y =
            ctx.animate_value_with_time(toast_id.with("stack_y"), target_y, toast_duration);

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

        let area_response = egui::Area::new(toast_id)
            .anchor(
                egui::Align2::RIGHT_BOTTOM,
                egui::vec2(-12.0, -36.0 - animated_y),
            )
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.multiply_opacity(alpha);
                egui::Frame::none()
                    .fill(bg)
                    .rounding(6.0)
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(&toast.message).color(text_color));
                    });
            });
        crate::ui::motion::apply_surface_transform(ctx, &area_response.response, surface_t, motion);

        stack_y = animated_y + area_response.response.rect.height() + 6.0;
    }

    ctx.request_repaint_after(std::time::Duration::from_millis(16));
}
