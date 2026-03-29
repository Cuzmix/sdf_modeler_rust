use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::SdfPrimitive;

/// Draws a floating Quick Primitives toolbar anchored to the left edge of the viewport.
/// Each button creates a primitive at world origin and selects it.
pub fn draw(ctx: &egui::Context, show: &mut bool, actions: &mut ActionSink) {
    let motion = crate::ui::motion::settings(ctx);
    let window_id = egui::Id::new("quick_primitives_window");
    let t = crate::ui::motion::surface_open_t(ctx, window_id, *show, motion);
    if !crate::ui::motion::should_draw_surface(*show, t) {
        crate::ui::motion::clear_surface_layers(ctx, window_id);
        return;
    }
    let alpha = crate::ui::motion::fade_alpha(t, motion.reduced_motion);

    let mut open = true;
    let mut window = egui::Window::new("Add Primitive")
        .id(window_id)
        .fade_in(false)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::LEFT_CENTER, egui::vec2(8.0, 0.0))
        .frame(crate::ui::motion::frame_with_alpha(
            egui::Frame::window(&ctx.style())
                .fill(ctx.style().visuals.window_fill.gamma_multiply(0.9)),
            t,
            motion,
        ));
    if *show {
        window = window.open(&mut open);
    }

    if let Some(window_response) = window.show(ctx, |ui| {
        ui.multiply_opacity(alpha);
        ui.vertical(|ui| {
            for primitive in SdfPrimitive::ALL {
                if ui
                    .add(egui::Button::new(primitive.base_name()).min_size(egui::vec2(100.0, 24.0)))
                    .clicked()
                {
                    actions.push(Action::CreatePrimitive(primitive.clone()));
                    *show = false;
                }
            }
        });
    }) {
        crate::ui::motion::apply_surface_transform(ctx, &window_response.response, t, motion);
    }

    if *show && !open {
        *show = false;
    }
}
