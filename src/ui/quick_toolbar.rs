use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::SdfPrimitive;

/// Draws a floating Quick Primitives toolbar anchored to the left edge of the viewport.
/// Each button creates a primitive at world origin and selects it.
pub fn draw(ctx: &egui::Context, show: &mut bool, actions: &mut ActionSink) {
    if !*show {
        return;
    }

    let mut open = true;
    egui::Window::new("Add Primitive")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::LEFT_CENTER, egui::vec2(8.0, 0.0))
        .frame(
            egui::Frame::window(&ctx.style())
                .fill(ctx.style().visuals.window_fill.gamma_multiply(0.9)),
        )
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                for primitive in SdfPrimitive::ALL {
                    if ui
                        .add(
                            egui::Button::new(primitive.base_name())
                                .min_size(egui::vec2(100.0, 24.0)),
                        )
                        .clicked()
                    {
                        actions.push(Action::CreatePrimitive(primitive.clone()));
                        *show = false;
                    }
                }
            });
        });

    if !open {
        *show = false;
    }
}
