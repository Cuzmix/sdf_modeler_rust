use eframe::egui;

pub enum RecoveryDialogAction {
    Recover,
    Discard,
    OpenLastProject,
}

/// Blocking startup recovery dialog shown after an unclean exit when autosave exists.
pub fn draw(
    ctx: &egui::Context,
    show_dialog: &mut bool,
    recovery_summary: &str,
    has_recent_project: bool,
) -> Option<RecoveryDialogAction> {
    let motion = crate::ui::motion::settings(ctx);
    let window_id = egui::Id::new("crash_recovery_dialog");
    let t = crate::ui::motion::surface_open_t(ctx, window_id, *show_dialog, motion);
    if !crate::ui::motion::should_draw_surface(*show_dialog, t) {
        crate::ui::motion::clear_surface_layers(ctx, window_id);
        return None;
    }
    let alpha = crate::ui::motion::fade_alpha(t, motion.reduced_motion);

    let visuals = ctx.style().visuals.clone();
    let backdrop_fill = if visuals.dark_mode {
        visuals.panel_fill.gamma_multiply(1.15)
    } else {
        visuals.panel_fill
    };
    let backdrop_tint = if visuals.dark_mode {
        egui::Color32::from_black_alpha(56)
    } else {
        egui::Color32::from_black_alpha(16)
    };

    egui::CentralPanel::default()
        .frame(egui::Frame::default().fill(backdrop_fill))
        .show(ctx, |ui| {
            let backdrop_rect = ui.max_rect();
            ui.allocate_rect(backdrop_rect, egui::Sense::click());
            ui.painter().rect_filled(
                backdrop_rect,
                0.0,
                egui::Color32::from_rgba_premultiplied(
                    backdrop_tint.r(),
                    backdrop_tint.g(),
                    backdrop_tint.b(),
                    (backdrop_tint.a() as f32 * alpha) as u8,
                ),
            );
        });

    let mut action = None;
    if let Some(window_response) = egui::Window::new(egui::RichText::new("Crash Recovery").strong())
        .id(window_id)
        .fade_in(false)
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::ui::motion::frame_with_alpha(
            egui::Frame::window(&ctx.style())
                .fill(visuals.window_fill.gamma_multiply(1.1))
                .inner_margin(egui::Margin::symmetric(18.0, 14.0))
                .shadow(egui::epaint::Shadow {
                    offset: egui::vec2(0.0, 10.0),
                    blur: 28.0,
                    spread: 2.0,
                    color: egui::Color32::from_black_alpha(96),
                }),
            t,
            motion,
        ))
        .show(ctx, |ui| {
            ui.multiply_opacity(alpha);
            ui.set_min_width(460.0);
            ui.label(recovery_summary);
            ui.add_space(8.0);
            ui.label("Choose how to continue:");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new("Recover").min_size(egui::vec2(92.0, 30.0)))
                    .clicked()
                {
                    action = Some(RecoveryDialogAction::Recover);
                }
                if ui
                    .add(egui::Button::new("Discard").min_size(egui::vec2(92.0, 30.0)))
                    .clicked()
                {
                    action = Some(RecoveryDialogAction::Discard);
                }
                let open_last = ui.add_enabled(
                    has_recent_project,
                    egui::Button::new("Open Last Project").min_size(egui::vec2(138.0, 30.0)),
                );
                if open_last.clicked() {
                    action = Some(RecoveryDialogAction::OpenLastProject);
                }
            });
            if !has_recent_project {
                ui.add_space(4.0);
                ui.weak("No recent project path available.");
            }
        })
    {
        crate::ui::motion::apply_surface_transform(ctx, &window_response.response, t, motion);
    }

    if action.is_some() {
        *show_dialog = false;
    }
    action
}
