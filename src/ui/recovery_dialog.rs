use crate::ui::egui_compat::{margin_symmetric, shadow};

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
    if !*show_dialog {
        return None;
    }

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
            ui.painter().rect_filled(backdrop_rect, 0.0, backdrop_tint);
        });

    let mut action = None;
    egui::Window::new(egui::RichText::new("Crash Recovery").strong())
        .id(egui::Id::new("crash_recovery_dialog"))
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(
            egui::Frame::window(&ctx.style())
                .fill(visuals.window_fill.gamma_multiply(1.1))
                .inner_margin(margin_symmetric(18.0, 14.0))
                .shadow(shadow(
                    0.0,
                    10.0,
                    28.0,
                    2.0,
                    egui::Color32::from_black_alpha(96),
                )),
        )
        .show(ctx, |ui| {
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
        });

    if action.is_some() {
        *show_dialog = false;
    }
    action
}
