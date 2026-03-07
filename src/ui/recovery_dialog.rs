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
    if !*show_dialog {
        return None;
    }

    let dim_color = egui::Color32::from_black_alpha(180);
    let screen_rect = ctx.input(|input| input.screen_rect());
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("recovery_backdrop"),
    ));
    painter.rect_filled(screen_rect, 0.0, dim_color);

    let mut action = None;
    egui::Window::new("Crash Recovery")
        .id(egui::Id::new("crash_recovery_dialog"))
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_min_width(460.0);
            ui.label(recovery_summary);
            ui.add_space(8.0);
            ui.label("Choose how to continue:");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Recover").clicked() {
                    action = Some(RecoveryDialogAction::Recover);
                }
                if ui.button("Discard").clicked() {
                    action = Some(RecoveryDialogAction::Discard);
                }
                let open_last = ui.add_enabled(
                    has_recent_project,
                    egui::Button::new("Open Last Project"),
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
