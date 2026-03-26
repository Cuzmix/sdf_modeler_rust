use super::super::super::host_state::SlintHostState;
use super::super::{mutate_host_and_tick, CallbackContext};
use crate::app::actions::Action;
use crate::app::slint_frontend::{ImportDialogAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_import_dialog_action(move |action, value| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_import_dialog_action(host_state, action, value);
        });
    });
}

fn handle_import_dialog_action(
    host_state: &mut SlintHostState,
    action: ImportDialogAction,
    value: f32,
) {
    match action {
        ImportDialogAction::NudgeImportResolution => {
            let Some(dialog) = host_state.app.ui.import_dialog.as_mut() else {
                return;
            };
            dialog.use_auto = false;
            let max_resolution = host_state.app.settings.max_sculpt_resolution.max(8) as i32;
            dialog.resolution =
                (dialog.resolution as i32 + value as i32).clamp(8, max_resolution) as u32;
        }
        ImportDialogAction::ConfirmImport => {
            let resolution = host_state
                .app
                .ui
                .import_dialog
                .as_ref()
                .map(|dialog| dialog.resolution.max(8))
                .unwrap_or(8);
            host_state.queue_action(Action::CommitImport { resolution });
        }
        ImportDialogAction::CancelImport => {
            host_state.app.ui.import_dialog = None;
        }
    }
}
