use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{SceneTextAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_scene_text_action(move |action, text| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_scene_text_action(host_state, action, text.to_string());
        });
    });
}

fn handle_scene_text_action(
    host_state: &mut SlintHostState,
    action: SceneTextAction,
    text: String,
) {
    match action {
        SceneTextAction::RenameSelected => {
            let Some(target_id) = host_state.app.ui.scene_panel.renaming_node else {
                return;
            };
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                host_state.queue_action(Action::RenameNode {
                    id: target_id,
                    name: trimmed.to_string(),
                });
            }
            host_state.app.ui.scene_panel.cancel_rename();
        }
        SceneTextAction::UpdateRenameBuffer => {
            host_state.app.ui.scene_panel.rename_buffer = text;
        }
        SceneTextAction::CancelRename => {
            host_state.app.ui.scene_panel.cancel_rename();
        }
        SceneTextAction::FilterScene => {
            host_state.app.ui.scene_panel.filter_query = text;
        }
    }
}
