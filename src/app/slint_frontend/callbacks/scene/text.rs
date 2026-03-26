use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
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
        SceneTextAction::RenameSelected => host_state.app.rename_selected_object(text),
        SceneTextAction::FilterScene => {
            host_state.app.ui.scene_tree_search = text;
        }
    }
}
