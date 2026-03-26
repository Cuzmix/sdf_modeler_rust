use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::super::scene_lookup::scene_row_at;
use crate::app::actions::Action;
use crate::app::slint_frontend::{SceneRowAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_scene_row_action(move |action, index| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_scene_row_action(host_state, action, index);
        });
    });
}

fn handle_scene_row_action(host_state: &mut SlintHostState, action: SceneRowAction, index: i32) {
    match action {
        SceneRowAction::SelectRow => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::Select(Some(row.host_id)));
        }
        SceneRowAction::ToggleRowSelection => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::ToggleSelection(row.host_id));
        }
        SceneRowAction::ToggleRowVisibility => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::ToggleVisibility(row.object_root_id));
        }
        SceneRowAction::ToggleRowLock => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::ToggleLock(row.host_id));
        }
        SceneRowAction::DuplicateRow => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::DuplicatePresentedObject(row.object_root_id));
        }
        SceneRowAction::DeleteRow => {
            let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                return;
            };
            host_state.queue_action(Action::DeletePresentedObject(row.object_root_id));
        }
    }
}
