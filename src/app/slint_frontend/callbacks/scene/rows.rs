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
    let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index).cloned() else {
        return;
    };

    match action {
        SceneRowAction::SelectRow => {
            host_state.queue_action(Action::Select(Some(row.host_id)));
        }
        SceneRowAction::ToggleRowSelection => {
            host_state.queue_action(Action::ToggleSelection(row.host_id));
        }
        SceneRowAction::ToggleRowExpanded => {
            host_state
                .app
                .ui
                .scene_panel
                .set_expanded(row.host_id, !row.expanded);
        }
        SceneRowAction::ToggleRowVisibility => {
            host_state.queue_action(Action::ToggleVisibility(row.object_root_id));
        }
        SceneRowAction::ToggleRowLock => {
            host_state.queue_action(Action::ToggleLock(row.host_id));
        }
        SceneRowAction::BeginRowRename => {
            host_state
                .app
                .ui
                .scene_panel
                .begin_rename(row.host_id, row.label);
        }
        SceneRowAction::DuplicateRow => {
            host_state.queue_action(Action::DuplicatePresentedObject(row.object_root_id));
        }
        SceneRowAction::DeleteRow => {
            host_state.queue_action(Action::DeletePresentedObject(row.object_root_id));
        }
        SceneRowAction::BeginRowDrag => {
            host_state.app.ui.scene_panel.begin_drag(row.object_root_id);
        }
        SceneRowAction::DropOnRow => {
            let Some(dragged) = host_state.app.ui.scene_panel.drag_source else {
                return;
            };
            if host_state
                .app
                .doc
                .scene
                .is_valid_drop_target(row.object_root_id, dragged)
            {
                host_state.app.ui.scene_panel.drop_target = Some(row.object_root_id);
                host_state.queue_action(Action::ReparentNode {
                    dragged,
                    new_parent: row.object_root_id,
                });
            }
            host_state.app.ui.scene_panel.clear_drag();
        }
    }
}
