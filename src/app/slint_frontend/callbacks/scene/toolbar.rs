use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{SceneToolbarAction, SlintHostWindow};
use crate::graph::scene::{LightType, SdfPrimitive};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_scene_toolbar_action(move |action| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_scene_toolbar_action(host_state, action);
        });
    });
}

fn handle_scene_toolbar_action(host_state: &mut SlintHostState, action: SceneToolbarAction) {
    match action {
        SceneToolbarAction::CreateSphere => {
            host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Sphere));
        }
        SceneToolbarAction::CreateBox => {
            host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Box));
        }
        SceneToolbarAction::CreateLight => {
            host_state.queue_action(Action::CreateLight(LightType::Point));
        }
        SceneToolbarAction::DuplicateSelected => host_state.queue_action(Action::Duplicate),
        SceneToolbarAction::DeleteSelected => host_state.queue_action(Action::DeleteSelected),
        SceneToolbarAction::StartRenameSelected => {
            let Some(selected_id) = host_state.app.ui.selection.selected else {
                return;
            };
            let Some(object) = crate::graph::presented_object::resolve_presented_object(
                &host_state.app.doc.scene,
                selected_id,
            ) else {
                return;
            };
            let Some(node) = host_state.app.doc.scene.nodes.get(&object.host_id) else {
                return;
            };
            host_state
                .app
                .ui
                .scene_panel
                .begin_rename(object.host_id, node.name.clone());
        }
        SceneToolbarAction::OpenNodeWorkspace => {
            host_state.queue_action(Action::ToggleExpertPanel(
                crate::app::state::ExpertPanelKind::NodeGraph,
            ));
        }
        SceneToolbarAction::OpenLightWorkspace => {
            host_state.queue_action(Action::ToggleExpertPanel(
                crate::app::state::ExpertPanelKind::LightGraph,
            ));
        }
        SceneToolbarAction::CancelRowDrag => {
            host_state.app.ui.scene_panel.clear_drag();
        }
    }
}
