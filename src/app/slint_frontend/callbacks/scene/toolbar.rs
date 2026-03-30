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

fn mapped_scene_toolbar_action(action: &SceneToolbarAction) -> Option<Action> {
    match action {
        SceneToolbarAction::CreateSphere => Some(Action::CreatePrimitive(SdfPrimitive::Sphere)),
        SceneToolbarAction::CreateBox => Some(Action::CreatePrimitive(SdfPrimitive::Box)),
        SceneToolbarAction::CreateLight => Some(Action::CreateLight(LightType::Point)),
        SceneToolbarAction::DuplicateSelected => Some(Action::Duplicate),
        SceneToolbarAction::DeleteSelected => Some(Action::DeleteSelected),
        SceneToolbarAction::OpenNodeWorkspace => Some(Action::TogglePanel(
            crate::app::state::PanelKind::NodeGraph,
            crate::app::state::PanelBarId::PrimaryRight,
        )),
        SceneToolbarAction::OpenLightWorkspace => {
            Some(Action::ToggleExpertPanel(crate::app::state::ExpertPanelKind::LightGraph))
        }
        SceneToolbarAction::StartRenameSelected | SceneToolbarAction::CancelRowDrag => None,
    }
}

fn handle_scene_toolbar_action(host_state: &mut SlintHostState, action: SceneToolbarAction) {
    if let Some(mapped) = mapped_scene_toolbar_action(&action) {
        host_state.queue_action(mapped);
        return;
    }

    match action {
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
        SceneToolbarAction::CancelRowDrag => {
            host_state.app.ui.scene_panel.clear_drag();
        }
        SceneToolbarAction::CreateSphere
        | SceneToolbarAction::CreateBox
        | SceneToolbarAction::CreateLight
        | SceneToolbarAction::DuplicateSelected
        | SceneToolbarAction::DeleteSelected
        | SceneToolbarAction::OpenNodeWorkspace
        | SceneToolbarAction::OpenLightWorkspace => {}
    }
}

#[cfg(test)]
mod tests {
    use super::mapped_scene_toolbar_action;
    use crate::app::actions::Action;
    use crate::app::slint_frontend::SceneToolbarAction;
    use crate::app::state::{PanelBarId, PanelKind};

    #[test]
    fn open_node_workspace_maps_to_node_graph_panel_toggle() {
        let mapped = mapped_scene_toolbar_action(&SceneToolbarAction::OpenNodeWorkspace);
        match mapped {
            Some(Action::TogglePanel(kind, bar_id)) => {
                assert_eq!(kind, PanelKind::NodeGraph);
                assert_eq!(bar_id, PanelBarId::PrimaryRight);
            }
            _ => panic!("expected node graph panel toggle action"),
        }
    }
}
