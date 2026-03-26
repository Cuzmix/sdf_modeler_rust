use super::super::super::host_state::SlintHostState;
use super::super::{mutate_host_and_tick, CallbackContext};
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
    }
}
