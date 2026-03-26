use super::super::super::host_state::SlintHostState;
use super::super::{mutate_host_and_tick, CallbackContext};
use crate::app::actions::Action;
use crate::app::slint_frontend::{SlintHostWindow, TopBarAction};
use crate::gizmo::GizmoMode;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_top_bar_action(move |action| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_top_bar_action(host_state, action);
        });
    });
}

fn handle_top_bar_action(host_state: &mut SlintHostState, action: TopBarAction) {
    match action {
        TopBarAction::NewScene => host_state.queue_action(Action::NewScene),
        TopBarAction::OpenProject => host_state.queue_action(Action::OpenProject),
        TopBarAction::SaveProject => host_state.queue_action(Action::SaveProject),
        TopBarAction::ImportMesh => host_state.queue_action(Action::ImportMesh),
        TopBarAction::ExportMesh => host_state.app.start_export(),
        TopBarAction::TakeScreenshot => host_state.queue_action(Action::TakeScreenshot),
        TopBarAction::AddReferenceImage => host_state.queue_action(Action::AddReferenceImage),
        TopBarAction::ToggleAllReferenceImages => {
            host_state.queue_action(Action::ToggleAllReferenceImages);
        }
        TopBarAction::FrameAll => host_state.queue_action(Action::FrameAll),
        TopBarAction::FocusSelected => host_state.queue_action(Action::FocusSelected),
        TopBarAction::Undo => host_state.queue_action(Action::Undo),
        TopBarAction::Redo => host_state.queue_action(Action::Redo),
        TopBarAction::CameraFront => host_state.queue_action(Action::CameraFront),
        TopBarAction::CameraTop => host_state.queue_action(Action::CameraTop),
        TopBarAction::CameraRight => host_state.queue_action(Action::CameraRight),
        TopBarAction::ToggleOrtho => host_state.queue_action(Action::ToggleOrtho),
        TopBarAction::ToggleMeasurement => {
            host_state.queue_action(Action::ToggleMeasurementTool);
        }
        TopBarAction::ToggleTurntable => host_state.queue_action(Action::ToggleTurntable),
        TopBarAction::SetGizmoTranslate => {
            host_state.queue_action(Action::SetGizmoMode(GizmoMode::Translate));
        }
        TopBarAction::SetGizmoRotate => {
            host_state.queue_action(Action::SetGizmoMode(GizmoMode::Rotate));
        }
        TopBarAction::SetGizmoScale => {
            host_state.queue_action(Action::SetGizmoMode(GizmoMode::Scale));
        }
        TopBarAction::ToggleGizmoSpace => host_state.queue_action(Action::ToggleGizmoSpace),
    }
}
