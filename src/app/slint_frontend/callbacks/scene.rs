use super::{mutate_host_and_tick, scene_row_at, CallbackContext};
use crate::app::actions::Action;
use crate::app::slint_frontend::{
    SceneRowAction, SceneTextAction, SceneToolbarAction, SlintHostWindow, TopBarAction,
};
use crate::gizmo::GizmoMode;
use crate::graph::scene::{LightType, SdfPrimitive};

use super::super::host_state::SlintHostState;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_top_bar_action(move |action| {
            mutate_host_and_tick(&context, move |host_state| {
                handle_top_bar_action(host_state, action);
            });
        });
    }

    {
        let context = context.clone();
        window.on_scene_toolbar_action(move |action| {
            mutate_host_and_tick(&context, move |host_state| {
                handle_scene_toolbar_action(host_state, action);
            });
        });
    }

    {
        let context = context.clone();
        window.on_scene_row_action(move |action, index| {
            mutate_host_and_tick(&context, move |host_state| {
                handle_scene_row_action(host_state, action, index);
            });
        });
    }

    {
        let context = context.clone();
        window.on_scene_text_action(move |action, text| {
            mutate_host_and_tick(&context, move |host_state| {
                handle_scene_text_action(host_state, action, text.to_string());
            });
        });
    }
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
