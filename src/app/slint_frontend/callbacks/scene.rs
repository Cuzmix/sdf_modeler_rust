use super::{mutate_host_and_tick, scene_row_at, CallbackContext};
use crate::app::actions::Action;
use crate::app::slint_frontend::SlintHostWindow;
use crate::gizmo::GizmoMode;
use crate::graph::scene::{LightType, SdfPrimitive};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_new_scene(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::NewScene);
            });
        });
    }

    {
        let context = context.clone();
        window.on_open_project(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::OpenProject);
            });
        });
    }

    {
        let context = context.clone();
        window.on_save_project(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::SaveProject);
            });
        });
    }

    {
        let context = context.clone();
        window.on_import_mesh(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ImportMesh);
            });
        });
    }

    {
        let context = context.clone();
        window.on_export_mesh(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.app.start_export();
            });
        });
    }

    {
        let context = context.clone();
        window.on_take_screenshot(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::TakeScreenshot);
            });
        });
    }

    {
        let context = context.clone();
        window.on_add_reference_image(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::AddReferenceImage);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_all_reference_images(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ToggleAllReferenceImages);
            });
        });
    }

    {
        let context = context.clone();
        window.on_frame_all(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::FrameAll);
            });
        });
    }

    {
        let context = context.clone();
        window.on_focus_selected(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::FocusSelected);
            });
        });
    }

    {
        let context = context.clone();
        window.on_undo(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::Undo);
            });
        });
    }

    {
        let context = context.clone();
        window.on_redo(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::Redo);
            });
        });
    }

    {
        let context = context.clone();
        window.on_camera_front(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CameraFront);
            });
        });
    }

    {
        let context = context.clone();
        window.on_camera_top(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CameraTop);
            });
        });
    }

    {
        let context = context.clone();
        window.on_camera_right(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CameraRight);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_ortho(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ToggleOrtho);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_measurement(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ToggleMeasurementTool);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_turntable(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ToggleTurntable);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_gizmo_translate(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::SetGizmoMode(GizmoMode::Translate));
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_gizmo_rotate(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::SetGizmoMode(GizmoMode::Rotate));
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_gizmo_scale(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::SetGizmoMode(GizmoMode::Scale));
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_gizmo_space(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::ToggleGizmoSpace);
            });
        });
    }

    {
        let context = context.clone();
        window.on_scene_filter_edited(move |text| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.ui.scene_tree_search = text.to_string();
            });
        });
    }

    {
        let context = context.clone();
        window.on_create_sphere(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Sphere));
            });
        });
    }

    {
        let context = context.clone();
        window.on_create_box(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Box));
            });
        });
    }

    {
        let context = context.clone();
        window.on_create_light(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::CreateLight(LightType::Point));
            });
        });
    }

    {
        let context = context.clone();
        window.on_duplicate_selected(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::Duplicate);
            });
        });
    }

    {
        let context = context.clone();
        window.on_delete_selected(move || {
            mutate_host_and_tick(&context, |host_state| {
                host_state.queue_action(Action::DeleteSelected);
            });
        });
    }

    {
        let context = context.clone();
        window.on_rename_selected(move |text| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.rename_selected_object(text.to_string());
            });
        });
    }

    {
        let context = context.clone();
        window.on_select_scene_row(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::Select(Some(row.host_id)));
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_scene_row(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::ToggleSelection(row.host_id));
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_scene_visibility(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::ToggleVisibility(row.object_root_id));
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_scene_lock(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::ToggleLock(row.host_id));
            });
        });
    }

    {
        let context = context.clone();
        window.on_duplicate_scene_row(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::DuplicatePresentedObject(row.object_root_id));
            });
        });
    }

    {
        let context = context.clone();
        window.on_delete_scene_row(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(row) = scene_row_at(host_state.last_snapshot.as_ref(), index) else {
                    return;
                };
                host_state.queue_action(Action::DeletePresentedObject(row.object_root_id));
            });
        });
    }
}
