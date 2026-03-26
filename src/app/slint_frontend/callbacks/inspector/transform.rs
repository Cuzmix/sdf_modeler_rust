use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::edit_helpers::apply_vector_edit;
use crate::app::slint_frontend::{InspectorEditMode, SlintHostWindow, TransformEditKind};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_transform_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_transform_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn handle_transform_edit(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    kind: TransformEditKind,
    mode: InspectorEditMode,
    axis: i32,
    value: f32,
) {
    match kind {
        TransformEditKind::Position => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.transform.pos_x.value,
                        state.transform.pos_y.value,
                        state.transform.pos_z.value,
                    ]
                },
                |app, component, next| {
                    app.set_selected_position_component(component, next);
                },
            );
        }
        TransformEditKind::Rotation => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.transform.rot_x.value,
                        state.transform.rot_y.value,
                        state.transform.rot_z.value,
                    ]
                },
                |app, component, next| {
                    app.set_selected_rotation_deg_component(component, next);
                },
            );
        }
        TransformEditKind::Scale => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.transform.scale_x.value,
                        state.transform.scale_y.value,
                        state.transform.scale_z.value,
                    ]
                },
                |app, component, next| {
                    app.set_selected_scale_component(component, next);
                },
            );
        }
    }
}
