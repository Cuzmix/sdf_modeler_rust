use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::edit_helpers::apply_scalar_edit;
use crate::app::slint_frontend::{InspectorEditMode, OperationEditKind, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_operation_edit(move |kind, mode, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_operation_edit(host_state, &edit_context, kind, mode, value);
        });
    });
}

fn handle_operation_edit(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    kind: OperationEditKind,
    mode: InspectorEditMode,
    value: f32,
) {
    match kind {
        OperationEditKind::SmoothK => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().operation.smooth_k.value,
                |app, next| {
                    app.set_selected_operation_smooth_k(next);
                },
            );
        }
        OperationEditKind::Steps => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().operation.steps.value,
                |app, next| {
                    app.set_selected_operation_steps(next);
                },
            );
        }
        OperationEditKind::ColorBlend => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| {
                    window
                        .get_inspector_panel_state()
                        .operation
                        .color_blend
                        .value
                },
                |app, next| {
                    app.set_selected_operation_color_blend(next);
                },
            );
        }
    }
}
