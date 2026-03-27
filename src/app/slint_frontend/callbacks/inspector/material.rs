use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::edit_helpers::{apply_scalar_edit, apply_vector_edit};
use crate::app::slint_frontend::{InspectorEditMode, MaterialEditKind, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_material_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_material_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn handle_material_edit(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    kind: MaterialEditKind,
    mode: InspectorEditMode,
    axis: i32,
    value: f32,
) {
    match kind {
        MaterialEditKind::Color => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let tool_state = window.get_tool_panel_state();
                    let state = if tool_state.material.visible {
                        tool_state.material
                    } else {
                        window.get_inspector_panel_state().material
                    };
                    [
                        state.color_r.value,
                        state.color_g.value,
                        state.color_b.value,
                    ]
                },
                |app, component, next| {
                    app.set_selected_material_color_component(component, next);
                },
            );
        }
        MaterialEditKind::Roughness => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| {
                    let tool_state = window.get_tool_panel_state();
                    if tool_state.material.visible {
                        tool_state.material.roughness.value
                    } else {
                        window.get_inspector_panel_state().material.roughness.value
                    }
                },
                |app, next| {
                    app.set_selected_material_roughness(next);
                },
            );
        }
        MaterialEditKind::Metallic => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| {
                    let tool_state = window.get_tool_panel_state();
                    if tool_state.material.visible {
                        tool_state.material.metallic.value
                    } else {
                        window.get_inspector_panel_state().material.metallic.value
                    }
                },
                |app, next| {
                    app.set_selected_material_metallic(next);
                },
            );
        }
    }
}
