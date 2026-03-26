use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::edit_helpers::{apply_scalar_edit, apply_scalar_value};
use crate::app::slint_frontend::{InspectorEditMode, SculptEditKind, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_sculpt_edit(move |kind, mode, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_sculpt_edit(host_state, &edit_context, kind, mode, value);
        });
    });
}

fn handle_sculpt_edit(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    kind: SculptEditKind,
    mode: InspectorEditMode,
    value: f32,
) {
    match kind {
        SculptEditKind::Resolution => {
            let next = apply_scalar_value(context, mode, value, |window| {
                window.get_inspector_panel_state().sculpt_resolution as f32
            })
            .max(8.0) as u32;
            host_state.app.set_selected_sculpt_resolution(next);
        }
        SculptEditKind::LayerIntensity => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().sculpt_layer_intensity,
                |app, next| {
                    app.set_selected_sculpt_layer_intensity(next);
                },
            );
        }
        SculptEditKind::BrushRadius => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().sculpt_brush_radius,
                |app, next| {
                    app.set_selected_brush_radius(next);
                },
            );
        }
        SculptEditKind::BrushStrength => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().sculpt_brush_strength,
                |app, next| {
                    app.set_selected_brush_strength(next);
                },
            );
        }
    }
}
