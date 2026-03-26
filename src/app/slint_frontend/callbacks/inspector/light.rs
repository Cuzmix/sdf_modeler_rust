use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use super::edit_helpers::{apply_scalar_edit, apply_vector_edit};
use crate::app::slint_frontend::{InspectorEditMode, LightEditKind, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_light_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_light_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn handle_light_edit(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    kind: LightEditKind,
    mode: InspectorEditMode,
    axis: i32,
    value: f32,
) {
    match kind {
        LightEditKind::Color => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.light.color_r.value,
                        state.light.color_g.value,
                        state.light.color_b.value,
                    ]
                },
                |app, component, next| {
                    app.set_selected_light_color_component(component, next);
                },
            );
        }
        LightEditKind::Intensity => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().light.intensity.value,
                |app, next| {
                    app.set_selected_light_intensity(next);
                },
            );
        }
        LightEditKind::Range => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().light.range.value,
                |app, next| {
                    app.set_selected_light_range(next);
                },
            );
        }
        LightEditKind::CastShadows => {
            host_state.app.set_selected_light_cast_shadows(value >= 0.5);
        }
        LightEditKind::Volumetric => {
            host_state.app.set_selected_light_volumetric(value >= 0.5);
        }
        LightEditKind::VolumetricDensity => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| {
                    window
                        .get_inspector_panel_state()
                        .light
                        .volumetric_density
                        .value
                },
                |app, next| {
                    app.set_selected_light_volumetric_density(next);
                },
            );
        }
    }
}
