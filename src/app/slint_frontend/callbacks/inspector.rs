use super::{axis_value, mutate_host_and_tick, CallbackContext};
use crate::app::slint_frontend::{InspectorEditKind, InspectorEditMode, SlintHostWindow};

use super::super::host_state::SlintHostState;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_inspector_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_inspector_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn handle_inspector_edit(
    host_state: &mut SlintHostState,
    context: &CallbackContext,
    kind: InspectorEditKind,
    mode: InspectorEditMode,
    axis: i32,
    value: f32,
) {
    match kind {
        InspectorEditKind::Position => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.transform_pos_x,
                        state.transform_pos_y,
                        state.transform_pos_z,
                    ]
                },
                |app, component, next| {
                    app.set_selected_position_component(component, next);
                },
            );
        }
        InspectorEditKind::Rotation => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.transform_rot_x,
                        state.transform_rot_y,
                        state.transform_rot_z,
                    ]
                },
                |app, component, next| {
                    app.set_selected_rotation_deg_component(component, next);
                },
            );
        }
        InspectorEditKind::Scale => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.selected_scale_x,
                        state.selected_scale_y,
                        state.selected_scale_z,
                    ]
                },
                |app, component, next| {
                    app.set_selected_scale_component(component, next);
                },
            );
        }
        InspectorEditKind::MaterialColor => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.material_color_r,
                        state.material_color_g,
                        state.material_color_b,
                    ]
                },
                |app, component, next| {
                    app.set_selected_material_color_component(component, next);
                },
            );
        }
        InspectorEditKind::MaterialRoughness => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().material_roughness,
                |app, next| {
                    app.set_selected_material_roughness(next);
                },
            );
        }
        InspectorEditKind::MaterialMetallic => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().material_metallic,
                |app, next| {
                    app.set_selected_material_metallic(next);
                },
            );
        }
        InspectorEditKind::OperationSmoothK => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().operation_smooth_k,
                |app, next| {
                    app.set_selected_operation_smooth_k(next);
                },
            );
        }
        InspectorEditKind::OperationSteps => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().operation_steps,
                |app, next| {
                    app.set_selected_operation_steps(next);
                },
            );
        }
        InspectorEditKind::OperationColorBlend => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().operation_color_blend,
                |app, next| {
                    app.set_selected_operation_color_blend(next);
                },
            );
        }
        InspectorEditKind::SculptResolution => {
            let next = apply_scalar_value(context, mode, value, |window| {
                window.get_inspector_panel_state().sculpt_resolution as f32
            })
            .max(8.0) as u32;
            host_state.app.set_selected_sculpt_resolution(next);
        }
        InspectorEditKind::SculptLayerIntensity => {
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
        InspectorEditKind::BrushRadius => {
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
        InspectorEditKind::BrushStrength => {
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
        InspectorEditKind::LightColor => {
            apply_vector_edit(
                host_state,
                context,
                axis,
                mode,
                value,
                |window| {
                    let state = window.get_inspector_panel_state();
                    [
                        state.light_color_r,
                        state.light_color_g,
                        state.light_color_b,
                    ]
                },
                |app, component, next| {
                    app.set_selected_light_color_component(component, next);
                },
            );
        }
        InspectorEditKind::LightIntensity => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().light_intensity,
                |app, next| {
                    app.set_selected_light_intensity(next);
                },
            );
        }
        InspectorEditKind::LightRange => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().light_range,
                |app, next| {
                    app.set_selected_light_range(next);
                },
            );
        }
        InspectorEditKind::LightCastShadows => {
            host_state.app.set_selected_light_cast_shadows(value >= 0.5);
        }
        InspectorEditKind::LightVolumetric => {
            host_state.app.set_selected_light_volumetric(value >= 0.5);
        }
        InspectorEditKind::LightVolumetricDensity => {
            apply_scalar_edit(
                host_state,
                context,
                mode,
                value,
                |window| window.get_inspector_panel_state().light_volumetric_density,
                |app, next| {
                    app.set_selected_light_volumetric_density(next);
                },
            );
        }
    }
}

fn apply_vector_edit<ReadCurrent, Apply>(
    host_state: &mut SlintHostState,
    context: &CallbackContext,
    axis: i32,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
    apply: Apply,
) where
    ReadCurrent: Fn(&SlintHostWindow) -> [f32; 3],
    Apply: Fn(&mut crate::app::SdfApp, usize, f32),
{
    let Some(window) = context.window_weak.upgrade() else {
        return;
    };
    let component = axis.max(0) as usize;
    let current = axis_value(axis, read_current(&window));
    let next = match mode {
        InspectorEditMode::Nudge => current + value,
        InspectorEditMode::Set | InspectorEditMode::Toggle => value,
    };
    apply(&mut host_state.app, component, next);
}

fn apply_scalar_edit<ReadCurrent, Apply>(
    host_state: &mut SlintHostState,
    context: &CallbackContext,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
    apply: Apply,
) where
    ReadCurrent: Fn(&SlintHostWindow) -> f32,
    Apply: Fn(&mut crate::app::SdfApp, f32),
{
    let next = apply_scalar_value(context, mode, value, read_current);
    apply(&mut host_state.app, next);
}

fn apply_scalar_value<ReadCurrent>(
    context: &CallbackContext,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
) -> f32
where
    ReadCurrent: Fn(&SlintHostWindow) -> f32,
{
    let Some(window) = context.window_weak.upgrade() else {
        return value;
    };
    match mode {
        InspectorEditMode::Nudge => read_current(&window) + value,
        InspectorEditMode::Set | InspectorEditMode::Toggle => value,
    }
}
