use super::{axis_value, mutate_host_and_tick, CallbackContext};
use crate::app::slint_frontend::{
    InspectorEditMode, LightEditKind, MaterialEditKind, OperationEditKind, SculptEditKind,
    SlintHostWindow, TransformEditKind,
};

use super::super::host_state::SlintHostState;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    install_transform_callbacks(window, context);
    install_material_callbacks(window, context);
    install_operation_callbacks(window, context);
    install_sculpt_callbacks(window, context);
    install_light_callbacks(window, context);
}

fn install_transform_callbacks(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_transform_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_transform_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn install_material_callbacks(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_material_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_material_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn install_operation_callbacks(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_operation_edit(move |kind, mode, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_operation_edit(host_state, &edit_context, kind, mode, value);
        });
    });
}

fn install_sculpt_callbacks(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_sculpt_edit(move |kind, mode, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_sculpt_edit(host_state, &edit_context, kind, mode, value);
        });
    });
}

fn install_light_callbacks(window: &SlintHostWindow, context: &CallbackContext) {
    let callback_context = context.clone();
    window.on_light_edit(move |kind, mode, axis, value| {
        let edit_context = callback_context.clone();
        mutate_host_and_tick(&callback_context, move |host_state| {
            handle_light_edit(host_state, &edit_context, kind, mode, axis, value);
        });
    });
}

fn handle_transform_edit(
    host_state: &mut SlintHostState,
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
    }
}

fn handle_material_edit(
    host_state: &mut SlintHostState,
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
        MaterialEditKind::Roughness => {
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
        MaterialEditKind::Metallic => {
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
    }
}

fn handle_operation_edit(
    host_state: &mut SlintHostState,
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
                |window| window.get_inspector_panel_state().operation_smooth_k,
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
                |window| window.get_inspector_panel_state().operation_steps,
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
                |window| window.get_inspector_panel_state().operation_color_blend,
                |app, next| {
                    app.set_selected_operation_color_blend(next);
                },
            );
        }
    }
}

fn handle_sculpt_edit(
    host_state: &mut SlintHostState,
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

fn handle_light_edit(
    host_state: &mut SlintHostState,
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
        LightEditKind::Intensity => {
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
        LightEditKind::Range => {
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
