use super::{axis_value, mutate_host_and_tick, CallbackContext};
use crate::app::slint_frontend::SlintHostWindow;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_nudge_selected_position(move |axis, delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = axis_value(
                axis,
                [
                    window.get_transform_pos_x(),
                    window.get_transform_pos_y(),
                    window.get_transform_pos_z(),
                ],
            ) + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_position_component(axis.max(0) as usize, next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_rotation(move |axis, delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = axis_value(
                axis,
                [
                    window.get_transform_rot_x(),
                    window.get_transform_rot_y(),
                    window.get_transform_rot_z(),
                ],
            ) + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_rotation_deg_component(axis.max(0) as usize, next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_scale(move |axis, delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = axis_value(
                axis,
                [
                    window.get_selected_scale_x(),
                    window.get_selected_scale_y(),
                    window.get_selected_scale_z(),
                ],
            ) + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_scale_component(axis.max(0) as usize, next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_material_color(move |axis, delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = axis_value(
                axis,
                [
                    window.get_material_color_r(),
                    window.get_material_color_g(),
                    window.get_material_color_b(),
                ],
            ) + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_material_color_component(axis.max(0) as usize, next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_material_roughness(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_material_roughness() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_material_roughness(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_material_metallic(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_material_metallic() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_material_metallic(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_material_roughness(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_material_roughness(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_material_metallic(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_material_metallic(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_operation_smooth_k(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_operation_smooth_k() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_smooth_k(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_operation_steps(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_operation_steps() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_steps(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_operation_color_blend(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_operation_color_blend() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_color_blend(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_operation_smooth_k(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_smooth_k(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_operation_steps(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_steps(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_operation_color_blend(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_operation_color_blend(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_sculpt_resolution(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = (window.get_sculpt_resolution() + delta).max(8) as u32;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_sculpt_resolution(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_sculpt_layer_intensity(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_sculpt_layer_intensity() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_sculpt_layer_intensity(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_brush_radius(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_sculpt_brush_radius() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_brush_radius(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_brush_strength(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_sculpt_brush_strength() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_brush_strength(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_sculpt_resolution(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_sculpt_resolution(value.max(8) as u32);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_sculpt_layer_intensity(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_sculpt_layer_intensity(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_brush_radius(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_brush_radius(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_brush_strength(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_brush_strength(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_light_color(move |axis, delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = axis_value(
                axis,
                [
                    window.get_light_color_r(),
                    window.get_light_color_g(),
                    window.get_light_color_b(),
                ],
            ) + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state
                    .app
                    .set_selected_light_color_component(axis.max(0) as usize, next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_light_intensity(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_light_intensity() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_intensity(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_light_range(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_light_range() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_range(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_selected_light_shadows(move || {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = !window.get_light_cast_shadows();
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_cast_shadows(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_selected_light_volumetric(move || {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = !window.get_light_volumetric();
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_volumetric(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_selected_light_volumetric_density(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = window.get_light_volumetric_density() + delta;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_volumetric_density(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_light_intensity(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_intensity(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_light_range(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_range(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_light_shadows(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_cast_shadows(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_light_volumetric(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_volumetric(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_selected_light_volumetric_density(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_selected_light_volumetric_density(value);
            });
        });
    }
}
