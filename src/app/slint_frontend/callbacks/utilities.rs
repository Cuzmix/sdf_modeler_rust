use super::{mutate_host_and_tick, CallbackContext};
use crate::app::actions::Action;
use crate::app::reference_images::RefPlane;
use crate::app::slint_frontend::SlintHostWindow;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_set_render_show_grid(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_show_grid(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_render_show_node_labels(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_show_node_labels(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_render_show_bounding_box(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_show_bounding_box(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_render_show_light_gizmos(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_show_light_gizmos(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_render_shadows_enabled(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_shadows_enabled(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_render_ao_enabled(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_render_ao_enabled(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_export_resolution(move |delta| {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let next = (window.get_export_resolution() + delta).max(16) as u32;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_export_resolution(next);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_export_resolution(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_export_resolution(value.max(16) as u32);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_adaptive_export(move |value| {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.set_adaptive_export(value);
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_reference_visibility(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                host_state.queue_action(Action::ToggleReferenceImageVisibility(index as usize));
            });
        });
    }

    {
        let context = context.clone();
        window.on_toggle_reference_lock(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                let Some(reference) = host_state
                    .app
                    .ui
                    .reference_images
                    .images
                    .get_mut(index as usize)
                else {
                    return;
                };
                reference.locked = !reference.locked;
            });
        });
    }

    {
        let context = context.clone();
        window.on_cycle_reference_plane(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                let Some(reference) = host_state
                    .app
                    .ui
                    .reference_images
                    .images
                    .get_mut(index as usize)
                else {
                    return;
                };
                reference.plane = next_reference_plane(reference.plane);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_reference_opacity(move |index, value| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                let Some(reference) = host_state
                    .app
                    .ui
                    .reference_images
                    .images
                    .get_mut(index as usize)
                else {
                    return;
                };
                reference.opacity = value.clamp(0.0, 1.0);
            });
        });
    }

    {
        let context = context.clone();
        window.on_set_reference_scale(move |index, value| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                let Some(reference) = host_state
                    .app
                    .ui
                    .reference_images
                    .images
                    .get_mut(index as usize)
                else {
                    return;
                };
                reference.scale = value.clamp(0.05, 20.0);
            });
        });
    }

    {
        let context = context.clone();
        window.on_remove_reference_image(move |index| {
            mutate_host_and_tick(&context, move |host_state| {
                if index < 0 {
                    return;
                }
                host_state.queue_action(Action::RemoveReferenceImage(index as usize));
            });
        });
    }

    {
        let context = context.clone();
        window.on_nudge_import_resolution(move |delta| {
            mutate_host_and_tick(&context, move |host_state| {
                let Some(dialog) = host_state.app.ui.import_dialog.as_mut() else {
                    return;
                };
                dialog.use_auto = false;
                let max_resolution = host_state.app.settings.max_sculpt_resolution.max(8) as i32;
                dialog.resolution =
                    (dialog.resolution as i32 + delta).clamp(8, max_resolution) as u32;
            });
        });
    }

    {
        let context = context.clone();
        window.on_confirm_import(move || {
            let Some(window) = context.window_weak.upgrade() else {
                return;
            };
            let resolution = window.get_import_resolution().max(8) as u32;
            mutate_host_and_tick(&context, move |host_state| {
                host_state.queue_action(Action::CommitImport { resolution });
            });
        });
    }

    {
        let context = context.clone();
        window.on_cancel_import(move || {
            mutate_host_and_tick(&context, move |host_state| {
                host_state.app.ui.import_dialog = None;
            });
        });
    }
}

fn next_reference_plane(plane: RefPlane) -> RefPlane {
    match plane {
        RefPlane::Front => RefPlane::Back,
        RefPlane::Back => RefPlane::Left,
        RefPlane::Left => RefPlane::Right,
        RefPlane::Right => RefPlane::Top,
        RefPlane::Top => RefPlane::Bottom,
        RefPlane::Bottom => RefPlane::Front,
    }
}
