use super::super::super::host_state::SlintHostState;
use super::super::{mutate_host_and_tick, CallbackContext};
use crate::app::slint_frontend::{RenderSettingsAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_render_settings_action(move |action, value| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_render_settings_action(host_state, action, value);
        });
    });
}

fn handle_render_settings_action(
    host_state: &mut SlintHostState,
    action: RenderSettingsAction,
    value: f32,
) {
    match action {
        RenderSettingsAction::SetShowGrid => {
            host_state.app.set_render_show_grid(value >= 0.5);
        }
        RenderSettingsAction::SetShowNodeLabels => {
            host_state.app.set_render_show_node_labels(value >= 0.5);
        }
        RenderSettingsAction::SetShowBoundingBox => {
            host_state.app.set_render_show_bounding_box(value >= 0.5);
        }
        RenderSettingsAction::SetShowLightGizmos => {
            host_state.app.set_render_show_light_gizmos(value >= 0.5);
        }
        RenderSettingsAction::SetShadowsEnabled => {
            host_state.app.set_render_shadows_enabled(value >= 0.5);
        }
        RenderSettingsAction::SetAoEnabled => {
            host_state.app.set_render_ao_enabled(value >= 0.5);
        }
        RenderSettingsAction::NudgeExportResolution => {
            let next = (host_state.app.settings.export_resolution as f32 + value).max(16.0) as u32;
            host_state.app.set_export_resolution(next);
        }
        RenderSettingsAction::SetExportResolution => {
            host_state.app.set_export_resolution(value.max(16.0) as u32);
        }
        RenderSettingsAction::SetAdaptiveExport => {
            host_state.app.set_adaptive_export(value >= 0.5);
        }
    }
}
