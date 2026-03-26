use super::{mutate_host_and_tick, CallbackContext};
use crate::app::actions::Action;
use crate::app::reference_images::RefPlane;
use crate::app::slint_frontend::{SlintHostWindow, UtilityAction};

use super::super::host_state::SlintHostState;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_utility_action(move |action, index, value| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_utility_action(host_state, action, index, value);
        });
    });
}

fn handle_utility_action(
    host_state: &mut SlintHostState,
    action: UtilityAction,
    index: i32,
    value: f32,
) {
    match action {
        UtilityAction::SetRenderShowGrid => {
            host_state.app.set_render_show_grid(value >= 0.5);
        }
        UtilityAction::SetRenderShowNodeLabels => {
            host_state.app.set_render_show_node_labels(value >= 0.5);
        }
        UtilityAction::SetRenderShowBoundingBox => {
            host_state.app.set_render_show_bounding_box(value >= 0.5);
        }
        UtilityAction::SetRenderShowLightGizmos => {
            host_state.app.set_render_show_light_gizmos(value >= 0.5);
        }
        UtilityAction::SetRenderShadowsEnabled => {
            host_state.app.set_render_shadows_enabled(value >= 0.5);
        }
        UtilityAction::SetRenderAoEnabled => {
            host_state.app.set_render_ao_enabled(value >= 0.5);
        }
        UtilityAction::NudgeExportResolution => {
            let next = (host_state.app.settings.export_resolution as f32 + value).max(16.0) as u32;
            host_state.app.set_export_resolution(next);
        }
        UtilityAction::SetExportResolution => {
            host_state.app.set_export_resolution(value.max(16.0) as u32);
        }
        UtilityAction::SetAdaptiveExport => {
            host_state.app.set_adaptive_export(value >= 0.5);
        }
        UtilityAction::ToggleReferenceVisibility => {
            if index < 0 {
                return;
            }
            host_state.queue_action(Action::ToggleReferenceImageVisibility(index as usize));
        }
        UtilityAction::ToggleReferenceLock => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.locked = !reference.locked;
        }
        UtilityAction::CycleReferencePlane => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.plane = next_reference_plane(reference.plane);
        }
        UtilityAction::SetReferenceOpacity => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.opacity = value.clamp(0.0, 1.0);
        }
        UtilityAction::SetReferenceScale => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.scale = value.clamp(0.05, 20.0);
        }
        UtilityAction::RemoveReferenceImage => {
            if index < 0 {
                return;
            }
            host_state.queue_action(Action::RemoveReferenceImage(index as usize));
        }
        UtilityAction::NudgeImportResolution => {
            let Some(dialog) = host_state.app.ui.import_dialog.as_mut() else {
                return;
            };
            dialog.use_auto = false;
            let max_resolution = host_state.app.settings.max_sculpt_resolution.max(8) as i32;
            dialog.resolution =
                (dialog.resolution as i32 + value as i32).clamp(8, max_resolution) as u32;
        }
        UtilityAction::ConfirmImport => {
            let resolution = host_state
                .app
                .ui
                .import_dialog
                .as_ref()
                .map(|dialog| dialog.resolution.max(8))
                .unwrap_or(8);
            host_state.queue_action(Action::CommitImport { resolution });
        }
        UtilityAction::CancelImport => {
            host_state.app.ui.import_dialog = None;
        }
    }
}

fn reference_at(
    host_state: &mut SlintHostState,
    index: i32,
) -> Option<&mut crate::app::reference_images::ReferenceImageEntry> {
    if index < 0 {
        return None;
    }
    host_state
        .app
        .ui
        .reference_images
        .images
        .get_mut(index as usize)
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
