use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{ReferenceImageAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_reference_image_action(move |action, index, value| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_reference_image_action(host_state, action, index, value);
        });
    });
}

fn handle_reference_image_action(
    host_state: &mut SlintHostState,
    action: ReferenceImageAction,
    index: i32,
    value: f32,
) {
    match action {
        ReferenceImageAction::ToggleReferenceVisibility => {
            if index < 0 {
                return;
            }
            host_state.queue_action(Action::ToggleReferenceImageVisibility(index as usize));
        }
        ReferenceImageAction::ToggleReferenceLock => {
            if index < 0 {
                return;
            }
            host_state.app.toggle_reference_image_lock(index as usize);
        }
        ReferenceImageAction::CycleReferencePlane => {
            if index < 0 {
                return;
            }
            host_state.app.cycle_reference_image_plane(index as usize);
        }
        ReferenceImageAction::SetReferenceOpacity => {
            if index < 0 {
                return;
            }
            host_state
                .app
                .set_reference_image_opacity(index as usize, value);
        }
        ReferenceImageAction::SetReferenceScale => {
            if index < 0 {
                return;
            }
            host_state
                .app
                .set_reference_image_scale(index as usize, value);
        }
        ReferenceImageAction::RemoveReferenceImage => {
            if index < 0 {
                return;
            }
            host_state.queue_action(Action::RemoveReferenceImage(index as usize));
        }
    }
}
