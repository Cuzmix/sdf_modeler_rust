use super::super::super::host_state::SlintHostState;
use super::super::{mutate_host_and_tick, CallbackContext};
use crate::app::actions::Action;
use crate::app::reference_images::{RefPlane, ReferenceImageEntry};
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
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.locked = !reference.locked;
        }
        ReferenceImageAction::CycleReferencePlane => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.plane = next_reference_plane(reference.plane);
        }
        ReferenceImageAction::SetReferenceOpacity => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.opacity = value.clamp(0.0, 1.0);
        }
        ReferenceImageAction::SetReferenceScale => {
            let Some(reference) = reference_at(host_state, index) else {
                return;
            };
            reference.scale = value.clamp(0.05, 20.0);
        }
        ReferenceImageAction::RemoveReferenceImage => {
            if index < 0 {
                return;
            }
            host_state.queue_action(Action::RemoveReferenceImage(index as usize));
        }
    }
}

fn reference_at(host_state: &mut SlintHostState, index: i32) -> Option<&mut ReferenceImageEntry> {
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
