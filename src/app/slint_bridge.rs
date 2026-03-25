use super::actions::{Action, ActionSink};
use super::backend_frame::{FrameInputSnapshot, UiFrameFeedback};
use super::frontend_models::ShellSnapshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlintUiEvent {
    FrameAll,
    Undo,
    Redo,
    SelectPreviousSceneRow,
    SelectNextSceneRow,
}

pub(super) fn capture_frame_input(now_seconds: f64) -> FrameInputSnapshot {
    FrameInputSnapshot {
        now_seconds,
        pointer_primary_down: false,
        is_dragging_ui: false,
    }
}

pub(super) fn empty_feedback() -> UiFrameFeedback {
    UiFrameFeedback {
        pending_pick: None,
        sculpt_ctrl_held: false,
        sculpt_shift_held: false,
        sculpt_pressure: 0.0,
        is_hover_pick: false,
        gizmo_drag_active: false,
    }
}

pub(super) fn dispatch_event(
    event: SlintUiEvent,
    snapshot: Option<&ShellSnapshot>,
    actions: &mut ActionSink,
) {
    match event {
        SlintUiEvent::FrameAll => actions.push(Action::FrameAll),
        SlintUiEvent::Undo => actions.push(Action::Undo),
        SlintUiEvent::Redo => actions.push(Action::Redo),
        SlintUiEvent::SelectPreviousSceneRow => {
            let Some(shell) = snapshot else {
                return;
            };
            let row_count = shell.scene_panel.rows.len();
            if row_count == 0 {
                return;
            }
            let current_index = selected_scene_index(shell).unwrap_or(0);
            let target_index = if current_index == 0 {
                row_count - 1
            } else {
                current_index - 1
            };
            actions.push(Action::Select(Some(
                shell.scene_panel.rows[target_index].host_id,
            )));
        }
        SlintUiEvent::SelectNextSceneRow => {
            let Some(shell) = snapshot else {
                return;
            };
            let row_count = shell.scene_panel.rows.len();
            if row_count == 0 {
                return;
            }
            let current_index = selected_scene_index(shell).unwrap_or(row_count - 1);
            let target_index = (current_index + 1) % row_count;
            actions.push(Action::Select(Some(
                shell.scene_panel.rows[target_index].host_id,
            )));
        }
    }
}

fn selected_scene_index(snapshot: &ShellSnapshot) -> Option<usize> {
    let selected_host = snapshot.scene_panel.selected_host?;
    snapshot
        .scene_panel
        .rows
        .iter()
        .position(|row| row.host_id == selected_host)
}
