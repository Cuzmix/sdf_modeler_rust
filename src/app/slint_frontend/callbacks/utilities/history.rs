use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{HistoryAction, SlintHostWindow};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_history_action(move |action, index| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_history_action(host_state, action, index);
        });
    });
}

fn handle_history_action(host_state: &mut SlintHostState, action: HistoryAction, index: i32) {
    if !matches!(action, HistoryAction::JumpToEntry) || index < 0 {
        return;
    }
    let Some(snapshot) = host_state.last_snapshot.as_ref() else {
        return;
    };
    let Some(entry) = snapshot.utility.history_rows.get(index as usize) else {
        return;
    };
    if !entry.jump_enabled || entry.jump_steps == 0 {
        return;
    }
    let repeated = if entry.direction_label == "Undo" {
        Action::Undo
    } else {
        Action::Redo
    };
    for _ in 0..entry.jump_steps {
        host_state.queue_action(repeated.clone());
    }
}
