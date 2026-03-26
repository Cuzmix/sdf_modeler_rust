use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{HistoryEntry, ScenePanelRow, ShellSnapshot};
use crate::app::slint_frontend::{HistoryRowView, ScenePanelState, SceneRowView, SlintHostWindow};

pub(super) fn build_scene_panel_state(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> ScenePanelState {
    let current_state = window.get_scene_panel_state();
    let scene_rows = snapshot
        .scene_panel
        .rows
        .iter()
        .map(scene_row_view)
        .collect::<Vec<_>>();
    let history_rows = snapshot
        .utility
        .history_rows
        .iter()
        .map(history_row_view)
        .collect::<Vec<_>>();

    ScenePanelState {
        selection_summary: selection_summary(snapshot).into(),
        selected_name: snapshot.inspector.name.clone().into(),
        scene_filter: current_state.scene_filter,
        scene_rows: Rc::new(VecModel::from(scene_rows)).into(),
        history_rows: Rc::new(VecModel::from(history_rows)).into(),
    }
}

fn selection_summary(snapshot: &ShellSnapshot) -> String {
    let row_count = snapshot.scene_panel.rows.len();
    match snapshot.scene_panel.selection_count {
        0 => format!("No selection | {row_count} visible rows"),
        1 => format!("{} | {row_count} visible rows", snapshot.inspector.title),
        count => format!("{count} selected | {row_count} visible rows"),
    }
}

fn scene_row_view(row: &ScenePanelRow) -> SceneRowView {
    SceneRowView {
        label: scene_row_label(row).into(),
        selected: row.selected,
        hidden: row.hidden,
        locked: row.locked,
    }
}

fn scene_row_label(row: &ScenePanelRow) -> String {
    let indent = "  ".repeat(row.depth);
    let visibility = if row.hidden { "[hidden] " } else { "" };
    let locked = if row.locked { "[locked] " } else { "" };
    let selection = if row.selected { "> " } else { "  " };
    format!("{selection}{indent}{visibility}{locked}{}", row.label)
}

fn history_row_view(entry: &HistoryEntry) -> HistoryRowView {
    HistoryRowView {
        label: entry.label.clone().into(),
        is_undo: entry.is_undo,
    }
}
