use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{ScenePanelRow, ShellSnapshot};
use crate::app::slint_frontend::{ScenePanelState, SceneRowView};

pub(super) fn build_scene_panel_state(snapshot: &ShellSnapshot) -> ScenePanelState {
    ScenePanelState {
        selection_summary: selection_summary(snapshot).into(),
        scene_filter: snapshot.scene_panel.filter_query.clone().into(),
        drag_summary: drag_summary(snapshot).into(),
        scene_rows: Rc::new(VecModel::from(
            snapshot
                .scene_panel
                .rows
                .iter()
                .map(scene_row_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
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

fn drag_summary(snapshot: &ShellSnapshot) -> String {
    if snapshot.scene_panel.drag_active {
        "Drag source armed. Use a row Drop action on a valid target.".to_string()
    } else {
        String::new()
    }
}

fn scene_row_view(row: &ScenePanelRow) -> SceneRowView {
    SceneRowView {
        label: row.label.clone().into(),
        kind_label: row.kind_label.clone().into(),
        depth: row.depth as i32,
        has_children: row.has_children,
        expanded: row.expanded,
        selected: row.selected,
        hidden: row.hidden,
        locked: row.locked,
        renaming: row.renaming,
        rename_value: row.rename_value.clone().into(),
        dragging: row.dragging,
        drop_allowed: row.drop_allowed,
        drop_target: row.drop_target,
    }
}
