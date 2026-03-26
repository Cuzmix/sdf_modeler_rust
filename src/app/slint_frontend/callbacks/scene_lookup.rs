use crate::app::frontend_models::{ScenePanelRow, ShellSnapshot};

pub(super) fn scene_row_at(snapshot: Option<&ShellSnapshot>, index: i32) -> Option<&ScenePanelRow> {
    if index < 0 {
        return None;
    }
    snapshot?.scene_panel.rows.get(index as usize)
}
