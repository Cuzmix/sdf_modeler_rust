use crate::app::frontend_models::ShellSnapshot;
use crate::app::slint_frontend::{SlintHostWindow, TopBarState};

pub(super) fn build_top_bar_snapshot(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> TopBarState {
    let mut state = window.get_top_bar_state();
    state.viewport_status = format!(
        "{} / {} / {}",
        snapshot.viewport_status.interaction_label,
        snapshot.viewport_status.transform_label,
        snapshot.viewport_status.space_label
    )
    .into();
    state
}
