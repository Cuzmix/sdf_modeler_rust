use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{HistoryEntry, ReferenceImageRow, ShellSnapshot};
use crate::app::slint_frontend::{
    HistoryRowView, ReferenceRowView, RenderSettingsState, SlintHostWindow, UtilityPanelState,
};

pub(super) fn build_utility_panel_snapshot(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> UtilityPanelState {
    UtilityPanelState {
        history_summary: snapshot.utility.history_summary.clone().into(),
        history_rows: Rc::new(VecModel::from(
            snapshot
                .utility
                .history_rows
                .iter()
                .map(history_row_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
        reference_summary: snapshot.utility.reference_summary.clone().into(),
        reference_rows: Rc::new(VecModel::from(
            snapshot
                .utility
                .reference_rows
                .iter()
                .map(reference_row_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
        render_settings: render_settings_state(snapshot),
        import_dialog: window.get_utility_panel_state().import_dialog,
    }
}

fn render_settings_state(snapshot: &ShellSnapshot) -> RenderSettingsState {
    let render = &snapshot.utility.render_settings;
    RenderSettingsState {
        show_grid: render.show_grid,
        show_node_labels: render.show_node_labels,
        show_bounding_box: render.show_bounding_box,
        show_light_gizmos: render.show_light_gizmos,
        shadows_enabled: render.shadows_enabled,
        ao_enabled: render.ao_enabled,
        export_resolution: render.export_resolution as i32,
        adaptive_export: render.adaptive_export,
        environment_is_hdri: render.environment_is_hdri,
        hdri_path_display: render.hdri_path_display.clone().into(),
        environment_rotation_degrees: render.environment_rotation_degrees,
        environment_exposure: render.environment_exposure,
        environment_bake_resolution: render.environment_bake_resolution as i32,
        background_is_procedural: render.background_is_procedural,
        environment_background_blur: render.environment_background_blur,
    }
}

fn history_row_view(entry: &HistoryEntry) -> HistoryRowView {
    HistoryRowView {
        label: entry.label.clone().into(),
        direction_label: entry.direction_label.clone().into(),
        is_current: entry.is_current,
        jump_enabled: entry.jump_enabled,
    }
}

fn reference_row_view(entry: &ReferenceImageRow) -> ReferenceRowView {
    ReferenceRowView {
        label: entry.label.clone().into(),
        plane_label: entry.plane_label.clone().into(),
        status_label: entry.status_label.clone().into(),
        visible: entry.visible,
        locked: entry.locked,
        opacity: entry.opacity,
        scale: entry.scale,
    }
}
