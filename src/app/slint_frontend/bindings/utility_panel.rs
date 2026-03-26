use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{ReferenceImageRow, ShellSnapshot};
use crate::app::slint_frontend::{
    ReferenceRowView, RenderSettingsState, SlintHostWindow, UtilityPanelState,
};

pub(super) fn build_utility_panel_snapshot(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> UtilityPanelState {
    let reference_rows = snapshot
        .utility
        .reference_rows
        .iter()
        .map(reference_row_view)
        .collect::<Vec<_>>();
    let render = &snapshot.utility.render_settings;

    UtilityPanelState {
        reference_rows: Rc::new(VecModel::from(reference_rows)).into(),
        render_settings: RenderSettingsState {
            show_grid: render.show_grid,
            show_node_labels: render.show_node_labels,
            show_bounding_box: render.show_bounding_box,
            show_light_gizmos: render.show_light_gizmos,
            shadows_enabled: render.shadows_enabled,
            ao_enabled: render.ao_enabled,
            export_resolution: render.export_resolution as i32,
            adaptive_export: render.adaptive_export,
        },
        import_dialog: window.get_utility_panel_state().import_dialog,
    }
}

fn reference_row_view(entry: &ReferenceImageRow) -> ReferenceRowView {
    ReferenceRowView {
        label: entry.label.clone().into(),
        plane_label: entry.plane_label.clone().into(),
        visible: entry.visible,
        locked: entry.locked,
        opacity: entry.opacity,
        scale: entry.scale,
    }
}
