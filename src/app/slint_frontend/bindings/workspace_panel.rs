use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{ShellSnapshot, WorkspaceSummaryEntry};
use crate::app::slint_frontend::{WorkspacePanelState, WorkspaceSummaryRowView};

pub(super) fn build_workspace_panel_state(snapshot: &ShellSnapshot) -> WorkspacePanelState {
    WorkspacePanelState {
        visible: snapshot.workspace.visible,
        route_label: snapshot.workspace.route_label.clone().into(),
        selection_summary: snapshot.workspace.selection_summary.clone().into(),
        detail_text: snapshot.workspace.detail_text.clone().into(),
        node_graph_active: matches!(
            snapshot.workspace.route,
            crate::app::state::WorkspaceRoute::NodeGraph
        ),
        light_graph_active: matches!(
            snapshot.workspace.route,
            crate::app::state::WorkspaceRoute::LightGraph
        ),
        context_rows: rows_to_model(&snapshot.workspace.context_rows),
        input_rows: rows_to_model(&snapshot.workspace.input_rows),
        output_rows: rows_to_model(&snapshot.workspace.output_rows),
    }
}

fn rows_to_model(rows: &[WorkspaceSummaryEntry]) -> slint::ModelRc<WorkspaceSummaryRowView> {
    Rc::new(VecModel::from(
        rows.iter()
            .map(|row| WorkspaceSummaryRowView {
                label: row.label.clone().into(),
                value: row.value.clone().into(),
            })
            .collect::<Vec<_>>(),
    ))
    .into()
}
