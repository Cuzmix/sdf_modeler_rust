use crate::app::frontend_models::ShellSnapshot;
use crate::app::slint_frontend::InspectorPanelState;

use super::property_sections::{
    light_section, material_section, operation_section, sculpt_section, transform_section,
};

pub(super) fn build_inspector_panel_state(snapshot: &ShellSnapshot) -> InspectorPanelState {
    InspectorPanelState {
        title: snapshot.inspector.title.clone().into(),
        chips: snapshot.inspector.chips.join(" | ").into(),
        display: join_lines(&snapshot.inspector.display_lines).into(),
        property_summary: join_lines(&snapshot.inspector.property_lines).into(),
        multi_selection_summary: snapshot
            .inspector
            .multi_selection_summary
            .clone()
            .unwrap_or_default()
            .into(),
        transform: transform_section(snapshot.inspector.transform.as_ref()),
        material: material_section(snapshot.inspector.material.as_ref()),
        operation: operation_section(snapshot.inspector.operation.as_ref()),
        sculpt: sculpt_section(snapshot.inspector.sculpt.as_ref()),
        light: light_section(snapshot.inspector.light.as_ref()),
    }
}

fn join_lines(lines: &[String]) -> String {
    lines.join("\n")
}
