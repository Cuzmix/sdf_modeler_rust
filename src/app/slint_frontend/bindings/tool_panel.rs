use crate::app::frontend_models::{ShellSnapshot, ToolPanelMode};
use crate::app::slint_frontend::{ToolPanelModeView, ToolPanelState};

use super::property_sections::{
    light_section, material_section, operation_section, sculpt_section, transform_section,
};

pub(super) fn build_tool_panel_state(snapshot: &ShellSnapshot) -> ToolPanelState {
    ToolPanelState {
        title: snapshot.tool_panel.title.clone().into(),
        mode: match snapshot.tool_panel.mode {
            ToolPanelMode::Select => ToolPanelModeView::Select,
            ToolPanelMode::Sculpt => ToolPanelModeView::Sculpt,
        },
        summary: snapshot.tool_panel.summary.clone().into(),
        empty_state: snapshot.tool_panel.empty_state.clone().into(),
        show_sculpt_target_fields: snapshot.tool_panel.show_sculpt_target_fields,
        transform: transform_section(snapshot.tool_panel.transform.as_ref()),
        material: material_section(snapshot.tool_panel.material.as_ref()),
        operation: operation_section(snapshot.tool_panel.operation.as_ref()),
        sculpt: sculpt_section(snapshot.tool_panel.sculpt.as_ref()),
        light: light_section(snapshot.tool_panel.light.as_ref()),
    }
}
