mod gizmo_overlay;
mod inspector_panel;
mod overlay_layout;
mod panel_framework;
mod property_sections;
mod runtime_state;
mod scene_panel;
mod tool_palette;
mod tool_panel;
mod top_bar;
mod utility_panel;
mod workspace_panel;

use crate::app::frontend_models::ShellSnapshot;
use crate::app::SdfApp;
use crate::gizmo::ViewportGizmoOverlay;

use super::SlintHostWindow;

pub(super) fn apply_shell_snapshot(window: &SlintHostWindow, snapshot: &ShellSnapshot) {
    window.set_top_bar_state(top_bar::build_top_bar_snapshot(window, snapshot));
    window.set_tool_palette_state(tool_palette::build_tool_palette_state(snapshot));
    window.set_overlay_layout_state(overlay_layout::build_overlay_layout_state(snapshot));
    window.set_panel_framework_state(panel_framework::build_panel_framework_state(snapshot));
    window.set_scene_panel_state(scene_panel::build_scene_panel_state(snapshot));
    window.set_inspector_panel_state(inspector_panel::build_inspector_panel_state(snapshot));
    window.set_tool_panel_state(tool_panel::build_tool_panel_state(snapshot));
    window.set_utility_panel_state(utility_panel::build_utility_panel_snapshot(
        window, snapshot,
    ));
    window.set_workspace_panel_state(workspace_panel::build_workspace_panel_state(snapshot));
}

pub(super) fn apply_runtime_ui_state(window: &SlintHostWindow, app: &SdfApp) {
    runtime_state::apply_runtime_ui_state(window, app);
}

pub(super) fn apply_gizmo_overlay(
    window: &SlintHostWindow,
    overlay: Option<&ViewportGizmoOverlay>,
) {
    gizmo_overlay::apply_gizmo_overlay(window, overlay);
}
