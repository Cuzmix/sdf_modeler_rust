use egui_dock::DockState;

use crate::app::actions::WorkspacePreset;
use crate::app::state::{
    ExpertPanelKind, ExpertPanelRegistry, PrimaryShellState, SceneGraphViewState,
    SceneSelectionState, ShellPanelKind,
};
use crate::app::ui_geometry::FloatingPanelBounds;
use crate::ui::dock::{self, Tab};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::reference_image::EguiReferenceImageCache;

pub(super) struct EguiFrontendState {
    pub dock_state: DockState<Tab>,
    pub node_graph_state: NodeGraphState,
    pub light_graph_state: NodeGraphState,
    pub reference_image_cache: EguiReferenceImageCache,
}

impl Default for EguiFrontendState {
    fn default() -> Self {
        Self {
            dock_state: dock::create_primary_shell_dock(),
            node_graph_state: NodeGraphState::new(),
            light_graph_state: NodeGraphState::new(),
            reference_image_cache: EguiReferenceImageCache::default(),
        }
    }
}

pub(super) const fn dock_tab_for_shell_panel(panel: ShellPanelKind) -> Tab {
    match panel {
        ShellPanelKind::Tool => Tab::ToolPanel,
        ShellPanelKind::Inspector => Tab::InspectorPanel,
        ShellPanelKind::Drawer => Tab::DrawerPanel,
    }
}

pub(super) fn reconcile_docked_panels(
    primary_shell: &mut PrimaryShellState,
    dock_state: &DockState<Tab>,
) {
    for panel in ShellPanelKind::ALL {
        let exists = dock_state
            .find_tab(&dock_tab_for_shell_panel(panel))
            .is_some();
        let shell_state = primary_shell.panel_mut(panel);
        if exists {
            shell_state.dock();
        } else if shell_state.is_docked() {
            shell_state.hide();
        }
    }
}

pub(super) const fn expert_panel_tab(panel: ExpertPanelKind) -> Tab {
    match panel {
        ExpertPanelKind::NodeGraph => Tab::NodeGraph,
        ExpertPanelKind::LightGraph => Tab::LightGraph,
        ExpertPanelKind::Properties => Tab::Properties,
        ExpertPanelKind::ReferenceImages => Tab::ReferenceImages,
        ExpertPanelKind::SceneTree => Tab::SceneTree,
        ExpertPanelKind::RenderSettings => Tab::RenderSettings,
        ExpertPanelKind::History => Tab::History,
        ExpertPanelKind::BrushSettings => Tab::BrushSettings,
        ExpertPanelKind::Lights => Tab::Lights,
        ExpertPanelKind::LightLinking => Tab::LightLinking,
        ExpertPanelKind::SceneStats => Tab::SceneStats,
    }
}

impl EguiFrontendState {
    pub fn prepare_scene_graph(
        &mut self,
        selection: &SceneSelectionState,
        scene_graph_view: &mut SceneGraphViewState,
    ) {
        self.node_graph_state.selected = selection.selected;
        self.node_graph_state.selected_set = selection.selected_set.clone();
        if scene_graph_view.needs_initial_rebuild {
            self.node_graph_state.needs_initial_rebuild = true;
        }
        if scene_graph_view.pending_center_node.is_some() {
            self.node_graph_state.pending_center_node = scene_graph_view.pending_center_node;
        }
        scene_graph_view.needs_initial_rebuild = false;
        scene_graph_view.pending_center_node = None;
    }

    pub fn commit_scene_graph(
        &self,
        selection: &mut SceneSelectionState,
        scene_graph_view: &mut SceneGraphViewState,
    ) {
        selection.selected = self.node_graph_state.selected;
        selection.selected_set = self.node_graph_state.selected_set.clone();
        scene_graph_view.needs_initial_rebuild = self.node_graph_state.needs_initial_rebuild;
        scene_graph_view.pending_center_node = self.node_graph_state.pending_center_node;
    }

    pub fn is_expert_panel_open(&self, panel: ExpertPanelKind) -> bool {
        self.dock_state.find_tab(&expert_panel_tab(panel)).is_some()
    }

    pub fn toggle_expert_panel(
        &mut self,
        expert_panels: &mut ExpertPanelRegistry,
        panel: ExpertPanelKind,
    ) {
        let tab = expert_panel_tab(panel);
        if let Some(location) = self.dock_state.find_tab(&tab) {
            self.dock_state.remove_tab(location);
        } else {
            self.dock_state.push_to_focused_leaf(tab);
        }
        expert_panels.set_open(panel, self.is_expert_panel_open(panel));
    }

    pub fn open_expert_panel(
        &mut self,
        expert_panels: &mut ExpertPanelRegistry,
        panel: ExpertPanelKind,
    ) {
        if !self.is_expert_panel_open(panel) {
            self.dock_state
                .push_to_focused_leaf(expert_panel_tab(panel));
        }
        expert_panels.set_open(panel, true);
    }

    pub fn set_workspace(
        &mut self,
        preset: WorkspacePreset,
        expert_panels: &mut ExpertPanelRegistry,
    ) {
        self.dock_state = match preset {
            WorkspacePreset::Modeling => dock::create_dock_state(),
            WorkspacePreset::Sculpting => dock::create_dock_sculpting(),
            WorkspacePreset::Rendering => dock::create_dock_rendering(),
        };
        self.sync_expert_panel_registry(expert_panels);
    }

    pub fn hide_shell_panel(
        &mut self,
        primary_shell: &mut PrimaryShellState,
        panel: ShellPanelKind,
    ) {
        if let Some(location) = self.dock_state.find_tab(&dock_tab_for_shell_panel(panel)) {
            self.dock_state.remove_tab(location);
        }
        primary_shell.panel_mut(panel).hide();
    }

    pub fn dock_shell_panel(
        &mut self,
        primary_shell: &mut PrimaryShellState,
        panel: ShellPanelKind,
        rect: FloatingPanelBounds,
    ) {
        if let Some(location) = self.dock_state.find_tab(&dock_tab_for_shell_panel(panel)) {
            self.dock_state.remove_tab(location);
        }

        primary_shell.panel_mut(panel).remember_floating_rect(rect);
        dock::add_window_with_rect(&mut self.dock_state, dock_tab_for_shell_panel(panel), rect);
        primary_shell.panel_mut(panel).dock();
    }

    pub fn undock_shell_panel(
        &mut self,
        primary_shell: &mut PrimaryShellState,
        panel: ShellPanelKind,
    ) {
        let fallback_rect = primary_shell.panel(panel).last_floating_rect;

        if let Some((surface, node, tab_index)) =
            self.dock_state.find_tab(&dock_tab_for_shell_panel(panel))
        {
            let detached_rect =
                self.dock_state
                    .get_window_state_mut(surface)
                    .and_then(|window_state| {
                        let rect = window_state.rect();
                        let bounds = FloatingPanelBounds::from_min_size(
                            rect.min.x,
                            rect.min.y,
                            rect.width(),
                            rect.height(),
                        );
                        bounds.is_valid().then_some(bounds)
                    });
            self.dock_state.remove_tab((surface, node, tab_index));
            primary_shell
                .panel_mut(panel)
                .show_floating(detached_rect.or(fallback_rect));
        } else {
            primary_shell.panel_mut(panel).show_floating(fallback_rect);
        }
    }

    pub fn reset_primary_shell_layout(&mut self, primary_shell: &mut PrimaryShellState) {
        primary_shell.reset_layout();
        self.dock_state = dock::create_primary_shell_dock();
    }

    pub fn sync_expert_panel_registry(&self, expert_panels: &mut ExpertPanelRegistry) {
        expert_panels.clear();
        for panel in ExpertPanelKind::ALL {
            expert_panels.set_open(panel, self.is_expert_panel_open(panel));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconcile_docked_panels_hides_missing_docked_tab() {
        let mut shell = PrimaryShellState::default();
        shell.tool_panel.dock();
        let dock_state = DockState::new(vec![Tab::Viewport]);

        reconcile_docked_panels(&mut shell, &dock_state);

        assert!(shell.tool_panel.is_hidden());
    }

    #[test]
    fn sync_expert_panel_registry_tracks_open_tabs() {
        let mut egui = EguiFrontendState::default();
        let mut registry = ExpertPanelRegistry::default();

        egui.toggle_expert_panel(&mut registry, ExpertPanelKind::History);

        assert!(registry.is_open(ExpertPanelKind::History));
    }
}
