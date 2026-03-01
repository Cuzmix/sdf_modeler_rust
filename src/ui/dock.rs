use eframe::egui;
use egui_dock::{DockState, Node, NodeIndex, Split, TabViewer};
use glam::Vec3;

use crate::app::actions::ActionSink;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::{ActiveTool, SculptState};
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::{self, NodeGraphState};
use crate::settings::Settings;
use crate::ui::{properties, render_settings, scene_tree, viewport};

#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    Viewport,
    NodeGraph,
    Properties,
    SceneTree,
    RenderSettings,
}

pub fn create_dock_state() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();

    let [center, right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.8,
        Node::leaf_with(vec![Tab::Properties, Tab::RenderSettings]),
    );

    let [_props, _tree] = surface.split(
        right,
        Split::Below,
        0.5,
        Node::leaf(Tab::SceneTree),
    );

    let [_viewport, _graph] = surface.split(
        center,
        Split::Below,
        0.7,
        Node::leaf(Tab::NodeGraph),
    );

    state
}

// ---------------------------------------------------------------------------
// Context bundles — group related refs so SdfTabViewer stays manageable.
// ---------------------------------------------------------------------------

/// Refs needed only by the viewport tab.
pub struct ViewportContext<'a> {
    pub gizmo_state: &'a mut GizmoState,
    pub gizmo_mode: &'a GizmoMode,
    pub gizmo_space: &'a GizmoSpace,
    pub pivot_offset: &'a mut Vec3,
    pub pending_pick: &'a mut Option<PendingPick>,
    pub sculpt_count: usize,
    pub fps_info: Option<(f64, f64)>,
}

/// Refs needed only by the scene tree tab.
pub struct SceneTreeContext<'a> {
    pub renaming_node: &'a mut Option<NodeId>,
    pub rename_buf: &'a mut String,
    pub drag_state: &'a mut Option<NodeId>,
}

// ---------------------------------------------------------------------------
// Tab viewer
// ---------------------------------------------------------------------------

pub struct SdfTabViewer<'a> {
    pub camera: &'a mut Camera,
    pub scene: &'a mut Scene,
    pub node_graph_state: &'a mut NodeGraphState,
    pub active_tool: &'a ActiveTool,
    pub sculpt_state: &'a mut SculptState,
    pub settings: &'a mut Settings,
    pub time: f32,
    /// (done_slices, total_slices) when a bake is in progress, None when idle.
    pub bake_progress: Option<(u32, u32)>,
    /// Viewport-specific refs (gizmo, pick, fps overlay).
    pub viewport: ViewportContext<'a>,
    /// Scene tree-specific refs (rename, drag & drop).
    pub scene_tree: SceneTreeContext<'a>,
    /// Action sink — structural mutations flow through here.
    pub actions: &'a mut ActionSink,
}

impl<'a> TabViewer for SdfTabViewer<'a> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "Viewport".into(),
            Tab::NodeGraph => "Node Graph".into(),
            Tab::Properties => "Properties".into(),
            Tab::SceneTree => "Scene Tree".into(),
            Tab::RenderSettings => "Render Settings".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewport => {
                let vp_output = viewport::draw(
                    ui,
                    self.camera,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.viewport.gizmo_state,
                    self.viewport.gizmo_mode,
                    self.viewport.gizmo_space,
                    self.viewport.pivot_offset,
                    self.sculpt_state,
                    self.active_tool,
                    self.time,
                    &self.settings.render,
                    self.viewport.sculpt_count,
                    self.viewport.fps_info,
                    self.actions,
                );
                if let Some(pick) = vp_output.pending_pick {
                    *self.viewport.pending_pick = Some(pick);
                }
            }
            Tab::NodeGraph => {
                node_graph::draw(ui, self.scene, self.node_graph_state, self.actions);
            }
            Tab::Properties => {
                properties::draw(
                    ui,
                    self.scene,
                    self.node_graph_state.selected,
                    self.sculpt_state,
                    self.bake_progress,
                    self.actions,
                );
                // Defensive: clear selection if the node was deleted by properties panel
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.needs_initial_rebuild = true;
                    }
                }
            }
            Tab::SceneTree => {
                let prev_selected = self.node_graph_state.selected;
                scene_tree::draw(
                    ui,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.scene_tree.renaming_node,
                    self.scene_tree.rename_buf,
                    self.scene_tree.drag_state,
                    self.actions,
                );
                // If scene tree changed selection, scroll graph to it
                if self.node_graph_state.selected != prev_selected {
                    if let Some(sel) = self.node_graph_state.selected {
                        self.node_graph_state.pending_center_node = Some(sel);
                    }
                }
                // Defensive: mark layout dirty if a node was deleted via context menu
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.needs_initial_rebuild = true;
                    }
                }
            }
            Tab::RenderSettings => {
                render_settings::draw(ui, self.settings, self.actions);
            }
        }
    }
}
