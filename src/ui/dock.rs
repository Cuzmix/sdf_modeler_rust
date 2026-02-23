use eframe::egui;
use egui_dock::{DockState, Node, NodeIndex, Split, TabViewer};
use glam::Vec3;

use crate::app::BakeRequest;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::SculptState;
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

pub struct SdfTabViewer<'a> {
    pub camera: &'a mut Camera,
    pub scene: &'a mut Scene,
    pub node_graph_state: &'a mut NodeGraphState,
    pub gizmo_state: &'a mut GizmoState,
    pub gizmo_mode: &'a GizmoMode,
    pub gizmo_space: &'a GizmoSpace,
    pub pivot_offset: &'a mut Vec3,
    pub sculpt_state: &'a mut SculptState,
    pub settings: &'a mut Settings,
    pub settings_dirty: &'a mut bool,
    pub time: f32,
    pub pending_pick: &'a mut Option<PendingPick>,
    pub bake_request: &'a mut Option<BakeRequest>,
    /// (done_slices, total_slices) when a bake is in progress, None when idle.
    pub bake_progress: Option<(u32, u32)>,
    /// Number of sculpt nodes in the scene (for auto step reduction).
    pub sculpt_count: usize,
    /// Node currently being renamed in scene tree (None = not renaming).
    pub renaming_node: &'a mut Option<NodeId>,
    /// Rename text buffer.
    pub rename_buf: &'a mut String,
    /// FPS info for viewport overlay: (fps, frame_ms).
    pub fps_info: Option<(f64, f64)>,
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
                if let Some(pick) = viewport::draw(
                    ui,
                    self.camera,
                    self.scene,
                    self.node_graph_state.selected,
                    self.gizmo_state,
                    self.gizmo_mode,
                    self.gizmo_space,
                    self.pivot_offset,
                    self.sculpt_state,
                    self.time,
                    &self.settings.render,
                    self.sculpt_count,
                    self.fps_info,
                ) {
                    *self.pending_pick = Some(pick);
                }
            }
            Tab::NodeGraph => {
                node_graph::draw(ui, self.scene, self.node_graph_state);
            }
            Tab::Properties => {
                properties::draw(
                    ui,
                    self.scene,
                    self.node_graph_state.selected,
                    self.sculpt_state,
                    self.bake_request,
                    self.bake_progress,
                );
                // Defensive: clear selection if the node was deleted by properties panel
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.layout_dirty = true;
                    }
                }
            }
            Tab::SceneTree => {
                scene_tree::draw(
                    ui,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.renaming_node,
                    self.rename_buf,
                );
                // Defensive: mark layout dirty if a node was deleted via context menu
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.layout_dirty = true;
                    }
                }
            }
            Tab::RenderSettings => {
                if render_settings::draw(ui, self.settings) {
                    *self.settings_dirty = true;
                }
            }
        }
    }
}
