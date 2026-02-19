use eframe::egui;
use egui_dock::{DockState, Node, NodeIndex, Split, TabViewer};

use crate::gpu::camera::Camera;
use crate::graph::scene::Scene;
use crate::ui::node_graph::{self, NodeGraphState};
use crate::ui::{properties, scene_tree, viewport};

#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    Viewport,
    NodeGraph,
    Properties,
    SceneTree,
}

pub fn create_dock_state() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();

    let [center, right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.8,
        Node::leaf(Tab::Properties),
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
    pub time: f32,
}

impl<'a> TabViewer for SdfTabViewer<'a> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "Viewport".into(),
            Tab::NodeGraph => "Node Graph".into(),
            Tab::Properties => "Properties".into(),
            Tab::SceneTree => "Scene Tree".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewport => {
                viewport::draw(ui, self.camera, self.time);
            }
            Tab::NodeGraph => {
                node_graph::draw(ui, self.scene, self.node_graph_state);
            }
            Tab::Properties => {
                properties::draw(ui, self.scene, self.node_graph_state.selected);
            }
            Tab::SceneTree => {
                scene_tree::draw(ui, self.scene, &mut self.node_graph_state.selected);
            }
        }
    }
}
