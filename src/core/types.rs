use glam::Vec3;

use crate::graph::scene::NodeId;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoreSelection {
    pub primary: Option<NodeId>,
    pub set: std::collections::HashSet<NodeId>,
}

impl CoreSelection {
    pub fn select_single(&mut self, id: NodeId) {
        self.primary = Some(id);
        self.set.clear();
        self.set.insert(id);
    }

    pub fn clear(&mut self) {
        self.primary = None;
        self.set.clear();
    }

    pub fn remove(&mut self, id: NodeId) {
        self.set.remove(&id);
        if self.primary == Some(id) {
            self.primary = self.set.iter().next().copied();
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoreAsyncState {
    pub bake_in_progress: bool,
    pub export_in_progress: bool,
    pub import_in_progress: bool,
    pub pick_in_progress: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreSnapshot {
    pub node_count: usize,
    pub hidden_node_count: usize,
    pub selected_count: usize,
    pub selected_primary: Option<NodeId>,
    pub active_tool: String,
    pub undo_count: usize,
    pub redo_count: usize,
    pub camera_target: Vec3,
    pub camera_distance: f32,
    pub bake_in_progress: bool,
    pub export_in_progress: bool,
    pub import_in_progress: bool,
    pub show_debug: bool,
    pub show_settings: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ViewportInput {
    pub pointer_x: f32,
    pub pointer_y: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub primary_down: bool,
    pub ctrl_held: bool,
    pub shift_held: bool,
    pub pressure: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ViewportOutput {
    pub request_redraw: bool,
    pub consume_primary_press: bool,
    pub pending_pick_node: Option<NodeId>,
    pub hover_world_position: Option<Vec3>,
}
