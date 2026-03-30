use std::collections::{BTreeMap, HashMap, HashSet};

use crate::graph::scene::{NodeData, NodeId, Scene};

pub const NODE_CARD_WIDTH: f32 = 176.0;
pub const NODE_CARD_HEIGHT: f32 = 78.0;
pub const HANDLE_HIT_RADIUS: f32 = 12.0;
pub const DEFAULT_CANVAS_WIDTH: f32 = 640.0;
pub const DEFAULT_CANVAS_HEIGHT: f32 = 420.0;
pub const MIN_ZOOM: f32 = 0.35;
pub const MAX_ZOOM: f32 = 2.4;

const LAYOUT_X_SPACING: f32 = 280.0;
const LAYOUT_Y_SPACING: f32 = 146.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeSlot {
    Left,
    Right,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeSocketKind {
    Output,
    LeftInput,
    RightInput,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeScreenFrame {
    pub origin: [f32; 2],
    pub size: [f32; 2],
}

impl NodeScreenFrame {
    pub fn contains(self, point: [f32; 2]) -> bool {
        point[0] >= self.origin[0]
            && point[0] <= self.origin[0] + self.size[0]
            && point[1] >= self.origin[1]
            && point[1] <= self.origin[1] + self.size[1]
    }
}

pub fn clamp_zoom(zoom: f32) -> f32 {
    zoom.clamp(MIN_ZOOM, MAX_ZOOM)
}

pub fn grid_gap_for_zoom(zoom: f32) -> f32 {
    (24.0 * clamp_zoom(zoom).clamp(0.5, 1.5)).clamp(14.0, 44.0)
}

pub fn grid_zoom_bucket(zoom: f32) -> i32 {
    (grid_gap_for_zoom(zoom) * 10.0).round() as i32
}

pub fn grid_canvas_bucket(canvas_size: [f32; 2]) -> [u32; 2] {
    [canvas_size[0].max(1.0).round() as u32, canvas_size[1].max(1.0).round() as u32]
}

pub fn build_grid_base_dots(canvas_size: [f32; 2], gap: f32) -> Vec<[f32; 2]> {
    let width = canvas_size[0].max(1.0);
    let height = canvas_size[1].max(1.0);
    let mut dots = Vec::new();
    let mut x = 0.0;
    while x <= width + gap {
        let mut y = 0.0;
        while y <= height + gap {
            dots.push([x, y]);
            y += gap;
        }
        x += gap;
    }
    dots
}

pub fn default_node_positions(scene: &Scene) -> HashMap<NodeId, [f32; 2]> {
    let mut memo = HashMap::<NodeId, usize>::new();
    let mut ids = scene.nodes.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();

    let mut visiting = HashSet::<NodeId>::new();
    for id in &ids {
        let _ = depth_for_node(*id, scene, &mut memo, &mut visiting);
    }

    let mut by_depth = BTreeMap::<usize, Vec<NodeId>>::new();
    for id in ids {
        by_depth.entry(memo[&id]).or_default().push(id);
    }

    let mut positions = HashMap::new();
    for (depth, layer_ids) in by_depth {
        let layer_count = layer_ids.len();
        let layer_height = (layer_count.saturating_sub(1) as f32) * LAYOUT_Y_SPACING;
        for (index, node_id) in layer_ids.into_iter().enumerate() {
            let y = index as f32 * LAYOUT_Y_SPACING - layer_height * 0.5;
            let x = depth as f32 * LAYOUT_X_SPACING;
            positions.insert(node_id, [x, y]);
        }
    }

    positions
}

pub fn fill_missing_node_positions(
    scene: &Scene,
    node_positions: &mut HashMap<NodeId, [f32; 2]>,
) -> bool {
    let has_missing_positions = node_positions.len() < scene.nodes.len()
        || scene
            .nodes
            .keys()
            .any(|node_id| !node_positions.contains_key(node_id));
    if !has_missing_positions {
        return false;
    }

    let previous_len = node_positions.len();
    let defaults = default_node_positions(scene);
    for (node_id, position) in defaults {
        node_positions.entry(node_id).or_insert(position);
    }
    node_positions.len() != previous_len
}

fn depth_for_node(
    id: NodeId,
    scene: &Scene,
    memo: &mut HashMap<NodeId, usize>,
    visiting: &mut HashSet<NodeId>,
) -> usize {
    if let Some(depth) = memo.get(&id).copied() {
        return depth;
    }
    if !visiting.insert(id) {
        return 0;
    }
    let depth = scene
        .nodes
        .get(&id)
        .map(|node| match &node.data {
            NodeData::Operation { left, right, .. } => {
                let left_depth =
                    left.map(|child_id| depth_for_node(child_id, scene, memo, visiting))
                        .unwrap_or(0);
                let right_depth =
                    right.map(|child_id| depth_for_node(child_id, scene, memo, visiting))
                        .unwrap_or(0);
                1 + left_depth.max(right_depth)
            }
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => {
                1 + input
                    .map(|child_id| depth_for_node(child_id, scene, memo, visiting))
                    .unwrap_or(0)
            }
            NodeData::Primitive { .. } | NodeData::Light { .. } => 0,
        })
        .unwrap_or(0);
    visiting.remove(&id);
    memo.insert(id, depth);
    depth
}

pub fn output_handle_position(node_x: f32, node_y: f32) -> [f32; 2] {
    [node_x + NODE_CARD_WIDTH, node_y + NODE_CARD_HEIGHT * 0.5]
}

pub fn single_input_handle_position(node_x: f32, node_y: f32) -> [f32; 2] {
    [node_x, node_y + NODE_CARD_HEIGHT * 0.5]
}

pub fn left_input_handle_position(node_x: f32, node_y: f32) -> [f32; 2] {
    [node_x, node_y + NODE_CARD_HEIGHT * 0.34]
}

pub fn right_input_handle_position(node_x: f32, node_y: f32) -> [f32; 2] {
    [node_x, node_y + NODE_CARD_HEIGHT * 0.66]
}

pub fn input_handle_position_for_slot(node_x: f32, node_y: f32, slot: EdgeSlot) -> [f32; 2] {
    match slot {
        EdgeSlot::Left => left_input_handle_position(node_x, node_y),
        EdgeSlot::Right => right_input_handle_position(node_x, node_y),
        EdgeSlot::Input => single_input_handle_position(node_x, node_y),
    }
}

pub fn screen_from_canvas(point: [f32; 2], pan: [f32; 2], zoom: f32) -> [f32; 2] {
    [pan[0] + point[0] * zoom, pan[1] + point[1] * zoom]
}

pub fn edge_curve_screen(
    parent_canvas: [f32; 2],
    child_canvas: [f32; 2],
    slot: EdgeSlot,
    pan: [f32; 2],
    zoom: f32,
) -> CubicEdgeCurve {
    let parent_frame = project_node_screen_frame(parent_canvas, pan, zoom);
    let child_frame = project_node_screen_frame(child_canvas, pan, zoom);
    edge_curve_from_projected_frames(parent_frame, child_frame, slot)
}

pub fn project_node_screen_frame(node_canvas: [f32; 2], pan: [f32; 2], zoom: f32) -> NodeScreenFrame {
    NodeScreenFrame {
        origin: screen_from_canvas(node_canvas, pan, zoom),
        size: [NODE_CARD_WIDTH * zoom, NODE_CARD_HEIGHT * zoom],
    }
}

pub fn socket_screen_position(frame: NodeScreenFrame, socket: NodeSocketKind) -> [f32; 2] {
    match socket {
        NodeSocketKind::Output => [frame.origin[0] + frame.size[0], frame.origin[1] + frame.size[1] * 0.5],
        NodeSocketKind::LeftInput => [frame.origin[0], frame.origin[1] + frame.size[1] * 0.34],
        NodeSocketKind::RightInput => [frame.origin[0], frame.origin[1] + frame.size[1] * 0.66],
        NodeSocketKind::Input => [frame.origin[0], frame.origin[1] + frame.size[1] * 0.5],
    }
}

pub fn edge_curve_from_projected_frames(
    parent_frame: NodeScreenFrame,
    child_frame: NodeScreenFrame,
    slot: EdgeSlot,
) -> CubicEdgeCurve {
    let input_socket = match slot {
        EdgeSlot::Left => NodeSocketKind::LeftInput,
        EdgeSlot::Right => NodeSocketKind::RightInput,
        EdgeSlot::Input => NodeSocketKind::Input,
    };
    let start = socket_screen_position(child_frame, NodeSocketKind::Output);
    let end = socket_screen_position(parent_frame, input_socket);
    CubicEdgeCurve::new(start, end)
}

pub fn edge_curve_from_screen_frames(
    parent_origin: [f32; 2],
    parent_size: [f32; 2],
    child_origin: [f32; 2],
    child_size: [f32; 2],
    slot: EdgeSlot,
) -> CubicEdgeCurve {
    edge_curve_from_projected_frames(
        NodeScreenFrame {
            origin: parent_origin,
            size: parent_size,
        },
        NodeScreenFrame {
            origin: child_origin,
            size: child_size,
        },
        slot,
    )
}

pub fn node_has_output_handle(_data: &NodeData) -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicEdgeCurve {
    pub start: [f32; 2],
    pub control_a: [f32; 2],
    pub control_b: [f32; 2],
    pub end: [f32; 2],
}

impl CubicEdgeCurve {
    pub fn new(start: [f32; 2], end: [f32; 2]) -> Self {
        let x_span = (end[0] - start[0]).abs();
        let control_offset = (x_span * 0.5).max(42.0);
        Self {
            start,
            control_a: [start[0] + control_offset, start[1]],
            control_b: [end[0] - control_offset, end[1]],
            end,
        }
    }

    pub fn to_path_commands(self) -> String {
        format!(
            "M {:.2} {:.2} C {:.2} {:.2}, {:.2} {:.2}, {:.2} {:.2}",
            self.start[0],
            self.start[1],
            self.control_a[0],
            self.control_a[1],
            self.control_b[0],
            self.control_b[1],
            self.end[0],
            self.end[1]
        )
    }

    pub fn point_at(self, t: f32) -> [f32; 2] {
        let u = 1.0 - t;
        let uu = u * u;
        let tt = t * t;
        let uuu = uu * u;
        let ttt = tt * t;
        let x = uuu * self.start[0]
            + 3.0 * uu * t * self.control_a[0]
            + 3.0 * u * tt * self.control_b[0]
            + ttt * self.end[0];
        let y = uuu * self.start[1]
            + 3.0 * uu * t * self.control_a[1]
            + 3.0 * u * tt * self.control_b[1]
            + ttt * self.end[1];
        [x, y]
    }

    pub fn distance_to_point(self, point: [f32; 2], samples: u32) -> f32 {
        let mut min_distance = f32::INFINITY;
        let mut previous = self.start;
        let sample_count = samples.max(6);
        for step in 1..=sample_count {
            let t = step as f32 / sample_count as f32;
            let current = self.point_at(t);
            let distance = distance_to_segment(point, previous, current);
            min_distance = min_distance.min(distance);
            previous = current;
        }
        min_distance
    }
}

fn distance_to_segment(point: [f32; 2], start: [f32; 2], end: [f32; 2]) -> f32 {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let length_sq = dx * dx + dy * dy;
    if length_sq <= f32::EPSILON {
        return ((point[0] - start[0]).powi(2) + (point[1] - start[1]).powi(2)).sqrt();
    }
    let t = (((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq).clamp(0.0, 1.0);
    let projection = [start[0] + t * dx, start[1] + t * dy];
    ((point[0] - projection[0]).powi(2) + (point[1] - projection[1]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_point_near(actual: [f32; 2], expected: [f32; 2]) {
        let epsilon = 0.001;
        assert!(
            (actual[0] - expected[0]).abs() <= epsilon,
            "x mismatch: actual={}, expected={}",
            actual[0],
            expected[0]
        );
        assert!(
            (actual[1] - expected[1]).abs() <= epsilon,
            "y mismatch: actual={}, expected={}",
            actual[1],
            expected[1]
        );
    }

    #[test]
    fn edge_curve_endpoints_match_handle_centers_for_each_slot() {
        let parent = [180.0, 60.0];
        let child = [20.0, -12.0];
        let pan = [32.0, 18.0];
        let zoom = 1.25;

        for slot in [EdgeSlot::Left, EdgeSlot::Right, EdgeSlot::Input] {
            let curve = edge_curve_screen(parent, child, slot, pan, zoom);
            let expected_start = screen_from_canvas(output_handle_position(child[0], child[1]), pan, zoom);
            let expected_end = screen_from_canvas(
                input_handle_position_for_slot(parent[0], parent[1], slot),
                pan,
                zoom,
            );
            assert_point_near(curve.start, expected_start);
            assert_point_near(curve.end, expected_end);
        }
    }

    #[test]
    fn edge_curve_screen_and_projected_frame_paths_match() {
        let parent = [240.0, -30.0];
        let child = [45.0, 125.0];
        let pan = [18.0, 44.0];
        let zoom = 0.9;
        let parent_frame = project_node_screen_frame(parent, pan, zoom);
        let child_frame = project_node_screen_frame(child, pan, zoom);

        for slot in [EdgeSlot::Left, EdgeSlot::Right, EdgeSlot::Input] {
            let from_canvas = edge_curve_screen(parent, child, slot, pan, zoom);
            let from_frames = edge_curve_from_projected_frames(parent_frame, child_frame, slot);
            assert_eq!(from_canvas.to_path_commands(), from_frames.to_path_commands());
        }
    }

    #[test]
    fn fill_missing_node_positions_preserves_existing_manual_positions() {
        let scene = Scene::new();
        let mut ids = scene.nodes.keys().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        let anchored = ids[0];

        let mut positions = HashMap::new();
        positions.insert(anchored, [512.0, -128.0]);

        assert!(fill_missing_node_positions(&scene, &mut positions));
        assert_eq!(positions[&anchored], [512.0, -128.0]);
        assert_eq!(positions.len(), scene.nodes.len());
    }

    #[test]
    fn fill_missing_node_positions_skips_when_scene_is_already_fully_positioned() {
        let scene = Scene::new();
        let mut positions = default_node_positions(&scene);
        let snapshot = positions.clone();

        assert!(!fill_missing_node_positions(&scene, &mut positions));
        assert_eq!(positions, snapshot);
    }
}
