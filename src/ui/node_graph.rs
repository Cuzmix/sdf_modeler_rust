use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_WIDTH: f32 = 140.0;
const NODE_HEIGHT: f32 = 48.0;
const PORT_RADIUS: f32 = 5.0;
const COL_SPACING: f32 = 200.0;
const ROW_SPACING: f32 = 70.0;

const COLOR_BG: Color32 = Color32::from_rgb(30, 30, 38);
const COLOR_NODE: Color32 = Color32::from_rgb(50, 50, 58);
const COLOR_NODE_SEL: Color32 = Color32::from_rgb(60, 60, 80);
const COLOR_PRIM_BADGE: Color32 = Color32::from_rgb(70, 130, 180);
const COLOR_OP_BADGE: Color32 = Color32::from_rgb(200, 120, 50);
const COLOR_PORT_OUT: Color32 = Color32::from_rgb(100, 200, 100);
const COLOR_PORT_IN: Color32 = Color32::from_rgb(200, 100, 100);
const COLOR_WIRE: Color32 = Color32::from_rgb(160, 160, 180);
const COLOR_WIRE_DRAG: Color32 = Color32::from_rgb(100, 180, 255);
const COLOR_SEL_BORDER: Color32 = Color32::from_rgb(255, 200, 60);
const COLOR_SCULPT_BADGE: Color32 = Color32::from_rgb(150, 100, 200);

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum DragState {
    None,
    MovingNode(NodeId),
    WireDrag {
        from_node: NodeId,
        from_port: PortKind,
    },
    Panning,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PortKind {
    Output,
    InputLeft,
    InputRight,
    InputSingle,
}

pub struct NodeGraphState {
    pub node_positions: HashMap<NodeId, Pos2>,
    pub pan_offset: Vec2,
    pub selected: Option<NodeId>,
    pub drag: DragState,
    pub layout_dirty: bool,
    pub pinned_positions: HashSet<NodeId>,
    pub needs_center: bool,
}

impl NodeGraphState {
    pub fn new() -> Self {
        Self {
            node_positions: HashMap::new(),
            pan_offset: Vec2::ZERO,
            selected: None,
            drag: DragState::None,
            layout_dirty: true,
            pinned_positions: HashSet::new(),
            needs_center: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-layout
// ---------------------------------------------------------------------------

fn compute_depth(scene: &Scene, id: NodeId, cache: &mut HashMap<NodeId, u32>) -> u32 {
    if let Some(&d) = cache.get(&id) {
        return d;
    }
    let depth = match scene.nodes.get(&id).map(|n| &n.data) {
        Some(NodeData::Operation { left, right, .. }) => {
            let ld = compute_depth(scene, *left, cache);
            let rd = compute_depth(scene, *right, cache);
            1 + ld.max(rd)
        }
        Some(NodeData::Sculpt { input, .. }) => {
            1 + compute_depth(scene, *input, cache)
        }
        _ => 0,
    };
    cache.insert(id, depth);
    depth
}

fn auto_layout(scene: &Scene, state: &mut NodeGraphState) {
    let mut depth_cache: HashMap<NodeId, u32> = HashMap::new();

    // Compute depth for all nodes
    for &id in scene.nodes.keys() {
        compute_depth(scene, id, &mut depth_cache);
    }

    // Group by column
    let mut columns: HashMap<u32, Vec<NodeId>> = HashMap::new();
    for (&id, &col) in &depth_cache {
        columns.entry(col).or_default().push(id);
    }
    // Sort each column by id for stable ordering
    for col in columns.values_mut() {
        col.sort();
    }

    // Assign positions
    for (&col, nodes) in &columns {
        for (row, &id) in nodes.iter().enumerate() {
            if !state.pinned_positions.contains(&id) {
                let x = col as f32 * COL_SPACING;
                let y = row as f32 * ROW_SPACING;
                state.node_positions.insert(id, Pos2::new(x, y));
            }
        }
    }

    state.layout_dirty = false;
}

// ---------------------------------------------------------------------------
// Port position helpers
// ---------------------------------------------------------------------------

fn output_port_pos(node_pos: Pos2, pan: Vec2) -> Pos2 {
    let screen = node_pos + pan;
    Pos2::new(screen.x + NODE_WIDTH, screen.y + NODE_HEIGHT / 2.0)
}

fn input_left_port_pos(node_pos: Pos2, pan: Vec2) -> Pos2 {
    let screen = node_pos + pan;
    Pos2::new(screen.x, screen.y + NODE_HEIGHT * 0.33)
}

fn input_right_port_pos(node_pos: Pos2, pan: Vec2) -> Pos2 {
    let screen = node_pos + pan;
    Pos2::new(screen.x, screen.y + NODE_HEIGHT * 0.67)
}

fn input_single_port_pos(node_pos: Pos2, pan: Vec2) -> Pos2 {
    let screen = node_pos + pan;
    Pos2::new(screen.x, screen.y + NODE_HEIGHT * 0.5)
}

fn node_screen_rect(node_pos: Pos2, pan: Vec2) -> Rect {
    let tl = node_pos + pan;
    Rect::from_min_size(tl, Vec2::new(NODE_WIDTH, NODE_HEIGHT))
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

fn draw_bezier(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    let cp_offset = (to.x - from.x).abs() * 0.5;
    let cp1 = Pos2::new(from.x + cp_offset, from.y);
    let cp2 = Pos2::new(to.x - cp_offset, to.y);
    let bezier = egui::epaint::CubicBezierShape::from_points_stroke(
        [from, cp1, cp2, to],
        false,
        Color32::TRANSPARENT,
        Stroke::new(2.0, color),
    );
    painter.add(bezier);
}

fn node_type_label(data: &NodeData) -> &str {
    match data {
        NodeData::Primitive { kind, .. } => kind.base_name(),
        NodeData::Operation { op, .. } => op.base_name(),
        NodeData::Sculpt { .. } => "Sculpt",
    }
}

fn badge_color(data: &NodeData) -> Color32 {
    match data {
        NodeData::Primitive { .. } => COLOR_PRIM_BADGE,
        NodeData::Operation { .. } => COLOR_OP_BADGE,
        NodeData::Sculpt { .. } => COLOR_SCULPT_BADGE,
    }
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

fn draw_toolbar(ui: &mut egui::Ui, scene: &mut Scene, state: &mut NodeGraphState) {
    ui.horizontal(|ui| {
        ui.style_mut().spacing.button_padding = Vec2::new(4.0, 2.0);

        // Primitive buttons
        for kind in [
            SdfPrimitive::Sphere,
            SdfPrimitive::Box,
            SdfPrimitive::Cylinder,
            SdfPrimitive::Torus,
            SdfPrimitive::Cone,
            SdfPrimitive::Capsule,
        ] {
            let label = format!("+{}", kind.base_name());
            if ui.small_button(&label).clicked() {
                scene.create_primitive(kind);
                state.layout_dirty = true;
                state.needs_center = true;
            }
        }
        ui.separator();

        // Operation buttons
        for op in [
            CsgOp::Union,
            CsgOp::SmoothUnion,
            CsgOp::Subtract,
            CsgOp::Intersect,
        ] {
            let label = format!("+{}", op.base_name());
            if ui.small_button(&label).clicked() {
                create_op_from_selection(scene, state, op);
            }
        }
        ui.separator();

        if ui.small_button("Delete").clicked() {
            if let Some(sel) = state.selected {
                scene.remove_node(sel);
                state.selected = None;
                state.layout_dirty = true;
                state.pinned_positions.remove(&sel);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Connections (wires between nodes)
// ---------------------------------------------------------------------------

fn draw_connections(
    painter: &egui::Painter,
    node_snapshot: &[(NodeId, NodeData)],
    state: &NodeGraphState,
    pan: Vec2,
) {
    for (id, data) in node_snapshot {
        match data {
            NodeData::Operation { left, right, .. } => {
                if let (Some(&left_pos), Some(&op_pos)) =
                    (state.node_positions.get(left), state.node_positions.get(id))
                {
                    draw_bezier(
                        painter,
                        output_port_pos(left_pos, pan),
                        input_left_port_pos(op_pos, pan),
                        COLOR_WIRE,
                    );
                }
                if let (Some(&right_pos), Some(&op_pos)) =
                    (state.node_positions.get(right), state.node_positions.get(id))
                {
                    draw_bezier(
                        painter,
                        output_port_pos(right_pos, pan),
                        input_right_port_pos(op_pos, pan),
                        COLOR_WIRE,
                    );
                }
            }
            NodeData::Sculpt { input, .. } => {
                if let (Some(&input_pos), Some(&sculpt_pos)) =
                    (state.node_positions.get(input), state.node_positions.get(id))
                {
                    draw_bezier(
                        painter,
                        output_port_pos(input_pos, pan),
                        input_single_port_pos(sculpt_pos, pan),
                        COLOR_WIRE,
                    );
                }
            }
            _ => {}
        }
    }

    // Wire drag preview
    if let DragState::WireDrag {
        from_node,
        from_port,
    } = &state.drag
    {
        if let Some(&from_pos) = state.node_positions.get(from_node) {
            let start = match from_port {
                PortKind::Output => output_port_pos(from_pos, pan),
                PortKind::InputLeft => input_left_port_pos(from_pos, pan),
                PortKind::InputRight => input_right_port_pos(from_pos, pan),
                PortKind::InputSingle => input_single_port_pos(from_pos, pan),
            };
            if let Some(mouse) = painter.ctx().pointer_hover_pos() {
                draw_bezier(painter, start, mouse, COLOR_WIRE_DRAG);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single node card
// ---------------------------------------------------------------------------

fn draw_node_card(
    painter: &egui::Painter,
    scene: &Scene,
    id: NodeId,
    data: &NodeData,
    node_pos: Pos2,
    pan: Vec2,
    is_selected: bool,
) -> Rect {
    let rect = node_screen_rect(node_pos, pan);

    // Node body
    let bg = if is_selected { COLOR_NODE_SEL } else { COLOR_NODE };
    painter.rect_filled(rect, 4.0, bg);
    if is_selected {
        painter.rect_stroke(rect, 4.0, Stroke::new(2.0, COLOR_SEL_BORDER));
    } else {
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_rgb(70, 70, 80)));
    }

    // Badge bar
    let badge_rect = Rect::from_min_size(rect.min, Vec2::new(NODE_WIDTH, 18.0));
    painter.rect_filled(
        badge_rect,
        egui::Rounding { nw: 4.0, ne: 4.0, sw: 0.0, se: 0.0 },
        badge_color(data),
    );

    // Type label
    painter.text(
        badge_rect.center(),
        egui::Align2::CENTER_CENTER,
        node_type_label(data),
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );

    // Node name
    let name = scene.nodes.get(&id).map(|n| n.name.as_str()).unwrap_or("?");
    painter.text(
        Pos2::new(rect.center().x, rect.min.y + 32.0),
        egui::Align2::CENTER_CENTER,
        name,
        egui::FontId::proportional(10.0),
        Color32::from_rgb(200, 200, 210),
    );

    // Output port (all nodes)
    painter.circle_filled(output_port_pos(node_pos, pan), PORT_RADIUS, COLOR_PORT_OUT);

    // Input ports
    match data {
        NodeData::Operation { .. } => {
            painter.circle_filled(input_left_port_pos(node_pos, pan), PORT_RADIUS, COLOR_PORT_IN);
            painter.circle_filled(input_right_port_pos(node_pos, pan), PORT_RADIUS, COLOR_PORT_IN);
        }
        NodeData::Sculpt { .. } => {
            painter.circle_filled(input_single_port_pos(node_pos, pan), PORT_RADIUS, COLOR_PORT_IN);
        }
        _ => {}
    }

    rect
}

// ---------------------------------------------------------------------------
// Interaction handling
// ---------------------------------------------------------------------------

fn handle_interaction(
    response: &egui::Response,
    scene: &mut Scene,
    state: &mut NodeGraphState,
    node_rects: &[(NodeId, Rect)],
    node_snapshot: &[(NodeId, NodeData)],
    pan: Vec2,
) {
    let pointer = response
        .interact_pointer_pos()
        .or_else(|| response.hover_pos());

    // Drag started: check ports first, then node bodies
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(pos) = pointer {
            let mut handled = false;

            // Check ports (output ports for wire dragging)
            for (id, data) in node_snapshot {
                let Some(&np) = state.node_positions.get(id) else {
                    continue;
                };
                let out = output_port_pos(np, pan);
                if pos.distance(out) < PORT_RADIUS * 3.0 {
                    state.drag = DragState::WireDrag {
                        from_node: *id,
                        from_port: PortKind::Output,
                    };
                    handled = true;
                    break;
                }
                // Input ports (for rewiring)
                if matches!(data, NodeData::Operation { .. }) {
                    let in_l = input_left_port_pos(np, pan);
                    let in_r = input_right_port_pos(np, pan);
                    if pos.distance(in_l) < PORT_RADIUS * 3.0 {
                        state.drag = DragState::WireDrag {
                            from_node: *id,
                            from_port: PortKind::InputLeft,
                        };
                        handled = true;
                        break;
                    }
                    if pos.distance(in_r) < PORT_RADIUS * 3.0 {
                        state.drag = DragState::WireDrag {
                            from_node: *id,
                            from_port: PortKind::InputRight,
                        };
                        handled = true;
                        break;
                    }
                }
                if matches!(data, NodeData::Sculpt { .. }) {
                    let in_s = input_single_port_pos(np, pan);
                    if pos.distance(in_s) < PORT_RADIUS * 3.0 {
                        state.drag = DragState::WireDrag {
                            from_node: *id,
                            from_port: PortKind::InputSingle,
                        };
                        handled = true;
                        break;
                    }
                }
            }

            // Check node bodies (iterate in reverse draw order for z-priority)
            if !handled {
                for &(id, rect) in node_rects.iter().rev() {
                    if rect.contains(pos) {
                        state.selected = Some(id);
                        state.drag = DragState::MovingNode(id);
                        handled = true;
                        break;
                    }
                }
            }

            // Click empty space: deselect
            if !handled {
                state.selected = None;
            }
        }
    }

    // Dragging: apply movement
    if response.dragged_by(egui::PointerButton::Primary) {
        match &state.drag {
            DragState::MovingNode(id) => {
                let id = *id;
                if let Some(pos) = state.node_positions.get_mut(&id) {
                    *pos += response.drag_delta();
                    state.pinned_positions.insert(id);
                }
            }
            DragState::WireDrag { .. } => {
                // Preview wire drawn above — nothing else to do here
            }
            _ => {}
        }
    }

    // Drag ended: complete wire connection
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        if let DragState::WireDrag {
            from_node,
            from_port,
        } = &state.drag
        {
            let from_node = *from_node;
            let from_port = from_port.clone();
            if let Some(pos) = pointer {
                try_complete_wire(scene, state, node_snapshot, from_node, &from_port, pos);
            }
        }
        state.drag = DragState::None;
    }

    // Simple click (no drag): select node or deselect
    if response.clicked() {
        if let Some(pos) = pointer {
            let mut clicked_node = false;
            for &(id, rect) in node_rects.iter().rev() {
                if rect.contains(pos) {
                    state.selected = Some(id);
                    clicked_node = true;
                    break;
                }
            }
            if !clicked_node {
                state.selected = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main draw function
// ---------------------------------------------------------------------------

pub fn draw(ui: &mut egui::Ui, scene: &mut Scene, state: &mut NodeGraphState) {
    draw_toolbar(ui, scene, state);

    // Canvas setup
    let canvas_rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(canvas_rect, egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas_rect);
    painter.rect_filled(canvas_rect, 0.0, COLOR_BG);

    if state.layout_dirty {
        auto_layout(scene, state);
    }

    // Auto-center graph in canvas after first layout
    if state.needs_center && !state.node_positions.is_empty() {
        let mut min = Pos2::new(f32::MAX, f32::MAX);
        let mut max = Pos2::new(f32::MIN, f32::MIN);
        for pos in state.node_positions.values() {
            min.x = min.x.min(pos.x);
            min.y = min.y.min(pos.y);
            max.x = max.x.max(pos.x + NODE_WIDTH);
            max.y = max.y.max(pos.y + NODE_HEIGHT);
        }
        let graph_center = Pos2::new((min.x + max.x) / 2.0, (min.y + max.y) / 2.0);
        let canvas_center = canvas_rect.center();
        state.pan_offset = Vec2::new(
            canvas_center.x - graph_center.x,
            canvas_center.y - graph_center.y,
        );
        state.needs_center = false;
    }

    // Pan: right-drag, middle-drag, or scroll
    if response.dragged_by(egui::PointerButton::Secondary)
        || response.dragged_by(egui::PointerButton::Middle)
    {
        if !matches!(state.drag, DragState::WireDrag { .. }) {
            state.pan_offset += response.drag_delta();
        }
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta);
        if scroll != Vec2::ZERO {
            state.pan_offset += scroll;
        }
    }

    let pan = state.pan_offset;

    // Snapshot node data for drawing (avoids borrow conflicts)
    let node_snapshot: Vec<(NodeId, NodeData)> = scene
        .nodes
        .values()
        .map(|n| (n.id, n.data.clone()))
        .collect();

    draw_connections(&painter, &node_snapshot, state, pan);

    // Draw node cards + collect rects for hit testing
    let mut node_rects: Vec<(NodeId, Rect)> = Vec::new();
    for (id, data) in &node_snapshot {
        let Some(&node_pos) = state.node_positions.get(id) else {
            continue;
        };
        let rect = node_screen_rect(node_pos, pan);
        if !canvas_rect.intersects(rect) {
            continue;
        }
        let is_selected = state.selected == Some(*id);
        draw_node_card(&painter, scene, *id, data, node_pos, pan, is_selected);
        node_rects.push((*id, rect));
    }

    handle_interaction(&response, scene, state, &node_rects, &node_snapshot, pan);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_op_from_selection(scene: &mut Scene, state: &mut NodeGraphState, op: CsgOp) {
    let mut prim_ids: Vec<NodeId> = scene
        .nodes
        .values()
        .filter(|n| matches!(n.data, NodeData::Primitive { .. }))
        .map(|n| n.id)
        .collect();
    prim_ids.sort();

    if prim_ids.len() >= 2 {
        let left = prim_ids[prim_ids.len() - 2];
        let right = prim_ids[prim_ids.len() - 1];
        let op_id = scene.create_operation(op, left, right);
        state.selected = Some(op_id);
        state.layout_dirty = true;
    }
}

fn try_complete_wire(
    scene: &mut Scene,
    state: &mut NodeGraphState,
    node_snapshot: &[(NodeId, NodeData)],
    from_node: NodeId,
    from_port: &PortKind,
    release_pos: Pos2,
) {
    let pan = state.pan_offset;

    for (id, data) in node_snapshot {
        let Some(&np) = state.node_positions.get(id) else {
            continue;
        };

        match from_port {
            PortKind::Output => {
                if *id == from_node {
                    continue;
                }
                // Dragged from output → looking for input ports on operations
                if matches!(data, NodeData::Operation { .. }) {
                    let in_l = input_left_port_pos(np, pan);
                    let in_r = input_right_port_pos(np, pan);
                    if release_pos.distance(in_l) < PORT_RADIUS * 4.0 {
                        scene.set_left_child(*id, from_node);
                        state.layout_dirty = true;
                        return;
                    }
                    if release_pos.distance(in_r) < PORT_RADIUS * 4.0 {
                        scene.set_right_child(*id, from_node);
                        state.layout_dirty = true;
                        return;
                    }
                }
                // Dragged from output → looking for Sculpt input port
                if matches!(data, NodeData::Sculpt { .. }) {
                    let in_s = input_single_port_pos(np, pan);
                    if release_pos.distance(in_s) < PORT_RADIUS * 4.0 {
                        scene.set_sculpt_input(*id, from_node);
                        state.layout_dirty = true;
                        return;
                    }
                }
            }
            PortKind::InputLeft | PortKind::InputRight => {
                // Dragged from input → looking for output ports on any node
                if *id != from_node {
                    let out = output_port_pos(np, pan);
                    if release_pos.distance(out) < PORT_RADIUS * 4.0 {
                        match from_port {
                            PortKind::InputLeft => scene.set_left_child(from_node, *id),
                            PortKind::InputRight => scene.set_right_child(from_node, *id),
                            _ => {}
                        }
                        state.layout_dirty = true;
                        return;
                    }
                }
            }
            PortKind::InputSingle => {
                // Dragged from Sculpt input → looking for output ports
                if *id != from_node {
                    let out = output_port_pos(np, pan);
                    if release_pos.distance(out) < PORT_RADIUS * 4.0 {
                        scene.set_sculpt_input(from_node, *id);
                        state.layout_dirty = true;
                        return;
                    }
                }
            }
        }
    }
}
