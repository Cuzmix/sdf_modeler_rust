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
const COLOR_ROOT_BADGE: Color32 = Color32::from_rgb(255, 220, 80);

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
}

pub struct NodeGraphState {
    pub node_positions: HashMap<NodeId, Pos2>,
    pub pan_offset: Vec2,
    pub selected: Option<NodeId>,
    pub drag: DragState,
    pub layout_dirty: bool,
    pub pinned_positions: HashSet<NodeId>,
}

impl NodeGraphState {
    pub fn new() -> Self {
        Self {
            node_positions: HashMap::new(),
            pan_offset: Vec2::new(50.0, 50.0),
            selected: None,
            drag: DragState::None,
            layout_dirty: true,
            pinned_positions: HashSet::new(),
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

fn node_screen_rect(node_pos: Pos2, pan: Vec2) -> Rect {
    let tl = node_pos + pan;
    Rect::from_min_size(tl, Vec2::new(NODE_WIDTH, NODE_HEIGHT))
}

// ---------------------------------------------------------------------------
// Drawing
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
        NodeData::Primitive { kind, .. } => match kind {
            SdfPrimitive::Sphere => "Sphere",
            SdfPrimitive::Box => "Box",
            SdfPrimitive::Cylinder => "Cylinder",
            SdfPrimitive::Torus => "Torus",
            SdfPrimitive::Plane => "Plane",
            SdfPrimitive::Cone => "Cone",
            SdfPrimitive::Capsule => "Capsule",
        },
        NodeData::Operation { op, .. } => match op {
            CsgOp::Union => "Union",
            CsgOp::SmoothUnion => "Smooth U",
            CsgOp::Subtract => "Subtract",
            CsgOp::Intersect => "Intersect",
        },
    }
}

fn badge_color(data: &NodeData) -> Color32 {
    match data {
        NodeData::Primitive { .. } => COLOR_PRIM_BADGE,
        NodeData::Operation { .. } => COLOR_OP_BADGE,
    }
}

// ---------------------------------------------------------------------------
// Main draw function
// ---------------------------------------------------------------------------

pub fn draw(ui: &mut egui::Ui, scene: &mut Scene, state: &mut NodeGraphState) {
    // Toolbar
    ui.horizontal(|ui| {
        ui.style_mut().spacing.button_padding = Vec2::new(4.0, 2.0);
        if ui.small_button("+Sphere").clicked() {
            scene.create_sphere();
            state.layout_dirty = true;
        }
        if ui.small_button("+Box").clicked() {
            scene.create_box();
            state.layout_dirty = true;
        }
        if ui.small_button("+Cyl").clicked() {
            scene.create_cylinder();
            state.layout_dirty = true;
        }
        if ui.small_button("+Torus").clicked() {
            scene.create_torus();
            state.layout_dirty = true;
        }
        if ui.small_button("+Cone").clicked() {
            scene.create_cone();
            state.layout_dirty = true;
        }
        if ui.small_button("+Capsule").clicked() {
            scene.create_capsule();
            state.layout_dirty = true;
        }
        ui.separator();

        // Operation buttons: need two selected/existing primitives
        if ui.small_button("+Union").clicked() {
            create_op_from_selection(scene, state, |s, l, r| s.create_union(l, r));
        }
        if ui.small_button("+SmoothU").clicked() {
            create_op_from_selection(scene, state, |s, l, r| s.create_smooth_union(l, r));
        }
        if ui.small_button("+Sub").clicked() {
            create_op_from_selection(scene, state, |s, l, r| s.create_subtract(l, r));
        }
        if ui.small_button("+Inter").clicked() {
            create_op_from_selection(scene, state, |s, l, r| s.create_intersect(l, r));
        }
        ui.separator();

        if ui.small_button("Set Root").clicked() {
            if let Some(sel) = state.selected {
                scene.root = Some(sel);
            }
        }
        if ui.small_button("Delete").clicked() {
            if let Some(sel) = state.selected {
                scene.remove_node(sel);
                state.selected = None;
                state.layout_dirty = true;
                state.pinned_positions.remove(&sel);
            }
        }
    });

    // Canvas
    let canvas_rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(canvas_rect, egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas_rect);

    // Background
    painter.rect_filled(canvas_rect, 0.0, COLOR_BG);

    // Auto-layout if dirty
    if state.layout_dirty {
        auto_layout(scene, state);
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

    // Draw connections
    let node_snapshot: Vec<(NodeId, NodeData)> = scene
        .nodes
        .values()
        .map(|n| (n.id, n.data.clone()))
        .collect();

    for (id, data) in &node_snapshot {
        if let NodeData::Operation { left, right, .. } = data {
            if let (Some(&left_pos), Some(&op_pos)) =
                (state.node_positions.get(left), state.node_positions.get(id))
            {
                draw_bezier(
                    &painter,
                    output_port_pos(left_pos, pan),
                    input_left_port_pos(op_pos, pan),
                    COLOR_WIRE,
                );
            }
            if let (Some(&right_pos), Some(&op_pos)) =
                (state.node_positions.get(right), state.node_positions.get(id))
            {
                draw_bezier(
                    &painter,
                    output_port_pos(right_pos, pan),
                    input_right_port_pos(op_pos, pan),
                    COLOR_WIRE,
                );
            }
        }
    }

    // Draw wire drag preview
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
            };
            if let Some(mouse) = response.hover_pos() {
                draw_bezier(&painter, start, mouse, COLOR_WIRE_DRAG);
            }
        }
    }

    // Draw node cards + ports
    let mut node_rects: Vec<(NodeId, Rect)> = Vec::new();

    for (id, data) in &node_snapshot {
        let Some(&node_pos) = state.node_positions.get(id) else {
            continue;
        };
        let rect = node_screen_rect(node_pos, pan);
        if !canvas_rect.intersects(rect) {
            continue;
        }
        node_rects.push((*id, rect));

        let is_selected = state.selected == Some(*id);
        let is_root = scene.root == Some(*id);

        // Node body
        let bg = if is_selected { COLOR_NODE_SEL } else { COLOR_NODE };
        painter.rect_filled(rect, 4.0, bg);
        if is_selected {
            painter.rect_stroke(rect, 4.0, Stroke::new(2.0, COLOR_SEL_BORDER));
        } else {
            painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_rgb(70, 70, 80)));
        }

        // Badge bar
        let badge_rect =
            Rect::from_min_size(rect.min, Vec2::new(NODE_WIDTH, 18.0));
        painter.rect_filled(badge_rect, egui::Rounding { nw: 4.0, ne: 4.0, sw: 0.0, se: 0.0 }, badge_color(data));

        // Type label
        let label = node_type_label(data);
        painter.text(
            badge_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(11.0),
            Color32::WHITE,
        );

        // Node name
        let name = scene.nodes.get(id).map(|n| n.name.as_str()).unwrap_or("?");
        painter.text(
            Pos2::new(rect.center().x, rect.min.y + 32.0),
            egui::Align2::CENTER_CENTER,
            name,
            egui::FontId::proportional(10.0),
            Color32::from_rgb(200, 200, 210),
        );

        // Root badge
        if is_root {
            painter.text(
                Pos2::new(rect.max.x - 8.0, rect.min.y + 32.0),
                egui::Align2::RIGHT_CENTER,
                "R",
                egui::FontId::proportional(9.0),
                COLOR_ROOT_BADGE,
            );
        }

        // Output port (all nodes)
        let out_pos = output_port_pos(node_pos, pan);
        painter.circle_filled(out_pos, PORT_RADIUS, COLOR_PORT_OUT);

        // Input ports (operations only)
        if matches!(data, NodeData::Operation { .. }) {
            let in_l = input_left_port_pos(node_pos, pan);
            let in_r = input_right_port_pos(node_pos, pan);
            painter.circle_filled(in_l, PORT_RADIUS, COLOR_PORT_IN);
            painter.circle_filled(in_r, PORT_RADIUS, COLOR_PORT_IN);
        }
    }

    // --- Interaction handling ---

    let pointer = response
        .interact_pointer_pos()
        .or_else(|| response.hover_pos());

    // Drag started: check ports first, then node bodies
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(pos) = pointer {
            let mut handled = false;

            // Check ports (output ports for wire dragging)
            for (id, data) in &node_snapshot {
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
                try_complete_wire(scene, state, &node_snapshot, from_node, &from_port, pos);
            }
        }
        state.drag = DragState::None;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_op_from_selection(
    scene: &mut Scene,
    state: &mut NodeGraphState,
    factory: impl FnOnce(&mut Scene, NodeId, NodeId) -> NodeId,
) {
    // Get two most recently created primitives, or use first two nodes
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
        let op_id = factory(scene, left, right);
        scene.root = Some(op_id);
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
                // Dragged from output → looking for input ports on operations
                if matches!(data, NodeData::Operation { .. }) && *id != from_node {
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
        }
    }
}
