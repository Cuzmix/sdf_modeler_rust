use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick_with_viewport_dirty;
use crate::app::actions::Action;
use crate::app::node_graph::{
    build_grid_base_dots, clamp_zoom, edge_curve_screen, grid_canvas_bucket, grid_gap_for_zoom,
    grid_zoom_bucket, input_handle_position_for_slot, node_has_output_handle, output_handle_position,
    screen_from_canvas, EdgeSlot, HANDLE_HIT_RADIUS, NODE_CARD_HEIGHT, NODE_CARD_WIDTH,
};
use crate::app::slint_frontend::{
    NodeGraphAction, NodeGraphPointerButton, NodeGraphPointerPayload, NodeGraphPointerPhase,
    SlintHostWindow,
};
use crate::app::state::{GraphInputSlot, NodeGraphConnectionPreview};
use crate::graph::scene::{CsgOp, NodeData, NodeId, SdfPrimitive};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointerMode {
    None,
    CanvasPan,
    NodeDrag(NodeId),
    Connect(NodeId),
}

#[derive(Clone, Copy, Debug)]
struct NodeGraphPointerState {
    mode: PointerMode,
    last_pointer: Option<[f32; 2]>,
}

impl Default for NodeGraphPointerState {
    fn default() -> Self {
        Self {
            mode: PointerMode::None,
            last_pointer: None,
        }
    }
}

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    let pointer_state = Rc::new(RefCell::new(NodeGraphPointerState::default()));
    window.on_node_graph_action(move |action, payload| {
        let pointer_state = pointer_state.clone();
        let viewport_dirty = node_graph_action_requires_viewport_dirty(action, &payload);
        mutate_host_and_tick_with_viewport_dirty(&context, viewport_dirty, move |host_state| {
            handle_node_graph_action(host_state, action, payload, &pointer_state);
        });
    });
}

fn node_graph_action_requires_viewport_dirty(
    action: NodeGraphAction,
    payload: &NodeGraphPointerPayload,
) -> bool {
    match action {
        NodeGraphAction::CanvasPointer => !matches!(payload.phase, NodeGraphPointerPhase::Move),
        NodeGraphAction::CanvasScroll
        | NodeGraphAction::ZoomIn
        | NodeGraphAction::ZoomOut
        | NodeGraphAction::FitView => false,
        NodeGraphAction::DisconnectSelectedEdge
        | NodeGraphAction::QuickAddSphere
        | NodeGraphAction::QuickAddBox
        | NodeGraphAction::QuickAddUnion => true,
    }
}

fn handle_node_graph_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    action: NodeGraphAction,
    payload: NodeGraphPointerPayload,
    pointer_state: &Rc<RefCell<NodeGraphPointerState>>,
) {
    if host_state.app.ui.menu.has_open_surface() {
        return;
    }
    sync_canvas_size(host_state, &payload);
    ensure_node_positions(host_state);

    match action {
        NodeGraphAction::CanvasPointer => handle_canvas_pointer(host_state, payload, pointer_state),
        NodeGraphAction::CanvasScroll => handle_canvas_scroll(host_state, payload),
        NodeGraphAction::ZoomIn => {
            apply_zoom(host_state, [payload.x, payload.y], 1.15);
        }
        NodeGraphAction::ZoomOut => {
            apply_zoom(host_state, [payload.x, payload.y], 0.87);
        }
        NodeGraphAction::FitView => fit_view(host_state),
        NodeGraphAction::DisconnectSelectedEdge => {
            if let Some(edge) = host_state.app.ui.node_graph_view.selected_edge {
                host_state.queue_action(Action::DisconnectGraphInput {
                    parent: edge.parent,
                    slot: edge.slot,
                });
            }
        }
        NodeGraphAction::QuickAddSphere => {
            host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Sphere));
        }
        NodeGraphAction::QuickAddBox => {
            host_state.queue_action(Action::CreatePrimitive(SdfPrimitive::Box));
        }
        NodeGraphAction::QuickAddUnion => {
            host_state.queue_action(Action::CreateOperation {
                op: CsgOp::Union,
                left: host_state.app.ui.selection.selected,
                right: None,
            });
        }
    }
}

fn sync_canvas_size(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    payload: &NodeGraphPointerPayload,
) {
    if payload.canvas_width > 1.0 && payload.canvas_height > 1.0 {
        host_state.app.ui.node_graph_view.canvas_size = [payload.canvas_width, payload.canvas_height];
        refresh_grid_cache(&mut host_state.app.ui.node_graph_view);
    }
}

fn handle_canvas_pointer(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    payload: NodeGraphPointerPayload,
    pointer_state: &Rc<RefCell<NodeGraphPointerState>>,
) {
    let pointer = [payload.x, payload.y];
    let mut state = pointer_state.borrow_mut();
    match payload.phase {
        NodeGraphPointerPhase::Down => {
            if payload.button != NodeGraphPointerButton::Primary {
                state.mode = PointerMode::None;
                state.last_pointer = None;
                return;
            }

            if let Some(source_node) = hit_output_handle(host_state, pointer) {
                host_state.app.ui.node_graph_view.connection_preview = Some(NodeGraphConnectionPreview {
                    source_node,
                    pointer_screen: pointer,
                });
                host_state.app.ui.node_graph_view.selected_edge = None;
                state.mode = PointerMode::Connect(source_node);
                state.last_pointer = Some(pointer);
                return;
            }

            if let Some(node_id) = hit_node(host_state, pointer) {
                host_state.queue_action(graph_selection_action(node_id, payload.modifiers.ctrl));
                host_state.app.ui.node_graph_view.selected_edge = None;
                state.mode = PointerMode::NodeDrag(node_id);
                state.last_pointer = Some(pointer);
                return;
            }

            if let Some((parent, slot)) = hit_edge(host_state, pointer) {
                host_state.app.ui.node_graph_view.selected_edge =
                    Some(crate::app::state::NodeGraphEdgeSelection { parent, slot });
                state.mode = PointerMode::None;
                state.last_pointer = Some(pointer);
                return;
            }

            host_state.app.ui.node_graph_view.selected_edge = None;
            state.mode = PointerMode::CanvasPan;
            state.last_pointer = Some(pointer);
        }
        NodeGraphPointerPhase::Move => {
            let Some(previous) = state.last_pointer else {
                return;
            };
            let delta = [pointer[0] - previous[0], pointer[1] - previous[1]];
            state.last_pointer = Some(pointer);
            match state.mode {
                PointerMode::CanvasPan => {
                    host_state.app.ui.node_graph_view.pan[0] += delta[0];
                    host_state.app.ui.node_graph_view.pan[1] += delta[1];
                }
                PointerMode::NodeDrag(node_id) => {
                    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
                    let position = host_state
                        .app
                        .ui
                        .node_graph_view
                        .node_positions
                        .entry(node_id)
                        .or_insert([0.0, 0.0]);
                    position[0] += delta[0] / zoom;
                    position[1] += delta[1] / zoom;
                }
                PointerMode::Connect(source_node) => {
                    host_state.app.ui.node_graph_view.connection_preview =
                        Some(NodeGraphConnectionPreview {
                            source_node,
                            pointer_screen: pointer,
                        });
                }
                PointerMode::None => {}
            }
        }
        NodeGraphPointerPhase::Up | NodeGraphPointerPhase::Cancel => {
            if matches!(state.mode, PointerMode::Connect(_))
                && payload.phase == NodeGraphPointerPhase::Up
                && payload.button == NodeGraphPointerButton::Primary
            {
                if let PointerMode::Connect(source_node) = state.mode {
                    if let Some((parent, slot)) = hit_input_handle(host_state, pointer) {
                        host_state.queue_action(Action::ConnectGraphInput {
                            parent,
                            slot,
                            child: source_node,
                        });
                    }
                }
            }
            host_state.app.ui.node_graph_view.connection_preview = None;
            state.mode = PointerMode::None;
            state.last_pointer = None;
        }
    }
}

fn handle_canvas_scroll(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    payload: NodeGraphPointerPayload,
) {
    if payload.delta_y.abs() <= f32::EPSILON {
        return;
    }
    let step = (-payload.delta_y / 120.0).clamp(-4.0, 4.0);
    let factor = 1.12_f32.powf(step);
    apply_zoom(host_state, [payload.x, payload.y], factor);
}

fn apply_zoom(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    anchor: [f32; 2],
    factor: f32,
) {
    let view = &mut host_state.app.ui.node_graph_view;
    let old_zoom = view.zoom.max(0.001);
    let new_zoom = clamp_zoom(old_zoom * factor);
    if (new_zoom - old_zoom).abs() <= f32::EPSILON {
        return;
    }
    let canvas_x = (anchor[0] - view.pan[0]) / old_zoom;
    let canvas_y = (anchor[1] - view.pan[1]) / old_zoom;
    view.zoom = new_zoom;
    view.pan = [
        anchor[0] - canvas_x * new_zoom,
        anchor[1] - canvas_y * new_zoom,
    ];
    refresh_grid_cache(view);
}

fn fit_view(host_state: &mut crate::app::slint_frontend::host_state::SlintHostState) {
    let mut node_ids = host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .copied()
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    if node_ids.is_empty() {
        return;
    }
    ensure_node_positions(host_state);
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for node_id in node_ids {
        let position = host_state
            .app
            .ui
            .node_graph_view
            .node_positions
            .get(&node_id)
            .copied()
            .unwrap_or([0.0, 0.0]);
        min_x = min_x.min(position[0]);
        min_y = min_y.min(position[1]);
        max_x = max_x.max(position[0] + NODE_CARD_WIDTH);
        max_y = max_y.max(position[1] + NODE_CARD_HEIGHT);
    }

    let content_width = (max_x - min_x).max(1.0);
    let content_height = (max_y - min_y).max(1.0);
    let canvas = host_state.app.ui.node_graph_view.canvas_size;
    let available_width = (canvas[0] - 72.0).max(40.0);
    let available_height = (canvas[1] - 72.0).max(40.0);
    let target_zoom = clamp_zoom((available_width / content_width).min(available_height / content_height));
    let center_x = min_x + content_width * 0.5;
    let center_y = min_y + content_height * 0.5;

    host_state.app.ui.node_graph_view.zoom = target_zoom;
    host_state.app.ui.node_graph_view.pan = [
        canvas[0] * 0.5 - center_x * target_zoom,
        canvas[1] * 0.5 - center_y * target_zoom,
    ];
    refresh_grid_cache(&mut host_state.app.ui.node_graph_view);
}

fn graph_selection_action(node_id: NodeId, toggle: bool) -> Action {
    if toggle {
        Action::ToggleSelection(node_id)
    } else {
        Action::Select(Some(node_id))
    }
}

fn hit_node(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
    pointer: [f32; 2],
) -> Option<NodeId> {
    let (positions, zoom) = projected_node_positions(host_state);
    let mut node_ids = positions.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    for node_id in node_ids.into_iter().rev() {
        let [x, y] = positions[&node_id];
        let width = NODE_CARD_WIDTH * zoom;
        let height = NODE_CARD_HEIGHT * zoom;
        if pointer[0] >= x && pointer[0] <= x + width && pointer[1] >= y && pointer[1] <= y + height {
            return Some(node_id);
        }
    }
    None
}

fn hit_output_handle(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
    pointer: [f32; 2],
) -> Option<NodeId> {
    let (positions, zoom) = projected_node_positions(host_state);
    let canvas_positions = all_canvas_positions(host_state);
    let radius = HANDLE_HIT_RADIUS * zoom.clamp(0.65, 1.45);
    let mut node_ids = positions.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    for node_id in node_ids.into_iter().rev() {
        let Some(node) = host_state.app.doc.scene.nodes.get(&node_id) else {
            continue;
        };
        if !node_has_output_handle(&node.data) {
            continue;
        }
        let [x, y] = *canvas_positions.get(&node_id)?;
        let handle_canvas = output_handle_position(x, y);
        let handle = screen_from_canvas_with_view(host_state, handle_canvas);
        if distance(pointer, handle) <= radius {
            return Some(node_id);
        }
    }
    None
}

fn hit_input_handle(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
    pointer: [f32; 2],
) -> Option<(NodeId, GraphInputSlot)> {
    let (positions, zoom) = projected_node_positions(host_state);
    let canvas_positions = all_canvas_positions(host_state);
    let radius = HANDLE_HIT_RADIUS * zoom.clamp(0.65, 1.45);
    let mut node_ids = positions.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    for node_id in node_ids.into_iter().rev() {
        let Some(node) = host_state.app.doc.scene.nodes.get(&node_id) else {
            continue;
        };
        let [x, y] = *canvas_positions.get(&node_id)?;
        match &node.data {
            NodeData::Operation { .. } => {
                let left = screen_from_canvas_with_view(
                    host_state,
                    input_handle_position_for_slot(x, y, EdgeSlot::Left),
                );
                if distance(pointer, left) <= radius {
                    return Some((node_id, GraphInputSlot::Left));
                }
                let right = screen_from_canvas_with_view(
                    host_state,
                    input_handle_position_for_slot(x, y, EdgeSlot::Right),
                );
                if distance(pointer, right) <= radius {
                    return Some((node_id, GraphInputSlot::Right));
                }
            }
            NodeData::Sculpt { .. } | NodeData::Transform { .. } | NodeData::Modifier { .. } => {
                let input = screen_from_canvas_with_view(
                    host_state,
                    input_handle_position_for_slot(x, y, EdgeSlot::Input),
                );
                if distance(pointer, input) <= radius {
                    return Some((node_id, GraphInputSlot::Input));
                }
            }
            NodeData::Primitive { .. } | NodeData::Light { .. } => {}
        }
    }
    None
}

fn hit_edge(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
    pointer: [f32; 2],
) -> Option<(NodeId, GraphInputSlot)> {
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    let threshold = 10.0 * zoom.clamp(0.7, 1.3);
    let canvas_positions = all_canvas_positions(host_state);
    let mut node_ids = host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .copied()
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    let mut nearest: Option<(NodeId, GraphInputSlot, f32)> = None;
    for parent_id in node_ids {
        let Some(parent_node) = host_state.app.doc.scene.nodes.get(&parent_id) else {
            continue;
        };
        let slots = node_slots(parent_node);
        for (slot, child_id) in slots {
            let Some(child_id) = child_id else {
                continue;
            };
            let Some(parent_pos) = canvas_positions.get(&parent_id).copied() else {
                continue;
            };
            let Some(child_pos) = canvas_positions.get(&child_id).copied() else {
                continue;
            };
            let edge_slot = match slot {
                GraphInputSlot::Left => EdgeSlot::Left,
                GraphInputSlot::Right => EdgeSlot::Right,
                GraphInputSlot::Input => EdgeSlot::Input,
            };
            let curve = edge_curve_screen(
                parent_pos,
                child_pos,
                edge_slot,
                host_state.app.ui.node_graph_view.pan,
                zoom,
            );
            let distance = curve.distance_to_point(pointer, 20);
            if distance <= threshold {
                match nearest {
                    Some((_, _, best)) if best <= distance => {}
                    _ => nearest = Some((parent_id, slot, distance)),
                }
            }
        }
    }
    nearest.map(|(parent, slot, _)| (parent, slot))
}

fn node_slots(node: &crate::graph::scene::SceneNode) -> [(GraphInputSlot, Option<NodeId>); 2] {
    match &node.data {
        NodeData::Operation { left, right, .. } => {
            [(GraphInputSlot::Left, *left), (GraphInputSlot::Right, *right)]
        }
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => {
            [(GraphInputSlot::Input, *input), (GraphInputSlot::Input, None)]
        }
        NodeData::Primitive { .. } | NodeData::Light { .. } => {
            [(GraphInputSlot::Input, None), (GraphInputSlot::Input, None)]
        }
    }
}

fn projected_node_positions(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
) -> (HashMap<NodeId, [f32; 2]>, f32) {
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    let positions = all_canvas_positions(host_state)
        .into_iter()
        .map(|(node_id, canvas)| (node_id, screen_from_canvas_with_view(host_state, canvas)))
        .collect::<HashMap<_, _>>();
    (positions, zoom)
}

fn all_canvas_positions(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
)-> HashMap<NodeId, [f32; 2]> {
    host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .copied()
        .filter_map(|node_id| {
            host_state
                .app
        .ui
        .node_graph_view
        .node_positions
        .get(&node_id)
        .copied()
                .map(|position| (node_id, position))
        })
        .collect()
}

fn screen_from_canvas_with_view(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
    point: [f32; 2],
) -> [f32; 2] {
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    screen_from_canvas(point, host_state.app.ui.node_graph_view.pan, zoom)
}

fn refresh_grid_cache(view: &mut crate::app::state::NodeGraphViewState) {
    let zoom_bucket = grid_zoom_bucket(view.zoom);
    let canvas_bucket = grid_canvas_bucket(view.canvas_size);
    if !view.grid_dots.is_empty()
        && view.grid_zoom_bucket == zoom_bucket
        && view.grid_canvas_bucket == canvas_bucket
    {
        return;
    }
    let gap = grid_gap_for_zoom(view.zoom);
    view.grid_dots = build_grid_base_dots(view.canvas_size, gap);
    view.grid_gap = gap;
    view.grid_zoom_bucket = zoom_bucket;
    view.grid_canvas_bucket = canvas_bucket;
}

fn ensure_node_positions(host_state: &mut crate::app::slint_frontend::host_state::SlintHostState) {
    let scene_node_count = host_state.app.doc.scene.nodes.len();
    if host_state.app.ui.node_graph_view.node_positions.len() >= scene_node_count
        && host_state
            .app
            .doc
            .scene
            .nodes
            .keys()
            .all(|node_id| host_state.app.ui.node_graph_view.node_positions.contains_key(node_id))
    {
        return;
    }
    let needs_fill = host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .any(|node_id| !host_state.app.ui.node_graph_view.node_positions.contains_key(node_id));
    if !needs_fill {
        return;
    }
    let defaults = crate::app::node_graph::default_node_positions(&host_state.app.doc.scene);
    for (node_id, position) in defaults {
        host_state
            .app
            .ui
            .node_graph_view
            .node_positions
            .entry(node_id)
            .or_insert(position);
    }
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{graph_selection_action, node_graph_action_requires_viewport_dirty};
    use crate::app::actions::Action;
    use crate::app::slint_frontend::{
        NodeGraphAction, NodeGraphModifiers, NodeGraphPointerButton, NodeGraphPointerPayload,
        NodeGraphPointerPhase,
    };

    fn payload(phase: NodeGraphPointerPhase) -> NodeGraphPointerPayload {
        NodeGraphPointerPayload {
            phase,
            button: NodeGraphPointerButton::Primary,
            x: 0.0,
            y: 0.0,
            canvas_width: 640.0,
            canvas_height: 420.0,
            delta_x: 0.0,
            delta_y: 0.0,
            modifiers: NodeGraphModifiers {
                ctrl: false,
                shift: false,
                alt: false,
            },
        }
    }

    #[test]
    fn graph_click_dispatches_select_action() {
        match graph_selection_action(42, false) {
            Action::Select(Some(node_id)) => assert_eq!(node_id, 42),
            _ => panic!("expected select action"),
        }
    }

    #[test]
    fn graph_ctrl_click_dispatches_toggle_selection_action() {
        match graph_selection_action(42, true) {
            Action::ToggleSelection(node_id) => assert_eq!(node_id, 42),
            _ => panic!("expected toggle selection action"),
        }
    }

    #[test]
    fn pointer_move_does_not_force_viewport_redraw() {
        assert!(!node_graph_action_requires_viewport_dirty(
            NodeGraphAction::CanvasPointer,
            &payload(NodeGraphPointerPhase::Move)
        ));
    }

    #[test]
    fn pointer_press_keeps_viewport_redraw_enabled() {
        assert!(node_graph_action_requires_viewport_dirty(
            NodeGraphAction::CanvasPointer,
            &payload(NodeGraphPointerPhase::Down)
        ));
    }
}
