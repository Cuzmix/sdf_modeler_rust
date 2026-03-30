use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick_with_viewport_dirty;
use crate::app::actions::Action;
use crate::app::node_graph::{
    build_grid_base_dots, clamp_zoom, edge_curve_from_projected_frames, grid_canvas_bucket,
    grid_gap_for_zoom, grid_zoom_bucket, node_has_output_handle, project_node_screen_frame,
    socket_screen_position, EdgeSlot, HANDLE_HIT_RADIUS, NODE_CARD_HEIGHT, NODE_CARD_WIDTH,
    NodeScreenFrame, NodeSocketKind as GeometrySocketKind,
};
use crate::app::slint_frontend::{
    NodeGraphAction, NodeGraphPointerButton, NodeGraphPointerPayload, NodeGraphPointerPhase,
    SlintHostWindow,
};
use crate::app::state::{
    GraphInputSlot, NodeGraphConnectionPreview, NodeGraphEdgeSelection, NodeGraphMarqueeRect,
    NodeGraphSocketHover, NodeGraphSocketKind, PanelBarId, PanelKind,
};
use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene};

type HostState = crate::app::slint_frontend::host_state::SlintHostState;

const NODE_DRAG_THRESHOLD_PX: f32 = 4.0;
const MARQUEE_SELECT_MIN_PX: f32 = 3.0;
const EDGE_HIT_SAMPLES: u32 = 24;
const EDGE_HIT_THRESHOLD_PX: f32 = 10.0;

const OPERATION_INPUT_SOCKETS: &[(GraphInputSlot, GeometrySocketKind, NodeGraphSocketKind)] = &[
    (
        GraphInputSlot::Left,
        GeometrySocketKind::LeftInput,
        NodeGraphSocketKind::LeftInput,
    ),
    (
        GraphInputSlot::Right,
        GeometrySocketKind::RightInput,
        NodeGraphSocketKind::RightInput,
    ),
];
const SINGLE_INPUT_SOCKET: &[(GraphInputSlot, GeometrySocketKind, NodeGraphSocketKind)] = &[(
    GraphInputSlot::Input,
    GeometrySocketKind::Input,
    NodeGraphSocketKind::Input,
)];
const NO_INPUT_SOCKETS: &[(GraphInputSlot, GeometrySocketKind, NodeGraphSocketKind)] = &[];

#[derive(Clone, Copy, Debug, PartialEq)]
enum PointerMode {
    None,
    CanvasPan,
    NodePress { node_id: NodeId },
    NodeDrag { node_id: NodeId },
    Connect { source_node: NodeId },
    Marquee { start: [f32; 2], additive: bool },
}

#[derive(Clone, Debug)]
struct ConnectionTargetCandidate {
    parent: NodeId,
    slot: GraphInputSlot,
    socket_kind: NodeGraphSocketKind,
    center_screen: [f32; 2],
    valid: bool,
    invalid_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct NodeGraphPointerState {
    mode: PointerMode,
    press_origin: Option<[f32; 2]>,
    last_pointer: Option<[f32; 2]>,
    connection_target: Option<ConnectionTargetCandidate>,
}

impl Default for NodeGraphPointerState {
    fn default() -> Self {
        Self {
            mode: PointerMode::None,
            press_origin: None,
            last_pointer: None,
            connection_target: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NodeGraphShortcutIntent {
    FitSelected,
    FitAll,
    DisconnectEdge,
    CancelInteraction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphNodeFamily {
    Geometry,
    Light,
    Unknown,
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

pub(super) fn clear_node_graph_keyboard_focus(host_state: &mut HostState) {
    host_state.app.ui.node_graph_view.graph_keyboard_focus = false;
}

pub(super) fn apply_node_graph_shortcut(
    host_state: &mut HostState,
    intent: NodeGraphShortcutIntent,
) -> bool {
    if !node_graph_shortcut_target_active(host_state) {
        return false;
    }
    ensure_node_positions(host_state);
    match intent {
        NodeGraphShortcutIntent::FitSelected => {
            fit_selected_or_all(host_state);
            true
        }
        NodeGraphShortcutIntent::FitAll => {
            fit_view(host_state);
            true
        }
        NodeGraphShortcutIntent::DisconnectEdge => disconnect_selected_edge(host_state),
        NodeGraphShortcutIntent::CancelInteraction => {
            clear_graph_transient_state(&mut host_state.app.ui.node_graph_view, true)
        }
    }
}

fn node_graph_shortcut_target_active(host_state: &HostState) -> bool {
    host_state.app.ui.node_graph_view.graph_keyboard_focus && node_graph_panel_is_open(host_state)
}

fn node_graph_panel_is_open(host_state: &HostState) -> bool {
    host_state
        .app
        .ui
        .panel_framework
        .pinned_instance(PanelKind::NodeGraph)
        .is_some()
        || host_state
            .app
            .ui
            .panel_framework
            .active_transient(PanelBarId::PrimaryRight)
            == Some(PanelKind::NodeGraph)
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
        | NodeGraphAction::FitView
        | NodeGraphAction::FitSelected
        | NodeGraphAction::CancelInteraction
        | NodeGraphAction::MarqueeBegin
        | NodeGraphAction::MarqueeUpdate
        | NodeGraphAction::MarqueeEnd => false,
        NodeGraphAction::DisconnectSelectedEdge
        | NodeGraphAction::QuickAddSphere
        | NodeGraphAction::QuickAddBox
        | NodeGraphAction::QuickAddUnion => true,
    }
}

fn handle_node_graph_action(
    host_state: &mut HostState,
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
        NodeGraphAction::CanvasScroll => {
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            handle_canvas_scroll(host_state, payload);
        }
        NodeGraphAction::ZoomIn => {
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            apply_zoom(host_state, [payload.x, payload.y], 1.15);
        }
        NodeGraphAction::ZoomOut => {
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            apply_zoom(host_state, [payload.x, payload.y], 0.87);
        }
        NodeGraphAction::FitView => {
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            fit_view(host_state);
        }
        NodeGraphAction::FitSelected => {
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            fit_selected_or_all(host_state);
        }
        NodeGraphAction::CancelInteraction => {
            clear_graph_transient_state(&mut host_state.app.ui.node_graph_view, true);
            reset_pointer_state(&mut pointer_state.borrow_mut());
        }
        NodeGraphAction::MarqueeBegin => {
            begin_marquee_interaction(host_state, payload, &mut pointer_state.borrow_mut());
        }
        NodeGraphAction::MarqueeUpdate => {
            update_marquee_interaction(host_state, payload, &mut pointer_state.borrow_mut());
        }
        NodeGraphAction::MarqueeEnd => {
            end_marquee_interaction(host_state, payload, &mut pointer_state.borrow_mut());
        }
        NodeGraphAction::DisconnectSelectedEdge => {
            disconnect_selected_edge(host_state);
        }
        NodeGraphAction::QuickAddSphere => {
            host_state.queue_action(Action::CreatePrimitive(crate::graph::scene::SdfPrimitive::Sphere));
        }
        NodeGraphAction::QuickAddBox => {
            host_state.queue_action(Action::CreatePrimitive(crate::graph::scene::SdfPrimitive::Box));
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

fn disconnect_selected_edge(host_state: &mut HostState) -> bool {
    if let Some(edge) = host_state.app.ui.node_graph_view.selected_edge {
        host_state.queue_action(Action::DisconnectGraphInput {
            parent: edge.parent,
            slot: edge.slot,
        });
        return true;
    }
    false
}

fn begin_marquee_interaction(
    host_state: &mut HostState,
    payload: NodeGraphPointerPayload,
    pointer_state: &mut NodeGraphPointerState,
) {
    host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
    let pointer = [payload.x, payload.y];
    let additive = payload.modifiers.ctrl || payload.modifiers.shift;
    begin_marquee(&mut host_state.app.ui.node_graph_view, pointer, additive);
    pointer_state.mode = PointerMode::Marquee {
        start: pointer,
        additive,
    };
    pointer_state.press_origin = Some(pointer);
    pointer_state.last_pointer = Some(pointer);
}

fn update_marquee_interaction(
    host_state: &mut HostState,
    payload: NodeGraphPointerPayload,
    pointer_state: &mut NodeGraphPointerState,
) {
    if let PointerMode::Marquee { start, additive } = pointer_state.mode {
        update_marquee(
            &mut host_state.app.ui.node_graph_view,
            start,
            [payload.x, payload.y],
            additive,
        );
    }
}

fn end_marquee_interaction(
    host_state: &mut HostState,
    payload: NodeGraphPointerPayload,
    pointer_state: &mut NodeGraphPointerState,
) {
    if let PointerMode::Marquee { start, additive } = pointer_state.mode {
        let rect = compute_marquee_rect(start, [payload.x, payload.y], additive);
        finalize_marquee_selection(host_state, rect);
    }
    host_state.app.ui.node_graph_view.marquee_rect = None;
    reset_pointer_state(pointer_state);
}

fn sync_canvas_size(host_state: &mut HostState, payload: &NodeGraphPointerPayload) {
    if payload.canvas_width <= 1.0 || payload.canvas_height <= 1.0 {
        return;
    }
    let next_size = [payload.canvas_width, payload.canvas_height];
    if host_state.app.ui.node_graph_view.canvas_size == next_size {
        return;
    }
    host_state.app.ui.node_graph_view.canvas_size = next_size;
    refresh_grid_cache(&mut host_state.app.ui.node_graph_view);
}

fn handle_canvas_pointer(
    host_state: &mut HostState,
    payload: NodeGraphPointerPayload,
    pointer_state: &Rc<RefCell<NodeGraphPointerState>>,
) {
    let pointer = [payload.x, payload.y];
    let mut state = pointer_state.borrow_mut();
    match payload.phase {
        NodeGraphPointerPhase::Down => {
            if payload.button != NodeGraphPointerButton::Primary {
                reset_pointer_state(&mut state);
                return;
            }
            host_state.app.ui.node_graph_view.graph_keyboard_focus = true;
            host_state.app.ui.node_graph_view.hovered_edge = None;
            host_state.app.ui.node_graph_view.hovered_socket = None;
            state.press_origin = Some(pointer);
            state.last_pointer = Some(pointer);
            state.connection_target = None;

            if payload.modifiers.shift
                && hit_node(host_state, pointer).is_none()
                && hit_output_handle(host_state, pointer).is_none()
            {
                let additive = true;
                begin_marquee(&mut host_state.app.ui.node_graph_view, pointer, additive);
                state.mode = PointerMode::Marquee {
                    start: pointer,
                    additive,
                };
                return;
            }

            if let Some(source_node) = hit_output_handle(host_state, pointer) {
                host_state.app.ui.node_graph_view.selected_edge = None;
                state.mode = PointerMode::Connect { source_node };
                update_connection_drag_feedback(
                    host_state,
                    source_node,
                    pointer,
                    &mut state.connection_target,
                );
                return;
            }

            if let Some(node_id) = hit_node(host_state, pointer) {
                host_state.queue_action(graph_selection_action(node_id, payload.modifiers.ctrl));
                host_state.app.ui.node_graph_view.selected_edge = None;
                state.mode = PointerMode::NodePress { node_id };
                return;
            }

            if let Some(edge) = hit_edge(host_state, pointer) {
                host_state.app.ui.node_graph_view.selected_edge = Some(edge);
                state.mode = PointerMode::None;
                return;
            }

            host_state.app.ui.node_graph_view.selected_edge = None;
            state.mode = PointerMode::CanvasPan;
        }
        NodeGraphPointerPhase::Move => {
            let previous = state.last_pointer.unwrap_or(pointer);
            let delta = [pointer[0] - previous[0], pointer[1] - previous[1]];
            state.last_pointer = Some(pointer);
            match state.mode {
                PointerMode::None => {
                    update_hover_feedback(host_state, pointer);
                }
                PointerMode::CanvasPan => {
                    host_state.app.ui.node_graph_view.pan[0] += delta[0];
                    host_state.app.ui.node_graph_view.pan[1] += delta[1];
                }
                PointerMode::NodePress { node_id } => {
                    if drag_distance(state.press_origin.unwrap_or(pointer), pointer)
                        >= NODE_DRAG_THRESHOLD_PX
                    {
                        state.mode = PointerMode::NodeDrag { node_id };
                        state.last_pointer = Some(pointer);
                    }
                }
                PointerMode::NodeDrag { node_id } => {
                    if delta[0].abs() <= f32::EPSILON && delta[1].abs() <= f32::EPSILON {
                        return;
                    }
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
                PointerMode::Connect { source_node } => {
                    update_connection_drag_feedback(
                        host_state,
                        source_node,
                        pointer,
                        &mut state.connection_target,
                    );
                }
                PointerMode::Marquee { start, additive } => {
                    update_marquee(&mut host_state.app.ui.node_graph_view, start, pointer, additive);
                }
            }
        }
        NodeGraphPointerPhase::Up => {
            if payload.button == NodeGraphPointerButton::Primary {
                match state.mode {
                    PointerMode::Connect { source_node } => {
                        finalize_connection_drag(
                            host_state,
                            source_node,
                            pointer,
                            &mut state.connection_target,
                        );
                    }
                    PointerMode::Marquee { start, additive } => {
                        let rect = compute_marquee_rect(start, pointer, additive);
                        finalize_marquee_selection(host_state, rect);
                        host_state.app.ui.node_graph_view.marquee_rect = None;
                    }
                    _ => {}
                }
            }
            host_state.app.ui.node_graph_view.connection_preview = None;
            host_state.app.ui.node_graph_view.hovered_socket = None;
            if !matches!(state.mode, PointerMode::CanvasPan | PointerMode::NodeDrag { .. }) {
                update_hover_feedback(host_state, pointer);
            }
            reset_pointer_state(&mut state);
        }
        NodeGraphPointerPhase::Cancel => {
            clear_graph_transient_state(&mut host_state.app.ui.node_graph_view, false);
            reset_pointer_state(&mut state);
        }
    }
}

fn handle_canvas_scroll(host_state: &mut HostState, payload: NodeGraphPointerPayload) {
    if payload.delta_y.abs() <= f32::EPSILON {
        return;
    }
    let step = (-payload.delta_y / 120.0).clamp(-4.0, 4.0);
    let factor = 1.12_f32.powf(step);
    apply_zoom(host_state, [payload.x, payload.y], factor);
}

fn apply_zoom(host_state: &mut HostState, anchor: [f32; 2], factor: f32) {
    let view = &mut host_state.app.ui.node_graph_view;
    let old_zoom = view.zoom.max(0.001);
    let new_zoom = clamp_zoom(old_zoom * factor);
    if (new_zoom - old_zoom).abs() <= f32::EPSILON {
        return;
    }
    let canvas_x = (anchor[0] - view.pan[0]) / old_zoom;
    let canvas_y = (anchor[1] - view.pan[1]) / old_zoom;
    view.zoom = new_zoom;
    view.pan = [anchor[0] - canvas_x * new_zoom, anchor[1] - canvas_y * new_zoom];
    refresh_grid_cache(view);
}

fn fit_view(host_state: &mut HostState) {
    let mut node_ids = host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .copied()
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    fit_nodes(host_state, &node_ids);
}

fn fit_selected_or_all(host_state: &mut HostState) {
    let mut selected_ids = host_state
        .app
        .ui
        .selection
        .selected_set
        .iter()
        .copied()
        .filter(|node_id| host_state.app.doc.scene.nodes.contains_key(node_id))
        .collect::<Vec<_>>();
    selected_ids.sort_unstable();
    selected_ids.dedup();
    if selected_ids.is_empty() {
        fit_view(host_state);
        return;
    }
    fit_nodes(host_state, &selected_ids);
}

fn fit_nodes(host_state: &mut HostState, node_ids: &[NodeId]) {
    if node_ids.is_empty() {
        return;
    }
    ensure_node_positions(host_state);
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut matched = 0_u32;
    for node_id in node_ids {
        let Some(position) = host_state.app.ui.node_graph_view.node_positions.get(node_id).copied() else {
            continue;
        };
        matched += 1;
        min_x = min_x.min(position[0]);
        min_y = min_y.min(position[1]);
        max_x = max_x.max(position[0] + NODE_CARD_WIDTH);
        max_y = max_y.max(position[1] + NODE_CARD_HEIGHT);
    }
    if matched == 0 {
        return;
    }

    let content_width = (max_x - min_x).max(1.0);
    let content_height = (max_y - min_y).max(1.0);
    let canvas = host_state.app.ui.node_graph_view.canvas_size;
    let available_width = (canvas[0] - 72.0).max(40.0);
    let available_height = (canvas[1] - 72.0).max(40.0);
    let target_zoom =
        clamp_zoom((available_width / content_width).min(available_height / content_height));
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

fn hit_node(host_state: &HostState, pointer: [f32; 2]) -> Option<NodeId> {
    let frames = projected_node_frames(host_state);
    let mut node_ids = frames.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    node_ids
        .into_iter()
        .rev()
        .find(|node_id| frames[node_id].contains(pointer))
}

fn hit_output_handle(host_state: &HostState, pointer: [f32; 2]) -> Option<NodeId> {
    let frames = projected_node_frames(host_state);
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    let radius = HANDLE_HIT_RADIUS * zoom.clamp(0.65, 1.45);
    let mut node_ids = frames.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    for node_id in node_ids.into_iter().rev() {
        let Some(node) = host_state.app.doc.scene.nodes.get(&node_id) else {
            continue;
        };
        if !node_has_output_handle(&node.data) {
            continue;
        }
        let center = socket_screen_position(frames[&node_id], GeometrySocketKind::Output);
        if distance(pointer, center) <= radius {
            return Some(node_id);
        }
    }
    None
}

fn hit_input_socket_for_connection(
    host_state: &HostState,
    pointer: [f32; 2],
    source_node: NodeId,
) -> Option<ConnectionTargetCandidate> {
    let frames = projected_node_frames(host_state);
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    let radius = HANDLE_HIT_RADIUS * zoom.clamp(0.65, 1.45);
    let mut node_ids = frames.keys().copied().collect::<Vec<_>>();
    node_ids.sort_unstable();
    let mut nearest: Option<(ConnectionTargetCandidate, f32)> = None;

    for node_id in node_ids.into_iter().rev() {
        let Some(node) = host_state.app.doc.scene.nodes.get(&node_id) else {
            continue;
        };
        for (slot, geometry_socket, ui_socket) in input_socket_descriptors(&node.data) {
            let center = socket_screen_position(frames[&node_id], *geometry_socket);
            let handle_distance = distance(pointer, center);
            if handle_distance > radius {
                continue;
            }
            let validation =
                validate_connection_target(&host_state.app.doc.scene, source_node, node_id, *slot);
            let candidate = ConnectionTargetCandidate {
                parent: node_id,
                slot: *slot,
                socket_kind: *ui_socket,
                center_screen: center,
                valid: validation.is_ok(),
                invalid_reason: validation.err(),
            };
            match nearest {
                Some((_, best_distance)) if best_distance <= handle_distance => {}
                _ => nearest = Some((candidate, handle_distance)),
            }
        }
    }

    nearest.map(|(candidate, _)| candidate)
}

fn input_socket_descriptors(
    node_data: &NodeData,
) -> &'static [(GraphInputSlot, GeometrySocketKind, NodeGraphSocketKind)] {
    match node_data {
        NodeData::Operation { .. } => OPERATION_INPUT_SOCKETS,
        NodeData::Sculpt { .. } | NodeData::Transform { .. } | NodeData::Modifier { .. } => {
            SINGLE_INPUT_SOCKET
        }
        NodeData::Primitive { .. } | NodeData::Light { .. } => NO_INPUT_SOCKETS,
    }
}

fn hit_edge(host_state: &HostState, pointer: [f32; 2]) -> Option<NodeGraphEdgeSelection> {
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    let threshold = EDGE_HIT_THRESHOLD_PX * zoom.clamp(0.7, 1.3);
    let frames = projected_node_frames(host_state);
    let mut node_ids = host_state
        .app
        .doc
        .scene
        .nodes
        .keys()
        .copied()
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    let mut nearest: Option<(NodeGraphEdgeSelection, f32)> = None;

    for parent_id in node_ids {
        let Some(parent_node) = host_state.app.doc.scene.nodes.get(&parent_id) else {
            continue;
        };
        let Some(parent_frame) = frames.get(&parent_id).copied() else {
            continue;
        };
        for (slot, child_id) in connected_slots(parent_node) {
            let Some(child_id) = child_id else {
                continue;
            };
            let Some(child_frame) = frames.get(&child_id).copied() else {
                continue;
            };
            let curve = edge_curve_from_projected_frames(parent_frame, child_frame, edge_slot(slot));
            let distance = curve.distance_to_point(pointer, EDGE_HIT_SAMPLES);
            if distance > threshold {
                continue;
            }
            let selection = NodeGraphEdgeSelection {
                parent: parent_id,
                slot,
            };
            match nearest {
                Some((_, best_distance)) if best_distance <= distance => {}
                _ => nearest = Some((selection, distance)),
            }
        }
    }

    nearest.map(|(selection, _)| selection)
}

fn connected_slots(node: &crate::graph::scene::SceneNode) -> [(GraphInputSlot, Option<NodeId>); 2] {
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

fn edge_slot(slot: GraphInputSlot) -> EdgeSlot {
    match slot {
        GraphInputSlot::Left => EdgeSlot::Left,
        GraphInputSlot::Right => EdgeSlot::Right,
        GraphInputSlot::Input => EdgeSlot::Input,
    }
}

fn projected_node_frames(host_state: &HostState) -> HashMap<NodeId, NodeScreenFrame> {
    let zoom = host_state.app.ui.node_graph_view.zoom.max(0.001);
    all_canvas_positions(host_state)
        .into_iter()
        .map(|(node_id, position)| {
            (
                node_id,
                project_node_screen_frame(position, host_state.app.ui.node_graph_view.pan, zoom),
            )
        })
        .collect()
}

fn all_canvas_positions(host_state: &HostState) -> HashMap<NodeId, [f32; 2]> {
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

fn update_connection_drag_feedback(
    host_state: &mut HostState,
    source_node: NodeId,
    pointer: [f32; 2],
    connection_target: &mut Option<ConnectionTargetCandidate>,
) {
    let candidate = hit_input_socket_for_connection(host_state, pointer, source_node);
    let preview_pointer = preview_pointer_for_candidate(pointer, candidate.as_ref());
    host_state.app.ui.node_graph_view.connection_preview = Some(NodeGraphConnectionPreview {
        source_node,
        pointer_screen: preview_pointer,
    });
    host_state.app.ui.node_graph_view.hovered_socket = candidate.as_ref().map(|target| {
        NodeGraphSocketHover {
            node: target.parent,
            socket: target.socket_kind,
            valid: target.valid,
        }
    });
    host_state.app.ui.node_graph_view.hovered_edge = None;
    *connection_target = candidate;
}

fn preview_pointer_for_candidate(
    pointer: [f32; 2],
    candidate: Option<&ConnectionTargetCandidate>,
) -> [f32; 2] {
    candidate
        .filter(|target| target.valid)
        .map(|target| target.center_screen)
        .unwrap_or(pointer)
}

fn finalize_connection_drag(
    host_state: &mut HostState,
    source_node: NodeId,
    pointer: [f32; 2],
    connection_target: &mut Option<ConnectionTargetCandidate>,
) {
    let candidate = connection_target
        .take()
        .or_else(|| hit_input_socket_for_connection(host_state, pointer, source_node));
    if let Some(target) = candidate {
        if target.valid {
            host_state.queue_action(Action::ConnectGraphInput {
                parent: target.parent,
                slot: target.slot,
                child: source_node,
            });
        } else {
            host_state.queue_action(Action::ShowToast {
                message: target
                    .invalid_reason
                    .unwrap_or_else(|| "Invalid connection target.".to_string()),
                is_error: true,
            });
        }
    }
    host_state.app.ui.node_graph_view.connection_preview = None;
    host_state.app.ui.node_graph_view.hovered_socket = None;
}

fn update_hover_feedback(host_state: &mut HostState, pointer: [f32; 2]) {
    if host_state.app.ui.node_graph_view.connection_preview.is_some() {
        return;
    }
    host_state.app.ui.node_graph_view.hovered_socket = None;
    host_state.app.ui.node_graph_view.hovered_edge = hit_edge(host_state, pointer);
}

fn begin_marquee(
    view: &mut crate::app::state::NodeGraphViewState,
    pointer: [f32; 2],
    additive: bool,
) {
    view.selected_edge = None;
    view.marquee_rect = Some(NodeGraphMarqueeRect {
        x: pointer[0],
        y: pointer[1],
        width: 0.0,
        height: 0.0,
        additive,
    });
}

fn update_marquee(
    view: &mut crate::app::state::NodeGraphViewState,
    start: [f32; 2],
    pointer: [f32; 2],
    additive: bool,
) {
    view.marquee_rect = Some(compute_marquee_rect(start, pointer, additive));
}

fn compute_marquee_rect(start: [f32; 2], pointer: [f32; 2], additive: bool) -> NodeGraphMarqueeRect {
    let min_x = start[0].min(pointer[0]);
    let min_y = start[1].min(pointer[1]);
    NodeGraphMarqueeRect {
        x: min_x,
        y: min_y,
        width: (pointer[0] - start[0]).abs(),
        height: (pointer[1] - start[1]).abs(),
        additive,
    }
}

fn finalize_marquee_selection(host_state: &mut HostState, rect: NodeGraphMarqueeRect) {
    if rect.width < MARQUEE_SELECT_MIN_PX && rect.height < MARQUEE_SELECT_MIN_PX {
        return;
    }

    let mut selected = projected_node_frames(host_state)
        .into_iter()
        .filter_map(|(node_id, frame)| node_inside_rect(frame, rect).then_some(node_id))
        .collect::<Vec<_>>();
    selected.sort_unstable();

    if !rect.additive {
        host_state.queue_action(Action::Select(None));
        for node_id in selected {
            host_state.queue_action(Action::ToggleSelection(node_id));
        }
        return;
    }

    let existing_selection = host_state.app.ui.selection.selected_set.clone();
    for node_id in selected {
        if !existing_selection.contains(&node_id) {
            host_state.queue_action(Action::ToggleSelection(node_id));
        }
    }
}

fn node_inside_rect(frame: NodeScreenFrame, rect: NodeGraphMarqueeRect) -> bool {
    let right = rect.x + rect.width;
    let bottom = rect.y + rect.height;
    frame.origin[0] >= rect.x
        && frame.origin[1] >= rect.y
        && frame.origin[0] + frame.size[0] <= right
        && frame.origin[1] + frame.size[1] <= bottom
}

fn clear_graph_transient_state(
    view: &mut crate::app::state::NodeGraphViewState,
    clear_selected_edge: bool,
) -> bool {
    let mut changed = false;
    changed |= view.connection_preview.take().is_some();
    changed |= view.hovered_socket.take().is_some();
    changed |= view.hovered_edge.take().is_some();
    changed |= view.marquee_rect.take().is_some();
    if clear_selected_edge {
        changed |= view.selected_edge.take().is_some();
    }
    changed
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

fn ensure_node_positions(host_state: &mut HostState) {
    crate::app::node_graph::fill_missing_node_positions(
        &host_state.app.doc.scene,
        &mut host_state.app.ui.node_graph_view.node_positions,
    );
}

fn graph_slot_child(scene: &Scene, parent: NodeId, slot: GraphInputSlot) -> Option<Option<NodeId>> {
    let node = scene.nodes.get(&parent)?;
    match (&node.data, slot) {
        (NodeData::Operation { left, .. }, GraphInputSlot::Left) => Some(*left),
        (NodeData::Operation { right, .. }, GraphInputSlot::Right) => Some(*right),
        (
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. },
            GraphInputSlot::Input,
        ) => Some(*input),
        _ => None,
    }
}

fn node_output_family(
    scene: &Scene,
    node_id: NodeId,
    visiting: &mut HashSet<NodeId>,
) -> GraphNodeFamily {
    if !visiting.insert(node_id) {
        return GraphNodeFamily::Unknown;
    }

    let family = match scene.nodes.get(&node_id).map(|node| &node.data) {
        Some(NodeData::Light { .. }) => GraphNodeFamily::Light,
        Some(
            NodeData::Primitive { .. }
            | NodeData::Operation { .. }
            | NodeData::Modifier { .. }
            | NodeData::Sculpt { .. },
        ) => GraphNodeFamily::Geometry,
        Some(NodeData::Transform { input, .. }) => input
            .map(|child_id| node_output_family(scene, child_id, visiting))
            .unwrap_or(GraphNodeFamily::Unknown),
        None => GraphNodeFamily::Unknown,
    };

    visiting.remove(&node_id);
    family
}

fn child_allowed_for_parent(scene: &Scene, parent: NodeId, child: NodeId) -> Result<(), &'static str> {
    let mut visiting = HashSet::new();
    let child_family = node_output_family(scene, child, &mut visiting);
    let Some(parent_node) = scene.nodes.get(&parent) else {
        return Err("Target node no longer exists.");
    };

    match &parent_node.data {
        NodeData::Operation { .. } | NodeData::Sculpt { .. } | NodeData::Modifier { .. } => {
            if child_family != GraphNodeFamily::Geometry {
                return Err("Target slot only accepts geometry inputs.");
            }
        }
        NodeData::Transform { input, .. } => {
            let expected_family = input.and_then(|existing_child| {
                let mut expected_visiting = HashSet::new();
                let family = node_output_family(scene, existing_child, &mut expected_visiting);
                (family != GraphNodeFamily::Unknown).then_some(family)
            });
            if let Some(expected_family) = expected_family {
                if expected_family != child_family {
                    return Err("Input type does not match this transform chain.");
                }
            } else if child_family == GraphNodeFamily::Unknown {
                return Err("Unable to resolve source output type.");
            }
        }
        NodeData::Primitive { .. } | NodeData::Light { .. } => {
            return Err("Target node does not expose input slots.");
        }
    }

    Ok(())
}

fn validate_connection_target(
    scene: &Scene,
    source_node: NodeId,
    parent: NodeId,
    slot: GraphInputSlot,
) -> Result<(), String> {
    if parent == source_node {
        return Err("Cannot connect a node to itself.".to_string());
    }
    if !scene.nodes.contains_key(&parent) {
        return Err("Target node no longer exists.".to_string());
    }
    if !scene.nodes.contains_key(&source_node) {
        return Err("Source node no longer exists.".to_string());
    }
    if graph_slot_child(scene, parent, slot).is_none() {
        return Err(format!("Target does not expose a {} input slot.", slot.label()));
    }
    if scene.is_descendant(parent, source_node) {
        return Err("Connection would create a cycle.".to_string());
    }
    if let Err(message) = child_allowed_for_parent(scene, parent, source_node) {
        return Err(message.to_string());
    }
    Ok(())
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn drag_distance(origin: [f32; 2], pointer: [f32; 2]) -> f32 {
    distance(origin, pointer)
}

fn reset_pointer_state(state: &mut NodeGraphPointerState) {
    state.mode = PointerMode::None;
    state.press_origin = None;
    state.last_pointer = None;
    state.connection_target = None;
}

#[cfg(test)]
mod tests {
    use super::{
        clear_graph_transient_state, compute_marquee_rect, drag_distance, graph_selection_action,
        node_graph_action_requires_viewport_dirty, preview_pointer_for_candidate,
        ConnectionTargetCandidate,
    };
    use crate::app::actions::Action;
    use crate::app::state::{
        GraphInputSlot, NodeGraphEdgeSelection, NodeGraphSocketKind, NodeGraphViewState,
    };
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

    #[test]
    fn drag_threshold_distance_is_measured_in_screen_pixels() {
        assert!(drag_distance([0.0, 0.0], [5.0, 0.0]) > 4.0);
        assert!(drag_distance([0.0, 0.0], [2.0, 2.0]) < 4.0);
    }

    #[test]
    fn marquee_rect_normalizes_negative_drag_direction() {
        let rect = compute_marquee_rect([100.0, 80.0], [40.0, 30.0], true);
        assert_eq!(rect.x, 40.0);
        assert_eq!(rect.y, 30.0);
        assert_eq!(rect.width, 60.0);
        assert_eq!(rect.height, 50.0);
        assert!(rect.additive);
    }

    #[test]
    fn preview_pointer_snaps_only_for_valid_target() {
        let pointer = [200.0, 150.0];
        let valid_target = ConnectionTargetCandidate {
            parent: 3,
            slot: GraphInputSlot::Input,
            socket_kind: NodeGraphSocketKind::Input,
            center_screen: [120.0, 88.0],
            valid: true,
            invalid_reason: None,
        };
        let invalid_target = ConnectionTargetCandidate {
            valid: false,
            ..valid_target.clone()
        };
        assert_eq!(
            preview_pointer_for_candidate(pointer, Some(&valid_target)),
            [120.0, 88.0]
        );
        assert_eq!(
            preview_pointer_for_candidate(pointer, Some(&invalid_target)),
            pointer
        );
        assert_eq!(preview_pointer_for_candidate(pointer, None), pointer);
    }

    #[test]
    fn cancel_interaction_clears_preview_marquee_and_edge_selection() {
        let mut view = NodeGraphViewState::default();
        view.connection_preview = Some(crate::app::state::NodeGraphConnectionPreview {
            source_node: 2,
            pointer_screen: [4.0, 9.0],
        });
        view.selected_edge = Some(NodeGraphEdgeSelection {
            parent: 7,
            slot: GraphInputSlot::Left,
        });
        view.hovered_edge = Some(NodeGraphEdgeSelection {
            parent: 8,
            slot: GraphInputSlot::Input,
        });
        view.marquee_rect = Some(crate::app::state::NodeGraphMarqueeRect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            additive: false,
        });

        assert!(clear_graph_transient_state(&mut view, true));
        assert!(view.connection_preview.is_none());
        assert!(view.selected_edge.is_none());
        assert!(view.hovered_edge.is_none());
        assert!(view.marquee_rect.is_none());
    }
}
