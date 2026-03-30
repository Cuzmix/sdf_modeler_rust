use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{NodeGraphSlotModel, NodeGraphSocketStateModel, ShellSnapshot};
use crate::app::slint_frontend::{
    NodeGraphEdgeView, NodeGraphGridDotView, NodeGraphNodeView, NodeGraphPanelState,
    NodeGraphSlotView, NodeGraphSocketStateView,
};

pub(super) fn build_node_graph_panel_state(snapshot: &ShellSnapshot) -> NodeGraphPanelState {
    NodeGraphPanelState {
        pan_x: snapshot.node_graph.pan_x,
        pan_y: snapshot.node_graph.pan_y,
        zoom: snapshot.node_graph.zoom,
        grid_offset_x: snapshot.node_graph.grid_offset_x,
        grid_offset_y: snapshot.node_graph.grid_offset_y,
        nodes: Rc::new(VecModel::from(
            snapshot
                .node_graph
                .nodes
                .iter()
                .map(|node| NodeGraphNodeView {
                    label: node.label.clone().into(),
                    kind_label: node.kind_label.clone().into(),
                    x: node.x,
                    y: node.y,
                    width: node.width,
                    height: node.height,
                    selected: node.selected,
                    has_left_input: node.has_left_input,
                    has_right_input: node.has_right_input,
                    has_input: node.has_input,
                    has_output: node.has_output,
                    output_state: node_graph_socket_state_view(node.output_state),
                    left_input_state: node_graph_socket_state_view(node.left_input_state),
                    right_input_state: node_graph_socket_state_view(node.right_input_state),
                    input_state: node_graph_socket_state_view(node.input_state),
                })
                .collect::<Vec<_>>(),
        ))
        .into(),
        edges: Rc::new(VecModel::from(
            snapshot
                .node_graph
                .edges
                .iter()
                .map(|edge| NodeGraphEdgeView {
                    path_commands: edge.path_commands.clone().into(),
                    midpoint_x: edge.midpoint_x,
                    midpoint_y: edge.midpoint_y,
                    selected: edge.selected,
                    hovered: edge.hovered,
                    deemphasized: edge.deemphasized,
                    slot: node_graph_slot_view(edge.slot),
                })
                .collect::<Vec<_>>(),
        ))
        .into(),
        grid_dots: Rc::new(VecModel::from(
            snapshot
                .node_graph
                .grid_dots
                .iter()
                .map(|dot| NodeGraphGridDotView { x: dot.x, y: dot.y })
                .collect::<Vec<_>>(),
        ))
        .into(),
        preview_path_commands: snapshot.node_graph.preview_path_commands.clone().into(),
        preview_visible: snapshot.node_graph.preview_visible,
        can_disconnect_selected_edge: snapshot.node_graph.can_disconnect_selected_edge,
        marquee_visible: snapshot.node_graph.marquee_visible,
        marquee_x: snapshot.node_graph.marquee_x,
        marquee_y: snapshot.node_graph.marquee_y,
        marquee_width: snapshot.node_graph.marquee_width,
        marquee_height: snapshot.node_graph.marquee_height,
        marquee_additive: snapshot.node_graph.marquee_additive,
    }
}

fn node_graph_slot_view(slot: NodeGraphSlotModel) -> NodeGraphSlotView {
    match slot {
        NodeGraphSlotModel::Left => NodeGraphSlotView::Left,
        NodeGraphSlotModel::Right => NodeGraphSlotView::Right,
        NodeGraphSlotModel::Input => NodeGraphSlotView::Input,
    }
}

fn node_graph_socket_state_view(state: NodeGraphSocketStateModel) -> NodeGraphSocketStateView {
    match state {
        NodeGraphSocketStateModel::Hidden => NodeGraphSocketStateView::Hidden,
        NodeGraphSocketStateModel::Idle => NodeGraphSocketStateView::Idle,
        NodeGraphSocketStateModel::SourceActive => NodeGraphSocketStateView::SourceActive,
        NodeGraphSocketStateModel::TargetValid => NodeGraphSocketStateView::TargetValid,
        NodeGraphSocketStateModel::TargetInvalid => NodeGraphSocketStateView::TargetInvalid,
    }
}
