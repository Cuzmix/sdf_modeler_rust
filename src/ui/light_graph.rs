use std::collections::HashSet;

use crate::app::actions::ActionSink;
use crate::graph::scene::{NodeId, Scene};
use crate::ui::node_graph::{self, NodeGraphState};

/// Dedicated light node graph panel.
/// Reuses the full node graph UI/UX (finder, ports, minimap, inline editing),
/// filtered to light-related scene nodes only.
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    state: &mut NodeGraphState,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
    actions: &mut ActionSink,
) {
    node_graph::draw_lights(ui, scene, state, selected, selected_set, actions);
}
