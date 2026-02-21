use std::collections::HashSet;

use eframe::egui;

use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode};

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(200, 200, 210);

pub fn draw(ui: &mut egui::Ui, scene: &Scene, selected: &mut Option<NodeId>) {
    ui.heading("Scene Tree");
    ui.separator();

    if scene.nodes.is_empty() {
        ui.label("Empty scene");
        return;
    }

    let tops = scene.top_level_nodes();
    if tops.is_empty() {
        ui.label("Empty scene");
        return;
    }

    let mut visited = HashSet::new();
    for id in tops {
        draw_node_recursive(ui, scene, id, selected, &mut visited);
    }
}

fn draw_node_recursive(
    ui: &mut egui::Ui,
    scene: &Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(id) {
        ui.label(format!("  (cycle: #{})", id));
        return;
    }

    let Some(node) = scene.nodes.get(&id) else {
        ui.label(format!("  (missing: #{})", id));
        return;
    };

    let is_selected = *selected == Some(id);

    match &node.data {
        NodeData::Operation { left, right, .. } => {
            let header_text = format_node_label(node);
            let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };

            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&header_text).color(color),
            )
            .default_open(true)
            .id_salt(id)
            .show(ui, |ui| {
                if let Some(left) = *left {
                    draw_node_recursive(ui, scene, left, selected, visited);
                } else {
                    ui.label("  (empty)");
                }
                if let Some(right) = *right {
                    draw_node_recursive(ui, scene, right, selected, visited);
                } else {
                    ui.label("  (empty)");
                }
            });

            if header.header_response.clicked() {
                *selected = Some(id);
            }
        }
        NodeData::Primitive { .. } => {
            draw_leaf_item(ui, node, id, selected);
        }
        NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => {
            let header_text = format_node_label(node);
            let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };
            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&header_text).color(color),
            )
            .default_open(true)
            .id_salt(id)
            .show(ui, |ui| {
                if let Some(input) = *input {
                    draw_node_recursive(ui, scene, input, selected, visited);
                } else {
                    ui.label("  (empty)");
                }
            });
            if header.header_response.clicked() {
                *selected = Some(id);
            }
        }
    }
}

fn draw_leaf_item(
    ui: &mut egui::Ui,
    node: &SceneNode,
    id: NodeId,
    selected: &mut Option<NodeId>,
) {
    let is_selected = *selected == Some(id);
    let label_text = format_node_label(node);
    let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };

    let label = egui::Label::new(egui::RichText::new(&label_text).color(color))
        .sense(egui::Sense::click());

    if ui.add(label).clicked() {
        *selected = Some(id);
    }
}

fn format_node_label(node: &SceneNode) -> String {
    let badge = match &node.data {
        NodeData::Primitive { kind, .. } => kind.badge(),
        NodeData::Operation { op, .. } => op.badge(),
        NodeData::Sculpt { .. } => "[Scl]",
        NodeData::Transform { kind, .. } => kind.badge(),
    };
    format!("{} {}", badge, node.name)
}
