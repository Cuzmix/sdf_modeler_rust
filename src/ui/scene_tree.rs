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

    // Draw the root tree
    if let Some(root_id) = scene.root {
        let mut visited = HashSet::new();
        draw_node_recursive(ui, scene, root_id, selected, &mut visited, true);
    } else {
        ui.label("No root node set");
    }

    // Orphans section
    let reachable = scene.reachable_from_root();
    let mut orphans: Vec<NodeId> = scene
        .nodes
        .keys()
        .filter(|id| !reachable.contains(id))
        .cloned()
        .collect();
    orphans.sort();

    if !orphans.is_empty() {
        ui.separator();
        egui::CollapsingHeader::new("Orphans")
            .default_open(true)
            .show(ui, |ui| {
                for id in &orphans {
                    draw_leaf_item(ui, scene, *id, selected);
                }
            });
    }
}

fn draw_node_recursive(
    ui: &mut egui::Ui,
    scene: &Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    visited: &mut HashSet<NodeId>,
    is_root: bool,
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
            let header_text = format_node_label(node, is_root);
            let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };

            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&header_text).color(color),
            )
            .default_open(true)
            .id_salt(id)
            .show(ui, |ui| {
                let left = *left;
                let right = *right;
                draw_node_recursive(ui, scene, left, selected, visited, false);
                draw_node_recursive(ui, scene, right, selected, visited, false);
            });

            if header.header_response.clicked() {
                *selected = Some(id);
            }
        }
        NodeData::Primitive { .. } => {
            draw_leaf_item_with_root(ui, node, id, selected, is_root);
        }
        NodeData::Sculpt { input, .. } => {
            let header_text = format_node_label(node, is_root);
            let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };
            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&header_text).color(color),
            )
            .default_open(true)
            .id_salt(id)
            .show(ui, |ui| {
                let input = *input;
                draw_node_recursive(ui, scene, input, selected, visited, false);
            });
            if header.header_response.clicked() {
                *selected = Some(id);
            }
        }
    }
}

fn draw_leaf_item(
    ui: &mut egui::Ui,
    scene: &Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
) {
    let Some(node) = scene.nodes.get(&id) else {
        return;
    };
    draw_leaf_item_with_root(ui, node, id, selected, false);
}

fn draw_leaf_item_with_root(
    ui: &mut egui::Ui,
    node: &SceneNode,
    id: NodeId,
    selected: &mut Option<NodeId>,
    is_root: bool,
) {
    let is_selected = *selected == Some(id);
    let label_text = format_node_label(node, is_root);
    let color = if is_selected { COLOR_SELECTED } else { COLOR_NORMAL };

    let label = egui::Label::new(egui::RichText::new(&label_text).color(color))
        .sense(egui::Sense::click());

    if ui.add(label).clicked() {
        *selected = Some(id);
    }
}

fn format_node_label(node: &SceneNode, is_root: bool) -> String {
    let badge = match &node.data {
        NodeData::Primitive { kind, .. } => kind.badge(),
        NodeData::Operation { op, .. } => op.badge(),
        NodeData::Sculpt { .. } => "[Scl]",
    };
    let root_marker = if is_root { " (R)" } else { "" };
    format!("{} {}{}", badge, node.name, root_marker)
}
