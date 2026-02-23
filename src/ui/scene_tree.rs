use std::collections::HashSet;

use eframe::egui;

use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode};

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(200, 200, 210);

pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
) {
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
        draw_node_recursive(ui, scene, id, selected, renaming, rename_buf, &mut visited);
    }
}

/// Lightweight node info extracted before mutable scene operations.
struct NodeInfo {
    label: String,
    is_selected: bool,
    is_leaf: bool,
    children: Vec<Option<NodeId>>,
}

fn extract_info(scene: &Scene, id: NodeId, selected: Option<NodeId>) -> Option<NodeInfo> {
    let node = scene.nodes.get(&id)?;
    let label = format_node_label(node);
    let is_selected = selected == Some(id);
    let (is_leaf, children) = match &node.data {
        NodeData::Primitive { .. } => (true, vec![]),
        NodeData::Operation { left, right, .. } => (false, vec![*left, *right]),
        NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => (false, vec![*input]),
    };
    Some(NodeInfo { label, is_selected, is_leaf, children })
}

fn node_context_menu(
    response: &egui::Response,
    _ui_ctx: &egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
) {
    response.context_menu(|ui| {
        if ui.button("Rename").clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
            ui.close_menu();
        }
        if ui.button("Delete").clicked() {
            scene.remove_node(id);
            if *selected == Some(id) { *selected = None; }
            if *renaming == Some(id) { *renaming = None; }
            ui.close_menu();
        }
    });
}

fn draw_node_recursive(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(id) {
        ui.label(format!("  (cycle: #{})", id));
        return;
    }

    let Some(info) = extract_info(scene, id, *selected) else {
        ui.label(format!("  (missing: #{})", id));
        return;
    };

    // Inline rename editor
    if *renaming == Some(id) {
        ui.horizontal(|ui| {
            let te = ui.text_edit_singleline(rename_buf);
            if te.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Some(node) = scene.nodes.get_mut(&id) {
                    node.name = rename_buf.clone();
                }
                *renaming = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                *renaming = None;
            }
            // Auto-focus the text edit
            te.request_focus();
        });
        return;
    }

    if info.is_leaf {
        let color = if info.is_selected { COLOR_SELECTED } else { COLOR_NORMAL };
        let label = egui::Label::new(egui::RichText::new(&info.label).color(color))
            .sense(egui::Sense::click());
        let response = ui.add(label);
        if response.clicked() {
            *selected = Some(id);
        }
        if response.double_clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
        }
        node_context_menu(&response, ui, scene, id, selected, renaming, rename_buf);
    } else {
        let color = if info.is_selected { COLOR_SELECTED } else { COLOR_NORMAL };
        let header = egui::CollapsingHeader::new(
            egui::RichText::new(&info.label).color(color),
        )
        .default_open(true)
        .id_salt(id)
        .show(ui, |ui| {
            for child_opt in &info.children {
                if let Some(child_id) = child_opt {
                    draw_node_recursive(ui, scene, *child_id, selected, renaming, rename_buf, visited);
                } else {
                    ui.label("  (empty)");
                }
            }
        });

        if header.header_response.clicked() {
            *selected = Some(id);
        }
        if header.header_response.double_clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
        }
        node_context_menu(&header.header_response, ui, scene, id, selected, renaming, rename_buf);
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
