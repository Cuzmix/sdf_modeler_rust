use std::collections::HashSet;

use eframe::egui;

use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode};

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(200, 200, 210);
const COLOR_HIDDEN: egui::Color32 = egui::Color32::from_gray(100);
const COLOR_DROP_TARGET: egui::Color32 = egui::Color32::from_rgb(80, 140, 255);

pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    drag_state: &mut Option<NodeId>,
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
        draw_node_recursive(ui, scene, id, selected, renaming, rename_buf, &mut visited, drag_state);
    }

    // "Drop here to make top-level" area when dragging
    if drag_state.is_some() {
        ui.add_space(4.0);
        let drop_response = ui.allocate_response(
            egui::vec2(ui.available_width(), 20.0),
            egui::Sense::hover(),
        );
        let is_hovered = drop_response.hovered();
        let rect = drop_response.rect;
        let color = if is_hovered { COLOR_DROP_TARGET } else { egui::Color32::from_gray(60) };
        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(1.5, color));
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drop here for top-level",
            egui::FontId::proportional(11.0),
            color,
        );

        // Handle drop: detach from parent, making it top-level
        if is_hovered && ui.input(|i| i.pointer.any_released()) {
            if let Some(dragged_id) = drag_state.take() {
                detach_from_parent(scene, dragged_id);
            }
        }
    }

    // Clear drag state when pointer is released (even if not over a target)
    if ui.input(|i| i.pointer.any_released()) {
        *drag_state = None;
    }
}

/// Lightweight node info extracted before mutable scene operations.
struct NodeInfo {
    label: String,
    is_selected: bool,
    is_hidden: bool,
    is_leaf: bool,
    children: Vec<Option<NodeId>>,
}

fn extract_info(scene: &Scene, id: NodeId, selected: Option<NodeId>) -> Option<NodeInfo> {
    let node = scene.nodes.get(&id)?;
    let label = format_node_label(node);
    let is_selected = selected == Some(id);
    let is_hidden = scene.is_hidden(id);
    let (is_leaf, children) = match &node.data {
        NodeData::Primitive { .. } => (true, vec![]),
        NodeData::Operation { left, right, .. } => (false, vec![*left, *right]),
        NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } | NodeData::Modifier { input, .. } => (false, vec![*input]),
    };
    Some(NodeInfo { label, is_selected, is_hidden, is_leaf, children })
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
        let hidden = scene.is_hidden(id);
        if ui.button(if hidden { "Show" } else { "Hide" }).clicked() {
            scene.toggle_visibility(id);
            ui.close_menu();
        }
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

/// Check if `target_id` is a valid drop target for `dragged_id`.
fn is_valid_drop_target(scene: &Scene, target_id: NodeId, dragged_id: NodeId) -> bool {
    // Can't drop on self
    if target_id == dragged_id {
        return false;
    }
    // Can't drop on a descendant (would create a cycle)
    if is_descendant(scene, target_id, dragged_id) {
        return false;
    }
    // Target must accept children and have a free slot
    let Some(target_node) = scene.nodes.get(&target_id) else {
        return false;
    };
    match &target_node.data {
        NodeData::Primitive { .. } => false,  // Primitives can't have children
        NodeData::Operation { left, right, .. } => left.is_none() || right.is_none(),
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => input.is_none(),
    }
}

/// Returns true if `candidate` is a descendant of `ancestor`.
fn is_descendant(scene: &Scene, candidate: NodeId, ancestor: NodeId) -> bool {
    let Some(node) = scene.nodes.get(&ancestor) else {
        return false;
    };
    for child_id in node.data.children() {
        if child_id == candidate || is_descendant(scene, candidate, child_id) {
            return true;
        }
    }
    false
}

/// Detach a node from its parent (null out the reference).
fn detach_from_parent(scene: &mut Scene, child_id: NodeId) {
    let parent_map = scene.build_parent_map();
    let Some(&parent_id) = parent_map.get(&child_id) else {
        return; // Already top-level
    };
    if let Some(parent) = scene.nodes.get_mut(&parent_id) {
        match &mut parent.data {
            NodeData::Operation { left, right, .. } => {
                if *left == Some(child_id) {
                    *left = None;
                }
                if *right == Some(child_id) {
                    *right = None;
                }
            }
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => {
                if *input == Some(child_id) {
                    *input = None;
                }
            }
            _ => {}
        }
    }
}

/// Reparent: detach from old parent, attach to first free slot of new parent.
fn perform_reparent(scene: &mut Scene, dragged_id: NodeId, target_id: NodeId) {
    detach_from_parent(scene, dragged_id);
    if let Some(target) = scene.nodes.get_mut(&target_id) {
        match &mut target.data {
            NodeData::Operation { left, right, .. } => {
                if left.is_none() {
                    *left = Some(dragged_id);
                } else if right.is_none() {
                    *right = Some(dragged_id);
                }
            }
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => {
                if input.is_none() {
                    *input = Some(dragged_id);
                }
            }
            _ => {}
        }
    }
}

fn handle_drag_drop(
    ui: &egui::Ui,
    response: &egui::Response,
    scene: &mut Scene,
    id: NodeId,
    drag_state: &mut Option<NodeId>,
) {
    // Drag source: start dragging on drag
    if response.dragged() && drag_state.is_none() {
        *drag_state = Some(id);
    }

    // Drop target: highlight and handle release
    if let Some(dragged_id) = *drag_state {
        if dragged_id != id && response.hovered() {
            if is_valid_drop_target(scene, id, dragged_id) {
                // Visual highlight
                ui.painter().rect_stroke(
                    response.rect,
                    2.0,
                    egui::Stroke::new(2.0, COLOR_DROP_TARGET),
                );

                // Handle drop on release
                if ui.input(|i| i.pointer.any_released()) {
                    perform_reparent(scene, dragged_id, id);
                    *drag_state = None;
                }
            }
        }
    }
}

fn draw_node_recursive(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    visited: &mut HashSet<NodeId>,
    drag_state: &mut Option<NodeId>,
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

    let node_color = if info.is_hidden {
        COLOR_HIDDEN
    } else if info.is_selected {
        COLOR_SELECTED
    } else {
        COLOR_NORMAL
    };

    if info.is_leaf {
        ui.horizontal(|ui| {
            let mut visible = !info.is_hidden;
            if ui.checkbox(&mut visible, "").changed() {
                scene.toggle_visibility(id);
            }
            let label = egui::Label::new(egui::RichText::new(&info.label).color(node_color))
                .sense(egui::Sense::click_and_drag());
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
            handle_drag_drop(ui, &response, scene, id, drag_state);
            node_context_menu(&response, ui, scene, id, selected, renaming, rename_buf);
        });
    } else {
        // Visibility checkbox before collapsing header
        ui.horizontal(|ui| {
            let mut visible = !info.is_hidden;
            if ui.checkbox(&mut visible, "").changed() {
                scene.toggle_visibility(id);
            }
            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&info.label).color(node_color),
            )
            .default_open(true)
            .id_salt(id)
            .show(ui, |ui| {
                for child_opt in &info.children {
                    if let Some(child_id) = child_opt {
                        draw_node_recursive(ui, scene, *child_id, selected, renaming, rename_buf, visited, drag_state);
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
            handle_drag_drop(ui, &header.header_response, scene, id, drag_state);
            node_context_menu(&header.header_response, ui, scene, id, selected, renaming, rename_buf);
        });
    }
}

fn format_node_label(node: &SceneNode) -> String {
    let badge = match &node.data {
        NodeData::Primitive { kind, .. } => kind.badge(),
        NodeData::Operation { op, .. } => op.badge(),
        NodeData::Sculpt { .. } => "[Scl]",
        NodeData::Transform { kind, .. } => kind.badge(),
        NodeData::Modifier { kind, .. } => kind.badge(),
    };
    format!("{} {}", badge, node.name)
}
