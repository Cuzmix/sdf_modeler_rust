use std::collections::HashSet;

use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::egui_theme::{resolve_scaled_font_id, AppTextRole};
use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode};
use crate::ui::chrome::{self, BadgeTone};

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(200, 200, 210);
const COLOR_HIDDEN: egui::Color32 = egui::Color32::from_gray(100);
const COLOR_DROP_TARGET: egui::Color32 = egui::Color32::from_rgb(80, 140, 255);

// Type-coded dot colors
const DOT_PRIMITIVE: egui::Color32 = egui::Color32::from_rgb(90, 140, 255); // blue
const DOT_OPERATION: egui::Color32 = egui::Color32::from_rgb(80, 200, 120); // green
const DOT_SCULPT: egui::Color32 = egui::Color32::from_rgb(240, 160, 60); // orange
const DOT_TRANSFORM: egui::Color32 = egui::Color32::from_rgb(180, 120, 240); // purple
const DOT_MODIFIER: egui::Color32 = egui::Color32::from_rgb(230, 210, 60); // yellow
const DOT_LIGHT: egui::Color32 = egui::Color32::from_rgb(255, 220, 50); // warm yellow

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    selected_set: &mut std::collections::HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    drag_state: &mut Option<NodeId>,
    actions: &mut ActionSink,
    search_filter: &mut String,
    active_light_ids: &std::collections::HashSet<NodeId>,
    soloed_light: Option<NodeId>,
) {
    chrome::panel_header(
        ui,
        "Scene Tree",
        "Filter, inspect, and reorganize the scene hierarchy from a cleaner docked view.",
    );
    ui.add_space(10.0);

    chrome::section_card(
        ui,
        "Search",
        "Find nodes by name across the current scene.",
        |ui| {
            ui.horizontal(|ui| {
                chrome::search_field(ui, "scene_tree_filter", search_filter, "Filter nodes...");
                if !search_filter.is_empty() && ui.button("Clear").clicked() {
                    search_filter.clear();
                }
            });
        },
    );

    ui.add_space(10.0);

    if scene.nodes.is_empty() {
        chrome::empty_state(
            ui,
            "Scene is empty",
            "Create a primitive, import geometry, or paste an existing node to begin.",
        );
        return;
    }

    let tops = scene.top_level_nodes();
    if tops.is_empty() {
        chrome::empty_state(
            ui,
            "No top-level nodes",
            "Drag a node back to the root or create a new one to rebuild the hierarchy.",
        );
        return;
    }

    // If search filter is active, show flat filtered list
    if !search_filter.is_empty() {
        let filter_lower = search_filter.to_lowercase();
        let matching: Vec<NodeId> = scene
            .nodes
            .keys()
            .filter(|id| {
                if let Some(node) = scene.nodes.get(id) {
                    node.name.to_lowercase().contains(&filter_lower)
                } else {
                    false
                }
            })
            .copied()
            .collect();

        if matching.is_empty() {
            chrome::empty_state(
                ui,
                "No matching nodes",
                "Try a broader search term or clear the filter to browse the whole tree.",
            );
            return;
        }

        for id in matching {
            draw_flat_node(
                ui,
                scene,
                id,
                selected,
                selected_set,
                renaming,
                rename_buf,
                actions,
                active_light_ids,
                soloed_light,
            );
        }
        return;
    }

    let mut visited = HashSet::new();
    for id in tops {
        draw_node_recursive(
            ui,
            scene,
            id,
            selected,
            selected_set,
            renaming,
            rename_buf,
            &mut visited,
            drag_state,
            actions,
            active_light_ids,
            soloed_light,
        );
    }

    // "Drop here to make top-level" area when dragging
    if drag_state.is_some() {
        ui.add_space(6.0);
        let drop_response = chrome::inset_frame(ui)
            .show(ui, |ui| {
                ui.add_sized(
                    [ui.available_width(), 24.0],
                    egui::Label::new(
                        egui::RichText::new("Drop here to move the node to the top level")
                            .color(chrome::tokens(ui).muted_text),
                    ),
                )
            })
            .response;
        let is_hovered = drop_response.hovered();
        let rect = drop_response.rect;
        let color = if is_hovered {
            COLOR_DROP_TARGET
        } else {
            chrome::tokens(ui).border
        };
        ui.painter().rect_stroke(
            rect,
            chrome::tokens(ui).radius_md,
            egui::Stroke::new(1.5, color),
        );
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drop here to move the node to the top level",
            resolve_scaled_font_id(ui.style().as_ref(), AppTextRole::SceneLabel, 11.0),
            color,
        );

        // Handle drop: detach from parent, making it top-level
        if is_hovered && ui.input(|i| i.pointer.any_released()) {
            if let Some(dragged_id) = drag_state.take() {
                scene.detach_from_parent(dragged_id);
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

fn extract_info(
    scene: &Scene,
    id: NodeId,
    selected_set: &std::collections::HashSet<NodeId>,
) -> Option<NodeInfo> {
    let node = scene.nodes.get(&id)?;
    let label = format_node_label(node);
    let is_selected = selected_set.contains(&id);
    let is_hidden = scene.is_hidden(id);
    let (is_leaf, children) = match &node.data {
        NodeData::Primitive { .. } | NodeData::Light { .. } => (true, vec![]),
        NodeData::Operation { left, right, .. } => (false, vec![*left, *right]),
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => (false, vec![*input]),
    };
    Some(NodeInfo {
        label,
        is_selected,
        is_hidden,
        is_leaf,
        children,
    })
}

#[allow(clippy::too_many_arguments)]
fn node_context_menu(
    response: &egui::Response,
    _ui_ctx: &egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    _selected: &mut Option<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
) {
    response.context_menu(|ui| {
        let hidden = scene.is_hidden(id);
        if ui.button(if hidden { "Show" } else { "Hide" }).clicked() {
            actions.push(Action::ToggleVisibility(id));
            ui.close_menu();
        }
        let locked = scene.nodes.get(&id).is_some_and(|n| n.locked);
        if ui.button(if locked { "Unlock" } else { "Lock" }).clicked() {
            actions.push(Action::ToggleLock(id));
            ui.close_menu();
        }
        if ui.button("Rename").clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
            ui.close_menu();
        }
        if ui.button("Save as Preset...").clicked() {
            actions.push(Action::SaveNodePreset(id));
            ui.close_menu();
        }
        let delete_enabled = !locked;
        if ui
            .add_enabled(delete_enabled, egui::Button::new("Delete"))
            .clicked()
        {
            actions.push(Action::DeleteNode(id));
            if *renaming == Some(id) {
                *renaming = None;
            }
            ui.close_menu();
        }
    });
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
        if dragged_id != id && response.hovered() && scene.is_valid_drop_target(id, dragged_id) {
            // Visual highlight
            ui.painter().rect_stroke(
                response.rect,
                chrome::tokens(ui).radius_md,
                egui::Stroke::new(2.0, COLOR_DROP_TARGET),
            );

            // Handle drop on release
            if ui.input(|i| i.pointer.any_released()) {
                scene.reparent(dragged_id, id);
                *drag_state = None;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_node_recursive(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    selected_set: &mut std::collections::HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    visited: &mut HashSet<NodeId>,
    drag_state: &mut Option<NodeId>,
    actions: &mut ActionSink,
    active_light_ids: &std::collections::HashSet<NodeId>,
    soloed_light: Option<NodeId>,
) {
    if !visited.insert(id) {
        ui.label(format!("  (cycle: #{})", id));
        return;
    }

    let Some(info) = extract_info(scene, id, selected_set) else {
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
                    scene.mark_data_changed();
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

    // Pre-fetch type color before mutable borrows
    // Dim inactive light nodes (lights beyond the MAX_SCENE_LIGHTS nearest to camera)
    let is_inactive_light = scene.nodes.get(&id).is_some_and(|n| {
        matches!(n.data, NodeData::Light { .. }) && !active_light_ids.contains(&id)
    });
    let type_dot_color = if is_inactive_light {
        egui::Color32::from_gray(100) // dimmed gray for inactive lights
    } else {
        scene
            .nodes
            .get(&id)
            .map(|n| node_type_color(&n.data))
            .unwrap_or(COLOR_NORMAL)
    };

    if info.is_leaf {
        let row = chrome::item_frame(ui, info.is_selected).show(ui, |ui| {
            ui.horizontal(|ui| {
            let mut visible = !info.is_hidden;
            if ui.checkbox(&mut visible, "").changed() {
                scene.toggle_visibility(id);
            }
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, type_dot_color);

                let label = egui::Label::new(egui::RichText::new(&info.label).color(node_color))
                    .sense(egui::Sense::click_and_drag())
                    .truncate();
                let response = ui.add_sized([ui.available_width() - 72.0, 20.0], label);

                if scene
                    .nodes
                    .get(&id)
                    .is_some_and(|node| matches!(node.data, NodeData::Light { .. }))
                {
                    chrome::badge(
                        ui,
                        if active_light_ids.contains(&id) {
                            BadgeTone::Success
                        } else {
                            BadgeTone::Muted
                        },
                        "Light",
                    );
                    let is_soloed = soloed_light == Some(id);
                    if ui
                        .small_button(if is_soloed { "Solo" } else { "S" })
                        .clicked()
                    {
                        actions.push(Action::SoloLight(Some(id)));
                    }
                }

                response
            })
            .inner
        });
        let response = row.inner;
        if response.clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            if ctrl {
                if selected_set.remove(&id) {
                    if *selected == Some(id) {
                        *selected = selected_set.iter().copied().min();
                    }
                } else {
                    selected_set.insert(id);
                    *selected = Some(id);
                }
            } else {
                *selected = Some(id);
                selected_set.clear();
                selected_set.insert(id);
            }
        }
        if response.double_clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
        }
        handle_drag_drop(ui, &response, scene, id, drag_state);
        node_context_menu(
            &response, ui, scene, id, selected, renaming, rename_buf, actions,
        );
    } else {
        let row = chrome::item_frame(ui, info.is_selected).show(ui, |ui| {
            ui.horizontal(|ui| {
            let mut visible = !info.is_hidden;
            if ui.checkbox(&mut visible, "").changed() {
                scene.toggle_visibility(id);
            }
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, type_dot_color);

                egui::CollapsingHeader::new(egui::RichText::new(&info.label).color(node_color))
                    .default_open(true)
                    .id_salt(id)
                    .show(ui, |ui| {
                        for child_opt in &info.children {
                            if let Some(child_id) = child_opt {
                                draw_node_recursive(
                                    ui,
                                    scene,
                                    *child_id,
                                    selected,
                                    selected_set,
                                    renaming,
                                    rename_buf,
                                    visited,
                                    drag_state,
                                    actions,
                                    active_light_ids,
                                    soloed_light,
                                );
                            } else {
                                ui.label("  (empty)");
                            }
                        }
                    })
            })
            .inner
        });

        let header = row.inner;
        if header.header_response.clicked() {
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            if ctrl {
                if selected_set.remove(&id) {
                    if *selected == Some(id) {
                        *selected = selected_set.iter().copied().min();
                    }
                } else {
                    selected_set.insert(id);
                    *selected = Some(id);
                }
            } else {
                *selected = Some(id);
                selected_set.clear();
                selected_set.insert(id);
            }
        }
        if header.header_response.double_clicked() {
            if let Some(node) = scene.nodes.get(&id) {
                *rename_buf = node.name.clone();
                *renaming = Some(id);
            }
        }
        handle_drag_drop(ui, &header.header_response, scene, id, drag_state);
        node_context_menu(
            &header.header_response,
            ui,
            scene,
            id,
            selected,
            renaming,
            rename_buf,
            actions,
        );
    }
}

fn format_node_label(node: &SceneNode) -> String {
    let badge = match &node.data {
        NodeData::Primitive { kind, .. } => kind.badge(),
        NodeData::Operation { op, .. } => op.badge(),
        NodeData::Sculpt { .. } => "[Scl]",
        NodeData::Transform { .. } => "[Xfm]",
        NodeData::Modifier { kind, .. } => kind.badge(),
        NodeData::Light { light_type, .. } => light_type.badge(),
    };
    if node.locked {
        format!("{} \u{1F512} {}", badge, node.name)
    } else {
        format!("{} {}", badge, node.name)
    }
}

const DOT_LIGHT_NEGATIVE: egui::Color32 = egui::Color32::from_rgb(255, 80, 80); // red for negative/subtractive
const DOT_LIGHT_ARRAY: egui::Color32 = egui::Color32::from_rgb(200, 180, 255); // light purple for array

fn node_type_color(data: &NodeData) -> egui::Color32 {
    match data {
        NodeData::Primitive { .. } => DOT_PRIMITIVE,
        NodeData::Operation { .. } => DOT_OPERATION,
        NodeData::Sculpt { .. } => DOT_SCULPT,
        NodeData::Transform { .. } => DOT_TRANSFORM,
        NodeData::Modifier { .. } => DOT_MODIFIER,
        NodeData::Light {
            light_type,
            intensity,
            ..
        } => {
            if matches!(light_type, crate::graph::scene::LightType::Array) {
                DOT_LIGHT_ARRAY
            } else if *intensity < 0.0 {
                DOT_LIGHT_NEGATIVE
            } else {
                DOT_LIGHT
            }
        }
    }
}

/// Flat node row (used in search results).
#[allow(clippy::too_many_arguments)]
fn draw_flat_node(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    id: NodeId,
    selected: &mut Option<NodeId>,
    selected_set: &mut std::collections::HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
    active_light_ids: &std::collections::HashSet<NodeId>,
    soloed_light: Option<NodeId>,
) {
    let Some(info) = extract_info(scene, id, selected_set) else {
        return;
    };
    let is_inactive_light = scene.nodes.get(&id).is_some_and(|n| {
        matches!(n.data, NodeData::Light { .. }) && !active_light_ids.contains(&id)
    });
    let dot_color = if is_inactive_light {
        egui::Color32::from_gray(100)
    } else {
        scene
            .nodes
            .get(&id)
            .map(|n| node_type_color(&n.data))
            .unwrap_or(COLOR_NORMAL)
    };

    // Inline rename editor
    if *renaming == Some(id) {
        ui.horizontal(|ui| {
            let te = ui.text_edit_singleline(rename_buf);
            if te.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Some(node) = scene.nodes.get_mut(&id) {
                    node.name = rename_buf.clone();
                    scene.mark_data_changed();
                }
                *renaming = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                *renaming = None;
            }
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

    let row = chrome::item_frame(ui, info.is_selected).show(ui, |ui| {
        ui.horizontal(|ui| {
            let (dot_rect, _) =
                ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter()
                .circle_filled(dot_rect.center(), 4.0, dot_color);

            let label = egui::Label::new(egui::RichText::new(&info.label).color(node_color))
                .sense(egui::Sense::click())
                .truncate();
            let response = ui.add_sized([ui.available_width() - 72.0, 20.0], label);
            if scene
                .nodes
                .get(&id)
                .is_some_and(|node| matches!(node.data, NodeData::Light { .. }))
            {
                chrome::badge(
                    ui,
                    if active_light_ids.contains(&id) {
                        BadgeTone::Success
                    } else {
                        BadgeTone::Muted
                    },
                    "Light",
                );
                let is_soloed = soloed_light == Some(id);
                if ui
                    .small_button(if is_soloed { "Solo" } else { "S" })
                    .clicked()
                {
                    actions.push(Action::SoloLight(Some(id)));
                }
            }
            response
        })
        .inner
    });
    let response = row.inner;
    if response.clicked() {
        let ctrl = ui.input(|i| i.modifiers.ctrl);
        if ctrl {
            if selected_set.remove(&id) {
                if *selected == Some(id) {
                    *selected = selected_set.iter().copied().min();
                }
            } else {
                selected_set.insert(id);
                *selected = Some(id);
            }
        } else {
            *selected = Some(id);
            selected_set.clear();
            selected_set.insert(id);
        }
    }
    if response.double_clicked() {
        if let Some(node) = scene.nodes.get(&id) {
            *rename_buf = node.name.clone();
            *renaming = Some(id);
        }
    }
    node_context_menu(
        &response, ui, scene, id, selected, renaming, rename_buf, actions,
    );
}
