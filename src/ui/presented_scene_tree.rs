use std::collections::HashSet;

use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::graph::presented_object::{
    presented_children, presented_top_level_objects, resolve_presented_object, PresentedObjectKind,
    PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::ui::{chips, chrome, presented_object_actions};

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 224, 160);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(218, 224, 236);
const COLOR_HIDDEN: egui::Color32 = egui::Color32::from_gray(114);
const TREE_INDENT: f32 = 14.0;
const ROW_SPACING: f32 = 6.0;

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    _drag_state: &mut Option<NodeId>,
    actions: &mut ActionSink,
    search_filter: &mut String,
    _active_light_ids: &HashSet<NodeId>,
    soloed_light: Option<NodeId>,
) {
    let selected_object =
        (*selected).and_then(|selected_id| resolve_presented_object(scene, selected_id));

    draw_quick_row(ui, scene, selected_object, actions, search_filter);
    ui.add_space(8.0);

    let roots = presented_top_level_objects(scene);
    if roots.is_empty() {
        chrome::tree_row_frame(false, true).show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.strong("Empty scene");
                ui.small("Create a primitive or bring a sculpted object in from the viewport.");
            });
        });
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if !search_filter.is_empty() {
                let filter_lower = search_filter.to_lowercase();
                let mut flattened = Vec::new();
                collect_flattened_objects(scene, &roots, &mut flattened);
                let matching: Vec<_> = flattened
                    .into_iter()
                    .filter(|object| {
                        scene
                            .nodes
                            .get(&object.host_id)
                            .is_some_and(|node| node.name.to_lowercase().contains(&filter_lower))
                    })
                    .collect();

                if matching.is_empty() {
                    chrome::tree_row_frame(false, true).show(ui, |ui| {
                        ui.small("No matching objects.");
                    });
                    return;
                }

                for object in matching {
                    draw_object_row(
                        ui,
                        scene,
                        object,
                        0,
                        false,
                        selected,
                        selected_set,
                        renaming,
                        rename_buf,
                        actions,
                        soloed_light,
                    );
                    ui.add_space(ROW_SPACING);
                }
                return;
            }

            let mut visited = HashSet::new();
            for object in roots {
                draw_object_recursive(
                    ui,
                    scene,
                    object,
                    0,
                    selected,
                    selected_set,
                    renaming,
                    rename_buf,
                    actions,
                    soloed_light,
                    &mut visited,
                );
            }
        });
}

fn draw_quick_row(
    ui: &mut egui::Ui,
    scene: &Scene,
    selected_object: Option<PresentedObjectRef>,
    actions: &mut ActionSink,
    search_filter: &mut String,
) {
    chrome::tree_row_frame(false, false).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Search").small().strong());
            ui.add(
                egui::TextEdit::singleline(search_filter)
                    .hint_text("Filter objects")
                    .desired_width(150.0),
            );
            if !search_filter.is_empty() && chrome::action_button(ui, "Clear", false).clicked() {
                search_filter.clear();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                chrome::with_action_button_style(ui, |ui| {
                    presented_object_actions::draw_object_create_menu_button(
                        ui,
                        scene,
                        selected_object,
                        actions,
                        "+ Create",
                    );
                });
            });
        });
    });
}

fn collect_flattened_objects(
    scene: &Scene,
    roots: &[PresentedObjectRef],
    out: &mut Vec<PresentedObjectRef>,
) {
    let mut visited = HashSet::new();
    let mut stack: Vec<_> = roots.iter().copied().rev().collect();

    while let Some(object) = stack.pop() {
        if !visited.insert(object.host_id) {
            continue;
        }
        out.push(object);
        let mut children = presented_children(scene, object);
        children.reverse();
        stack.extend(children);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_object_recursive(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    depth: usize,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
    soloed_light: Option<NodeId>,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(object.host_id) {
        return;
    }

    let children = presented_children(scene, object);
    let has_children = !children.is_empty();
    let expanded = has_children && object_is_expanded(ui, object.host_id);

    draw_object_row(
        ui,
        scene,
        object,
        depth,
        has_children,
        selected,
        selected_set,
        renaming,
        rename_buf,
        actions,
        soloed_light,
    );
    ui.add_space(ROW_SPACING);

    if expanded {
        for child in children {
            draw_object_recursive(
                ui,
                scene,
                child,
                depth + 1,
                selected,
                selected_set,
                renaming,
                rename_buf,
                actions,
                soloed_light,
                visited,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_object_row(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    depth: usize,
    has_children: bool,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
    soloed_light: Option<NodeId>,
) {
    let indent = depth as f32 * TREE_INDENT;

    ui.horizontal(|ui| {
        if indent > 0.0 {
            ui.add_space(indent);
        }

        let is_selected = selected_set.contains(&object.host_id) || *selected == Some(object.host_id);
        let dimmed = scene.is_hidden(object.object_root_id);

        if *renaming == Some(object.host_id) {
            chrome::tree_row_frame(is_selected, dimmed).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    if has_children {
                        draw_expand_button(ui, object.host_id);
                    } else {
                        ui.add_space(28.0);
                    }
                    draw_visibility_toggle(ui, scene, object, actions);
                    let response = ui.text_edit_singleline(rename_buf);
                    if response.lost_focus() || ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                        if let Some(node) = scene.nodes.get_mut(&object.host_id) {
                            node.name = rename_buf.clone();
                        }
                        *renaming = None;
                    }
                    if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                        *renaming = None;
                    }
                    response.request_focus();
                });
            });
            return;
        }

        chrome::tree_row_frame(is_selected, dimmed).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                if has_children {
                    draw_expand_button(ui, object.host_id);
                } else {
                    ui.add_space(28.0);
                }

                draw_visibility_toggle(ui, scene, object, actions);

                let label = egui::RichText::new(object_label(scene, object))
                    .color(object_label_color(scene, object, is_selected))
                    .strong();
                let label_response = ui.add(
                    egui::Button::new(label)
                        .frame(false)
                        .min_size(egui::vec2(120.0, 20.0)),
                );

                if label_response.clicked() {
                    apply_presented_selection(ui, object.host_id, selected, selected_set);
                }
                if label_response.double_clicked() {
                    begin_rename(scene, object.host_id, renaming, rename_buf);
                }

                object_context_menu(&label_response, scene, object, renaming, rename_buf, actions);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    draw_light_solo_button(ui, object, actions, soloed_light);
                    draw_object_add_button(ui, scene, object, actions);
                    draw_object_state_chips(ui, object);
                });
            });
        });
    });
}

fn draw_expand_button(ui: &mut egui::Ui, host_id: NodeId) {
    let expanded = object_is_expanded(ui, host_id);
    let label = if expanded { "v" } else { ">" };
    if chrome::action_button(ui, label, expanded).clicked() {
        set_object_expanded(ui, host_id, !expanded);
    }
}

fn draw_visibility_toggle(
    ui: &mut egui::Ui,
    scene: &Scene,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
) {
    let mut visible = !scene.is_hidden(object.object_root_id);
    if ui.checkbox(&mut visible, "").changed() {
        actions.push(Action::ToggleVisibility(object.object_root_id));
    }
}

fn object_is_expanded(ui: &egui::Ui, host_id: NodeId) -> bool {
    ui.ctx()
        .data_mut(|data| data.get_persisted::<bool>(object_expanded_id(host_id)).unwrap_or(true))
}

fn set_object_expanded(ui: &egui::Ui, host_id: NodeId, expanded: bool) {
    ui.ctx()
        .data_mut(|data| data.insert_persisted(object_expanded_id(host_id), expanded));
}

fn object_expanded_id(host_id: NodeId) -> egui::Id {
    egui::Id::new(("presented_object_expanded", host_id))
}

fn apply_presented_selection(
    ui: &egui::Ui,
    host_id: NodeId,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
) {
    let ctrl_held = ui.input(|input| input.modifiers.ctrl);
    if ctrl_held {
        if selected_set.remove(&host_id) {
            if *selected == Some(host_id) {
                *selected = selected_set.iter().copied().min();
            }
        } else {
            selected_set.insert(host_id);
            *selected = Some(host_id);
        }
    } else {
        *selected = Some(host_id);
        selected_set.clear();
        selected_set.insert(host_id);
    }
}

fn begin_rename(
    scene: &Scene,
    host_id: NodeId,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
) {
    if let Some(node) = scene.nodes.get(&host_id) {
        *rename_buf = node.name.clone();
        *renaming = Some(host_id);
    }
}

fn object_label(scene: &Scene, object: PresentedObjectRef) -> String {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return format!("[Missing] #{:?}", object.host_id);
    };
    let badge = match &node.data {
        NodeData::Primitive { kind, .. } => kind.badge(),
        NodeData::Operation { op, .. } => op.badge(),
        NodeData::Sculpt { .. } => "[Vox]",
        NodeData::Light { light_type, .. } => light_type.badge(),
        NodeData::Transform { .. } => "[Xfm]",
        NodeData::Modifier { kind, .. } => kind.badge(),
    };
    format!("{badge} {}", node.name)
}

fn object_label_color(scene: &Scene, object: PresentedObjectRef, is_selected: bool) -> egui::Color32 {
    if scene.is_hidden(object.object_root_id) {
        COLOR_HIDDEN
    } else if is_selected {
        COLOR_SELECTED
    } else {
        COLOR_NORMAL
    }
}

fn object_context_menu(
    response: &egui::Response,
    scene: &Scene,
    object: PresentedObjectRef,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
) {
    response.context_menu(|ui| {
        let hidden = scene.is_hidden(object.object_root_id);
        if ui.button(if hidden { "Show" } else { "Hide" }).clicked() {
            actions.push(Action::ToggleVisibility(object.object_root_id));
            ui.close_menu();
        }
        if ui.button("Rename").clicked() {
            begin_rename(scene, object.host_id, renaming, rename_buf);
            ui.close_menu();
        }
        if ui.button("Duplicate").clicked() {
            actions.push(Action::DuplicatePresentedObject(object.host_id));
            ui.close_menu();
        }
        if !matches!(object.kind, PresentedObjectKind::Light) {
            ui.menu_button("Add", |ui| {
                presented_object_actions::draw_host_add_menu_contents(ui, scene, object, actions);
            });
        }
        if let Some(sculpt_id) = object.attached_sculpt_id {
            if ui.button("Convert to Voxel Object").clicked() {
                presented_object_actions::push_convert_to_voxel_action(
                    scene, object, sculpt_id, actions,
                );
                ui.close_menu();
            }
        }
        if object.can_remove_attached_sculpt() && ui.button("Remove Sculpt Layer").clicked() {
            actions.push(Action::RemoveAttachedSculpt {
                host: object.host_id,
            });
            ui.close_menu();
        }
        if ui.button("Delete").clicked() {
            actions.push(Action::DeletePresentedObject(object.object_root_id));
            ui.close_menu();
        }
    });
}

fn draw_object_add_button(
    ui: &mut egui::Ui,
    scene: &Scene,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
) {
    if matches!(object.kind, PresentedObjectKind::Light) {
        return;
    }
    chrome::with_action_button_style(ui, |ui| {
        presented_object_actions::draw_host_add_menu_button(ui, scene, object, actions, "+");
    });
}

fn draw_object_state_chips(ui: &mut egui::Ui, object: PresentedObjectRef) {
    if matches!(object.kind, PresentedObjectKind::Light) {
        chips::draw_chip(
            ui,
            "LIGHT",
            egui::Color32::from_rgb(64, 58, 26),
            egui::Color32::from_rgb(252, 230, 144),
        );
    }
    if object.attached_sculpt_id.is_some() {
        chips::draw_chip(
            ui,
            "SCULPT",
            egui::Color32::from_rgb(98, 67, 24),
            egui::Color32::from_rgb(255, 220, 140),
        );
    }
    if matches!(object.kind, PresentedObjectKind::Voxel) {
        chips::draw_chip(
            ui,
            "VOXEL",
            egui::Color32::from_rgb(28, 72, 92),
            egui::Color32::from_rgb(180, 232, 255),
        );
    }
}

fn draw_light_solo_button(
    ui: &mut egui::Ui,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
    soloed_light: Option<NodeId>,
) {
    if !matches!(object.kind, PresentedObjectKind::Light) {
        return;
    }

    let is_soloed = soloed_light == Some(object.host_id);
    if chrome::action_button(ui, "S", is_soloed).clicked() {
        actions.push(Action::SoloLight(Some(object.host_id)));
    }
}
