use std::collections::HashSet;

use eframe::egui;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::graph::presented_object::{
    presented_children, presented_top_level_objects, PresentedObjectKind, PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel;

const COLOR_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
const COLOR_NORMAL: egui::Color32 = egui::Color32::from_rgb(200, 200, 210);
const COLOR_HIDDEN: egui::Color32 = egui::Color32::from_gray(100);

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
    ui.heading("Items");

    ui.horizontal(|ui| {
        ui.label("\u{1F50D}");
        ui.add(
            egui::TextEdit::singleline(search_filter)
                .hint_text("Filter objects...")
                .desired_width(ui.available_width() - 24.0),
        );
        if !search_filter.is_empty() && ui.small_button("\u{2715}").clicked() {
            search_filter.clear();
        }
    });
    ui.separator();

    let roots = presented_top_level_objects(scene);
    if roots.is_empty() {
        ui.label("Empty scene");
        return;
    }

    if !search_filter.is_empty() {
        let filter_lower = search_filter.to_lowercase();
        let mut flattened = Vec::new();
        collect_flattened_objects(scene, &roots, &mut flattened);
        let matching: Vec<_> = flattened
            .into_iter()
            .filter(|object| {
                scene.nodes
                    .get(&object.host_id)
                    .is_some_and(|node| node.name.to_lowercase().contains(&filter_lower))
            })
            .collect();

        if matching.is_empty() {
            ui.weak("No matches");
            return;
        }

        for object in matching {
            draw_flat_object_row(
                ui,
                scene,
                object,
                selected,
                selected_set,
                renaming,
                rename_buf,
                actions,
                soloed_light,
            );
        }
        return;
    }

    let mut visited = HashSet::new();
    for object in roots {
        draw_object_recursive(
            ui,
            scene,
            object,
            selected,
            selected_set,
            renaming,
            rename_buf,
            actions,
            soloed_light,
            &mut visited,
        );
    }
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
    if children.is_empty() {
        draw_flat_object_row(
            ui,
            scene,
            object,
            selected,
            selected_set,
            renaming,
            rename_buf,
            actions,
            soloed_light,
        );
        return;
    }

    let is_selected = selected_set.contains(&object.host_id) || *selected == Some(object.host_id);
    let color = object_label_color(scene, object, is_selected);
    let label = egui::RichText::new(object_label(scene, object)).color(color);

    ui.horizontal(|ui| {
        let mut visible = !scene.is_hidden(object.object_root_id);
        if ui.checkbox(&mut visible, "").changed() {
            actions.push(Action::ToggleVisibility(object.object_root_id));
        }

        let header = egui::CollapsingHeader::new(label)
            .default_open(true)
            .id_salt(("presented_object", object.host_id))
            .show(ui, |ui| {
                for child in children {
                    draw_object_recursive(
                        ui,
                        scene,
                        child,
                        selected,
                        selected_set,
                        renaming,
                        rename_buf,
                        actions,
                        soloed_light,
                        visited,
                    );
                }
            });

        if header.header_response.clicked() {
            apply_presented_selection(ui, object.host_id, selected, selected_set);
        }
        if header.header_response.double_clicked() {
            begin_rename(scene, object.host_id, renaming, rename_buf);
        }
        draw_light_solo_button(ui, object, actions, soloed_light);
        object_context_menu(
            &header.header_response,
            scene,
            object,
            renaming,
            rename_buf,
            actions,
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn draw_flat_object_row(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    selected: &mut Option<NodeId>,
    selected_set: &mut HashSet<NodeId>,
    renaming: &mut Option<NodeId>,
    rename_buf: &mut String,
    actions: &mut ActionSink,
    soloed_light: Option<NodeId>,
) {
    if *renaming == Some(object.host_id) {
        ui.horizontal(|ui| {
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
        return;
    }

    let is_selected = selected_set.contains(&object.host_id) || *selected == Some(object.host_id);
    let color = object_label_color(scene, object, is_selected);
    let label = egui::RichText::new(object_label(scene, object)).color(color);

    ui.horizontal(|ui| {
        let mut visible = !scene.is_hidden(object.object_root_id);
        if ui.checkbox(&mut visible, "").changed() {
            actions.push(Action::ToggleVisibility(object.object_root_id));
        }

        let response = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
        if response.clicked() {
            apply_presented_selection(ui, object.host_id, selected, selected_set);
        }
        if response.double_clicked() {
            begin_rename(scene, object.host_id, renaming, rename_buf);
        }
        draw_light_solo_button(ui, object, actions, soloed_light);
        object_context_menu(&response, scene, object, renaming, rename_buf, actions);
    });
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
    let sculpt_badge = if object.attached_sculpt_id.is_some() {
        "  [Sculpt]"
    } else {
        ""
    };
    format!("{badge} {}{sculpt_badge}", node.name)
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

fn sculpt_color_for_object(scene: &Scene, object: PresentedObjectRef) -> Vec3 {
    scene
        .nodes
        .get(&object.host_id)
        .and_then(|node| node.data.material())
        .map(|material| material.base_color)
        .unwrap_or(Vec3::splat(0.6))
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
        if object.supports_add_sculpt() && ui.button("Add Sculpt").clicked() {
            actions.push(Action::RequestBake(crate::app::BakeRequest {
                subtree_root: object.object_root_id,
                resolution: voxel::DEFAULT_RESOLUTION,
                color: sculpt_color_for_object(scene, object),
                existing_sculpt: None,
                flatten: false,
            }));
            ui.close_menu();
        }
        if object.can_remove_attached_sculpt() && ui.button("Remove Sculpt").clicked() {
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
    let solo_text = egui::RichText::new("S").small();
    let solo_text = if is_soloed {
        solo_text.color(egui::Color32::from_rgb(255, 220, 50))
    } else {
        solo_text.color(egui::Color32::from_gray(120))
    };
    if ui.small_button(solo_text).clicked() {
        actions.push(Action::SoloLight(Some(object.host_id)));
    }
}
