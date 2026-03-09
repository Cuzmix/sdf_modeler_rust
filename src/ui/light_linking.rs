use eframe::egui;
use std::collections::HashSet;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{NodeData, NodeId, Scene};

/// Draw the Light Linking matrix panel (Maya/Houdini-style).
/// Rows = geometry nodes (Primitives + Sculpts), Columns = active lights (up to 8).
/// Each cell is a checkbox controlling whether that light affects that object.
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
) {
    ui.add_space(4.0);

    // Collect geometry nodes (Primitives and Sculpts)
    let mut geometry_nodes: Vec<(NodeId, String)> = Vec::new();
    for (&id, node) in &scene.nodes {
        match &node.data {
            NodeData::Primitive { .. } | NodeData::Sculpt { .. } => {
                geometry_nodes.push((id, node.name.clone()));
            }
            _ => {}
        }
    }
    geometry_nodes.sort_by_key(|(id, _)| *id);

    // Collect lights in stable order, tracking which slot index they map to
    let parent_map = scene.build_parent_map();
    let mut light_entries: Vec<(NodeId, String)> = Vec::new();
    for (&id, node) in &scene.nodes {
        if matches!(node.data, NodeData::Light { .. }) && parent_map.contains_key(&id) {
            light_entries.push((id, node.name.clone()));
        }
    }
    light_entries.sort_by_key(|(id, _)| *id);

    // Only show active lights (the ones actually sent to GPU, max 8)
    let active_lights: Vec<(usize, NodeId, String)> = light_entries
        .iter()
        .enumerate()
        .filter(|(_, (id, _))| active_light_ids.contains(id))
        .map(|(slot, (id, name))| (slot, *id, name.clone()))
        .collect();

    if geometry_nodes.is_empty() || active_lights.is_empty() {
        if geometry_nodes.is_empty() {
            ui.weak("No geometry nodes in scene");
        }
        if active_lights.is_empty() {
            ui.weak("No active lights in scene");
        }
        return;
    }

    // Bulk actions
    ui.horizontal(|ui| {
        if ui.button("Link All").clicked() {
            for &(geo_id, _) in &geometry_nodes {
                actions.push(Action::SetLightMask {
                    node_id: geo_id,
                    mask: 0xFF,
                });
            }
        }
        if ui.button("Unlink All").clicked() {
            for &(geo_id, _) in &geometry_nodes {
                actions.push(Action::SetLightMask {
                    node_id: geo_id,
                    mask: 0x00,
                });
            }
        }
        if ui.button("Reset All").clicked() {
            for &(geo_id, _) in &geometry_nodes {
                actions.push(Action::SetLightMask {
                    node_id: geo_id,
                    mask: 0xFF,
                });
            }
        }
    });

    ui.separator();

    // Matrix grid
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("light_linking_matrix")
                .striped(true)
                .min_col_width(24.0)
                .show(ui, |ui| {
                    // Header row: empty corner cell + light names
                    ui.label("");
                    for (_, _, light_name) in &active_lights {
                        // Vertical-ish header: truncate long names
                        let short_name = if light_name.len() > 8 {
                            format!("{}...", &light_name[..6])
                        } else {
                            light_name.clone()
                        };
                        ui.vertical(|ui| {
                            ui.set_min_width(24.0);
                            ui.weak(&short_name);
                        });
                    }
                    // Per-row "Link All" column header
                    ui.weak("All");
                    ui.end_row();

                    // Data rows: one per geometry node
                    for &(geo_id, ref geo_name) in &geometry_nodes {
                        let current_mask = scene.get_light_mask(geo_id);

                        // Row header: type-colored dot + node name
                        let type_color = if scene
                            .nodes
                            .get(&geo_id)
                            .map(|n| matches!(n.data, NodeData::Sculpt { .. }))
                            .unwrap_or(false)
                        {
                            egui::Color32::from_rgb(230, 140, 40) // orange for sculpt
                        } else {
                            egui::Color32::from_rgb(80, 140, 230) // blue for primitive
                        };

                        ui.horizontal(|ui| {
                            let (dot_rect, _) =
                                ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                            ui.painter()
                                .circle_filled(dot_rect.center(), 3.5, type_color);

                            let short_name = if geo_name.len() > 14 {
                                format!("{}...", &geo_name[..12])
                            } else {
                                geo_name.clone()
                            };
                            ui.label(&short_name);
                        });

                        // Checkbox cells for each active light
                        for &(light_slot, _, _) in &active_lights {
                            let slot_u8 = light_slot as u8;
                            let mut linked = (current_mask & (1 << slot_u8)) != 0;
                            if ui.checkbox(&mut linked, "").changed() {
                                actions.push(Action::ToggleLightMaskBit {
                                    node_id: geo_id,
                                    light_slot: slot_u8,
                                    enabled: linked,
                                });
                            }
                        }

                        // Per-row link/unlink toggle
                        let all_linked = active_lights
                            .iter()
                            .all(|&(slot, _, _)| (current_mask & (1 << slot as u8)) != 0);
                        let mut row_all = all_linked;
                        if ui.checkbox(&mut row_all, "").changed() {
                            let new_mask = if row_all { 0xFF } else { 0x00 };
                            actions.push(Action::SetLightMask {
                                node_id: geo_id,
                                mask: new_mask,
                            });
                        }

                        ui.end_row();
                    }

                    // Footer row: per-column link/unlink toggles
                    ui.weak("All");
                    for &(light_slot, _, _) in &active_lights {
                        let slot_u8 = light_slot as u8;
                        let all_geo_linked = geometry_nodes.iter().all(|&(geo_id, _)| {
                            (scene.get_light_mask(geo_id) & (1 << slot_u8)) != 0
                        });
                        let mut col_all = all_geo_linked;
                        if ui.checkbox(&mut col_all, "").changed() {
                            for &(geo_id, _) in &geometry_nodes {
                                actions.push(Action::ToggleLightMaskBit {
                                    node_id: geo_id,
                                    light_slot: slot_u8,
                                    enabled: col_all,
                                });
                            }
                        }
                    }
                    ui.end_row();
                });
        });
}
