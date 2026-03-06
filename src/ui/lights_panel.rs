use eframe::egui;
use std::collections::HashSet;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{LightType, NodeData, NodeId, Scene};

/// Draw the Lights management panel. Lists all Light nodes with inline controls
/// for quick editing, plus buttons to create new lights.
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
) {
    ui.add_space(4.0);

    // Quick-add buttons
    ui.horizontal(|ui| {
        if ui.button("+ Point").clicked() {
            actions.push(Action::CreateLight(LightType::Point));
        }
        if ui.button("+ Spot").clicked() {
            actions.push(Action::CreateLight(LightType::Spot));
        }
        if ui.button("+ Directional").clicked() {
            actions.push(Action::CreateLight(LightType::Directional));
        }
        if ui.button("+ Ambient").clicked() {
            actions.push(Action::CreateLight(LightType::Ambient));
        }
    });

    ui.separator();

    // Collect lights and their parent transforms
    let parent_map = scene.build_parent_map();
    let mut light_entries: Vec<(NodeId, NodeId)> = Vec::new(); // (light_id, transform_id)

    for (&id, node) in &scene.nodes {
        if matches!(node.data, NodeData::Light { .. }) {
            if let Some(&transform_id) = parent_map.get(&id) {
                light_entries.push((id, transform_id));
            }
        }
    }

    // Sort by node ID for stable ordering
    light_entries.sort_by_key(|(light_id, _)| *light_id);

    if light_entries.is_empty() {
        ui.weak("No lights in scene");
        return;
    }

    let total_lights = light_entries.len();

    // Light list
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (light_id, transform_id) in &light_entries {
                let is_active = active_light_ids.contains(light_id);
                let is_selected = *selected == Some(*transform_id) || *selected == Some(*light_id);
                let is_hidden = scene.is_hidden(*light_id) || scene.is_hidden(*transform_id);

                // Read current light properties
                let Some(light_node) = scene.nodes.get(light_id) else {
                    continue;
                };
                let (light_type, color, intensity) = match &light_node.data {
                    NodeData::Light {
                        light_type,
                        color,
                        intensity,
                        ..
                    } => (light_type.clone(), *color, *intensity),
                    _ => continue,
                };
                let light_name = light_node.name.clone();

                // Row frame with selection highlight
                let row_response = ui.horizontal(|ui| {
                    // Selection indicator
                    if is_selected {
                        ui.painter().rect_filled(
                            ui.available_rect_before_wrap(),
                            2.0,
                            egui::Color32::from_rgba_unmultiplied(80, 120, 200, 40),
                        );
                    }

                    // Visibility toggle (eye icon)
                    let eye_label = if is_hidden { "\u{1F441}\u{FE0F}" } else { "\u{1F441}" };
                    let eye_text = if is_hidden {
                        egui::RichText::new(eye_label).color(egui::Color32::from_gray(80))
                    } else {
                        egui::RichText::new(eye_label)
                    };
                    if ui.small_button(eye_text).clicked() {
                        scene.toggle_visibility(*transform_id);
                    }

                    // Active/inactive indicator dot
                    let dot_color = if !is_active {
                        egui::Color32::from_gray(100)
                    } else {
                        egui::Color32::from_rgb(50, 200, 50)
                    };
                    let (dot_rect, _) = ui.allocate_exact_size(
                        egui::vec2(8.0, 8.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().circle_filled(dot_rect.center(), 3.5, dot_color);

                    // Color swatch
                    let mut rgb = [color.x, color.y, color.z];
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        if let Some(node) = scene.nodes.get_mut(light_id) {
                            if let NodeData::Light { color: c, .. } = &mut node.data {
                                *c = glam::Vec3::new(rgb[0], rgb[1], rgb[2]);
                            }
                        }
                    }

                    // Type badge
                    ui.weak(light_type.badge());

                    // Light name (clickable for selection)
                    let name_response = ui.selectable_label(is_selected, &light_name);
                    if name_response.clicked() {
                        *selected = Some(*transform_id);
                    }

                    // Intensity slider (compact)
                    let mut inten = intensity;
                    let slider_response = ui.add(
                        egui::DragValue::new(&mut inten)
                            .speed(0.05)
                            .range(0.0..=10.0)
                            .prefix("I: "),
                    );
                    if slider_response.changed() {
                        if let Some(node) = scene.nodes.get_mut(light_id) {
                            if let NodeData::Light { intensity: i, .. } = &mut node.data {
                                *i = inten;
                            }
                        }
                    }
                });

                // Click on the row area to select
                if row_response.response.clicked() {
                    *selected = Some(*transform_id);
                }
            }
        });

    ui.separator();

    // Status bar
    let active_count = active_light_ids.len();
    let status = format!("{}/{} lights active (max 8)", active_count, total_lights);
    if total_lights > 8 {
        ui.colored_label(egui::Color32::YELLOW, status);
    } else {
        ui.weak(status);
    }
}
