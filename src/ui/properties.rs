use std::collections::HashSet;

use eframe::egui;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::app::BakeRequest;
use crate::graph::scene::{
    CsgOp, MaterialParams, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive,
};
use crate::graph::voxel;
use crate::material_preset::{self, MaterialLibrary};
use crate::sculpt::SculptState;

const SCALE_MIN: f32 = 0.01;
const SCALE_MAX: f32 = 100.0;

/// Common color presets as (name, [r, g, b]).
const COLOR_PRESETS: &[(&str, [f32; 3])] = &[
    ("R", [0.85, 0.15, 0.15]),
    ("G", [0.15, 0.75, 0.15]),
    ("B", [0.2, 0.3, 0.85]),
    ("Y", [0.9, 0.8, 0.1]),
    ("O", [0.9, 0.5, 0.1]),
    ("W", [0.9, 0.9, 0.9]),
    ("Gr", [0.5, 0.5, 0.5]),
    ("Tn", [0.76, 0.6, 0.42]),
];

fn color_presets_row(ui: &mut egui::Ui, color: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        for &(label, preset) in COLOR_PRESETS {
            let c32 = egui::Color32::from_rgb(
                (preset[0] * 255.0) as u8,
                (preset[1] * 255.0) as u8,
                (preset[2] * 255.0) as u8,
            );
            let is_match = (color[0] - preset[0]).abs() < 0.02
                && (color[1] - preset[1]).abs() < 0.02
                && (color[2] - preset[2]).abs() < 0.02;
            let btn = egui::Button::new(egui::RichText::new(label).small().color(
                if preset[0] + preset[1] + preset[2] > 1.5 {
                    egui::Color32::BLACK
                } else {
                    egui::Color32::WHITE
                },
            ))
            .fill(c32)
            .min_size(egui::vec2(20.0, 16.0))
            .stroke(if is_match {
                egui::Stroke::new(2.0, egui::Color32::WHITE)
            } else {
                egui::Stroke::NONE
            });
            if ui.add(btn).clicked() {
                *color = preset;
            }
        }
    });
}

fn vec3_editor(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut glam::Vec3,
    speed: f32,
    range: Option<std::ops::RangeInclusive<f32>>,
    suffix: &str,
) {
    ui.label(label);
    ui.horizontal(|ui| {
        for (axis_label, component) in [
            ("X:", &mut value.x),
            ("Y:", &mut value.y),
            ("Z:", &mut value.z),
        ] {
            ui.label(axis_label);
            let mut drag = egui::DragValue::new(component).speed(speed);
            if !suffix.is_empty() {
                drag = drag.suffix(suffix);
            }
            if let Some(ref r) = range {
                drag = drag.range(r.clone());
            }
            ui.add(drag);
        }
    });
}

/// Walk up from a leaf node, collecting Transform/Modifier ancestors
/// until hitting an Operation or top-level node. Returns innermost first.
fn collect_modifier_chain(scene: &Scene, leaf_id: NodeId) -> Vec<NodeId> {
    let parent_map = scene.build_parent_map();
    let mut chain = Vec::new();
    let mut current = leaf_id;
    while let Some(&parent_id) = parent_map.get(&current) {
        match scene.nodes.get(&parent_id).map(|n| &n.data) {
            Some(NodeData::Transform { .. } | NodeData::Modifier { .. }) => {
                chain.push(parent_id);
                current = parent_id;
            }
            _ => break,
        }
    }
    chain
}

/// Helper: extract position/rotation from a node (Primitive or Sculpt have position/rotation,
/// Transform has translation/rotation).
fn get_node_position(data: &NodeData) -> Option<glam::Vec3> {
    match data {
        NodeData::Primitive { position, .. } => Some(*position),
        NodeData::Sculpt { position, .. } => Some(*position),
        NodeData::Transform { translation, .. } => Some(*translation),
        _ => None,
    }
}

fn get_node_color(data: &NodeData) -> Option<glam::Vec3> {
    data.material().map(|material| material.base_color)
}

fn get_node_roughness(data: &NodeData) -> Option<f32> {
    data.material().map(|material| material.roughness)
}

fn get_node_metallic(data: &NodeData) -> Option<f32> {
    data.material().map(|material| material.metallic)
}

fn get_node_reflectance_f0(data: &NodeData) -> Option<f32> {
    data.material().map(|material| material.reflectance_f0)
}

/// Apply a position delta to a node (Primitive, Sculpt, or Transform).
fn apply_position_delta(data: &mut NodeData, delta: glam::Vec3) {
    match data {
        NodeData::Primitive { position, .. } => *position += delta,
        NodeData::Sculpt { position, .. } => *position += delta,
        NodeData::Transform { translation, .. } => *translation += delta,
        _ => {}
    }
}

/// Apply a rotation delta to a node.
fn apply_rotation_delta(data: &mut NodeData, delta: glam::Vec3) {
    match data {
        NodeData::Primitive { rotation, .. } => *rotation += delta,
        NodeData::Sculpt { rotation, .. } => *rotation += delta,
        NodeData::Transform { rotation, .. } => *rotation += delta,
        _ => {}
    }
}

/// Set color on a node (Primitive or Sculpt).
fn set_node_color(data: &mut NodeData, new_color: glam::Vec3) {
    if let Some(material) = data.material_mut() {
        material.base_color = new_color;
    }
}

/// Set roughness on a node (Primitive or Sculpt).
fn set_node_roughness(data: &mut NodeData, val: f32) {
    if let Some(material) = data.material_mut() {
        material.roughness = val;
    }
}

/// Set metallic on a node (Primitive or Sculpt).
fn set_node_metallic(data: &mut NodeData, val: f32) {
    if let Some(material) = data.material_mut() {
        material.metallic = val;
    }
}

/// Set reflectance F0 on a node (Primitive or Sculpt).
fn set_node_reflectance_f0(data: &mut NodeData, val: f32) {
    if let Some(material) = data.material_mut() {
        material.reflectance_f0 = val;
    }
}

fn apply_material_preset(material: &mut MaterialParams, preset: &material_preset::MaterialPreset) {
    let preserved_color = material.base_color;
    *material = preset.material.clone();
    if preset.preserve_existing_base_color {
        material.base_color = preserved_color;
    }
}

fn save_material_preset(material_library: &mut MaterialLibrary, material: &MaterialParams) {
    let preset_name = format!("Custom {}", material_library.user_presets.len() + 1);
    let preset =
        material_preset::MaterialPreset::from_node_material(&preset_name, material.clone());
    material_library.save_preset(preset);
    material_library.save();
}

fn normalize_direction_or_x(direction: Vec3) -> Vec3 {
    if direction.length_squared() <= 1e-6 {
        Vec3::X
    } else {
        direction.normalize()
    }
}

fn draw_material_editor(
    ui: &mut egui::Ui,
    id_prefix: &str,
    material: &mut MaterialParams,
    material_library: &mut MaterialLibrary,
) {
    let built_in = material_preset::built_in_presets();

    ui.horizontal(|ui| {
        ui.label("Preset:");
        egui::ComboBox::from_id_salt(format!("{id_prefix}_mat_preset"))
            .selected_text("Apply...")
            .width(120.0)
            .show_ui(ui, |ui| {
                for category in material_preset::CATEGORIES {
                    if *category == material_preset::CATEGORY_USER {
                        continue;
                    }
                    let in_category: Vec<_> = built_in
                        .iter()
                        .filter(|preset| preset.category == *category)
                        .collect();
                    if in_category.is_empty() {
                        continue;
                    }
                    ui.label(egui::RichText::new(*category).strong().small());
                    for preset in in_category {
                        if ui.selectable_label(false, &preset.name).clicked() {
                            apply_material_preset(material, preset);
                        }
                    }
                    ui.separator();
                }
                if !material_library.user_presets.is_empty() {
                    ui.label(egui::RichText::new("User").strong().small());
                    let mut remove_index = None;
                    for (idx, preset) in material_library.user_presets.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(false, &preset.name).clicked() {
                                apply_material_preset(material, preset);
                            }
                            if ui
                                .small_button("\u{2715}")
                                .on_hover_text("Delete preset")
                                .clicked()
                            {
                                remove_index = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove_index {
                        material_library.remove_preset(idx);
                        material_library.save();
                    }
                }
            });
        if ui
            .small_button("\u{1F4BE}")
            .on_hover_text("Save current material as preset")
            .clicked()
        {
            save_material_preset(material_library, material);
        }
    });

    egui::CollapsingHeader::new("Base")
        .default_open(true)
        .id_salt(format!("{id_prefix}_base"))
        .show(ui, |ui| {
            ui.label("Color");
            let mut color_arr = [
                material.base_color.x,
                material.base_color.y,
                material.base_color.z,
            ];
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut color_arr);
                color_presets_row(ui, &mut color_arr);
            });
            material.base_color = Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

            ui.horizontal(|ui| {
                ui.label("Metallic:");
                ui.add(egui::Slider::new(&mut material.metallic, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Roughness:");
                ui.add(egui::Slider::new(&mut material.roughness, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Reflectance F0:");
                ui.add(egui::Slider::new(&mut material.reflectance_f0, 0.0..=1.0));
            });
        });

    egui::CollapsingHeader::new("Emissive")
        .default_open(false)
        .id_salt(format!("{id_prefix}_emissive"))
        .show(ui, |ui| {
            let mut emissive_arr = [
                material.emissive_color.x,
                material.emissive_color.y,
                material.emissive_color.z,
            ];
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut emissive_arr);
                if ui
                    .small_button("= Base")
                    .on_hover_text("Set emissive to base color")
                    .clicked()
                {
                    emissive_arr = [
                        material.base_color.x,
                        material.base_color.y,
                        material.base_color.z,
                    ];
                }
            });
            material.emissive_color = Vec3::new(emissive_arr[0], emissive_arr[1], emissive_arr[2]);
            ui.horizontal(|ui| {
                ui.label("Intensity:");
                ui.add(egui::Slider::new(
                    &mut material.emissive_intensity,
                    0.0..=5.0,
                ));
            });
        });

    egui::CollapsingHeader::new("Clearcoat")
        .default_open(false)
        .id_salt(format!("{id_prefix}_clearcoat"))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Amount:");
                ui.add(egui::Slider::new(&mut material.clearcoat, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Roughness:");
                ui.add(egui::Slider::new(
                    &mut material.clearcoat_roughness,
                    0.0..=1.0,
                ));
            });
        });

    egui::CollapsingHeader::new("Sheen")
        .default_open(false)
        .id_salt(format!("{id_prefix}_sheen"))
        .show(ui, |ui| {
            let mut sheen_arr = [
                material.sheen_color.x,
                material.sheen_color.y,
                material.sheen_color.z,
            ];
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut sheen_arr);
                if ui
                    .small_button("= Base")
                    .on_hover_text("Set sheen tint to base color")
                    .clicked()
                {
                    sheen_arr = [
                        material.base_color.x,
                        material.base_color.y,
                        material.base_color.z,
                    ];
                }
            });
            material.sheen_color = Vec3::new(sheen_arr[0], sheen_arr[1], sheen_arr[2]);
            ui.horizontal(|ui| {
                ui.label("Roughness:");
                ui.add(egui::Slider::new(&mut material.sheen_roughness, 0.0..=1.0));
            });
        });

    egui::CollapsingHeader::new("Transmission")
        .default_open(false)
        .id_salt(format!("{id_prefix}_transmission"))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Amount:");
                ui.add(egui::Slider::new(&mut material.transmission, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Thickness:");
                ui.add(egui::Slider::new(&mut material.thickness, 0.0..=8.0));
            });
            ui.horizontal(|ui| {
                ui.label("IOR:");
                ui.add(egui::Slider::new(&mut material.ior, 1.0..=2.5));
            });
            ui.small("Base color tints transmitted light.");
        });

    egui::CollapsingHeader::new("Anisotropy")
        .default_open(false)
        .id_salt(format!("{id_prefix}_anisotropy"))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Amount:");
                ui.add(egui::Slider::new(
                    &mut material.anisotropy_strength,
                    -0.95..=0.95,
                ));
            });

            let mut direction = normalize_direction_or_x(material.anisotropy_direction_local);
            ui.horizontal(|ui| {
                ui.label("Direction:");
                ui.add(
                    egui::DragValue::new(&mut direction.x)
                        .speed(0.05)
                        .range(-1.0..=1.0)
                        .max_decimals(2),
                );
                ui.add(
                    egui::DragValue::new(&mut direction.y)
                        .speed(0.05)
                        .range(-1.0..=1.0)
                        .max_decimals(2),
                );
                ui.add(
                    egui::DragValue::new(&mut direction.z)
                        .speed(0.05)
                        .range(-1.0..=1.0)
                        .max_decimals(2),
                );
            });
            ui.horizontal(|ui| {
                if ui.small_button("X").clicked() {
                    direction = Vec3::X;
                }
                if ui.small_button("Y").clicked() {
                    direction = Vec3::Y;
                }
                if ui.small_button("Z").clicked() {
                    direction = Vec3::Z;
                }
            });
            material.anisotropy_direction_local = normalize_direction_or_x(direction);
        });
}

/// Draw properties panel for multiple selected nodes (batch editing).
fn draw_multi_properties(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected_set: &HashSet<NodeId>,
    actions: &mut ActionSink,
) {
    let count = selected_set.len();
    ui.heading(format!("{} nodes selected", count));
    ui.separator();

    // Determine which property groups are available across all selected nodes
    let ids: Vec<NodeId> = selected_set.iter().copied().collect();
    let all_have_position = ids.iter().all(|id| {
        scene
            .nodes
            .get(id)
            .is_some_and(|n| get_node_position(&n.data).is_some())
    });
    let all_have_color = ids.iter().all(|id| {
        scene
            .nodes
            .get(id)
            .is_some_and(|n| get_node_color(&n.data).is_some())
    });

    // --- Transform (delta-based) ---
    if all_have_position {
        egui::CollapsingHeader::new("Transform")
            .default_open(true)
            .show(ui, |ui| {
                // Position — delta mode: start at zero, apply delta to all
                let mut pos_delta = glam::Vec3::ZERO;
                ui.label("Position (delta)");
                ui.horizontal(|ui| {
                    for (axis_label, component) in [
                        ("X:", &mut pos_delta.x),
                        ("Y:", &mut pos_delta.y),
                        ("Z:", &mut pos_delta.z),
                    ] {
                        ui.label(axis_label);
                        ui.add(egui::DragValue::new(component).speed(0.05));
                    }
                });
                if pos_delta != glam::Vec3::ZERO {
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            apply_position_delta(&mut node.data, pos_delta);
                        }
                    }
                }

                // Rotation — delta mode (display in degrees)
                let mut rot_delta_deg = glam::Vec3::ZERO;
                ui.label("Rotation (delta)");
                ui.horizontal(|ui| {
                    for (axis_label, component) in [
                        ("X:", &mut rot_delta_deg.x),
                        ("Y:", &mut rot_delta_deg.y),
                        ("Z:", &mut rot_delta_deg.z),
                    ] {
                        ui.label(axis_label);
                        ui.add(
                            egui::DragValue::new(component)
                                .speed(1.0)
                                .suffix("\u{00B0}"),
                        );
                    }
                });
                if rot_delta_deg != glam::Vec3::ZERO {
                    let rot_delta_rad = glam::Vec3::new(
                        rot_delta_deg.x.to_radians(),
                        rot_delta_deg.y.to_radians(),
                        rot_delta_deg.z.to_radians(),
                    );
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            apply_rotation_delta(&mut node.data, rot_delta_rad);
                        }
                    }
                }
            });
    }

    // --- Material (absolute-value) ---
    if all_have_color {
        egui::CollapsingHeader::new("Material")
            .default_open(true)
            .show(ui, |ui| {
                // Color — use first selected node's color as starting value
                let first_color = ids
                    .iter()
                    .find_map(|id| scene.nodes.get(id).and_then(|n| get_node_color(&n.data)))
                    .unwrap_or(glam::Vec3::new(0.5, 0.5, 0.5));
                let mut color_arr = [first_color.x, first_color.y, first_color.z];

                ui.label("Color");
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgb(&mut color_arr);
                    color_presets_row(ui, &mut color_arr);
                });
                let new_color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);
                if new_color != first_color {
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            set_node_color(&mut node.data, new_color);
                        }
                    }
                }

                // Roughness — use first node's value
                let first_roughness = ids
                    .iter()
                    .find_map(|id| {
                        scene
                            .nodes
                            .get(id)
                            .and_then(|n| get_node_roughness(&n.data))
                    })
                    .unwrap_or(0.5);
                let mut roughness = first_roughness;
                ui.horizontal(|ui| {
                    ui.label("Roughness:");
                    ui.add(egui::Slider::new(&mut roughness, 0.0..=1.0));
                });
                if (roughness - first_roughness).abs() > f32::EPSILON {
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            set_node_roughness(&mut node.data, roughness);
                        }
                    }
                }

                // Metallic — use first node's value
                let first_metallic = ids
                    .iter()
                    .find_map(|id| scene.nodes.get(id).and_then(|n| get_node_metallic(&n.data)))
                    .unwrap_or(0.0);
                let mut metallic = first_metallic;
                ui.horizontal(|ui| {
                    ui.label("Metallic:");
                    ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0));
                });
                if (metallic - first_metallic).abs() > f32::EPSILON {
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            set_node_metallic(&mut node.data, metallic);
                        }
                    }
                }

                // Reflectance F0 — use first node's value
                let first_reflectance_f0 = ids
                    .iter()
                    .find_map(|id| {
                        scene
                            .nodes
                            .get(id)
                            .and_then(|n| get_node_reflectance_f0(&n.data))
                    })
                    .unwrap_or(0.04);
                let mut reflectance_f0 = first_reflectance_f0;
                ui.horizontal(|ui| {
                    ui.label("Reflectance F0:");
                    ui.add(egui::Slider::new(&mut reflectance_f0, 0.0..=1.0));
                });
                if (reflectance_f0 - first_reflectance_f0).abs() > f32::EPSILON {
                    for &id in &ids {
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            set_node_reflectance_f0(&mut node.data, reflectance_f0);
                        }
                    }
                }
            });
    }

    // --- Batch actions ---
    ui.separator();
    if ui.button("Delete Selected Nodes").clicked() {
        for &id in &ids {
            actions.push(Action::DeleteNode(id));
        }
    }
}

/// Format large voxel counts with K/M suffixes for readability.
fn format_voxel_count(voxels: u64) -> String {
    if voxels >= 1_000_000 {
        format!("{:.1}M", voxels as f64 / 1_000_000.0)
    } else if voxels >= 1_000 {
        format!("{:.0}K", voxels as f64 / 1_000.0)
    } else {
        format!("{}", voxels)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
    sculpt_state: &mut SculptState,
    bake_progress: Option<(u32, u32)>,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
    max_sculpt_resolution: u32,
    soloed_light: Option<NodeId>,
    material_library: &mut MaterialLibrary,
) {
    // Multi-select: show batch properties when more than 1 node is selected
    if selected_set.len() > 1 {
        draw_multi_properties(ui, scene, selected_set, actions);
        return;
    }

    let Some(id) = selected else {
        ui.vertical_centered(|ui| {
            ui.add_space(40.0);
            ui.label("No selection");
            ui.add_space(8.0);
            ui.weak("Click a node in the viewport or scene tree");
        });
        return;
    };

    let Some(node) = scene.nodes.get(&id) else {
        ui.centered_and_justified(|ui| {
            ui.label("Selected node not found");
        });
        return;
    };

    let is_locked = node.locked;

    // Clone data to avoid borrow conflicts with egui widgets
    let mut name = node.name.clone();
    let node_data = node.data.clone();

    ui.heading(format!("Node #{}", id));
    if is_locked {
        ui.colored_label(egui::Color32::from_rgb(255, 180, 80), "\u{1F512} Locked");
    }
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Name:");
        ui.add_enabled(!is_locked, egui::TextEdit::singleline(&mut name));
    });

    ui.separator();

    match node_data {
        NodeData::Primitive {
            mut kind,
            mut position,
            mut rotation,
            mut scale,
            mut material,
            ..
        } => {
            let mut new_kind = kind.clone();
            egui::ComboBox::from_id_salt("prop_prim_type")
                .selected_text(new_kind.base_name())
                .show_ui(ui, |ui| {
                    for v in SdfPrimitive::ALL {
                        ui.selectable_value(&mut new_kind, v.clone(), v.base_name());
                    }
                });
            if new_kind != kind {
                scale = new_kind.default_scale();
                kind = new_kind;
            }
            ui.separator();

            egui::CollapsingHeader::new("Transform")
                .default_open(true)
                .show(ui, |ui| {
                    vec3_editor(ui, "Position", &mut position, 0.05, None, "");

                    // Rotation: display in degrees, store in radians
                    let mut rot_deg = glam::Vec3::new(
                        rotation.x.to_degrees(),
                        rotation.y.to_degrees(),
                        rotation.z.to_degrees(),
                    );
                    vec3_editor(ui, "Rotation", &mut rot_deg, 1.0, None, "\u{00B0}");
                    rotation = glam::Vec3::new(
                        rot_deg.x.to_radians(),
                        rot_deg.y.to_radians(),
                        rot_deg.z.to_radians(),
                    );

                    let params = kind.scale_params();
                    if !params.is_empty() {
                        ui.label("Size");
                        ui.horizontal(|ui| {
                            for &(label, axis) in params {
                                ui.label(format!("{}:", label));
                                let val = match axis {
                                    0 => &mut scale.x,
                                    1 => &mut scale.y,
                                    _ => &mut scale.z,
                                };
                                ui.add(
                                    egui::DragValue::new(val)
                                        .speed(0.05)
                                        .range(SCALE_MIN..=SCALE_MAX),
                                );
                            }
                        });
                    }
                });

            egui::CollapsingHeader::new("Material")
                .default_open(true)
                .show(ui, |ui| {
                    draw_material_editor(ui, "prim", &mut material, material_library);
                });

            // Add Sculpt Modifier button
            ui.separator();
            if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)),
                );
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Add Sculpt Modifier").clicked() {
                        actions.push(Action::RequestBake(BakeRequest {
                            subtree_root: id,
                            resolution: voxel::DEFAULT_RESOLUTION,
                            color: material.base_color,
                            existing_sculpt: None,
                            flatten: false,
                        }));
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            node.name = name.clone();
                        }
                        return;
                    }
                    if ui
                        .button("Delete Node")
                        .on_hover_text("Remove this node from the scene")
                        .clicked()
                    {
                        actions.push(Action::DeleteNode(id));
                    }
                });
            }

            // Light Linking
            draw_light_linking_section(ui, scene, id, actions, active_light_ids);

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Primitive {
                    kind: ref mut k,
                    position: ref mut p,
                    rotation: ref mut r,
                    scale: ref mut s,
                    material: ref mut mat,
                    ..
                } = node.data
                {
                    *k = kind;
                    *p = position;
                    *r = rotation;
                    *s = scale;
                    *mat = material;
                }
            }
        }
        NodeData::Operation {
            mut op,
            mut smooth_k,
            mut steps,
            mut color_blend,
            left,
            right,
        } => {
            let mut new_op = op.clone();
            egui::ComboBox::from_id_salt("prop_op_type")
                .selected_text(new_op.base_name())
                .show_ui(ui, |ui| {
                    for v in CsgOp::ALL {
                        ui.selectable_value(&mut new_op, v.clone(), v.base_name());
                    }
                });
            if new_op != op {
                smooth_k = new_op.default_smooth_k();
                steps = new_op.default_steps();
                color_blend = -1.0;
                op = new_op;
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Smooth K:");
                ui.add(egui::Slider::new(&mut smooth_k, 0.0..=2.0));
            });

            if op.has_steps_param() {
                ui.horizontal(|ui| {
                    ui.label("Count:");
                    ui.add(egui::Slider::new(&mut steps, 2.0..=16.0).step_by(1.0));
                });
            }

            if op.has_color_blend_param() {
                let mut independent = color_blend >= 0.0;
                ui.checkbox(&mut independent, "Independent Color Blend");
                if independent {
                    if color_blend < 0.0 {
                        color_blend = smooth_k;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Color K:");
                        ui.add(egui::Slider::new(&mut color_blend, 0.0..=2.0));
                    });
                } else {
                    color_blend = -1.0;
                }
            }

            ui.separator();
            match left {
                Some(lid) => {
                    let left_name = scene
                        .nodes
                        .get(&lid)
                        .map(|n| n.name.as_str())
                        .unwrap_or("???");
                    ui.label(format!("Left: {} (#{})", left_name, lid));
                }
                None => {
                    ui.label("Left: (empty)");
                }
            }
            match right {
                Some(rid) => {
                    let right_name = scene
                        .nodes
                        .get(&rid)
                        .map(|n| n.name.as_str())
                        .unwrap_or("???");
                    ui.label(format!("Right: {} (#{})", right_name, rid));
                }
                None => {
                    ui.label("Right: (empty)");
                }
            }
            if left.is_some() && right.is_some() && ui.button("Swap Inputs").clicked() {
                actions.push(Action::SwapChildren(id));
            }

            // Add Sculpt Modifier / Flatten buttons
            ui.separator();
            if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)),
                );
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Add Sculpt Modifier").clicked() {
                        let sculpt_color = glam::Vec3::new(0.6, 0.6, 0.6);
                        actions.push(Action::RequestBake(BakeRequest {
                            subtree_root: id,
                            resolution: voxel::DEFAULT_RESOLUTION,
                            color: sculpt_color,
                            existing_sculpt: None,
                            flatten: false,
                        }));
                        return;
                    }
                    if ui.button("Flatten to Sculpt").clicked() {
                        let resolution = voxel::max_subtree_resolution(scene, id);
                        let sculpt_color = glam::Vec3::new(0.6, 0.6, 0.6);
                        actions.push(Action::RequestBake(BakeRequest {
                            subtree_root: id,
                            resolution,
                            color: sculpt_color,
                            existing_sculpt: None,
                            flatten: true,
                        }));
                    }
                });
                if ui
                    .button("Delete Node")
                    .on_hover_text("Remove this node from the scene")
                    .clicked()
                {
                    actions.push(Action::DeleteNode(id));
                    return;
                }
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Operation {
                    op: ref mut o,
                    smooth_k: ref mut k,
                    steps: ref mut s,
                    color_blend: ref mut cb,
                    ..
                } = node.data
                {
                    *o = op;
                    *k = smooth_k;
                    *s = steps;
                    *cb = color_blend;
                }
            }
        }
        NodeData::Sculpt {
            input,
            mut position,
            mut rotation,
            mut material,
            mut layer_intensity,
            ref voxel_grid,
            mut desired_resolution,
        } => {
            ui.label("Type: Sculpt Modifier");
            ui.separator();

            match input {
                Some(iid) => {
                    let input_name = scene
                        .nodes
                        .get(&iid)
                        .map(|n| n.name.as_str())
                        .unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => {
                    ui.label("Input: (empty)");
                }
            }
            ui.separator();

            // Layer intensity (sculpt layer control)
            ui.horizontal(|ui| {
                ui.label("Layer Intensity:");
                ui.add(egui::Slider::new(&mut layer_intensity, 0.0..=1.0).fixed_decimals(2));
            });
            ui.separator();

            egui::CollapsingHeader::new("Transform")
                .default_open(true)
                .id_salt("sculpt_transform")
                .show(ui, |ui| {
                    vec3_editor(ui, "Position", &mut position, 0.05, None, "");

                    let mut rot_deg = glam::Vec3::new(
                        rotation.x.to_degrees(),
                        rotation.y.to_degrees(),
                        rotation.z.to_degrees(),
                    );
                    vec3_editor(ui, "Rotation", &mut rot_deg, 1.0, None, "\u{00B0}");
                    rotation = glam::Vec3::new(
                        rot_deg.x.to_radians(),
                        rot_deg.y.to_radians(),
                        rot_deg.z.to_radians(),
                    );
                });

            egui::CollapsingHeader::new("Material")
                .default_open(true)
                .id_salt("sculpt_material")
                .show(ui, |ui| {
                    draw_material_editor(ui, "sculpt", &mut material, material_library);
                });

            egui::CollapsingHeader::new("Sculpting")
                .default_open(true)
                .show(ui, |ui| {
                    let detail_size = voxel_grid.voxel_pitch();
                    ui.horizontal(|ui| {
                        ui.label("Detail Size:");
                        ui.monospace(format!("{detail_size:.4}"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.monospace(format!("{}^3", desired_resolution));
                    });
                    let detail_state = sculpt_state.detail_state();
                    if sculpt_state.active_node() == Some(id) {
                        if let Some(previous_detail) = detail_state.last_pre_expand_detail_size {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!(
                                    "Volume expansion made detail coarser. Previous detail size was {:.4}.",
                                    previous_detail
                                ),
                            );
                        }
                        if detail_state.detail_limited_after_growth {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 170, 80),
                                "Remesh is limited by the current sculpt resolution cap.",
                            );
                        }
                    }
                    ui.weak("Expand Volume adds room but makes detail coarser until you remesh.");
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Increase Detail").clicked() {
                            actions.push(Action::IncreaseSculptDetail(id));
                        }
                        if ui.button("Decrease Detail").clicked() {
                            actions.push(Action::DecreaseSculptDetail(id));
                        }
                        if ui.button("Remesh at Current Detail").clicked() {
                            actions.push(Action::RemeshSculptAtCurrentDetail(id));
                        }
                        if ui.button("Expand Volume").clicked() {
                            actions.push(Action::ExpandSculptVolume(id));
                        }
                        if ui.button("Fit Volume to Sculpt").clicked() {
                            actions.push(Action::FitSculptVolume(id));
                        }
                    });
                    ui.add_space(6.0);
                    ui.separator();

                    // Resolution presets
                    ui.horizontal(|ui| {
                        ui.label("Manual Resolution:");
                        for &(label, res) in &[
                            ("Low", 32u32),
                            ("Med", 64),
                            ("High", 96),
                            ("Ultra", 128),
                            ("Max", 256),
                        ] {
                            if ui
                                .selectable_label(desired_resolution == res, label)
                                .clicked()
                            {
                                desired_resolution = res;
                            }
                        }
                    });
                    // Custom input (capped by max_sculpt_resolution setting)
                    let max_res = max_sculpt_resolution.max(16);
                    ui.horizontal(|ui| {
                        ui.label("Custom:");
                        let mut res_i32 = desired_resolution as i32;
                        if ui
                            .add(
                                egui::DragValue::new(&mut res_i32)
                                    .speed(1)
                                    .range(16..=max_res as i32)
                                    .suffix("^3"),
                            )
                            .changed()
                        {
                            desired_resolution = (res_i32 as u32).clamp(16, max_res);
                        }
                    });
                    let voxels = (desired_resolution as u64).pow(3);
                    let mem_mb = (voxels as f64 * 4.0) / (1024.0 * 1024.0);
                    ui.weak(format!(
                        "{} voxels ({:.1} MB)",
                        format_voxel_count(voxels),
                        mem_mb
                    ));
                    if desired_resolution > 256 {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 100, 100),
                            format!(
                                "Warning: {:.0} MB RAM — may cause slowdowns or crashes",
                                mem_mb
                            ),
                        );
                    } else if desired_resolution > 128 {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "High resolution — sculpting may be slower",
                        );
                    }
                });

            let sculpt_active = sculpt_state.active_node() == Some(id);

            if sculpt_active {
                if let Some((done, total)) = bake_progress {
                    let frac = done as f32 / total.max(1) as f32;
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(format!("Baking... {:.0}%", frac * 100.0)),
                    );
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("Exit Sculpt Mode").clicked() {
                            actions.push(Action::SetTool(crate::sculpt::ActiveTool::Select));
                        }
                        if let Some(input_id) = input {
                            if ui.button("Re-bake").clicked() {
                                actions.push(Action::RequestBake(BakeRequest {
                                    subtree_root: input_id,
                                    resolution: desired_resolution,
                                    color: material.base_color,
                                    existing_sculpt: Some(id),
                                    flatten: false,
                                }));
                            }
                        }
                    });
                }
            } else if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)),
                );
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Resume Sculpting").clicked() {
                        sculpt_state.activate_preserving_session(id, None);
                    }
                    if ui.button("Remove Modifier").clicked() {
                        actions.push(Action::DeleteNode(id));
                        sculpt_state.deactivate();
                    }
                });
                // Flatten: merge this sculpt + its input into a standalone Sculpt
                if input.is_some() && ui.button("Flatten (merge input + sculpt)").clicked() {
                    let resolution = voxel::max_subtree_resolution(scene, id);
                    actions.push(Action::RequestBake(BakeRequest {
                        subtree_root: id,
                        resolution,
                        color: material.base_color,
                        existing_sculpt: None,
                        flatten: true,
                    }));
                    sculpt_state.deactivate();
                }
            }

            // Light Linking
            draw_light_linking_section(ui, scene, id, actions, active_light_ids);

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Sculpt {
                    position: ref mut p,
                    rotation: ref mut r,
                    material: ref mut mat,
                    layer_intensity: ref mut li,
                    desired_resolution: ref mut dr,
                    ..
                } = node.data
                {
                    *p = position;
                    *r = rotation;
                    *mat = material;
                    *li = layer_intensity;
                    *dr = desired_resolution;
                }
            }
        }
        NodeData::Transform {
            input,
            mut translation,
            mut rotation,
            mut scale,
        } => {
            ui.label("Type: Transform");

            match input {
                Some(iid) => {
                    let input_name = scene
                        .nodes
                        .get(&iid)
                        .map(|n| n.name.as_str())
                        .unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => {
                    ui.label("Input: (empty)");
                }
            }
            ui.separator();

            vec3_editor(ui, "Translation", &mut translation, 0.05, None, "");

            let mut rot_deg = glam::Vec3::new(
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            );
            vec3_editor(ui, "Rotation", &mut rot_deg, 1.0, None, "\u{00B0}");
            rotation = glam::Vec3::new(
                rot_deg.x.to_radians(),
                rot_deg.y.to_radians(),
                rot_deg.z.to_radians(),
            );

            vec3_editor(
                ui,
                "Scale",
                &mut scale,
                0.05,
                Some(SCALE_MIN..=SCALE_MAX),
                "",
            );

            ui.separator();
            if ui
                .button("Delete Node")
                .on_hover_text("Remove this node from the scene")
                .clicked()
            {
                actions.push(Action::DeleteNode(id));
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Transform {
                    translation: ref mut t,
                    rotation: ref mut r,
                    scale: ref mut s,
                    ..
                } = node.data
                {
                    *t = translation;
                    *r = rotation;
                    *s = scale;
                }
            }
        }
        NodeData::Modifier {
            mut kind,
            input,
            mut value,
            mut extra,
        } => {
            let mut new_kind = kind.clone();
            egui::ComboBox::from_id_salt("prop_mod_type")
                .selected_text(new_kind.base_name())
                .show_ui(ui, |ui| {
                    for v in ModifierKind::ALL {
                        ui.selectable_value(&mut new_kind, v.clone(), v.base_name());
                    }
                });
            if new_kind != kind {
                value = new_kind.default_value();
                extra = new_kind.default_extra();
                kind = new_kind;
            }
            ui.separator();

            match input {
                Some(iid) => {
                    let input_name = scene
                        .nodes
                        .get(&iid)
                        .map(|n| n.name.as_str())
                        .unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => {
                    ui.label("Input: (empty)");
                }
            }
            ui.separator();

            match kind {
                ModifierKind::Twist => {
                    ui.horizontal(|ui| {
                        ui.label("Rate");
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.05));
                    });
                }
                ModifierKind::Bend => {
                    ui.horizontal(|ui| {
                        ui.label("Amount");
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.05));
                    });
                }
                ModifierKind::Taper => {
                    ui.horizontal(|ui| {
                        ui.label("Factor");
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.05));
                    });
                }
                ModifierKind::Round => {
                    ui.horizontal(|ui| {
                        ui.label("Radius");
                        ui.add(
                            egui::DragValue::new(&mut value.x)
                                .speed(0.01)
                                .range(0.0..=5.0),
                        );
                    });
                }
                ModifierKind::Onion => {
                    ui.horizontal(|ui| {
                        ui.label("Thickness");
                        ui.add(
                            egui::DragValue::new(&mut value.x)
                                .speed(0.01)
                                .range(0.001..=5.0),
                        );
                    });
                }
                ModifierKind::Elongate => {
                    vec3_editor(ui, "Elongation", &mut value, 0.05, Some(0.0..=10.0), "");
                }
                ModifierKind::Mirror => {
                    ui.horizontal(|ui| {
                        let mut mx = value.x > 0.5;
                        let mut my = value.y > 0.5;
                        let mut mz = value.z > 0.5;
                        ui.checkbox(&mut mx, "X");
                        ui.checkbox(&mut my, "Y");
                        ui.checkbox(&mut mz, "Z");
                        value.x = if mx { 1.0 } else { 0.0 };
                        value.y = if my { 1.0 } else { 0.0 };
                        value.z = if mz { 1.0 } else { 0.0 };
                    });
                }
                ModifierKind::Repeat => {
                    vec3_editor(ui, "Spacing", &mut value, 0.1, Some(0.0..=20.0), "");
                }
                ModifierKind::FiniteRepeat => {
                    vec3_editor(ui, "Spacing", &mut value, 0.1, Some(0.0..=20.0), "");
                    vec3_editor(ui, "Count", &mut extra, 1.0, Some(0.0..=50.0), "");
                }
                ModifierKind::RadialRepeat => {
                    ui.horizontal(|ui| {
                        ui.label("Count:");
                        ui.add(
                            egui::DragValue::new(&mut value.x)
                                .speed(1.0)
                                .range(1.0..=64.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Axis:");
                        let mut ax = value.y as i32;
                        ui.selectable_value(&mut ax, 0, "X");
                        ui.selectable_value(&mut ax, 1, "Y");
                        ui.selectable_value(&mut ax, 2, "Z");
                        value.y = ax as f32;
                    });
                }
                ModifierKind::Offset => {
                    ui.horizontal(|ui| {
                        ui.label("Offset");
                        ui.add(
                            egui::DragValue::new(&mut value.x)
                                .speed(0.01)
                                .range(-1.0..=1.0),
                        );
                    });
                }
                ModifierKind::Noise => {
                    ui.horizontal(|ui| {
                        ui.label("Frequency");
                        ui.add(
                            egui::DragValue::new(&mut value.x)
                                .speed(0.1)
                                .range(0.1..=20.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Amplitude");
                        ui.add(
                            egui::DragValue::new(&mut value.y)
                                .speed(0.01)
                                .range(0.0..=2.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Octaves");
                        let mut octaves_i32 = value.z as i32;
                        ui.add(egui::Slider::new(&mut octaves_i32, 1..=8));
                        value.z = octaves_i32 as f32;
                    });
                }
            }

            ui.separator();
            if ui
                .button("Delete Node")
                .on_hover_text("Remove this node from the scene")
                .clicked()
            {
                actions.push(Action::DeleteNode(id));
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Modifier {
                    kind: ref mut k,
                    value: ref mut v,
                    extra: ref mut e,
                    ..
                } = node.data
                {
                    *k = kind;
                    *v = value;
                    *e = extra;
                }
            }
        }
        NodeData::Light {
            mut light_type,
            mut color,
            mut intensity,
            mut range,
            mut spot_angle,
            mut cast_shadows,
            mut shadow_softness,
            mut shadow_color,
            mut volumetric,
            mut volumetric_density,
            cookie_node,
            mut proximity_mode,
            mut proximity_range,
            mut array_config,
            mut intensity_expr,
            mut color_hue_expr,
        } => {
            ui.horizontal(|ui| {
                ui.label("Type: Light");
                let is_soloed = soloed_light == Some(id);
                let solo_text = if is_soloed { "Unsolo" } else { "Solo" };
                let btn = egui::Button::new(egui::RichText::new(solo_text).small());
                let btn = if is_soloed {
                    btn.fill(egui::Color32::from_rgb(80, 60, 10))
                } else {
                    btn
                };
                if ui.add(btn).clicked() {
                    actions.push(Action::SoloLight(Some(id)));
                }
            });
            if !active_light_ids.contains(&id) {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!(
                        "This light is inactive (max {} active lights). Move it closer to the camera or remove other lights.",
                        crate::graph::scene::MAX_SCENE_LIGHTS,
                    ))
                    .color(egui::Color32::from_rgb(200, 160, 60))
                    .small()
                );
                ui.add_space(4.0);
            }
            ui.separator();

            let mut new_lt = light_type.clone();
            egui::ComboBox::from_id_salt("prop_light_type")
                .selected_text(new_lt.label())
                .show_ui(ui, |ui| {
                    for lt in crate::graph::scene::LightType::ALL {
                        ui.selectable_value(&mut new_lt, lt.clone(), lt.label());
                    }
                });
            if std::mem::discriminant(&new_lt) != std::mem::discriminant(&light_type) {
                light_type = new_lt.clone();
                // Initialize array_config when switching to Array type
                if matches!(new_lt, crate::graph::scene::LightType::Array) && array_config.is_none()
                {
                    array_config = Some(crate::graph::scene::LightArrayConfig::default());
                }
            }
            ui.separator();

            // Array-specific controls
            if matches!(light_type, crate::graph::scene::LightType::Array) {
                if let Some(ref mut cfg) = array_config {
                    ui.label(egui::RichText::new("Array Pattern").strong());
                    let mut new_pattern = cfg.pattern.clone();
                    egui::ComboBox::from_id_salt(format!("array_pattern_{}", id))
                        .selected_text(new_pattern.label())
                        .show_ui(ui, |ui| {
                            for p in crate::graph::scene::ArrayPattern::ALL {
                                ui.selectable_value(&mut new_pattern, p.clone(), p.label());
                            }
                        });
                    cfg.pattern = new_pattern;

                    ui.horizontal(|ui| {
                        ui.label("Count:");
                        let mut count_i32 = cfg.count as i32;
                        ui.add(egui::Slider::new(&mut count_i32, 2..=32));
                        cfg.count = count_i32 as u32;
                    });
                    ui.horizontal(|ui| {
                        ui.label("Radius:");
                        ui.add(egui::Slider::new(&mut cfg.radius, 0.1..=20.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Color Variation:");
                        ui.add(egui::Slider::new(&mut cfg.color_variation, 0.0..=1.0));
                    });
                    ui.separator();
                }
            }

            ui.label("Color");
            let mut color_arr = [color.x, color.y, color.z];
            ui.color_edit_button_rgb(&mut color_arr);
            color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

            ui.horizontal(|ui| {
                let intensity_label = if intensity < 0.0 {
                    egui::RichText::new("Intensity:").color(egui::Color32::from_rgb(255, 80, 80))
                } else {
                    egui::RichText::new("Intensity:").color(egui::Color32::from_rgb(80, 200, 120))
                };
                ui.label(intensity_label);
                ui.add(egui::Slider::new(&mut intensity, -10.0..=10.0));
            });

            // Range: relevant for Point, Spot, and Array
            if matches!(
                light_type,
                crate::graph::scene::LightType::Point
                    | crate::graph::scene::LightType::Spot
                    | crate::graph::scene::LightType::Array
            ) {
                ui.horizontal(|ui| {
                    ui.label("Range:");
                    ui.add(egui::Slider::new(&mut range, 0.1..=50.0));
                });
            }

            // Spot angle: only relevant for Spot
            if matches!(light_type, crate::graph::scene::LightType::Spot) {
                ui.horizontal(|ui| {
                    ui.label("Cone Angle:");
                    ui.add(egui::Slider::new(&mut spot_angle, 1.0..=179.0).suffix("\u{00B0}"));
                });
            }

            // Shadow controls (not shown for Ambient or Array lights)
            if !matches!(
                light_type,
                crate::graph::scene::LightType::Ambient | crate::graph::scene::LightType::Array
            ) {
                ui.separator();
                ui.label(egui::RichText::new("Shadows").strong());
                ui.checkbox(&mut cast_shadows, "Cast Shadows");
                if cast_shadows {
                    ui.horizontal(|ui| {
                        ui.label("Softness:");
                        ui.add(
                            egui::Slider::new(&mut shadow_softness, 1.0..=64.0).logarithmic(true),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Shadow Color:");
                        let mut sc_arr = [shadow_color.x, shadow_color.y, shadow_color.z];
                        ui.color_edit_button_rgb(&mut sc_arr);
                        shadow_color = glam::Vec3::new(sc_arr[0], sc_arr[1], sc_arr[2]);
                    });
                }
            }

            // Volumetric scattering (Point and Spot only)
            if matches!(
                light_type,
                crate::graph::scene::LightType::Point | crate::graph::scene::LightType::Spot
            ) {
                ui.separator();
                ui.label(egui::RichText::new("Volumetric").strong());
                ui.checkbox(&mut volumetric, "Enable Volumetric")
                    .on_hover_text("Volumetric light scattering (god rays)");
                if volumetric {
                    ui.horizontal(|ui| {
                        ui.label("Density:");
                        ui.add(
                            egui::Slider::new(&mut volumetric_density, 0.01..=1.0)
                                .logarithmic(true),
                        );
                    });
                }
            }

            // Proximity modulation (Point and Spot only — position-dependent)
            if matches!(
                light_type,
                crate::graph::scene::LightType::Point | crate::graph::scene::LightType::Spot
            ) {
                ui.separator();
                ui.label(egui::RichText::new("Proximity").strong());
                ui.label(
                    egui::RichText::new("Modulate intensity by distance to surfaces")
                        .weak()
                        .small(),
                );
                let mut new_pm = proximity_mode.clone();
                egui::ComboBox::from_id_salt(format!("proximity_mode_{}", id))
                    .selected_text(new_pm.label())
                    .show_ui(ui, |ui| {
                        for pm in crate::graph::scene::ProximityMode::ALL {
                            ui.selectable_value(&mut new_pm, pm.clone(), pm.label());
                        }
                    });
                if std::mem::discriminant(&new_pm) != std::mem::discriminant(&proximity_mode) {
                    proximity_mode = new_pm;
                }
                if !matches!(proximity_mode, crate::graph::scene::ProximityMode::Off) {
                    ui.horizontal(|ui| {
                        ui.label("Range:");
                        ui.add(egui::Slider::new(&mut proximity_range, 0.1..=10.0));
                    });
                }
            }

            // Cookie shape (Point and Spot only — Directional/Ambient don't benefit)
            if matches!(
                light_type,
                crate::graph::scene::LightType::Point | crate::graph::scene::LightType::Spot
            ) {
                ui.separator();
                ui.label(egui::RichText::new("Cookie Shape").strong());
                ui.label(
                    egui::RichText::new("Use an SDF primitive to shape this light's beam")
                        .weak()
                        .small(),
                );

                // Collect candidate nodes (Primitives and Operations)
                let cookie_label = cookie_node
                    .and_then(|cid| scene.nodes.get(&cid))
                    .map(|n| n.name.clone())
                    .unwrap_or_else(|| "None".to_string());

                egui::ComboBox::from_id_salt(format!("cookie_{}", id))
                    .selected_text(&cookie_label)
                    .show_ui(ui, |ui| {
                        // "None" option to clear the cookie
                        if ui.selectable_label(cookie_node.is_none(), "None").clicked() {
                            actions.push(Action::SetLightCookie {
                                light_id: id,
                                cookie: None,
                            });
                        }
                        // List all Primitive and Operation nodes
                        let mut candidates: Vec<(NodeId, String)> = scene
                            .nodes
                            .iter()
                            .filter_map(|(&nid, n)| {
                                if matches!(
                                    n.data,
                                    NodeData::Primitive { .. } | NodeData::Operation { .. }
                                ) {
                                    Some((nid, n.name.clone()))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        candidates.sort_by(|a, b| a.1.cmp(&b.1));
                        for (cid, name) in candidates {
                            let selected = cookie_node == Some(cid);
                            if ui.selectable_label(selected, &name).clicked() {
                                actions.push(Action::SetLightCookie {
                                    light_id: id,
                                    cookie: Some(cid),
                                });
                            }
                        }
                    });
            }

            // Expressions section (not for Ambient or Array)
            if !matches!(
                light_type,
                crate::graph::scene::LightType::Ambient | crate::graph::scene::LightType::Array
            ) {
                ui.separator();
                egui::CollapsingHeader::new(egui::RichText::new("Expressions").strong())
                    .default_open(intensity_expr.is_some() || color_hue_expr.is_some())
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "Animate properties with math expressions using t (time)",
                            )
                            .weak()
                            .small(),
                        );

                        // Preset dropdown
                        egui::ComboBox::from_id_salt(format!("expr_preset_{}", id))
                            .selected_text("Presets...")
                            .show_ui(ui, |ui| {
                                for preset in crate::expression::EXPRESSION_PRESETS {
                                    if ui.selectable_label(false, preset.name).clicked() {
                                        if !preset.intensity_expr.is_empty() {
                                            intensity_expr =
                                                Some(preset.intensity_expr.to_string());
                                        }
                                        if !preset.color_hue_expr.is_empty() {
                                            color_hue_expr =
                                                Some(preset.color_hue_expr.to_string());
                                        }
                                    }
                                }
                                if ui.selectable_label(false, "Clear All").clicked() {
                                    intensity_expr = None;
                                    color_hue_expr = None;
                                }
                            });
                        ui.add_space(4.0);

                        // Intensity expression
                        ui.label("Intensity Expression:");
                        let mut int_text = intensity_expr.clone().unwrap_or_default();
                        let int_response = ui.add(
                            egui::TextEdit::singleline(&mut int_text)
                                .hint_text("e.g. 0.5 + 0.5 * sin(t * 3.0)")
                                .desired_width(200.0),
                        );
                        if int_response.changed() {
                            intensity_expr = if int_text.trim().is_empty() {
                                None
                            } else {
                                Some(int_text.clone())
                            };
                        }
                        if let Some(ref expr_str) = intensity_expr {
                            match crate::expression::parse_expression(expr_str) {
                                Ok(expr) => {
                                    let now = ui.input(|i| i.time) as f32;
                                    let val = crate::expression::evaluate(&expr, now);
                                    ui.label(
                                        egui::RichText::new(format!("Current: {val:.3}"))
                                            .small()
                                            .color(egui::Color32::from_rgb(120, 200, 120)),
                                    );
                                }
                                Err(err) => {
                                    ui.label(
                                        egui::RichText::new(format!("Error: {err}"))
                                            .small()
                                            .color(egui::Color32::from_rgb(255, 80, 80)),
                                    );
                                }
                            }
                        }
                        ui.add_space(4.0);

                        // Color hue expression
                        ui.label("Color Hue Expression:");
                        let mut hue_text = color_hue_expr.clone().unwrap_or_default();
                        let hue_response = ui.add(
                            egui::TextEdit::singleline(&mut hue_text)
                                .hint_text("e.g. fract(t * 0.1) * 360.0")
                                .desired_width(200.0),
                        );
                        if hue_response.changed() {
                            color_hue_expr = if hue_text.trim().is_empty() {
                                None
                            } else {
                                Some(hue_text.clone())
                            };
                        }
                        if let Some(ref expr_str) = color_hue_expr {
                            match crate::expression::parse_expression(expr_str) {
                                Ok(expr) => {
                                    let now = ui.input(|i| i.time) as f32;
                                    let val = crate::expression::evaluate(&expr, now);
                                    ui.label(
                                        egui::RichText::new(format!("Current: {val:.1}\u{00B0}"))
                                            .small()
                                            .color(egui::Color32::from_rgb(120, 200, 120)),
                                    );
                                }
                                Err(err) => {
                                    ui.label(
                                        egui::RichText::new(format!("Error: {err}"))
                                            .small()
                                            .color(egui::Color32::from_rgb(255, 80, 80)),
                                    );
                                }
                            }
                        }
                    });
            }

            ui.separator();
            if ui
                .button("Delete Node")
                .on_hover_text("Remove this light from the scene")
                .clicked()
            {
                actions.push(Action::DeleteNode(id));
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Light {
                    light_type: ref mut lt,
                    color: ref mut c,
                    intensity: ref mut int,
                    range: ref mut r,
                    spot_angle: ref mut sa,
                    cast_shadows: ref mut cs,
                    shadow_softness: ref mut ss,
                    shadow_color: ref mut sc,
                    volumetric: ref mut vol,
                    volumetric_density: ref mut vd,
                    proximity_mode: ref mut pm,
                    proximity_range: ref mut pr,
                    array_config: ref mut ac,
                    intensity_expr: ref mut ie,
                    color_hue_expr: ref mut ce,
                    ..
                } = node.data
                {
                    *lt = light_type;
                    *c = color;
                    *int = intensity;
                    *r = range;
                    *sa = spot_angle;
                    *cs = cast_shadows;
                    *ss = shadow_softness;
                    *sc = shadow_color;
                    *vol = volumetric;
                    *vd = volumetric_density;
                    *pm = proximity_mode;
                    *pr = proximity_range;
                    *ac = array_config;
                    *ie = intensity_expr;
                    *ce = color_hue_expr;
                }
            }
        }
    }

    // --- Modifier Stack (for Primitive and Sculpt nodes) ---
    let show_stack = scene
        .nodes
        .get(&id)
        .is_some_and(|n| matches!(n.data, NodeData::Primitive { .. } | NodeData::Sculpt { .. }));
    if show_stack {
        let chain = collect_modifier_chain(scene, id);
        ui.separator();
        egui::CollapsingHeader::new("Modifier Stack")
            .default_open(true)
            .show(ui, |ui| {
                if chain.is_empty() {
                    ui.weak("No modifiers applied");
                } else {
                    let mut to_delete = None;
                    for &mod_id in &chain {
                        if let Some(mod_node) = scene.nodes.get(&mod_id) {
                            let badge = match &mod_node.data {
                                NodeData::Transform { .. } => "[Xfm]",
                                NodeData::Modifier { kind, .. } => kind.badge(),
                                _ => "???",
                            };
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{} {}", badge, mod_node.name))
                                        .small(),
                                );
                                if ui
                                    .small_button("X")
                                    .on_hover_text("Remove modifier")
                                    .clicked()
                                {
                                    to_delete = Some(mod_id);
                                }
                            });
                        }
                    }
                    if let Some(del_id) = to_delete {
                        actions.push(Action::DeleteNode(del_id));
                    }
                }
                // Add Modifier menu
                ui.horizontal(|ui| {
                    ui.menu_button("+ Modifier", |ui| {
                        ui.label("Deform");
                        for kind in &[
                            ModifierKind::Twist,
                            ModifierKind::Bend,
                            ModifierKind::Taper,
                            ModifierKind::Noise,
                        ] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove {
                                    target: id,
                                    kind: kind.clone(),
                                });
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        ui.label("Shape");
                        for kind in &[
                            ModifierKind::Round,
                            ModifierKind::Onion,
                            ModifierKind::Elongate,
                            ModifierKind::Offset,
                        ] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove {
                                    target: id,
                                    kind: kind.clone(),
                                });
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        ui.label("Repeat");
                        for kind in &[
                            ModifierKind::Mirror,
                            ModifierKind::Repeat,
                            ModifierKind::FiniteRepeat,
                            ModifierKind::RadialRepeat,
                        ] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove {
                                    target: id,
                                    kind: kind.clone(),
                                });
                                ui.close_menu();
                            }
                        }
                    });
                    if ui.button("+ Transform").clicked() {
                        actions.push(Action::InsertTransformAbove { target: id });
                    }
                });
            });
    }

    // Write name back
    if let Some(node) = scene.nodes.get_mut(&id) {
        node.name = name;
    }
}

/// Draw the Light Linking collapsing section for a geometry node.
/// Shows per-light checkboxes allowing the user to control which lights affect this object.
fn draw_light_linking_section(
    ui: &mut egui::Ui,
    scene: &Scene,
    node_id: NodeId,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
) {
    // Collect active lights in stable order
    let parent_map = scene.build_parent_map();
    let mut light_entries: Vec<(usize, NodeId, String, glam::Vec3)> = Vec::new();

    let mut all_lights: Vec<(NodeId, String, glam::Vec3)> = Vec::new();
    for (&lid, node) in &scene.nodes {
        if let NodeData::Light { color, .. } = &node.data {
            if parent_map.contains_key(&lid) {
                all_lights.push((lid, node.name.clone(), *color));
            }
        }
    }
    all_lights.sort_by_key(|(lid, _, _)| *lid);

    for (slot, (lid, name, color)) in all_lights.iter().enumerate() {
        if active_light_ids.contains(lid) {
            light_entries.push((slot, *lid, name.clone(), *color));
        }
    }

    if light_entries.is_empty() {
        return;
    }

    egui::CollapsingHeader::new("Light Linking")
        .default_open(false)
        .show(ui, |ui| {
            let current_mask = scene.get_light_mask(node_id);

            for &(slot, _, ref light_name, light_color) in &light_entries {
                let slot_u8 = slot as u8;
                let mut linked = (current_mask & (1 << slot_u8)) != 0;
                ui.horizontal(|ui| {
                    // Color swatch
                    let swatch_color = egui::Color32::from_rgb(
                        (light_color.x * 255.0) as u8,
                        (light_color.y * 255.0) as u8,
                        (light_color.z * 255.0) as u8,
                    );
                    let (swatch_rect, _) =
                        ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().rect_filled(swatch_rect, 2.0, swatch_color);

                    if ui.checkbox(&mut linked, light_name).changed() {
                        actions.push(Action::ToggleLightMaskBit {
                            node_id,
                            light_slot: slot_u8,
                            enabled: linked,
                        });
                    }
                });
            }

            // Quick buttons
            ui.horizontal(|ui| {
                if ui.small_button("Link All").clicked() {
                    actions.push(Action::SetLightMask {
                        node_id,
                        mask: 0xFF,
                    });
                }
                if ui.small_button("Unlink All").clicked() {
                    actions.push(Action::SetLightMask {
                        node_id,
                        mask: 0x00,
                    });
                }
            });
        });
}
