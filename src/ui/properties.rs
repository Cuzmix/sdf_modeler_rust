use eframe::egui;

use crate::app::BakeRequest;
use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive};
use crate::graph::voxel;
use crate::sculpt::SculptState;

const SCALE_MIN: f32 = 0.01;
const SCALE_MAX: f32 = 100.0;

/// Material preset: (name, optional_color, metallic, roughness, fresnel, emissive_intensity).
/// `None` for color means "keep current color".
struct MaterialPreset {
    name: &'static str,
    color: Option<[f32; 3]>,
    metallic: f32,
    roughness: f32,
    fresnel: f32,
    emissive_intensity: f32,
}

const MATERIAL_PRESETS: &[MaterialPreset] = &[
    MaterialPreset { name: "Default",  color: None,                        metallic: 0.0, roughness: 0.5,  fresnel: 0.04, emissive_intensity: 0.0 },
    MaterialPreset { name: "Gold",     color: Some([1.0, 0.76, 0.33]),     metallic: 1.0, roughness: 0.3,  fresnel: 0.95, emissive_intensity: 0.0 },
    MaterialPreset { name: "Silver",   color: Some([0.95, 0.93, 0.88]),    metallic: 1.0, roughness: 0.2,  fresnel: 0.97, emissive_intensity: 0.0 },
    MaterialPreset { name: "Chrome",   color: Some([0.77, 0.78, 0.78]),    metallic: 1.0, roughness: 0.05, fresnel: 0.98, emissive_intensity: 0.0 },
    MaterialPreset { name: "Plastic",  color: None,                        metallic: 0.0, roughness: 0.4,  fresnel: 0.04, emissive_intensity: 0.0 },
    MaterialPreset { name: "Ceramic",  color: None,                        metallic: 0.0, roughness: 0.1,  fresnel: 0.04, emissive_intensity: 0.0 },
    MaterialPreset { name: "Rubber",   color: None,                        metallic: 0.0, roughness: 0.9,  fresnel: 0.02, emissive_intensity: 0.0 },
    MaterialPreset { name: "Glow",     color: None,                        metallic: 0.0, roughness: 0.5,  fresnel: 0.04, emissive_intensity: 2.0 },
];

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
            let btn = egui::Button::new(
                egui::RichText::new(label).small().color(
                    if preset[0] + preset[1] + preset[2] > 1.5 {
                        egui::Color32::BLACK
                    } else {
                        egui::Color32::WHITE
                    },
                ),
            )
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
        for (axis_label, component) in [("X:", &mut value.x), ("Y:", &mut value.y), ("Z:", &mut value.z)] {
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

pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: Option<NodeId>,
    sculpt_state: &mut SculptState,
    bake_progress: Option<(u32, u32)>,
    actions: &mut ActionSink,
) {
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
            mut color,
            mut roughness,
            mut metallic,
            mut emissive,
            mut emissive_intensity,
            mut fresnel,
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
                    // Material presets
                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt("prim_mat_preset")
                            .selected_text("Apply...")
                            .width(80.0)
                            .show_ui(ui, |ui| {
                                for preset in MATERIAL_PRESETS {
                                    if ui.selectable_label(false, preset.name).clicked() {
                                        if let Some(c) = preset.color {
                                            color = glam::Vec3::new(c[0], c[1], c[2]);
                                        }
                                        metallic = preset.metallic;
                                        roughness = preset.roughness;
                                        fresnel = preset.fresnel;
                                        emissive_intensity = preset.emissive_intensity;
                                        if preset.emissive_intensity > 0.0 && emissive == glam::Vec3::ZERO {
                                            emissive = color;
                                        }
                                    }
                                }
                            });
                    });

                    ui.label("Color");
                    let mut color_arr = [color.x, color.y, color.z];
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut color_arr);
                        color_presets_row(ui, &mut color_arr);
                    });
                    color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

                    ui.horizontal(|ui| {
                        ui.label("Metallic:");
                        ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Roughness:");
                        ui.add(egui::Slider::new(&mut roughness, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Fresnel:");
                        ui.add(egui::Slider::new(&mut fresnel, 0.0..=1.0));
                    });

                    ui.separator();
                    ui.label("Emissive");
                    let mut emissive_arr = [emissive.x, emissive.y, emissive.z];
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut emissive_arr);
                        if ui.small_button("= Color").on_hover_text("Set emissive to object color").clicked() {
                            emissive_arr = [color.x, color.y, color.z];
                        }
                    });
                    emissive = glam::Vec3::new(emissive_arr[0], emissive_arr[1], emissive_arr[2]);
                    ui.horizontal(|ui| {
                        ui.label("Intensity:");
                        ui.add(egui::Slider::new(&mut emissive_intensity, 0.0..=5.0));
                    });
                });

            // Add Sculpt Modifier button
            ui.separator();
            if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Add Sculpt Modifier").clicked() {
                        actions.push(Action::RequestBake(BakeRequest {
                            subtree_root: id,
                            resolution: voxel::DEFAULT_RESOLUTION,
                            color,
                            existing_sculpt: None,
                            flatten: false,
                        }));
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            node.name = name.clone();
                        }
                        return;
                    }
                    if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                        actions.push(Action::DeleteNode(id));
                    }
                });
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Primitive {
                    kind: ref mut k,
                    position: ref mut p,
                    rotation: ref mut r,
                    scale: ref mut s,
                    color: ref mut c,
                    metallic: ref mut m,
                    roughness: ref mut rgh,
                    emissive: ref mut em,
                    emissive_intensity: ref mut ei,
                    fresnel: ref mut fr,
                    ..
                } = node.data
                {
                    *k = kind;
                    *p = position;
                    *r = rotation;
                    *s = scale;
                    *c = color;
                    *m = metallic;
                    *rgh = roughness;
                    *em = emissive;
                    *ei = emissive_intensity;
                    *fr = fresnel;
                }
            }
        }
        NodeData::Operation {
            mut op,
            mut smooth_k,
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
                op = new_op;
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Smooth K:");
                ui.add(egui::Slider::new(&mut smooth_k, 0.0..=2.0));
            });

            ui.separator();
            match left {
                Some(lid) => {
                    let left_name = scene.nodes.get(&lid).map(|n| n.name.as_str()).unwrap_or("???");
                    ui.label(format!("Left: {} (#{})", left_name, lid));
                }
                None => { ui.label("Left: (empty)"); }
            }
            match right {
                Some(rid) => {
                    let right_name = scene.nodes.get(&rid).map(|n| n.name.as_str()).unwrap_or("???");
                    ui.label(format!("Right: {} (#{})", right_name, rid));
                }
                None => { ui.label("Right: (empty)"); }
            }
            if left.is_some() && right.is_some() && ui.button("Swap Inputs").clicked() {
                actions.push(Action::SwapChildren(id));
            }

            // Add Sculpt Modifier / Flatten buttons
            ui.separator();
            if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
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
                if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                    actions.push(Action::DeleteNode(id));
                    return;
                }
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Operation {
                    op: ref mut o,
                    smooth_k: ref mut k,
                    ..
                } = node.data
                {
                    *o = op;
                    *k = smooth_k;
                }
            }
        }
        NodeData::Sculpt {
            input,
            mut position,
            mut rotation,
            mut color,
            mut roughness,
            mut metallic,
            mut emissive,
            mut emissive_intensity,
            mut fresnel,
            mut layer_intensity,
            voxel_grid: _,
            mut desired_resolution,
        } => {
            ui.label("Type: Sculpt Modifier");
            ui.separator();

            match input {
                Some(iid) => {
                    let input_name = scene.nodes.get(&iid).map(|n| n.name.as_str()).unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => { ui.label("Input: (empty)"); }
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
                    // Material presets
                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt("sculpt_mat_preset")
                            .selected_text("Apply...")
                            .width(80.0)
                            .show_ui(ui, |ui| {
                                for preset in MATERIAL_PRESETS {
                                    if ui.selectable_label(false, preset.name).clicked() {
                                        if let Some(c) = preset.color {
                                            color = glam::Vec3::new(c[0], c[1], c[2]);
                                        }
                                        metallic = preset.metallic;
                                        roughness = preset.roughness;
                                        fresnel = preset.fresnel;
                                        emissive_intensity = preset.emissive_intensity;
                                        if preset.emissive_intensity > 0.0 && emissive == glam::Vec3::ZERO {
                                            emissive = color;
                                        }
                                    }
                                }
                            });
                    });

                    ui.label("Color");
                    let mut color_arr = [color.x, color.y, color.z];
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut color_arr);
                        color_presets_row(ui, &mut color_arr);
                    });
                    color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

                    ui.horizontal(|ui| {
                        ui.label("Metallic:");
                        ui.add(egui::Slider::new(&mut metallic, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Roughness:");
                        ui.add(egui::Slider::new(&mut roughness, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Fresnel:");
                        ui.add(egui::Slider::new(&mut fresnel, 0.0..=1.0));
                    });

                    ui.separator();
                    ui.label("Emissive");
                    let mut emissive_arr = [emissive.x, emissive.y, emissive.z];
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut emissive_arr);
                        if ui.small_button("= Color").on_hover_text("Set emissive to object color").clicked() {
                            emissive_arr = [color.x, color.y, color.z];
                        }
                    });
                    emissive = glam::Vec3::new(emissive_arr[0], emissive_arr[1], emissive_arr[2]);
                    ui.horizontal(|ui| {
                        ui.label("Intensity:");
                        ui.add(egui::Slider::new(&mut emissive_intensity, 0.0..=5.0));
                    });
                });

            egui::CollapsingHeader::new("Sculpting")
                .default_open(true)
                .show(ui, |ui| {
                    let mut res_i32 = desired_resolution as i32;
                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.add(egui::Slider::new(&mut res_i32, 32..=256).suffix("^3"));
                    });
                    desired_resolution = res_i32 as u32;
                    let voxels = (desired_resolution as u64).pow(3);
                    let mem_mb = (voxels * 4) as f64 / (1024.0 * 1024.0);
                    ui.weak(format!("{} voxels ({:.1} MB)", voxels, mem_mb));
                });

            let sculpt_active = sculpt_state.active_node() == Some(id);

            if sculpt_active {
                if let Some((done, total)) = bake_progress {
                    let frac = done as f32 / total.max(1) as f32;
                    ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
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
                                    color,
                                    existing_sculpt: Some(id),
                                    flatten: false,
                                }));
                            }
                        }
                    });
                }
            } else if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Resume Sculpting").clicked() {
                        *sculpt_state = SculptState::new_active(id);
                    }
                    if ui.button("Remove Modifier").clicked() {
                        actions.push(Action::DeleteNode(id));
                        *sculpt_state = SculptState::Inactive;
                    }
                });
                // Flatten: merge this sculpt + its input into a standalone Sculpt
                if input.is_some() && ui.button("Flatten (merge input + sculpt)").clicked() {
                    let resolution = voxel::max_subtree_resolution(scene, id);
                    actions.push(Action::RequestBake(BakeRequest {
                        subtree_root: id,
                        resolution,
                        color,
                        existing_sculpt: None,
                        flatten: true,
                    }));
                    *sculpt_state = SculptState::Inactive;
                }
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Sculpt {
                    position: ref mut p,
                    rotation: ref mut r,
                    color: ref mut c,
                    metallic: ref mut m,
                    roughness: ref mut rgh,
                    emissive: ref mut em,
                    emissive_intensity: ref mut ei,
                    fresnel: ref mut fr,
                    layer_intensity: ref mut li,
                    desired_resolution: ref mut dr,
                    ..
                } = node.data
                {
                    *p = position;
                    *r = rotation;
                    *c = color;
                    *m = metallic;
                    *rgh = roughness;
                    *em = emissive;
                    *ei = emissive_intensity;
                    *fr = fresnel;
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
                    let input_name = scene.nodes.get(&iid).map(|n| n.name.as_str()).unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => { ui.label("Input: (empty)"); }
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

            vec3_editor(ui, "Scale", &mut scale, 0.05, Some(SCALE_MIN..=SCALE_MAX), "");

            ui.separator();
            if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                actions.push(Action::DeleteNode(id));
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Transform {
                    translation: ref mut t,
                    rotation: ref mut r,
                    scale: ref mut s, ..
                } = node.data {
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
                    let input_name = scene.nodes.get(&iid).map(|n| n.name.as_str()).unwrap_or("???");
                    ui.label(format!("Input: {} (#{})", input_name, iid));
                }
                None => { ui.label("Input: (empty)"); }
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
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.01).range(0.0..=5.0));
                    });
                }
                ModifierKind::Onion => {
                    ui.horizontal(|ui| {
                        ui.label("Thickness");
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.01).range(0.001..=5.0));
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
                        ui.add(egui::DragValue::new(&mut value.x).speed(1.0).range(1.0..=64.0));
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
                        ui.add(egui::DragValue::new(&mut value.x).speed(0.01).range(-1.0..=1.0));
                    });
                }
            }

            ui.separator();
            if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                actions.push(Action::DeleteNode(id));
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Modifier { kind: ref mut k, value: ref mut v, extra: ref mut e, .. } = node.data {
                    *k = kind;
                    *v = value;
                    *e = extra;
                }
            }
        }
    }

    // --- Modifier Stack (for Primitive and Sculpt nodes) ---
    let show_stack = scene.nodes.get(&id).is_some_and(|n| {
        matches!(n.data, NodeData::Primitive { .. } | NodeData::Sculpt { .. })
    });
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
                                if ui.small_button("X").on_hover_text("Remove modifier").clicked() {
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
                        for kind in &[ModifierKind::Twist, ModifierKind::Bend, ModifierKind::Taper] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove { target: id, kind: kind.clone() });
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        ui.label("Shape");
                        for kind in &[ModifierKind::Round, ModifierKind::Onion, ModifierKind::Elongate, ModifierKind::Offset] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove { target: id, kind: kind.clone() });
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        ui.label("Repeat");
                        for kind in &[ModifierKind::Mirror, ModifierKind::Repeat, ModifierKind::FiniteRepeat, ModifierKind::RadialRepeat] {
                            if ui.button(kind.base_name()).clicked() {
                                actions.push(Action::InsertModifierAbove { target: id, kind: kind.clone() });
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
