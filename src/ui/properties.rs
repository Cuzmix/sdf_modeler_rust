use eframe::egui;

use crate::app::BakeRequest;
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};
use crate::graph::voxel;
use crate::sculpt::{BrushMode, FalloffMode, SculptState, DEFAULT_BRUSH_STRENGTH};

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
            .min_size(egui::vec2(20.0, 16.0));
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

pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: Option<NodeId>,
    sculpt_state: &mut SculptState,
    bake_request: &mut Option<BakeRequest>,
    bake_progress: Option<(u32, u32)>,
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

    // Clone data to avoid borrow conflicts with egui widgets
    let mut name = node.name.clone();
    let node_data = node.data.clone();

    ui.heading(format!("Node #{}", id));
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Name:");
        ui.text_edit_singleline(&mut name);
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

            // Add Sculpt Modifier button
            ui.separator();
            if let Some((done, total)) = bake_progress {
                let frac = done as f32 / total.max(1) as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Add Sculpt Modifier").clicked() {
                        *bake_request = Some(BakeRequest {
                            subtree_root: id,
                            resolution: voxel::DEFAULT_RESOLUTION,
                            color,
                            existing_sculpt: None,
                            flatten: false,
                        });
                        if let Some(node) = scene.nodes.get_mut(&id) {
                            node.name = name.clone();
                        }
                        return;
                    }
                    if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                        scene.remove_node(id);
                        return;
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
            if left.is_some() && right.is_some() {
                if ui.button("Swap Inputs").clicked() {
                    scene.swap_children(id);
                }
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
                        *bake_request = Some(BakeRequest {
                            subtree_root: id,
                            resolution: voxel::DEFAULT_RESOLUTION,
                            color: sculpt_color,
                            existing_sculpt: None,
                            flatten: false,
                        });
                        return;
                    }
                    if ui.button("Flatten to Sculpt").clicked() {
                        let resolution = voxel::max_subtree_resolution(scene, id);
                        let sculpt_color = glam::Vec3::new(0.6, 0.6, 0.6);
                        *bake_request = Some(BakeRequest {
                            subtree_root: id,
                            resolution,
                            color: sculpt_color,
                            existing_sculpt: None,
                            flatten: true,
                        });
                        return;
                    }
                });
                if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                    scene.remove_node(id);
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

            ui.separator();
            ui.label("Sculpting");
            let mut res_i32 = desired_resolution as i32;
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                ui.add(egui::Slider::new(&mut res_i32, 32..=256).suffix("^3"));
            });
            desired_resolution = res_i32 as u32;
            let voxels = (desired_resolution as u64).pow(3);
            let mem_mb = (voxels * 4) as f64 / (1024.0 * 1024.0);
            ui.weak(format!("{} voxels ({:.1} MB)", voxels, mem_mb));

            let sculpt_active = sculpt_state.active_node() == Some(id);

            // Brush settings (only when actively sculpting)
            if let SculptState::Active {
                ref mut brush_mode,
                ref mut brush_radius,
                ref mut brush_strength,
                ref mut falloff_mode,
                ref mut smooth_iterations,
                ref mut lazy_radius,
                ref mut surface_constraint,
                ref mut symmetry_axis,
                ..
            } = sculpt_state
            {
                let prev_mode = brush_mode.clone();
                ui.horizontal(|ui| {
                    ui.label("Brush:");
                    ui.selectable_value(brush_mode, BrushMode::Add, "Add");
                    ui.selectable_value(brush_mode, BrushMode::Carve, "Carve");
                    ui.selectable_value(brush_mode, BrushMode::Smooth, "Smooth");
                    ui.selectable_value(brush_mode, BrushMode::Flatten, "Flatten");
                    ui.selectable_value(brush_mode, BrushMode::Inflate, "Inflate");
                    ui.selectable_value(brush_mode, BrushMode::Grab, "Grab");
                });
                if *brush_mode != prev_mode {
                    if *brush_mode == BrushMode::Grab && *brush_strength < 0.5 {
                        *brush_strength = 1.0;
                    } else if prev_mode == BrushMode::Grab && *brush_strength > 0.5 {
                        *brush_strength = DEFAULT_BRUSH_STRENGTH;
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Falloff:");
                    ui.selectable_value(falloff_mode, FalloffMode::Smooth, "Smooth");
                    ui.selectable_value(falloff_mode, FalloffMode::Linear, "Linear");
                    ui.selectable_value(falloff_mode, FalloffMode::Sharp, "Sharp");
                    ui.selectable_value(falloff_mode, FalloffMode::Flat, "Flat");
                });
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(egui::Slider::new(brush_radius, 0.05..=2.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Strength:");
                    let range = if *brush_mode == BrushMode::Grab {
                        0.1..=3.0
                    } else {
                        0.01..=0.5
                    };
                    ui.add(egui::Slider::new(brush_strength, range));
                });
                if *brush_mode == BrushMode::Smooth {
                    ui.horizontal(|ui| {
                        ui.label("Iterations:");
                        let mut iters = *smooth_iterations as i32;
                        ui.add(egui::Slider::new(&mut iters, 1..=10));
                        *smooth_iterations = iters as u32;
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Stabilize:");
                    ui.add(egui::Slider::new(lazy_radius, 0.0..=0.5));
                });
                ui.horizontal(|ui| {
                    ui.label("Surface:");
                    ui.add(egui::Slider::new(surface_constraint, 0.0..=1.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Symmetry:");
                    ui.selectable_value(symmetry_axis, None, "Off");
                    ui.selectable_value(symmetry_axis, Some(0), "X");
                    ui.selectable_value(symmetry_axis, Some(1), "Y");
                    ui.selectable_value(symmetry_axis, Some(2), "Z");
                });
            }

            if sculpt_active {
                if let Some((done, total)) = bake_progress {
                    let frac = done as f32 / total.max(1) as f32;
                    ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("Exit Sculpt Mode").clicked() {
                            *sculpt_state = SculptState::Inactive;
                        }
                        if let Some(input_id) = input {
                            if ui.button("Re-bake").clicked() {
                                *bake_request = Some(BakeRequest {
                                    subtree_root: input_id,
                                    resolution: desired_resolution,
                                    color,
                                    existing_sculpt: Some(id),
                                    flatten: false,
                                });
                            }
                        }
                    });
                }
            } else {
                if let Some((done, total)) = bake_progress {
                    let frac = done as f32 / total.max(1) as f32;
                    ui.add(egui::ProgressBar::new(frac).text(format!("Baking... {:.0}%", frac * 100.0)));
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("Resume Sculpting").clicked() {
                            *sculpt_state = SculptState::new_active(id);
                        }
                        if ui.button("Remove Modifier").clicked() {
                            scene.remove_node(id);
                            *sculpt_state = SculptState::Inactive;
                            return;
                        }
                    });
                    // Flatten: merge this sculpt + its input into a standalone Sculpt
                    if input.is_some() {
                        if ui.button("Flatten (merge input + sculpt)").clicked() {
                            let resolution = voxel::max_subtree_resolution(scene, id);
                            *bake_request = Some(BakeRequest {
                                subtree_root: id,
                                resolution,
                                color,
                                existing_sculpt: None,
                                flatten: true,
                            });
                            *sculpt_state = SculptState::Inactive;
                            return;
                        }
                    }
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
                    desired_resolution: ref mut dr,
                    ..
                } = node.data
                {
                    *p = position;
                    *r = rotation;
                    *c = color;
                    *m = metallic;
                    *rgh = roughness;
                    *dr = desired_resolution;
                }
            }
        }
        NodeData::Transform {
            mut kind,
            input,
            mut value,
        } => {
            let mut new_kind = kind.clone();
            egui::ComboBox::from_id_salt("prop_xform_type")
                .selected_text(new_kind.base_name())
                .show_ui(ui, |ui| {
                    for v in TransformKind::ALL {
                        ui.selectable_value(&mut new_kind, v.clone(), v.base_name());
                    }
                });
            if new_kind != kind {
                value = new_kind.default_value();
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
                TransformKind::Translate => {
                    vec3_editor(ui, "Offset", &mut value, 0.05, None, "");
                }
                TransformKind::Rotate => {
                    let mut rot_deg = glam::Vec3::new(
                        value.x.to_degrees(),
                        value.y.to_degrees(),
                        value.z.to_degrees(),
                    );
                    vec3_editor(ui, "Rotation", &mut rot_deg, 1.0, None, "\u{00B0}");
                    value = glam::Vec3::new(
                        rot_deg.x.to_radians(),
                        rot_deg.y.to_radians(),
                        rot_deg.z.to_radians(),
                    );
                }
                TransformKind::Scale => {
                    vec3_editor(ui, "Scale", &mut value, 0.05, Some(SCALE_MIN..=SCALE_MAX), "");
                }
            }

            ui.separator();
            if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                scene.remove_node(id);
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Transform { kind: ref mut k, value: ref mut v, .. } = node.data {
                    *k = kind;
                    *v = value;
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
            }

            ui.separator();
            if ui.button("Delete Node").on_hover_text("Remove this node from the scene").clicked() {
                scene.remove_node(id);
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

    // Write name back
    if let Some(node) = scene.nodes.get_mut(&id) {
        node.name = name;
    }
}
