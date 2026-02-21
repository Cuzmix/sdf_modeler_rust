use eframe::egui;

use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel;
use crate::sculpt::{self, BrushMode, SculptState};

const SCALE_MIN: f32 = 0.01;
const SCALE_MAX: f32 = 100.0;

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
) {
    let Some(id) = selected else {
        ui.centered_and_justified(|ui| {
            ui.label("No selection");
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
            kind,
            mut position,
            mut rotation,
            mut scale,
            mut color,
            ..
        } => {
            ui.label(format!("Type: {}", kind.base_name()));
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

            vec3_editor(ui, "Scale", &mut scale, 0.05, Some(SCALE_MIN..=SCALE_MAX), "");

            ui.label("Color");
            let mut color_arr = [color.x, color.y, color.z];
            ui.color_edit_button_rgb(&mut color_arr);
            color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

            // Add Sculpt Modifier button
            ui.separator();
            if ui.button("Add Sculpt Modifier").clicked() {
                let (grid, center) = voxel::bake_subtree(scene, id, voxel::DEFAULT_RESOLUTION);
                let sculpt_id = scene.insert_sculpt_above(
                    id, center, glam::Vec3::ZERO, color, grid,
                );
                *sculpt_state = SculptState::Active {
                    node_id: sculpt_id,
                    brush_mode: BrushMode::Add,
                    brush_radius: sculpt::DEFAULT_BRUSH_RADIUS,
                    brush_strength: sculpt::DEFAULT_BRUSH_STRENGTH,
                };
                if let Some(node) = scene.nodes.get_mut(&id) {
                    node.name = name;
                }
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Primitive {
                    position: ref mut p,
                    rotation: ref mut r,
                    scale: ref mut s,
                    color: ref mut c,
                    ..
                } = node.data
                {
                    *p = position;
                    *r = rotation;
                    *s = scale;
                    *c = color;
                }
            }
        }
        NodeData::Operation {
            op,
            mut smooth_k,
            left,
            right,
        } => {
            ui.label(format!("Operation: {}", op.base_name()));
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

            // Add Sculpt Modifier button
            ui.separator();
            if ui.button("Add Sculpt Modifier").clicked() {
                let (grid, center) = voxel::bake_subtree(scene, id, voxel::DEFAULT_RESOLUTION);
                // Use a neutral color for operation sculpts
                let sculpt_color = glam::Vec3::new(0.6, 0.6, 0.6);
                let sculpt_id = scene.insert_sculpt_above(
                    id, center, glam::Vec3::ZERO, sculpt_color, grid,
                );
                *sculpt_state = SculptState::Active {
                    node_id: sculpt_id,
                    brush_mode: BrushMode::Add,
                    brush_radius: sculpt::DEFAULT_BRUSH_RADIUS,
                    brush_strength: sculpt::DEFAULT_BRUSH_STRENGTH,
                };
                return;
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Operation {
                    smooth_k: ref mut k,
                    ..
                } = node.data
                {
                    *k = smooth_k;
                }
            }
        }
        NodeData::Sculpt {
            input,
            mut position,
            mut rotation,
            mut color,
            voxel_grid,
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
            ui.color_edit_button_rgb(&mut color_arr);
            color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

            ui.separator();
            ui.label("Sculpting");
            ui.label(format!("Resolution: {}^3", voxel_grid.resolution));

            let sculpt_active = sculpt_state.active_node() == Some(id);

            // Brush settings (only when actively sculpting)
            if let SculptState::Active {
                ref mut brush_mode,
                ref mut brush_radius,
                ref mut brush_strength,
                ..
            } = sculpt_state
            {
                ui.horizontal(|ui| {
                    ui.label("Brush:");
                    ui.selectable_value(brush_mode, BrushMode::Add, "Add");
                    ui.selectable_value(brush_mode, BrushMode::Carve, "Carve");
                });
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(egui::Slider::new(brush_radius, 0.05..=2.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Strength:");
                    ui.add(egui::Slider::new(brush_strength, 0.01..=0.5));
                });
            }

            if sculpt_active {
                ui.horizontal(|ui| {
                    if ui.button("Exit Sculpt Mode").clicked() {
                        *sculpt_state = SculptState::Inactive;
                    }
                    if let Some(input_id) = input {
                        if ui.button("Re-bake").clicked() {
                            let (grid, center) =
                                voxel::bake_subtree(scene, input_id, voxel::DEFAULT_RESOLUTION);
                            if let Some(node) = scene.nodes.get_mut(&id) {
                                if let NodeData::Sculpt {
                                    voxel_grid: ref mut vg,
                                    position: ref mut p,
                                    ..
                                } = node.data
                                {
                                    *vg = grid;
                                    *p = center;
                                }
                            }
                            position = center;
                        }
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Resume Sculpting").clicked() {
                        *sculpt_state = SculptState::Active {
                            node_id: id,
                            brush_mode: BrushMode::Add,
                            brush_radius: sculpt::DEFAULT_BRUSH_RADIUS,
                            brush_strength: sculpt::DEFAULT_BRUSH_STRENGTH,
                        };
                    }
                    if ui.button("Remove Modifier").clicked() {
                        // Remove the sculpt node, reconnect input to parents
                        // For simplicity: just delete the sculpt node (its parent will be disconnected)
                        scene.remove_node(id);
                        *sculpt_state = SculptState::Inactive;
                        return;
                    }
                });
            }

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Sculpt {
                    position: ref mut p,
                    rotation: ref mut r,
                    color: ref mut c,
                    ..
                } = node.data
                {
                    *p = position;
                    *r = rotation;
                    *c = color;
                }
            }
        }
    }

    // Write name back
    if let Some(node) = scene.nodes.get_mut(&id) {
        node.name = name;
    }
}
