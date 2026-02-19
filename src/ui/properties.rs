use eframe::egui;

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};

pub fn draw(ui: &mut egui::Ui, scene: &mut Scene, selected: Option<NodeId>) {
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
            mut scale,
            mut color,
        } => {
            let type_str = match kind {
                SdfPrimitive::Sphere => "Sphere",
                SdfPrimitive::Box => "Box",
                SdfPrimitive::Cylinder => "Cylinder",
                SdfPrimitive::Torus => "Torus",
                SdfPrimitive::Plane => "Plane",
            };
            ui.label(format!("Type: {}", type_str));
            ui.separator();

            ui.label("Position");
            ui.horizontal(|ui| {
                ui.label("X:");
                ui.add(egui::DragValue::new(&mut position.x).speed(0.05));
                ui.label("Y:");
                ui.add(egui::DragValue::new(&mut position.y).speed(0.05));
                ui.label("Z:");
                ui.add(egui::DragValue::new(&mut position.z).speed(0.05));
            });

            ui.label("Scale");
            ui.horizontal(|ui| {
                ui.label("X:");
                ui.add(
                    egui::DragValue::new(&mut scale.x)
                        .speed(0.05)
                        .range(0.01..=100.0),
                );
                ui.label("Y:");
                ui.add(
                    egui::DragValue::new(&mut scale.y)
                        .speed(0.05)
                        .range(0.01..=100.0),
                );
                ui.label("Z:");
                ui.add(
                    egui::DragValue::new(&mut scale.z)
                        .speed(0.05)
                        .range(0.01..=100.0),
                );
            });

            ui.label("Color");
            let mut color_arr = [color.x, color.y, color.z];
            ui.color_edit_button_rgb(&mut color_arr);
            color = glam::Vec3::new(color_arr[0], color_arr[1], color_arr[2]);

            // Write back
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Primitive {
                    position: ref mut p,
                    scale: ref mut s,
                    color: ref mut c,
                    ..
                } = node.data
                {
                    *p = position;
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
            let op_str = match op {
                CsgOp::Union => "Union",
                CsgOp::SmoothUnion => "Smooth Union",
                CsgOp::Subtract => "Subtract",
                CsgOp::Intersect => "Intersect",
            };
            ui.label(format!("Operation: {}", op_str));
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Smooth K:");
                ui.add(egui::Slider::new(&mut smooth_k, 0.0..=2.0));
            });

            ui.separator();
            let left_name = scene
                .nodes
                .get(&left)
                .map(|n| n.name.as_str())
                .unwrap_or("???");
            let right_name = scene
                .nodes
                .get(&right)
                .map(|n| n.name.as_str())
                .unwrap_or("???");
            ui.label(format!("Left: {} (#{})", left_name, left));
            ui.label(format!("Right: {} (#{})", right_name, right));

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
    }

    // Write name back
    if let Some(node) = scene.nodes.get_mut(&id) {
        node.name = name;
    }
}
