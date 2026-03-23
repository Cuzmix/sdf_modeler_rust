use std::collections::HashSet;

use eframe::egui;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::app::state::MultiTransformSessionState;
use crate::app::BakeRequest;
use crate::graph::presented_object::{
    collect_presented_selection, collect_presented_wrapper_chain, PresentedObjectKind,
    PresentedObjectRef,
};
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive};
use crate::graph::voxel;
use crate::material_preset::MaterialLibrary;
use crate::sculpt::SculptState;
use crate::settings::SelectionBehaviorSettings;
use crate::ui::gizmo::GizmoSpace;

use super::properties;

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
    multi_transform_edit: &mut MultiTransformSessionState,
    gizmo_space: &GizmoSpace,
    selection_behavior: &SelectionBehaviorSettings,
) {
    let presented_selection = collect_presented_selection(scene, selected, selected_set);
    if presented_selection.ordered.len() > 1 {
        draw_multi_selection(ui, &presented_selection.ordered, scene, actions);
        return;
    }

    let Some(object) = presented_selection.primary else {
        ui.vertical_centered(|ui| {
            ui.add_space(40.0);
            ui.label("No selection");
            ui.add_space(8.0);
            ui.weak("Click an object in the viewport or scene tray");
        });
        return;
    };

    if matches!(object.kind, PresentedObjectKind::Light) {
        let light_selected_set = HashSet::from([object.host_id]);
        properties::draw(
            ui,
            scene,
            Some(object.host_id),
            &light_selected_set,
            sculpt_state,
            bake_progress,
            actions,
            active_light_ids,
            max_sculpt_resolution,
            soloed_light,
            material_library,
            multi_transform_edit,
            gizmo_space,
            selection_behavior,
        );
        return;
    }

    draw_presented_object(
        ui,
        scene,
        object,
        sculpt_state,
        bake_progress,
        actions,
        active_light_ids,
        max_sculpt_resolution,
        material_library,
    );
}

fn draw_multi_selection(
    ui: &mut egui::Ui,
    objects: &[PresentedObjectRef],
    scene: &Scene,
    actions: &mut ActionSink,
) {
    ui.heading(format!("{} Objects Selected", objects.len()));
    ui.separator();
    ui.weak("Use the viewport gizmo to move, rotate, or scale the selected objects together.");
    ui.add_space(8.0);

    egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
        for object in objects {
            let Some(node) = scene.nodes.get(&object.host_id) else {
                continue;
            };
            let sculpt_badge = if object.attached_sculpt_id.is_some() {
                " [Sculpt]"
            } else {
                ""
            };
            ui.label(format!("- {}{}", node.name, sculpt_badge));
        }
    });

    ui.separator();
    if ui.button("Delete Selected Objects").clicked() {
        for object in objects {
            actions.push(Action::DeletePresentedObject(object.object_root_id));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_presented_object(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    sculpt_state: &mut SculptState,
    bake_progress: Option<(u32, u32)>,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
    max_sculpt_resolution: u32,
    material_library: &mut MaterialLibrary,
) {
    let Some(host_node) = scene.nodes.get(&object.host_id) else {
        ui.centered_and_justified(|ui| {
            ui.label("Selected object not found");
        });
        return;
    };

    let mut name = host_node.name.clone();
    let host_data = host_node.data.clone();
    let is_locked = host_node.locked;

    ui.heading(format!("Object #{}", object.host_id));
    if is_locked {
        ui.colored_label(egui::Color32::from_rgb(255, 180, 80), "Locked");
    }
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Name:");
        ui.add_enabled(!is_locked, egui::TextEdit::singleline(&mut name));
    });

    ui.separator();

    match host_data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            material,
            ..
        } => {
            draw_primitive_host(
                ui,
                scene,
                object,
                kind,
                position,
                rotation,
                scale,
                material,
                actions,
                active_light_ids,
                material_library,
            );
        }
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            left,
            right,
        } => {
            draw_operation_host(
                ui,
                scene,
                object,
                op,
                smooth_k,
                steps,
                color_blend,
                left,
                right,
                actions,
                active_light_ids,
            );
        }
        NodeData::Sculpt {
            input: None,
            position,
            rotation,
            material,
            layer_intensity,
            voxel_grid,
            desired_resolution,
        } => {
            draw_voxel_host(
                ui,
                scene,
                object,
                sculpt_state,
                position,
                rotation,
                material,
                layer_intensity,
                voxel_grid,
                desired_resolution,
                actions,
                active_light_ids,
                max_sculpt_resolution,
                material_library,
            );
        }
        _ => {
            ui.weak("This object type is only editable in the advanced raw-node inspector.");
        }
    }

    ui.separator();
    draw_host_actions(ui, scene, object, bake_progress, actions);
    ui.separator();
    draw_object_stack(ui, scene, object, sculpt_state, actions, bake_progress);

    if let Some(node) = scene.nodes.get_mut(&object.host_id) {
        node.name = name;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_primitive_host(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    mut kind: SdfPrimitive,
    mut position: Vec3,
    mut rotation: Vec3,
    mut scale: Vec3,
    mut material: crate::graph::scene::MaterialParams,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
    material_library: &mut MaterialLibrary,
) {
    let has_sculpt_layer = object.attached_sculpt_id.is_some();
    egui::ComboBox::from_id_salt(("presented_primitive_type", object.host_id))
        .selected_text(kind.base_name())
        .show_ui(ui, |ui| {
            for primitive in SdfPrimitive::ALL {
                ui.selectable_value(&mut kind, primitive.clone(), primitive.base_name());
            }
        });

    ui.separator();
    egui::CollapsingHeader::new(if has_sculpt_layer {
        "Base Shape"
    } else {
        "Base Transform"
    })
        .default_open(true)
        .show(ui, |ui| {
            if has_sculpt_layer {
                ui.small(
                    "Object movement now belongs to the outer object transform. These controls only change the analytical base shape.",
                );
            } else {
                properties::vec3_editor(ui, "Position", &mut position, 0.05, None, "");
                let mut rotation_deg = Vec3::new(
                    rotation.x.to_degrees(),
                    rotation.y.to_degrees(),
                    rotation.z.to_degrees(),
                );
                properties::vec3_editor(ui, "Rotation", &mut rotation_deg, 1.0, None, " deg");
                rotation = Vec3::new(
                    rotation_deg.x.to_radians(),
                    rotation_deg.y.to_radians(),
                    rotation_deg.z.to_radians(),
                );
            }

            let params = kind.scale_params();
            if params.is_empty() {
                ui.weak("This primitive does not expose base size parameters.");
            } else {
                ui.label("Size");
                ui.horizontal(|ui| {
                    for &(label, axis) in params {
                        ui.label(format!("{label}:"));
                        let component = match axis {
                            0 => &mut scale.x,
                            1 => &mut scale.y,
                            _ => &mut scale.z,
                        };
                        ui.add(egui::DragValue::new(component).speed(0.05));
                    }
                });
            }
        });

    egui::CollapsingHeader::new("Material")
        .default_open(true)
        .show(ui, |ui| {
            properties::draw_material_editor(
                ui,
                "presented_primitive",
                &mut material,
                material_library,
            );
        });

    properties::draw_light_linking_section(ui, scene, object.host_id, actions, active_light_ids);

    if let Some(node) = scene.nodes.get_mut(&object.host_id) {
        if let NodeData::Primitive {
            kind: node_kind,
            position: node_position,
            rotation: node_rotation,
            scale: node_scale,
            material: node_material,
            ..
        } = &mut node.data
        {
            *node_kind = kind;
            *node_position = position;
            *node_rotation = rotation;
            *node_scale = scale;
            *node_material = material;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_operation_host(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    mut op: CsgOp,
    mut smooth_k: f32,
    mut steps: f32,
    mut color_blend: f32,
    left: Option<NodeId>,
    right: Option<NodeId>,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
) {
    egui::ComboBox::from_id_salt(("presented_operation_type", object.host_id))
        .selected_text(op.base_name())
        .show_ui(ui, |ui| {
            for operation in CsgOp::ALL {
                if ui
                    .selectable_label(op == *operation, operation.base_name())
                    .clicked()
                {
                    op = operation.clone();
                    smooth_k = op.default_smooth_k();
                    steps = op.default_steps();
                    color_blend = -1.0;
                }
            }
        });

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
    ui.label(format!(
        "Left: {}",
        left.and_then(|id| scene.nodes.get(&id))
            .map(|node| node.name.clone())
            .unwrap_or_else(|| "(empty)".to_string())
    ));
    ui.label(format!(
        "Right: {}",
        right.and_then(|id| scene.nodes.get(&id))
            .map(|node| node.name.clone())
            .unwrap_or_else(|| "(empty)".to_string())
    ));
    if left.is_some() && right.is_some() && ui.button("Swap Inputs").clicked() {
        actions.push(Action::SwapChildren(object.host_id));
    }

    properties::draw_light_linking_section(ui, scene, object.host_id, actions, active_light_ids);

    if let Some(node) = scene.nodes.get_mut(&object.host_id) {
        if let NodeData::Operation {
            op: node_op,
            smooth_k: node_smooth_k,
            steps: node_steps,
            color_blend: node_color_blend,
            ..
        } = &mut node.data
        {
            *node_op = op;
            *node_smooth_k = smooth_k;
            *node_steps = steps;
            *node_color_blend = color_blend;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_voxel_host(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    sculpt_state: &mut SculptState,
    mut position: Vec3,
    mut rotation: Vec3,
    mut material: crate::graph::scene::MaterialParams,
    mut layer_intensity: f32,
    voxel_grid: crate::graph::voxel::VoxelGrid,
    mut desired_resolution: u32,
    actions: &mut ActionSink,
    active_light_ids: &HashSet<NodeId>,
    max_sculpt_resolution: u32,
    material_library: &mut MaterialLibrary,
) {
    ui.label("Type: Voxel Object");
    ui.separator();

    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            properties::vec3_editor(ui, "Position", &mut position, 0.05, None, "");
            let mut rotation_deg = Vec3::new(
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            );
            properties::vec3_editor(ui, "Rotation", &mut rotation_deg, 1.0, None, " deg");
            rotation = Vec3::new(
                rotation_deg.x.to_radians(),
                rotation_deg.y.to_radians(),
                rotation_deg.z.to_radians(),
            );
        });

    egui::CollapsingHeader::new("Material")
        .default_open(true)
        .show(ui, |ui| {
            properties::draw_material_editor(ui, "presented_voxel", &mut material, material_library);
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
            ui.horizontal(|ui| {
                ui.label("Layer Intensity:");
                ui.add(egui::Slider::new(&mut layer_intensity, 0.0..=1.0).fixed_decimals(2));
            });

            let detail_state = sculpt_state.detail_state();
            if sculpt_state.active_node() == Some(object.host_id) {
                if let Some(previous_detail) = detail_state.last_pre_expand_detail_size {
                    let coarsening = (detail_size / previous_detail).max(1.0);
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        format!(
                            "Volume expansion made detail coarser. Previous {:.4}, current {:.4} ({:.2}x larger voxels).",
                            previous_detail, detail_size, coarsening
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

            ui.horizontal_wrapped(|ui| {
                if ui.button("Increase Detail").clicked() {
                    actions.push(Action::IncreaseSculptDetail(object.host_id));
                }
                if ui.button("Decrease Detail").clicked() {
                    actions.push(Action::DecreaseSculptDetail(object.host_id));
                }
                if ui.button("Remesh at Current Detail").clicked() {
                    actions.push(Action::RemeshSculptAtCurrentDetail(object.host_id));
                }
                if ui.button("Expand Volume").clicked() {
                    actions.push(Action::ExpandSculptVolume(object.host_id));
                }
                if ui.button("Fit Volume to Sculpt").clicked() {
                    actions.push(Action::FitSculptVolume(object.host_id));
                }
            });

            let max_resolution = max_sculpt_resolution.max(16);
            ui.horizontal(|ui| {
                ui.label("Manual Resolution:");
                ui.add(
                    egui::DragValue::new(&mut desired_resolution)
                        .range(16..=max_resolution)
                        .speed(1.0),
                );
            });
            ui.small(format!(
                "Voxel count: {}",
                properties::format_voxel_count((desired_resolution as u64).pow(3))
            ));
        });

    properties::draw_light_linking_section(ui, scene, object.host_id, actions, active_light_ids);

    if let Some(node) = scene.nodes.get_mut(&object.host_id) {
        if let NodeData::Sculpt {
            position: node_position,
            rotation: node_rotation,
            material: node_material,
            layer_intensity: node_layer_intensity,
            desired_resolution: node_desired_resolution,
            ..
        } = &mut node.data
        {
            *node_position = position;
            *node_rotation = rotation;
            *node_material = material;
            *node_layer_intensity = layer_intensity;
            *node_desired_resolution = desired_resolution;
        }
    }
}

fn draw_host_actions(
    ui: &mut egui::Ui,
    scene: &Scene,
    object: PresentedObjectRef,
    bake_progress: Option<(u32, u32)>,
    actions: &mut ActionSink,
) {
    ui.label(egui::RichText::new("Object Actions").small().strong());
    if let Some((done, total)) = bake_progress {
        let progress = done as f32 / total.max(1) as f32;
        ui.add(egui::ProgressBar::new(progress).text(format!("Baking... {:.0}%", progress * 100.0)));
        return;
    }

    ui.horizontal_wrapped(|ui| {
        if matches!(object.kind, PresentedObjectKind::Voxel) {
            if ui.button("Enter Sculpt").clicked() {
                actions.push(Action::EnterSculptMode);
            }
        } else if let Some(sculpt_id) = object.attached_sculpt_id {
            if ui.button("Enter Sculpt").clicked() {
                actions.push(Action::EnterSculptMode);
            }
            if ui.button("Convert to Voxel Object").clicked() {
                actions.push(Action::RequestBake(BakeRequest {
                    subtree_root: object.object_root_id,
                    resolution: voxel::max_subtree_resolution(scene, object.object_root_id),
                    color: sculpt_color_for_object(scene, object),
                    existing_sculpt: Some(sculpt_id),
                    flatten: true,
                }));
            }
            if ui.button("Remove Sculpt Layer").clicked() {
                actions.push(Action::RemoveAttachedSculpt {
                    host: object.host_id,
                });
            }
        } else if object.supports_add_sculpt() && ui.button("Add Sculpt Layer").clicked() {
            actions.push(Action::RequestBake(BakeRequest {
                subtree_root: object.object_root_id,
                resolution: voxel::DEFAULT_RESOLUTION,
                color: sculpt_color_for_object(scene, object),
                existing_sculpt: None,
                flatten: false,
            }));
        }

        if ui
            .button("Delete Object")
            .on_hover_text("Remove this object and its hidden internal wrapper nodes")
            .clicked()
        {
            actions.push(Action::DeletePresentedObject(object.object_root_id));
        }
    });
}

fn draw_object_stack(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    sculpt_state: &mut SculptState,
    actions: &mut ActionSink,
    bake_progress: Option<(u32, u32)>,
) {
    let wrappers = collect_presented_wrapper_chain(scene, object);
    egui::CollapsingHeader::new("Object Stack")
        .default_open(true)
        .show(ui, |ui| {
            if wrappers.is_empty() {
                ui.weak("No transforms, modifiers, or sculpt attachment on this object.");
            }

            for wrapper_id in wrappers {
                let Some(wrapper_node) = scene.nodes.get(&wrapper_id) else {
                    continue;
                };
                let wrapper_name = wrapper_node.name.clone();
                let wrapper_locked = wrapper_node.locked;
                let wrapper_data = wrapper_node.data.clone();

                match wrapper_data {
                    NodeData::Transform {
                        mut translation,
                        mut rotation,
                        mut scale,
                        ..
                    } => {
                        egui::CollapsingHeader::new(format!("[Xfm] {wrapper_name}"))
                            .default_open(false)
                            .id_salt(("presented_transform_wrapper", wrapper_id))
                            .show(ui, |ui| {
                                if wrapper_locked {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 180, 80),
                                        "Locked",
                                    );
                                }
                                properties::vec3_editor(
                                    ui,
                                    "Translate",
                                    &mut translation,
                                    0.05,
                                    None,
                                    "",
                                );
                                let mut rotation_deg = Vec3::new(
                                    rotation.x.to_degrees(),
                                    rotation.y.to_degrees(),
                                    rotation.z.to_degrees(),
                                );
                                properties::vec3_editor(
                                    ui,
                                    "Rotate",
                                    &mut rotation_deg,
                                    1.0,
                                    None,
                                    " deg",
                                );
                                rotation = Vec3::new(
                                    rotation_deg.x.to_radians(),
                                    rotation_deg.y.to_radians(),
                                    rotation_deg.z.to_radians(),
                                );
                                properties::vec3_editor(ui, "Scale", &mut scale, 0.05, None, "");
                                if ui
                                    .add_enabled(!wrapper_locked, egui::Button::new("Remove Transform"))
                                    .clicked()
                                {
                                    actions.push(Action::RemoveWrapperNode(wrapper_id));
                                }
                            });

                        if let Some(node) = scene.nodes.get_mut(&wrapper_id) {
                            if let NodeData::Transform {
                                translation: node_translation,
                                rotation: node_rotation,
                                scale: node_scale,
                                ..
                            } = &mut node.data
                            {
                                *node_translation = translation;
                                *node_rotation = rotation;
                                *node_scale = scale;
                            }
                        }
                    }
                    NodeData::Modifier {
                        kind,
                        mut value,
                        mut extra,
                        ..
                    } => {
                        egui::CollapsingHeader::new(format!("{} {wrapper_name}", kind.badge()))
                            .default_open(false)
                            .id_salt(("presented_modifier_wrapper", wrapper_id))
                            .show(ui, |ui| {
                                if wrapper_locked {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(255, 180, 80),
                                        "Locked",
                                    );
                                }
                                ui.small(kind.base_name());
                                properties::vec3_editor(ui, "Value", &mut value, 0.05, None, "");
                                properties::vec3_editor(ui, "Extra", &mut extra, 0.05, None, "");
                                if ui
                                    .add_enabled(!wrapper_locked, egui::Button::new("Remove Modifier"))
                                    .clicked()
                                {
                                    actions.push(Action::RemoveWrapperNode(wrapper_id));
                                }
                            });

                        if let Some(node) = scene.nodes.get_mut(&wrapper_id) {
                            if let NodeData::Modifier {
                                value: node_value,
                                extra: node_extra,
                                ..
                            } = &mut node.data
                            {
                                *node_value = value;
                                *node_extra = extra;
                            }
                        }
                    }
                    NodeData::Sculpt {
                        mut layer_intensity,
                        voxel_grid,
                        desired_resolution,
                        ..
                    } => {
                        egui::CollapsingHeader::new("[Sculpt] Sculpt Layer")
                            .default_open(true)
                            .id_salt(("presented_attached_sculpt", wrapper_id))
                            .show(ui, |ui| {
                                let is_active = sculpt_state.active_node() == Some(wrapper_id);
                                ui.small(if is_active {
                                    "This sculpt layer is currently active."
                                } else {
                                    "This sculpt layer is attached to the host object."
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Layer Intensity:");
                                    ui.add(
                                        egui::Slider::new(&mut layer_intensity, 0.0..=1.0)
                                            .fixed_decimals(2),
                                    );
                                });
                                ui.small(format!(
                                    "Resolution: {}^3 | Detail Size: {:.4}",
                                    desired_resolution,
                                    voxel_grid.voxel_pitch()
                                ));
                                if let Some((done, total)) = bake_progress {
                                    let progress = done as f32 / total.max(1) as f32;
                                    ui.add(
                                        egui::ProgressBar::new(progress)
                                            .text(format!("Baking... {:.0}%", progress * 100.0)),
                                    );
                                } else {
                                    ui.horizontal_wrapped(|ui| {
                                        if ui.button("Enter Sculpt").clicked() {
                                            actions.push(Action::EnterSculptMode);
                                        }
                                        if ui.button("Convert to Voxel Object").clicked() {
                                            actions.push(Action::RequestBake(BakeRequest {
                                                subtree_root: object.object_root_id,
                                                resolution: voxel::max_subtree_resolution(
                                                    scene,
                                                    object.object_root_id,
                                                ),
                                                color: sculpt_color_for_object(scene, object),
                                                existing_sculpt: Some(wrapper_id),
                                                flatten: true,
                                            }));
                                        }
                                        if ui.button("Remove Sculpt Layer").clicked() {
                                            actions.push(Action::RemoveAttachedSculpt {
                                                host: object.host_id,
                                            });
                                        }
                                    });
                                }
                                ui.separator();
                                ui.small("Internal sculpt transform and material stay hidden in the default object inspector.");
                            });

                        if let Some(node) = scene.nodes.get_mut(&wrapper_id) {
                            if let NodeData::Sculpt {
                                layer_intensity: node_layer_intensity,
                                ..
                            } = &mut node.data
                            {
                                *node_layer_intensity = layer_intensity;
                            }
                        }
                    }
                    _ => {}
                }
            }

            ui.separator();
            ui.menu_button("+ Add", |ui| {
                if ui.button("Transform").clicked() {
                    actions.push(Action::InsertTransformAbove {
                        target: object.object_root_id,
                    });
                    ui.close_menu();
                }
                ui.menu_button("Modifier", |ui| {
                    for modifier in ModifierKind::ALL {
                        if ui.button(modifier.base_name()).clicked() {
                            actions.push(Action::InsertModifierAbove {
                                target: object.object_root_id,
                                kind: modifier.clone(),
                            });
                            ui.close_menu();
                        }
                    }
                });
                if object.supports_add_sculpt() && ui.button("Sculpt Layer").clicked() {
                    actions.push(Action::RequestBake(BakeRequest {
                        subtree_root: object.object_root_id,
                        resolution: voxel::DEFAULT_RESOLUTION,
                        color: sculpt_color_for_object(scene, object),
                        existing_sculpt: None,
                        flatten: false,
                    }));
                    ui.close_menu();
                }
            });
        });
}

fn sculpt_color_for_object(scene: &Scene, object: PresentedObjectRef) -> Vec3 {
    scene
        .nodes
        .get(&object.host_id)
        .and_then(|node| node.data.material())
        .map(|material| material.base_color)
        .unwrap_or(Vec3::splat(0.6))
}
