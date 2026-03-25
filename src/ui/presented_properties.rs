use std::collections::HashSet;

use glam::Vec3;

use crate::app::actions::{Action, ActionSink, OperationInputSlot};
use crate::app::state::MultiTransformSessionState;
use crate::graph::presented_object::{
    collect_presented_base_wrapper_chain, collect_presented_selection, object_transform_wrapper,
    resolve_presented_object, PresentedObjectKind, PresentedObjectRef,
};
use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};
use crate::material_preset::MaterialLibrary;
use crate::sculpt::SculptState;
use crate::settings::SelectionBehaviorSettings;
use crate::ui::gizmo::GizmoSpace;
use crate::ui::presented_object_actions;
use crate::ui::{chips, chrome};

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
        chrome::card_frame().show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label("No selection");
                ui.add_space(6.0);
                ui.weak("Click an object in the viewport or scene panel.");
                ui.add_space(12.0);
            });
        });
        return;
    };

    if matches!(object.kind, PresentedObjectKind::Light) {
        let light_selected_set = HashSet::from([object.host_id]);
        show_property_card(ui, "presented_light_host", "Light", true, None, |ui| {
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
        });
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
    _actions: &mut ActionSink,
) {
    show_property_card(
        ui,
        "presented_multi_selection",
        "Selection",
        true,
        Some("Use the viewport gizmo to move, rotate, or scale the selected objects together."),
        |ui| {
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
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
        },
    );
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

    draw_identity_card(ui, &mut name, is_locked, object.host_id);

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
            show_property_card(
                ui,
                ("presented_unsupported", object.host_id),
                "Unsupported",
                true,
                None,
                |ui| {
                    ui.weak(
                        "This object type is only editable in the advanced raw-node inspector.",
                    );
                },
            );
        }
    }

    draw_object_stack(
        ui,
        scene,
        object,
        sculpt_state,
        actions,
        bake_progress,
        object.attached_sculpt_id.is_some(),
    );

    if let Some(node) = scene.nodes.get_mut(&object.host_id) {
        node.name = name;
    }
}

fn draw_identity_card(ui: &mut egui::Ui, name: &mut String, is_locked: bool, host_id: NodeId) {
    show_property_card(
        ui,
        ("presented_identity", host_id),
        "Identity",
        true,
        None,
        |ui| {
            if is_locked {
                chips::draw_chip(
                    ui,
                    "LOCKED",
                    egui::Color32::from_rgb(80, 56, 22),
                    egui::Color32::from_rgb(255, 214, 148),
                );
                ui.add_space(6.0);
            }

            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.add_enabled(!is_locked, egui::TextEdit::singleline(name));
            });
        },
    );
}

fn show_property_card(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    title: &str,
    default_open: bool,
    subtitle: Option<&str>,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    chrome::card_frame().show(ui, |ui| {
        egui::CollapsingHeader::new(title)
            .default_open(default_open)
            .id_salt(id_source)
            .show(ui, |ui| {
                if let Some(subtitle) = subtitle {
                    ui.small(subtitle);
                    ui.add_space(6.0);
                }
                add_contents(ui);
            });
    });
    ui.add_space(8.0);
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
    show_property_card(
        ui,
        ("presented_primitive_core", object.host_id),
        "Core",
        true,
        Some(if has_sculpt_layer {
            "Object motion belongs to the outer transform. These controls reshape the analytical base."
        } else {
            "Edit the primitive type and its primary shape controls."
        }),
        |ui| {
            egui::ComboBox::from_id_salt(("presented_primitive_type", object.host_id))
                .selected_text(kind.base_name())
                .show_ui(ui, |ui| {
                    for primitive in SdfPrimitive::ALL {
                        ui.selectable_value(&mut kind, primitive.clone(), primitive.base_name());
                    }
                });
            ui.add_space(6.0);

            if !has_sculpt_layer {
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
                ui.label(if has_sculpt_layer {
                    "Base Shape"
                } else {
                    "Size"
                });
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
        },
    );

    show_property_card(
        ui,
        ("presented_primitive_material", object.host_id),
        "Material",
        false,
        None,
        |ui| {
            properties::draw_material_editor(
                ui,
                "presented_primitive",
                &mut material,
                material_library,
            );
        },
    );

    show_property_card(
        ui,
        ("presented_primitive_lighting", object.host_id),
        "Lighting",
        false,
        None,
        |ui| {
            properties::draw_light_linking_section(
                ui,
                scene,
                object.host_id,
                actions,
                active_light_ids,
            )
        },
    );

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
    show_property_card(
        ui,
        ("presented_operation_core", object.host_id),
        "Operation",
        true,
        None,
        |ui| {
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

            ui.add_space(6.0);
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
        },
    );

    show_property_card(
        ui,
        ("presented_operation_inputs", object.host_id),
        "Inputs",
        true,
        Some("Treat each operand as an object. Select it, swap it, or replace it inline."),
        |ui| {
            draw_operation_input_row(
                ui,
                scene,
                object.host_id,
                OperationInputSlot::Left,
                left,
                actions,
            );
            draw_operation_input_row(
                ui,
                scene,
                object.host_id,
                OperationInputSlot::Right,
                right,
                actions,
            );
            if left.is_some() && right.is_some() && ui.button("Swap Inputs").clicked() {
                actions.push(Action::SwapChildren(object.host_id));
            }
        },
    );

    show_property_card(
        ui,
        ("presented_operation_lighting", object.host_id),
        "Lighting",
        false,
        None,
        |ui| {
            properties::draw_light_linking_section(
                ui,
                scene,
                object.host_id,
                actions,
                active_light_ids,
            )
        },
    );

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
    show_property_card(
        ui,
        ("presented_voxel_transform", object.host_id),
        "Transform",
        true,
        Some("Standalone voxel objects keep their own transform and sculpt state."),
        |ui| {
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
        },
    );

    show_property_card(
        ui,
        ("presented_voxel_sculpting", object.host_id),
        "Sculpt",
        true,
        None,
        |ui| {
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
        },
    );

    show_property_card(
        ui,
        ("presented_voxel_material", object.host_id),
        "Material",
        false,
        None,
        |ui| {
            properties::draw_material_editor(
                ui,
                "presented_voxel",
                &mut material,
                material_library,
            );
        },
    );

    show_property_card(
        ui,
        ("presented_voxel_lighting", object.host_id),
        "Lighting",
        false,
        None,
        |ui| {
            properties::draw_light_linking_section(
                ui,
                scene,
                object.host_id,
                actions,
                active_light_ids,
            )
        },
    );

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

fn draw_object_stack(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    sculpt_state: &mut SculptState,
    actions: &mut ActionSink,
    bake_progress: Option<(u32, u32)>,
    default_open: bool,
) {
    let object_transform_id = object_transform_wrapper(scene, object.host_id);
    let base_wrappers = collect_presented_base_wrapper_chain(scene, object);
    let has_entries = object_transform_id.is_some()
        || !base_wrappers.is_empty()
        || object.attached_sculpt_id.is_some();

    show_property_card(
        ui,
        ("presented_object_layers", object.host_id),
        "Object Layers",
        default_open,
        Some("Move the object, shape the base, then sculpt on top."),
        |ui| {
            ui.horizontal(|ui| {
                chrome::section_title(ui, "Object Model", None);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    presented_object_actions::draw_host_add_menu_button(
                        ui, scene, object, actions, "+ Add",
                    );
                });
            });
            ui.add_space(6.0);

            if !has_entries {
                ui.weak("No transforms, modifiers, or sculpt attachment on this object yet.");
                return;
            }

            if let Some(wrapper_id) = object_transform_id {
                draw_transform_wrapper_row(
                    ui,
                    scene,
                    wrapper_id,
                    "Object Transform".to_string(),
                    true,
                    false,
                    Some(
                        "Primary transform owner for this sculpted object. Move, rotate, and scale the object here.",
                    ),
                    actions,
                );
                ui.add_space(6.0);
            }

            if object.attached_sculpt_id.is_some() && !base_wrappers.is_empty() {
                chrome::section_title(
                    ui,
                    "Base Shape Stack",
                    Some("These transforms and modifiers reshape the analytical base under the sculpt layer."),
                );
                ui.add_space(6.0);
            }

            for (index, &wrapper_id) in base_wrappers.iter().enumerate() {
                let Some(wrapper_node) = scene.nodes.get(&wrapper_id) else {
                    continue;
                };
                let wrapper_name = wrapper_node.name.clone();
                let wrapper_data = wrapper_node.data.clone();

                match wrapper_data {
                    NodeData::Transform { .. } => {
                        let header = if object.attached_sculpt_id.is_some() {
                            format!("[Base Xfm] {wrapper_name}")
                        } else {
                            format!("[Xfm] {wrapper_name}")
                        };
                        draw_transform_wrapper_row(
                            ui, scene, wrapper_id, header, false, true, None, actions,
                        );
                    }
                    NodeData::Modifier { .. } => {
                        draw_modifier_wrapper_row(ui, scene, wrapper_id, wrapper_name, actions);
                    }
                    _ => {}
                }

                if index + 1 != base_wrappers.len() {
                    ui.add_space(6.0);
                }
            }

            if let Some(sculpt_id) = object.attached_sculpt_id {
                if object_transform_id.is_some() || !base_wrappers.is_empty() {
                    ui.add_space(6.0);
                }
                draw_attached_sculpt_row(
                    ui,
                    scene,
                    object,
                    sculpt_id,
                    sculpt_state,
                    actions,
                    bake_progress,
                );
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_transform_wrapper_row(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    wrapper_id: NodeId,
    header: String,
    default_open: bool,
    allow_remove: bool,
    description: Option<&str>,
    actions: &mut ActionSink,
) {
    let Some(wrapper_node) = scene.nodes.get(&wrapper_id) else {
        return;
    };
    let wrapper_locked = wrapper_node.locked;
    let NodeData::Transform {
        mut translation,
        mut rotation,
        mut scale,
        ..
    } = wrapper_node.data.clone()
    else {
        return;
    };

    egui::CollapsingHeader::new(header)
        .default_open(default_open)
        .id_salt(("presented_transform_wrapper", wrapper_id))
        .show(ui, |ui| {
            if wrapper_locked {
                ui.colored_label(egui::Color32::from_rgb(255, 180, 80), "Locked");
            }
            if let Some(text) = description {
                ui.small(text);
            }
            properties::vec3_editor(ui, "Translate", &mut translation, 0.05, None, "");
            let mut rotation_deg = Vec3::new(
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            );
            properties::vec3_editor(ui, "Rotate", &mut rotation_deg, 1.0, None, " deg");
            rotation = Vec3::new(
                rotation_deg.x.to_radians(),
                rotation_deg.y.to_radians(),
                rotation_deg.z.to_radians(),
            );
            properties::vec3_editor(ui, "Scale", &mut scale, 0.05, None, "");
            if allow_remove
                && ui
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

fn draw_operation_input_row(
    ui: &mut egui::Ui,
    scene: &Scene,
    operation: NodeId,
    slot: OperationInputSlot,
    child: Option<NodeId>,
    actions: &mut ActionSink,
) {
    let slot_label = match slot {
        OperationInputSlot::Left => "Left Operand",
        OperationInputSlot::Right => "Right Operand",
    };
    let child_object = child.and_then(|child_id| resolve_presented_object(scene, child_id));

    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(slot_label).small().strong());
        if let Some(object) = child_object {
            let object_name = scene
                .nodes
                .get(&object.host_id)
                .map(|node| node.name.as_str())
                .unwrap_or("Missing Object");
            ui.label(object_name);
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
            if ui.small_button("Select").clicked() {
                actions.push(Action::Select(Some(object.host_id)));
            }
        } else {
            ui.weak("(empty)");
        }

        ui.menu_button("Replace", |ui| {
            for primitive in SdfPrimitive::ALL {
                if ui.button(primitive.base_name()).clicked() {
                    actions.push(Action::ReplaceOperationInputWithPrimitive {
                        operation,
                        slot,
                        primitive: primitive.clone(),
                    });
                    ui.close_menu();
                }
            }
        });
    });
}

fn draw_modifier_wrapper_row(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    wrapper_id: NodeId,
    wrapper_name: String,
    actions: &mut ActionSink,
) {
    let Some(wrapper_node) = scene.nodes.get(&wrapper_id) else {
        return;
    };
    let wrapper_locked = wrapper_node.locked;
    let NodeData::Modifier {
        kind,
        mut value,
        mut extra,
        ..
    } = wrapper_node.data.clone()
    else {
        return;
    };

    egui::CollapsingHeader::new(format!("{} {wrapper_name}", kind.badge()))
        .default_open(false)
        .id_salt(("presented_modifier_wrapper", wrapper_id))
        .show(ui, |ui| {
            if wrapper_locked {
                ui.colored_label(egui::Color32::from_rgb(255, 180, 80), "Locked");
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

fn draw_attached_sculpt_row(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    object: PresentedObjectRef,
    sculpt_id: NodeId,
    sculpt_state: &mut SculptState,
    actions: &mut ActionSink,
    bake_progress: Option<(u32, u32)>,
) {
    let Some(sculpt_node) = scene.nodes.get(&sculpt_id) else {
        return;
    };
    let NodeData::Sculpt {
        mut layer_intensity,
        voxel_grid,
        desired_resolution,
        ..
    } = sculpt_node.data.clone()
    else {
        return;
    };

    egui::CollapsingHeader::new("Sculpt Layer")
        .default_open(true)
        .id_salt(("presented_attached_sculpt", sculpt_id))
        .show(ui, |ui| {
            let is_active = sculpt_state.active_node() == Some(sculpt_id);
            ui.horizontal(|ui| {
                chips::draw_chip(
                    ui,
                    if is_active { "ACTIVE" } else { "ATTACHED" },
                    if is_active {
                        egui::Color32::from_rgb(32, 92, 52)
                    } else {
                        egui::Color32::from_rgb(98, 67, 24)
                    },
                    if is_active {
                        egui::Color32::from_rgb(198, 255, 208)
                    } else {
                        egui::Color32::from_rgb(255, 220, 140)
                    },
                );
                ui.small(if is_active {
                    "Brushes now affect this layer."
                } else {
                    "This sculpt layer stays attached to the object."
                });
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Layer Intensity:");
                ui.add(egui::Slider::new(&mut layer_intensity, 0.0..=1.0).fixed_decimals(2));
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
                        presented_object_actions::push_convert_to_voxel_action(
                            scene, object, sculpt_id, actions,
                        );
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

    if let Some(node) = scene.nodes.get_mut(&sculpt_id) {
        if let NodeData::Sculpt {
            layer_intensity: node_layer_intensity,
            ..
        } = &mut node.data
        {
            *node_layer_intensity = layer_intensity;
        }
    }
}
