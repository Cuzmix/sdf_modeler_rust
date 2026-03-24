use eframe::egui;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::app::BakeRequest;
use crate::graph::presented_object::{presented_wrap_target, PresentedObjectKind, PresentedObjectRef};
use crate::graph::scene::{CsgOp, ModifierKind, NodeId, Scene, SdfPrimitive};
use crate::graph::voxel;

pub fn draw_host_add_menu_button(
    ui: &mut egui::Ui,
    scene: &Scene,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
    label: &str,
) {
    ui.menu_button(label, |ui| draw_host_add_menu_contents(ui, scene, object, actions));
}

pub fn draw_host_add_menu_contents(
    ui: &mut egui::Ui,
    scene: &Scene,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
) {
    if let Some(target_id) = presented_wrap_target(scene, object.host_id) {
        if ui.button("Add Transform").clicked() {
            actions.push(Action::InsertTransformAbove { target: target_id });
            ui.close_menu();
        }
        ui.menu_button("Add Modifier", |ui| {
            for modifier in ModifierKind::ALL {
                if ui.button(modifier.base_name()).clicked() {
                    actions.push(Action::InsertModifierAbove {
                        target: target_id,
                        kind: modifier.clone(),
                    });
                    ui.close_menu();
                }
            }
        });
    }

    if object.supports_add_sculpt() && ui.button("Add Sculpt Layer").clicked() {
        push_add_sculpt_layer_action(scene, object, actions);
        ui.close_menu();
    }
}

pub fn draw_object_create_menu_button(
    ui: &mut egui::Ui,
    scene: &Scene,
    selected_object: Option<PresentedObjectRef>,
    actions: &mut ActionSink,
    label: &str,
) {
    ui.menu_button(label, |ui| draw_object_create_menu_contents(ui, scene, selected_object, actions));
}

pub fn draw_object_create_menu_contents(
    ui: &mut egui::Ui,
    scene: &Scene,
    selected_object: Option<PresentedObjectRef>,
    actions: &mut ActionSink,
) {
    draw_object_create_actions(ui, scene, selected_object, actions, true);
}

pub fn draw_object_create_actions(
    ui: &mut egui::Ui,
    scene: &Scene,
    selected_object: Option<PresentedObjectRef>,
    actions: &mut ActionSink,
    close_after_action: bool,
) {
    ui.horizontal_wrapped(|ui| {
        ui.menu_button("New Primitive", |ui| {
            for primitive in SdfPrimitive::ALL {
                if ui.button(primitive.base_name()).clicked() {
                    actions.push(Action::CreatePrimitive(primitive.clone()));
                    if close_after_action {
                        ui.close_menu();
                    }
                }
            }
        });

        if let Some(object) = selected_object {
            if ui.button("Duplicate Selected").clicked() {
                actions.push(Action::DuplicatePresentedObject(object.host_id));
                if close_after_action {
                    ui.close_menu();
                }
            }

            if !matches!(object.kind, PresentedObjectKind::Light) {
                ui.menu_button("Boolean From Selection", |ui| {
                    for op in CsgOp::ALL {
                        ui.menu_button(op.base_name(), |ui| {
                            for primitive in SdfPrimitive::ALL {
                                if ui.button(primitive.base_name()).clicked() {
                                    actions.push(Action::ShellCreateBooleanPrimitive {
                                        op: op.clone(),
                                        primitive: primitive.clone(),
                                    });
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                });

                if object.supports_add_sculpt() && ui.button("Add Sculpt Layer").clicked() {
                    push_add_sculpt_layer_action(scene, object, actions);
                    if close_after_action {
                        ui.close_menu();
                    }
                } else if let Some(sculpt_id) = object.attached_sculpt_id {
                    if ui.button("Enter Sculpt").clicked() {
                        actions.push(Action::EnterSculptMode);
                        if close_after_action {
                            ui.close_menu();
                        }
                    }
                    if ui.button("Convert to Voxel Object").clicked() {
                        push_convert_to_voxel_action(scene, object, sculpt_id, actions);
                        if close_after_action {
                            ui.close_menu();
                        }
                    }
                }
            }
        }
    });
}

pub fn push_add_sculpt_layer_action(
    scene: &Scene,
    object: PresentedObjectRef,
    actions: &mut ActionSink,
) {
    actions.push(Action::RequestBake(BakeRequest {
        subtree_root: object.object_root_id,
        resolution: voxel::DEFAULT_RESOLUTION,
        color: sculpt_color_for_object(scene, object),
        existing_sculpt: None,
        flatten: false,
    }));
}

pub fn push_convert_to_voxel_action(
    scene: &Scene,
    object: PresentedObjectRef,
    sculpt_id: NodeId,
    actions: &mut ActionSink,
) {
    actions.push(Action::RequestBake(BakeRequest {
        subtree_root: object.object_root_id,
        resolution: voxel::max_subtree_resolution(scene, object.object_root_id),
        color: sculpt_color_for_object(scene, object),
        existing_sculpt: Some(sculpt_id),
        flatten: true,
    }));
}

pub fn sculpt_color_for_object(scene: &Scene, object: PresentedObjectRef) -> Vec3 {
    scene
        .nodes
        .get(&object.host_id)
        .and_then(|node| node.data.material())
        .map(|material| material.base_color)
        .unwrap_or(Vec3::splat(0.6))
}
