// src/app_slint_sync.rs — Helpers to convert Rust state into Slint-compatible data
//
// These functions produce intermediate Rust structs that app_slint.rs converts
// to Slint types via the generated bindings.

use std::collections::HashSet;

use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};
use crate::settings::RenderConfig;

// ---------------------------------------------------------------------------
// Scene tree
// ---------------------------------------------------------------------------

pub struct SceneNodeInfoData {
    pub id: i32,
    pub name: String,
    pub node_type: String,
    pub badge: String,
    pub depth: i32,
    pub is_selected: bool,
    pub is_hidden: bool,
    pub has_children: bool,
}

pub fn build_scene_tree_flat(scene: &Scene, selected: Option<NodeId>) -> Vec<SceneNodeInfoData> {
    let mut result = Vec::new();
    let tops = scene.top_level_nodes();
    let mut visited = HashSet::new();
    for id in tops {
        flatten_tree_recursive(scene, id, selected, 0, &mut result, &mut visited);
    }
    result
}

fn flatten_tree_recursive(
    scene: &Scene,
    id: NodeId,
    selected: Option<NodeId>,
    depth: i32,
    out: &mut Vec<SceneNodeInfoData>,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }
    let Some(node) = scene.nodes.get(&id) else {
        return;
    };

    let (node_type, badge) = match &node.data {
        NodeData::Primitive { kind, .. } => ("Primitive".to_string(), kind.badge().to_string()),
        NodeData::Operation { op, .. } => ("Operation".to_string(), op.badge().to_string()),
        NodeData::Transform { kind, .. } => ("Transform".to_string(), kind.badge().to_string()),
        NodeData::Modifier { kind, .. } => ("Modifier".to_string(), kind.badge().to_string()),
        NodeData::Sculpt { .. } => ("Sculpt".to_string(), "[Scu]".to_string()),
    };

    let children: Vec<NodeId> = node.data.children().collect();
    let has_children = !children.is_empty();

    out.push(SceneNodeInfoData {
        id: id as i32,
        name: node.name.clone(),
        node_type,
        badge,
        depth,
        is_selected: selected == Some(id),
        is_hidden: scene.is_hidden(id),
        has_children,
    });

    for child_id in children {
        flatten_tree_recursive(scene, child_id, selected, depth + 1, out, visited);
    }
}

// ---------------------------------------------------------------------------
// Selected node properties
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct SelectedNodePropsData {
    pub id: i32,
    pub name: String,
    pub node_type: i32, // 0=none, 1=primitive, 2=operation, 3=transform, 4=modifier, 5=sculpt

    pub primitive_kind: i32,
    pub csg_op: i32,
    pub smooth_k: f32,
    pub left_name: String,
    pub right_name: String,
    pub has_left: bool,
    pub has_right: bool,
    pub transform_kind: i32,
    pub modifier_kind: i32,

    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub rot_x: f32,
    pub rot_y: f32,
    pub rot_z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,

    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub roughness: f32,
    pub metallic: f32,
    pub fresnel: f32,
    pub emissive_r: f32,
    pub emissive_g: f32,
    pub emissive_b: f32,
    pub emissive_intensity: f32,

    pub value_x: f32,
    pub value_y: f32,
    pub value_z: f32,
    pub extra_x: f32,
    pub extra_y: f32,
    pub extra_z: f32,

    pub scale_label_x: String,
    pub scale_label_y: String,
    pub scale_label_z: String,
    pub scale_count: i32,
}

pub fn build_selected_props(scene: &Scene, selected: Option<NodeId>) -> SelectedNodePropsData {
    let Some(id) = selected else {
        return SelectedNodePropsData::default();
    };
    let Some(node) = scene.nodes.get(&id) else {
        return SelectedNodePropsData::default();
    };

    let mut p = SelectedNodePropsData::default();
    p.id = id as i32;
    p.name = node.name.clone();

    match &node.data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            color,
            roughness,
            metallic,
            emissive,
            emissive_intensity,
            fresnel,
            ..
        } => {
            p.node_type = 1;
            p.primitive_kind =
                SdfPrimitive::ALL.iter().position(|v| v == kind).unwrap_or(0) as i32;
            p.pos_x = position.x;
            p.pos_y = position.y;
            p.pos_z = position.z;
            p.rot_x = rotation.x.to_degrees();
            p.rot_y = rotation.y.to_degrees();
            p.rot_z = rotation.z.to_degrees();
            p.scale_x = scale.x;
            p.scale_y = scale.y;
            p.scale_z = scale.z;
            p.color_r = color.x;
            p.color_g = color.y;
            p.color_b = color.z;
            p.roughness = *roughness;
            p.metallic = *metallic;
            p.fresnel = *fresnel;
            p.emissive_r = emissive.x;
            p.emissive_g = emissive.y;
            p.emissive_b = emissive.z;
            p.emissive_intensity = *emissive_intensity;

            let params = kind.scale_params();
            p.scale_count = params.len() as i32;
            if !params.is_empty() {
                p.scale_label_x = params[0].0.to_string();
            }
            if params.len() > 1 {
                p.scale_label_y = params[1].0.to_string();
            }
            if params.len() > 2 {
                p.scale_label_z = params[2].0.to_string();
            }
        }
        NodeData::Operation {
            op,
            smooth_k,
            left,
            right,
        } => {
            p.node_type = 2;
            p.csg_op = CsgOp::ALL.iter().position(|v| v == op).unwrap_or(0) as i32;
            p.smooth_k = *smooth_k;
            p.has_left = left.is_some();
            p.has_right = right.is_some();
            p.left_name = left
                .and_then(|id| scene.nodes.get(&id))
                .map(|n| n.name.clone())
                .unwrap_or_default();
            p.right_name = right
                .and_then(|id| scene.nodes.get(&id))
                .map(|n| n.name.clone())
                .unwrap_or_default();
        }
        NodeData::Transform { kind, value, .. } => {
            p.node_type = 3;
            p.transform_kind =
                TransformKind::ALL.iter().position(|v| v == kind).unwrap_or(0) as i32;
            if *kind == TransformKind::Rotate {
                p.value_x = value.x.to_degrees();
                p.value_y = value.y.to_degrees();
                p.value_z = value.z.to_degrees();
            } else {
                p.value_x = value.x;
                p.value_y = value.y;
                p.value_z = value.z;
            }
        }
        NodeData::Modifier {
            kind, value, extra, ..
        } => {
            p.node_type = 4;
            p.modifier_kind =
                ModifierKind::ALL.iter().position(|v| v == kind).unwrap_or(0) as i32;
            p.value_x = value.x;
            p.value_y = value.y;
            p.value_z = value.z;
            p.extra_x = extra.x;
            p.extra_y = extra.y;
            p.extra_z = extra.z;
        }
        NodeData::Sculpt {
            position,
            rotation,
            color,
            ..
        } => {
            p.node_type = 5;
            p.pos_x = position.x;
            p.pos_y = position.y;
            p.pos_z = position.z;
            p.rot_x = rotation.x.to_degrees();
            p.rot_y = rotation.y.to_degrees();
            p.rot_z = rotation.z.to_degrees();
            p.color_r = color.x;
            p.color_g = color.y;
            p.color_b = color.z;
            // Sculpt material: we need to read from the Sculpt variant directly
            // For now use defaults since Sculpt doesn't store roughness/metallic/fresnel separately
            // (it reuses the defaults). If the Sculpt variant has them, add here.
            p.roughness = 0.5;
            p.metallic = 0.0;
            p.fresnel = 0.04;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Render settings
// ---------------------------------------------------------------------------

pub struct RenderSettingsDataRust {
    pub shadows_enabled: bool,
    pub shadow_steps: f32,
    pub shadow_penumbra_k: f32,
    pub shadow_bias: f32,
    pub shadow_mint: f32,
    pub shadow_maxt: f32,
    pub ao_enabled: bool,
    pub ao_samples: f32,
    pub ao_step: f32,
    pub ao_decay: f32,
    pub ao_intensity: f32,
    pub march_max_steps: f32,
    pub march_epsilon: f32,
    pub march_step_multiplier: f32,
    pub march_max_distance: f32,
    pub key_light_dir_x: f32,
    pub key_light_dir_y: f32,
    pub key_light_dir_z: f32,
    pub key_diffuse: f32,
    pub key_spec_power: f32,
    pub key_spec_intensity: f32,
    pub fill_light_dir_x: f32,
    pub fill_light_dir_y: f32,
    pub fill_light_dir_z: f32,
    pub fill_intensity: f32,
    pub ambient: f32,
    pub sky_horizon_r: f32,
    pub sky_horizon_g: f32,
    pub sky_horizon_b: f32,
    pub sky_zenith_r: f32,
    pub sky_zenith_g: f32,
    pub sky_zenith_b: f32,
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_color_r: f32,
    pub fog_color_g: f32,
    pub fog_color_b: f32,
    pub gamma: f32,
    pub tonemapping_aces: bool,
    pub outline_color_r: f32,
    pub outline_color_g: f32,
    pub outline_color_b: f32,
    pub outline_thickness: f32,
    pub show_grid: bool,
}

pub fn build_render_settings(c: &RenderConfig) -> RenderSettingsDataRust {
    RenderSettingsDataRust {
        shadows_enabled: c.shadows_enabled,
        shadow_steps: c.shadow_steps as f32,
        shadow_penumbra_k: c.shadow_penumbra_k,
        shadow_bias: c.shadow_bias,
        shadow_mint: c.shadow_mint,
        shadow_maxt: c.shadow_maxt,
        ao_enabled: c.ao_enabled,
        ao_samples: c.ao_samples as f32,
        ao_step: c.ao_step,
        ao_decay: c.ao_decay,
        ao_intensity: c.ao_intensity,
        march_max_steps: c.march_max_steps as f32,
        march_epsilon: c.march_epsilon,
        march_step_multiplier: c.march_step_multiplier,
        march_max_distance: c.march_max_distance,
        key_light_dir_x: c.key_light_dir[0],
        key_light_dir_y: c.key_light_dir[1],
        key_light_dir_z: c.key_light_dir[2],
        key_diffuse: c.key_diffuse,
        key_spec_power: c.key_spec_power,
        key_spec_intensity: c.key_spec_intensity,
        fill_light_dir_x: c.fill_light_dir[0],
        fill_light_dir_y: c.fill_light_dir[1],
        fill_light_dir_z: c.fill_light_dir[2],
        fill_intensity: c.fill_intensity,
        ambient: c.ambient,
        sky_horizon_r: c.sky_horizon[0],
        sky_horizon_g: c.sky_horizon[1],
        sky_horizon_b: c.sky_horizon[2],
        sky_zenith_r: c.sky_zenith[0],
        sky_zenith_g: c.sky_zenith[1],
        sky_zenith_b: c.sky_zenith[2],
        fog_enabled: c.fog_enabled,
        fog_density: c.fog_density,
        fog_color_r: c.fog_color[0],
        fog_color_g: c.fog_color[1],
        fog_color_b: c.fog_color[2],
        gamma: c.gamma,
        tonemapping_aces: c.tonemapping_aces,
        outline_color_r: c.outline_color[0],
        outline_color_g: c.outline_color[1],
        outline_color_b: c.outline_color[2],
        outline_thickness: c.outline_thickness,
        show_grid: c.show_grid,
    }
}

// ---------------------------------------------------------------------------
// Param dispatch helpers (called from app_slint.rs callbacks)
// ---------------------------------------------------------------------------

pub fn apply_float_update(scene: &mut Scene, node_id: NodeId, param: i32, value: f32) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            ref mut color,
            ref mut roughness,
            ref mut metallic,
            ref mut fresnel,
            ref mut emissive,
            ref mut emissive_intensity,
            ..
        } => match param {
            10 => color.x = value,
            11 => color.y = value,
            12 => color.z = value,
            13 => *roughness = value,
            14 => *metallic = value,
            15 => *fresnel = value,
            16 => emissive.x = value,
            17 => emissive.y = value,
            18 => emissive.z = value,
            19 => *emissive_intensity = value,
            _ => {}
        },
        NodeData::Operation {
            ref mut smooth_k, ..
        } => {
            if param == 20 {
                *smooth_k = value;
            }
        }
        NodeData::Sculpt {
            ref mut color, ..
        } => match param {
            10 => color.x = value,
            11 => color.y = value,
            12 => color.z = value,
            _ => {}
        },
        _ => {}
    }
}

pub fn apply_vec3_update(
    scene: &mut Scene,
    node_id: NodeId,
    param_group: i32,
    axis: i32,
    text: &str,
) {
    let Ok(val) = text.parse::<f32>() else {
        return;
    };
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };

    match &mut node.data {
        NodeData::Primitive {
            ref mut position,
            ref mut rotation,
            ref mut scale,
            ..
        } => match param_group {
            0 => set_axis(position, axis, val),
            1 => set_axis(rotation, axis, val.to_radians()),
            2 => set_axis(scale, axis, val),
            _ => {}
        },
        NodeData::Transform {
            ref mut value,
            kind,
            ..
        } => {
            if param_group == 3 {
                if *kind == TransformKind::Rotate {
                    set_axis(value, axis, val.to_radians());
                } else {
                    set_axis(value, axis, val);
                }
            }
        }
        NodeData::Modifier {
            ref mut value,
            ref mut extra,
            ..
        } => match param_group {
            4 => set_axis(value, axis, val),
            5 => set_axis(extra, axis, val),
            _ => {}
        },
        NodeData::Sculpt {
            ref mut position,
            ref mut rotation,
            ..
        } => match param_group {
            0 => set_axis(position, axis, val),
            1 => set_axis(rotation, axis, val.to_radians()),
            _ => {}
        },
        _ => {}
    }
}

fn set_axis(v: &mut glam::Vec3, axis: i32, val: f32) {
    match axis {
        0 => v.x = val,
        1 => v.y = val,
        2 => v.z = val,
        _ => {}
    }
}

pub fn apply_kind_update(scene: &mut Scene, node_id: NodeId, kind_index: i32) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive { ref mut kind, .. } => {
            if let Some(new_kind) = SdfPrimitive::ALL.get(kind_index as usize) {
                *kind = new_kind.clone();
            }
        }
        NodeData::Operation { ref mut op, .. } => {
            if let Some(new_op) = CsgOp::ALL.get(kind_index as usize) {
                *op = new_op.clone();
            }
        }
        NodeData::Transform { ref mut kind, .. } => {
            if let Some(new_kind) = TransformKind::ALL.get(kind_index as usize) {
                *kind = new_kind.clone();
            }
        }
        NodeData::Modifier { ref mut kind, .. } => {
            if let Some(new_kind) = ModifierKind::ALL.get(kind_index as usize) {
                *kind = new_kind.clone();
            }
        }
        _ => {}
    }
}

pub fn apply_render_bool(config: &mut RenderConfig, param: i32, value: bool) {
    match param {
        0 => config.shadows_enabled = value,
        10 => config.ao_enabled = value,
        50 => config.show_grid = value,
        60 => config.fog_enabled = value,
        71 => config.tonemapping_aces = value,
        _ => {}
    }
}

pub fn apply_render_float(config: &mut RenderConfig, param: i32, value: f32) {
    match param {
        2 => config.shadow_penumbra_k = value,
        3 => config.shadow_bias = value,
        4 => config.shadow_mint = value,
        5 => config.shadow_maxt = value,
        12 => config.ao_step = value,
        13 => config.ao_decay = value,
        14 => config.ao_intensity = value,
        21 => config.march_epsilon = value,
        22 => config.march_step_multiplier = value,
        23 => config.march_max_distance = value,
        30 => config.key_light_dir[0] = value,
        31 => config.key_light_dir[1] = value,
        32 => config.key_light_dir[2] = value,
        33 => config.key_diffuse = value,
        34 => config.key_spec_power = value,
        35 => config.key_spec_intensity = value,
        36 => config.fill_light_dir[0] = value,
        37 => config.fill_light_dir[1] = value,
        38 => config.fill_light_dir[2] = value,
        39 => config.fill_intensity = value,
        40 => config.ambient = value,
        51 => config.sky_horizon[0] = value,
        52 => config.sky_horizon[1] = value,
        53 => config.sky_horizon[2] = value,
        54 => config.sky_zenith[0] = value,
        55 => config.sky_zenith[1] = value,
        56 => config.sky_zenith[2] = value,
        61 => config.fog_density = value,
        62 => config.fog_color[0] = value,
        63 => config.fog_color[1] = value,
        64 => config.fog_color[2] = value,
        70 => config.gamma = value,
        80 => config.outline_color[0] = value,
        81 => config.outline_color[1] = value,
        82 => config.outline_color[2] = value,
        83 => config.outline_thickness = value,
        _ => {}
    }
}

pub fn apply_render_int(config: &mut RenderConfig, param: i32, value: i32) {
    match param {
        1 => config.shadow_steps = value,
        11 => config.ao_samples = value,
        20 => config.march_max_steps = value,
        _ => {}
    }
}

pub fn apply_render_preset(config: &mut RenderConfig, preset: i32) {
    match preset {
        0 => {
            // Fast
            config.shadows_enabled = false;
            config.ao_enabled = false;
            config.march_max_steps = 64;
            config.interaction_render_scale = 0.35;
        }
        1 => {
            // Balanced
            config.shadows_enabled = false;
            config.ao_enabled = true;
            config.march_max_steps = 128;
            config.interaction_render_scale = 0.5;
        }
        2 => {
            // Quality
            config.shadows_enabled = true;
            config.ao_enabled = true;
            config.march_max_steps = 256;
            config.interaction_render_scale = 0.5;
            config.tonemapping_aces = true;
        }
        _ => {}
    }
}

pub fn apply_render_reset_section(config: &mut RenderConfig, section: i32) {
    match section {
        0 => config.reset_shadows(),
        1 => config.reset_ao(),
        2 => config.reset_raymarching(),
        3 => config.reset_lighting(),
        4 => config.reset_sky(),
        5 => config.reset_fog(),
        6 => config.reset_gamma(),
        7 => config.reset_outline(),
        _ => {}
    }
}
