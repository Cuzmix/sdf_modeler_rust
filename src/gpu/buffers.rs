use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::gpu::codegen::build_cookie_mapping;
use crate::graph::scene::{LightType, NodeData, NodeId, ProximityMode, Scene, MAX_SCENE_LIGHTS};
use crate::graph::voxel::evaluate_scene_sdf_at_point;

/// 208-byte GPU node (13 x vec4f).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SdfNodeGpu {
    pub type_op: [f32; 4],   // [type_val, smooth_k, metallic, roughness]
    pub position: [f32; 4],  // [x, y, z, 0]
    pub rotation: [f32; 4],  // [rx, ry, rz, 0] (radians)
    pub scale: [f32; 4],     // [sx, sy, sz, 0]
    pub color: [f32; 4],     // [base_color.r, base_color.g, base_color.b, is_selected]
    pub extra0: [f32; 4],    // prim: unused; sculpt: [voxel_offset, resolution, 0, 0]
    pub extra1: [f32; 4],    // prim: unused; sculpt: [bounds_min.xyz, 0]
    pub extra2: [f32; 4],    // prim: unused; sculpt: [bounds_max.xyz, 0]
    pub material0: [f32; 4], // [emissive.rgb, emissive_intensity]
    pub material1: [f32; 4], // [reflectance_f0, clearcoat, clearcoat_roughness, sheen_roughness]
    pub material2: [f32; 4], // [sheen_color.rgb, transmission]
    pub material3: [f32; 4], // [anisotropy_direction_local.xyz, thickness]
    pub material4: [f32; 4], // [anisotropy_strength, ior, 0, 0]
}

/// Info about sculpt nodes for texture binding.
pub struct SculptTexInfo {
    pub node_id: NodeId,
    pub tex_idx: usize,
    /// true if sculpt has an analytical child (differential SDF), false if standalone (total SDF).
    pub has_input: bool,
}

/// Count sculpt nodes in topo order and return their info for texture creation.
pub fn collect_sculpt_tex_info(scene: &Scene) -> Vec<SculptTexInfo> {
    let order = scene.visible_topo_order();
    let mut infos = Vec::new();
    for &node_id in &order {
        if let Some(node) = scene.nodes.get(&node_id) {
            if let NodeData::Sculpt { input, .. } = node.data {
                infos.push(SculptTexInfo {
                    node_id,
                    tex_idx: infos.len(),
                    has_input: input.is_some(),
                });
            }
        }
    }
    infos
}

/// Build the concatenated voxel data buffer.
/// Returns (flat_data, offset_map) where offset_map maps NodeId → offset in flat_data (f32 elements).
pub fn build_voxel_buffer(scene: &Scene) -> (Vec<f32>, HashMap<NodeId, u32>) {
    let order = scene.visible_topo_order();
    let mut flat_data = Vec::new();
    let mut offsets = HashMap::new();

    for &node_id in &order {
        if let Some(node) = scene.nodes.get(&node_id) {
            if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                let offset = flat_data.len() as u32;
                offsets.insert(node_id, offset);
                flat_data.extend_from_slice(&voxel_grid.data);
            }
        }
    }

    (flat_data, offsets)
}

pub fn build_node_buffer(
    scene: &Scene,
    selected_set: &std::collections::HashSet<NodeId>,
    voxel_offsets: &HashMap<NodeId, u32>,
) -> Vec<SdfNodeGpu> {
    let order = scene.visible_topo_order();
    let mut buffer = Vec::with_capacity(order.len().max(1));

    for &node_id in &order {
        let Some(node) = scene.nodes.get(&node_id) else {
            buffer.push(SdfNodeGpu::zeroed());
            continue;
        };
        let is_sel = if selected_set.contains(&node_id) {
            1.0
        } else {
            0.0
        };
        let light_mask = scene.get_light_mask(node_id) as f32;

        match &node.data {
            NodeData::Primitive {
                kind,
                position,
                rotation,
                scale,
                material,
                ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [
                        kind.gpu_type_id(),
                        0.0,
                        material.metallic,
                        material.roughness,
                    ],
                    position: [position.x, position.y, position.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, light_mask],
                    color: [
                        material.base_color.x,
                        material.base_color.y,
                        material.base_color.z,
                        is_sel,
                    ],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                    material0: [
                        material.emissive_color.x,
                        material.emissive_color.y,
                        material.emissive_color.z,
                        material.emissive_intensity,
                    ],
                    material1: [
                        material.reflectance_f0,
                        material.clearcoat,
                        material.clearcoat_roughness,
                        material.sheen_roughness,
                    ],
                    material2: [
                        material.sheen_color.x,
                        material.sheen_color.y,
                        material.sheen_color.z,
                        material.transmission,
                    ],
                    material3: [
                        material.anisotropy_direction_local.x,
                        material.anisotropy_direction_local.y,
                        material.anisotropy_direction_local.z,
                        material.thickness,
                    ],
                    material4: [material.anisotropy_strength, material.ior, 0.0, 0.0],
                });
            }
            NodeData::Operation {
                op,
                smooth_k,
                steps,
                color_blend,
                ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [op.gpu_op_id(), *smooth_k, *steps, *color_blend],
                    position: [0.0; 4],
                    rotation: [0.0; 4],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                    material0: [0.0; 4],
                    material1: [0.0; 4],
                    material2: [0.0; 4],
                    material3: [0.0; 4],
                    material4: [0.0; 4],
                });
            }
            NodeData::Sculpt {
                position,
                rotation,
                material,
                layer_intensity,
                voxel_grid,
                ..
            } => {
                let offset = voxel_offsets.get(&node_id).copied().unwrap_or(0);
                buffer.push(SdfNodeGpu {
                    type_op: [20.0, 0.0, material.metallic, material.roughness],
                    position: [position.x, position.y, position.z, *layer_intensity],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [1.0, 1.0, 1.0, light_mask],
                    color: [
                        material.base_color.x,
                        material.base_color.y,
                        material.base_color.z,
                        is_sel,
                    ],
                    extra0: [offset as f32, voxel_grid.resolution as f32, 0.0, 0.0],
                    extra1: [
                        voxel_grid.bounds_min.x,
                        voxel_grid.bounds_min.y,
                        voxel_grid.bounds_min.z,
                        0.0,
                    ],
                    extra2: [
                        voxel_grid.bounds_max.x,
                        voxel_grid.bounds_max.y,
                        voxel_grid.bounds_max.z,
                        0.0,
                    ],
                    material0: [
                        material.emissive_color.x,
                        material.emissive_color.y,
                        material.emissive_color.z,
                        material.emissive_intensity,
                    ],
                    material1: [
                        material.reflectance_f0,
                        material.clearcoat,
                        material.clearcoat_roughness,
                        material.sheen_roughness,
                    ],
                    material2: [
                        material.sheen_color.x,
                        material.sheen_color.y,
                        material.sheen_color.z,
                        material.transmission,
                    ],
                    material3: [
                        material.anisotropy_direction_local.x,
                        material.anisotropy_direction_local.y,
                        material.anisotropy_direction_local.z,
                        material.thickness,
                    ],
                    material4: [material.anisotropy_strength, material.ior, 0.0, 0.0],
                });
            }
            NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [21.0, 0.0, 0.0, 0.0],
                    position: [translation.x, translation.y, translation.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                    material0: [0.0; 4],
                    material1: [0.0; 4],
                    material2: [0.0; 4],
                    material3: [0.0; 4],
                    material4: [0.0; 4],
                });
            }
            NodeData::Modifier {
                kind, value, extra, ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [kind.gpu_type_id(), 0.0, 0.0, 0.0],
                    position: [value.x, value.y, value.z, 0.0],
                    rotation: [extra.x, extra.y, extra.z, 0.0],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                    material0: [0.0; 4],
                    material1: [0.0; 4],
                    material2: [0.0; 4],
                    material3: [0.0; 4],
                    material4: [0.0; 4],
                });
            }
            NodeData::Light {
                color,
                intensity,
                range,
                spot_angle,
                light_type,
                ..
            } => {
                let type_val = match light_type {
                    crate::graph::scene::LightType::Point => 50.0,
                    crate::graph::scene::LightType::Spot => 51.0,
                    crate::graph::scene::LightType::Directional => 52.0,
                    crate::graph::scene::LightType::Ambient => 53.0,
                    crate::graph::scene::LightType::Array => 54.0,
                };
                buffer.push(SdfNodeGpu {
                    type_op: [type_val, *intensity, *range, *spot_angle],
                    position: [0.0; 4],
                    rotation: [0.0; 4],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                    material0: [0.0; 4],
                    material1: [0.0; 4],
                    material2: [0.0; 4],
                    material3: [0.0; 4],
                    material4: [0.0; 4],
                });
            }
        }
    }

    // Ensure at least one element (avoid zero-sized buffer)
    if buffer.is_empty() {
        buffer.push(SdfNodeGpu::zeroed());
    }

    buffer
}

/// GPU representation of a scene light (5 × vec4f = 80 bytes).
/// Packed into the camera uniform buffer as a flat array of vec4f.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SceneLightGpu {
    /// xyz = world position, w = light_type (0=point, 1=spot, 2=directional)
    pub position_type: [f32; 4],
    /// xyz = direction (normalized), w = intensity
    pub direction_intensity: [f32; 4],
    /// rgb = color, w = range
    pub color_range: [f32; 4],
    /// x = cos(half_spot_angle), y = cast_shadows (0/1), z = shadow_softness,
    /// w = packed shadow_color (RGB8 encoded as floor(r*255)*65536 + floor(g*255)*256 + floor(b*255))
    pub params: [f32; 4],
    /// x = volumetric (0.0 = off, 1.0 = on), y = volumetric_density,
    /// z = has_cookie (0.0 = no, 1.0 = yes), w = cookie_sdf_index (-1 = none, 0+ = index)
    pub volumetric: [f32; 4],
}

/// Collected ambient contribution from scene Ambient light nodes.
#[derive(Default)]
pub struct SceneAmbient {
    /// Total ambient color (sum of all visible Ambient lights: color * intensity).
    pub color: glam::Vec3,
}

/// Compute local positions for a Light Array pattern.
fn expand_array_pattern(
    pattern: &crate::graph::scene::ArrayPattern,
    count: u32,
    radius: f32,
) -> Vec<glam::Vec3> {
    use crate::graph::scene::ArrayPattern;
    let n = count.max(1) as usize;
    let mut positions = Vec::with_capacity(n);
    match pattern {
        ArrayPattern::Ring => {
            for i in 0..n {
                let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
                positions.push(glam::Vec3::new(
                    angle.cos() * radius,
                    0.0,
                    angle.sin() * radius,
                ));
            }
        }
        ArrayPattern::Line => {
            let total_length = radius * 2.0;
            for i in 0..n {
                let t = if n > 1 {
                    i as f32 / (n - 1) as f32
                } else {
                    0.5
                };
                positions.push(glam::Vec3::new(-radius + t * total_length, 0.0, 0.0));
            }
        }
        ArrayPattern::Grid => {
            let side = (n as f32).sqrt().ceil() as usize;
            let mut placed = 0;
            for row in 0..side {
                for col in 0..side {
                    if placed >= n {
                        break;
                    }
                    let tx = if side > 1 {
                        col as f32 / (side - 1) as f32
                    } else {
                        0.5
                    };
                    let tz = if side > 1 {
                        row as f32 / (side - 1) as f32
                    } else {
                        0.5
                    };
                    positions.push(glam::Vec3::new(
                        -radius + tx * radius * 2.0,
                        0.0,
                        -radius + tz * radius * 2.0,
                    ));
                    placed += 1;
                }
            }
        }
        ArrayPattern::Spiral => {
            for i in 0..n {
                let t = i as f32 / n.max(1) as f32;
                let angle = t * std::f32::consts::TAU * 2.0; // 2 full revolutions
                let r = t * radius;
                positions.push(glam::Vec3::new(angle.cos() * r, 0.0, angle.sin() * r));
            }
        }
    }
    positions
}

/// Rotate a color's hue by a given number of degrees (0-360).
fn hue_rotate_color(color: glam::Vec3, degrees: f32) -> glam::Vec3 {
    // Convert RGB to HSV, shift hue, convert back
    let r = color.x;
    let g = color.y;
    let b = color.z;
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    let delta = max_c - min_c;

    // Compute hue (0-360)
    let hue = if delta < 1e-6 {
        0.0
    } else if (max_c - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max_c - g).abs() < 1e-6 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let saturation = if max_c < 1e-6 { 0.0 } else { delta / max_c };
    let value = max_c;

    // Shift hue
    let new_hue = (hue + degrees).rem_euclid(360.0);

    // HSV to RGB
    let c = value * saturation;
    let x = c * (1.0 - ((new_hue / 60.0) % 2.0 - 1.0).abs());
    let m = value - c;
    let (r1, g1, b1) = if new_hue < 60.0 {
        (c, x, 0.0)
    } else if new_hue < 120.0 {
        (x, c, 0.0)
    } else if new_hue < 180.0 {
        (0.0, c, x)
    } else if new_hue < 240.0 {
        (0.0, x, c)
    } else if new_hue < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    glam::Vec3::new(r1 + m, g1 + m, b1 + m)
}

/// Collect visible Light nodes from the scene, sorted by distance to camera (nearest first).
/// Returns (directional/point/spot count, gpu lights, accumulated ambient).
/// Ambient-type lights contribute to the returned `SceneAmbient` instead of the light array.
pub fn collect_scene_lights(
    scene: &Scene,
    camera_pos: glam::Vec3,
    soloed_light: Option<NodeId>,
    time: f32,
) -> (u32, Vec<SceneLightGpu>, SceneAmbient) {
    let parent_map = scene.build_parent_map();
    let cookie_map = build_cookie_mapping(scene);
    let mut lights: Vec<(f32, SceneLightGpu)> = Vec::new();
    let mut ambient = SceneAmbient::default();

    for (&id, node) in &scene.nodes {
        if let NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            spot_angle,
            cast_shadows,
            shadow_softness,
            shadow_color,
            volumetric,
            volumetric_density,
            proximity_mode,
            proximity_range,
            array_config,
            intensity_expr,
            color_hue_expr,
            ..
        } = &node.data
        {
            if scene.is_hidden(id) {
                continue;
            }

            // Solo mode: skip all lights except the soloed one
            if let Some(solo_id) = soloed_light {
                if id != solo_id {
                    continue;
                }
            }

            // Evaluate expression overrides for intensity and color hue
            let effective_base_intensity = if let Some(ref expr_str) = intensity_expr {
                crate::expression::parse_expression(expr_str)
                    .map(|expr| crate::expression::evaluate(&expr, time))
                    .unwrap_or(*intensity)
            } else {
                *intensity
            };
            let effective_color = if let Some(ref expr_str) = color_hue_expr {
                crate::expression::parse_expression(expr_str)
                    .map(|expr| {
                        let hue_degrees = crate::expression::evaluate(&expr, time);
                        hue_rotate_color(*color, hue_degrees)
                    })
                    .unwrap_or(*color)
            } else {
                *color
            };

            // Ambient lights accumulate into scene ambient — no position/direction needed
            if *light_type == LightType::Ambient {
                // Suppress ambient when soloing a non-ambient light
                if soloed_light.is_none() {
                    ambient.color += effective_color * effective_base_intensity;
                }
                continue;
            }

            // Find the parent Transform to get world position and rotation
            let Some(&transform_id) = parent_map.get(&id) else {
                continue;
            };
            let Some(transform_node) = scene.nodes.get(&transform_id) else {
                continue;
            };
            let NodeData::Transform {
                translation,
                rotation,
                ..
            } = &transform_node.data
            else {
                continue;
            };

            // Light Array: expand into N individual point lights
            if *light_type == LightType::Array {
                if let Some(cfg) = array_config {
                    let positions = expand_array_pattern(&cfg.pattern, cfg.count, cfg.radius);
                    for (instance_index, local_pos) in positions.iter().enumerate() {
                        let world_pos = *translation + *local_pos;
                        let dist_to_camera = (world_pos - camera_pos).length();
                        // Apply hue variation across instances
                        let instance_color = if cfg.color_variation > 0.0 {
                            let hue_shift = (instance_index as f32 / positions.len() as f32)
                                * cfg.color_variation;
                            hue_rotate_color(effective_color, hue_shift * 360.0)
                        } else {
                            effective_color
                        };
                        lights.push((
                            dist_to_camera,
                            SceneLightGpu {
                                position_type: [world_pos.x, world_pos.y, world_pos.z, 0.0], // Point type
                                direction_intensity: [0.0, -1.0, 0.0, effective_base_intensity],
                                color_range: [
                                    instance_color.x,
                                    instance_color.y,
                                    instance_color.z,
                                    *range,
                                ],
                                params: [1.0, 0.0, 8.0, 0.0], // No shadows, no cookie
                                volumetric: [0.0, 0.0, 0.0, -1.0],
                            },
                        ));
                    }
                }
                continue;
            }

            // Compute direction from rotation (default light direction is -Y).
            // Use inverse rotation so GPU direction matches gizmo drag convention.
            let direction = inverse_rotate_euler_light(glam::Vec3::NEG_Y, *rotation);
            let direction = if direction.length_squared() > 0.001 {
                direction.normalize()
            } else {
                glam::Vec3::NEG_Y
            };

            let type_val = match light_type {
                LightType::Point => 0.0,
                LightType::Spot => 1.0,
                LightType::Directional => 2.0,
                LightType::Ambient | LightType::Array => unreachable!(), // handled above
            };

            let half_angle_rad = (spot_angle * 0.5).to_radians();
            let cos_half_angle = half_angle_rad.cos();

            // Compute proximity-modulated intensity on CPU (zero GPU cost).
            let effective_intensity = match proximity_mode {
                ProximityMode::Off => effective_base_intensity,
                ProximityMode::Brighten | ProximityMode::Dim => {
                    let sdf_dist = evaluate_scene_sdf_at_point(scene, *translation);
                    let proximity_range_clamped = proximity_range.max(0.001);
                    // smoothstep(edge0, edge1, x): 0 at edge0, 1 at edge1
                    let t = (sdf_dist / proximity_range_clamped).clamp(0.0, 1.0);
                    let smooth_t = t * t * (3.0 - 2.0 * t);
                    match proximity_mode {
                        // Brighten: factor = 1.0 + (1.0 - smooth_t), so max 2x at surface
                        ProximityMode::Brighten => {
                            effective_base_intensity * (1.0 + (1.0 - smooth_t))
                        }
                        // Dim: factor = smooth_t, so 0 at surface, 1 at range
                        ProximityMode::Dim => effective_base_intensity * smooth_t,
                        ProximityMode::Off => unreachable!(),
                    }
                }
            };

            let dist_to_camera = (*translation - camera_pos).length();

            lights.push((
                dist_to_camera,
                SceneLightGpu {
                    position_type: [translation.x, translation.y, translation.z, type_val],
                    direction_intensity: [
                        direction.x,
                        direction.y,
                        direction.z,
                        effective_intensity,
                    ],
                    color_range: [
                        effective_color.x,
                        effective_color.y,
                        effective_color.z,
                        *range,
                    ],
                    params: [
                        cos_half_angle,
                        if *cast_shadows { 1.0 } else { 0.0 },
                        *shadow_softness,
                        pack_rgb8(*shadow_color),
                    ],
                    volumetric: [
                        if *volumetric { 1.0 } else { 0.0 },
                        *volumetric_density,
                        if cookie_map.contains_key(&id) {
                            1.0
                        } else {
                            0.0
                        },
                        cookie_map.get(&id).map(|&idx| idx as f32).unwrap_or(-1.0),
                    ],
                },
            ));
        }
    }

    // Sort by distance to camera (nearest first)
    lights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let count = lights.len().min(MAX_SCENE_LIGHTS) as u32;
    let gpu_lights: Vec<SceneLightGpu> = lights
        .into_iter()
        .take(MAX_SCENE_LIGHTS)
        .map(|(_, l)| l)
        .collect();

    (count, gpu_lights, ambient)
}

/// Identify which Light nodes are active (nearest to camera, up to MAX_SCENE_LIGHTS).
/// Returns (active_light_node_ids, total_light_count).
#[allow(dead_code)]
pub fn identify_active_lights(
    scene: &Scene,
    camera_pos: glam::Vec3,
) -> (std::collections::HashSet<NodeId>, usize) {
    let parent_map = scene.build_parent_map();
    let mut light_distances: Vec<(f32, NodeId)> = Vec::new();

    for (&id, node) in &scene.nodes {
        if !matches!(node.data, NodeData::Light { .. }) {
            continue;
        }
        if scene.is_hidden(id) {
            continue;
        }
        let Some(&transform_id) = parent_map.get(&id) else {
            continue;
        };
        let Some(transform_node) = scene.nodes.get(&transform_id) else {
            continue;
        };
        let NodeData::Transform { translation, .. } = &transform_node.data else {
            continue;
        };
        let dist = (*translation - camera_pos).length();
        light_distances.push((dist, id));
    }

    let total_count = light_distances.len();
    light_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let active_ids: std::collections::HashSet<NodeId> = light_distances
        .iter()
        .take(MAX_SCENE_LIGHTS)
        .map(|(_, id)| *id)
        .collect();

    (active_ids, total_count)
}

/// Pack an RGB color (each component 0.0–1.0) into a single f32.
/// Encoding: floor(r*255)*65536 + floor(g*255)*256 + floor(b*255).
/// Unpacked in WGSL: r = floor(v / 65536.0) / 255.0, etc.
pub fn pack_rgb8(color: glam::Vec3) -> f32 {
    let r = (color.x.clamp(0.0, 1.0) * 255.0).floor() as u32;
    let g = (color.y.clamp(0.0, 1.0) * 255.0).floor() as u32;
    let b = (color.z.clamp(0.0, 1.0) * 255.0).floor() as u32;
    (r * 65536 + g * 256 + b) as f32
}

/// Inverse Euler XYZ rotation (applies -Z, -Y, -X — matches gizmo drag convention).
fn inverse_rotate_euler_light(p: glam::Vec3, r: glam::Vec3) -> glam::Vec3 {
    let mut q = p;
    let (sz, cz) = (-r.z).sin_cos();
    q = glam::Vec3::new(cz * q.x - sz * q.y, sz * q.x + cz * q.y, q.z);
    let (sy, cy) = (-r.y).sin_cos();
    q = glam::Vec3::new(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
    let (sx, cx) = (-r.x).sin_cos();
    glam::Vec3::new(q.x, cx * q.y - sx * q.z, sx * q.y + cx * q.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{LightType, NodeData, Scene};
    use glam::Vec3;
    use std::collections::{HashMap, HashSet};

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        }
    }

    #[test]
    fn build_node_buffer_packs_transmission_and_anisotropy_material_fields() {
        let mut scene = empty_scene();
        scene.add_node(
            "Sphere".to_string(),
            NodeData::Primitive {
                kind: crate::graph::scene::SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::new(0.1, 0.2, 0.3),
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::new(0.8, 0.9, 1.0),
                    roughness: 0.35,
                    metallic: 0.0,
                    transmission: 0.85,
                    thickness: 1.75,
                    ior: 1.45,
                    anisotropy_strength: 0.6,
                    anisotropy_direction_local: Vec3::new(0.0, 1.0, 0.0),
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );

        let buffer = build_node_buffer(&scene, &HashSet::new(), &HashMap::new());
        assert_eq!(buffer.len(), 1);
        let node = buffer[0];
        assert!((node.material2[3] - 0.85).abs() < 1e-6);
        assert!((node.material3[1] - 1.0).abs() < 1e-6);
        assert!((node.material3[3] - 1.75).abs() < 1e-6);
        assert!((node.material4[0] - 0.6).abs() < 1e-6);
        assert!((node.material4[1] - 1.45).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // collect_scene_lights
    // -----------------------------------------------------------------------

    #[test]
    fn collect_scene_lights_empty_scene_returns_zero() {
        let scene = empty_scene();
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 0);
        assert!(lights.is_empty());
    }

    #[test]
    fn collect_scene_lights_single_point_light() {
        let mut scene = empty_scene();
        let (_light_id, _transform_id) = scene.create_light(LightType::Point);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        assert_eq!(lights.len(), 1);
        // Point light type = 0.0
        assert!((lights[0].position_type[3] - 0.0).abs() < 1e-5);
        // Default intensity = 1.0
        assert!((lights[0].direction_intensity[3] - 1.0).abs() < 1e-5);
        // Default color = white
        assert!((lights[0].color_range[0] - 1.0).abs() < 1e-5);
        assert!((lights[0].color_range[1] - 1.0).abs() < 1e-5);
        assert!((lights[0].color_range[2] - 1.0).abs() < 1e-5);
        // Default range = 10.0
        assert!((lights[0].color_range[3] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn collect_scene_lights_spot_light_type() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Spot);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Spot light type = 1.0
        assert!((lights[0].position_type[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn collect_scene_lights_directional_type() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Directional);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Directional light type = 2.0
        assert!((lights[0].position_type[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn collect_scene_lights_respects_max_limit() {
        let mut scene = empty_scene();
        // Create 10 lights (exceeds MAX_SCENE_LIGHTS = 8)
        for _ in 0..10 {
            scene.create_light(LightType::Point);
        }
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, MAX_SCENE_LIGHTS as u32);
        assert_eq!(lights.len(), MAX_SCENE_LIGHTS);
    }

    #[test]
    fn collect_scene_lights_sorted_by_distance_to_camera() {
        let mut scene = empty_scene();
        // Create light at (10, 0, 0) — far from camera
        let (_, far_transform) = scene.create_light(LightType::Point);
        if let Some(node) = scene.nodes.get_mut(&far_transform) {
            if let NodeData::Transform { translation, .. } = &mut node.data {
                *translation = Vec3::new(10.0, 0.0, 0.0);
            }
        }
        // Create light at (1, 0, 0) — close to camera
        let (_, near_transform) = scene.create_light(LightType::Point);
        if let Some(node) = scene.nodes.get_mut(&near_transform) {
            if let NodeData::Transform { translation, .. } = &mut node.data {
                *translation = Vec3::new(1.0, 0.0, 0.0);
            }
        }
        let camera_pos = Vec3::ZERO;
        let (count, lights, _ambient) = collect_scene_lights(&scene, camera_pos, None, 0.0);
        assert_eq!(count, 2);
        // First light should be the nearest one (at x=1)
        assert!(
            (lights[0].position_type[0] - 1.0).abs() < 1e-5,
            "nearest light should be first, got x={}",
            lights[0].position_type[0]
        );
        // Second light should be farther (at x=10)
        assert!(
            (lights[1].position_type[0] - 10.0).abs() < 1e-5,
            "farther light should be second, got x={}",
            lights[1].position_type[0]
        );
    }

    #[test]
    fn collect_scene_lights_hidden_lights_excluded() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        scene.hidden_nodes.insert(light_id);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 0);
        assert!(lights.is_empty());
    }

    #[test]
    fn collect_scene_lights_ambient_contributes_to_scene_ambient() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Ambient);
        // Set intensity on the ambient light
        for node in scene.nodes.values_mut() {
            if let NodeData::Light {
                light_type,
                intensity,
                ..
            } = &mut node.data
            {
                if *light_type == LightType::Ambient {
                    *intensity = 0.1;
                }
            }
        }
        let (count, lights, ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        // Ambient lights don't go into the light array
        assert_eq!(count, 0);
        assert!(lights.is_empty());
        // Ambient contribution should be non-zero (white * 0.1 = (0.1, 0.1, 0.1))
        assert!(ambient.color.x > 0.0);
        assert!(ambient.color.y > 0.0);
        assert!(ambient.color.z > 0.0);
    }

    // -----------------------------------------------------------------------
    // identify_active_lights
    // -----------------------------------------------------------------------

    #[test]
    fn identify_active_lights_empty_scene() {
        let scene = empty_scene();
        let (active, total) = identify_active_lights(&scene, Vec3::ZERO);
        assert_eq!(total, 0);
        assert!(active.is_empty());
    }

    #[test]
    fn identify_active_lights_within_limit() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        let (active, total) = identify_active_lights(&scene, Vec3::ZERO);
        assert_eq!(total, 1);
        assert_eq!(active.len(), 1);
        assert!(active.contains(&light_id));
    }

    #[test]
    fn identify_active_lights_exceeding_limit() {
        let mut scene = empty_scene();
        let mut all_light_ids = Vec::new();
        for _ in 0..10 {
            let (light_id, _) = scene.create_light(LightType::Point);
            all_light_ids.push(light_id);
        }
        let (active, total) = identify_active_lights(&scene, Vec3::ZERO);
        assert_eq!(total, 10);
        assert_eq!(active.len(), MAX_SCENE_LIGHTS);
        // All active IDs should be valid light node IDs
        for id in &active {
            assert!(all_light_ids.contains(id));
        }
    }

    // -----------------------------------------------------------------------
    // SceneLightGpu spot angle encoding
    // -----------------------------------------------------------------------

    #[test]
    fn scene_light_gpu_spot_angle_encoded_as_cos_half() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Spot);
        // Default spot_angle = 45 degrees
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { spot_angle, .. } = &mut node.data {
                *spot_angle = 90.0;
            }
        }
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        // cos(90/2 degrees) = cos(45 degrees) = sqrt(2)/2 ≈ 0.7071
        let expected_cos = (45.0_f32.to_radians()).cos();
        assert!(
            (lights[0].params[0] - expected_cos).abs() < 1e-3,
            "spot angle cosine encoding: expected {expected_cos}, got {}",
            lights[0].params[0]
        );
    }

    #[test]
    fn collect_scene_lights_negative_intensity_preserved() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        // Set negative intensity (subtractive light)
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { intensity, .. } = &mut node.data {
                *intensity = -3.5;
            }
        }
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Negative intensity must be preserved in the GPU buffer (direction_intensity.w)
        assert!(
            (lights[0].direction_intensity[3] - (-3.5)).abs() < 1e-5,
            "negative intensity must be preserved: expected -3.5, got {}",
            lights[0].direction_intensity[3]
        );
    }

    #[test]
    fn collect_scene_lights_shadow_params_packed() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Directional);
        // Set shadow params
        if let NodeData::Light {
            cast_shadows,
            shadow_softness,
            shadow_color,
            ..
        } = &mut scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cast_shadows = true;
            *shadow_softness = 16.0;
            *shadow_color = Vec3::new(0.0, 0.0, 1.0); // blue shadows
        }
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // params.y = cast_shadows = 1.0
        assert!((lights[0].params[1] - 1.0).abs() < 1e-5);
        // params.z = shadow_softness = 16.0
        assert!((lights[0].params[2] - 16.0).abs() < 1e-5);
        // params.w = packed shadow color (blue = r=0, g=0, b=255 → 0*65536 + 0*256 + 255 = 255.0)
        assert!((lights[0].params[3] - 255.0).abs() < 1e-3);
    }

    #[test]
    fn collect_scene_lights_no_shadows_by_default_for_point() {
        let mut scene = empty_scene();
        let (_light_id, _) = scene.create_light(LightType::Point);
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Point lights default to cast_shadows=false
        assert!(
            lights[0].params[1] < 0.5,
            "point light should default to no shadows"
        );
    }

    #[test]
    fn solo_light_filters_to_single_light() {
        let mut scene = empty_scene();
        let (light_a, _) = scene.create_light(LightType::Point);
        let (light_b, _) = scene.create_light(LightType::Spot);
        let (_light_c, _) = scene.create_light(LightType::Directional);

        // Without solo: all 3 lights returned
        let (count, _, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 3);

        // Solo light A: only 1 light returned
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, Some(light_a), 0.0);
        assert_eq!(count, 1);
        // Point light type = 0.0
        assert!((lights[0].position_type[3] - 0.0).abs() < 0.01);

        // Solo light B: only 1 light returned (Spot)
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, Some(light_b), 0.0);
        assert_eq!(count, 1);
        assert!((lights[0].position_type[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn solo_light_suppresses_ambient() {
        let mut scene = empty_scene();
        let (point_id, _) = scene.create_light(LightType::Point);
        let (ambient_id, _) = scene.create_light(LightType::Ambient);
        // Set ambient color
        if let Some(node) = scene.nodes.get_mut(&ambient_id) {
            if let NodeData::Light {
                ref mut color,
                ref mut intensity,
                ..
            } = node.data
            {
                *color = Vec3::new(1.0, 0.5, 0.2);
                *intensity = 2.0;
            }
        }

        // Without solo: ambient contributes
        let (_, _, ambient) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert!(ambient.color.length() > 0.0);

        // Solo the point light: ambient suppressed
        let (_, _, ambient) = collect_scene_lights(&scene, Vec3::ZERO, Some(point_id), 0.0);
        assert!(ambient.color.length() < 0.01);
    }

    #[test]
    fn pack_rgb8_encodes_correctly() {
        // Pure red
        let red = pack_rgb8(Vec3::new(1.0, 0.0, 0.0));
        assert!((red - (255.0 * 65536.0)).abs() < 1.0);
        // Pure green
        let green = pack_rgb8(Vec3::new(0.0, 1.0, 0.0));
        assert!((green - (255.0 * 256.0)).abs() < 1.0);
        // Pure blue
        let blue = pack_rgb8(Vec3::new(0.0, 0.0, 1.0));
        assert!((blue - 255.0).abs() < 1.0);
        // Black
        let black = pack_rgb8(Vec3::ZERO);
        assert!(black.abs() < 1.0);
    }

    #[test]
    fn proximity_off_preserves_intensity() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        // Default proximity_mode is Off, so intensity should be unchanged
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { intensity, .. } = &mut node.data {
                *intensity = 5.0;
            }
        }
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        assert!((lights[0].direction_intensity[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn proximity_brighten_increases_intensity_near_surface() {
        use crate::graph::scene::{ProximityMode, SdfPrimitive};
        let mut scene = empty_scene();
        // Create a sphere at origin with scale 1.0
        let _sphere_id = scene.add_node(
            "Sphere".to_string(),
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::ONE,
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        // Place light very close to sphere surface (sphere radius=1, light at x=1.2)
        let light_id = scene.add_node(
            "Light".to_string(),
            NodeData::Light {
                light_type: LightType::Point,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Brighten,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _transform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::new(1.2, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Light at 1.2 from sphere surface (d≈0.2), Brighten should boost intensity > 1.0
        assert!(lights[0].direction_intensity[3] > 1.0);
    }

    #[test]
    fn proximity_dim_decreases_intensity_near_surface() {
        use crate::graph::scene::{ProximityMode, SdfPrimitive};
        let mut scene = empty_scene();
        let _sphere_id = scene.add_node(
            "Sphere".to_string(),
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::ONE,
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        let light_id = scene.add_node(
            "Light".to_string(),
            NodeData::Light {
                light_type: LightType::Point,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Dim,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _transform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::new(1.2, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // Light at 1.2 from sphere surface (d≈0.2), Dim should reduce intensity < 1.0
        assert!(lights[0].direction_intensity[3] < 1.0);
        assert!(lights[0].direction_intensity[3] > 0.0);
    }

    #[test]
    fn light_array_ring_expands_to_n_point_lights() {
        use crate::graph::scene::{ArrayPattern, LightArrayConfig};
        let mut scene = empty_scene();
        let light_id = scene.add_node(
            "Array".to_string(),
            NodeData::Light {
                light_type: LightType::Array,
                color: Vec3::ONE,
                intensity: 2.0,
                range: 5.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: crate::graph::scene::ProximityMode::Off,
                proximity_range: 2.0,
                array_config: Some(LightArrayConfig {
                    pattern: ArrayPattern::Ring,
                    count: 4,
                    radius: 3.0,
                    color_variation: 0.0,
                }),
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _transform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 4, "Ring with 4 should produce 4 point lights");
        // All should be point type (0.0)
        for light in &lights {
            assert!(
                (light.position_type[3] - 0.0).abs() < 1e-5,
                "Array instances are point lights"
            );
            assert!(
                (light.direction_intensity[3] - 2.0).abs() < 1e-5,
                "Intensity preserved"
            );
            assert!((light.color_range[3] - 5.0).abs() < 1e-5, "Range preserved");
        }
    }

    #[test]
    fn light_array_respects_max_lights() {
        use crate::graph::scene::{ArrayPattern, LightArrayConfig};
        let mut scene = empty_scene();
        let light_id = scene.add_node(
            "Array".to_string(),
            NodeData::Light {
                light_type: LightType::Array,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 5.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: crate::graph::scene::ProximityMode::Off,
                proximity_range: 2.0,
                array_config: Some(LightArrayConfig {
                    pattern: ArrayPattern::Ring,
                    count: 20, // More than MAX_SCENE_LIGHTS
                    radius: 3.0,
                    color_variation: 0.0,
                }),
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _transform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(
            count as usize, MAX_SCENE_LIGHTS,
            "Should cap at MAX_SCENE_LIGHTS"
        );
        assert_eq!(lights.len(), MAX_SCENE_LIGHTS);
    }

    #[test]
    fn light_array_color_variation_shifts_hue() {
        use crate::graph::scene::{ArrayPattern, LightArrayConfig};
        let mut scene = empty_scene();
        let light_id = scene.add_node(
            "Array".to_string(),
            NodeData::Light {
                light_type: LightType::Array,
                color: Vec3::new(1.0, 0.0, 0.0), // Red
                intensity: 1.0,
                range: 5.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: crate::graph::scene::ProximityMode::Off,
                proximity_range: 2.0,
                array_config: Some(LightArrayConfig {
                    pattern: ArrayPattern::Ring,
                    count: 3,
                    radius: 2.0,
                    color_variation: 1.0, // Full rainbow spread
                }),
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _transform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 3);
        // First light should still be red-ish (hue shift 0)
        assert!(
            lights[0].color_range[0] > 0.8,
            "First instance should be red"
        );
        // Second light should be shifted significantly (hue shift ~120°)
        let second_is_different = (lights[1].color_range[0] - lights[0].color_range[0]).abs() > 0.3
            || (lights[1].color_range[1] - lights[0].color_range[1]).abs() > 0.3;
        assert!(
            second_is_different,
            "Color variation should shift hue for different instances"
        );
    }

    #[test]
    fn expand_array_pattern_line() {
        use crate::graph::scene::ArrayPattern;
        let positions = expand_array_pattern(&ArrayPattern::Line, 3, 2.0);
        assert_eq!(positions.len(), 3);
        // First at -radius, last at +radius
        assert!((positions[0].x - (-2.0)).abs() < 1e-5);
        assert!((positions[2].x - 2.0).abs() < 1e-5);
        // All at y=0, z=0
        for p in &positions {
            assert!((p.y).abs() < 1e-5);
            assert!((p.z).abs() < 1e-5);
        }
    }

    #[test]
    fn expand_array_pattern_grid() {
        use crate::graph::scene::ArrayPattern;
        let positions = expand_array_pattern(&ArrayPattern::Grid, 4, 1.0);
        assert_eq!(positions.len(), 4);
        // 4 lights → 2×2 grid
        // All should be at y=0
        for p in &positions {
            assert!((p.y).abs() < 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // Volumetric scattering: flag/density packing + volumetric light count
    // -----------------------------------------------------------------------

    #[test]
    fn volumetric_flag_and_density_packed_into_buffer() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut volumetric,
            ref mut volumetric_density,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *volumetric = true;
            *volumetric_density = 0.42;
        }
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 1);
        // volumetric.x = 1.0 (enabled)
        assert!(
            (lights[0].volumetric[0] - 1.0).abs() < 1e-5,
            "volumetric flag should be 1.0"
        );
        // volumetric.y = density
        assert!(
            (lights[0].volumetric[1] - 0.42).abs() < 1e-5,
            "volumetric density should be 0.42"
        );
    }

    #[test]
    fn volumetric_disabled_packs_zero_flag() {
        let mut scene = empty_scene();
        let (_light_id, _) = scene.create_light(LightType::Point);
        // Default: volumetric = false
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert!(
            (lights[0].volumetric[0]).abs() < 1e-5,
            "volumetric flag should be 0.0 by default"
        );
    }

    #[test]
    fn volumetric_light_count_computed_correctly() {
        let mut scene = empty_scene();
        // Create 3 lights: 2 volumetric, 1 not
        let (l1, _) = scene.create_light(LightType::Point);
        let (l2, _) = scene.create_light(LightType::Spot);
        let (_l3, _) = scene.create_light(LightType::Point);

        if let NodeData::Light {
            ref mut volumetric, ..
        } = scene.nodes.get_mut(&l1).unwrap().data
        {
            *volumetric = true;
        }
        if let NodeData::Light {
            ref mut volumetric, ..
        } = scene.nodes.get_mut(&l2).unwrap().data
        {
            *volumetric = true;
        }
        // l3 stays default (volumetric=false)

        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        let vol_count = lights.iter().filter(|l| l.volumetric[0] > 0.5).count();
        assert_eq!(vol_count, 2, "should have exactly 2 volumetric lights");
    }

    // -----------------------------------------------------------------------
    // Cookie SDF: buffer packing
    // -----------------------------------------------------------------------

    #[test]
    fn cookie_node_packs_into_volumetric_fields() {
        use crate::graph::scene::SdfPrimitive;
        let mut scene = empty_scene();
        let prim_id = scene.create_primitive(SdfPrimitive::Sphere);
        let (light_id, _) = scene.create_light(LightType::Spot);
        if let NodeData::Light {
            ref mut cookie_node,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cookie_node = Some(prim_id);
        }
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(lights.len(), 1);
        // volumetric.z = has_cookie = 1.0
        assert!(
            (lights[0].volumetric[2] - 1.0).abs() < 1e-5,
            "has_cookie should be 1.0"
        );
        // volumetric.w = cookie index >= 0
        assert!(
            lights[0].volumetric[3] >= 0.0,
            "cookie index should be >= 0"
        );
    }

    #[test]
    fn no_cookie_packs_negative_index() {
        let mut scene = empty_scene();
        let (_light_id, _) = scene.create_light(LightType::Point);
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        // volumetric.z = has_cookie = 0.0
        assert!(
            (lights[0].volumetric[2]).abs() < 1e-5,
            "has_cookie should be 0.0"
        );
        // volumetric.w = cookie index = -1.0
        assert!(
            (lights[0].volumetric[3] - (-1.0)).abs() < 1e-5,
            "cookie index should be -1.0"
        );
    }

    // -----------------------------------------------------------------------
    // Shadow: max 2 shadow-casters packed correctly
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_shadow_casters_all_marked_in_buffer() {
        let mut scene = empty_scene();
        let (l1, _) = scene.create_light(LightType::Directional);
        let (l2, _) = scene.create_light(LightType::Spot);
        let (l3, _) = scene.create_light(LightType::Point);

        // Enable shadows on all 3
        for lid in [l1, l2, l3] {
            if let NodeData::Light {
                ref mut cast_shadows,
                ..
            } = scene.nodes.get_mut(&lid).unwrap().data
            {
                *cast_shadows = true;
            }
        }
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert_eq!(count, 3);
        // All 3 lights should have cast_shadows=1.0 in the buffer
        // (the shader-side budget of max 2 is enforced in WGSL, not in Rust)
        let shadow_count = lights.iter().filter(|l| l.params[1] > 0.5).count();
        assert_eq!(
            shadow_count, 3,
            "all 3 lights should have cast_shadows=1.0 in buffer"
        );
    }

    #[test]
    fn shadow_softness_packed_correctly() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Spot);
        if let NodeData::Light {
            ref mut cast_shadows,
            ref mut shadow_softness,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cast_shadows = true;
            *shadow_softness = 32.5;
        }
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert!(
            (lights[0].params[2] - 32.5).abs() < 1e-5,
            "shadow softness should be 32.5"
        );
    }

    // -----------------------------------------------------------------------
    // Expression: time-based intensity override
    // -----------------------------------------------------------------------

    #[test]
    fn expression_overrides_intensity_at_time() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut intensity_expr,
            ref mut intensity,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *intensity = 5.0; // static value (should be overridden)
            *intensity_expr = Some("sin(t * 3.0) * 0.5 + 0.5".to_string());
        }
        // At t=0: sin(0)*0.5+0.5 = 0.5
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert!(
            (lights[0].direction_intensity[3] - 0.5).abs() < 1e-3,
            "intensity at t=0 should be 0.5, got {}",
            lights[0].direction_intensity[3]
        );
    }

    #[test]
    fn expression_color_hue_rotates_at_time() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut color,
            ref mut color_hue_expr,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *color = Vec3::new(1.0, 0.0, 0.0); // Red
            *color_hue_expr = Some("120.0".to_string()); // Constant 120° hue shift
        }
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        // Red shifted 120° → should be approximately green
        let r = lights[0].color_range[0];
        let g = lights[0].color_range[1];
        assert!(
            g > r,
            "120° hue shift of red should be more green than red, r={r}, g={g}"
        );
    }

    #[test]
    fn invalid_expression_falls_back_to_static_intensity() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut intensity_expr,
            ref mut intensity,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *intensity = 7.0;
            *intensity_expr = Some("invalid+++".to_string()); // Bad expression
        }
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None, 0.0);
        assert!(
            (lights[0].direction_intensity[3] - 7.0).abs() < 1e-3,
            "invalid expression should fall back to static intensity 7.0"
        );
    }
}
