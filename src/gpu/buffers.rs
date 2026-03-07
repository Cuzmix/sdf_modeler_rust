use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::graph::scene::{LightType, NodeData, NodeId, Scene, MAX_SCENE_LIGHTS};

/// 128-byte GPU node (8 x vec4f).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SdfNodeGpu {
    pub type_op: [f32; 4],   // [type_val, smooth_k, metallic, roughness]
    pub position: [f32; 4],  // [x, y, z, 0]
    pub rotation: [f32; 4],  // [rx, ry, rz, 0] (radians)
    pub scale: [f32; 4],     // [sx, sy, sz, 0]
    pub color: [f32; 4],     // [r, g, b, is_selected]
    pub extra0: [f32; 4],    // prim: [0,0,0,0]; sculpt: [voxel_offset, resolution, emissive.x, emissive.y]
    pub extra1: [f32; 4],    // prim: [emissive.xyz, emissive_intensity]; sculpt: [bounds_min.xyz, emissive.z]
    pub extra2: [f32; 4],    // prim: [fresnel,0,0,0]; sculpt: [bounds_max.xyz, fresnel]
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
        let is_sel = if selected_set.contains(&node_id) { 1.0 } else { 0.0 };
        let light_mask = scene.get_light_mask(node_id) as f32;

        match &node.data {
            NodeData::Primitive {
                kind,
                position,
                rotation,
                scale,
                color,
                metallic,
                roughness,
                emissive,
                emissive_intensity,
                fresnel,
                ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [kind.gpu_type_id(), 0.0, *metallic, *roughness],
                    position: [position.x, position.y, position.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, light_mask],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [0.0; 4],
                    extra1: [emissive.x, emissive.y, emissive.z, *emissive_intensity],
                    extra2: [*fresnel, 0.0, 0.0, 0.0],
                });
            }
            NodeData::Operation { op, smooth_k, steps, color_blend, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [op.gpu_op_id(), *smooth_k, *steps, *color_blend],
                    position: [0.0; 4],
                    rotation: [0.0; 4],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                });
            }
            NodeData::Sculpt {
                position,
                rotation,
                color,
                metallic,
                roughness,
                emissive,
                emissive_intensity,
                fresnel,
                layer_intensity,
                voxel_grid,
                ..
            } => {
                let offset = voxel_offsets.get(&node_id).copied().unwrap_or(0);
                buffer.push(SdfNodeGpu {
                    type_op: [20.0, *emissive_intensity, *metallic, *roughness],
                    position: [position.x, position.y, position.z, *layer_intensity],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [1.0, 1.0, 1.0, light_mask],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [offset as f32, voxel_grid.resolution as f32, emissive.x, emissive.y],
                    extra1: [voxel_grid.bounds_min.x, voxel_grid.bounds_min.y, voxel_grid.bounds_min.z, emissive.z],
                    extra2: [voxel_grid.bounds_max.x, voxel_grid.bounds_max.y, voxel_grid.bounds_max.z, *fresnel],
                });
            }
            NodeData::Transform { translation, rotation, scale, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [21.0, 0.0, 0.0, 0.0],
                    position: [translation.x, translation.y, translation.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                });
            }
            NodeData::Modifier { kind, value, extra, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [kind.gpu_type_id(), 0.0, 0.0, 0.0],
                    position: [value.x, value.y, value.z, 0.0],
                    rotation: [extra.x, extra.y, extra.z, 0.0],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                });
            }
            NodeData::Light { color, intensity, range, spot_angle, light_type, .. } => {
                let type_val = match light_type {
                    crate::graph::scene::LightType::Point => 50.0,
                    crate::graph::scene::LightType::Spot => 51.0,
                    crate::graph::scene::LightType::Directional => 52.0,
                    crate::graph::scene::LightType::Ambient => 53.0,
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
    /// x = volumetric (0.0 = off, 1.0 = on), y = volumetric_density, zw = reserved
    pub volumetric: [f32; 4],
}

/// Collected ambient contribution from scene Ambient light nodes.
#[derive(Default)]
pub struct SceneAmbient {
    /// Total ambient color (sum of all visible Ambient lights: color * intensity).
    pub color: glam::Vec3,
}

/// Collect visible Light nodes from the scene, sorted by distance to camera (nearest first).
/// Returns (directional/point/spot count, gpu lights, accumulated ambient).
/// Ambient-type lights contribute to the returned `SceneAmbient` instead of the light array.
pub fn collect_scene_lights(
    scene: &Scene,
    camera_pos: glam::Vec3,
    soloed_light: Option<NodeId>,
) -> (u32, Vec<SceneLightGpu>, SceneAmbient) {
    let parent_map = scene.build_parent_map();
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

            // Ambient lights accumulate into scene ambient — no position/direction needed
            if *light_type == LightType::Ambient {
                // Suppress ambient when soloing a non-ambient light
                if soloed_light.is_none() {
                    ambient.color += *color * *intensity;
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
                LightType::Ambient => unreachable!(), // handled above
            };

            let half_angle_rad = (spot_angle * 0.5).to_radians();
            let cos_half_angle = half_angle_rad.cos();

            let dist_to_camera = (*translation - camera_pos).length();

            lights.push((
                dist_to_camera,
                SceneLightGpu {
                    position_type: [translation.x, translation.y, translation.z, type_val],
                    direction_intensity: [direction.x, direction.y, direction.z, *intensity],
                    color_range: [color.x, color.y, color.z, *range],
                    params: [
                        cos_half_angle,
                        if *cast_shadows { 1.0 } else { 0.0 },
                        *shadow_softness,
                        pack_rgb8(*shadow_color),
                    ],
                    volumetric: [
                        if *volumetric { 1.0 } else { 0.0 },
                        *volumetric_density,
                        0.0,
                        0.0,
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

    // -----------------------------------------------------------------------
    // collect_scene_lights
    // -----------------------------------------------------------------------

    #[test]
    fn collect_scene_lights_empty_scene_returns_zero() {
        let scene = empty_scene();
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 0);
        assert!(lights.is_empty());
    }

    #[test]
    fn collect_scene_lights_single_point_light() {
        let mut scene = empty_scene();
        let (_light_id, _transform_id) = scene.create_light(LightType::Point);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
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
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 1);
        // Spot light type = 1.0
        assert!((lights[0].position_type[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn collect_scene_lights_directional_type() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Directional);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
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
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
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
        let (count, lights, _ambient) = collect_scene_lights(&scene, camera_pos, None);
        assert_eq!(count, 2);
        // First light should be the nearest one (at x=1)
        assert!((lights[0].position_type[0] - 1.0).abs() < 1e-5,
            "nearest light should be first, got x={}", lights[0].position_type[0]);
        // Second light should be farther (at x=10)
        assert!((lights[1].position_type[0] - 10.0).abs() < 1e-5,
            "farther light should be second, got x={}", lights[1].position_type[0]);
    }

    #[test]
    fn collect_scene_lights_hidden_lights_excluded() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        scene.hidden_nodes.insert(light_id);
        let (count, lights, _ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 0);
        assert!(lights.is_empty());
    }

    #[test]
    fn collect_scene_lights_ambient_contributes_to_scene_ambient() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Ambient);
        // Set intensity on the ambient light
        for node in scene.nodes.values_mut() {
            if let NodeData::Light { light_type, intensity, .. } = &mut node.data {
                if *light_type == LightType::Ambient {
                    *intensity = 0.1;
                }
            }
        }
        let (count, lights, ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
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
        let (_, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None);
        // cos(90/2 degrees) = cos(45 degrees) = sqrt(2)/2 ≈ 0.7071
        let expected_cos = (45.0_f32.to_radians()).cos();
        assert!((lights[0].params[0] - expected_cos).abs() < 1e-3,
            "spot angle cosine encoding: expected {expected_cos}, got {}", lights[0].params[0]);
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
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 1);
        // Negative intensity must be preserved in the GPU buffer (direction_intensity.w)
        assert!((lights[0].direction_intensity[3] - (-3.5)).abs() < 1e-5,
            "negative intensity must be preserved: expected -3.5, got {}", lights[0].direction_intensity[3]);
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
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None);
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
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 1);
        // Point lights default to cast_shadows=false
        assert!(lights[0].params[1] < 0.5, "point light should default to no shadows");
    }

    #[test]
    fn solo_light_filters_to_single_light() {
        let mut scene = empty_scene();
        let (light_a, _) = scene.create_light(LightType::Point);
        let (light_b, _) = scene.create_light(LightType::Spot);
        let (_light_c, _) = scene.create_light(LightType::Directional);

        // Without solo: all 3 lights returned
        let (count, _, _) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert_eq!(count, 3);

        // Solo light A: only 1 light returned
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, Some(light_a));
        assert_eq!(count, 1);
        // Point light type = 0.0
        assert!((lights[0].position_type[3] - 0.0).abs() < 0.01);

        // Solo light B: only 1 light returned (Spot)
        let (count, lights, _) = collect_scene_lights(&scene, Vec3::ZERO, Some(light_b));
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
            if let NodeData::Light { ref mut color, ref mut intensity, .. } = node.data {
                *color = Vec3::new(1.0, 0.5, 0.2);
                *intensity = 2.0;
            }
        }

        // Without solo: ambient contributes
        let (_, _, ambient) = collect_scene_lights(&scene, Vec3::ZERO, None);
        assert!(ambient.color.length() > 0.0);

        // Solo the point light: ambient suppressed
        let (_, _, ambient) = collect_scene_lights(&scene, Vec3::ZERO, Some(point_id));
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
}
