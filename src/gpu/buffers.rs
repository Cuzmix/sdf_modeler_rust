use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::graph::scene::{NodeData, NodeId, Scene};

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
                    scale: [scale.x, scale.y, scale.z, 0.0],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [0.0; 4],
                    extra1: [emissive.x, emissive.y, emissive.z, *emissive_intensity],
                    extra2: [*fresnel, 0.0, 0.0, 0.0],
                });
            }
            NodeData::Operation { op, smooth_k, steps, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [op.gpu_op_id(), *smooth_k, *steps, 0.0],
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
                    scale: [1.0, 1.0, 1.0, 0.0],
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
            NodeData::Light { color, intensity, range, spot_angle, light_type } => {
                let type_val = match light_type {
                    crate::graph::scene::LightType::Point => 50.0,
                    crate::graph::scene::LightType::Spot => 51.0,
                    crate::graph::scene::LightType::Directional => 52.0,
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
