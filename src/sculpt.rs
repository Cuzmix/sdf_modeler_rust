use glam::Vec3;

use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel::VoxelGrid;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_BRUSH_RADIUS: f32 = 0.3;
pub const DEFAULT_BRUSH_STRENGTH: f32 = 0.05;

// ---------------------------------------------------------------------------
// Brush types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum BrushMode {
    Add,
    Carve,
}

impl BrushMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Carve => "Carve",
        }
    }

    fn sign(&self) -> f32 {
        match self {
            Self::Add => -1.0,   // decrease distance = add material
            Self::Carve => 1.0,  // increase distance = remove material
        }
    }
}

// ---------------------------------------------------------------------------
// Sculpt state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum SculptState {
    Inactive,
    Active {
        node_id: NodeId,
        brush_mode: BrushMode,
        brush_radius: f32,
        brush_strength: f32,
    },
}

impl SculptState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    pub fn active_node(&self) -> Option<NodeId> {
        match self {
            Self::Active { node_id, .. } => Some(*node_id),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Brush application
// ---------------------------------------------------------------------------

/// Apply a spherical brush stroke at `hit_world` position.
/// Returns the (z0, z1) inclusive dirty z-slab range for incremental GPU upload.
pub fn apply_brush(
    scene: &mut Scene,
    node_id: NodeId,
    hit_world: Vec3,
    brush_mode: &BrushMode,
    brush_radius: f32,
    brush_strength: f32,
) -> Option<(u32, u32)> {
    // Read transform to convert hit point to local space
    let (position, rotation) = match scene.nodes.get(&node_id).map(|n| &n.data) {
        Some(NodeData::Sculpt {
            position, rotation, ..
        }) => (*position, *rotation),
        _ => return None,
    };

    let local_hit = inverse_rotate_euler(hit_world - position, rotation);

    // Get mutable reference to grid and apply brush
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return None;
    };
    if let NodeData::Sculpt {
        ref mut voxel_grid, ..
    } = node.data
    {
        Some(apply_brush_to_grid(voxel_grid, local_hit, brush_mode, brush_radius, brush_strength))
    } else {
        None
    }
}

/// Returns (z0, z1) inclusive range of z-slabs that were modified.
fn apply_brush_to_grid(
    grid: &mut VoxelGrid,
    center: Vec3,
    brush_mode: &BrushMode,
    radius: f32,
    strength: f32,
) -> (u32, u32) {
    let res = grid.resolution;
    let sign = brush_mode.sign();

    // Compute grid-space bounding box of the brush sphere
    let brush_min = center - Vec3::splat(radius);
    let brush_max = center + Vec3::splat(radius);
    let g_min = grid.world_to_grid(brush_min);
    let g_max = grid.world_to_grid(brush_max);

    let x0 = (g_min.x.floor().max(0.0) as u32).min(res - 1);
    let y0 = (g_min.y.floor().max(0.0) as u32).min(res - 1);
    let z0 = (g_min.z.floor().max(0.0) as u32).min(res - 1);
    let x1 = (g_max.x.ceil().max(0.0) as u32).min(res - 1);
    let y1 = (g_max.y.ceil().max(0.0) as u32).min(res - 1);
    let z1 = (g_max.z.ceil().max(0.0) as u32).min(res - 1);

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let world_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let dist = (world_pos - center).length();
                if dist < radius {
                    let falloff = 1.0 - (dist / radius);
                    let falloff = falloff * falloff; // quadratic for smoother feel
                    let delta = sign * strength * falloff;
                    let idx = VoxelGrid::index(x, y, z, res);
                    grid.data[idx] += delta;
                }
            }
        }
    }

    (z0, z1)
}

/// Inverse of rotate_euler: undo Z rotation, then Y, then X.
pub fn inverse_rotate_euler(p: Vec3, r: Vec3) -> Vec3 {
    let mut q = p;
    // Inverse Z rotation
    let (sz, cz) = r.z.sin_cos();
    q = Vec3::new(cz * q.x + sz * q.y, -sz * q.x + cz * q.y, q.z);
    // Inverse Y rotation
    let (sy, cy) = r.y.sin_cos();
    q = Vec3::new(cy * q.x - sy * q.z, q.y, sy * q.x + cy * q.z);
    // Inverse X rotation
    let (sx, cx) = r.x.sin_cos();
    q = Vec3::new(q.x, cx * q.y + sx * q.z, -sx * q.y + cx * q.z);
    q
}
