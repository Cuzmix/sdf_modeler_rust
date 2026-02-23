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
    Smooth,
    Flatten,
    Inflate,
    Grab,
}

impl BrushMode {
    pub fn sign(&self) -> f32 {
        match self {
            Self::Add => -1.0,   // decrease distance = add material
            Self::Carve => 1.0,  // increase distance = remove material
            Self::Smooth => 0.0,
            Self::Flatten => 0.0,
            Self::Inflate => -1.0,
            Self::Grab => 0.0,
        }
    }

    /// GPU brush_mode encoding: 0=Add, 1=Carve, 2=Smooth, 3=Flatten, 4=Inflate.
    pub fn gpu_mode(&self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Carve => 1.0,
            Self::Smooth => 2.0,
            Self::Flatten => 3.0,
            Self::Inflate => 4.0,
            Self::Grab => 5.0,  // CPU-only, never dispatched to GPU
        }
    }
}

// ---------------------------------------------------------------------------
// Falloff types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum FalloffMode {
    Smooth,
    Linear,
    Sharp,
    Flat,
}

impl FalloffMode {
    /// GPU falloff_mode encoding: 0=Smooth, 1=Linear, 2=Sharp, 3=Flat.
    pub fn gpu_mode(&self) -> f32 {
        match self {
            Self::Smooth => 0.0,
            Self::Linear => 1.0,
            Self::Sharp => 2.0,
            Self::Flat => 3.0,
        }
    }

    /// Evaluate falloff at normalized distance nt in [0, 1).
    pub fn evaluate(&self, nt: f32) -> f32 {
        match self {
            Self::Smooth => 1.0 - nt * nt * (3.0 - 2.0 * nt),
            Self::Linear => 1.0 - nt,
            Self::Sharp => (1.0 - nt) * (1.0 - nt),
            Self::Flat => 1.0,
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
        falloff_mode: FalloffMode,
        smooth_iterations: u32,
        /// SDF value at brush center when Flatten drag started. Reset on mouse release.
        flatten_reference: Option<f32>,
        /// Lazy brush dead zone radius (0.0 = disabled).
        lazy_radius: f32,
        /// Surface constraint multiplier (0.0 = off). Attenuates brush near surface only.
        surface_constraint: f32,
        /// Mirror axis: None = off, Some(0) = X, Some(1) = Y, Some(2) = Z.
        symmetry_axis: Option<u8>,
        /// Snapshot of grid data for grab brush (cloned on grab start).
        grab_snapshot: Option<Vec<f32>>,
        /// World position where grab stroke started.
        grab_start: Option<Vec3>,
    },
}

impl SculptState {
    /// Create a new Active sculpt state with default brush settings.
    pub fn new_active(node_id: NodeId) -> Self {
        Self::Active {
            node_id,
            brush_mode: BrushMode::Add,
            brush_radius: DEFAULT_BRUSH_RADIUS,
            brush_strength: DEFAULT_BRUSH_STRENGTH,
            falloff_mode: FalloffMode::Sharp,
            smooth_iterations: 3,
            flatten_reference: None,
            lazy_radius: 0.0,
            surface_constraint: 0.0,
            symmetry_axis: None,
            grab_snapshot: None,
            grab_start: None,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    pub fn active_node(&self) -> Option<NodeId> {
        match self {
            Self::Active { node_id, .. } => Some(*node_id),
            _ => None,
        }
    }

    pub fn symmetry_axis(&self) -> Option<u8> {
        match self {
            Self::Active { symmetry_axis, .. } => *symmetry_axis,
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
    falloff_mode: &FalloffMode,
    smooth_iterations: u32,
    flatten_ref: f32,
    surface_constraint: f32,
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
        match brush_mode {
            BrushMode::Smooth => Some(apply_smooth_to_grid(
                voxel_grid,
                local_hit,
                brush_radius,
                brush_strength,
                falloff_mode,
                smooth_iterations,
                surface_constraint,
            )),
            _ => Some(apply_brush_to_grid(
                voxel_grid,
                local_hit,
                brush_mode,
                brush_radius,
                brush_strength,
                falloff_mode,
                flatten_ref,
                surface_constraint,
            )),
        }
    } else {
        None
    }
}

/// Compute surface constraint factor for a voxel value.
fn surface_factor(voxel_val: f32, radius: f32, constraint: f32) -> f32 {
    if constraint > 0.0 {
        let threshold = radius * constraint;
        1.0 - (voxel_val.abs() / threshold).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Returns (z0, z1) inclusive range of z-slabs that were modified.
fn apply_brush_to_grid(
    grid: &mut VoxelGrid,
    center: Vec3,
    brush_mode: &BrushMode,
    radius: f32,
    strength: f32,
    falloff_mode: &FalloffMode,
    flatten_ref: f32,
    surface_constraint: f32,
) -> (u32, u32) {
    let res = grid.resolution;

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
                    let nt = dist / radius;
                    let falloff = falloff_mode.evaluate(nt);
                    let idx = VoxelGrid::index(x, y, z, res);
                    match brush_mode {
                        BrushMode::Add | BrushMode::Carve => {
                            let sf = surface_factor(grid.data[idx], radius, surface_constraint);
                            grid.data[idx] += brush_mode.sign() * strength * falloff * sf;
                        }
                        BrushMode::Flatten => {
                            let sf = surface_factor(grid.data[idx], radius, surface_constraint);
                            grid.data[idx] +=
                                (flatten_ref - grid.data[idx]) * falloff * strength * sf;
                        }
                        BrushMode::Inflate => {
                            // Inflate = Add with implicit surface constraint (always on)
                            let threshold = radius * 0.5;
                            let sf = 1.0 - (grid.data[idx].abs() / threshold).clamp(0.0, 1.0);
                            grid.data[idx] += -1.0 * strength * falloff * sf;
                        }
                        BrushMode::Smooth | BrushMode::Grab => unreachable!(),
                    }
                }
            }
        }
    }

    (z0, z1)
}

/// Apply Laplacian smoothing within the brush sphere.
fn apply_smooth_to_grid(
    grid: &mut VoxelGrid,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff_mode: &FalloffMode,
    iterations: u32,
    surface_constraint: f32,
) -> (u32, u32) {
    let res = grid.resolution;

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

    for _ in 0..iterations {
        let snapshot = grid.data.clone();

        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let world_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                    let dist = (world_pos - center).length();
                    if dist >= radius {
                        continue;
                    }
                    let nt = dist / radius;
                    let falloff = falloff_mode.evaluate(nt);

                    // 6-neighbor Laplacian average (clamped at grid edges)
                    let xm = if x > 0 { x - 1 } else { x };
                    let xp = if x < res - 1 { x + 1 } else { x };
                    let ym = if y > 0 { y - 1 } else { y };
                    let yp = if y < res - 1 { y + 1 } else { y };
                    let zm = if z > 0 { z - 1 } else { z };
                    let zp = if z < res - 1 { z + 1 } else { z };

                    let avg = (snapshot[VoxelGrid::index(xm, y, z, res)]
                        + snapshot[VoxelGrid::index(xp, y, z, res)]
                        + snapshot[VoxelGrid::index(x, ym, z, res)]
                        + snapshot[VoxelGrid::index(x, yp, z, res)]
                        + snapshot[VoxelGrid::index(x, y, zm, res)]
                        + snapshot[VoxelGrid::index(x, y, zp, res)])
                        / 6.0;

                    let idx = VoxelGrid::index(x, y, z, res);
                    let current = snapshot[idx];
                    let sf = surface_factor(current, radius, surface_constraint);
                    grid.data[idx] = current + (avg - current) * falloff * strength * sf;
                }
            }
        }
    }

    (z0, z1)
}

/// Apply grab brush: shift surface content by sampling from a snapshot at an offset position.
/// Returns (z0, z1) inclusive range of z-slabs modified.
pub fn apply_grab_to_grid(
    grid: &mut VoxelGrid,
    snapshot: &[f32],
    center: Vec3,
    radius: f32,
    strength: f32,
    grab_delta: Vec3,
    falloff_mode: &FalloffMode,
) -> (u32, u32) {
    let res = grid.resolution;
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

    let max_c = (res - 1) as f32;

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let world_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let dist = (world_pos - center).length();
                if dist >= radius {
                    continue;
                }
                let nt = dist / radius;
                let falloff = falloff_mode.evaluate(nt);

                // Sample from snapshot at offset position
                let sample_pos = world_pos - grab_delta * falloff * strength;
                let gc = grid.world_to_grid(sample_pos);

                // Trilinear interpolation from snapshot
                let gx = gc.x.clamp(0.0, max_c);
                let gy = gc.y.clamp(0.0, max_c);
                let gz = gc.z.clamp(0.0, max_c);

                let ix0 = gx.floor() as u32;
                let iy0 = gy.floor() as u32;
                let iz0 = gz.floor() as u32;
                let ix1 = (ix0 + 1).min(res - 1);
                let iy1 = (iy0 + 1).min(res - 1);
                let iz1 = (iz0 + 1).min(res - 1);
                let fx = gx.fract();
                let fy = gy.fract();
                let fz = gz.fract();

                let c000 = snapshot[VoxelGrid::index(ix0, iy0, iz0, res)];
                let c100 = snapshot[VoxelGrid::index(ix1, iy0, iz0, res)];
                let c010 = snapshot[VoxelGrid::index(ix0, iy1, iz0, res)];
                let c110 = snapshot[VoxelGrid::index(ix1, iy1, iz0, res)];
                let c001 = snapshot[VoxelGrid::index(ix0, iy0, iz1, res)];
                let c101 = snapshot[VoxelGrid::index(ix1, iy0, iz1, res)];
                let c011 = snapshot[VoxelGrid::index(ix0, iy1, iz1, res)];
                let c111 = snapshot[VoxelGrid::index(ix1, iy1, iz1, res)];

                let c00 = c000 + (c100 - c000) * fx;
                let c10 = c010 + (c110 - c010) * fx;
                let c01 = c001 + (c101 - c001) * fx;
                let c11 = c011 + (c111 - c011) * fx;
                let c0 = c00 + (c10 - c00) * fy;
                let c1 = c01 + (c11 - c01) * fy;
                let sampled = c0 + (c1 - c0) * fz;

                let idx = VoxelGrid::index(x, y, z, res);
                grid.data[idx] = sampled;
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
