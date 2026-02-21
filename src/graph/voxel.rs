use glam::{Vec2, Vec3, Vec3Swizzles};
use serde::{Deserialize, Serialize};

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};

pub const DEFAULT_RESOLUTION: u32 = 96;
const GRID_PADDING: f32 = 0.5;

pub fn default_resolution() -> u32 {
    DEFAULT_RESOLUTION
}
const FAR_DISTANCE: f32 = 999.0;

/// A 3D signed distance field stored as a flat array.
/// Layout: data[z * res * res + y * res + x]
/// material_ids uses the same layout and stores the NodeId (as f32) of the closest primitive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub data: Vec<f32>,
    #[serde(default)]
    pub material_ids: Vec<f32>,
}

impl VoxelGrid {
    pub fn new(resolution: u32, bounds_min: Vec3, bounds_max: Vec3) -> Self {
        let total = (resolution * resolution * resolution) as usize;
        Self {
            resolution,
            bounds_min,
            bounds_max,
            data: vec![FAR_DISTANCE; total],
            material_ids: vec![-1.0; total],
        }
    }

    pub fn index(x: u32, y: u32, z: u32, resolution: u32) -> usize {
        (z * resolution * resolution + y * resolution + x) as usize
    }

    pub fn total_floats(&self) -> usize {
        (self.resolution * self.resolution * self.resolution) as usize
    }

    /// Maps a local-space position to continuous grid coordinates [0, res-1].
    pub fn world_to_grid(&self, local_pos: Vec3) -> Vec3 {
        let norm = (local_pos - self.bounds_min) / (self.bounds_max - self.bounds_min);
        norm * (self.resolution - 1) as f32
    }

    /// Maps grid coordinates back to local-space position.
    pub fn grid_to_world(&self, gx: f32, gy: f32, gz: f32) -> Vec3 {
        let norm = Vec3::new(gx, gy, gz) / (self.resolution - 1) as f32;
        self.bounds_min + norm * (self.bounds_max - self.bounds_min)
    }

    /// Bit-exact equality check for undo system.
    pub fn content_eq(&self, other: &VoxelGrid) -> bool {
        self.resolution == other.resolution
            && self.bounds_min == other.bounds_min
            && self.bounds_max == other.bounds_max
            && self.data.len() == other.data.len()
            && self.data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.material_ids.len() == other.material_ids.len()
            && self.material_ids
                .iter()
                .zip(other.material_ids.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Nearest-neighbor material ID lookup at a local-space position.
    pub fn sample_material(&self, local_pos: Vec3) -> f32 {
        if self.material_ids.len() != self.data.len() {
            return -1.0;
        }
        let gc = self.world_to_grid(local_pos);
        let max_coord = (self.resolution - 1) as f32;
        let gc = gc.clamp(Vec3::ZERO, Vec3::splat(max_coord));
        let ix = gc.x.round() as u32;
        let iy = gc.y.round() as u32;
        let iz = gc.z.round() as u32;
        self.material_ids[Self::index(ix, iy, iz, self.resolution)]
    }

    /// CPU-side trilinear interpolation at a local-space position.
    pub fn sample(&self, local_pos: Vec3) -> f32 {
        let gc = self.world_to_grid(local_pos);
        let res = self.resolution;
        let max_coord = (res - 1) as f32;

        let gc = gc.clamp(Vec3::ZERO, Vec3::splat(max_coord));

        let ix0 = gc.x.floor() as u32;
        let iy0 = gc.y.floor() as u32;
        let iz0 = gc.z.floor() as u32;
        let ix1 = (ix0 + 1).min(res - 1);
        let iy1 = (iy0 + 1).min(res - 1);
        let iz1 = (iz0 + 1).min(res - 1);

        let fx = gc.x.fract();
        let fy = gc.y.fract();
        let fz = gc.z.fract();

        let c000 = self.data[Self::index(ix0, iy0, iz0, res)];
        let c100 = self.data[Self::index(ix1, iy0, iz0, res)];
        let c010 = self.data[Self::index(ix0, iy1, iz0, res)];
        let c110 = self.data[Self::index(ix1, iy1, iz0, res)];
        let c001 = self.data[Self::index(ix0, iy0, iz1, res)];
        let c101 = self.data[Self::index(ix1, iy0, iz1, res)];
        let c011 = self.data[Self::index(ix0, iy1, iz1, res)];
        let c111 = self.data[Self::index(ix1, iy1, iz1, res)];

        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        c0 + (c1 - c0) * fz
    }
}

// ---------------------------------------------------------------------------
// CPU-side SDF evaluation (mirrors WGSL functions exactly)
// ---------------------------------------------------------------------------

fn rotate_euler(p: Vec3, r: Vec3) -> Vec3 {
    let mut q = p;
    let (sx, cx) = r.x.sin_cos();
    q = Vec3::new(q.x, cx * q.y - sx * q.z, sx * q.y + cx * q.z);
    let (sy, cy) = r.y.sin_cos();
    q = Vec3::new(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
    let (sz, cz) = r.z.sin_cos();
    q = Vec3::new(cz * q.x - sz * q.y, sz * q.x + cz * q.y, q.z);
    q
}

fn evaluate_sdf(kind: &SdfPrimitive, p: Vec3, s: Vec3) -> f32 {
    match kind {
        SdfPrimitive::Sphere => p.length() - s.x,
        SdfPrimitive::Box => {
            let q = p.abs() - s;
            q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
        }
        SdfPrimitive::Cylinder => {
            let d_x = Vec2::new(p.xz().length(), p.y).abs() - Vec2::new(s.x, s.y);
            d_x.x.max(d_x.y).min(0.0) + d_x.max(Vec2::ZERO).length()
        }
        SdfPrimitive::Torus => {
            let q = Vec2::new(p.xz().length() - s.x, p.y);
            q.length() - s.y
        }
        SdfPrimitive::Plane => p.y,
        SdfPrimitive::Cone => {
            let q = Vec2::new(p.xz().length(), p.y);
            let tip = Vec2::new(0.0, s.y);
            let base = Vec2::new(s.x, 0.0);
            let ab = base - tip;
            let aq = q - tip;
            let t = ab.dot(aq) / ab.dot(ab);
            let t = t.clamp(0.0, 1.0);
            let closest = tip + ab * t;
            let d_side = (q - closest).length();
            let cross2d = ab.x * aq.y - ab.y * aq.x;
            let sign_val = if cross2d < 0.0 && q.y > 0.0 && q.y < s.y {
                -1.0
            } else {
                1.0
            };
            d_side * sign_val
        }
        SdfPrimitive::Capsule => {
            let h = s.y;
            let r = s.x;
            let py = p.y.clamp(-h, h);
            (p - Vec3::new(0.0, py, 0.0)).length() - r
        }
    }
}

// ---------------------------------------------------------------------------
// CSG operations (CPU mirrors of WGSL)
// ---------------------------------------------------------------------------

fn csg_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

fn csg_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k.max(0.0001)).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

fn csg_subtract(a: f32, b: f32) -> f32 {
    a.max(-b)
}

fn csg_intersect(a: f32, b: f32) -> f32 {
    a.max(b)
}

// ---------------------------------------------------------------------------
// Recursive SDF tree evaluator
// ---------------------------------------------------------------------------

/// Evaluate the combined SDF of a subtree at a world-space point.
pub fn evaluate_sdf_tree(scene: &Scene, node_id: NodeId, p: Vec3) -> f32 {
    let Some(node) = scene.nodes.get(&node_id) else {
        return FAR_DISTANCE;
    };
    match &node.data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            ..
        } => {
            let local_p = rotate_euler(p - *position, *rotation);
            evaluate_sdf(kind, local_p, *scale)
        }
        NodeData::Operation {
            op,
            smooth_k,
            left,
            right,
        } => {
            let a = left.map(|l| evaluate_sdf_tree(scene, l, p));
            let b = right.map(|r| evaluate_sdf_tree(scene, r, p));
            match (a, b) {
                (Some(a), Some(b)) => match op {
                    CsgOp::Union => csg_union(a, b),
                    CsgOp::SmoothUnion => csg_smooth_union(a, b, *smooth_k),
                    CsgOp::Subtract => csg_subtract(a, b),
                    CsgOp::Intersect => csg_intersect(a, b),
                },
                (Some(v), None) | (None, Some(v)) => v,
                (None, None) => f32::MAX,
            }
        }
        NodeData::Sculpt {
            position,
            rotation,
            voxel_grid,
            ..
        } => {
            let local_p = rotate_euler(p - *position, *rotation);
            voxel_grid.sample(local_p)
        }
        NodeData::Transform { kind, input, value } => {
            let Some(child_id) = input else {
                return FAR_DISTANCE;
            };
            let tp = match kind {
                TransformKind::Translate => p - *value,
                TransformKind::Rotate => rotate_euler(p, *value),
                TransformKind::Scale => p / *value,
            };
            let d = evaluate_sdf_tree(scene, *child_id, tp);
            match kind {
                TransformKind::Scale => d * value.min_element(),
                _ => d,
            }
        }
    }
}

/// Evaluate the combined SDF of a subtree at a world-space point,
/// returning (distance, material_node_id as f32).
/// Material tracks which primitive's NodeId is closest at each point.
pub fn evaluate_sdf_tree_with_material(scene: &Scene, node_id: NodeId, p: Vec3) -> (f32, f32) {
    let Some(node) = scene.nodes.get(&node_id) else {
        return (FAR_DISTANCE, -1.0);
    };
    match &node.data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            ..
        } => {
            let local_p = rotate_euler(p - *position, *rotation);
            let dist = evaluate_sdf(kind, local_p, *scale);
            (dist, node_id as f32)
        }
        NodeData::Operation {
            op,
            smooth_k,
            left,
            right,
        } => {
            let a = left.map(|l| evaluate_sdf_tree_with_material(scene, l, p));
            let b = right.map(|r| evaluate_sdf_tree_with_material(scene, r, p));
            match (a, b) {
                (Some((da, ma)), Some((db, mb))) => match op {
                    CsgOp::Union => {
                        if da < db { (da, ma) } else { (db, mb) }
                    }
                    CsgOp::SmoothUnion => {
                        let k = smooth_k.max(0.0001);
                        let h = (0.5 + 0.5 * (db - da) / k).clamp(0.0, 1.0);
                        let d = db + (da - db) * h - k * h * (1.0 - h);
                        let mat = if da < db { ma } else { mb };
                        (d, mat)
                    }
                    CsgOp::Subtract => {
                        let d = da.max(-db);
                        (d, ma)
                    }
                    CsgOp::Intersect => {
                        if da > db { (da, ma) } else { (db, mb) }
                    }
                },
                (Some(v), None) | (None, Some(v)) => v,
                (None, None) => (f32::MAX, -1.0),
            }
        }
        NodeData::Sculpt {
            position,
            rotation,
            voxel_grid,
            ..
        } => {
            let local_p = rotate_euler(p - *position, *rotation);
            let dist = voxel_grid.sample(local_p);
            let mat = voxel_grid.sample_material(local_p);
            let mat = if mat < 0.0 { node_id as f32 } else { mat };
            (dist, mat)
        }
        NodeData::Transform { kind, input, value } => {
            let Some(child_id) = input else {
                return (FAR_DISTANCE, -1.0);
            };
            let tp = match kind {
                TransformKind::Translate => p - *value,
                TransformKind::Rotate => rotate_euler(p, *value),
                TransformKind::Scale => p / *value,
            };
            let (d, mat) = evaluate_sdf_tree_with_material(scene, *child_id, tp);
            let d = match kind {
                TransformKind::Scale => d * value.min_element(),
                _ => d,
            };
            (d, mat)
        }
    }
}

// ---------------------------------------------------------------------------
// Bounds estimation
// ---------------------------------------------------------------------------

/// Estimate the world-space bounding box of a subtree.
fn collect_bounds(scene: &Scene, id: NodeId, all_min: &mut Vec3, all_max: &mut Vec3) {
    let Some(node) = scene.nodes.get(&id) else {
        return;
    };
    match &node.data {
        NodeData::Primitive {
            position, scale, ..
        } => {
            let extent = *scale * 1.5;
            *all_min = all_min.min(*position - extent);
            *all_max = all_max.max(*position + extent);
        }
        NodeData::Operation { left, right, .. } => {
            if let Some(l) = left { collect_bounds(scene, *l, all_min, all_max); }
            if let Some(r) = right { collect_bounds(scene, *r, all_min, all_max); }
        }
        NodeData::Sculpt {
            position,
            voxel_grid,
            ..
        } => {
            *all_min = all_min.min(*position + voxel_grid.bounds_min);
            *all_max = all_max.max(*position + voxel_grid.bounds_max);
        }
        NodeData::Transform { input, .. } => {
            if let Some(i) = input { collect_bounds(scene, *i, all_min, all_max); }
        }
    }
}

fn bounds_for_subtree(scene: &Scene, root_id: NodeId) -> (Vec3, Vec3) {
    let mut all_min = Vec3::splat(f32::MAX);
    let mut all_max = Vec3::splat(f32::MIN);
    collect_bounds(scene, root_id, &mut all_min, &mut all_max);
    let pad = Vec3::splat(GRID_PADDING);
    (all_min - pad, all_max + pad)
}

// ---------------------------------------------------------------------------
// Baking
// ---------------------------------------------------------------------------

/// Sample the analytical SDF of a primitive into a VoxelGrid.
pub fn bake_from_analytical(kind: &SdfPrimitive, scale: Vec3, resolution: u32) -> VoxelGrid {
    let extent = scale * (1.0 + GRID_PADDING);
    let bounds_min = -extent;
    let bounds_max = extent;

    let mut grid = VoxelGrid::new(resolution, bounds_min, bounds_max);
    let res = resolution;

    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let dist = evaluate_sdf(kind, pos, scale);
                grid.data[VoxelGrid::index(x, y, z, res)] = dist;
            }
        }
    }

    grid
}

/// Bake any subtree's SDF into a VoxelGrid.
/// Returns (grid in local space, center position in world space).
/// Material IDs store the NodeId (as f32) of the closest primitive at each voxel.
pub fn bake_subtree(scene: &Scene, subtree_root: NodeId, resolution: u32) -> (VoxelGrid, Vec3) {
    let (world_min, world_max) = bounds_for_subtree(scene, subtree_root);
    let center = (world_min + world_max) * 0.5;
    let half_extent = (world_max - world_min) * 0.5;

    let local_min = -half_extent;
    let local_max = half_extent;

    let mut grid = VoxelGrid::new(resolution, local_min, local_max);
    let res = resolution;

    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let local_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let world_pos = local_pos + center;
                let (dist, mat) = evaluate_sdf_tree_with_material(scene, subtree_root, world_pos);
                let idx = VoxelGrid::index(x, y, z, res);
                grid.data[idx] = dist;
                grid.material_ids[idx] = mat;
            }
        }
    }

    (grid, center)
}
