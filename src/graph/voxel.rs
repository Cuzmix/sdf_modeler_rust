use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};

pub const DEFAULT_RESOLUTION: u32 = 96;
const GRID_PADDING: f32 = 0.5;

pub fn default_resolution() -> u32 {
    DEFAULT_RESOLUTION
}
const FAR_DISTANCE: f32 = 999.0;

/// Return the max voxel resolution of any Sculpt node in a subtree.
/// Falls back to DEFAULT_RESOLUTION if no Sculpt nodes exist.
pub fn max_subtree_resolution(scene: &Scene, root: NodeId) -> u32 {
    let mut max_res = 0u32;
    let mut stack = vec![root];
    let mut visited = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) { continue; }
        if let Some(node) = scene.nodes.get(&id) {
            if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                max_res = max_res.max(voxel_grid.resolution);
            }
            stack.extend(node.data.children());
        }
    }
    if max_res == 0 { DEFAULT_RESOLUTION } else { max_res }
}

/// A 3D signed distance field stored as a flat array.
/// Layout: data[z * res * res + y * res + x]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    #[serde(with = "sparse_voxel_data")]
    pub data: Vec<f32>,
}

/// Custom serde module: serializes voxel data sparsely when beneficial (skipping FAR_DISTANCE
/// voxels), and deserializes both old dense format and new sparse format.
mod sparse_voxel_data {
    use super::FAR_DISTANCE;
    use serde::de::{self, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    #[derive(serde::Serialize)]
    struct SparseRepr {
        total: usize,
        entries: Vec<(u32, f32)>,
    }

    pub fn serialize<S: Serializer>(data: &Vec<f32>, serializer: S) -> Result<S::Ok, S::Error> {
        let non_far: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .filter(|(_, &d)| (d - FAR_DISTANCE).abs() > 0.001)
            .map(|(i, &d)| (i as u32, d))
            .collect();

        // Use sparse if it saves space: each sparse entry = (u32, f32) = 2 values
        // vs dense = 1 value per voxel. Sparse wins when non_far * 2 < total.
        if non_far.len() * 2 < data.len() {
            let repr = SparseRepr { total: data.len(), entries: non_far };
            repr.serialize(serializer)
        } else {
            data.serialize(serializer)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        deserializer.deserialize_any(VoxelDataVisitor)
    }

    struct VoxelDataVisitor;

    impl<'de> Visitor<'de> for VoxelDataVisitor {
        type Value = Vec<f32>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a dense array of f32 or a sparse {total, entries} object")
        }

        // Old format: plain array of f32
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<f32>, A::Error> {
            let mut data = Vec::with_capacity(seq.size_hint().unwrap_or(0));
            while let Some(val) = seq.next_element::<f32>()? {
                data.push(val);
            }
            Ok(data)
        }

        // New sparse format: { "total": N, "entries": [[idx, val], ...] }
        fn visit_map<M: de::MapAccess<'de>>(self, mut map: M) -> Result<Vec<f32>, M::Error> {
            let mut total: Option<usize> = None;
            let mut entries: Option<Vec<(u32, f32)>> = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "total" => total = Some(map.next_value()?),
                    "entries" => entries = Some(map.next_value()?),
                    _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
                }
            }

            let total = total.ok_or_else(|| de::Error::missing_field("total"))?;
            let entries = entries.ok_or_else(|| de::Error::missing_field("entries"))?;

            let mut data = vec![FAR_DISTANCE; total];
            for (idx, val) in entries {
                if (idx as usize) < data.len() {
                    data[idx as usize] = val;
                }
            }
            Ok(data)
        }
    }
}

impl VoxelGrid {
    pub fn new(resolution: u32, bounds_min: Vec3, bounds_max: Vec3) -> Self {
        let total = (resolution * resolution * resolution) as usize;
        Self {
            resolution,
            bounds_min,
            bounds_max,
            data: vec![FAR_DISTANCE; total],
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
    let res = resolution;
    let res_f = (res - 1) as f32;
    let size = bounds_max - bounds_min;

    let data: Vec<f32> = (0..res)
        .into_par_iter()
        .flat_map(|z| {
            let mut slice = Vec::with_capacity((res * res) as usize);
            for y in 0..res {
                for x in 0..res {
                    let norm = Vec3::new(x as f32, y as f32, z as f32) / res_f;
                    let pos = bounds_min + norm * size;
                    slice.push(evaluate_sdf(kind, pos, scale));
                }
            }
            slice
        })
        .collect();

    VoxelGrid { resolution, bounds_min, bounds_max, data }
}

/// Bake any subtree's SDF into a VoxelGrid.
/// Returns (grid in local space, center position in world space).
pub fn bake_subtree(scene: &Scene, subtree_root: NodeId, resolution: u32) -> (VoxelGrid, Vec3) {
    let (world_min, world_max) = bounds_for_subtree(scene, subtree_root);
    let center = (world_min + world_max) * 0.5;
    let half_extent = (world_max - world_min) * 0.5;

    let local_min = -half_extent;
    let local_max = half_extent;
    let res = resolution;
    let res_f = (res - 1) as f32;
    let size = local_max - local_min;

    let data: Vec<f32> = (0..res)
        .into_par_iter()
        .flat_map(|z| {
            let mut slice = Vec::with_capacity((res * res) as usize);
            for y in 0..res {
                for x in 0..res {
                    let norm = Vec3::new(x as f32, y as f32, z as f32) / res_f;
                    let local_pos = local_min + norm * size;
                    let world_pos = local_pos + center;
                    slice.push(evaluate_sdf_tree(scene, subtree_root, world_pos));
                }
            }
            slice
        })
        .collect();

    let grid = VoxelGrid { resolution, bounds_min: local_min, bounds_max: local_max, data };
    (grid, center)
}

/// Same as `bake_subtree` but reports progress via an atomic counter (incremented per z-slice).
pub fn bake_subtree_with_progress(
    scene: &Scene,
    subtree_root: NodeId,
    resolution: u32,
    progress: Arc<AtomicU32>,
) -> (VoxelGrid, Vec3) {
    let (world_min, world_max) = bounds_for_subtree(scene, subtree_root);
    let center = (world_min + world_max) * 0.5;
    let half_extent = (world_max - world_min) * 0.5;

    let local_min = -half_extent;
    let local_max = half_extent;
    let res = resolution;
    let res_f = (res - 1) as f32;
    let size = local_max - local_min;

    let data: Vec<f32> = (0..res)
        .into_par_iter()
        .flat_map(|z| {
            let mut slice = Vec::with_capacity((res * res) as usize);
            for y in 0..res {
                for x in 0..res {
                    let norm = Vec3::new(x as f32, y as f32, z as f32) / res_f;
                    let local_pos = local_min + norm * size;
                    let world_pos = local_pos + center;
                    slice.push(evaluate_sdf_tree(scene, subtree_root, world_pos));
                }
            }
            progress.fetch_add(1, Ordering::Relaxed);
            slice
        })
        .collect();

    let grid = VoxelGrid { resolution, bounds_min: local_min, bounds_max: local_max, data };
    (grid, center)
}
