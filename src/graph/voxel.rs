use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec3Swizzles};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use crate::compat::maybe_par_iter;
use serde::{Deserialize, Serialize};

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};

pub const DEFAULT_RESOLUTION: u32 = 96;
const GRID_PADDING: f32 = 0.5;

pub fn default_resolution() -> u32 {
    DEFAULT_RESOLUTION
}
const FAR_DISTANCE: f32 = 999.0;

// ---------------------------------------------------------------------------
// CPU-side noise functions matching the WGSL noise.wgsl implementation
// ---------------------------------------------------------------------------

/// Hash-based pseudo-random: maps Vec3 to Vec3 (matches WGSL hash33).
/// The magic constant matches the WGSL version exactly for GPU/CPU consistency.
#[allow(clippy::excessive_precision)]
fn hash33(p: Vec3) -> Vec3 {
    let q = Vec3::new(
        p.dot(Vec3::new(127.1, 311.7, 74.7)),
        p.dot(Vec3::new(269.5, 183.3, 246.1)),
        p.dot(Vec3::new(113.5, 271.9, 124.6)),
    );
    Vec3::new(
        (q.x.sin() * 43758.5453123).fract() * 2.0 - 1.0,
        (q.y.sin() * 43758.5453123).fract() * 2.0 - 1.0,
        (q.z.sin() * 43758.5453123).fract() * 2.0 - 1.0,
    )
}

/// Quintic interpolation curve: 6t^5 - 15t^4 + 10t^3 (C2 continuous).
fn quintic(t: Vec3) -> Vec3 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 3D gradient noise with trilinear interpolation (matches WGSL noise3d).
fn noise3d(p: Vec3) -> f32 {
    let cell = p.floor();
    let local = p - cell;
    let fade = quintic(local);

    let g000 = hash33(cell + Vec3::new(0.0, 0.0, 0.0)).dot(local - Vec3::new(0.0, 0.0, 0.0));
    let g100 = hash33(cell + Vec3::new(1.0, 0.0, 0.0)).dot(local - Vec3::new(1.0, 0.0, 0.0));
    let g010 = hash33(cell + Vec3::new(0.0, 1.0, 0.0)).dot(local - Vec3::new(0.0, 1.0, 0.0));
    let g110 = hash33(cell + Vec3::new(1.0, 1.0, 0.0)).dot(local - Vec3::new(1.0, 1.0, 0.0));
    let g001 = hash33(cell + Vec3::new(0.0, 0.0, 1.0)).dot(local - Vec3::new(0.0, 0.0, 1.0));
    let g101 = hash33(cell + Vec3::new(1.0, 0.0, 1.0)).dot(local - Vec3::new(1.0, 0.0, 1.0));
    let g011 = hash33(cell + Vec3::new(0.0, 1.0, 1.0)).dot(local - Vec3::new(0.0, 1.0, 1.0));
    let g111 = hash33(cell + Vec3::new(1.0, 1.0, 1.0)).dot(local - Vec3::new(1.0, 1.0, 1.0));

    fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

    let mix_x0 = lerp(g000, g100, fade.x);
    let mix_x1 = lerp(g010, g110, fade.x);
    let mix_x2 = lerp(g001, g101, fade.x);
    let mix_x3 = lerp(g011, g111, fade.x);

    let mix_y0 = lerp(mix_x0, mix_x1, fade.y);
    let mix_y1 = lerp(mix_x2, mix_x3, fade.y);

    lerp(mix_y0, mix_y1, fade.z)
}

/// Fractal Brownian Motion returning 3D displacement (matches WGSL fbm_noise).
fn fbm_noise(p: Vec3, frequency: f32, amplitude: f32, octaves: i32) -> Vec3 {
    let max_octaves = octaves.min(8);
    let mut result = Vec3::ZERO;
    let mut freq = frequency;
    let mut amp = amplitude;

    let offset_y = Vec3::new(31.416, 67.281, 11.513);
    let offset_z = Vec3::new(73.156, 19.874, 53.129);

    for _ in 0..max_octaves {
        let sample_pos = p * freq;
        result.x += noise3d(sample_pos) * amp;
        result.y += noise3d(sample_pos + offset_y) * amp;
        result.z += noise3d(sample_pos + offset_z) * amp;

        freq *= 2.0;
        amp *= 0.5;
    }

    result
}

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
///
/// Two modes:
/// - `is_displacement = false` (total SDF): data stores complete distance values, fill = FAR_DISTANCE
/// - `is_displacement = true` (differential): data stores displacement from analytical base, fill = 0.0
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    #[serde(default)]
    pub is_displacement: bool,
    #[serde(with = "sparse_voxel_data")]
    pub data: Vec<f32>,
}

/// Custom serde module: serializes voxel data sparsely when beneficial.
/// Supports both total-SDF grids (fill = FAR_DISTANCE) and displacement grids (fill = 0.0).
/// Auto-detects the fill value based on data content.
mod sparse_voxel_data {
    use super::FAR_DISTANCE;
    use serde::de::{self, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    #[derive(serde::Serialize)]
    struct SparseRepr {
        total: usize,
        fill: f32,
        entries: Vec<(u32, f32)>,
    }

    pub fn serialize<S: Serializer>(data: &Vec<f32>, serializer: S) -> Result<S::Ok, S::Error> {
        // Auto-detect fill: count values near FAR_DISTANCE vs near 0.0
        let near_far = data.iter().filter(|&&d| (d - FAR_DISTANCE).abs() < 0.001).count();
        let near_zero = data.iter().filter(|&&d| d.abs() < 0.001).count();
        let fill = if near_far >= near_zero { FAR_DISTANCE } else { 0.0 };

        let non_fill: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .filter(|(_, &d)| (d - fill).abs() > 0.001)
            .map(|(i, &d)| (i as u32, d))
            .collect();

        // Use sparse if it saves space: each sparse entry = (u32, f32) = 2 values
        // vs dense = 1 value per voxel. Sparse wins when non_fill * 2 < total.
        if non_fill.len() * 2 < data.len() {
            let repr = SparseRepr { total: data.len(), fill, entries: non_fill };
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
            f.write_str("a dense array of f32 or a sparse {total, fill?, entries} object")
        }

        // Old format: plain array of f32
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<f32>, A::Error> {
            let mut data = Vec::with_capacity(seq.size_hint().unwrap_or(0));
            while let Some(val) = seq.next_element::<f32>()? {
                data.push(val);
            }
            Ok(data)
        }

        // Sparse format: { "total": N, "fill"?: F, "entries": [[idx, val], ...] }
        fn visit_map<M: de::MapAccess<'de>>(self, mut map: M) -> Result<Vec<f32>, M::Error> {
            let mut total: Option<usize> = None;
            let mut fill: Option<f32> = None;
            let mut entries: Option<Vec<(u32, f32)>> = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "total" => total = Some(map.next_value()?),
                    "fill" => fill = Some(map.next_value()?),
                    "entries" => entries = Some(map.next_value()?),
                    _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
                }
            }

            let total = total.ok_or_else(|| de::Error::missing_field("total"))?;
            let entries = entries.ok_or_else(|| de::Error::missing_field("entries"))?;
            let fill = fill.unwrap_or(FAR_DISTANCE); // backward compat: old files used FAR_DISTANCE

            let mut data = vec![fill; total];
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
    /// Create a displacement grid (fill = 0.0). Used for sculpts with an analytical child.
    /// O(1) — no per-voxel SDF evaluation needed.
    pub fn new_displacement(resolution: u32, bounds_min: Vec3, bounds_max: Vec3) -> Self {
        let total = (resolution * resolution * resolution) as usize;
        Self {
            resolution,
            bounds_min,
            bounds_max,
            is_displacement: true,
            data: vec![0.0; total],
        }
    }

    pub fn index(x: u32, y: u32, z: u32, resolution: u32) -> usize {
        (z * resolution * resolution + y * resolution + x) as usize
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
        SdfPrimitive::Ellipsoid => {
            let k0 = (p / s).length();
            let k1 = (p / (s * s)).length();
            if k1 < 0.0001 {
                p.length() - s.x.min(s.y).min(s.z)
            } else {
                k0 * (k0 - 1.0) / k1
            }
        }
        SdfPrimitive::HexPrism => {
            let k = Vec3::new(-0.866_025_4, 0.5, 0.577_350_27);
            let pa = p.abs();
            let dot2 = 2.0 * (k.x * pa.x + k.y * pa.z).min(0.0);
            let rx = pa.x - dot2 * k.x;
            let rz = pa.z - dot2 * k.y;
            let clamped_rx = rx.clamp(-k.z * s.x, k.z * s.x);
            let d1 = Vec2::new(rx - clamped_rx, rz - s.x).length() * (rz - s.x).signum();
            let d2 = pa.y - s.y;
            d1.max(d2).min(0.0) + Vec2::new(d1.max(0.0), d2.max(0.0)).length()
        }
        SdfPrimitive::Pyramid => {
            let h = s.y;
            let b = s.x;
            let m2 = h * h + 0.25;
            let mut xz = Vec2::new(p.x.abs(), p.z.abs());
            if xz.y > xz.x {
                xz = Vec2::new(xz.y, xz.x);
            }
            xz -= Vec2::splat(0.5) * b;
            let q = Vec3::new(xz.y, h * p.y - 0.5 * xz.x, h * xz.x + 0.5 * p.y);
            let ss = (-q.x).max(0.0);
            let t = ((q.y - 0.5 * xz.y) / (m2 + 0.25)).clamp(0.0, 1.0);
            let a = m2 * (q.x + ss) * (q.x + ss) + q.y * q.y;
            let bb = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
            let d2 = if q.y.min(-q.x * m2 - q.y * 0.5) > 0.0 { 0.0 } else { a.min(bb) };
            ((d2 + q.z * q.z) / m2).sqrt() * q.z.max(-p.y).signum()
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
            input,
            position,
            rotation,
            layer_intensity,
            voxel_grid,
            ..
        } => {
            let local_p = rotate_euler(p - *position, *rotation);
            if let Some(child_id) = input {
                // Differential: analytical child SDF + displacement from grid * layer_intensity
                let analytical = evaluate_sdf_tree(scene, *child_id, p);
                let gc = voxel_grid.world_to_grid(local_p);
                let max_c = (voxel_grid.resolution - 1) as f32;
                let outside = gc.x < 0.0 || gc.y < 0.0 || gc.z < 0.0
                    || gc.x > max_c || gc.y > max_c || gc.z > max_c;
                let disp = if outside { 0.0 } else { voxel_grid.sample(local_p) };
                analytical + disp * layer_intensity
            } else {
                // Standalone: grid IS the total SDF (unchanged)
                voxel_grid.sample(local_p)
            }
        }
        NodeData::Transform { input, translation, rotation, scale } => {
            let Some(child_id) = input else {
                return FAR_DISTANCE;
            };
            // Inverse TRS: subtract T, inverse-rotate R, divide S
            let tp = rotate_euler(p - *translation, *rotation) / *scale;
            let d = evaluate_sdf_tree(scene, *child_id, tp);
            // Scale distance correction
            d * scale.min_element()
        }
        NodeData::Modifier { kind, input, value, extra } => {
            let Some(child_id) = input else {
                return FAR_DISTANCE;
            };
            use crate::graph::scene::ModifierKind;
            match kind {
                // Point modifiers: transform p, then recurse
                ModifierKind::Twist => {
                    let rate = value.x;
                    let c = (rate * p.y).cos();
                    let s = (rate * p.y).sin();
                    let tp = Vec3::new(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Bend => {
                    let k = value.x;
                    let c = (k * p.x).cos();
                    let s = (k * p.x).sin();
                    let tp = Vec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Taper => {
                    let factor = value.x;
                    let s = 1.0 / (1.0 + factor * p.y);
                    let tp = Vec3::new(p.x * s, p.y, p.z * s);
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Elongate => {
                    let h = *value;
                    let tp = p - p.clamp(-h, h);
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Mirror => {
                    let axes = *value;
                    let tp = Vec3::new(
                        if axes.x > 0.5 { p.x.abs() } else { p.x },
                        if axes.y > 0.5 { p.y.abs() } else { p.y },
                        if axes.z > 0.5 { p.z.abs() } else { p.z },
                    );
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Repeat => {
                    let s = *value;
                    let mut q = p;
                    if s.x > 0.0 { q.x -= s.x * (q.x / s.x).round(); }
                    if s.y > 0.0 { q.y -= s.y * (q.y / s.y).round(); }
                    if s.z > 0.0 { q.z -= s.z * (q.z / s.z).round(); }
                    evaluate_sdf_tree(scene, *child_id, q)
                }
                ModifierKind::FiniteRepeat => {
                    let s = *value;
                    let c = *extra;
                    let mut q = p;
                    if s.x > 0.0 { q.x -= s.x * (q.x / s.x).round().clamp(-c.x, c.x); }
                    if s.y > 0.0 { q.y -= s.y * (q.y / s.y).round().clamp(-c.y, c.y); }
                    if s.z > 0.0 { q.z -= s.z * (q.z / s.z).round().clamp(-c.z, c.z); }
                    evaluate_sdf_tree(scene, *child_id, q)
                }
                ModifierKind::RadialRepeat => {
                    let count = value.x.max(1.0);
                    let axis = value.y;
                    let sector = std::f32::consts::TAU / count;
                    let (a, r, tp) = if axis < 0.5 {
                        let a = p.z.atan2(p.y);
                        let r = Vec2::new(p.y, p.z).length();
                        let a = a - sector * (a / sector).round();
                        (a, r, Vec3::new(p.x, r * a.cos(), r * a.sin()))
                    } else if axis < 1.5 {
                        let a = p.z.atan2(p.x);
                        let r = Vec2::new(p.x, p.z).length();
                        let a = a - sector * (a / sector).round();
                        (a, r, Vec3::new(r * a.cos(), p.y, r * a.sin()))
                    } else {
                        let a = p.y.atan2(p.x);
                        let r = Vec2::new(p.x, p.y).length();
                        let a = a - sector * (a / sector).round();
                        (a, r, Vec3::new(r * a.cos(), r * a.sin(), p.z))
                    };
                    let _ = (a, r); // suppress unused warnings
                    evaluate_sdf_tree(scene, *child_id, tp)
                }
                ModifierKind::Noise => {
                    let frequency = value.x;
                    let amplitude = value.y;
                    let octaves = value.z as i32;
                    let displacement = fbm_noise(p, frequency, amplitude, octaves);
                    evaluate_sdf_tree(scene, *child_id, p + displacement)
                }
                // Distance modifiers: recurse first, then modify result
                ModifierKind::Round => {
                    let d = evaluate_sdf_tree(scene, *child_id, p);
                    d - value.x
                }
                ModifierKind::Onion => {
                    let d = evaluate_sdf_tree(scene, *child_id, p);
                    d.abs() - value.x
                }
                ModifierKind::Offset => {
                    let d = evaluate_sdf_tree(scene, *child_id, p);
                    d + value.x
                }
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
        NodeData::Transform { input, .. } | NodeData::Modifier { input, .. } => {
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

/// Create a displacement grid for an arbitrary subtree (differential SDF).
/// Computes bounds from the subtree but does NOT evaluate SDF — grid starts at 0.0.
/// Returns (grid in local space, center in world space).
pub fn create_displacement_grid_for_subtree(
    scene: &Scene,
    subtree_root: NodeId,
    resolution: u32,
) -> (VoxelGrid, Vec3) {
    let (world_min, world_max) = bounds_for_subtree(scene, subtree_root);
    let center = (world_min + world_max) * 0.5;
    let half_extent = (world_max - world_min) * 0.5;
    let grid = VoxelGrid::new_displacement(resolution, -half_extent, half_extent);
    (grid, center)
}

/// Bake any subtree's SDF into a VoxelGrid with progress reporting (incremented per z-slice).
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

    let data: Vec<f32> = maybe_par_iter!(0..res)
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

    let grid = VoxelGrid { resolution, bounds_min: local_min, bounds_max: local_max, is_displacement: false, data };
    (grid, center)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
        }
    }

    /// Helper: create a scene with a single unit sphere at the origin.
    fn scene_with_sphere() -> (Scene, NodeId) {
        let mut scene = empty_scene();
        let name = scene.next_name("Sphere");
        let id = scene.add_node(name, NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        (scene, id)
    }

    /// Helper: create a scene with a unit box at the origin.
    fn scene_with_box() -> (Scene, NodeId) {
        let mut scene = empty_scene();
        let name = scene.next_name("Box");
        let id = scene.add_node(name, NodeData::Primitive {
            kind: SdfPrimitive::Box,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        (scene, id)
    }

    // -----------------------------------------------------------------------
    // VoxelGrid construction and indexing
    // -----------------------------------------------------------------------

    #[test]
    fn new_displacement_creates_zero_filled_grid() {
        let grid = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(grid.resolution, 4);
        assert_eq!(grid.data.len(), 64); // 4^3
        assert!(grid.is_displacement);
        assert!(grid.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn index_layout_z_major() {
        // data[z * res * res + y * res + x]
        assert_eq!(VoxelGrid::index(0, 0, 0, 4), 0);
        assert_eq!(VoxelGrid::index(1, 0, 0, 4), 1);
        assert_eq!(VoxelGrid::index(0, 1, 0, 4), 4);
        assert_eq!(VoxelGrid::index(0, 0, 1, 4), 16);
        assert_eq!(VoxelGrid::index(3, 3, 3, 4), 63);
    }

    #[test]
    fn world_to_grid_maps_bounds_to_grid_edges() {
        let grid = VoxelGrid::new_displacement(4, Vec3::splat(-2.0), Vec3::splat(2.0));
        let gc_min = grid.world_to_grid(Vec3::splat(-2.0));
        let gc_max = grid.world_to_grid(Vec3::splat(2.0));
        assert!((gc_min - Vec3::ZERO).length() < 1e-5);
        assert!((gc_max - Vec3::splat(3.0)).length() < 1e-5);
    }

    #[test]
    fn world_to_grid_center_maps_to_half_resolution() {
        let grid = VoxelGrid::new_displacement(5, Vec3::splat(-1.0), Vec3::splat(1.0));
        let gc = grid.world_to_grid(Vec3::ZERO);
        assert!((gc - Vec3::splat(2.0)).length() < 1e-5);
    }

    #[test]
    fn grid_to_world_roundtrip() {
        let grid = VoxelGrid::new_displacement(8, Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let original = Vec3::new(0.5, -1.0, 2.0);
        let gc = grid.world_to_grid(original);
        let back = grid.grid_to_world(gc.x, gc.y, gc.z);
        assert!((back - original).length() < 1e-4, "roundtrip failed: {back} != {original}");
    }

    #[test]
    fn grid_to_world_maps_edges_to_bounds() {
        let grid = VoxelGrid::new_displacement(4, Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let min = grid.grid_to_world(0.0, 0.0, 0.0);
        let max = grid.grid_to_world(3.0, 3.0, 3.0);
        assert!((min - grid.bounds_min).length() < 1e-5);
        assert!((max - grid.bounds_max).length() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // VoxelGrid sampling (trilinear interpolation)
    // -----------------------------------------------------------------------

    #[test]
    fn sample_returns_exact_value_at_grid_corner() {
        let mut grid = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        // Set value at corner (0,0,0)
        grid.data[VoxelGrid::index(0, 0, 0, 4)] = 5.0;
        let sampled = grid.sample(Vec3::splat(-1.0));
        assert!((sampled - 5.0).abs() < 1e-5, "expected 5.0, got {sampled}");
    }

    #[test]
    fn sample_interpolates_between_two_values() {
        let mut grid = VoxelGrid::new_displacement(2, Vec3::ZERO, Vec3::ONE);
        // Two adjacent voxels along X
        grid.data[VoxelGrid::index(0, 0, 0, 2)] = 0.0;
        grid.data[VoxelGrid::index(1, 0, 0, 2)] = 10.0;
        let mid_x = grid.sample(Vec3::new(0.5, 0.0, 0.0));
        assert!((mid_x - 5.0).abs() < 1e-4, "expected ~5.0, got {mid_x}");
    }

    #[test]
    fn sample_clamps_outside_bounds() {
        let mut grid = VoxelGrid::new_displacement(2, Vec3::ZERO, Vec3::ONE);
        grid.data[VoxelGrid::index(0, 0, 0, 2)] = 3.0;
        grid.data[VoxelGrid::index(1, 0, 0, 2)] = 7.0;
        // Sample well outside bounds — should clamp to edge
        let outside_low = grid.sample(Vec3::new(-10.0, 0.0, 0.0));
        assert!((outside_low - 3.0).abs() < 1e-4, "expected clamped to 3.0, got {outside_low}");
    }

    #[test]
    fn sample_uniform_grid_returns_constant() {
        let mut grid = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        for v in grid.data.iter_mut() {
            *v = 42.0;
        }
        let sampled = grid.sample(Vec3::new(0.3, -0.5, 0.7));
        assert!((sampled - 42.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // content_eq
    // -----------------------------------------------------------------------

    #[test]
    fn content_eq_identical_grids() {
        let grid_a = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid_b = grid_a.clone();
        assert!(grid_a.content_eq(&grid_b));
    }

    #[test]
    fn content_eq_detects_data_change() {
        let grid_a = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        let mut grid_b = grid_a.clone();
        grid_b.data[10] = 1.0;
        assert!(!grid_a.content_eq(&grid_b));
    }

    #[test]
    fn content_eq_detects_resolution_change() {
        let grid_a = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid_b = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(!grid_a.content_eq(&grid_b));
    }

    #[test]
    fn content_eq_detects_bounds_change() {
        let grid_a = VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid_b = VoxelGrid::new_displacement(4, Vec3::splat(-2.0), Vec3::splat(1.0));
        assert!(!grid_a.content_eq(&grid_b));
    }

    // -----------------------------------------------------------------------
    // SDF primitive evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_sdf_sphere_at_origin() {
        let dist = evaluate_sdf(&SdfPrimitive::Sphere, Vec3::ZERO, Vec3::ONE);
        assert!((dist - (-1.0)).abs() < 1e-5, "inside sphere center: {dist}");
    }

    #[test]
    fn evaluate_sdf_sphere_on_surface() {
        let dist = evaluate_sdf(&SdfPrimitive::Sphere, Vec3::X, Vec3::ONE);
        assert!(dist.abs() < 1e-5, "on sphere surface: {dist}");
    }

    #[test]
    fn evaluate_sdf_sphere_outside() {
        let dist = evaluate_sdf(&SdfPrimitive::Sphere, Vec3::new(3.0, 0.0, 0.0), Vec3::ONE);
        assert!((dist - 2.0).abs() < 1e-5, "outside sphere: {dist}");
    }

    #[test]
    fn evaluate_sdf_box_at_origin() {
        let dist = evaluate_sdf(&SdfPrimitive::Box, Vec3::ZERO, Vec3::ONE);
        assert!((dist - (-1.0)).abs() < 1e-5, "inside box center: {dist}");
    }

    #[test]
    fn evaluate_sdf_box_on_face() {
        let dist = evaluate_sdf(&SdfPrimitive::Box, Vec3::X, Vec3::ONE);
        assert!(dist.abs() < 1e-5, "on box face: {dist}");
    }

    #[test]
    fn evaluate_sdf_box_outside() {
        let dist = evaluate_sdf(&SdfPrimitive::Box, Vec3::new(2.0, 0.0, 0.0), Vec3::ONE);
        assert!((dist - 1.0).abs() < 1e-5, "outside box: {dist}");
    }

    #[test]
    fn evaluate_sdf_plane_above() {
        let dist = evaluate_sdf(&SdfPrimitive::Plane, Vec3::new(0.0, 2.0, 0.0), Vec3::ONE);
        assert!((dist - 2.0).abs() < 1e-5);
    }

    #[test]
    fn evaluate_sdf_plane_below() {
        let dist = evaluate_sdf(&SdfPrimitive::Plane, Vec3::new(0.0, -1.0, 0.0), Vec3::ONE);
        assert!((dist - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn evaluate_sdf_cylinder_on_surface() {
        // Cylinder with radius=1, half-height=1
        let dist = evaluate_sdf(&SdfPrimitive::Cylinder, Vec3::new(1.0, 0.0, 0.0), Vec3::ONE);
        assert!(dist.abs() < 1e-5, "on cylinder surface: {dist}");
    }

    #[test]
    fn evaluate_sdf_torus_on_surface() {
        // Torus with major_r=1, minor_r=0.25 (scale.x=major, scale.y=minor)
        let on_surface = Vec3::new(1.25, 0.0, 0.0);
        let dist = evaluate_sdf(&SdfPrimitive::Torus, on_surface, Vec3::new(1.0, 0.25, 0.0));
        assert!(dist.abs() < 1e-4, "on torus surface: {dist}");
    }

    #[test]
    fn evaluate_sdf_capsule_on_surface() {
        // Capsule: radius=0.5, half-height=1.0
        let on_tip = Vec3::new(0.0, 1.5, 0.0); // y=h+r = 1.0+0.5
        let dist = evaluate_sdf(&SdfPrimitive::Capsule, on_tip, Vec3::new(0.5, 1.0, 0.0));
        assert!(dist.abs() < 1e-4, "on capsule tip: {dist}");
    }

    // -----------------------------------------------------------------------
    // CSG operations
    // -----------------------------------------------------------------------

    #[test]
    fn csg_union_returns_min() {
        assert_eq!(csg_union(1.0, 2.0), 1.0);
        assert_eq!(csg_union(-1.0, 0.5), -1.0);
    }

    #[test]
    fn csg_subtract_returns_max_neg() {
        assert_eq!(csg_subtract(1.0, 2.0), 1.0); // max(1, -2) = 1
        assert_eq!(csg_subtract(0.5, -1.0), 1.0); // max(0.5, 1) = 1
    }

    #[test]
    fn csg_intersect_returns_max() {
        assert_eq!(csg_intersect(1.0, 2.0), 2.0);
        assert_eq!(csg_intersect(-1.0, 0.5), 0.5);
    }

    #[test]
    fn csg_smooth_union_blends() {
        let hard = csg_union(-0.1, 0.1);
        let smooth = csg_smooth_union(-0.1, 0.1, 0.5);
        // Smooth union should be <= hard union (it rounds inward)
        assert!(smooth <= hard, "smooth {smooth} should be <= hard {hard}");
    }

    #[test]
    fn csg_smooth_union_zero_k_equals_hard() {
        let a = 0.5_f32;
        let b = 1.5_f32;
        let smooth = csg_smooth_union(a, b, 0.0);
        let hard = csg_union(a, b);
        assert!((smooth - hard).abs() < 1e-3, "smooth={smooth}, hard={hard}");
    }

    // -----------------------------------------------------------------------
    // evaluate_sdf_tree — recursive tree evaluator
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_sdf_tree_sphere_at_origin() {
        let (scene, id) = scene_with_sphere();
        let dist = evaluate_sdf_tree(&scene, id, Vec3::ZERO);
        assert!((dist - (-1.0)).abs() < 1e-5, "sphere center: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_sphere_outside() {
        let (scene, id) = scene_with_sphere();
        let dist = evaluate_sdf_tree(&scene, id, Vec3::new(3.0, 0.0, 0.0));
        assert!((dist - 2.0).abs() < 1e-5, "outside sphere: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_missing_node_returns_far() {
        let scene = empty_scene();
        let dist = evaluate_sdf_tree(&scene, 999, Vec3::ZERO);
        assert_eq!(dist, FAR_DISTANCE);
    }

    #[test]
    fn evaluate_sdf_tree_translated_primitive() {
        let mut scene = empty_scene();
        let id = scene.add_node("Sphere".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        // At (5,0,0) should be on surface
        let dist = evaluate_sdf_tree(&scene, id, Vec3::new(6.0, 0.0, 0.0));
        assert!(dist.abs() < 1e-4, "on translated sphere surface: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_csg_union() {
        let mut scene = empty_scene();
        let sphere_a = scene.add_node("A".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(-1.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let sphere_b = scene.add_node("B".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let union_id = scene.add_node("Union".into(), NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            left: Some(sphere_a),
            right: Some(sphere_b),
        });
        // At origin, both unit spheres at ±1 just touch (dist=0). Should be <= 0.
        let dist = evaluate_sdf_tree(&scene, union_id, Vec3::ZERO);
        assert!(dist <= 0.0, "at union boundary: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_csg_subtract() {
        let mut scene = empty_scene();
        let sphere_a = scene.add_node("A".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::splat(2.0),
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let sphere_b = scene.add_node("B".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let sub_id = scene.add_node("Sub".into(), NodeData::Operation {
            op: CsgOp::Subtract,
            smooth_k: 0.0,
            left: Some(sphere_a),
            right: Some(sphere_b),
        });
        // At origin: A dist=-2, B dist=-1. Subtract = max(-2, -(-1)) = max(-2, 1) = 1 (outside)
        let dist = evaluate_sdf_tree(&scene, sub_id, Vec3::ZERO);
        assert!((dist - 1.0).abs() < 1e-5, "subtracted region at origin: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_operation_single_child() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let op_id = scene.add_node("Op".into(), NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            left: Some(sphere),
            right: None,
        });
        // With one child, result should be same as child
        let dist = evaluate_sdf_tree(&scene, op_id, Vec3::ZERO);
        assert!((dist - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn evaluate_sdf_tree_operation_no_children() {
        let mut scene = empty_scene();
        let op_id = scene.add_node("Op".into(), NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            left: None,
            right: None,
        });
        let dist = evaluate_sdf_tree(&scene, op_id, Vec3::ZERO);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn evaluate_sdf_tree_transform_translation() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let transform_id = scene.add_node("T".into(), NodeData::Transform {
            input: Some(sphere),
            translation: Vec3::new(3.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        });
        // Sphere should be centered at (3,0,0) after transform
        let dist_at_center = evaluate_sdf_tree(&scene, transform_id, Vec3::new(3.0, 0.0, 0.0));
        assert!((dist_at_center - (-1.0)).abs() < 1e-4, "sphere center via transform: {dist_at_center}");
    }

    #[test]
    fn evaluate_sdf_tree_transform_uniform_scale() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let transform_id = scene.add_node("T".into(), NodeData::Transform {
            input: Some(sphere),
            translation: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::splat(2.0),
        });
        // At distance 2.0 from origin should be on surface of scaled sphere
        let dist = evaluate_sdf_tree(&scene, transform_id, Vec3::new(2.0, 0.0, 0.0));
        assert!(dist.abs() < 1e-4, "on scaled sphere surface: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_transform_empty_returns_far() {
        let mut scene = empty_scene();
        let transform_id = scene.add_node("T".into(), NodeData::Transform {
            input: None,
            translation: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        });
        let dist = evaluate_sdf_tree(&scene, transform_id, Vec3::ZERO);
        assert_eq!(dist, FAR_DISTANCE);
    }

    // -----------------------------------------------------------------------
    // rotate_euler
    // -----------------------------------------------------------------------

    #[test]
    fn rotate_euler_identity() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let rotated = rotate_euler(p, Vec3::ZERO);
        assert!((rotated - p).length() < 1e-5);
    }

    #[test]
    fn rotate_euler_90_deg_y() {
        let p = Vec3::X; // (1,0,0)
        let rotated = rotate_euler(p, Vec3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0));
        // 90° around Y should send X → -Z
        let expected = Vec3::new(0.0, 0.0, -1.0);
        assert!((rotated - expected).length() < 1e-5, "got {rotated}, expected {expected}");
    }

    // -----------------------------------------------------------------------
    // Bounds estimation
    // -----------------------------------------------------------------------

    #[test]
    fn bounds_for_subtree_single_sphere() {
        let (scene, id) = scene_with_sphere();
        let (bmin, bmax) = bounds_for_subtree(&scene, id);
        // Sphere at origin with scale=1 → extent = 1*1.5 = 1.5, padded by 0.5
        assert!((bmin - Vec3::splat(-2.0)).length() < 1e-5, "bmin: {bmin}");
        assert!((bmax - Vec3::splat(2.0)).length() < 1e-5, "bmax: {bmax}");
    }

    #[test]
    fn bounds_for_subtree_translated_sphere() {
        let mut scene = empty_scene();
        let id = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let (bmin, bmax) = bounds_for_subtree(&scene, id);
        // center=5, extent=1.5, pad=0.5 → min=3.0, max=7.0 on x
        assert!((bmin.x - 3.0).abs() < 1e-5, "bmin.x: {}", bmin.x);
        assert!((bmax.x - 7.0).abs() < 1e-5, "bmax.x: {}", bmax.x);
    }

    #[test]
    fn bounds_for_subtree_operation_encloses_both_children() {
        let mut scene = empty_scene();
        let left = scene.add_node("L".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(-5.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let right = scene.add_node("R".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        let op = scene.add_node("U".into(), NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            left: Some(left),
            right: Some(right),
        });
        let (bmin, bmax) = bounds_for_subtree(&scene, op);
        // Left: -5-1.5-0.5=-7, Right: 5+1.5+0.5=7
        assert!(bmin.x <= -7.0 + 1e-5, "bmin.x should cover left sphere: {}", bmin.x);
        assert!(bmax.x >= 7.0 - 1e-5, "bmax.x should cover right sphere: {}", bmax.x);
    }

    // -----------------------------------------------------------------------
    // create_displacement_grid_for_subtree
    // -----------------------------------------------------------------------

    #[test]
    fn create_displacement_grid_centered_on_subtree() {
        let (scene, id) = scene_with_sphere();
        let (grid, center) = create_displacement_grid_for_subtree(&scene, id, 8);
        assert_eq!(grid.resolution, 8);
        assert!(grid.is_displacement);
        assert!(grid.data.iter().all(|&v| v == 0.0));
        // Sphere at origin → center should be near origin
        assert!(center.length() < 1e-5, "center: {center}");
        // Bounds should be symmetric around origin
        assert!((grid.bounds_min + grid.bounds_max).length() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // bake_subtree_with_progress
    // -----------------------------------------------------------------------

    #[test]
    fn bake_sphere_produces_negative_inside_positive_outside() {
        let (scene, id) = scene_with_sphere();
        let progress = Arc::new(AtomicU32::new(0));
        let (grid, center) = bake_subtree_with_progress(&scene, id, 16, progress.clone());

        assert_eq!(grid.resolution, 16);
        assert!(!grid.is_displacement);
        assert!(center.length() < 1e-5);

        // Sample at center of grid (origin in world space) — should be inside sphere
        let center_val = grid.sample(Vec3::ZERO);
        assert!(center_val < 0.0, "center of sphere should be negative: {center_val}");

        // Sample at a corner of the grid — should be outside sphere
        let corner_val = grid.sample(grid.bounds_max);
        assert!(corner_val > 0.0, "corner of grid should be positive: {corner_val}");

        // Progress should have been incremented once per z-slice
        assert_eq!(progress.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn bake_box_negative_at_center() {
        let (scene, id) = scene_with_box();
        let progress = Arc::new(AtomicU32::new(0));
        let (grid, _center) = bake_subtree_with_progress(&scene, id, 8, progress);
        let center_val = grid.sample(Vec3::ZERO);
        assert!(center_val < 0.0, "center of box should be negative: {center_val}");
    }

    // -----------------------------------------------------------------------
    // max_subtree_resolution
    // -----------------------------------------------------------------------

    #[test]
    fn max_subtree_resolution_no_sculpt_returns_default() {
        let (scene, id) = scene_with_sphere();
        let res = max_subtree_resolution(&scene, id);
        assert_eq!(res, DEFAULT_RESOLUTION);
    }

    #[test]
    fn max_subtree_resolution_finds_sculpt_node() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(64, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.add_node("Sculpt".into(), NodeData::Sculpt {
            input: None,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            layer_intensity: 1.0,
            voxel_grid: grid,
            desired_resolution: 64,
        });
        let res = max_subtree_resolution(&scene, sculpt_id);
        assert_eq!(res, 64);
    }

    // -----------------------------------------------------------------------
    // Sculpt node evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_sdf_tree_sculpt_standalone_samples_grid() {
        let mut grid = VoxelGrid::new_displacement(32, Vec3::splat(-2.0), Vec3::splat(2.0));
        // Mark grid as total SDF (not displacement)
        grid.is_displacement = false;
        // Fill with a simple distance: negative inside, positive outside
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                    grid.data[VoxelGrid::index(x, y, z, 32)] = pos.length() - 1.0; // unit sphere SDF
                }
            }
        }
        let mut scene = empty_scene();
        let sculpt_id = scene.add_node("Sculpt".into(), NodeData::Sculpt {
            input: None,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            layer_intensity: 1.0,
            voxel_grid: grid,
            desired_resolution: 32,
        });
        let dist = evaluate_sdf_tree(&scene, sculpt_id, Vec3::ZERO);
        assert!(dist < 0.0, "inside sculpt sphere: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_sculpt_with_child_adds_displacement() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        // Displacement grid with uniform +0.5 displacement
        let mut grid = VoxelGrid::new_displacement(4, Vec3::splat(-2.0), Vec3::splat(2.0));
        for v in grid.data.iter_mut() {
            *v = 0.5;
        }
        let sculpt_id = scene.add_node("Sculpt".into(), NodeData::Sculpt {
            input: Some(sphere),
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            layer_intensity: 1.0,
            voxel_grid: grid,
            desired_resolution: 4,
        });
        // At origin, sphere SDF = -1.0, displacement = 0.5*1.0 → total = -0.5
        let dist = evaluate_sdf_tree(&scene, sculpt_id, Vec3::ZERO);
        assert!((dist - (-0.5)).abs() < 1e-4, "sculpt with displacement: {dist}");
    }

    // -----------------------------------------------------------------------
    // Modifier evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_sdf_tree_modifier_round() {
        let mut scene = empty_scene();
        let box_id = scene.add_node("B".into(), NodeData::Primitive {
            kind: SdfPrimitive::Box,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let round_id = scene.add_node("Round".into(), NodeData::Modifier {
            kind: ModifierKind::Round,
            input: Some(box_id),
            value: Vec3::new(0.1, 0.0, 0.0),
            extra: Vec3::ZERO,
        });
        // Round subtracts the radius from the SDF
        let dist_box = evaluate_sdf_tree(&scene, box_id, Vec3::new(1.0, 0.0, 0.0));
        let dist_round = evaluate_sdf_tree(&scene, round_id, Vec3::new(1.0, 0.0, 0.0));
        assert!((dist_round - (dist_box - 0.1)).abs() < 1e-5, "round modifier: {dist_round}");
    }

    #[test]
    fn evaluate_sdf_tree_modifier_onion() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let onion_id = scene.add_node("Onion".into(), NodeData::Modifier {
            kind: ModifierKind::Onion,
            input: Some(sphere),
            value: Vec3::new(0.1, 0.0, 0.0),
            extra: Vec3::ZERO,
        });
        // At origin, sphere = -1.0, onion = |−1.0| − 0.1 = 0.9 (outside the shell)
        let dist = evaluate_sdf_tree(&scene, onion_id, Vec3::ZERO);
        assert!((dist - 0.9).abs() < 1e-5, "onion at center: {dist}");
    }

    #[test]
    fn evaluate_sdf_tree_modifier_offset() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let offset_id = scene.add_node("Offset".into(), NodeData::Modifier {
            kind: ModifierKind::Offset,
            input: Some(sphere),
            value: Vec3::new(0.5, 0.0, 0.0),
            extra: Vec3::ZERO,
        });
        // Offset adds the value to the distance
        let dist_sphere = evaluate_sdf_tree(&scene, sphere, Vec3::new(2.0, 0.0, 0.0));
        let dist_offset = evaluate_sdf_tree(&scene, offset_id, Vec3::new(2.0, 0.0, 0.0));
        assert!((dist_offset - (dist_sphere + 0.5)).abs() < 1e-5, "offset modifier: {dist_offset}");
    }

    #[test]
    fn evaluate_sdf_tree_modifier_mirror_x() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::new(2.0, 0.0, 0.0),
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let mirror_id = scene.add_node("Mirror".into(), NodeData::Modifier {
            kind: ModifierKind::Mirror,
            input: Some(sphere),
            value: Vec3::new(1.0, 0.0, 0.0), // mirror X
            extra: Vec3::ZERO,
        });
        // Sphere at (2,0,0). Mirror X: p.x = |p.x|
        // At (-3,0,0): mirrored to (3,0,0) → dist to sphere center = 1 → on surface
        let dist = evaluate_sdf_tree(&scene, mirror_id, Vec3::new(-3.0, 0.0, 0.0));
        let dist_pos = evaluate_sdf_tree(&scene, mirror_id, Vec3::new(3.0, 0.0, 0.0));
        assert!(dist.abs() < 1e-4, "mirrored point should be on surface: {dist}");
        assert!((dist - dist_pos).abs() < 1e-4, "mirrored should match positive: {dist} vs {dist_pos}");
    }

    #[test]
    fn evaluate_sdf_tree_modifier_noise_displaces_point() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let noise_id = scene.add_node("Noise".into(), NodeData::Modifier {
            kind: ModifierKind::Noise,
            input: Some(sphere),
            value: Vec3::new(2.0, 0.5, 3.0), // frequency=2, amplitude=0.5, octaves=3
            extra: Vec3::ZERO,
        });
        // With non-zero amplitude, the noise modifier should change the SDF distance
        // compared to evaluating without it
        let dist_plain = evaluate_sdf_tree(&scene, sphere, Vec3::new(1.5, 0.3, 0.7));
        let dist_noise = evaluate_sdf_tree(&scene, noise_id, Vec3::new(1.5, 0.3, 0.7));
        assert!(
            (dist_plain - dist_noise).abs() > 1e-6,
            "noise modifier should displace point: plain={dist_plain}, noise={dist_noise}"
        );
    }

    #[test]
    fn noise_modifier_zero_amplitude_is_identity() {
        let mut scene = empty_scene();
        let sphere = scene.add_node("S".into(), NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            color: Vec3::ONE,
            roughness: 0.5,
            metallic: 0.0,
            emissive: Vec3::ZERO,
            emissive_intensity: 0.0,
            fresnel: 0.04,
            voxel_grid: None,
        });
        use crate::graph::scene::ModifierKind;
        let noise_id = scene.add_node("Noise".into(), NodeData::Modifier {
            kind: ModifierKind::Noise,
            input: Some(sphere),
            value: Vec3::new(2.0, 0.0, 3.0), // amplitude=0 → no displacement
            extra: Vec3::ZERO,
        });
        let dist_plain = evaluate_sdf_tree(&scene, sphere, Vec3::new(1.5, 0.3, 0.7));
        let dist_noise = evaluate_sdf_tree(&scene, noise_id, Vec3::new(1.5, 0.3, 0.7));
        assert!(
            (dist_plain - dist_noise).abs() < 1e-6,
            "zero amplitude should not displace: plain={dist_plain}, noise={dist_noise}"
        );
    }

    #[test]
    fn evaluate_sdf_tree_modifier_empty_returns_far() {
        let mut scene = empty_scene();
        use crate::graph::scene::ModifierKind;
        let mod_id = scene.add_node("M".into(), NodeData::Modifier {
            kind: ModifierKind::Round,
            input: None,
            value: Vec3::new(0.1, 0.0, 0.0),
            extra: Vec3::ZERO,
        });
        let dist = evaluate_sdf_tree(&scene, mod_id, Vec3::ZERO);
        assert_eq!(dist, FAR_DISTANCE);
    }
}
