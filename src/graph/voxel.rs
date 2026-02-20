use glam::{Vec2, Vec3, Vec3Swizzles};
use serde::{Deserialize, Serialize};

use crate::graph::scene::SdfPrimitive;

pub const DEFAULT_RESOLUTION: u32 = 64;
const GRID_PADDING: f32 = 0.5;
const FAR_DISTANCE: f32 = 999.0;

/// A 3D signed distance field stored as a flat array.
/// Layout: data[z * res * res + y * res + x]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub data: Vec<f32>,
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
