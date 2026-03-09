use glam::Vec3;

use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel::{self, VoxelGrid};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_BRUSH_RADIUS: f32 = 0.3;
pub const DEFAULT_BRUSH_STRENGTH: f32 = 0.05;
pub const DEFAULT_STROKE_SPACING: f32 = 0.15;

// ---------------------------------------------------------------------------
// Tool system
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActiveTool {
    #[default]
    Select,
    Sculpt,
    // Future: Mask, Paint, Polygroup
}

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
            Self::Add => -1.0,  // decrease distance = add material
            Self::Carve => 1.0, // increase distance = remove material
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
            Self::Grab => 5.0, // CPU-only, never dispatched to GPU
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
// Brush shapes (alphas/stamps)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum BrushShape {
    Sphere,
    Cube,
    Diamond,
    Ring,
    Cylinder,
}

impl BrushShape {
    /// Compute the normalized distance [0, 1) for a voxel offset from brush center.
    /// Returns None if the voxel is outside the brush region.
    pub fn normalized_distance(&self, offset: Vec3, radius: f32) -> Option<f32> {
        match self {
            Self::Sphere => {
                let dist = offset.length();
                if dist < radius {
                    Some(dist / radius)
                } else {
                    None
                }
            }
            Self::Cube => {
                let nt = offset.abs().max_element() / radius;
                if nt < 1.0 {
                    Some(nt)
                } else {
                    None
                }
            }
            Self::Diamond => {
                let nt = (offset.x.abs() + offset.y.abs() + offset.z.abs()) / radius;
                if nt < 1.0 {
                    Some(nt)
                } else {
                    None
                }
            }
            Self::Ring => {
                let dist = offset.length();
                if dist < radius {
                    let nt = dist / radius;
                    // Ring peaks at 0.6 radius, falls off toward center and edge
                    let ring_val = 1.0 - (2.0 * (nt - 0.6).abs()).clamp(0.0, 1.0);
                    Some(1.0 - ring_val) // invert so 0 = strongest
                } else {
                    None
                }
            }
            Self::Cylinder => {
                // Uses XZ distance only (creates column-like strokes along Y)
                let dist_xz = (offset.x * offset.x + offset.z * offset.z).sqrt();
                let nt_xz = dist_xz / radius;
                let nt_y = offset.y.abs() / radius;
                if nt_xz < 1.0 && nt_y < 1.0 {
                    Some(nt_xz)
                } else {
                    None
                }
            }
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
        brush_shape: BrushShape,
        smooth_iterations: u32,
        /// SDF value at brush center when Flatten drag started. Reset on mouse release.
        flatten_reference: Option<f32>,
        /// Lazy brush dead zone radius (0.0 = disabled).
        lazy_radius: f32,
        /// Stroke interpolation spacing as a fraction of brush radius.
        /// Lower values produce denser, smoother stroke samples.
        stroke_spacing: f32,
        /// Surface constraint multiplier (0.0 = off). Attenuates brush near surface only.
        surface_constraint: f32,
        /// If true, attenuate brush influence toward back-facing voxels.
        /// Mimics "front faces only" behavior from DCC sculpt tools.
        front_faces_only: bool,
        /// Mirror axis: None = off, Some(0) = X, Some(1) = Y, Some(2) = Z.
        symmetry_axis: Option<u8>,
        /// Snapshot of grid data for grab brush (cloned on grab start).
        /// For differential sculpts, this is the total SDF (analytical + displacement).
        grab_snapshot: Option<Vec<f32>>,
        /// World position where grab stroke started.
        grab_start: Option<Vec3>,
        /// Child input node for differential grab (used to subtract analytical SDF on write-back).
        grab_child_input: Option<NodeId>,
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
            brush_shape: BrushShape::Sphere,
            smooth_iterations: 3,
            flatten_reference: None,
            lazy_radius: 0.0,
            stroke_spacing: DEFAULT_STROKE_SPACING,
            surface_constraint: 0.0,
            front_faces_only: true,
            symmetry_axis: None,
            grab_snapshot: None,
            grab_start: None,
            grab_child_input: None,
        }
    }

    /// Create Active state with adaptive brush radius based on object bounds.
    /// `extent` is the average half-extent of the bounding box (e.g. from `compute_bounds()`).
    pub fn new_active_with_radius(node_id: NodeId, extent: f32) -> Self {
        let radius = (extent * 0.15).clamp(0.05, 2.0);
        Self::Active {
            node_id,
            brush_mode: BrushMode::Add,
            brush_radius: radius,
            brush_strength: DEFAULT_BRUSH_STRENGTH,
            falloff_mode: FalloffMode::Sharp,
            brush_shape: BrushShape::Sphere,
            smooth_iterations: 3,
            flatten_reference: None,
            lazy_radius: 0.0,
            stroke_spacing: DEFAULT_STROKE_SPACING,
            surface_constraint: 0.0,
            front_faces_only: true,
            symmetry_axis: None,
            grab_snapshot: None,
            grab_start: None,
            grab_child_input: None,
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
#[allow(clippy::too_many_arguments)]
pub fn apply_brush(
    scene: &mut Scene,
    node_id: NodeId,
    hit_world: Vec3,
    view_dir_world: Vec3,
    brush_mode: &BrushMode,
    brush_radius: f32,
    brush_strength: f32,
    falloff_mode: &FalloffMode,
    brush_shape: &BrushShape,
    smooth_iterations: u32,
    flatten_ref: f32,
    surface_constraint: f32,
    front_faces_only: bool,
) -> Option<(u32, u32)> {
    // Read transform to convert hit point to local space
    let (position, rotation) = match scene.nodes.get(&node_id).map(|n| &n.data) {
        Some(NodeData::Sculpt {
            position, rotation, ..
        }) => (*position, *rotation),
        _ => return None,
    };

    let local_hit = inverse_rotate_euler(hit_world - position, rotation);
    let local_view_dir = if front_faces_only {
        inverse_rotate_euler(view_dir_world, rotation).normalize_or_zero()
    } else {
        Vec3::ZERO
    };

    // Get mutable reference to grid and apply brush
    let node = scene.nodes.get_mut(&node_id)?;
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
                brush_shape,
                smooth_iterations,
                surface_constraint,
                local_view_dir,
            )),
            _ => Some(apply_brush_to_grid(
                voxel_grid,
                local_hit,
                brush_mode,
                brush_radius,
                brush_strength,
                falloff_mode,
                brush_shape,
                flatten_ref,
                surface_constraint,
                local_view_dir,
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

/// Estimate the largest world-space voxel step for the grid.
fn max_voxel_step(grid: &VoxelGrid) -> f32 {
    let extent = grid.bounds_max - grid.bounds_min;
    let denom = grid.resolution.saturating_sub(1).max(1) as f32;
    (extent.max_element() / denom).max(1e-6)
}

/// Clamp per-sample signed SDF delta to avoid aggressive stepping artifacts.
fn clamp_brush_delta(delta: f32, radius: f32, voxel_step: f32) -> f32 {
    let max_delta = (voxel_step * 2.0).min(radius * 0.35).max(0.01);
    delta.clamp(-max_delta, max_delta)
}

/// Attenuate influence for back-facing voxels relative to camera view direction.
/// `view_dir_local = Vec3::ZERO` disables this term.
fn front_face_factor(offset: Vec3, view_dir_local: Vec3) -> f32 {
    let offset_len2 = offset.length_squared();
    let view_len2 = view_dir_local.length_squared();
    if offset_len2 < 1e-8 || view_len2 < 1e-8 {
        return 1.0;
    }
    let ndotv = offset.normalize().dot(view_dir_local.normalize());
    let hemi = (0.5 - 0.5 * ndotv).clamp(0.0, 1.0);
    hemi * hemi
}

/// Trilinear interpolation from arbitrary voxel data using the grid transform.
fn sample_from_data(grid: &VoxelGrid, data: &[f32], local_pos: Vec3) -> f32 {
    let gc = grid.world_to_grid(local_pos);
    let res = grid.resolution;
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

    let c000 = data[VoxelGrid::index(ix0, iy0, iz0, res)];
    let c100 = data[VoxelGrid::index(ix1, iy0, iz0, res)];
    let c010 = data[VoxelGrid::index(ix0, iy1, iz0, res)];
    let c110 = data[VoxelGrid::index(ix1, iy1, iz0, res)];
    let c001 = data[VoxelGrid::index(ix0, iy0, iz1, res)];
    let c101 = data[VoxelGrid::index(ix1, iy0, iz1, res)];
    let c011 = data[VoxelGrid::index(ix0, iy1, iz1, res)];
    let c111 = data[VoxelGrid::index(ix1, iy1, iz1, res)];

    let c00 = c000 + (c100 - c000) * fx;
    let c10 = c010 + (c110 - c010) * fx;
    let c01 = c001 + (c101 - c001) * fx;
    let c11 = c011 + (c111 - c011) * fx;
    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;
    c0 + (c1 - c0) * fz
}

/// Incompressible Kelvinlet displacement used by the grab/move brush.
/// This gives smoother, Blender-like move behavior without a hard support edge.
fn kelvinlet_displacement(offset: Vec3, displacement: Vec3, eps: f32) -> Vec3 {
    if displacement.length_squared() < 1e-10 || eps <= 1e-6 {
        return Vec3::ZERO;
    }
    let r2 = offset.length_squared();
    let eps2 = eps * eps;
    let r_eps = (r2 + eps2).sqrt();
    let r_eps3 = r_eps * r_eps * r_eps;
    if r_eps3 <= 1e-10 {
        return Vec3::ZERO;
    }
    let scale = eps / (3.0 * r_eps3);
    let x_dot_d = offset.dot(displacement);
    displacement * (scale * (r2 + 2.0 * eps2)) + offset * (scale * x_dot_d)
}

/// Estimate a finite AABB radius for Kelvinlet updates to keep CPU sculpting responsive.
fn kelvinlet_effect_radius(eps: f32, disp_len: f32, min_displacement: f32) -> f32 {
    if disp_len <= 1e-6 || min_displacement <= 1e-6 {
        return eps.max(1e-6);
    }
    // Conservative asymptotic bound: |u| <= 2*|d|*eps/(3*sqrt(r^2 + eps^2)).
    let k = 2.0 * disp_len * eps / (3.0 * min_displacement);
    if k <= eps {
        eps
    } else {
        (k * k - eps * eps).sqrt().max(eps)
    }
}
/// Returns (z0, z1) inclusive range of z-slabs that were modified.
#[allow(clippy::too_many_arguments)]
fn apply_brush_to_grid(
    grid: &mut VoxelGrid,
    center: Vec3,
    brush_mode: &BrushMode,
    radius: f32,
    strength: f32,
    falloff_mode: &FalloffMode,
    brush_shape: &BrushShape,
    flatten_ref: f32,
    surface_constraint: f32,
    view_dir_local: Vec3,
) -> (u32, u32) {
    let res = grid.resolution;
    let voxel_step = max_voxel_step(grid);

    // Compute grid-space bounding box of the brush region
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
                let offset = world_pos - center;
                if let Some(nt) = brush_shape.normalized_distance(offset, radius) {
                    let front = front_face_factor(offset, view_dir_local);
                    if front <= 0.0 {
                        continue;
                    }
                    let falloff = falloff_mode.evaluate(nt) * front;
                    let idx = VoxelGrid::index(x, y, z, res);
                    match brush_mode {
                        BrushMode::Add | BrushMode::Carve => {
                            let sf = surface_factor(grid.data[idx], radius, surface_constraint);
                            let delta = clamp_brush_delta(
                                brush_mode.sign() * strength * falloff * sf,
                                radius,
                                voxel_step,
                            );
                            grid.data[idx] += delta;
                        }
                        BrushMode::Flatten => {
                            let sf = surface_factor(grid.data[idx], radius, surface_constraint);
                            let delta = clamp_brush_delta(
                                (flatten_ref - grid.data[idx]) * falloff * strength * sf,
                                radius,
                                voxel_step,
                            );
                            grid.data[idx] += delta;
                        }
                        BrushMode::Inflate => {
                            let threshold = radius * 0.5;
                            let sf = 1.0 - (grid.data[idx].abs() / threshold).clamp(0.0, 1.0);
                            let delta =
                                clamp_brush_delta(-strength * falloff * sf, radius, voxel_step);
                            grid.data[idx] += delta;
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
#[allow(clippy::too_many_arguments)]
fn apply_smooth_to_grid(
    grid: &mut VoxelGrid,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff_mode: &FalloffMode,
    brush_shape: &BrushShape,
    iterations: u32,
    surface_constraint: f32,
    view_dir_local: Vec3,
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

    // Taubin smoothing (lambda/mu) preserves volume better than pure Laplacian.
    // A single "iteration" is two passes.
    const LAMBDA: f32 = 0.5;
    const MU: f32 = -0.53;
    for _ in 0..iterations {
        for pass_scale in [LAMBDA, MU] {
            let snapshot = grid.data.clone();

            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let world_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                        let offset = world_pos - center;
                        let Some(nt) = brush_shape.normalized_distance(offset, radius) else {
                            continue;
                        };
                        let front = front_face_factor(offset, view_dir_local);
                        if front <= 0.0 {
                            continue;
                        }
                        let falloff = falloff_mode.evaluate(nt) * front;

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
                        let blend = (falloff * strength * sf * pass_scale).clamp(-1.0, 1.0);
                        grid.data[idx] = current + (avg - current) * blend;
                    }
                }
            }
        }
    }

    (z0, z1)
}

/// Apply grab brush using a Kelvinlet warp sampled from the grab-start snapshot.
/// Returns (z0, z1) inclusive range of z-slabs modified.
#[allow(clippy::too_many_arguments)]
pub fn apply_grab_to_grid(
    grid: &mut VoxelGrid,
    snapshot: &[f32],
    center: Vec3,
    radius: f32,
    strength: f32,
    grab_delta: Vec3,
    _falloff_mode: &FalloffMode,
    _surface_constraint: f32,
    _view_dir_local: Vec3,
) -> (u32, u32) {
    let res = grid.resolution;
    let displacement = grab_delta * strength;
    let disp_len = displacement.length();
    if disp_len <= 1e-6 || radius <= 1e-6 {
        let z = grid.world_to_grid(center).z.clamp(0.0, (res - 1) as f32) as u32;
        return (z, z);
    }

    let extent = grid.bounds_max - grid.bounds_min;
    let voxel_step = extent.max_element() / (res.saturating_sub(1).max(1) as f32);
    let min_displacement = (voxel_step * 0.2).max(1e-4);
    let min_disp2 = min_displacement * min_displacement;
    let effect_radius = kelvinlet_effect_radius(radius, disp_len, min_displacement);

    let brush_min = center - Vec3::splat(effect_radius);
    let brush_max = center + Vec3::splat(effect_radius);
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
                let local_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let warp = kelvinlet_displacement(local_pos - center, displacement, radius);
                if warp.length_squared() <= min_disp2 {
                    continue;
                }

                let idx = VoxelGrid::index(x, y, z, res);
                let sample_pos = local_pos - warp;
                grid.data[idx] = sample_from_data(grid, snapshot, sample_pos);
            }
        }
    }

    (z0, z1)
}
/// Apply grab brush for differential sculpts: snapshot is total SDF, write back as displacement.
/// Subtracts the analytical child SDF at each voxel to convert total -> displacement.
#[allow(clippy::too_many_arguments)]
pub fn apply_grab_to_grid_differential(
    grid: &mut VoxelGrid,
    snapshot: &[f32],
    center: Vec3,
    radius: f32,
    strength: f32,
    grab_delta: Vec3,
    _falloff_mode: &FalloffMode,
    _surface_constraint: f32,
    _view_dir_local: Vec3,
    scene: &Scene,
    child_id: NodeId,
    sculpt_position: Vec3,
    sculpt_rotation: Vec3,
) -> (u32, u32) {
    let res = grid.resolution;
    let displacement = grab_delta * strength;
    let disp_len = displacement.length();
    if disp_len <= 1e-6 || radius <= 1e-6 {
        let z = grid.world_to_grid(center).z.clamp(0.0, (res - 1) as f32) as u32;
        return (z, z);
    }

    let extent = grid.bounds_max - grid.bounds_min;
    let voxel_step = extent.max_element() / (res.saturating_sub(1).max(1) as f32);
    let min_displacement = (voxel_step * 0.2).max(1e-4);
    let min_disp2 = min_displacement * min_displacement;
    let effect_radius = kelvinlet_effect_radius(radius, disp_len, min_displacement);

    let brush_min = center - Vec3::splat(effect_radius);
    let brush_max = center + Vec3::splat(effect_radius);
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
                let local_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let warp = kelvinlet_displacement(local_pos - center, displacement, radius);
                if warp.length_squared() <= min_disp2 {
                    continue;
                }

                let idx = VoxelGrid::index(x, y, z, res);
                let sample_pos = local_pos - warp;
                let total_sampled = sample_from_data(grid, snapshot, sample_pos);

                // Subtract analytical child SDF to get back to displacement.
                let world_pos = sculpt_position + inverse_rotate_euler(local_pos, sculpt_rotation);
                let analytical = voxel::evaluate_sdf_tree(scene, child_id, world_pos);
                grid.data[idx] = total_sampled - analytical;
            }
        }
    }

    (z0, z1)
}
/// Wrapper that handles the borrow conflict: need &Scene for evaluate_sdf_tree and &mut VoxelGrid.
/// Temporarily swaps the VoxelGrid out of the scene, applies grab, then swaps it back.
#[allow(clippy::too_many_arguments)]
pub fn apply_grab_to_grid_differential_scene(
    scene: &mut Scene,
    node_id: NodeId,
    snapshot: &[f32],
    center: Vec3,
    radius: f32,
    strength: f32,
    grab_delta: Vec3,
    falloff_mode: &FalloffMode,
    surface_constraint: f32,
    view_dir_local: Vec3,
    child_id: NodeId,
    sculpt_position: Vec3,
    sculpt_rotation: Vec3,
) -> Option<(u32, u32)> {
    // Extract VoxelGrid from the scene node temporarily
    let node = scene.nodes.get_mut(&node_id)?;
    let mut grid = if let NodeData::Sculpt {
        ref mut voxel_grid, ..
    } = node.data
    {
        std::mem::replace(
            voxel_grid,
            VoxelGrid {
                resolution: 1,
                bounds_min: Vec3::ZERO,
                bounds_max: Vec3::ONE,
                is_displacement: true,
                data: vec![0.0],
            },
        )
    } else {
        return None;
    };

    // Now scene doesn't hold our grid, so we can borrow &Scene and &mut grid simultaneously
    let result = apply_grab_to_grid_differential(
        &mut grid,
        snapshot,
        center,
        radius,
        strength,
        grab_delta,
        falloff_mode,
        surface_constraint,
        view_dir_local,
        scene,
        child_id,
        sculpt_position,
        sculpt_rotation,
    );

    // Put the grid back
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        if let NodeData::Sculpt {
            ref mut voxel_grid, ..
        } = node.data
        {
            *voxel_grid = grid;
        }
    }

    Some(result)
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
