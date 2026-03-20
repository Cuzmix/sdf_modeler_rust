use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel::{self, VoxelGrid};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_BRUSH_RADIUS: f32 = 0.3;
pub const DEFAULT_BRUSH_STRENGTH: f32 = 0.05;
pub const DEFAULT_SMOOTH_STRENGTH: f32 = 0.16;
pub const DEFAULT_GRAB_STRENGTH: f32 = 1.0;
pub const DEFAULT_STROKE_SPACING: f32 = 0.08;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BrushMode {
    #[default]
    Add,
    Carve,
    Smooth,
    Flatten,
    Inflate,
    Grab,
}

impl BrushMode {
    pub fn sign(self) -> f32 {
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
    pub fn gpu_mode(self) -> f32 {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FalloffMode {
    Smooth,
    Linear,
    Sharp,
    Flat,
}

impl FalloffMode {
    /// GPU falloff_mode encoding: 0=Smooth, 1=Linear, 2=Sharp, 3=Flat.
    pub fn gpu_mode(self) -> f32 {
        match self {
            Self::Smooth => 0.0,
            Self::Linear => 1.0,
            Self::Sharp => 2.0,
            Self::Flat => 3.0,
        }
    }

    /// Evaluate falloff at normalized distance nt in [0, 1).
    pub fn evaluate(self, nt: f32) -> f32 {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SculptBrushProfile {
    pub radius: f32,
    pub strength: f32,
    pub falloff_mode: FalloffMode,
    pub brush_shape: BrushShape,
    pub smooth_iterations: u32,
    pub lazy_radius: f32,
    pub stroke_spacing: f32,
    pub surface_constraint: f32,
    pub front_faces_only: bool,
}

impl SculptBrushProfile {
    pub fn default_for_mode(mode: BrushMode) -> Self {
        match mode {
            BrushMode::Add => Self {
                radius: DEFAULT_BRUSH_RADIUS,
                strength: DEFAULT_BRUSH_STRENGTH,
                falloff_mode: FalloffMode::Smooth,
                brush_shape: BrushShape::Sphere,
                smooth_iterations: 2,
                lazy_radius: 0.0,
                stroke_spacing: DEFAULT_STROKE_SPACING,
                surface_constraint: 0.0,
                front_faces_only: true,
            },
            BrushMode::Carve => Self {
                strength: DEFAULT_BRUSH_STRENGTH,
                ..Self::default_for_mode(BrushMode::Add)
            },
            BrushMode::Smooth => Self {
                radius: DEFAULT_BRUSH_RADIUS,
                strength: DEFAULT_SMOOTH_STRENGTH,
                falloff_mode: FalloffMode::Smooth,
                brush_shape: BrushShape::Sphere,
                smooth_iterations: 2,
                lazy_radius: 0.0,
                stroke_spacing: DEFAULT_STROKE_SPACING,
                surface_constraint: 0.2,
                front_faces_only: true,
            },
            BrushMode::Flatten => Self {
                radius: DEFAULT_BRUSH_RADIUS,
                strength: DEFAULT_BRUSH_STRENGTH,
                falloff_mode: FalloffMode::Smooth,
                brush_shape: BrushShape::Sphere,
                smooth_iterations: 2,
                lazy_radius: 0.0,
                stroke_spacing: DEFAULT_STROKE_SPACING,
                surface_constraint: 0.0,
                front_faces_only: true,
            },
            BrushMode::Inflate => Self {
                radius: DEFAULT_BRUSH_RADIUS,
                strength: 0.04,
                falloff_mode: FalloffMode::Smooth,
                brush_shape: BrushShape::Sphere,
                smooth_iterations: 2,
                lazy_radius: 0.0,
                stroke_spacing: DEFAULT_STROKE_SPACING,
                surface_constraint: 0.2,
                front_faces_only: true,
            },
            BrushMode::Grab => Self {
                radius: DEFAULT_BRUSH_RADIUS,
                strength: DEFAULT_GRAB_STRENGTH,
                falloff_mode: FalloffMode::Smooth,
                brush_shape: BrushShape::Sphere,
                smooth_iterations: 2,
                lazy_radius: 0.0,
                stroke_spacing: DEFAULT_STROKE_SPACING,
                surface_constraint: 0.0,
                front_faces_only: false,
            },
        }
    }

    pub fn strength_limits(mode: BrushMode) -> (f32, f32) {
        match mode {
            BrushMode::Grab => (0.1, 3.0),
            _ => (0.01, 0.5),
        }
    }

    pub fn clamp_strength_for_mode(&mut self, mode: BrushMode) {
        let (min_strength, max_strength) = Self::strength_limits(mode);
        self.strength = self.strength.clamp(min_strength, max_strength);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SculptBrushProfileSet {
    pub add: SculptBrushProfile,
    pub carve: SculptBrushProfile,
    pub smooth: SculptBrushProfile,
    pub flatten: SculptBrushProfile,
    pub inflate: SculptBrushProfile,
    pub grab: SculptBrushProfile,
}

impl Default for SculptBrushProfileSet {
    fn default() -> Self {
        Self {
            add: SculptBrushProfile::default_for_mode(BrushMode::Add),
            carve: SculptBrushProfile::default_for_mode(BrushMode::Carve),
            smooth: SculptBrushProfile::default_for_mode(BrushMode::Smooth),
            flatten: SculptBrushProfile::default_for_mode(BrushMode::Flatten),
            inflate: SculptBrushProfile::default_for_mode(BrushMode::Inflate),
            grab: SculptBrushProfile::default_for_mode(BrushMode::Grab),
        }
    }
}

impl SculptBrushProfileSet {
    pub fn profile(&self, mode: BrushMode) -> &SculptBrushProfile {
        match mode {
            BrushMode::Add => &self.add,
            BrushMode::Carve => &self.carve,
            BrushMode::Smooth => &self.smooth,
            BrushMode::Flatten => &self.flatten,
            BrushMode::Inflate => &self.inflate,
            BrushMode::Grab => &self.grab,
        }
    }

    pub fn profile_mut(&mut self, mode: BrushMode) -> &mut SculptBrushProfile {
        match mode {
            BrushMode::Add => &mut self.add,
            BrushMode::Carve => &mut self.carve,
            BrushMode::Smooth => &mut self.smooth,
            BrushMode::Flatten => &mut self.flatten,
            BrushMode::Inflate => &mut self.inflate,
            BrushMode::Grab => &mut self.grab,
        }
    }

    pub fn set_radius_for_all(&mut self, radius: f32) {
        for mode in [
            BrushMode::Add,
            BrushMode::Carve,
            BrushMode::Smooth,
            BrushMode::Flatten,
            BrushMode::Inflate,
            BrushMode::Grab,
        ] {
            self.profile_mut(mode).radius = radius;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SculptDetailState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pre_expand_detail_size: Option<f32>,
    #[serde(default)]
    pub detail_limited_after_growth: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SculptStrokeState {
    /// SDF value at brush center when Flatten drag started. Reset on mouse release.
    pub flatten_reference: Option<f32>,
    /// Snapshot of grid data for grab brush (cloned on grab start).
    /// For differential sculpts this stores displacement only; analytical
    /// SDF is sampled on demand during warp write-back.
    pub grab_snapshot: Option<Arc<[f32]>>,
    /// Optional analytical child SDF snapshot for differential grab.
    /// Cached once at stroke start to avoid per-voxel tree evals per frame.
    pub grab_analytical_snapshot: Option<Arc<[f32]>>,
    /// World position where grab stroke started.
    pub grab_start: Option<Vec3>,
    /// Child input node for differential grab (used to subtract analytical SDF on write-back).
    pub grab_child_input: Option<NodeId>,
    /// Deferred repair region for Grab so active drags stay responsive.
    pub pending_grab_repair_region: Option<VoxelEditRegion>,
    /// Shift-smooth inherits the current active brush radius for the duration of a stroke.
    pub temporary_smooth_radius: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SculptSessionData {
    pub selected_brush: BrushMode,
    pub brush_profiles: SculptBrushProfileSet,
    pub symmetry_axis: Option<u8>,
    #[serde(default)]
    pub detail_state: SculptDetailState,
}

impl Default for SculptSessionData {
    fn default() -> Self {
        Self {
            selected_brush: BrushMode::Add,
            brush_profiles: SculptBrushProfileSet::default(),
            symmetry_axis: None,
            detail_state: SculptDetailState::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedSculptState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_node: Option<NodeId>,
    #[serde(default)]
    pub session: SculptSessionData,
}

#[derive(Clone, Debug)]
pub struct ResolvedSculptBrush {
    pub mode: BrushMode,
    pub profile: SculptBrushProfile,
}

#[derive(Clone, Debug)]
pub enum SculptState {
    Inactive {
        session: SculptSessionData,
    },
    Active {
        node_id: NodeId,
        session: SculptSessionData,
        stroke_state: SculptStrokeState,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelEditRegion {
    pub x0: u32,
    pub y0: u32,
    pub z0: u32,
    pub x1: u32,
    pub y1: u32,
    pub z1: u32,
}

impl VoxelEditRegion {
    pub fn merge(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            z0: self.z0.min(other.z0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
            z1: self.z1.max(other.z1),
        }
    }

    pub fn padded(self, padding: u32, resolution: u32) -> Self {
        let max = resolution.saturating_sub(1);
        Self {
            x0: self.x0.saturating_sub(padding),
            y0: self.y0.saturating_sub(padding),
            z0: self.z0.saturating_sub(padding),
            x1: (self.x1 + padding).min(max),
            y1: (self.y1 + padding).min(max),
            z1: (self.z1 + padding).min(max),
        }
    }

    pub fn z_range(self) -> (u32, u32) {
        (self.z0, self.z1)
    }
}

impl SculptState {
    fn default_session_with_radius(extent: f32) -> SculptSessionData {
        let radius = (extent * 0.15).clamp(0.05, 2.0);
        let mut session = SculptSessionData::default();
        session.brush_profiles.set_radius_for_all(radius);
        session
    }

    pub fn new_inactive() -> Self {
        Self::Inactive {
            session: SculptSessionData::default(),
        }
    }

    /// Create a new Active sculpt state with default brush settings.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new_active(node_id: NodeId) -> Self {
        Self::Active {
            node_id,
            session: SculptSessionData::default(),
            stroke_state: SculptStrokeState::default(),
        }
    }

    /// Create Active state with adaptive brush radius based on object bounds.
    /// `extent` is the average half-extent of the bounding box (e.g. from `compute_bounds()`).
    #[allow(dead_code)]
    pub fn new_active_with_radius(node_id: NodeId, extent: f32) -> Self {
        Self::Active {
            node_id,
            session: Self::default_session_with_radius(extent),
            stroke_state: SculptStrokeState::default(),
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

    pub fn activate_preserving_session(&mut self, node_id: NodeId, extent: Option<f32>) {
        let mut session = match std::mem::replace(self, Self::new_inactive()) {
            Self::Inactive { session } | Self::Active { session, .. } => session,
        };
        if session == SculptSessionData::default() {
            if let Some(extent) = extent {
                session = Self::default_session_with_radius(extent);
            }
        }
        *self = Self::Active {
            node_id,
            session,
            stroke_state: SculptStrokeState::default(),
        };
    }

    pub fn deactivate(&mut self) {
        let session = match std::mem::replace(self, Self::new_inactive()) {
            Self::Inactive { session } | Self::Active { session, .. } => session,
        };
        *self = Self::Inactive { session };
    }

    pub fn session(&self) -> &SculptSessionData {
        match self {
            Self::Inactive { session } | Self::Active { session, .. } => session,
        }
    }

    pub fn session_mut(&mut self) -> &mut SculptSessionData {
        match self {
            Self::Inactive { session } | Self::Active { session, .. } => session,
        }
    }

    pub fn stroke_state(&self) -> Option<&SculptStrokeState> {
        match self {
            Self::Active { stroke_state, .. } => Some(stroke_state),
            Self::Inactive { .. } => None,
        }
    }

    pub fn stroke_state_mut(&mut self) -> Option<&mut SculptStrokeState> {
        match self {
            Self::Active { stroke_state, .. } => Some(stroke_state),
            Self::Inactive { .. } => None,
        }
    }

    pub fn clear_stroke_state(&mut self) {
        if let Some(stroke_state) = self.stroke_state_mut() {
            *stroke_state = SculptStrokeState::default();
        }
    }

    pub fn selected_brush(&self) -> BrushMode {
        self.session().selected_brush
    }

    pub fn set_selected_brush(&mut self, mode: BrushMode) {
        self.session_mut().selected_brush = mode;
    }

    pub fn profile(&self, mode: BrushMode) -> &SculptBrushProfile {
        self.session().brush_profiles.profile(mode)
    }

    pub fn profile_mut(&mut self, mode: BrushMode) -> &mut SculptBrushProfile {
        self.session_mut().brush_profiles.profile_mut(mode)
    }

    pub fn selected_profile(&self) -> &SculptBrushProfile {
        self.profile(self.selected_brush())
    }

    pub fn selected_profile_mut(&mut self) -> &mut SculptBrushProfile {
        let selected_brush = self.selected_brush();
        self.profile_mut(selected_brush)
    }

    pub fn symmetry_axis(&self) -> Option<u8> {
        self.session().symmetry_axis
    }

    pub fn set_symmetry_axis(&mut self, axis: Option<u8>) {
        self.session_mut().symmetry_axis = axis;
    }

    pub fn detail_state(&self) -> &SculptDetailState {
        &self.session().detail_state
    }

    pub fn detail_state_mut(&mut self) -> &mut SculptDetailState {
        &mut self.session_mut().detail_state
    }

    pub fn to_persisted(&self) -> PersistedSculptState {
        PersistedSculptState {
            active_node: self.active_node(),
            session: self.session().clone(),
        }
    }

    pub fn from_persisted(persisted: PersistedSculptState, scene: &Scene) -> Self {
        if let Some(active_node) = persisted.active_node {
            if scene
                .nodes
                .get(&active_node)
                .is_some_and(|node| matches!(node.data, NodeData::Sculpt { .. }))
            {
                return Self::Active {
                    node_id: active_node,
                    session: persisted.session,
                    stroke_state: SculptStrokeState::default(),
                };
            }
        }
        Self::Inactive {
            session: persisted.session,
        }
    }

    pub fn effective_brush_mode(&self, ctrl: bool, shift: bool) -> BrushMode {
        let base = self.selected_brush();
        if shift {
            return BrushMode::Smooth;
        }
        if ctrl {
            return match base {
                BrushMode::Add => BrushMode::Carve,
                BrushMode::Carve => BrushMode::Add,
                BrushMode::Inflate => BrushMode::Carve,
                other => other,
            };
        }
        base
    }

    pub fn preview_radius_with_modifiers(&self, ctrl: bool, shift: bool) -> f32 {
        let base_brush = self.selected_brush();
        if shift && base_brush != BrushMode::Smooth {
            return self.profile(base_brush).radius;
        }
        self.profile(self.effective_brush_mode(ctrl, shift)).radius
    }

    pub fn resolved_brush_for_stroke(&mut self, ctrl: bool, shift: bool) -> ResolvedSculptBrush {
        let selected_brush = self.selected_brush();
        let effective_mode = self.effective_brush_mode(ctrl, shift);
        let active_radius = self.profile(selected_brush).radius;
        let mut profile = self.profile(effective_mode).clone();

        if shift && selected_brush != BrushMode::Smooth {
            if let Some(stroke_state) = self.stroke_state_mut() {
                if stroke_state.temporary_smooth_radius.is_none() {
                    stroke_state.temporary_smooth_radius = Some(active_radius);
                }
                if let Some(temporary_radius) = stroke_state.temporary_smooth_radius {
                    profile.radius = temporary_radius;
                }
            } else {
                profile.radius = active_radius;
            }
        }

        ResolvedSculptBrush {
            mode: effective_mode,
            profile,
        }
    }
}

// ---------------------------------------------------------------------------
// Brush application
// ---------------------------------------------------------------------------

/// Apply brush directly in sculpt-local space.
#[allow(clippy::too_many_arguments)]
pub fn apply_brush_local(
    voxel_grid: &mut VoxelGrid,
    local_hit: Vec3,
    local_view_dir: Vec3,
    brush_mode: &BrushMode,
    brush_radius: f32,
    brush_strength: f32,
    falloff_mode: &FalloffMode,
    brush_shape: &BrushShape,
    smooth_iterations: u32,
    flatten_ref: f32,
    surface_constraint: f32,
) -> VoxelEditRegion {
    match brush_mode {
        BrushMode::Smooth => apply_smooth_to_grid(
            voxel_grid,
            local_hit,
            brush_radius,
            brush_strength,
            falloff_mode,
            brush_shape,
            smooth_iterations,
            surface_constraint,
            local_view_dir,
        ),
        _ => apply_brush_to_grid(
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
        ),
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

#[allow(clippy::too_many_arguments)]
fn smooth_region_index(
    x: u32,
    y: u32,
    z: u32,
    x0: u32,
    y0: u32,
    z0: u32,
    size_x: usize,
    size_y: usize,
) -> usize {
    let lx = (x - x0) as usize;
    let ly = (y - y0) as usize;
    let lz = (z - z0) as usize;
    (lz * size_y + ly) * size_x + lx
}

#[allow(clippy::too_many_arguments)]
fn sample_smooth_pass_value(
    snapshot: &[f32],
    grid_data: &[f32],
    x: u32,
    y: u32,
    z: u32,
    res: u32,
    x0: u32,
    y0: u32,
    z0: u32,
    x1: u32,
    y1: u32,
    z1: u32,
    size_x: usize,
    size_y: usize,
) -> f32 {
    if x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1 {
        let ridx = smooth_region_index(x, y, z, x0, y0, z0, size_x, size_y);
        snapshot[ridx]
    } else {
        grid_data[VoxelGrid::index(x, y, z, res)]
    }
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
) -> VoxelEditRegion {
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

    VoxelEditRegion {
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
    }
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
) -> VoxelEditRegion {
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

    let size_x = (x1 - x0 + 1) as usize;
    let size_y = (y1 - y0 + 1) as usize;
    let size_z = (z1 - z0 + 1) as usize;
    let mut snapshot = vec![0.0f32; size_x * size_y * size_z];

    // Taubin smoothing (lambda/mu) preserves volume better than pure Laplacian.
    // A single "iteration" is two passes.
    const LAMBDA: f32 = 0.5;
    const MU: f32 = -0.53;

    for _ in 0..iterations {
        for pass_scale in [LAMBDA, MU] {
            // Snapshot only the brush AABB region instead of cloning the full grid.
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let ridx = smooth_region_index(x, y, z, x0, y0, z0, size_x, size_y);
                        snapshot[ridx] = grid.data[VoxelGrid::index(x, y, z, res)];
                    }
                }
            }

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

                        // 6-neighbor Laplacian average (clamped at grid edges).
                        let xm = if x > 0 { x - 1 } else { x };
                        let xp = if x < res - 1 { x + 1 } else { x };
                        let ym = if y > 0 { y - 1 } else { y };
                        let yp = if y < res - 1 { y + 1 } else { y };
                        let zm = if z > 0 { z - 1 } else { z };
                        let zp = if z < res - 1 { z + 1 } else { z };

                        let avg = (sample_smooth_pass_value(
                            &snapshot, &grid.data, xm, y, z, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        ) + sample_smooth_pass_value(
                            &snapshot, &grid.data, xp, y, z, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        ) + sample_smooth_pass_value(
                            &snapshot, &grid.data, x, ym, z, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        ) + sample_smooth_pass_value(
                            &snapshot, &grid.data, x, yp, z, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        ) + sample_smooth_pass_value(
                            &snapshot, &grid.data, x, y, zm, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        ) + sample_smooth_pass_value(
                            &snapshot, &grid.data, x, y, zp, res, x0, y0, z0, x1, y1, z1, size_x,
                            size_y,
                        )) / 6.0;

                        let idx = VoxelGrid::index(x, y, z, res);
                        let current =
                            snapshot[smooth_region_index(x, y, z, x0, y0, z0, size_x, size_y)];
                        let sf = surface_factor(current, radius, surface_constraint);
                        let blend = (falloff * strength * sf * pass_scale).clamp(-1.0, 1.0);
                        grid.data[idx] = current + (avg - current) * blend;
                    }
                }
            }
        }
    }

    VoxelEditRegion {
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
    }
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
) -> VoxelEditRegion {
    let res = grid.resolution;
    let displacement = grab_delta * strength;
    let disp_len = displacement.length();
    if disp_len <= 1e-6 || radius <= 1e-6 {
        let z = grid.world_to_grid(center).z.clamp(0.0, (res - 1) as f32) as u32;
        let xy = grid
            .world_to_grid(center)
            .clamp(Vec3::ZERO, Vec3::splat((res - 1) as f32));
        return VoxelEditRegion {
            x0: xy.x as u32,
            y0: xy.y as u32,
            z0: z,
            x1: xy.x as u32,
            y1: xy.y as u32,
            z1: z,
        };
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

    VoxelEditRegion {
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
    }
}
/// Apply grab brush for differential sculpts using a displacement snapshot.
/// Reconstructs total SDF on demand: displacement(sample) + analytical(sample).
/// Uses cached analytical snapshot when available.
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
    analytical_snapshot: Option<&[f32]>,
    scene: &Scene,
    child_id: NodeId,
    sculpt_position: Vec3,
    sculpt_rotation: Vec3,
) -> VoxelEditRegion {
    let res = grid.resolution;
    let analytical_snapshot = analytical_snapshot.filter(|data| data.len() == grid.data.len());
    let displacement = grab_delta * strength;
    let disp_len = displacement.length();
    if disp_len <= 1e-6 || radius <= 1e-6 {
        let z = grid.world_to_grid(center).z.clamp(0.0, (res - 1) as f32) as u32;
        let xy = grid
            .world_to_grid(center)
            .clamp(Vec3::ZERO, Vec3::splat((res - 1) as f32));
        return VoxelEditRegion {
            x0: xy.x as u32,
            y0: xy.y as u32,
            z0: z,
            x1: xy.x as u32,
            y1: xy.y as u32,
            z1: z,
        };
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
                let displacement_sampled = sample_from_data(grid, snapshot, sample_pos);

                let analytical_sample = if let Some(snapshot) = analytical_snapshot {
                    sample_from_data(grid, snapshot, sample_pos)
                } else {
                    let sample_world =
                        sculpt_position + inverse_rotate_euler(sample_pos, sculpt_rotation);
                    voxel::evaluate_sdf_tree(scene, child_id, sample_world)
                };
                let total_sampled = displacement_sampled + analytical_sample;

                // Subtract analytical child SDF at destination to write back displacement.
                let analytical_dest = if let Some(snapshot) = analytical_snapshot {
                    snapshot[idx]
                } else {
                    let world_pos =
                        sculpt_position + inverse_rotate_euler(local_pos, sculpt_rotation);
                    voxel::evaluate_sdf_tree(scene, child_id, world_pos)
                };
                grid.data[idx] = total_sampled - analytical_dest;
            }
        }
    }

    VoxelEditRegion {
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
    }
}
/// Wrapper that handles the borrow conflict: need &Scene for evaluate_sdf_tree and &mut VoxelGrid.
/// Temporarily swaps the VoxelGrid out of the scene, applies grab, then swaps it back.
#[allow(clippy::too_many_arguments)]
pub fn apply_grab_to_grid_differential_scene(
    scene: &mut Scene,
    node_id: NodeId,
    snapshot: &[f32],
    analytical_snapshot: Option<&[f32]>,
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
) -> Option<VoxelEditRegion> {
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
        analytical_snapshot,
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

fn axis_voxel_step(grid: &VoxelGrid) -> Vec3 {
    let extent = grid.bounds_max - grid.bounds_min;
    let denom = grid.resolution.saturating_sub(1).max(1) as f32;
    Vec3::new(
        (extent.x / denom).max(1e-6),
        (extent.y / denom).max(1e-6),
        (extent.z / denom).max(1e-6),
    )
}

fn repair_region_index(
    x: u32,
    y: u32,
    z: u32,
    region: VoxelEditRegion,
    size_x: usize,
    size_y: usize,
) -> usize {
    let lx = (x - region.x0) as usize;
    let ly = (y - region.y0) as usize;
    let lz = (z - region.z0) as usize;
    (lz * size_y + ly) * size_x + lx
}

fn field_index(x: usize, y: usize, z: usize, size_x: usize, size_y: usize) -> usize {
    (z * size_y + y) * size_x + x
}

fn clamped_field_value(
    values: &[f32],
    size_x: usize,
    size_y: usize,
    size_z: usize,
    x: isize,
    y: isize,
    z: isize,
) -> f32 {
    let cx = x.clamp(0, size_x.saturating_sub(1) as isize) as usize;
    let cy = y.clamp(0, size_y.saturating_sub(1) as isize) as usize;
    let cz = z.clamp(0, size_z.saturating_sub(1) as isize) as usize;
    values[field_index(cx, cy, cz, size_x, size_y)]
}

fn has_sign_change_neighbor(
    values: &[f32],
    size_x: usize,
    size_y: usize,
    size_z: usize,
    x: usize,
    y: usize,
    z: usize,
) -> bool {
    let center = values[field_index(x, y, z, size_x, size_y)];
    let center_sign = center.signum();
    for (dx, dy, dz) in [
        (-1isize, 0isize, 0isize),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ] {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        let nz = z as isize + dz;
        if nx < 0
            || ny < 0
            || nz < 0
            || nx >= size_x as isize
            || ny >= size_y as isize
            || nz >= size_z as isize
        {
            continue;
        }
        let neighbor = values[field_index(nx as usize, ny as usize, nz as usize, size_x, size_y)];
        if neighbor.signum() != center_sign {
            return true;
        }
    }
    false
}

fn godunov_gradient_magnitude(
    values: &[f32],
    dimensions: (usize, usize, usize),
    x: usize,
    y: usize,
    z: usize,
    step: Vec3,
    sign: f32,
) -> f32 {
    let (size_x, size_y, size_z) = dimensions;
    let center = values[field_index(x, y, z, size_x, size_y)];
    let left = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize - 1,
        y as isize,
        z as isize,
    );
    let right = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize + 1,
        y as isize,
        z as isize,
    );
    let down = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize,
        y as isize - 1,
        z as isize,
    );
    let up = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize,
        y as isize + 1,
        z as isize,
    );
    let back = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize,
        y as isize,
        z as isize - 1,
    );
    let front = clamped_field_value(
        values,
        size_x,
        size_y,
        size_z,
        x as isize,
        y as isize,
        z as isize + 1,
    );

    let dx_minus = (center - left) / step.x;
    let dx_plus = (right - center) / step.x;
    let dy_minus = (center - down) / step.y;
    let dy_plus = (up - center) / step.y;
    let dz_minus = (center - back) / step.z;
    let dz_plus = (front - center) / step.z;

    let axis_term = |minus: f32, plus: f32| {
        if sign >= 0.0 {
            (minus.max(0.0).powi(2)).max(plus.min(0.0).powi(2))
        } else {
            (minus.min(0.0).powi(2)).max(plus.max(0.0).powi(2))
        }
    };

    (axis_term(dx_minus, dx_plus) + axis_term(dy_minus, dy_plus) + axis_term(dz_minus, dz_plus))
        .sqrt()
}

fn reinitialize_signed_distance_field(
    values: &mut [f32],
    size_x: usize,
    size_y: usize,
    size_z: usize,
    step: Vec3,
) -> bool {
    let count = values.len();
    if count == 0 {
        return false;
    }

    let original = values.to_vec();
    let mut current = original.clone();
    let mut next = original.clone();
    let min_step = step.min_element().max(1e-6);
    let interface_band = min_step * 0.75;
    let pseudo_dt = min_step * 0.3;
    let max_dim = size_x.max(size_y).max(size_z);
    let iterations = max_dim.clamp(6, 24);
    let mut locked = vec![false; count];
    let mut seed_count = 0usize;

    for z in 0..size_z {
        for y in 0..size_y {
            for x in 0..size_x {
                let idx = field_index(x, y, z, size_x, size_y);
                let phi0 = original[idx];
                let is_border = x == 0
                    || y == 0
                    || z == 0
                    || x + 1 == size_x
                    || y + 1 == size_y
                    || z + 1 == size_z;
                let is_surface_anchor = phi0.abs() <= interface_band
                    || has_sign_change_neighbor(&original, size_x, size_y, size_z, x, y, z);
                if is_border || is_surface_anchor {
                    locked[idx] = true;
                    seed_count += usize::from(is_surface_anchor);
                }
            }
        }
    }

    if seed_count == 0 {
        return false;
    }

    for _ in 0..iterations {
        for z in 0..size_z {
            for y in 0..size_y {
                for x in 0..size_x {
                    let idx = field_index(x, y, z, size_x, size_y);
                    let phi0 = original[idx];
                    if locked[idx] {
                        next[idx] = phi0;
                        continue;
                    }

                    let sign = phi0 / (phi0 * phi0 + min_step * min_step).sqrt();
                    let grad = godunov_gradient_magnitude(
                        &current,
                        (size_x, size_y, size_z),
                        x,
                        y,
                        z,
                        step,
                        sign,
                    );
                    let updated = current[idx] - pseudo_dt * sign * (grad - 1.0);
                    next[idx] = if phi0 >= 0.0 {
                        updated.max(0.0)
                    } else {
                        updated.min(0.0)
                    };
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    values.copy_from_slice(&current);
    true
}

fn repair_total_sdf_region(
    grid: &mut VoxelGrid,
    dirty_region: VoxelEditRegion,
    padding: u32,
) -> VoxelEditRegion {
    let region = dirty_region.padded(padding, grid.resolution);
    let size_x = (region.x1 - region.x0 + 1) as usize;
    let size_y = (region.y1 - region.y0 + 1) as usize;
    let size_z = (region.z1 - region.z0 + 1) as usize;

    let mut values = vec![0.0f32; size_x * size_y * size_z];
    for z in region.z0..=region.z1 {
        for y in region.y0..=region.y1 {
            for x in region.x0..=region.x1 {
                let local_idx = repair_region_index(x, y, z, region, size_x, size_y);
                values[local_idx] = grid.data[VoxelGrid::index(x, y, z, grid.resolution)];
            }
        }
    }

    if !reinitialize_signed_distance_field(
        &mut values,
        size_x,
        size_y,
        size_z,
        axis_voxel_step(grid),
    ) {
        return dirty_region;
    }

    for z in region.z0..=region.z1 {
        for y in region.y0..=region.y1 {
            for x in region.x0..=region.x1 {
                let local_idx = repair_region_index(x, y, z, region, size_x, size_y);
                let global_idx = VoxelGrid::index(x, y, z, grid.resolution);
                grid.data[global_idx] = values[local_idx];
            }
        }
    }

    region
}

#[allow(clippy::too_many_arguments)]
fn repair_differential_sdf_region(
    grid: &mut VoxelGrid,
    scene: &Scene,
    child_id: NodeId,
    sculpt_position: Vec3,
    sculpt_rotation: Vec3,
    layer_intensity: f32,
    dirty_region: VoxelEditRegion,
    padding: u32,
) -> VoxelEditRegion {
    if layer_intensity.abs() <= 1e-5 {
        return dirty_region;
    }

    let region = dirty_region.padded(padding, grid.resolution);
    let size_x = (region.x1 - region.x0 + 1) as usize;
    let size_y = (region.y1 - region.y0 + 1) as usize;
    let size_z = (region.z1 - region.z0 + 1) as usize;

    let mut analytical = vec![0.0f32; size_x * size_y * size_z];
    let mut total_values = vec![0.0f32; size_x * size_y * size_z];

    for z in region.z0..=region.z1 {
        for y in region.y0..=region.y1 {
            for x in region.x0..=region.x1 {
                let local_idx = repair_region_index(x, y, z, region, size_x, size_y);
                let local_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let world_pos = sculpt_position + inverse_rotate_euler(local_pos, sculpt_rotation);
                let analytical_value = voxel::evaluate_sdf_tree(scene, child_id, world_pos);
                let global_idx = VoxelGrid::index(x, y, z, grid.resolution);
                analytical[local_idx] = analytical_value;
                total_values[local_idx] =
                    analytical_value + grid.data[global_idx] * layer_intensity;
            }
        }
    }

    if !reinitialize_signed_distance_field(
        &mut total_values,
        size_x,
        size_y,
        size_z,
        axis_voxel_step(grid),
    ) {
        return dirty_region;
    }

    for z in region.z0..=region.z1 {
        for y in region.y0..=region.y1 {
            for x in region.x0..=region.x1 {
                let local_idx = repair_region_index(x, y, z, region, size_x, size_y);
                let global_idx = VoxelGrid::index(x, y, z, grid.resolution);
                grid.data[global_idx] =
                    (total_values[local_idx] - analytical[local_idx]) / layer_intensity;
            }
        }
    }

    region
}

pub fn repair_sdf_region_scene(
    scene: &mut Scene,
    node_id: NodeId,
    dirty_region: VoxelEditRegion,
    padding: u32,
) -> Option<VoxelEditRegion> {
    let (child_id, sculpt_position, sculpt_rotation, layer_intensity) = {
        let node = scene.nodes.get(&node_id)?;
        let NodeData::Sculpt {
            input,
            position,
            rotation,
            layer_intensity,
            ..
        } = node.data
        else {
            return None;
        };
        (input, position, rotation, layer_intensity)
    };

    if let Some(child_id) = child_id {
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

        let repaired = repair_differential_sdf_region(
            &mut grid,
            scene,
            child_id,
            sculpt_position,
            sculpt_rotation,
            layer_intensity,
            dirty_region,
            padding,
        );

        if let Some(node) = scene.nodes.get_mut(&node_id) {
            if let NodeData::Sculpt {
                ref mut voxel_grid, ..
            } = node.data
            {
                *voxel_grid = grid;
            }
        }
        Some(repaired)
    } else {
        let node = scene.nodes.get_mut(&node_id)?;
        let NodeData::Sculpt {
            ref mut voxel_grid, ..
        } = node.data
        else {
            return None;
        };
        Some(repair_total_sdf_region(voxel_grid, dirty_region, padding))
    }
}

/// Build an analytical child-SDF snapshot in sculpt-local voxel space.
/// Used to amortize differential grab cost over the whole stroke.
pub fn build_analytical_snapshot(
    grid: &VoxelGrid,
    scene: &Scene,
    child_id: NodeId,
    sculpt_position: Vec3,
    sculpt_rotation: Vec3,
) -> Vec<f32> {
    let res = grid.resolution;
    let mut snapshot = vec![0.0f32; grid.data.len()];
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let idx = VoxelGrid::index(x, y, z, res);
                let local_pos = grid.grid_to_world(x as f32, y as f32, z as f32);
                let world_pos = sculpt_position + inverse_rotate_euler(local_pos, sculpt_rotation);
                snapshot[idx] = voxel::evaluate_sdf_tree(scene, child_id, world_pos);
            }
        }
    }
    snapshot
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{MaterialParams, NodeData, Scene, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;

    fn scene_with_sculpt_node() -> (Scene, NodeId) {
        let mut scene = Scene::new();
        let input = scene.add_node(
            "Sphere".into(),
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: MaterialParams::default(),
                voxel_grid: None,
            },
        );
        let sculpt = scene.create_sculpt(
            input,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ONE,
            VoxelGrid::new_displacement(16, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );
        (scene, sculpt)
    }

    #[test]
    fn voxel_edit_region_merge_and_padding_work() {
        let a = VoxelEditRegion {
            x0: 2,
            y0: 3,
            z0: 4,
            x1: 5,
            y1: 6,
            z1: 7,
        };
        let b = VoxelEditRegion {
            x0: 1,
            y0: 4,
            z0: 2,
            x1: 8,
            y1: 5,
            z1: 9,
        };

        let merged = a.merge(b);
        assert_eq!(
            merged,
            VoxelEditRegion {
                x0: 1,
                y0: 3,
                z0: 2,
                x1: 8,
                y1: 6,
                z1: 9,
            }
        );

        let padded = merged.padded(2, 10);
        assert_eq!(
            padded,
            VoxelEditRegion {
                x0: 0,
                y0: 1,
                z0: 0,
                x1: 9,
                y1: 8,
                z1: 9,
            }
        );
    }

    #[test]
    fn repair_signed_distance_values_keeps_simple_line_field_ordered() {
        let mut values = vec![-1.0, 0.0, 1.0];
        let repaired = reinitialize_signed_distance_field(&mut values, 3, 1, 1, Vec3::ONE);
        assert!(repaired);
        assert!(values[0] < 0.0);
        assert!(values[1].abs() < 1e-5);
        assert!(values[2] > 0.0);
        assert!(values[0].abs() <= values[2].abs() + 1e-5);
    }

    #[test]
    fn repair_signed_distance_values_reduces_obvious_drift() {
        let line = [
            -2.5_f32, -1.9_f32, -0.5_f32, 0.5_f32, 2.1_f32, 3.0_f32, 3.5_f32,
        ];
        let mut values = vec![0.0_f32; 7 * 3 * 3];
        for z in 0..3 {
            for y in 0..3 {
                for (x, value) in line.iter().enumerate() {
                    values[field_index(x, y, z, 7, 3)] = *value;
                }
            }
        }
        let target_error = |samples: &[f32]| -> f32 {
            let line_sample = |x: usize| samples[field_index(x, 1, 1, 7, 3)];
            (line_sample(1) + 1.5).abs()
                + (line_sample(4) - 1.5).abs()
                + (line_sample(5) - 2.5).abs()
        };
        let before_target_error = target_error(&values);
        let repaired = reinitialize_signed_distance_field(&mut values, 7, 3, 3, Vec3::ONE);
        assert!(repaired);
        let after_target_error = target_error(&values);
        assert!(after_target_error < before_target_error);
        let line_sample = |x: usize| values[field_index(x, 1, 1, 7, 3)];
        assert!(line_sample(0) < 0.0 && line_sample(1) < 0.0 && line_sample(2) < 0.0);
        assert!(line_sample(3) > 0.0);
        assert!(line_sample(4) > 0.0 && line_sample(5) > 0.0 && line_sample(6) > 0.0);
    }

    #[test]
    fn brush_profiles_keep_independent_settings() {
        let mut sculpt_state = SculptState::new_inactive();
        sculpt_state.profile_mut(BrushMode::Add).radius = 0.6;
        sculpt_state.profile_mut(BrushMode::Smooth).radius = 1.1;
        sculpt_state.profile_mut(BrushMode::Grab).strength = 2.4;

        assert!((sculpt_state.profile(BrushMode::Add).radius - 0.6).abs() < 1e-5);
        assert!((sculpt_state.profile(BrushMode::Smooth).radius - 1.1).abs() < 1e-5);
        assert!((sculpt_state.profile(BrushMode::Grab).strength - 2.4).abs() < 1e-5);
    }

    #[test]
    fn shift_smooth_uses_smooth_profile_but_keeps_active_radius() {
        let mut sculpt_state = SculptState::new_active(7);
        sculpt_state.set_selected_brush(BrushMode::Add);
        sculpt_state.profile_mut(BrushMode::Add).radius = 0.85;
        sculpt_state.profile_mut(BrushMode::Smooth).radius = 0.25;
        sculpt_state.profile_mut(BrushMode::Smooth).strength = 0.21;

        let resolved = sculpt_state.resolved_brush_for_stroke(false, true);
        assert_eq!(resolved.mode, BrushMode::Smooth);
        assert!((resolved.profile.radius - 0.85).abs() < 1e-5);
        assert!((resolved.profile.strength - 0.21).abs() < 1e-5);
    }

    #[test]
    fn ctrl_invert_swaps_add_and_carve_but_preserves_other_modes() {
        let mut sculpt_state = SculptState::new_inactive();
        sculpt_state.set_selected_brush(BrushMode::Add);
        assert_eq!(sculpt_state.effective_brush_mode(true, false), BrushMode::Carve);

        sculpt_state.set_selected_brush(BrushMode::Carve);
        assert_eq!(sculpt_state.effective_brush_mode(true, false), BrushMode::Add);

        sculpt_state.set_selected_brush(BrushMode::Grab);
        assert_eq!(sculpt_state.effective_brush_mode(true, false), BrushMode::Grab);
    }

    #[test]
    fn persisted_state_round_trips_active_node_and_session() {
        let (scene, sculpt_node) = scene_with_sculpt_node();
        let mut sculpt_state = SculptState::new_active(sculpt_node);
        sculpt_state.set_selected_brush(BrushMode::Flatten);
        sculpt_state.profile_mut(BrushMode::Flatten).strength = 0.19;
        sculpt_state.detail_state_mut().last_pre_expand_detail_size = Some(0.125);
        sculpt_state.detail_state_mut().detail_limited_after_growth = true;

        let restored = SculptState::from_persisted(sculpt_state.to_persisted(), &scene);
        assert_eq!(restored.active_node(), Some(sculpt_node));
        assert_eq!(restored.selected_brush(), BrushMode::Flatten);
        assert!((restored.profile(BrushMode::Flatten).strength - 0.19).abs() < 1e-5);
        assert_eq!(
            restored.detail_state().last_pre_expand_detail_size,
            Some(0.125)
        );
        assert!(restored.detail_state().detail_limited_after_growth);
    }
}
