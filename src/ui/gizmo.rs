use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use glam::{Mat4, Quat, Vec3, Vec4};

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::SnapConfig;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Translate => "Move",
            Self::Rotate => "Rotate",
            Self::Scale => "Scale",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoSpace {
    Local,
    World,
}

impl GizmoSpace {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Local => "Local",
            Self::World => "World",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    fn color(&self) -> Color32 {
        match self {
            Self::X => COLOR_X,
            Self::Y => COLOR_Y,
            Self::Z => COLOR_Z,
        }
    }

    fn hover_color(&self) -> Color32 {
        match self {
            Self::X => COLOR_X_HOVER,
            Self::Y => COLOR_Y_HOVER,
            Self::Z => COLOR_Z_HOVER,
        }
    }

    fn euler_index(&self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

const ALL_AXES: [GizmoAxis; 3] = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];

fn axis_color(
    axis: &GizmoAxis,
    active: &Option<GizmoAxis>,
    hovered: &Option<GizmoAxis>,
) -> Color32 {
    if active.as_ref() == Some(axis) || hovered.as_ref() == Some(axis) {
        axis.hover_color()
    } else {
        axis.color()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum GizmoState {
    Idle,
    DraggingSingle {
        axis: GizmoAxis,
        node_id: NodeId,
        _start_screen_pos: Pos2,
        _start_world_pos: Vec3,
        _start_rotation: Vec3,
        _start_scale: Vec3,
    },
    DraggingMulti {
        axis: GizmoAxis,
        drag_session: GizmoDragSession,
    },
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct GizmoDragSession {
    start_center_world: Vec3,
    start_origin_screen: Pos2,
    start_pivot_offset: Vec3,
    axis_directions: [Vec3; 3],
    targets: Vec<GizmoTarget>,
    accumulated_drag_delta: Vec2,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AXIS_LENGTH: f32 = 1.2;
const HIT_THRESHOLD: f32 = 12.0;
const ARROW_SIZE: f32 = 8.0;
const AXIS_STROKE_WIDTH: f32 = 2.5;
const SCALE_BOX_SIZE: f32 = 6.0;
const RING_RADIUS: f32 = 1.0;
const RING_SEGMENTS: usize = 48;
const RING_HIT_THRESHOLD: f32 = 14.0;
const TRANSLATE_SENSITIVITY: f32 = 0.003;
const SCALE_SENSITIVITY: f32 = 0.005;
const ROTATE_SENSITIVITY: f32 = 0.01;

const COLOR_X: Color32 = Color32::from_rgb(220, 60, 60);
const COLOR_Y: Color32 = Color32::from_rgb(60, 200, 60);
const COLOR_Z: Color32 = Color32::from_rgb(60, 100, 240);
const COLOR_X_HOVER: Color32 = Color32::from_rgb(255, 120, 120);
const COLOR_Y_HOVER: Color32 = Color32::from_rgb(120, 255, 120);
const COLOR_Z_HOVER: Color32 = Color32::from_rgb(120, 160, 255);

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
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

/// Convert Euler XYZ rotation (radians) to quaternion.
/// Shader applies X then Y then Z, so combined = Qz * Qy * Qx.
fn euler_to_quat(r: Vec3) -> Quat {
    Quat::from_euler(glam::EulerRot::ZYX, r.z, r.y, r.x)
}

/// Convert quaternion back to Euler XYZ rotation (radians),
/// choosing the representation closest to `prev` to avoid jumps at singularities.
fn quat_to_euler_stable(q: Quat, prev: Vec3) -> Vec3 {
    use std::f32::consts::{PI, TAU};

    fn wrap_near(angle: f32, reference: f32) -> f32 {
        let mut a = angle;
        while a - reference > PI {
            a -= TAU;
        }
        while a - reference < -PI {
            a += TAU;
        }
        a
    }

    fn normalize_near(v: Vec3, prev: Vec3) -> Vec3 {
        Vec3::new(
            wrap_near(v.x, prev.x),
            wrap_near(v.y, prev.y),
            wrap_near(v.z, prev.z),
        )
    }

    let (rz, ry, rx) = q.to_euler(glam::EulerRot::ZYX);
    let a = normalize_near(Vec3::new(rx, ry, rz), prev);

    // Alternative Euler representation for the same rotation
    let b = normalize_near(Vec3::new(rx + PI, PI - ry, rz + PI), prev);

    if (a - prev).length_squared() <= (b - prev).length_squared() {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// Projection helpers
// ---------------------------------------------------------------------------

pub fn world_to_screen(world_pos: Vec3, view_proj: &Mat4, viewport_rect: Rect) -> Option<Pos2> {
    let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let x = viewport_rect.min.x + (ndc.x * 0.5 + 0.5) * viewport_rect.width();
    let y = viewport_rect.min.y + (-ndc.y * 0.5 + 0.5) * viewport_rect.height();
    Some(Pos2::new(x, y))
}

fn point_to_segment_dist(point: Pos2, seg_start: Pos2, seg_end: Pos2) -> f32 {
    let ab = seg_end - seg_start;
    let ap = point - seg_start;
    let len_sq = ab.dot(ab);
    if len_sq < 0.001 {
        return point.distance(seg_start);
    }
    let t = (ab.dot(ap) / len_sq).clamp(0.0, 1.0);
    let closest = seg_start + ab * t;
    point.distance(closest)
}

fn screen_axis_dir(
    origin: Vec3,
    axis_dir: Vec3,
    origin_screen: Pos2,
    vp: &Mat4,
    rect: Rect,
) -> Vec2 {
    let end = world_to_screen(origin + axis_dir, vp, rect).unwrap_or(origin_screen);
    let dir = end - origin_screen;
    let len = dir.length();
    if len > 0.1 {
        dir / len
    } else {
        Vec2::ZERO
    }
}

// ---------------------------------------------------------------------------
// Node transform extraction
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct NodeTransform {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    has_scale: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct GizmoTarget {
    node_id: NodeId,
    local_transform: NodeTransform,
    parent_world_inverse: Mat4,
    parent_world_rotation: Quat,
    world_position: Vec3,
    world_rotation: Quat,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct GizmoSelection {
    targets: Vec<GizmoTarget>,
    base_center_world: Vec3,
    reference_rotation_world: Quat,
}

#[allow(dead_code)]
fn local_to_world_rotation(rotation: Vec3) -> Quat {
    euler_to_quat(rotation).inverse()
}

#[allow(dead_code)]
fn transform_local_to_parent_matrix(transform: &NodeTransform) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        transform.scale,
        local_to_world_rotation(transform.rotation),
        transform.position,
    )
}

fn extract_node_transform(scene: &Scene, node_id: NodeId) -> Option<NodeTransform> {
    match scene.nodes.get(&node_id).map(|n| &n.data) {
        Some(NodeData::Primitive {
            position,
            rotation,
            scale,
            ..
        }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: *scale,
            has_scale: true,
        }),
        Some(NodeData::Sculpt {
            position, rotation, ..
        }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: Vec3::ONE,
            has_scale: false,
        }),
        Some(NodeData::Transform {
            translation,
            rotation,
            scale,
            ..
        }) => Some(NodeTransform {
            position: *translation,
            rotation: *rotation,
            scale: *scale,
            has_scale: true,
        }),
        _ => None,
    }
}

#[allow(dead_code)]
fn build_parent_world_transform(
    scene: &Scene,
    node_id: NodeId,
    parent_map: &std::collections::HashMap<NodeId, NodeId>,
) -> (Mat4, Mat4, Quat) {
    let mut transform_chain = Vec::new();
    let mut current = node_id;

    while let Some(&parent_id) = parent_map.get(&current) {
        if let Some(NodeData::Transform {
            translation,
            rotation,
            scale,
            ..
        }) = scene.nodes.get(&parent_id).map(|node| &node.data)
        {
            transform_chain.push(NodeTransform {
                position: *translation,
                rotation: *rotation,
                scale: *scale,
                has_scale: true,
            });
        }
        current = parent_id;
    }

    let mut parent_world_matrix = Mat4::IDENTITY;
    let mut parent_world_rotation = Quat::IDENTITY;
    for transform in transform_chain.iter().rev() {
        parent_world_matrix *= transform_local_to_parent_matrix(transform);
        parent_world_rotation *= local_to_world_rotation(transform.rotation);
    }

    (
        parent_world_matrix,
        parent_world_matrix.inverse(),
        parent_world_rotation,
    )
}

#[allow(dead_code)]
fn build_gizmo_target(
    scene: &Scene,
    node_id: NodeId,
    parent_map: &std::collections::HashMap<NodeId, NodeId>,
) -> Option<GizmoTarget> {
    let local_transform = extract_node_transform(scene, node_id)?;
    let (parent_world_matrix, parent_world_inverse, parent_world_rotation) =
        build_parent_world_transform(scene, node_id, parent_map);
    let world_position = parent_world_matrix.transform_point3(local_transform.position);
    let world_rotation = parent_world_rotation * local_to_world_rotation(local_transform.rotation);

    Some(GizmoTarget {
        node_id,
        local_transform,
        parent_world_inverse,
        parent_world_rotation,
        world_position,
        world_rotation,
    })
}

#[allow(dead_code)]
pub(crate) fn collect_gizmo_selection(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &std::collections::HashSet<NodeId>,
) -> Option<GizmoSelection> {
    let mut ordered_ids = Vec::new();
    if let Some(primary_selected) = selected {
        ordered_ids.push(primary_selected);
    }

    let mut extra_ids: Vec<_> = selected_set
        .iter()
        .copied()
        .filter(|node_id| Some(*node_id) != selected)
        .collect();
    extra_ids.sort_unstable();
    ordered_ids.extend(extra_ids);

    if ordered_ids.is_empty() {
        return None;
    }

    let parent_map = scene.build_parent_map();
    let ordered_id_set: std::collections::HashSet<_> = ordered_ids.iter().copied().collect();
    let mut targets = Vec::new();
    for node_id in ordered_ids {
        let mut current = node_id;
        let mut has_selected_ancestor = false;
        while let Some(&parent_id) = parent_map.get(&current) {
            if ordered_id_set.contains(&parent_id) {
                has_selected_ancestor = true;
                break;
            }
            current = parent_id;
        }
        if has_selected_ancestor {
            continue;
        }
        if let Some(target) = build_gizmo_target(scene, node_id, &parent_map) {
            targets.push(target);
        }
    }

    if targets.is_empty() {
        return None;
    }

    let mut base_center_world = Vec3::ZERO;
    for target in &targets {
        base_center_world += target.world_position;
    }
    base_center_world /= targets.len() as f32;

    Some(GizmoSelection {
        reference_rotation_world: targets[0].world_rotation,
        targets,
        base_center_world,
    })
}

impl GizmoSelection {
    pub(crate) fn supports_scale(&self) -> bool {
        self.targets
            .iter()
            .all(|target| target.local_transform.has_scale)
    }

    pub(crate) fn target_count(&self) -> usize {
        self.targets.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MultiTransformReadout {
    pub position_delta: Vec3,
    pub rotation_delta_rad: Vec3,
    pub scale_factor: Vec3,
    pub scale_enabled: bool,
}

fn current_targets_from_baseline(
    scene: &Scene,
    baseline: &GizmoSelection,
) -> Option<Vec<GizmoTarget>> {
    let parent_map = scene.build_parent_map();
    baseline
        .targets
        .iter()
        .map(|target| build_gizmo_target(scene, target.node_id, &parent_map))
        .collect()
}

fn rotation_delta_quat_from_euler(
    rotation_delta_rad: Vec3,
    gizmo_space: &GizmoSpace,
    reference_rotation_world: Quat,
) -> Quat {
    let frame_delta = local_to_world_rotation(rotation_delta_rad);
    match gizmo_space {
        GizmoSpace::World => frame_delta,
        GizmoSpace::Local => {
            reference_rotation_world * frame_delta * reference_rotation_world.inverse()
        }
    }
}

fn rotation_delta_euler_from_world_quat(
    rotation_delta_world: Quat,
    gizmo_space: &GizmoSpace,
    reference_rotation_world: Quat,
    previous_rotation_delta_rad: Vec3,
) -> Vec3 {
    let frame_delta = match gizmo_space {
        GizmoSpace::World => rotation_delta_world,
        GizmoSpace::Local => {
            reference_rotation_world.inverse() * rotation_delta_world * reference_rotation_world
        }
    };
    quat_to_euler_stable(frame_delta.inverse(), previous_rotation_delta_rad)
}

fn scale_offset_in_basis(offset: Vec3, basis_axes: [Vec3; 3], scale_factor: Vec3) -> Vec3 {
    basis_axes[0] * offset.dot(basis_axes[0]) * scale_factor.x
        + basis_axes[1] * offset.dot(basis_axes[1]) * scale_factor.y
        + basis_axes[2] * offset.dot(basis_axes[2]) * scale_factor.z
}

#[allow(dead_code)]
fn build_drag_session(
    selection: &GizmoSelection,
    axis_directions: [Vec3; 3],
    origin_screen: Pos2,
    pivot_offset: Vec3,
) -> GizmoDragSession {
    let targets = selection.targets.to_vec();

    GizmoDragSession {
        start_center_world: selection.base_center_world + pivot_offset,
        start_origin_screen: origin_screen,
        start_pivot_offset: pivot_offset,
        axis_directions,
        targets,
        accumulated_drag_delta: Vec2::ZERO,
    }
}

// ---------------------------------------------------------------------------
// Axis direction computation
// ---------------------------------------------------------------------------

fn compute_axis_directions(node_rotation: Vec3, space: &GizmoSpace) -> [Vec3; 3] {
    match space {
        GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
        GizmoSpace::Local => [
            inverse_rotate_euler(Vec3::X, node_rotation),
            inverse_rotate_euler(Vec3::Y, node_rotation),
            inverse_rotate_euler(Vec3::Z, node_rotation),
        ],
    }
}

fn compute_world_axis_directions(reference_rotation_world: Quat, space: &GizmoSpace) -> [Vec3; 3] {
    match space {
        GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
        GizmoSpace::Local => [
            (reference_rotation_world * Vec3::X).normalize_or_zero(),
            (reference_rotation_world * Vec3::Y).normalize_or_zero(),
            (reference_rotation_world * Vec3::Z).normalize_or_zero(),
        ],
    }
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

fn hit_test_axes(
    mouse_pos: Pos2,
    origin_screen: Pos2,
    axis_screens: &[Pos2; 3],
    threshold: f32,
) -> Option<GizmoAxis> {
    let dx = point_to_segment_dist(mouse_pos, origin_screen, axis_screens[0]);
    let dy = point_to_segment_dist(mouse_pos, origin_screen, axis_screens[1]);
    let dz = point_to_segment_dist(mouse_pos, origin_screen, axis_screens[2]);
    let min = dx.min(dy).min(dz);
    if min > threshold {
        None
    } else if (min - dx).abs() < 0.01 {
        Some(GizmoAxis::X)
    } else if (min - dy).abs() < 0.01 {
        Some(GizmoAxis::Y)
    } else {
        Some(GizmoAxis::Z)
    }
}

fn ring_tangent_bitangent(axis_dir: Vec3) -> (Vec3, Vec3) {
    let up = if axis_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = axis_dir.cross(up).normalize();
    let bitangent = axis_dir.cross(tangent).normalize();
    (tangent, bitangent)
}

fn hit_test_rings(
    mouse_pos: Pos2,
    center: Vec3,
    axes: &[Vec3; 3],
    vp: &Mat4,
    rect: Rect,
) -> Option<GizmoAxis> {
    let mut best_dist = f32::MAX;
    let mut best_axis = None;

    for (i, axis) in ALL_AXES.iter().enumerate() {
        let dist = ring_distance_to_point(mouse_pos, center, axes[i], vp, rect);
        if dist < best_dist {
            best_dist = dist;
            best_axis = Some(axis.clone());
        }
    }

    if best_dist > RING_HIT_THRESHOLD {
        None
    } else {
        best_axis
    }
}

fn ring_distance_to_point(
    mouse_pos: Pos2,
    center: Vec3,
    axis_dir: Vec3,
    vp: &Mat4,
    rect: Rect,
) -> f32 {
    let (tangent, bitangent) = ring_tangent_bitangent(axis_dir);
    let mut min_dist = f32::MAX;
    let mut prev_screen: Option<Pos2> = None;

    for i in 0..=RING_SEGMENTS {
        let angle = (i as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
        let world_pt = center + (tangent * angle.cos() + bitangent * angle.sin()) * RING_RADIUS;
        if let Some(screen_pt) = world_to_screen(world_pt, vp, rect) {
            if let Some(prev) = prev_screen {
                let d = point_to_segment_dist(mouse_pos, prev, screen_pt);
                min_dist = min_dist.min(d);
            }
            prev_screen = Some(screen_pt);
        }
    }
    min_dist
}

// ---------------------------------------------------------------------------
// Drawing: translate gizmo
// ---------------------------------------------------------------------------

fn draw_translate_gizmo(
    painter: &egui::Painter,
    origin_screen: Pos2,
    axis_screens: &[Pos2; 3],
    colors: &[Color32; 3],
) {
    for i in 0..3 {
        painter.line_segment(
            [origin_screen, axis_screens[i]],
            Stroke::new(AXIS_STROKE_WIDTH, colors[i]),
        );
        draw_arrow_head(painter, origin_screen, axis_screens[i], colors[i]);
    }
}

fn draw_arrow_head(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    let dir = to - from;
    let len = dir.length();
    if len < 1.0 {
        return;
    }
    let dir = dir / len;
    let perp = Vec2::new(-dir.y, dir.x);
    let tip = to;
    let left = to - dir * ARROW_SIZE + perp * ARROW_SIZE * 0.5;
    let right = to - dir * ARROW_SIZE - perp * ARROW_SIZE * 0.5;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, left, right],
        color,
        Stroke::NONE,
    ));
}

// ---------------------------------------------------------------------------
// Drawing: scale gizmo
// ---------------------------------------------------------------------------

fn draw_scale_gizmo(
    painter: &egui::Painter,
    origin_screen: Pos2,
    axis_screens: &[Pos2; 3],
    colors: &[Color32; 3],
) {
    for i in 0..3 {
        painter.line_segment(
            [origin_screen, axis_screens[i]],
            Stroke::new(AXIS_STROKE_WIDTH, colors[i]),
        );
        draw_scale_box(painter, axis_screens[i], colors[i]);
    }
}

fn draw_scale_box(painter: &egui::Painter, center: Pos2, color: Color32) {
    painter.rect_filled(
        Rect::from_center_size(center, Vec2::splat(SCALE_BOX_SIZE)),
        0.0,
        color,
    );
}

// ---------------------------------------------------------------------------
// Drawing: rotation gizmo
// ---------------------------------------------------------------------------

fn draw_rotate_gizmo(
    painter: &egui::Painter,
    center: Vec3,
    axes: &[Vec3; 3],
    colors: &[Color32; 3],
    vp: &Mat4,
    rect: Rect,
) {
    for i in 0..3 {
        draw_rotation_ring(painter, center, axes[i], colors[i], vp, rect);
    }
}

fn draw_rotation_ring(
    painter: &egui::Painter,
    center: Vec3,
    axis_dir: Vec3,
    color: Color32,
    vp: &Mat4,
    rect: Rect,
) {
    let (tangent, bitangent) = ring_tangent_bitangent(axis_dir);
    let mut prev_screen: Option<Pos2> = None;

    for i in 0..=RING_SEGMENTS {
        let angle = (i as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
        let world_pt = center + (tangent * angle.cos() + bitangent * angle.sin()) * RING_RADIUS;
        if let Some(screen_pt) = world_to_screen(world_pt, vp, rect) {
            if let Some(prev) = prev_screen {
                painter.line_segment([prev, screen_pt], Stroke::new(AXIS_STROKE_WIDTH, color));
            }
            prev_screen = Some(screen_pt);
        } else {
            prev_screen = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Drag handlers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn handle_translate_drag(
    response: &egui::Response,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Pos2,
    camera: &Camera,
    vp: &Mat4,
    rect: Rect,
    scene: &mut Scene,
    drag_node: NodeId,
    pivot_offset: &Vec3,
    node_rotation: Vec3,
) {
    let axis_sd = screen_axis_dir(origin, axis_dir, origin_screen, vp, rect);
    let delta = response.drag_delta();
    let projected = delta.dot(axis_sd);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let world_delta = axis_dir * projected * world_scale;

    let _ = pivot_offset;
    let _ = node_rotation;

    apply_position_delta(scene, drag_node, world_delta);
}

#[allow(clippy::too_many_arguments)]
fn handle_scale_drag(
    response: &egui::Response,
    axis: &GizmoAxis,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Pos2,
    _camera: &Camera,
    vp: &Mat4,
    rect: Rect,
    scene: &mut Scene,
    drag_node: NodeId,
    pivot_offset: &Vec3,
    node_rotation: Vec3,
) {
    let axis_sd = screen_axis_dir(origin, axis_dir, origin_screen, vp, rect);
    let delta = response.drag_delta();
    let projected = delta.dot(axis_sd);
    let factor = 1.0 + projected * SCALE_SENSITIVITY;

    if let Some(node) = scene.nodes.get_mut(&drag_node) {
        let (scale, position) = match &mut node.data {
            NodeData::Primitive {
                ref mut scale,
                ref mut position,
                ..
            } => (scale, position),
            NodeData::Transform {
                ref mut scale,
                ref mut translation,
                ..
            } => (scale, translation),
            _ => return,
        };

        if pivot_offset.length_squared() > 1e-6 {
            let pivot_world = *position + inverse_rotate_euler(*pivot_offset, node_rotation);
            let offset = *position - pivot_world;
            let mut scale_vec = Vec3::ONE;
            match axis {
                GizmoAxis::X => scale_vec.x = factor,
                GizmoAxis::Y => scale_vec.y = factor,
                GizmoAxis::Z => scale_vec.z = factor,
            }
            *position = pivot_world + offset * scale_vec;
        }

        match axis {
            GizmoAxis::X => scale.x = (scale.x * factor).max(0.01),
            GizmoAxis::Y => scale.y = (scale.y * factor).max(0.01),
            GizmoAxis::Z => scale.z = (scale.z * factor).max(0.01),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_rotate_drag(
    response: &egui::Response,
    axis: &GizmoAxis,
    axis_dir: Vec3,
    origin_screen: Pos2,
    scene: &mut Scene,
    drag_node: NodeId,
    pivot_offset: &Vec3,
    node_rotation: Vec3,
    gizmo_space: &GizmoSpace,
    view_dir: Vec3,
) {
    let mouse_pos = response.hover_pos().unwrap_or(origin_screen);
    let radius_vec = mouse_pos - origin_screen;
    let radius_len = radius_vec.length();
    if radius_len < 1.0 {
        return;
    }
    let radius_norm = radius_vec / radius_len;
    let tangent_dir = Vec2::new(-radius_norm.y, radius_norm.x);
    let delta = response.drag_delta();
    let projected = delta.dot(tangent_dir);

    // Flip sign when the ring axis faces away from camera (right-hand rule)
    // Multi-rotate drag direction is intentionally inverted to match the
    // single-selection gizmo feel in viewport interaction.
    let sign = if axis_dir.dot(view_dir) >= 0.0 {
        -1.0
    } else {
        1.0
    };
    let angle_delta = projected * ROTATE_SENSITIVITY * sign;

    let delta_quat = Quat::from_axis_angle(axis_dir, angle_delta);

    if let Some(node) = scene.nodes.get_mut(&drag_node) {
        let (rotation, position) = match &mut node.data {
            NodeData::Primitive {
                ref mut rotation,
                ref mut position,
                ..
            } => (rotation, position),
            NodeData::Sculpt {
                ref mut rotation,
                ref mut position,
                ..
            } => (rotation, position),
            NodeData::Transform {
                ref mut rotation,
                ref mut translation,
                ..
            } => (rotation, translation),
            _ => return,
        };
        if pivot_offset.length_squared() > 1e-6 {
            let pivot_world = *position + inverse_rotate_euler(*pivot_offset, node_rotation);
            let offset = *position - pivot_world;
            *position = pivot_world + delta_quat * offset;
        }
        apply_rotation_delta(rotation, axis, angle_delta, axis_dir, gizmo_space);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_multi_translate_drag(
    response: &egui::Response,
    drag_session: &mut GizmoDragSession,
    axis_dir: Vec3,
    camera: &Camera,
    vp: &Mat4,
    rect: Rect,
    scene: &mut Scene,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let axis_sd = screen_axis_dir(
        drag_session.start_center_world,
        axis_dir,
        drag_session.start_origin_screen,
        vp,
        rect,
    );
    drag_session.accumulated_drag_delta += response.drag_delta();
    let delta = drag_session.accumulated_drag_delta;
    let projected = delta.dot(axis_sd);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let mut projected_world = projected * world_scale;
    if ctrl_held {
        projected_world = snap_value(projected_world, snap_config.translate_snap);
    }
    let world_delta = axis_dir * projected_world;

    apply_world_translation_to_targets(scene, &drag_session.targets, world_delta);
}

#[allow(clippy::too_many_arguments)]
fn handle_multi_rotate_drag(
    response: &egui::Response,
    drag_session: &mut GizmoDragSession,
    axis_dir: Vec3,
    scene: &mut Scene,
    view_dir: Vec3,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let mouse_pos = response
        .hover_pos()
        .unwrap_or(drag_session.start_origin_screen);
    let radius_vec = mouse_pos - drag_session.start_origin_screen;
    let radius_len = radius_vec.length();
    if radius_len < 1.0 {
        return;
    }
    let radius_norm = radius_vec / radius_len;
    let tangent_dir = Vec2::new(-radius_norm.y, radius_norm.x);
    drag_session.accumulated_drag_delta += response.drag_delta();
    let delta = drag_session.accumulated_drag_delta;
    let projected = delta.dot(tangent_dir);

    let sign = if axis_dir.dot(view_dir) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut angle_delta = projected * ROTATE_SENSITIVITY * sign;
    if ctrl_held {
        angle_delta = snap_value(angle_delta, snap_config.rotate_snap.to_radians());
    }

    let delta_quat = Quat::from_axis_angle(axis_dir, angle_delta);
    apply_world_rotation_to_targets(
        scene,
        &drag_session.targets,
        drag_session.start_center_world,
        delta_quat,
    );
}

#[allow(clippy::too_many_arguments)]
fn handle_multi_scale_drag(
    response: &egui::Response,
    drag_session: &mut GizmoDragSession,
    axis: &GizmoAxis,
    axis_dir: Vec3,
    vp: &Mat4,
    rect: Rect,
    scene: &mut Scene,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let axis_sd = screen_axis_dir(
        drag_session.start_center_world,
        axis_dir,
        drag_session.start_origin_screen,
        vp,
        rect,
    );
    drag_session.accumulated_drag_delta += response.drag_delta();
    let delta = drag_session.accumulated_drag_delta;
    let projected = delta.dot(axis_sd);
    let factor = 1.0 + projected * SCALE_SENSITIVITY;

    apply_axis_scale_to_targets(
        scene,
        &drag_session.targets,
        drag_session.start_center_world,
        axis_dir,
        axis,
        factor,
    );
    if ctrl_held {
        snap_target_scales(
            scene,
            &drag_session.targets,
            axis.euler_index(),
            snap_config.scale_snap,
        );
    }
}

#[allow(dead_code)]
fn apply_rotation_delta(
    rotation: &mut Vec3,
    _axis: &GizmoAxis,
    angle_delta: f32,
    axis_dir: Vec3,
    gizmo_space: &GizmoSpace,
) {
    let prev = *rotation;
    let delta_quat = Quat::from_axis_angle(axis_dir, angle_delta);
    let current = euler_to_quat(*rotation);
    let new_quat = match gizmo_space {
        // Local: rotation in object's frame — current * delta
        GizmoSpace::Local => current * delta_quat,
        // World: rotation in world frame — delta * current
        GizmoSpace::World => delta_quat * current,
    };
    *rotation = quat_to_euler_stable(new_quat, prev);
}

#[allow(dead_code)]
fn apply_position_delta(scene: &mut Scene, node_id: NodeId, delta: Vec3) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        match &mut node.data {
            NodeData::Primitive {
                ref mut position, ..
            } => *position += delta,
            NodeData::Sculpt {
                ref mut position, ..
            } => *position += delta,
            NodeData::Transform {
                ref mut translation,
                ..
            } => *translation += delta,
            _ => {}
        }
    }
}

#[allow(dead_code)]
fn set_node_position(scene: &mut Scene, node_id: NodeId, position: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            position: node_position,
            ..
        } => *node_position = position,
        NodeData::Sculpt {
            position: node_position,
            ..
        } => *node_position = position,
        NodeData::Transform { translation, .. } => *translation = position,
        _ => {}
    }
}

#[allow(dead_code)]
fn set_node_rotation(scene: &mut Scene, node_id: NodeId, rotation: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        NodeData::Sculpt {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        NodeData::Transform {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        _ => {}
    }
}

#[allow(dead_code)]
fn set_node_scale(scene: &mut Scene, node_id: NodeId, scale: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            scale: node_scale, ..
        } => *node_scale = scale,
        NodeData::Transform {
            scale: node_scale, ..
        } => *node_scale = scale,
        _ => {}
    }
}

#[allow(dead_code)]
fn apply_world_translation_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    world_delta: Vec3,
) {
    for target in targets {
        let new_world_position = target.world_position + world_delta;
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(new_world_position);
        set_node_position(scene, target.node_id, new_local_position);
    }
}

#[allow(dead_code)]
fn apply_world_rotation_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    pivot_world: Vec3,
    delta_quat: Quat,
) {
    for target in targets {
        let rotated_world_position = pivot_world + delta_quat * (target.world_position - pivot_world);
        let rotated_world_orientation = delta_quat * target.world_rotation;
        let local_world_rotation =
            target.parent_world_rotation.inverse() * rotated_world_orientation;
        let local_rotation_quat = local_world_rotation.inverse();
        let local_rotation =
            quat_to_euler_stable(local_rotation_quat, target.local_transform.rotation);
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(rotated_world_position);

        set_node_position(scene, target.node_id, new_local_position);
        set_node_rotation(scene, target.node_id, local_rotation);
    }
}

#[allow(dead_code)]
fn apply_axis_scale_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    pivot_world: Vec3,
    axis_dir: Vec3,
    axis: &GizmoAxis,
    factor: f32,
) {
    for target in targets {
        let offset = target.world_position - pivot_world;
        let parallel = axis_dir * offset.dot(axis_dir);
        let perpendicular = offset - parallel;
        let scaled_world_position = pivot_world + perpendicular + parallel * factor;
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(scaled_world_position);
        set_node_position(scene, target.node_id, new_local_position);

        if !target.local_transform.has_scale {
            continue;
        }

        let mut scaled_local = target.local_transform.scale;
        match axis {
            GizmoAxis::X => scaled_local.x = (scaled_local.x * factor).max(0.01),
            GizmoAxis::Y => scaled_local.y = (scaled_local.y * factor).max(0.01),
            GizmoAxis::Z => scaled_local.z = (scaled_local.z * factor).max(0.01),
        }
        set_node_scale(scene, target.node_id, scaled_local);
    }
}

#[allow(dead_code)]
fn snap_target_scales(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    axis_index: usize,
    snap: f32,
) {
    for target in targets {
        if target.local_transform.has_scale {
            snap_scale(scene, target.node_id, axis_index, snap);
        }
    }
}

pub(crate) fn derive_multi_transform_readout(
    scene: &Scene,
    baseline: &GizmoSelection,
    gizmo_space: &GizmoSpace,
    previous_rotation_delta_rad: Vec3,
) -> Option<MultiTransformReadout> {
    let current_targets = current_targets_from_baseline(scene, baseline)?;
    if current_targets.is_empty() {
        return None;
    }

    let mut current_center_world = Vec3::ZERO;
    for target in &current_targets {
        current_center_world += target.world_position;
    }
    current_center_world /= current_targets.len() as f32;
    let position_delta = current_center_world - baseline.base_center_world;

    let reference_current = &current_targets[0];
    let reference_baseline = &baseline.targets[0];
    let rotation_delta_world =
        reference_current.world_rotation * reference_baseline.world_rotation.inverse();
    let rotation_delta_rad = rotation_delta_euler_from_world_quat(
        rotation_delta_world,
        gizmo_space,
        baseline.reference_rotation_world,
        previous_rotation_delta_rad,
    );

    let scale_enabled = baseline.supports_scale();
    let scale_factor = if scale_enabled {
        let current_scale = current_targets[0].local_transform.scale;
        let baseline_scale = baseline.targets[0].local_transform.scale;
        Vec3::new(
            current_scale.x / baseline_scale.x.max(0.01),
            current_scale.y / baseline_scale.y.max(0.01),
            current_scale.z / baseline_scale.z.max(0.01),
        )
    } else {
        Vec3::ONE
    };

    Some(MultiTransformReadout {
        position_delta,
        rotation_delta_rad,
        scale_factor,
        scale_enabled,
    })
}

pub(crate) fn restore_multi_transform_baseline(scene: &mut Scene, baseline: &GizmoSelection) {
    for target in &baseline.targets {
        set_node_position(scene, target.node_id, target.local_transform.position);
        set_node_rotation(scene, target.node_id, target.local_transform.rotation);
        if target.local_transform.has_scale {
            set_node_scale(scene, target.node_id, target.local_transform.scale);
        }
    }
}

pub(crate) fn apply_multi_transform_from_baseline(
    scene: &mut Scene,
    baseline: &GizmoSelection,
    gizmo_space: &GizmoSpace,
    position_delta: Vec3,
    rotation_delta_rad: Vec3,
    scale_factor: Vec3,
) {
    let clamped_scale_factor = Vec3::new(
        scale_factor.x.max(0.01),
        scale_factor.y.max(0.01),
        scale_factor.z.max(0.01),
    );
    let rotation_delta_world = rotation_delta_quat_from_euler(
        rotation_delta_rad,
        gizmo_space,
        baseline.reference_rotation_world,
    );
    let scale_axes = compute_world_axis_directions(baseline.reference_rotation_world, gizmo_space);
    let scale_enabled = baseline.supports_scale();

    for target in &baseline.targets {
        let scaled_offset = if scale_enabled {
            scale_offset_in_basis(
                target.world_position - baseline.base_center_world,
                scale_axes,
                clamped_scale_factor,
            )
        } else {
            target.world_position - baseline.base_center_world
        };
        let rotated_offset = rotation_delta_world * scaled_offset;
        let new_world_position =
            baseline.base_center_world + rotated_offset + position_delta;
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(new_world_position);

        let rotated_world_orientation = rotation_delta_world * target.world_rotation;
        let local_world_rotation =
            target.parent_world_rotation.inverse() * rotated_world_orientation;
        let local_rotation_quat = local_world_rotation.inverse();
        let local_rotation =
            quat_to_euler_stable(local_rotation_quat, target.local_transform.rotation);

        set_node_position(scene, target.node_id, new_local_position);
        set_node_rotation(scene, target.node_id, local_rotation);

        if scale_enabled && target.local_transform.has_scale {
            let scaled_local = target.local_transform.scale * clamped_scale_factor;
            set_node_scale(scene, target.node_id, scaled_local);
        }
    }
}

// ---------------------------------------------------------------------------
// Pivot drag (Alt + drag moves pivot instead of object)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn handle_pivot_drag(
    response: &egui::Response,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Pos2,
    camera: &Camera,
    vp: &Mat4,
    rect: Rect,
    pivot_offset: &mut Vec3,
    node_rotation: Vec3,
) {
    let axis_sd = screen_axis_dir(origin, axis_dir, origin_screen, vp, rect);
    let delta = response.drag_delta();
    let projected = delta.dot(axis_sd);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let world_delta = axis_dir * projected * world_scale;

    let inv_delta = inverse_rotate_euler(world_delta, node_rotation);
    *pivot_offset += inv_delta;
}

#[allow(dead_code)]
fn inverse_rotate_euler(p: Vec3, r: Vec3) -> Vec3 {
    let mut q = p;
    let (sz, cz) = r.z.sin_cos();
    q = Vec3::new(cz * q.x + sz * q.y, -sz * q.x + cz * q.y, q.z);
    let (sy, cy) = r.y.sin_cos();
    q = Vec3::new(cy * q.x - sy * q.z, q.y, sy * q.x + cy * q.z);
    let (sx, cx) = r.x.sin_cos();
    q = Vec3::new(q.x, cx * q.y + sx * q.z, -sx * q.y + cx * q.z);
    q
}

// ---------------------------------------------------------------------------
// Main gizmo function
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn draw_and_interact(
    painter: &egui::Painter,
    response: &egui::Response,
    camera: &Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    selected_set: &std::collections::HashSet<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: &mut Vec3,
    rect: Rect,
    snap_config: &SnapConfig,
    gizmo_visible: bool,
) -> bool {
    if !gizmo_visible {
        *gizmo_state = GizmoState::Idle;
        return false;
    }

    let aspect = rect.width() / rect.height().max(1.0);
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;

    let mut raw_selected_count = selected_set.len();
    if let Some(primary_selected) = selected {
        if !selected_set.contains(&primary_selected) {
            raw_selected_count += 1;
        }
    }

    let multi_drag_active = matches!(gizmo_state, GizmoState::DraggingMulti { .. });
    let multi_selection_active = match gizmo_state {
        GizmoState::DraggingSingle { .. } => false,
        GizmoState::DraggingMulti { .. } => true,
        GizmoState::Idle => raw_selected_count > 1,
    };

    let selection = if multi_selection_active {
        collect_gizmo_selection(scene, selected, selected_set)
    } else {
        None
    };

    let single_node_id = match gizmo_state {
        GizmoState::DraggingSingle { node_id, .. } => Some(*node_id),
        _ => selected,
    };

    let mut single_transform = None;
    let axes;
    let gizmo_center;

    if multi_selection_active {
        if let Some(selection) = selection.as_ref() {
            if *gizmo_mode == GizmoMode::Scale && !selection.supports_scale() && !multi_drag_active {
                *gizmo_state = GizmoState::Idle;
                return false;
            }
            // Multi-selection rotation should always use world-space axes so the group
            // rotates consistently regardless of per-object local orientation.
            let multi_axes_space = if *gizmo_mode == GizmoMode::Rotate {
                GizmoSpace::World
            } else {
                gizmo_space.clone()
            };
            axes = compute_world_axis_directions(selection.reference_rotation_world, &multi_axes_space);
            gizmo_center = selection.base_center_world;
        } else if let GizmoState::DraggingMulti { drag_session, .. } = gizmo_state {
            axes = drag_session.axis_directions;
            gizmo_center = drag_session.start_center_world;
        } else {
            *gizmo_state = GizmoState::Idle;
            return false;
        }
    } else {
        let Some(node_id) = single_node_id else {
            *gizmo_state = GizmoState::Idle;
            return false;
        };

        let Some(node_transform) = extract_node_transform(scene, node_id) else {
            *gizmo_state = GizmoState::Idle;
            return false;
        };

        if *gizmo_mode == GizmoMode::Scale && !node_transform.has_scale {
            *gizmo_state = GizmoState::Idle;
            return false;
        }

        axes = compute_axis_directions(node_transform.rotation, gizmo_space);
        gizmo_center =
            node_transform.position + rotate_euler(*pivot_offset, node_transform.rotation);
        single_transform = Some((node_id, node_transform));
    }

    let Some(origin_screen) = world_to_screen(gizmo_center, &vp, rect) else {
        return false;
    };

    let allow_pivot_drag = !multi_selection_active && response.ctx.input(|i| i.modifiers.alt);

    // Project axis endpoints for translate/scale modes
    let axis_screens = {
        let mut screens = [origin_screen; 3];
        for i in 0..3 {
            let end = gizmo_center + axes[i] * AXIS_LENGTH;
            screens[i] = world_to_screen(end, &vp, rect).unwrap_or(origin_screen);
        }
        screens
    };

    // Hit test
    let hover_pos = response.hover_pos();
    let hovered_axis = hover_pos.and_then(|pos| match gizmo_mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            hit_test_axes(pos, origin_screen, &axis_screens, HIT_THRESHOLD)
        }
        GizmoMode::Rotate => hit_test_rings(pos, gizmo_center, &axes, &vp, rect),
    });

    // Determine colors
    let dragging_axis = match gizmo_state {
        GizmoState::DraggingSingle { ref axis, .. }
        | GizmoState::DraggingMulti { ref axis, .. } => Some(axis.clone()),
        _ => None,
    };
    let colors = [
        axis_color(&GizmoAxis::X, &dragging_axis, &hovered_axis),
        axis_color(&GizmoAxis::Y, &dragging_axis, &hovered_axis),
        axis_color(&GizmoAxis::Z, &dragging_axis, &hovered_axis),
    ];

    // Draw
    match gizmo_mode {
        GizmoMode::Translate => {
            draw_translate_gizmo(painter, origin_screen, &axis_screens, &colors)
        }
        GizmoMode::Scale => draw_scale_gizmo(painter, origin_screen, &axis_screens, &colors),
        GizmoMode::Rotate => draw_rotate_gizmo(painter, gizmo_center, &axes, &colors, &vp, rect),
    }

    // Draw pivot indicator when offset
    if let Some((_, node_transform)) = single_transform
        .as_ref()
        .filter(|_| pivot_offset.length_squared() > 1e-6)
    {
        let node_origin_screen = world_to_screen(node_transform.position, &vp, rect);
        if let Some(nos) = node_origin_screen {
            painter.circle_stroke(nos, 4.0, Stroke::new(1.5, Color32::from_rgb(255, 200, 50)));
            painter.line_segment(
                [nos, origin_screen],
                Stroke::new(1.0, Color32::from_rgba_premultiplied(255, 200, 50, 100)),
            );
        }
    }

    // --- Interaction ---
    let mut consumed = false;

    // Start dragging
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(ref axis) = hovered_axis {
            if multi_selection_active {
                if let Some(selection) = selection.as_ref() {
                    *gizmo_state = GizmoState::DraggingMulti {
                        axis: axis.clone(),
                        drag_session: build_drag_session(
                            selection,
                            axes,
                            origin_screen,
                            Vec3::ZERO,
                        ),
                    };
                    consumed = true;
                }
            } else if let Some((node_id, node_transform)) = single_transform.as_ref() {
                *gizmo_state = GizmoState::DraggingSingle {
                    axis: axis.clone(),
                    node_id: *node_id,
                    _start_screen_pos: hover_pos.unwrap_or(origin_screen),
                    _start_world_pos: node_transform.position,
                    _start_rotation: node_transform.rotation,
                    _start_scale: node_transform.scale,
                };
                consumed = true;
            }
        }
    }

    // During drag
    if response.dragged_by(egui::PointerButton::Primary) {
        match gizmo_state {
            GizmoState::DraggingSingle {
                axis,
                node_id: drag_node,
                ..
            } => {
                if let Some((_, node_transform)) = single_transform.as_ref() {
                    let axis_idx = axis.euler_index();
                    let axis_dir = axes[axis_idx];
                    let axis_clone = axis.clone();

                    if allow_pivot_drag {
                        handle_pivot_drag(
                            response,
                            axis_dir,
                            gizmo_center,
                            origin_screen,
                            camera,
                            &vp,
                            rect,
                            pivot_offset,
                            node_transform.rotation,
                        );
                    } else {
                        let ctrl_held = response.ctx.input(|i| i.modifiers.ctrl);
                        match gizmo_mode {
                            GizmoMode::Translate => {
                                handle_translate_drag(
                                    response,
                                    axis_dir,
                                    gizmo_center,
                                    origin_screen,
                                    camera,
                                    &vp,
                                    rect,
                                    scene,
                                    *drag_node,
                                    pivot_offset,
                                    node_transform.rotation,
                                );
                                if ctrl_held {
                                    snap_position(
                                        scene,
                                        *drag_node,
                                        axis_idx,
                                        snap_config.translate_snap,
                                    );
                                }
                            }
                            GizmoMode::Scale => {
                                handle_scale_drag(
                                    response,
                                    &axis_clone,
                                    axis_dir,
                                    gizmo_center,
                                    origin_screen,
                                    camera,
                                    &vp,
                                    rect,
                                    scene,
                                    *drag_node,
                                    pivot_offset,
                                    node_transform.rotation,
                                );
                                if ctrl_held {
                                    snap_scale(
                                        scene,
                                        *drag_node,
                                        axis_idx,
                                        snap_config.scale_snap,
                                    );
                                }
                            }
                            GizmoMode::Rotate => {
                                let view_dir = camera.eye() - gizmo_center;
                                handle_rotate_drag(
                                    response,
                                    &axis_clone,
                                    axis_dir,
                                    origin_screen,
                                    scene,
                                    *drag_node,
                                    pivot_offset,
                                    node_transform.rotation,
                                    gizmo_space,
                                    view_dir,
                                );
                                if ctrl_held {
                                    snap_rotation(
                                        scene,
                                        *drag_node,
                                        axis_idx,
                                        snap_config.rotate_snap.to_radians(),
                                    );
                                }
                            }
                        }
                    }
                    consumed = true;
                }
            }
            GizmoState::DraggingMulti { axis, drag_session } => {
                let axis_idx = axis.euler_index();
                let axis_dir = drag_session.axis_directions[axis_idx];
                let ctrl_held = response.ctx.input(|i| i.modifiers.ctrl);

                match gizmo_mode {
                    GizmoMode::Translate => {
                        handle_multi_translate_drag(
                            response,
                            drag_session,
                            axis_dir,
                            camera,
                            &vp,
                            rect,
                            scene,
                            snap_config,
                            ctrl_held,
                        );
                    }
                    GizmoMode::Rotate => {
                        let view_dir = camera.eye() - drag_session.start_center_world;
                        handle_multi_rotate_drag(
                            response,
                            drag_session,
                            axis_dir,
                            scene,
                            view_dir,
                            snap_config,
                            ctrl_held,
                        );
                    }
                    GizmoMode::Scale => {
                        handle_multi_scale_drag(
                            response,
                            drag_session,
                            axis,
                            axis_dir,
                            &vp,
                            rect,
                            scene,
                            snap_config,
                            ctrl_held,
                        );
                    }
                }
                consumed = true;
            }
            GizmoState::Idle => {}
        }
    }

    // End drag
    if response.drag_stopped_by(egui::PointerButton::Primary)
        && !matches!(gizmo_state, GizmoState::Idle)
    {
        *gizmo_state = GizmoState::Idle;
        consumed = true;
    }

    consumed
}

// ---------------------------------------------------------------------------
// Snap helpers — quantize position/rotation/scale to grid increments
// ---------------------------------------------------------------------------

fn snap_value(val: f32, snap: f32) -> f32 {
    (val / snap).round() * snap
}

#[allow(dead_code)]
fn snap_position(scene: &mut Scene, node_id: NodeId, axis_idx: usize, snap: f32) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        let pos = match &mut node.data {
            NodeData::Primitive {
                ref mut position, ..
            } => position,
            NodeData::Sculpt {
                ref mut position, ..
            } => position,
            NodeData::Transform {
                ref mut translation,
                ..
            } => translation,
            _ => return,
        };
        match axis_idx {
            0 => pos.x = snap_value(pos.x, snap),
            1 => pos.y = snap_value(pos.y, snap),
            _ => pos.z = snap_value(pos.z, snap),
        }
    }
}

#[allow(dead_code)]
fn snap_rotation(scene: &mut Scene, node_id: NodeId, axis_idx: usize, snap_rad: f32) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        let rot = match &mut node.data {
            NodeData::Primitive {
                ref mut rotation, ..
            } => rotation,
            NodeData::Sculpt {
                ref mut rotation, ..
            } => rotation,
            NodeData::Transform {
                ref mut rotation, ..
            } => rotation,
            _ => return,
        };
        match axis_idx {
            0 => rot.x = snap_value(rot.x, snap_rad),
            1 => rot.y = snap_value(rot.y, snap_rad),
            _ => rot.z = snap_value(rot.z, snap_rad),
        }
    }
}

fn snap_scale(scene: &mut Scene, node_id: NodeId, axis_idx: usize, snap: f32) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        let scale = match &mut node.data {
            NodeData::Primitive { ref mut scale, .. } => scale,
            NodeData::Transform { ref mut scale, .. } => scale,
            _ => return,
        };
        match axis_idx {
            0 => scale.x = snap_value(scale.x, snap).max(0.01),
            1 => scale.y = snap_value(scale.y, snap).max(0.01),
            _ => scale.z = snap_value(scale.z, snap).max(0.01),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{Scene, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
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

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let delta = actual - expected;
        assert!(
            delta.length() < 1e-4,
            "expected {expected:?}, got {actual:?}, delta={delta:?}"
        );
    }

    fn primitive_position(scene: &Scene, node_id: NodeId) -> Vec3 {
        match &scene.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            other => panic!("expected primitive, got {other:?}"),
        }
    }

    #[test]
    fn group_translation_moves_all_selected_targets() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Sphere);

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *position = Vec3::new(-1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }

        let selected_set = HashSet::from([left, right]);
        let selection = collect_gizmo_selection(&scene, Some(left), &selected_set).unwrap();
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Pos2::ZERO,
            Vec3::ZERO,
        );

        apply_world_translation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::new(0.0, 2.0, 0.0),
        );

        assert_vec3_close(primitive_position(&scene, left), Vec3::new(-1.0, 2.0, 0.0));
        assert_vec3_close(primitive_position(&scene, right), Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn group_rotation_uses_shared_pivot() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Sphere);

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *position = Vec3::new(-1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }

        let selected_set = HashSet::from([left, right]);
        let selection = collect_gizmo_selection(&scene, Some(left), &selected_set).unwrap();
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Pos2::ZERO,
            Vec3::ZERO,
        );

        apply_world_rotation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        );

        assert_vec3_close(primitive_position(&scene, left), Vec3::new(0.0, -1.0, 0.0));
        assert_vec3_close(primitive_position(&scene, right), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn child_under_parent_transform_writes_back_local_translation() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_transform(Some(child));

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&child).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Transform { translation, .. }) =
            scene.nodes.get_mut(&parent).map(|node| &mut node.data)
        {
            *translation = Vec3::new(10.0, 0.0, 0.0);
        }

        let selected_set = HashSet::from([child]);
        let selection = collect_gizmo_selection(&scene, Some(child), &selected_set).unwrap();
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Pos2::ZERO,
            Vec3::ZERO,
        );

        apply_world_translation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::new(2.0, 0.0, 0.0),
        );

        assert_vec3_close(primitive_position(&scene, child), Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn collect_selection_skips_descendant_when_ancestor_selected() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_transform(Some(child));

        let selected_set = HashSet::from([child, parent]);
        let selection = collect_gizmo_selection(&scene, Some(child), &selected_set).unwrap();

        assert_eq!(selection.targets.len(), 1);
        assert_eq!(selection.targets[0].node_id, parent);
    }

    #[test]
    fn child_under_rotated_parent_preserves_world_translation_delta() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_transform(Some(child));

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&child).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Transform {
            translation,
            rotation,
            ..
        }) = scene.nodes.get_mut(&parent).map(|node| &mut node.data)
        {
            *translation = Vec3::new(10.0, 0.0, 0.0);
            *rotation = Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        }

        let selected_set = HashSet::from([child]);
        let selection = collect_gizmo_selection(&scene, Some(child), &selected_set).unwrap();
        let start_world_position = selection.targets[0].world_position;
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Pos2::ZERO,
            Vec3::ZERO,
        );

        apply_world_translation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::new(2.0, 0.0, 0.0),
        );

        let moved_selection = collect_gizmo_selection(&scene, Some(child), &selected_set).unwrap();
        assert_vec3_close(
            moved_selection.targets[0].world_position,
            start_world_position + Vec3::new(2.0, 0.0, 0.0),
        );
    }

    #[test]
    fn multi_transform_readout_round_trips_group_transform() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Sphere);

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *position = Vec3::new(-1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }

        let selected_set = HashSet::from([left, right]);
        let baseline = collect_gizmo_selection(&scene, Some(left), &selected_set).unwrap();

        apply_multi_transform_from_baseline(
            &mut scene,
            &baseline,
            &GizmoSpace::Local,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            Vec3::new(2.0, 1.0, 1.0),
        );

        let readout =
            derive_multi_transform_readout(&scene, &baseline, &GizmoSpace::Local, Vec3::ZERO)
                .unwrap();
        assert_vec3_close(readout.position_delta, Vec3::new(0.0, 2.0, 0.0));
        assert_vec3_close(
            readout.rotation_delta_rad,
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
        );
        assert_vec3_close(readout.scale_factor, Vec3::new(2.0, 1.0, 1.0));
        assert!(readout.scale_enabled);
    }

    #[test]
    fn sculpt_selection_disables_multi_scale() {
        let mut scene = empty_scene();
        let input = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt = scene.create_sculpt(
            input,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ONE,
            VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );
        let primitive = scene.create_primitive(SdfPrimitive::Box);

        let selected_set = HashSet::from([sculpt, primitive]);
        let baseline = collect_gizmo_selection(&scene, Some(primitive), &selected_set).unwrap();
        let readout =
            derive_multi_transform_readout(&scene, &baseline, &GizmoSpace::World, Vec3::ZERO)
                .unwrap();

        assert!(!baseline.supports_scale());
        assert!(!readout.scale_enabled);
        assert_vec3_close(readout.scale_factor, Vec3::ONE);
    }
}
