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

fn axis_color(axis: &GizmoAxis, active: &Option<GizmoAxis>, hovered: &Option<GizmoAxis>) -> Color32 {
    if active.as_ref() == Some(axis) || hovered.as_ref() == Some(axis) {
        axis.hover_color()
    } else {
        axis.color()
    }
}

#[derive(Clone, Debug)]
pub enum GizmoState {
    Idle,
    Dragging {
        axis: GizmoAxis,
        node_id: NodeId,
        _start_screen_pos: Pos2,
        _start_world_pos: Vec3,
        _start_rotation: Vec3,
        _start_scale: Vec3,
    },
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
        while a - reference > PI { a -= TAU; }
        while a - reference < -PI { a += TAU; }
        a
    }

    fn normalize_near(v: Vec3, prev: Vec3) -> Vec3 {
        Vec3::new(wrap_near(v.x, prev.x), wrap_near(v.y, prev.y), wrap_near(v.z, prev.z))
    }

    let (rz, ry, rx) = q.to_euler(glam::EulerRot::ZYX);
    let a = normalize_near(Vec3::new(rx, ry, rz), prev);

    // Alternative Euler representation for the same rotation
    let b = normalize_near(Vec3::new(rx + PI, PI - ry, rz + PI), prev);

    if (a - prev).length_squared() <= (b - prev).length_squared() { a } else { b }
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

fn screen_axis_dir(origin: Vec3, axis_dir: Vec3, origin_screen: Pos2, vp: &Mat4, rect: Rect) -> Vec2 {
    let end = world_to_screen(origin + axis_dir, vp, rect).unwrap_or(origin_screen);
    let dir = end - origin_screen;
    let len = dir.length();
    if len > 0.1 { dir / len } else { Vec2::ZERO }
}

// ---------------------------------------------------------------------------
// Node transform extraction
// ---------------------------------------------------------------------------

struct NodeTransform {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    has_scale: bool,
}

fn extract_node_transform(scene: &Scene, node_id: NodeId) -> Option<NodeTransform> {
    match scene.nodes.get(&node_id).map(|n| &n.data) {
        Some(NodeData::Primitive { position, rotation, scale, .. }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: *scale,
            has_scale: true,
        }),
        Some(NodeData::Sculpt { position, rotation, .. }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: Vec3::ONE,
            has_scale: false,
        }),
        _ => None,
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
    let up = if axis_dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
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

    if best_dist > RING_HIT_THRESHOLD { None } else { best_axis }
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
                painter.line_segment(
                    [prev, screen_pt],
                    Stroke::new(AXIS_STROKE_WIDTH, color),
                );
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

    // When pivot is offset, translate moves position normally (pivot follows)
    let _ = pivot_offset;
    let _ = node_rotation;

    apply_position_delta(scene, drag_node, world_delta);
}

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
        if let NodeData::Primitive { ref mut scale, ref mut position, .. } = node.data {
            // When pivot is offset, scaling around pivot shifts position
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
}

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
    let sign = if axis_dir.dot(view_dir) >= 0.0 { 1.0 } else { -1.0 };
    let angle_delta = projected * ROTATE_SENSITIVITY * sign;

    let delta_quat = Quat::from_axis_angle(axis_dir, angle_delta);

    if let Some(node) = scene.nodes.get_mut(&drag_node) {
        match &mut node.data {
            NodeData::Primitive { ref mut rotation, ref mut position, .. } => {
                if pivot_offset.length_squared() > 1e-6 {
                    let pivot_world = *position + inverse_rotate_euler(*pivot_offset, node_rotation);
                    let offset = *position - pivot_world;
                    *position = pivot_world + delta_quat * offset;
                }
                apply_rotation_delta(rotation, axis, angle_delta, axis_dir, gizmo_space);
            }
            NodeData::Sculpt { ref mut rotation, ref mut position, .. } => {
                if pivot_offset.length_squared() > 1e-6 {
                    let pivot_world = *position + inverse_rotate_euler(*pivot_offset, node_rotation);
                    let offset = *position - pivot_world;
                    *position = pivot_world + delta_quat * offset;
                }
                apply_rotation_delta(rotation, axis, angle_delta, axis_dir, gizmo_space);
            }
            _ => {}
        }
    }
}

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

fn apply_position_delta(scene: &mut Scene, node_id: NodeId, delta: Vec3) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        match &mut node.data {
            NodeData::Primitive { ref mut position, .. } => *position += delta,
            NodeData::Sculpt { ref mut position, .. } => *position += delta,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Pivot drag (Alt + drag moves pivot instead of object)
// ---------------------------------------------------------------------------

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

    // Convert world delta to local space for the pivot offset
    let inv_delta = inverse_rotate_euler(world_delta, node_rotation);
    *pivot_offset += inv_delta;
}

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

pub fn draw_and_interact(
    painter: &egui::Painter,
    response: &egui::Response,
    camera: &Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: &mut Vec3,
    rect: Rect,
    snap_config: &SnapConfig,
) -> bool {
    let Some(node_id) = selected else {
        *gizmo_state = GizmoState::Idle;
        return false;
    };

    let Some(nt) = extract_node_transform(scene, node_id) else {
        *gizmo_state = GizmoState::Idle;
        return false;
    };

    // Skip scale gizmo for nodes without scale
    if *gizmo_mode == GizmoMode::Scale && !nt.has_scale {
        *gizmo_state = GizmoState::Idle;
        return false;
    }

    let axes = compute_axis_directions(nt.rotation, gizmo_space);
    let gizmo_center = nt.position + rotate_euler(*pivot_offset, nt.rotation);

    let aspect = rect.width() / rect.height().max(1.0);
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;

    let Some(origin_screen) = world_to_screen(gizmo_center, &vp, rect) else {
        return false;
    };

    // Check if Alt is held (pivot drag mode)
    let alt_held = response.ctx.input(|i| i.modifiers.alt);

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
        GizmoMode::Rotate => {
            hit_test_rings(pos, gizmo_center, &axes, &vp, rect)
        }
    });

    // Determine colors
    let dragging_axis = match gizmo_state {
        GizmoState::Dragging { ref axis, .. } => Some(axis.clone()),
        _ => None,
    };
    let colors = [
        axis_color(&GizmoAxis::X, &dragging_axis, &hovered_axis),
        axis_color(&GizmoAxis::Y, &dragging_axis, &hovered_axis),
        axis_color(&GizmoAxis::Z, &dragging_axis, &hovered_axis),
    ];

    // Draw
    match gizmo_mode {
        GizmoMode::Translate => draw_translate_gizmo(painter, origin_screen, &axis_screens, &colors),
        GizmoMode::Scale => draw_scale_gizmo(painter, origin_screen, &axis_screens, &colors),
        GizmoMode::Rotate => draw_rotate_gizmo(painter, gizmo_center, &axes, &colors, &vp, rect),
    }

    // Draw pivot indicator when offset
    if pivot_offset.length_squared() > 1e-6 {
        let node_origin_screen = world_to_screen(nt.position, &vp, rect);
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
            *gizmo_state = GizmoState::Dragging {
                axis: axis.clone(),
                node_id,
                _start_screen_pos: hover_pos.unwrap_or(origin_screen),
                _start_world_pos: nt.position,
                _start_rotation: nt.rotation,
                _start_scale: nt.scale,
            };
            consumed = true;
        }
    }

    // During drag
    if response.dragged_by(egui::PointerButton::Primary) {
        if let GizmoState::Dragging {
            ref axis,
            node_id: drag_node,
            ..
        } = gizmo_state
        {
            let drag_node = *drag_node;
            let axis_idx = axis.euler_index();
            let axis_dir = axes[axis_idx];
            let axis_clone = axis.clone();

            if alt_held {
                // Alt+drag: move pivot
                handle_pivot_drag(
                    response, axis_dir, gizmo_center, origin_screen,
                    camera, &vp, rect, pivot_offset, nt.rotation,
                );
            } else {
                let ctrl_held = response.ctx.input(|i| i.modifiers.ctrl);
                match gizmo_mode {
                    GizmoMode::Translate => {
                        handle_translate_drag(
                            response, axis_dir, gizmo_center, origin_screen,
                            camera, &vp, rect, scene, drag_node,
                            pivot_offset, nt.rotation,
                        );
                        if ctrl_held {
                            snap_position(scene, drag_node, axis_idx, snap_config.translate_snap);
                        }
                    }
                    GizmoMode::Scale => {
                        handle_scale_drag(
                            response, &axis_clone, axis_dir, gizmo_center, origin_screen,
                            camera, &vp, rect, scene, drag_node,
                            pivot_offset, nt.rotation,
                        );
                        if ctrl_held {
                            snap_scale(scene, drag_node, axis_idx, snap_config.scale_snap);
                        }
                    }
                    GizmoMode::Rotate => {
                        let view_dir = camera.eye() - gizmo_center;
                        handle_rotate_drag(
                            response, &axis_clone, axis_dir, origin_screen,
                            scene, drag_node, pivot_offset, nt.rotation, gizmo_space,
                            view_dir,
                        );
                        if ctrl_held {
                            snap_rotation(scene, drag_node, axis_idx, snap_config.rotate_snap.to_radians());
                        }
                    }
                }
            }
            consumed = true;
        }
    }

    // End drag
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        if matches!(gizmo_state, GizmoState::Dragging { .. }) {
            *gizmo_state = GizmoState::Idle;
            consumed = true;
        }
    }

    consumed
}

// ---------------------------------------------------------------------------
// Snap helpers — quantize position/rotation/scale to grid increments
// ---------------------------------------------------------------------------

fn snap_value(val: f32, snap: f32) -> f32 {
    (val / snap).round() * snap
}

fn snap_position(scene: &mut Scene, node_id: NodeId, axis_idx: usize, snap: f32) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        let pos = match &mut node.data {
            NodeData::Primitive { ref mut position, .. } => position,
            NodeData::Sculpt { ref mut position, .. } => position,
            _ => return,
        };
        match axis_idx {
            0 => pos.x = snap_value(pos.x, snap),
            1 => pos.y = snap_value(pos.y, snap),
            _ => pos.z = snap_value(pos.z, snap),
        }
    }
}

fn snap_rotation(scene: &mut Scene, node_id: NodeId, axis_idx: usize, snap_rad: f32) {
    if let Some(node) = scene.nodes.get_mut(&node_id) {
        let rot = match &mut node.data {
            NodeData::Primitive { ref mut rotation, .. } => rotation,
            NodeData::Sculpt { ref mut rotation, .. } => rotation,
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
        if let NodeData::Primitive { ref mut scale, .. } = node.data {
            match axis_idx {
                0 => scale.x = snap_value(scale.x, snap).max(0.01),
                1 => scale.y = snap_value(scale.y, snap).max(0.01),
                _ => scale.z = snap_value(scale.z, snap).max(0.01),
            }
        }
    }
}
