use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use glam::{Mat4, Vec3, Vec4};

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, NodeId, Scene};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoMode {
    Translate,
    // Future: Rotate, Scale
}

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    fn direction(&self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
        }
    }

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
}

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
        start_screen_pos: Pos2,
        start_world_pos: Vec3,
    },
}

// ---------------------------------------------------------------------------
// Projection helpers
// ---------------------------------------------------------------------------

fn world_to_screen(world_pos: Vec3, view_proj: &Mat4, viewport_rect: Rect) -> Option<Pos2> {
    let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    if clip.w <= 0.0 {
        return None; // Behind camera
    }
    let ndc = clip.truncate() / clip.w;
    // ndc: [-1, 1] → screen coords
    let x = viewport_rect.min.x + (ndc.x * 0.5 + 0.5) * viewport_rect.width();
    let y = viewport_rect.min.y + (-ndc.y * 0.5 + 0.5) * viewport_rect.height();
    Some(Pos2::new(x, y))
}

fn point_to_segment_dist(point: Pos2, seg_start: Pos2, seg_end: Pos2) -> f32 {
    let ab = seg_end - seg_start;
    let ap = point - seg_start;
    let t = ab.dot(ap) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    let closest = seg_start + ab * t;
    point.distance(closest)
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AXIS_LENGTH: f32 = 1.2;
const HIT_THRESHOLD: f32 = 12.0;
const ARROW_SIZE: f32 = 8.0;
const AXIS_STROKE_WIDTH: f32 = 2.5;

const COLOR_X: Color32 = Color32::from_rgb(220, 60, 60);
const COLOR_Y: Color32 = Color32::from_rgb(60, 200, 60);
const COLOR_Z: Color32 = Color32::from_rgb(60, 100, 240);
const COLOR_X_HOVER: Color32 = Color32::from_rgb(255, 120, 120);
const COLOR_Y_HOVER: Color32 = Color32::from_rgb(120, 255, 120);
const COLOR_Z_HOVER: Color32 = Color32::from_rgb(120, 160, 255);

// ---------------------------------------------------------------------------
// Main gizmo function
// ---------------------------------------------------------------------------

/// Draw the gizmo overlay and handle interaction.
/// Returns true if the gizmo consumed the interaction (suppresses orbit + pick).
pub fn draw_and_interact(
    painter: &egui::Painter,
    response: &egui::Response,
    camera: &Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    gizmo_state: &mut GizmoState,
    _gizmo_mode: &GizmoMode,
    rect: Rect,
) -> bool {
    let Some(node_id) = selected else {
        *gizmo_state = GizmoState::Idle;
        return false;
    };

    // Show gizmo for nodes that have position
    let origin = match scene.nodes.get(&node_id).map(|n| &n.data) {
        Some(NodeData::Primitive { position, .. }) => *position,
        Some(NodeData::Sculpt { position, .. }) => *position,
        _ => {
            *gizmo_state = GizmoState::Idle;
            return false;
        }
    };

    let aspect = rect.width() / rect.height().max(1.0);
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;

    // Project origin + axis endpoints
    let Some(origin_screen) = world_to_screen(origin, &vp, rect) else {
        return false;
    };
    let x_end = origin + Vec3::X * AXIS_LENGTH;
    let y_end = origin + Vec3::Y * AXIS_LENGTH;
    let z_end = origin + Vec3::Z * AXIS_LENGTH;

    let Some(x_screen) = world_to_screen(x_end, &vp, rect) else {
        return false;
    };
    let Some(y_screen) = world_to_screen(y_end, &vp, rect) else {
        return false;
    };
    let Some(z_screen) = world_to_screen(z_end, &vp, rect) else {
        return false;
    };

    // Determine which axis is hovered
    let hover_pos = response.hover_pos();
    let hovered_axis = hover_pos.and_then(|pos| {
        let dx = point_to_segment_dist(pos, origin_screen, x_screen);
        let dy = point_to_segment_dist(pos, origin_screen, y_screen);
        let dz = point_to_segment_dist(pos, origin_screen, z_screen);
        let min = dx.min(dy).min(dz);
        if min > HIT_THRESHOLD {
            None
        } else if (min - dx).abs() < 0.01 {
            Some(GizmoAxis::X)
        } else if (min - dy).abs() < 0.01 {
            Some(GizmoAxis::Y)
        } else {
            Some(GizmoAxis::Z)
        }
    });

    // Determine axis colors (hover highlight)
    let dragging_axis = match gizmo_state {
        GizmoState::Dragging { ref axis, .. } => Some(axis.clone()),
        _ => None,
    };

    let x_color = axis_color(&GizmoAxis::X, &dragging_axis, &hovered_axis);
    let y_color = axis_color(&GizmoAxis::Y, &dragging_axis, &hovered_axis);
    let z_color = axis_color(&GizmoAxis::Z, &dragging_axis, &hovered_axis);

    // Draw axis lines
    painter.line_segment(
        [origin_screen, x_screen],
        Stroke::new(AXIS_STROKE_WIDTH, x_color),
    );
    painter.line_segment(
        [origin_screen, y_screen],
        Stroke::new(AXIS_STROKE_WIDTH, y_color),
    );
    painter.line_segment(
        [origin_screen, z_screen],
        Stroke::new(AXIS_STROKE_WIDTH, z_color),
    );

    // Draw arrow heads
    draw_arrow_head(painter, origin_screen, x_screen, x_color);
    draw_arrow_head(painter, origin_screen, y_screen, y_color);
    draw_arrow_head(painter, origin_screen, z_screen, z_color);

    // --- Interaction ---
    let mut consumed = false;

    // Start dragging
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(ref axis) = hovered_axis {
            *gizmo_state = GizmoState::Dragging {
                axis: axis.clone(),
                node_id,
                start_screen_pos: hover_pos.unwrap_or(origin_screen),
                start_world_pos: origin,
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
            let axis_dir = axis.direction();

            // Project axis direction to screen space
            let axis_screen_dir = {
                let end = world_to_screen(origin + axis_dir, &vp, rect)
                    .unwrap_or(origin_screen);
                let dir = end - origin_screen;
                let len = dir.length();
                if len > 0.1 { dir / len } else { Vec2::ZERO }
            };

            // Project mouse delta onto axis screen direction
            let delta = response.drag_delta();
            let projected = delta.dot(axis_screen_dir);

            // Scale: screen pixels → world units (approximate)
            let world_scale = camera.distance * 0.003;
            let world_delta = axis_dir * projected * world_scale;

            if let Some(node) = scene.nodes.get_mut(&drag_node) {
                match &mut node.data {
                    NodeData::Primitive { ref mut position, .. } => {
                        *position += world_delta;
                    }
                    NodeData::Sculpt { ref mut position, .. } => {
                        *position += world_delta;
                    }
                    _ => {}
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
