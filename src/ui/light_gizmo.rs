use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
use glam::Vec3;
use std::collections::HashMap;

use crate::gpu::camera::Camera;
use crate::graph::scene::{LightType, NodeData, NodeId, Scene};
use crate::ui::gizmo::world_to_screen;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Radius (in screen pixels) for billboard icon hit testing.
const ICON_HIT_RADIUS: f32 = 16.0;
/// Minimum icon size in pixels.
const ICON_SIZE_MIN: f32 = 12.0;
/// Maximum icon size in pixels.
const ICON_SIZE_MAX: f32 = 48.0;
/// Distance at which icons start fading.
const FADE_START_DISTANCE: f32 = 50.0;
/// Number of segments for wireframe circles.
const CIRCLE_SEGMENTS: usize = 48;
/// Stroke width for wireframe gizmos.
const WIREFRAME_STROKE: f32 = 1.5;

// ---------------------------------------------------------------------------
// Light info extraction
// ---------------------------------------------------------------------------

/// Extracted light data needed for drawing.
struct LightInfo {
    light_id: NodeId,
    transform_id: NodeId,
    world_pos: Vec3,
    light_type: LightType,
    color: Vec3,
    range: f32,
    spot_angle: f32,
    intensity: f32,
    /// Direction the light points (from Transform rotation). Default -Y.
    direction: Vec3,
    /// Whether this light has an SDF cookie shape attached.
    has_cookie: bool,
    /// For Array lights: the expanded instance positions in world space.
    array_positions: Vec<Vec3>,
    /// For Array lights: per-instance colors after hue variation.
    array_colors: Vec<Vec3>,
}

/// Collect all visible Light nodes and their world-space transforms.
fn collect_lights(scene: &Scene, parent_map: &HashMap<NodeId, NodeId>) -> Vec<LightInfo> {
    let mut lights = Vec::new();

    for (&id, node) in &scene.nodes {
        if let NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            spot_angle,
            cookie_node,
            array_config,
            ..
        } = &node.data
        {
            if scene.is_hidden(id) {
                continue;
            }

            // Find the parent Transform to get world position and rotation
            let Some(&transform_id) = parent_map.get(&id) else {
                continue;
            };
            if scene.is_hidden(transform_id) {
                continue;
            }
            let Some(transform_node) = scene.nodes.get(&transform_id) else {
                continue;
            };
            let NodeData::Transform {
                translation,
                rotation,
                ..
            } = &transform_node.data
            else {
                continue;
            };

            // Compute direction from rotation (default light direction is -Y).
            // Use inverse rotation so direction arrows match gizmo drag convention.
            let direction = inverse_rotate_euler(Vec3::NEG_Y, *rotation).normalize_or_zero();

            // Compute array instance positions and colors for Array lights
            let (array_positions, array_colors) = if *light_type == LightType::Array {
                if let Some(cfg) = array_config {
                    let local_positions =
                        compute_array_positions(&cfg.pattern, cfg.count, cfg.radius);
                    let world_positions: Vec<Vec3> = local_positions
                        .iter()
                        .map(|lp| *translation + *lp)
                        .collect();
                    let colors: Vec<Vec3> = (0..local_positions.len())
                        .map(|i| {
                            if cfg.color_variation > 0.0 {
                                let hue_shift = (i as f32 / local_positions.len() as f32)
                                    * cfg.color_variation
                                    * 360.0;
                                hue_rotate(*color, hue_shift)
                            } else {
                                *color
                            }
                        })
                        .collect();
                    (world_positions, colors)
                } else {
                    (Vec::new(), Vec::new())
                }
            } else {
                (Vec::new(), Vec::new())
            };

            lights.push(LightInfo {
                light_id: id,
                transform_id,
                world_pos: *translation,
                light_type: light_type.clone(),
                color: *color,
                intensity: *intensity,
                range: *range,
                spot_angle: *spot_angle,
                direction: if direction.length_squared() > 0.5 {
                    direction
                } else {
                    Vec3::NEG_Y
                },
                has_cookie: cookie_node.is_some(),
                array_positions,
                array_colors,
            });
        }
    }

    lights
}

/// Inverse Euler XYZ rotation (applies -Z, -Y, -X — matches gizmo drag convention).
fn inverse_rotate_euler(p: Vec3, r: Vec3) -> Vec3 {
    let mut q = p;
    let (sz, cz) = (-r.z).sin_cos();
    q = Vec3::new(cz * q.x - sz * q.y, sz * q.x + cz * q.y, q.z);
    let (sy, cy) = (-r.y).sin_cos();
    q = Vec3::new(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
    let (sx, cx) = (-r.x).sin_cos();
    Vec3::new(q.x, cx * q.y - sx * q.z, sx * q.y + cx * q.z)
}

/// Compute local positions for a Light Array pattern (same logic as buffers.rs).
fn compute_array_positions(
    pattern: &crate::graph::scene::ArrayPattern,
    count: u32,
    radius: f32,
) -> Vec<Vec3> {
    use crate::graph::scene::ArrayPattern;
    let n = count.max(1) as usize;
    let mut positions = Vec::with_capacity(n);
    match pattern {
        ArrayPattern::Ring => {
            for i in 0..n {
                let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
                positions.push(Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius));
            }
        }
        ArrayPattern::Line => {
            let total_length = radius * 2.0;
            for i in 0..n {
                let t = if n > 1 {
                    i as f32 / (n - 1) as f32
                } else {
                    0.5
                };
                positions.push(Vec3::new(-radius + t * total_length, 0.0, 0.0));
            }
        }
        ArrayPattern::Grid => {
            let side = (n as f32).sqrt().ceil() as usize;
            let mut placed = 0;
            for row in 0..side {
                for col in 0..side {
                    if placed >= n {
                        break;
                    }
                    let tx = if side > 1 {
                        col as f32 / (side - 1) as f32
                    } else {
                        0.5
                    };
                    let tz = if side > 1 {
                        row as f32 / (side - 1) as f32
                    } else {
                        0.5
                    };
                    positions.push(Vec3::new(
                        -radius + tx * radius * 2.0,
                        0.0,
                        -radius + tz * radius * 2.0,
                    ));
                    placed += 1;
                }
            }
        }
        ArrayPattern::Spiral => {
            for i in 0..n {
                let t = i as f32 / n.max(1) as f32;
                let angle = t * std::f32::consts::TAU * 2.0;
                let r = t * radius;
                positions.push(Vec3::new(angle.cos() * r, 0.0, angle.sin() * r));
            }
        }
    }
    positions
}

/// Rotate a color's hue by a given number of degrees.
fn hue_rotate(color: Vec3, degrees: f32) -> Vec3 {
    let r = color.x;
    let g = color.y;
    let b = color.z;
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    let delta = max_c - min_c;
    let hue = if delta < 1e-6 {
        0.0
    } else if (max_c - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max_c - g).abs() < 1e-6 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let saturation = if max_c < 1e-6 { 0.0 } else { delta / max_c };
    let value = max_c;
    let new_hue = (hue + degrees).rem_euclid(360.0);
    let c = value * saturation;
    let x = c * (1.0 - ((new_hue / 60.0) % 2.0 - 1.0).abs());
    let m = value - c;
    let (r1, g1, b1) = if new_hue < 60.0 {
        (c, x, 0.0)
    } else if new_hue < 120.0 {
        (x, c, 0.0)
    } else if new_hue < 180.0 {
        (0.0, c, x)
    } else if new_hue < 240.0 {
        (0.0, x, c)
    } else if new_hue < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    Vec3::new(r1 + m, g1 + m, b1 + m)
}

// ---------------------------------------------------------------------------
// Billboard icon drawing
// ---------------------------------------------------------------------------

/// Compute icon size based on distance to camera (closer = larger).
fn icon_size(distance: f32) -> f32 {
    (30.0 / (distance * 0.15 + 1.0)).clamp(ICON_SIZE_MIN, ICON_SIZE_MAX)
}

/// Compute icon alpha based on distance to camera (fade at far distances).
fn icon_alpha(distance: f32) -> f32 {
    if distance > FADE_START_DISTANCE {
        ((60.0 - distance) / 10.0).clamp(0.1, 1.0)
    } else {
        1.0
    }
}

/// Convert a Vec3 color (0-1) to egui Color32 with the given alpha.
fn light_color_to_egui(color: Vec3, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (color.x.clamp(0.0, 1.0) * 255.0) as u8,
        (color.y.clamp(0.0, 1.0) * 255.0) as u8,
        (color.z.clamp(0.0, 1.0) * 255.0) as u8,
        (alpha * 255.0) as u8,
    )
}

/// Draw a point light billboard: filled circle with radiating ray lines.
fn draw_point_light_icon(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let core_radius = size * 0.3;
    let ray_inner = size * 0.4;
    let ray_outer = size * 0.7;
    let ray_count = 8;

    // Filled center circle
    painter.circle_filled(center, core_radius, color);

    // Radiating rays
    let ray_stroke = Stroke::new(1.5, color);
    for i in 0..ray_count {
        let angle = (i as f32 / ray_count as f32) * std::f32::consts::TAU;
        let dir = egui::vec2(angle.cos(), angle.sin());
        let start = center + dir * ray_inner;
        let end = center + dir * ray_outer;
        painter.line_segment([start, end], ray_stroke);
    }
}

/// Draw a spot light billboard: small triangle/cone icon.
fn draw_spot_light_icon(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let half = size * 0.5;
    // Triangle pointing down (default light direction is -Y)
    let top_left = Pos2::new(center.x - half * 0.4, center.y - half * 0.5);
    let top_right = Pos2::new(center.x + half * 0.4, center.y - half * 0.5);
    let bottom = Pos2::new(center.x, center.y + half * 0.7);

    painter.add(egui::Shape::convex_polygon(
        vec![top_left, top_right, bottom],
        color,
        Stroke::NONE,
    ));

    // Small circle at top to represent the light source
    painter.circle_filled(center + egui::vec2(0.0, -half * 0.3), size * 0.15, color);
}

/// Draw a directional light billboard: sun icon (circle with parallel ray lines).
fn draw_directional_light_icon(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let core_radius = size * 0.25;
    let ray_inner = size * 0.35;
    let ray_outer = size * 0.65;
    let ray_count = 12;

    // Filled center circle (sun)
    painter.circle_filled(center, core_radius, color);
    // Outline ring
    painter.circle_stroke(center, core_radius, Stroke::new(1.0, color));

    // Radiating rays (longer, thinner)
    let ray_stroke = Stroke::new(1.0, color);
    for i in 0..ray_count {
        let angle = (i as f32 / ray_count as f32) * std::f32::consts::TAU;
        let dir = egui::vec2(angle.cos(), angle.sin());
        let start = center + dir * ray_inner;
        let end = center + dir * ray_outer;
        painter.line_segment([start, end], ray_stroke);
    }
}

/// Draw an ambient light billboard: concentric rings (represents omnidirectional illumination).
fn draw_ambient_light_icon(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let ring_stroke = Stroke::new(1.2, color);
    // Three concentric rings
    painter.circle_stroke(center, size * 0.2, ring_stroke);
    painter.circle_stroke(center, size * 0.4, ring_stroke);
    painter.circle_stroke(center, size * 0.6, ring_stroke);
    // Small filled center dot
    painter.circle_filled(center, size * 0.08, color);
}

/// Draw a stacked-dots icon for a Light Array (center billboard).
fn draw_array_light_icon(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    // Three stacked circles representing multiple lights
    let dot_radius = size * 0.15;
    let spacing = size * 0.28;
    painter.circle_filled(center + egui::vec2(0.0, -spacing), dot_radius, color);
    painter.circle_filled(center, dot_radius, color);
    painter.circle_filled(center + egui::vec2(0.0, spacing), dot_radius, color);
    // Enclosing dashed circle
    let dash_count = 12;
    let dash_stroke = Stroke::new(1.2, color);
    let radius = size * 0.55;
    for i in 0..dash_count {
        let a0 = (i as f32 / dash_count as f32) * std::f32::consts::TAU;
        let a1 = ((i as f32 + 0.5) / dash_count as f32) * std::f32::consts::TAU;
        let p0 = center + egui::vec2(a0.cos(), a0.sin()) * radius;
        let p1 = center + egui::vec2(a1.cos(), a1.sin()) * radius;
        painter.line_segment([p0, p1], dash_stroke);
    }
}

// ---------------------------------------------------------------------------
// Wireframe gizmo drawing (selected lights only)
// ---------------------------------------------------------------------------

/// Draw a wireframe circle in 3D, projected to screen.
fn draw_wireframe_circle(
    painter: &egui::Painter,
    center: Vec3,
    axis: Vec3,
    radius: f32,
    vp: &glam::Mat4,
    rect: Rect,
    stroke: Stroke,
) {
    let up = if axis.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = axis.cross(up).normalize();
    let bitangent = axis.cross(tangent).normalize();

    let mut prev_screen: Option<Pos2> = None;
    for i in 0..=CIRCLE_SEGMENTS {
        let angle = (i as f32 / CIRCLE_SEGMENTS as f32) * std::f32::consts::TAU;
        let world_pt = center + (tangent * angle.cos() + bitangent * angle.sin()) * radius;
        if let Some(screen_pt) = world_to_screen(world_pt, vp, rect) {
            if let Some(prev) = prev_screen {
                painter.line_segment([prev, screen_pt], stroke);
            }
            prev_screen = Some(screen_pt);
        } else {
            prev_screen = None;
        }
    }
}

/// Draw wireframe range sphere for point light (3 orthogonal circles).
fn draw_point_light_wireframe(
    painter: &egui::Painter,
    center: Vec3,
    range: f32,
    vp: &glam::Mat4,
    rect: Rect,
    color: Color32,
) {
    let stroke = Stroke::new(WIREFRAME_STROKE, color.gamma_multiply(0.5));
    draw_wireframe_circle(painter, center, Vec3::X, range, vp, rect, stroke);
    draw_wireframe_circle(painter, center, Vec3::Y, range, vp, rect, stroke);
    draw_wireframe_circle(painter, center, Vec3::Z, range, vp, rect, stroke);
}

/// Draw wireframe cone for spot light.
#[allow(clippy::too_many_arguments)]
fn draw_spot_light_wireframe(
    painter: &egui::Painter,
    center: Vec3,
    direction: Vec3,
    range: f32,
    spot_angle_deg: f32,
    vp: &glam::Mat4,
    rect: Rect,
    color: Color32,
) {
    let outer_stroke = Stroke::new(WIREFRAME_STROKE, color.gamma_multiply(0.5));
    let inner_stroke = Stroke::new(WIREFRAME_STROKE * 0.7, color.gamma_multiply(0.25));
    let half_angle_rad = (spot_angle_deg * 0.5).to_radians();
    let cone_base_radius = range * half_angle_rad.tan();

    // Inner cone at 80% of outer angle (matches shader falloff)
    let inner_half_angle_rad = (spot_angle_deg * 0.5 * 0.8).to_radians();
    let inner_cone_radius = range * inner_half_angle_rad.tan();

    // Compute cone base center
    let base_center = center + direction * range;

    // Draw outer circle at cone base
    draw_wireframe_circle(
        painter,
        base_center,
        direction,
        cone_base_radius,
        vp,
        rect,
        outer_stroke,
    );
    // Draw inner circle at cone base (dimmer)
    draw_wireframe_circle(
        painter,
        base_center,
        direction,
        inner_cone_radius,
        vp,
        rect,
        inner_stroke,
    );

    // Draw cone lines from tip to base edge
    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = direction.cross(up).normalize();
    let bitangent = direction.cross(tangent).normalize();

    let line_count = 8;
    for i in 0..line_count {
        let angle = (i as f32 / line_count as f32) * std::f32::consts::TAU;
        // Outer cone silhouette lines
        let edge =
            base_center + (tangent * angle.cos() + bitangent * angle.sin()) * cone_base_radius;
        if let (Some(tip_screen), Some(edge_screen)) = (
            world_to_screen(center, vp, rect),
            world_to_screen(edge, vp, rect),
        ) {
            painter.line_segment([tip_screen, edge_screen], outer_stroke);
        }
        // Inner cone silhouette lines (dimmer, every other line)
        if i % 2 == 0 {
            let inner_edge =
                base_center + (tangent * angle.cos() + bitangent * angle.sin()) * inner_cone_radius;
            if let (Some(tip_screen), Some(edge_screen)) = (
                world_to_screen(center, vp, rect),
                world_to_screen(inner_edge, vp, rect),
            ) {
                painter.line_segment([tip_screen, edge_screen], inner_stroke);
            }
        }
    }
}

/// Draw parallel arrow lines for directional light.
fn draw_directional_light_wireframe(
    painter: &egui::Painter,
    center: Vec3,
    direction: Vec3,
    vp: &glam::Mat4,
    rect: Rect,
    color: Color32,
) {
    let stroke = Stroke::new(WIREFRAME_STROKE, color.gamma_multiply(0.5));
    let arrow_length = 5.0;
    let spread = 0.8;

    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = direction.cross(up).normalize();
    let bitangent = direction.cross(tangent).normalize();

    // Draw 5 parallel arrows (center + 4 corners)
    let offsets = [
        Vec3::ZERO,
        tangent * spread,
        tangent * -spread,
        bitangent * spread,
        bitangent * -spread,
    ];

    for offset in &offsets {
        let start = center + *offset;
        let end = start + direction * arrow_length;
        if let (Some(s), Some(e)) = (
            world_to_screen(start, vp, rect),
            world_to_screen(end, vp, rect),
        ) {
            painter.line_segment([s, e], stroke);
            // Small arrowhead
            let dir_2d = e - s;
            let len = dir_2d.length();
            if len > 5.0 {
                let d = dir_2d / len;
                let perp = egui::vec2(-d.y, d.x);
                let tip_size = 5.0;
                let left = e - d * tip_size + perp * tip_size * 0.4;
                let right = e - d * tip_size - perp * tip_size * 0.4;
                painter.add(egui::Shape::convex_polygon(
                    vec![e, left, right],
                    color.gamma_multiply(0.5),
                    Stroke::NONE,
                ));
            }
        }
    }
}

/// Draw dashed lines connecting array light instance positions (wireframe pattern outline).
fn draw_array_wireframe(
    painter: &egui::Painter,
    positions: &[Vec3],
    vp: &glam::Mat4,
    rect: Rect,
    color: Color32,
) {
    if positions.len() < 2 {
        return;
    }
    let stroke = Stroke::new(WIREFRAME_STROKE, color.gamma_multiply(0.5));
    // Connect instances in order with dashed lines
    for i in 0..positions.len() {
        let next = (i + 1) % positions.len();
        if let (Some(s), Some(e)) = (
            world_to_screen(positions[i], vp, rect),
            world_to_screen(positions[next], vp, rect),
        ) {
            // Dashed line
            let dir = e - s;
            let len = dir.length();
            if len < 1.0 {
                continue;
            }
            let d = dir / len;
            let dash_len = 6.0;
            let gap_len = 4.0;
            let mut t = 0.0;
            while t < len {
                let end_t = (t + dash_len).min(len);
                let p0 = s + d * t;
                let p1 = s + d * end_t;
                painter.line_segment([p0, p1], stroke);
                t = end_t + gap_len;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of light gizmo interaction — the transform_id of a clicked light billboard.
pub struct LightGizmoResult {
    pub clicked_transform_id: Option<NodeId>,
}

/// Draw all light billboard icons and wireframe gizmos for selected lights.
/// Returns the transform_id of a clicked light billboard (for selection).
#[allow(clippy::too_many_arguments)]
pub fn draw_and_interact(
    painter: &egui::Painter,
    camera: &Camera,
    scene: &Scene,
    selected: Option<NodeId>,
    rect: Rect,
    mouse_pos: Option<Pos2>,
    mouse_clicked: bool,
    active_light_ids: &std::collections::HashSet<NodeId>,
) -> LightGizmoResult {
    let parent_map = scene.build_parent_map();
    let lights = collect_lights(scene, &parent_map);

    let view = camera.view_matrix();
    let aspect = rect.width() / rect.height().max(1.0);
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;
    let cam_pos = camera.eye();

    let mut clicked_transform_id: Option<NodeId> = None;
    let mut closest_click_dist = f32::MAX;

    for light in &lights {
        let dist = (light.world_pos - cam_pos).length();
        let size = icon_size(dist);
        let alpha = icon_alpha(dist);

        let Some(screen_pos) = world_to_screen(light.world_pos, &vp, rect) else {
            continue;
        };

        // Skip if outside viewport
        if !rect.contains(screen_pos) {
            continue;
        }

        let is_active = active_light_ids.contains(&light.light_id);

        let color = light_color_to_egui(light.color, alpha);
        // Ensure icon is visible even for dark light colors
        let draw_color = if !is_active {
            // Inactive lights (beyond 8-light limit) get desaturated gray
            Color32::from_rgba_unmultiplied(120, 120, 120, (alpha * 0.5 * 255.0) as u8)
        } else if light.color.length() < 0.3 {
            Color32::from_rgba_unmultiplied(200, 200, 200, (alpha * 255.0) as u8)
        } else {
            color
        };

        // Check if this light (or its transform) is selected
        let is_selected = selected == Some(light.light_id) || selected == Some(light.transform_id);

        // Draw highlight ring when selected
        if is_selected {
            painter.circle_stroke(
                screen_pos,
                size * 0.85,
                Stroke::new(2.0, Color32::from_rgb(255, 200, 50)),
            );
        }

        // Draw billboard icon
        match light.light_type {
            LightType::Point => draw_point_light_icon(painter, screen_pos, size, draw_color),
            LightType::Spot => draw_spot_light_icon(painter, screen_pos, size, draw_color),
            LightType::Directional => {
                draw_directional_light_icon(painter, screen_pos, size, draw_color);
            }
            LightType::Ambient => draw_ambient_light_icon(painter, screen_pos, size, draw_color),
            LightType::Array => {
                // Draw a stacked-dots icon at the array center
                draw_array_light_icon(painter, screen_pos, size, draw_color);
                // Draw smaller point light icons at each instance position
                let instance_size = size * 0.6;
                for (i, pos) in light.array_positions.iter().enumerate() {
                    if let Some(inst_screen) = world_to_screen(*pos, &vp, rect) {
                        if rect.contains(inst_screen) {
                            let inst_color = if i < light.array_colors.len() {
                                light_color_to_egui(light.array_colors[i], alpha)
                            } else {
                                draw_color
                            };
                            draw_point_light_icon(painter, inst_screen, instance_size, inst_color);
                        }
                    }
                }
            }
        }

        // Draw minus sign overlay for negative/subtractive lights
        if light.intensity < 0.0 {
            let minus_half = size * 0.35;
            let minus_stroke = Stroke::new(2.5, Color32::from_rgb(255, 60, 60));
            painter.line_segment(
                [
                    screen_pos + egui::vec2(-minus_half, 0.0),
                    screen_pos + egui::vec2(minus_half, 0.0),
                ],
                minus_stroke,
            );
            // Dashed circle to indicate subtractive nature
            let dash_radius = size * 0.75;
            let dash_count = 12;
            let dash_stroke = Stroke::new(
                1.5,
                Color32::from_rgba_unmultiplied(255, 60, 60, (alpha * 200.0) as u8),
            );
            for i in 0..dash_count {
                let a0 = (i as f32 / dash_count as f32) * std::f32::consts::TAU;
                let a1 = ((i as f32 + 0.5) / dash_count as f32) * std::f32::consts::TAU;
                let p0 = screen_pos + egui::vec2(a0.cos(), a0.sin()) * dash_radius;
                let p1 = screen_pos + egui::vec2(a1.cos(), a1.sin()) * dash_radius;
                painter.line_segment([p0, p1], dash_stroke);
            }
        }

        // Draw cookie badge (small "C" in a circle) when light has an SDF cookie
        if light.has_cookie {
            let badge_offset = egui::vec2(size * 0.6, -size * 0.6);
            let badge_center = screen_pos + badge_offset;
            let badge_radius = size * 0.25;
            let badge_bg = Color32::from_rgba_unmultiplied(40, 120, 200, (alpha * 220.0) as u8);
            let badge_text_col =
                Color32::from_rgba_unmultiplied(255, 255, 255, (alpha * 255.0) as u8);
            painter.circle_filled(badge_center, badge_radius + 1.0, badge_bg);
            painter.text(
                badge_center,
                egui::Align2::CENTER_CENTER,
                "C",
                egui::FontId::proportional(badge_radius * 1.6),
                badge_text_col,
            );
        }

        // Draw small "X" over inactive light icons
        if !is_active {
            let x_size = size * 0.3;
            let x_stroke = Stroke::new(2.0, Color32::from_rgb(255, 80, 80));
            painter.line_segment(
                [
                    screen_pos + egui::vec2(-x_size, -x_size),
                    screen_pos + egui::vec2(x_size, x_size),
                ],
                x_stroke,
            );
            painter.line_segment(
                [
                    screen_pos + egui::vec2(x_size, -x_size),
                    screen_pos + egui::vec2(-x_size, x_size),
                ],
                x_stroke,
            );
        }

        // Draw wireframe gizmo when selected
        if is_selected {
            let wireframe_color = Color32::from_rgb(255, 220, 50);
            match light.light_type {
                LightType::Point => {
                    draw_point_light_wireframe(
                        painter,
                        light.world_pos,
                        light.range,
                        &vp,
                        rect,
                        wireframe_color,
                    );
                }
                LightType::Spot => {
                    draw_spot_light_wireframe(
                        painter,
                        light.world_pos,
                        light.direction,
                        light.range,
                        light.spot_angle,
                        &vp,
                        rect,
                        wireframe_color,
                    );
                }
                LightType::Directional => {
                    draw_directional_light_wireframe(
                        painter,
                        light.world_pos,
                        light.direction,
                        &vp,
                        rect,
                        wireframe_color,
                    );
                }
                LightType::Ambient => {
                    // Ambient has no spatial extent — no wireframe gizmo needed.
                    // Just show a highlight ring around the billboard icon.
                    painter.circle_stroke(
                        screen_pos,
                        size * 0.8,
                        Stroke::new(WIREFRAME_STROKE, wireframe_color),
                    );
                }
                LightType::Array => {
                    // Draw dashed outline connecting array instance positions
                    draw_array_wireframe(
                        painter,
                        &light.array_positions,
                        &vp,
                        rect,
                        wireframe_color,
                    );
                }
            }
        }

        // Hit test for billboard click (screen-space distance to icon center)
        if mouse_clicked {
            if let Some(mp) = mouse_pos {
                let click_dist = mp.distance(screen_pos);
                if click_dist < ICON_HIT_RADIUS.max(size * 0.5) && click_dist < closest_click_dist {
                    closest_click_dist = click_dist;
                    // Return the transform_id since that's what gets selected
                    // (industry standard: select the positionable parent)
                    clicked_transform_id = Some(light.transform_id);
                }
            }
        }
    }

    LightGizmoResult {
        clicked_transform_id,
    }
}
