use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::app::actions::{Action, ActionSink};
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{
    CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive,
};
use crate::sculpt::{self, ActiveTool, BrushMode, SculptState};
use crate::settings::SnapConfig;
use crate::ui::gizmo::{self, GizmoMode, GizmoSpace, GizmoState};

use super::ViewportResources;

// ---------------------------------------------------------------------------
// Safety border helper
// ---------------------------------------------------------------------------

/// Returns true if the given position is within the safety border zone
/// (a fraction of viewport size along each edge). Used in sculpt mode to
/// guarantee navigation even when geometry fills the viewport.
fn in_safety_border(pos: egui::Pos2, rect: egui::Rect, fraction: f32) -> bool {
    if fraction <= 0.0 {
        return false;
    }
    let border = rect.width().min(rect.height()) * fraction;
    pos.x < rect.min.x + border
        || pos.x > rect.max.x - border
        || pos.y < rect.min.y + border
        || pos.y > rect.max.y - border
}

// ---------------------------------------------------------------------------
// CPU-side sculpt mesh hit test (voxel raycast)
// ---------------------------------------------------------------------------

/// Standard slab-method ray-AABB intersection. Returns (t_enter, t_exit).
/// If t_enter >= t_exit, the ray misses the box.
fn ray_aabb(origin: glam::Vec3, dir: glam::Vec3, box_min: glam::Vec3, box_max: glam::Vec3) -> (f32, f32) {
    let inv_dir = glam::Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    (t_enter, t_exit)
}

/// CPU-side voxel raycast: unproject cursor to a world ray, transform into
/// local voxel space, and march through the SDF grid. Returns true if the ray
/// hits actual solid geometry (SDF near zero), false if it passes through
/// empty voxels. This gives instant orbit-vs-sculpt with zero GPU latency.
///
/// For displacement grids (differential sculpt), the total SDF is:
///   analytical_sdf(input_child, world_pos) + displacement
/// so we evaluate both and combine them.
fn cursor_in_sculpt_bounds(
    cursor: egui::Pos2,
    sculpt_state: &SculptState,
    scene: &Scene,
    camera: &Camera,
    rect: egui::Rect,
) -> bool {
    use crate::graph::voxel;

    let node_id = match sculpt_state {
        SculptState::Active { node_id, .. } => *node_id,
        _ => return false,
    };
    let node = match scene.nodes.get(&node_id) {
        Some(n) => n,
        None => return false,
    };
    let (position, rotation, voxel_grid, input_child) = match &node.data {
        NodeData::Sculpt { position, rotation, voxel_grid, input, .. } => {
            (*position, *rotation, voxel_grid, *input)
        }
        _ => return false,
    };

    // Unproject cursor to world-space ray
    let aspect = rect.width() / rect.height().max(1.0);
    let ndc_x = ((cursor.x - rect.min.x) / rect.width()) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((cursor.y - rect.min.y) / rect.height()) * 2.0;
    let inv_vp = (camera.projection_matrix(aspect) * camera.view_matrix()).inverse();
    let near_h = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near = near_h.truncate() / near_h.w;
    let far = far_h.truncate() / far_h.w;
    let ray_dir = (far - near).normalize();

    // Transform ray into local voxel space (undo position + rotation)
    let local_origin = sculpt::inverse_rotate_euler(near - position, rotation);
    let local_dir = sculpt::inverse_rotate_euler(ray_dir, rotation).normalize();

    // Ray-AABB intersection with voxel grid bounds
    let (t_enter, t_exit) = ray_aabb(local_origin, local_dir, voxel_grid.bounds_min, voxel_grid.bounds_max);
    if t_enter >= t_exit {
        return false; // Ray misses the voxel grid entirely
    }

    // March through voxel grid, sphere-trace style with SDF sampling
    let cell_size = (voxel_grid.bounds_max - voxel_grid.bounds_min) / voxel_grid.resolution as f32;
    let min_step = cell_size.x.min(cell_size.y).min(cell_size.z);
    let threshold = min_step * 0.5;
    let mut t = t_enter.max(0.0);
    while t < t_exit {
        let local_pos = local_origin + local_dir * t;
        let displacement = voxel_grid.sample(local_pos);

        // For displacement grids, combine with analytical SDF from input child.
        // For total SDF grids, displacement IS the total SDF.
        let sdf = if voxel_grid.is_displacement {
            if let Some(child_id) = input_child {
                // Transform back to world space for analytical evaluation
                let world_pos = position + sculpt::inverse_rotate_euler(local_pos, rotation);
                let analytical = voxel::evaluate_sdf_tree(scene, child_id, world_pos);
                analytical + displacement
            } else {
                displacement // No input child — treat as total SDF
            }
        } else {
            displacement // Total SDF grid — value is already the distance
        };

        if sdf <= threshold {
            return true; // Near surface → sculpt
        }
        // Sphere-trace: jump by SDF distance (clamped to min step for safety)
        t += min_step.max(sdf.abs() * 0.9);
    }
    false // Ray passed through empty voxels → orbit
}

// ---------------------------------------------------------------------------
// Paint callback
// ---------------------------------------------------------------------------

struct ViewportCallback {
    /// Camera uniform with viewport set to render dimensions (offscreen texture size).
    render_uniform: CameraUniform,
    /// Display viewport in physical pixels: [x, y, width, height].
    display_viewport: [f32; 4],
    /// Render scale factor (0.25 - 1.0).
    render_scale: f32,
    /// Outline color (RGB).
    outline_color: [f32; 3],
    /// Outline width in pixels.
    outline_width: f32,
    /// Bloom parameters: [threshold, intensity, radius, enabled].
    bloom_params: [f32; 4],
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources.get_mut::<ViewportResources>().unwrap();

        let display_w = self.display_viewport[2] as u32;
        let display_h = self.display_viewport[3] as u32;
        let render_w = ((display_w as f32) * self.render_scale).max(1.0) as u32;
        let render_h = ((display_h as f32) * self.render_scale).max(1.0) as u32;

        // Ensure offscreen texture + blit bind group are the right size
        resources.ensure_offscreen_texture(device, render_w, render_h);

        // Write camera uniform (viewport = render dimensions for the SDF shader)
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&self.render_uniform),
        );

        // Write blit params (display viewport + outline settings + bloom for the blit shader)
        let blit_data: [f32; 12] = [
            self.display_viewport[0],
            self.display_viewport[1],
            self.display_viewport[2],
            self.display_viewport[3],
            self.outline_color[0],
            self.outline_color[1],
            self.outline_color[2],
            self.outline_width,
            self.bloom_params[0],
            self.bloom_params[1],
            self.bloom_params[2],
            self.bloom_params[3],
        ];
        queue.write_buffer(
            &resources.blit_params_buffer,
            0,
            bytemuck::cast_slice(&blit_data),
        );

        // Render SDF scene to offscreen texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Offscreen Encoder"),
        });
        {
            let view = resources.offscreen_view.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Offscreen SDF Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Use composite render pipeline when enabled, otherwise direct render
            if resources.use_composite {
                if let Some(ref comp) = resources.composite {
                    pass.set_pipeline(&comp.render_pipeline);
                    pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                    pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                    pass.set_bind_group(2, &comp.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&resources.pipeline);
                pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                pass.set_bind_group(2, &resources.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        if let Some(ref blit_bg) = resources.blit_bind_group {
            render_pass.set_pipeline(&resources.blit_pipeline);
            render_pass.set_bind_group(0, blit_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

fn brush_cursor_color(mode: &BrushMode) -> egui::Color32 {
    match mode {
        BrushMode::Add => egui::Color32::from_rgba_premultiplied(100, 200, 100, 160),
        BrushMode::Carve => egui::Color32::from_rgba_premultiplied(200, 100, 100, 160),
        BrushMode::Smooth => egui::Color32::from_rgba_premultiplied(100, 150, 255, 160),
        BrushMode::Flatten => egui::Color32::from_rgba_premultiplied(255, 200, 80, 160),
        BrushMode::Inflate => egui::Color32::from_rgba_premultiplied(200, 140, 255, 160),
        BrushMode::Grab => egui::Color32::from_rgba_premultiplied(255, 160, 60, 160),
    }
}

/// Compute the effective brush mode given modifier key state.
/// Shift → Smooth (overrides everything), Ctrl → invert (Add↔Carve, Inflate→Carve).
fn effective_brush_mode(base: &BrushMode, ctrl: bool, shift: bool) -> BrushMode {
    if shift {
        return BrushMode::Smooth;
    }
    if ctrl {
        return match base {
            BrushMode::Add => BrushMode::Carve,
            BrushMode::Carve => BrushMode::Add,
            BrushMode::Inflate => BrushMode::Carve,
            other => other.clone(),
        };
    }
    base.clone()
}

/// Draw a semi-transparent symmetry plane overlay at the mirror axis with axis label.
/// `scene_bounds` is used to adaptively size the plane to the scene.
fn draw_symmetry_plane(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    axis: u8,
    scene_bounds: ([f32; 3], [f32; 3]),
) {
    let aspect = rect.width() / rect.height();
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    // Adaptive extent: 1.5x the largest scene half-extent, minimum 2.0
    let half_extents = [
        (scene_bounds.1[0] - scene_bounds.0[0]) * 0.5,
        (scene_bounds.1[1] - scene_bounds.0[1]) * 0.5,
        (scene_bounds.1[2] - scene_bounds.0[2]) * 0.5,
    ];
    let max_half = half_extents[0].max(half_extents[1]).max(half_extents[2]);
    let extent = (max_half * 1.5).max(2.0);

    let corners: [glam::Vec3; 4] = match axis {
        0 => [
            glam::Vec3::new(0.0, -extent, -extent),
            glam::Vec3::new(0.0, -extent, extent),
            glam::Vec3::new(0.0, extent, extent),
            glam::Vec3::new(0.0, extent, -extent),
        ],
        1 => [
            glam::Vec3::new(-extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, extent),
            glam::Vec3::new(-extent, 0.0, extent),
        ],
        _ => [
            glam::Vec3::new(-extent, -extent, 0.0),
            glam::Vec3::new(extent, -extent, 0.0),
            glam::Vec3::new(extent, extent, 0.0),
            glam::Vec3::new(-extent, extent, 0.0),
        ],
    };

    let screen_pts: Vec<egui::Pos2> = corners
        .iter()
        .filter_map(|&c| gizmo::world_to_screen(c, &view_proj, rect))
        .collect();

    if screen_pts.len() < 3 {
        return;
    }

    let (fill, border, label, label_color) = match axis {
        0 => (
            egui::Color32::from_rgba_premultiplied(255, 50, 50, 20),
            egui::Color32::from_rgba_premultiplied(255, 80, 80, 80),
            "X",
            egui::Color32::from_rgb(255, 100, 100),
        ),
        1 => (
            egui::Color32::from_rgba_premultiplied(50, 255, 50, 20),
            egui::Color32::from_rgba_premultiplied(80, 255, 80, 80),
            "Y",
            egui::Color32::from_rgb(100, 255, 100),
        ),
        _ => (
            egui::Color32::from_rgba_premultiplied(50, 50, 255, 20),
            egui::Color32::from_rgba_premultiplied(80, 80, 255, 80),
            "Z",
            egui::Color32::from_rgb(100, 100, 255),
        ),
    };

    painter.add(egui::Shape::convex_polygon(
        screen_pts.clone(),
        fill,
        egui::Stroke::new(1.0, border),
    ));

    // Draw axis label at the top edge of the plane
    let label_world = match axis {
        0 => glam::Vec3::new(0.0, extent * 0.9, 0.0),
        1 => glam::Vec3::new(0.0, 0.0, extent * 0.9),
        _ => glam::Vec3::new(0.0, extent * 0.9, 0.0),
    };
    if let Some(label_pos) = gizmo::world_to_screen(label_world, &view_proj, rect) {
        let font = egui::FontId::proportional(13.0);
        painter.text(
            label_pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_CENTER,
            label,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            label,
            font,
            label_color,
        );
    }
}

pub struct ViewportOutput {
    pub pending_pick: Option<PendingPick>,
    /// Modifier keys at time of sculpt drag (Ctrl = invert, Shift = smooth).
    pub sculpt_ctrl_held: bool,
    pub sculpt_shift_held: bool,
    /// Pen pressure (0.0 = no pressure data, 1.0 = max).
    pub sculpt_pressure: f32,
    /// Brush radius delta from Ctrl+right-drag (horizontal movement).
    pub brush_radius_delta: f32,
    /// Brush strength delta from Ctrl+right-drag (vertical movement).
    pub brush_strength_delta: f32,
    /// True if this pick is a hover-only pick (no brush application).
    pub is_hover_pick: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    gizmo_visible: bool,
    pivot_offset: &mut glam::Vec3,
    sculpt_state: &SculptState,
    active_tool: &ActiveTool,
    time: f32,
    render_config: &crate::settings::RenderConfig,
    sculpt_count: usize,
    fps_info: Option<(f64, f64)>, // (fps, frame_ms)
    actions: &mut ActionSink,
    snap_config: &SnapConfig,
    isolation_label: Option<&str>,
    turntable_active: bool,
    last_sculpt_hit: Option<glam::Vec3>,
    hover_world_pos: Option<glam::Vec3>,
    _cursor_over_geometry: bool,
    active_light_ids: &std::collections::HashSet<crate::graph::scene::NodeId>,
    soloed_light: Option<NodeId>,
    solo_label: Option<&str>,
) -> ViewportOutput {
    let mut output = ViewportOutput {
        pending_pick: None,
        sculpt_ctrl_held: false,
        sculpt_shift_held: false,
        sculpt_pressure: 0.0,
        brush_radius_delta: 0.0,
        brush_strength_delta: 0.0,
        is_hover_pick: false,
    };
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

    // --- Paint the SDF viewport (WGPU callback) ---
    let pixels_per_point = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.width() * pixels_per_point,
        rect.height() * pixels_per_point,
    ];
    // Interaction detection
    let multi_touch_active = ui.input(|i| i.multi_touch()).is_some();
    let sculpt_active = sculpt_state.is_active();
    let camera_dragging = if sculpt_active {
        response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    } else {
        response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    };
    let sculpt_brushing = sculpt_active && response.dragged_by(egui::PointerButton::Primary);
    let multi_sculpt_reduce = render_config.auto_reduce_steps && sculpt_count >= 2;
    let is_interacting = camera_dragging || sculpt_brushing;
    // Stop turntable on any camera interaction
    if turntable_active && camera_dragging {
        actions.push(Action::ToggleTurntable);
    }
    // Fast quality mode: half steps + skip AO/shadows
    let quality_mode = if (is_interacting && render_config.sculpt_fast_mode) || multi_sculpt_reduce
    {
        1.0
    } else {
        0.0
    };

    // Resolution scaling: reduced resolution during interaction
    let render_scale = if is_interacting {
        render_config.interaction_render_scale.clamp(0.25, 1.0)
    } else {
        render_config.rest_render_scale.clamp(0.25, 1.0)
    };

    let scene_bounds = scene.compute_bounds();

    // Render uniform uses the RENDER viewport (offscreen texture dimensions)
    let render_w = (viewport[2] * render_scale).max(1.0);
    let render_h = (viewport[3] * render_scale).max(1.0);
    let render_viewport = [0.0, 0.0, render_w, render_h];
    let selected_idx = (*selected)
        .and_then(|id| {
            let order = scene.visible_topo_order();
            order.iter().position(|&nid| nid == id)
        })
        .map(|i| i as f32)
        .unwrap_or(-1.0);
    let shading_mode_val = render_config.shading_mode.gpu_value();

    // Compute brush_pos for 3D brush preview: [x, y, z, radius] or [0,0,0,0] when inactive.
    // Zero-latency tracking: project cursor through camera at last known surface depth.
    // GPU hover picks refine the depth each frame, but projection gives instant feedback.
    let brush_pos = if let SculptState::Active { brush_radius, .. } = sculpt_state {
        let reference_hit = hover_world_pos.or(last_sculpt_hit);
        let cursor = response.hover_pos();

        match (cursor, reference_hit) {
            (Some(cp), Some(hit)) if rect.contains(cp) => {
                // Cast ray through cursor at the depth of the last known hit
                let aspect = rect.width() / rect.height().max(1.0);
                let ndc_x = ((cp.x - rect.min.x) / rect.width()) * 2.0 - 1.0;
                let ndc_y = 1.0 - ((cp.y - rect.min.y) / rect.height()) * 2.0;
                let inv_vp = (camera.projection_matrix(aspect) * camera.view_matrix()).inverse();
                let near_h = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
                let near = near_h.truncate() / near_h.w;
                let far_h = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
                let far = far_h.truncate() / far_h.w;
                let ray_dir = (far - near).normalize();
                let depth = (hit - near).dot(ray_dir).max(0.001);
                let approx = near + ray_dir * depth;
                [approx.x, approx.y, approx.z, *brush_radius]
            }
            (_, Some(hit)) => [hit.x, hit.y, hit.z, *brush_radius],
            _ => [0.0; 4],
        }
    } else {
        [0.0; 4]
    };

    let cross_section = [
        render_config.cross_section_axis as f32,
        render_config.cross_section_position,
        0.0,
        0.0,
    ];
    // Collect scene lights for the GPU (up to 8, sorted by distance to camera)
    let (scene_light_count, scene_light_list, scene_ambient) =
        crate::gpu::buffers::collect_scene_lights(scene, camera.eye(), soloed_light, time);
    let volumetric_count = scene_light_list.iter().filter(|l| l.volumetric[0] > 0.5).count() as f32;
    let volumetric_steps = render_config.volumetric_steps as f32;
    let scene_light_info = [scene_light_count as f32, volumetric_count, volumetric_steps, 0.0];
    let mut scene_lights_flat = [[0.0_f32; 4]; 32];
    let mut scene_light_vol = [[0.0_f32; 4]; 8];
    for (i, light) in scene_light_list.iter().enumerate() {
        scene_lights_flat[i * 4] = light.position_type;
        scene_lights_flat[i * 4 + 1] = light.direction_intensity;
        scene_lights_flat[i * 4 + 2] = light.color_range;
        scene_lights_flat[i * 4 + 3] = light.params;
        scene_light_vol[i] = light.volumetric;
    }

    // Combine scene Ambient lights with render config fallback
    // Scene ambient color's luminance overrides config if any Ambient nodes exist
    let ambient_luminance = scene_ambient.color.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    let effective_ambient = if ambient_luminance > 0.0 {
        ambient_luminance
    } else {
        render_config.ambient
    };

    let render_uniform = camera.to_uniform(
        render_viewport,
        time,
        quality_mode,
        render_config.show_grid,
        scene_bounds,
        selected_idx,
        shading_mode_val,
        brush_pos,
        cross_section,
        effective_ambient,
        scene_light_info,
        scene_lights_flat,
        scene_light_vol,
    );

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback {
            render_uniform,
            display_viewport: viewport,
            render_scale,
            outline_color: render_config.outline_color,
            outline_width: render_config.outline_thickness,
            bloom_params: if render_config.bloom_enabled {
                [
                    render_config.bloom_threshold,
                    render_config.bloom_intensity,
                    render_config.bloom_radius,
                    1.0,
                ]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            },
        },
    ));

    // --- Symmetry plane overlay ---
    if let Some(axis) = sculpt_state.symmetry_axis() {
        draw_symmetry_plane(ui.painter(), camera, rect, axis, scene_bounds);
    }

    // --- Node labels overlay ---
    if render_config.show_node_labels {
        draw_node_labels(ui.painter(), camera, scene, *selected, rect);
    }

    // --- Light gizmo overlay ---
    let light_gizmo_result = if render_config.show_light_gizmos {
        let mouse_pos = ui.input(|i| i.pointer.hover_pos());
        let mouse_clicked = response.clicked();
        crate::ui::light_gizmo::draw_and_interact(
            ui.painter(),
            camera,
            scene,
            *selected,
            rect,
            mouse_pos,
            mouse_clicked,
            active_light_ids,
        )
    } else {
        crate::ui::light_gizmo::LightGizmoResult {
            clicked_transform_id: None,
        }
    };

    // --- Bounding box overlay ---
    if render_config.show_bounding_box {
        if let Some(sel_id) = *selected {
            draw_bounding_box(ui.painter(), camera, scene, sel_id, rect);
        }
    }

    // --- Gizmo overlay (drawn on top of WGPU content) ---

    let gizmo_consumed = gizmo::draw_and_interact(
        ui.painter(),
        &response,
        camera,
        scene,
        *selected,
        gizmo_state,
        gizmo_mode,
        gizmo_space,
        pivot_offset,
        rect,
        snap_config,
        gizmo_visible,
    );

    // --- Interaction priority: sculpt > gizmo > pick > orbit ---

    if sculpt_active {
        // Sculpt mode: same navigation as select mode, plus sculpt when over mesh.
        // CPU-side bounding sphere test gives instant orbit-vs-sculpt — zero GPU latency.
        if !gizmo_consumed && response.dragged_by(egui::PointerButton::Primary) && !multi_touch_active {
            let drag_origin = ui.input(|i| i.pointer.press_origin());
            let in_border = drag_origin
                .map(|origin| in_safety_border(origin, rect, render_config.sculpt_safety_border))
                .unwrap_or(false);

            // CPU bounding sphere test: project sculpt node bounds to screen,
            // check if drag origin is inside. Instant, no async pipeline needed.
            let cursor_on_mesh = if !in_border {
                drag_origin
                    .map(|origin| cursor_in_sculpt_bounds(origin, sculpt_state, scene, camera, rect))
                    .unwrap_or(false)
            } else {
                false
            };

            // Once a stroke is confirmed via GPU pick, keep sculpting even if
            // the cursor drifts outside the bounding sphere mid-stroke.
            let stroke_confirmed = last_sculpt_hit.is_some();

            if !cursor_on_mesh && !stroke_confirmed {
                // Outside mesh bounds and no active stroke: orbit (same as select mode)
                let delta = response.drag_delta();
                let modifiers = ui.input(|i| i.modifiers);
                if modifiers.ctrl && modifiers.alt {
                    let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
                    camera.roll_by(sign * delta.x, render_config.roll_sensitivity);
                } else {
                    camera.orbit(delta.x, delta.y);
                    if render_config.clamp_orbit_pitch {
                        camera.clamp_pitch();
                    }
                }
            } else if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                // Over mesh bounds or stroke already confirmed: sculpt
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    output.pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(
                            viewport,
                            time,
                            0.0,
                            false,
                            scene_bounds,
                            -1.0,
                            0.0,
                            [0.0; 4],
                            [0.0; 4],
                            0.0,
                            [0.0; 4],
                            [[0.0; 4]; 32],
                            [[0.0; 4]; 8],
                        ),
                        ctrl_held: false, // Ctrl used for sculpt inversion, not multi-select
                    });
                    // Capture modifier keys for Ctrl-invert / Shift-smooth
                    let modifiers = ui.input(|i| i.modifiers);
                    output.sculpt_ctrl_held = modifiers.ctrl;
                    output.sculpt_shift_held = modifiers.shift;
                    // Capture pen pressure from touch/stylus events
                    output.sculpt_pressure = ui.input(|i| {
                        if let Some(touch) = i.multi_touch() {
                            if touch.force > 0.0 {
                                return touch.force;
                            }
                        }
                        for event in &i.events {
                            if let egui::Event::Touch { force: Some(f), .. } = event {
                                if *f > 0.0 {
                                    return *f;
                                }
                            }
                        }
                        0.0
                    });
                }
            }
        }
        // Hover pick: when NOT dragging, submit pick for 3D brush preview position
        else if !gizmo_consumed && response.hovered() && output.pending_pick.is_none() {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    output.pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(
                            viewport,
                            time,
                            0.0,
                            false,
                            scene_bounds,
                            -1.0,
                            0.0,
                            [0.0; 4],
                            [0.0; 4],
                            0.0,
                            [0.0; 4],
                            [[0.0; 4]; 32],
                            [[0.0; 4]; 8],
                        ),
                        ctrl_held: false,
                    });
                    output.is_hover_pick = true;
                }
            }
        }

        // Ctrl+right-drag: horizontal = resize brush, vertical = adjust strength
        if response.dragged_by(egui::PointerButton::Secondary) && !multi_touch_active {
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.ctrl {
                let delta = response.drag_delta();
                // Horizontal → radius (scale by distance for consistent feel)
                let radius_sensitivity = 0.005 * camera.distance;
                output.brush_radius_delta = delta.x * radius_sensitivity;
                // Vertical → strength (inverted: drag up = stronger)
                let strength_sensitivity = 0.002;
                output.brush_strength_delta = -delta.y * strength_sensitivity;
            }
        }

        // Minimal 2D crosshair cursor (3D shader ring is the primary brush preview)
        if let SculptState::Active {
            ref brush_mode,
            ..
        } = sculpt_state
        {
            if let Some(hover_pos) = response.hover_pos() {
                let modifiers = ui.input(|i| i.modifiers);
                let effective_mode =
                    effective_brush_mode(brush_mode, modifiers.ctrl, modifiers.shift);
                let mode_color = brush_cursor_color(&effective_mode);

                // Small center dot
                ui.painter().circle_filled(
                    hover_pos,
                    2.5,
                    egui::Color32::from_white_alpha(180),
                );
                // Thin crosshair lines (6px each direction)
                let cross = 6.0;
                let stroke = egui::Stroke::new(1.0, mode_color);
                ui.painter().line_segment(
                    [
                        hover_pos - egui::vec2(cross, 0.0),
                        hover_pos + egui::vec2(cross, 0.0),
                    ],
                    stroke,
                );
                ui.painter().line_segment(
                    [
                        hover_pos - egui::vec2(0.0, cross),
                        hover_pos + egui::vec2(0.0, cross),
                    ],
                    stroke,
                );
            }
        }

        // Double-click in safety border → frame all
        if response.double_clicked() {
            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                if in_safety_border(pos, rect, render_config.sculpt_safety_border) {
                    actions.push(crate::app::actions::Action::FrameAll);
                }
            }
        }

        // Visual safety border indicator (subtle inner rect)
        if render_config.sculpt_safety_border > 0.0 {
            let border_px = rect.width().min(rect.height()) * render_config.sculpt_safety_border;
            let inner_rect = rect.shrink(border_px);
            ui.painter().rect_stroke(
                inner_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(20)),
            );
        }

        // Right-click still orbits in sculpt mode, secondary drag pans
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.ctrl && modifiers.alt {
                let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
                camera.roll_by(sign * delta.x, render_config.roll_sensitivity);
            } else {
                camera.orbit(delta.x, delta.y);
                if render_config.clamp_orbit_pitch {
                    camera.clamp_pitch();
                }
            }
        }
    } else if !gizmo_consumed {
        // Normal mode: click to pick
        if response.clicked() {
            // Light billboard click takes priority over GPU pick
            if let Some(transform_id) = light_gizmo_result.clicked_transform_id {
                actions.push(Action::Select(Some(transform_id)));
            } else if let Some(pos) = response.interact_pointer_pos() {
                let mouse_px = [
                    (pos.x - rect.min.x) * pixels_per_point,
                    (pos.y - rect.min.y) * pixels_per_point,
                ];
                let pick_uniform =
                    camera.to_uniform(viewport, time, 0.0, false, scene_bounds, -1.0, 0.0, [0.0; 4], [0.0; 4], 0.0, [0.0; 4], [[0.0; 4]; 32], [[0.0; 4]; 8]);
                let ctrl_held = ui.input(|i| i.modifiers.ctrl);
                output.pending_pick = Some(PendingPick {
                    mouse_pos: mouse_px,
                    camera_uniform: pick_uniform,
                    ctrl_held,
                });
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) && !multi_touch_active {
            let delta = response.drag_delta();
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.ctrl && modifiers.alt {
                let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
                camera.roll_by(sign * delta.x, render_config.roll_sensitivity);
            } else {
                camera.orbit(delta.x, delta.y);
                if render_config.clamp_orbit_pitch {
                    camera.clamp_pitch();
                }
            }
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }

    // --- Multi-touch: pinch-to-zoom + two-finger pan ---
    if let Some(touch) = ui.input(|i| i.multi_touch()) {
        if touch.zoom_delta != 1.0 {
            let zoom_amount = (touch.zoom_delta - 1.0) * render_config.touch_zoom_sensitivity;
            camera.zoom(zoom_amount);
        }
        if touch.translation_delta != egui::Vec2::ZERO {
            let sign = if render_config.invert_touch_pan {
                -1.0
            } else {
                1.0
            };
            camera.pan(
                sign * touch.translation_delta.x,
                sign * touch.translation_delta.y,
            );
        }
        if touch.rotation_delta != 0.0 {
            let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
            camera.roll += sign * touch.rotation_delta;
        }
    }

    // --- FPS counter overlay (top-left of viewport) ---
    if let Some((fps, frame_ms)) = fps_info {
        let text = format!("{:.0} FPS ({:.1} ms)", fps, frame_ms);
        let font = egui::FontId::monospace(11.0);
        let pos = rect.min + egui::vec2(6.0, 4.0);
        // Shadow for readability
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::LEFT_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        let color = if fps >= 55.0 {
            egui::Color32::from_rgb(120, 220, 120) // green
        } else if fps >= 30.0 {
            egui::Color32::from_rgb(220, 200, 80) // yellow
        } else {
            egui::Color32::from_rgb(220, 100, 100) // red
        };
        ui.painter()
            .text(pos, egui::Align2::LEFT_TOP, &text, font, color);
    }

    // --- Isolation mode indicator ---
    if let Some(label) = isolation_label {
        let text = format!("ISOLATED: {}", label);
        let font = egui::FontId::proportional(13.0);
        let pos = egui::pos2(rect.center().x, rect.min.y + 8.0);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::CENTER_TOP,
            &text,
            font,
            egui::Color32::from_rgb(255, 180, 60),
        );
    }

    // --- Solo light indicator ---
    if let Some(label) = solo_label {
        let text = format!("SOLO: {}", label);
        let font = egui::FontId::proportional(13.0);
        let y_offset = if isolation_label.is_some() { 24.0 } else { 8.0 };
        let pos = egui::pos2(rect.center().x, rect.min.y + y_offset);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::CENTER_TOP,
            &text,
            font,
            egui::Color32::from_rgb(255, 220, 50), // warm yellow
        );
    }

    // --- Sculpt context indicator (warm accent bar below isolation indicator) ---
    if let SculptState::Active { node_id, .. } = sculpt_state {
        let node_name = scene
            .nodes
            .get(node_id)
            .map(|n| n.name.as_str())
            .unwrap_or("Unknown");
        let text = format!("Sculpting: {}", node_name);
        let font = egui::FontId::proportional(13.0);
        let mut y_offset = 8.0;
        if isolation_label.is_some() { y_offset += 16.0; }
        if solo_label.is_some() { y_offset += 16.0; }
        let pos = egui::pos2(rect.center().x, rect.min.y + y_offset);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::CENTER_TOP,
            &text,
            font,
            egui::Color32::from_rgb(255, 160, 80),
        );
    }

    // --- Turntable indicator (top-right, below orientation gizmo area) ---
    if turntable_active {
        let text = "TURNTABLE";
        let font = egui::FontId::proportional(11.0);
        let pos = egui::pos2(rect.right() - 40.0, rect.top() + 100.0);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_TOP,
            text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::CENTER_TOP,
            text,
            font,
            egui::Color32::from_rgb(120, 200, 255),
        );
    }

    // --- Floating toolbar overlay ---
    let overlay_frame = egui::Frame::window(&ui.ctx().style())
        .fill(egui::Color32::from_rgba_premultiplied(30, 30, 38, 220));

    let toolbar_id = ui.id().with("viewport_toolbar");
    egui::Window::new(egui::RichText::new("Tools").size(11.0))
        .id(toolbar_id)
        .default_pos(rect.min + egui::vec2(8.0, 28.0))
        .resizable(true)
        .collapsible(true)
        .frame(overlay_frame)
        .show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                let select_active = *active_tool == ActiveTool::Select;
                let sculpt_tool_active = *active_tool == ActiveTool::Sculpt;

                if ui.selectable_label(select_active, "Select").clicked() && !select_active {
                    actions.push(Action::SetTool(ActiveTool::Select));
                }
                if ui.selectable_label(sculpt_tool_active, "Sculpt").clicked()
                    && !sculpt_tool_active
                {
                    actions.push(Action::SetTool(ActiveTool::Sculpt));
                }
            });

            ui.separator();
            let mode = &render_config.shading_mode;
            if ui
                .selectable_label(false, format!("Shading: {}", mode.label()))
                .on_hover_text("Click to cycle (Z key)")
                .clicked()
            {
                actions.push(Action::CycleShadingMode);
            }
        });

    // --- Shapes panel (hidden in sculpt mode) ---
    if *active_tool != ActiveTool::Sculpt {
        let shapes_id = ui.id().with("viewport_shapes");
        let overlay_frame = egui::Frame::window(&ui.ctx().style())
            .fill(egui::Color32::from_rgba_premultiplied(30, 30, 38, 220));

        egui::Window::new(egui::RichText::new("Shapes").size(11.0))
            .id(shapes_id)
            .default_pos(rect.min + egui::vec2(8.0, 130.0))
            .resizable(true)
            .collapsible(true)
            .frame(overlay_frame)
            .show(ui.ctx(), |ui| {
                let btn_size = egui::vec2(72.0, 22.0);

                // Primitives — flow layout wraps based on window width
                ui.horizontal_wrapped(|ui| {
                    for prim in SdfPrimitive::ALL {
                        if ui.add(egui::Button::new(prim.base_name()).min_size(btn_size)).clicked() {
                            actions.push(Action::CreatePrimitive(prim.clone()));
                        }
                    }
                });

                ui.separator();

                // Boolean operations
                ui.label(egui::RichText::new("Boolean").size(11.0).weak());
                ui.horizontal_wrapped(|ui| {
                    for op in CsgOp::ALL {
                        if ui.add(egui::Button::new(op.base_name()).min_size(btn_size)).clicked() {
                            let tops = scene.top_level_nodes();
                            if tops.len() >= 2 {
                                let left = Some(tops[tops.len() - 2]);
                                let right = Some(tops[tops.len() - 1]);
                                actions.push(Action::CreateOperation {
                                    op: op.clone(),
                                    left,
                                    right,
                                });
                            }
                        }
                    }
                });

                // Modifiers (needs selection)
                if let Some(sel_id) = *selected {
                    ui.separator();
                    ui.label(egui::RichText::new("Modify").size(11.0).weak());
                    ui.horizontal_wrapped(|ui| {
                        for kind in ModifierKind::ALL {
                            if ui.add(egui::Button::new(kind.base_name()).min_size(btn_size)).clicked() {
                                actions.push(Action::InsertModifierAbove {
                                    target: sel_id,
                                    kind: kind.clone(),
                                });
                            }
                        }
                    });
                }
            });
    }

    // --- Orientation Gizmo (top-right corner, interactive) ---
    {
        let gizmo_size = 60.0_f32;
        let gizmo_center = egui::pos2(
            rect.right() - gizmo_size * 0.6,
            rect.top() + gizmo_size * 0.6 + 4.0,
        );
        let arm_len = gizmo_size * 0.35;
        let hit_radius = 12.0_f32;

        let view = camera.view_matrix();
        let project_axis = |axis: glam::Vec3| -> egui::Vec2 {
            let v = view.transform_vector3(axis);
            egui::vec2(v.x, -v.y) * arm_len
        };

        // Axis definitions: positive and negative directions with their view actions
        let axes: [(glam::Vec3, egui::Color32, &str, Action, Action); 3] = [
            (
                glam::Vec3::X,
                egui::Color32::from_rgb(220, 60, 60),
                "X",
                Action::CameraRight,
                Action::CameraLeft,
            ),
            (
                glam::Vec3::Y,
                egui::Color32::from_rgb(60, 200, 60),
                "Y",
                Action::CameraTop,
                Action::CameraBottom,
            ),
            (
                glam::Vec3::Z,
                egui::Color32::from_rgb(60, 100, 220),
                "Z",
                Action::CameraFront,
                Action::CameraBack,
            ),
        ];

        let mut sorted_axes: Vec<_> = axes
            .iter()
            .map(|(axis, color, label, pos_action, neg_action)| {
                let v = view.transform_vector3(*axis);
                (v.z, *axis, *color, *label, pos_action, neg_action)
            })
            .collect();
        sorted_axes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        ui.painter().circle_filled(
            gizmo_center,
            arm_len + 8.0,
            egui::Color32::from_rgba_premultiplied(20, 20, 25, 180),
        );

        // Check for clicks on axis endpoints
        let click_pos = if response.clicked() {
            ui.input(|i| i.pointer.interact_pos())
        } else {
            None
        };
        let hover_pos = ui.input(|i| i.pointer.hover_pos());

        for (depth, axis, color, label, pos_action, neg_action) in &sorted_axes {
            let pos_end = gizmo_center + project_axis(*axis);
            let neg_end = gizmo_center + project_axis(-*axis);

            // Check hover for highlighting
            let pos_hovered = hover_pos.is_some_and(|p| p.distance(pos_end) < hit_radius);
            let neg_hovered = hover_pos.is_some_and(|p| p.distance(neg_end) < hit_radius);

            let pos_alpha = if *depth > 0.0 { 255u8 } else { 80u8 };
            let neg_alpha = if *depth < 0.0 { 255u8 } else { 80u8 };

            let pos_color = if pos_hovered {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), pos_alpha)
            };
            let neg_color = if neg_hovered {
                egui::Color32::from_rgba_unmultiplied(200, 200, 200, 180)
            } else {
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), neg_alpha)
            };

            let pos_thickness = if pos_hovered {
                3.0
            } else if *depth > 0.0 {
                2.0
            } else {
                1.0
            };

            ui.painter().line_segment(
                [gizmo_center, pos_end],
                egui::Stroke::new(pos_thickness, pos_color),
            );
            if *depth > -0.3 {
                ui.painter().text(
                    pos_end,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(if pos_hovered { 12.0 } else { 10.0 }),
                    pos_color,
                );
            }
            // Small dot for negative axis endpoint (clickable)
            let neg_radius = if neg_hovered { 4.0 } else { 3.0 };
            ui.painter().circle_filled(neg_end, neg_radius, neg_color);

            // Handle clicks on axis endpoints
            if let Some(cp) = click_pos {
                if cp.distance(pos_end) < hit_radius {
                    actions.push(match pos_action {
                        Action::CameraFront => Action::CameraFront,
                        Action::CameraTop => Action::CameraTop,
                        Action::CameraRight => Action::CameraRight,
                        Action::CameraBack => Action::CameraBack,
                        Action::CameraLeft => Action::CameraLeft,
                        Action::CameraBottom => Action::CameraBottom,
                        _ => Action::CameraFront,
                    });
                } else if cp.distance(neg_end) < hit_radius {
                    actions.push(match neg_action {
                        Action::CameraFront => Action::CameraFront,
                        Action::CameraTop => Action::CameraTop,
                        Action::CameraRight => Action::CameraRight,
                        Action::CameraBack => Action::CameraBack,
                        Action::CameraLeft => Action::CameraLeft,
                        Action::CameraBottom => Action::CameraBottom,
                        _ => Action::CameraFront,
                    });
                }
            }
        }

        // Ortho/Persp label below the gizmo
        let label_pos = egui::pos2(gizmo_center.x, gizmo_center.y + arm_len + 14.0);
        let proj_label = if camera.orthographic {
            "Ortho"
        } else {
            "Persp"
        };
        ui.painter().text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            proj_label,
            egui::FontId::proportional(9.0),
            egui::Color32::from_rgb(160, 160, 170),
        );
    }

    output
}

fn draw_node_labels(
    painter: &egui::Painter,
    camera: &Camera,
    scene: &Scene,
    selected: Option<NodeId>,
    rect: egui::Rect,
) {
    let view = camera.view_matrix();
    let aspect = rect.width() / rect.height().max(1.0);
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;
    let cam_pos = camera.eye();

    for (&id, node) in &scene.nodes {
        // Only show labels for geometry nodes with position
        let pos = match &node.data {
            NodeData::Primitive { position, .. } => *position,
            NodeData::Sculpt { position, .. } => *position,
            _ => continue,
        };

        // Skip hidden nodes
        if scene.is_hidden(id) {
            continue;
        }

        let dist = (pos - cam_pos).length();

        // Project to screen
        let Some(screen_pos) = gizmo::world_to_screen(pos, &vp, rect) else {
            continue;
        };

        // Skip if outside viewport
        if !rect.contains(screen_pos) {
            continue;
        }

        // Font size scales with distance
        let font_size: f32 = (14.0_f32 / (dist * 0.3 + 1.0)).clamp(8.0, 14.0);
        // Alpha fades with distance
        let alpha: f32 = ((10.0_f32 - dist) / 8.0).clamp(0.3, 0.9);

        let is_sel = selected == Some(id);
        let base_color = if is_sel {
            egui::Color32::from_rgb(255, 200, 60)
        } else {
            egui::Color32::from_rgb(200, 200, 210)
        };
        let color = base_color.gamma_multiply(alpha);
        let shadow_color = egui::Color32::from_rgba_premultiplied(0, 0, 0, (alpha * 180.0) as u8);

        let label_pos = egui::pos2(screen_pos.x, screen_pos.y - font_size - 4.0);
        let font = egui::FontId::proportional(font_size);

        // Text shadow
        painter.text(
            label_pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_CENTER,
            &node.name,
            font.clone(),
            shadow_color,
        );
        // Text
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            &node.name,
            font,
            color,
        );
    }
}

fn draw_bounding_box(
    painter: &egui::Painter,
    camera: &Camera,
    scene: &Scene,
    sel_id: NodeId,
    rect: egui::Rect,
) {
    let node = match scene.nodes.get(&sel_id) {
        Some(n) => n,
        None => return,
    };
    let (center, radius) = match node.data.geometry_local_sphere() {
        Some(v) => v,
        None => return,
    };
    let parent_map = scene.build_parent_map();
    let (wc, wr) = scene.walk_transforms_sphere(center, radius, sel_id, &parent_map);

    // Build 8 corners of the AABB
    let min = glam::Vec3::new(wc[0] - wr, wc[1] - wr, wc[2] - wr);
    let max = glam::Vec3::new(wc[0] + wr, wc[1] + wr, wc[2] + wr);
    let corners = [
        glam::Vec3::new(min.x, min.y, min.z),
        glam::Vec3::new(max.x, min.y, min.z),
        glam::Vec3::new(max.x, max.y, min.z),
        glam::Vec3::new(min.x, max.y, min.z),
        glam::Vec3::new(min.x, min.y, max.z),
        glam::Vec3::new(max.x, min.y, max.z),
        glam::Vec3::new(max.x, max.y, max.z),
        glam::Vec3::new(min.x, max.y, max.z),
    ];

    let aspect = rect.width() / rect.height().max(1.0);
    let vp = camera.projection_matrix(aspect) * camera.view_matrix();

    // Project all corners to screen
    let screen: Vec<Option<egui::Pos2>> = corners
        .iter()
        .map(|&c| gizmo::world_to_screen(c, &vp, rect))
        .collect();

    // 12 edges of a box: indices into corners
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // front face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // back face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // connecting edges
    ];

    let stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_rgba_premultiplied(255, 200, 60, 140),
    );
    for (a, b) in &edges {
        if let (Some(pa), Some(pb)) = (screen[*a], screen[*b]) {
            painter.line_segment([pa, pb], stroke);
        }
    }
}
