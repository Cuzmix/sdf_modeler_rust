use crate::app::actions::{Action, ActionSink};
use crate::app::backend_frame::ViewportUiFeedback;
use crate::app::runtime::ViewportResourceHandle;
use crate::app::state::{
    MultiTransformSessionState, SculptBrushAdjustMode, SculptBrushAdjustState,
    ViewportInteractionState,
};
use crate::app::viewport_interaction::{
    run_viewport_interaction_core, PointerButtonSnapshot, ViewportInputSnapshot,
    ViewportInteractionContext,
};
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::picking::PendingPick;
use crate::graph::presented_object::resolve_presented_object;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::sculpt::{self, ActiveTool, BrushMode, SculptBrushProfile, SculptState};
use crate::settings::{
    EnvironmentBackgroundMode, GroupRotateDirection, MultiAxisOrientation, MultiPivotMode,
    SelectionBehaviorSettings, SnapConfig,
};
use crate::ui::gizmo::{self, GizmoMode, GizmoSpace, GizmoState};

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

fn selection_set_key(selected_set: &std::collections::HashSet<NodeId>) -> Vec<NodeId> {
    let mut ids: Vec<_> = selected_set.iter().copied().collect();
    ids.sort_unstable();
    ids
}

fn sync_multi_transform_session_from_scene(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &std::collections::HashSet<NodeId>,
    gizmo_space: &GizmoSpace,
    selection_behavior: &SelectionBehaviorSettings,
    session: &mut MultiTransformSessionState,
) {
    if selected_set.len() <= 1 {
        session.reset_for_selection(&[], *selection_behavior);
        return;
    }

    let selection_key = selection_set_key(selected_set);
    session.reset_for_selection(&selection_key, *selection_behavior);
    if session.baseline_selection.is_none() {
        session.baseline_selection =
            gizmo::collect_gizmo_selection(scene, selected, selected_set, selection_behavior);
    }

    let Some(baseline_selection) = session.baseline_selection.as_ref() else {
        session.position_delta = glam::Vec3::ZERO;
        session.rotation_delta_deg = glam::Vec3::ZERO;
        session.scale_factor = glam::Vec3::ONE;
        return;
    };

    if let Some(readout) = gizmo::derive_multi_transform_readout(
        scene,
        baseline_selection,
        gizmo_space,
        glam::Vec3::new(
            session.rotation_delta_deg.x.to_radians(),
            session.rotation_delta_deg.y.to_radians(),
            session.rotation_delta_deg.z.to_radians(),
        ),
    ) {
        session.position_delta = readout.position_delta;
        session.rotation_delta_deg = glam::Vec3::new(
            readout.rotation_delta_rad.x.to_degrees(),
            readout.rotation_delta_rad.y.to_degrees(),
            readout.rotation_delta_rad.z.to_degrees(),
        );
        session.scale_factor = if readout.scale_enabled {
            readout.scale_factor
        } else {
            glam::Vec3::ONE
        };
    }
}

fn draw_selection_behavior_panel(
    ui: &mut egui::Ui,
    selected: Option<NodeId>,
    selected_set: &std::collections::HashSet<NodeId>,
    active_tool: &ActiveTool,
    selection_behavior: &SelectionBehaviorSettings,
    actions: &mut ActionSink,
) {
    let mut selected_count = selected_set.len();
    if let Some(primary_selected) = selected {
        if !selected_set.contains(&primary_selected) {
            selected_count += 1;
        }
    }
    if selected_count <= 1 || *active_tool != ActiveTool::Select {
        return;
    }

    let overlay_frame = egui::Frame::window(&ui.ctx().style())
        .fill(egui::Color32::from_rgba_premultiplied(30, 30, 38, 220));
    let panel_id = ui.id().with("selection_behavior_panel");
    let mut edited = *selection_behavior;
    let mut changed = false;

    egui::Window::new(egui::RichText::new("Selection Behaviour").size(11.0))
        .id(panel_id)
        .resizable(false)
        .collapsible(false)
        .frame(overlay_frame)
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-8.0, -8.0))
        .show(ui.ctx(), |ui| {
            ui.small(format!("{selected_count} selected"));

            ui.horizontal(|ui| {
                ui.label("Axes");
                changed |= ui
                    .selectable_value(
                        &mut edited.multi_axis_orientation,
                        MultiAxisOrientation::WorldZero,
                        MultiAxisOrientation::WorldZero.label(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut edited.multi_axis_orientation,
                        MultiAxisOrientation::ActiveObject,
                        MultiAxisOrientation::ActiveObject.label(),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("Rotate");
                changed |= ui
                    .selectable_value(
                        &mut edited.group_rotate_direction,
                        GroupRotateDirection::Standard,
                        GroupRotateDirection::Standard.label(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut edited.group_rotate_direction,
                        GroupRotateDirection::Inverted,
                        GroupRotateDirection::Inverted.label(),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("Pivot");
                changed |= ui
                    .selectable_value(
                        &mut edited.multi_pivot_mode,
                        MultiPivotMode::SelectionCenter,
                        MultiPivotMode::SelectionCenter.label(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut edited.multi_pivot_mode,
                        MultiPivotMode::ActiveObject,
                        MultiPivotMode::ActiveObject.label(),
                    )
                    .changed();
            });
        });

    if changed && edited != *selection_behavior {
        actions.push(Action::SetSelectionBehavior(edited));
    }
}

// ---------------------------------------------------------------------------
// CPU-side sculpt mesh hit test (voxel raycast)
// ---------------------------------------------------------------------------

/// Standard slab-method ray-AABB intersection. Returns (t_enter, t_exit).
/// If t_enter >= t_exit, the ray misses the box.
fn ray_aabb(
    origin: glam::Vec3,
    dir: glam::Vec3,
    box_min: glam::Vec3,
    box_max: glam::Vec3,
) -> (f32, f32) {
    let inv_dir = glam::Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    (t_enter, t_exit)
}

/// Build a world-space ray from a screen-space cursor position.
fn cursor_world_ray(
    camera: &Camera,
    rect: egui::Rect,
    cursor: egui::Pos2,
) -> (glam::Vec3, glam::Vec3) {
    let aspect = (rect.width() / rect.height().max(1.0)).max(1e-5);
    let ndc_x = ((cursor.x - rect.min.x) / rect.width()) * 2.0 - 1.0;
    let ndc_y = -(((cursor.y - rect.min.y) / rect.height()) * 2.0 - 1.0);
    let inv_vp = (camera.projection_matrix(aspect) * camera.view_matrix()).inverse();
    let near = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
    let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = (far - near).normalize_or_zero();
    (near, dir)
}

/// CPU sphere-trace against the scene SDF. Returns hit point on/near the surface.
fn raycast_scene_surface(scene: &Scene, origin: glam::Vec3, dir: glam::Vec3) -> Option<glam::Vec3> {
    if dir.length_squared() <= 1e-8 {
        return None;
    }
    let mut t = 0.0_f32;
    const MAX_DIST: f32 = 200.0;
    const HIT_EPS: f32 = 0.002;
    for _ in 0..128 {
        let p = origin + dir * t;
        let d = crate::graph::voxel::evaluate_scene_sdf_at_point(scene, p);
        if d.abs() <= HIT_EPS {
            return Some(p);
        }
        t += d.abs().max(0.005);
        if t >= MAX_DIST {
            break;
        }
    }
    None
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
        NodeData::Sculpt {
            position,
            rotation,
            voxel_grid,
            input,
            ..
        } => (*position, *rotation, voxel_grid, *input),
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
    let (t_enter, t_exit) = ray_aabb(
        local_origin,
        local_dir,
        voxel_grid.bounds_min,
        voxel_grid.bounds_max,
    );
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
                displacement // No input child Ã¢â‚¬â€ treat as total SDF
            }
        } else {
            displacement // Total SDF grid Ã¢â‚¬â€ value is already the distance
        };

        if sdf <= threshold {
            return true; // Near surface Ã¢â€ â€™ sculpt
        }
        // Sphere-trace: jump by SDF distance (clamped to min step for safety)
        t += min_step.max(sdf.abs() * 0.9);
    }
    false // Ray passed through empty voxels Ã¢â€ â€™ orbit
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
        let resource_handle = callback_resources
            .get::<ViewportResourceHandle>()
            .expect("viewport resources should be registered");
        let mut resources = resource_handle.write();

        let display_w = self.display_viewport[2] as u32;
        let display_h = self.display_viewport[3] as u32;
        let render_w = ((display_w as f32) * self.render_scale).max(1.0) as u32;
        let render_h = ((display_h as f32) * self.render_scale).max(1.0) as u32;

        // Ensure offscreen texture + blit bind group are the right size
        resources.ensure_offscreen_texture(device, render_w, render_h);

        let mut render_uniform = self.render_uniform;
        render_uniform.ambient_info[2] = resources
            .environment
            .prefiltered_mip_count
            .saturating_sub(1) as f32;
        render_uniform.environment_info[2] = if resources.environment.has_hdri_background_texture()
        {
            1.0
        } else {
            0.0
        };
        if resources.environment.uses_compatibility_fallback() {
            // Mobile compatibility mode keeps diffuse ambient, but avoids the
            // baked float IBL/specular path that can hard-fail older drivers.
            render_uniform.ambient_info[1] = 0.0;
            render_uniform.background_info[0] = 0.0;
            render_uniform.environment_info[2] = 0.0;
            render_uniform.environment_info[3] = 0.0;
        }

        // Write camera uniform (viewport = render dimensions for the SDF shader)
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&render_uniform),
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
                    depth_slice: None,
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
                    pass.set_bind_group(3, &resources.environment.bind_group, &[]);
                }
            } else {
                pass.set_pipeline(&resources.pipeline);
                pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                pass.set_bind_group(2, &resources.voxel_tex_bind_group, &[]);
                pass.set_bind_group(3, &resources.environment.bind_group, &[]);
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
        let resource_handle = callback_resources
            .get::<ViewportResourceHandle>()
            .expect("viewport resources should be registered");
        let resources = resource_handle.read();
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

fn active_sculpt_detail_size(scene: &Scene, sculpt_state: &SculptState) -> Option<f32> {
    let node_id = sculpt_state.active_node()?;
    let node = scene.nodes.get(&node_id)?;
    let NodeData::Sculpt { voxel_grid, .. } = &node.data else {
        return None;
    };
    Some(voxel_grid.voxel_pitch())
}

/// Compute the effective brush mode given modifier key state.
/// Shift Ã¢â€ â€™ Smooth (overrides everything), Ctrl Ã¢â€ â€™ invert (AddÃ¢â€ â€Carve, InflateÃ¢â€ â€™Carve).
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
    pub viewport_rect: egui::Rect,
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
    /// True while the transform gizmo is actively dragging.
    pub gizmo_drag_active: bool,
}

fn apply_modal_brush_adjustment(
    profile: &mut SculptBrushProfile,
    selected_mode: BrushMode,
    adjust_mode: SculptBrushAdjustMode,
    initial_value: f32,
    delta_x: f32,
    camera_distance: f32,
) {
    match adjust_mode {
        SculptBrushAdjustMode::Radius => {
            let sensitivity = 0.005 * camera_distance.max(0.1);
            profile.radius = (initial_value + delta_x * sensitivity).clamp(0.05, 2.0);
        }
        SculptBrushAdjustMode::Strength => {
            profile.strength = initial_value + delta_x * 0.002;
            profile.clamp_strength_for_mode(selected_mode);
        }
    }
}

fn capture_egui_viewport_input(
    ui: &egui::Ui,
    response: &egui::Response,
    rect: egui::Rect,
    pixels_per_point: f32,
    now_seconds: f64,
    interaction_state: &ViewportInteractionState,
) -> ViewportInputSnapshot {
    let pointer_position_logical = ui.input(|input| {
        input
            .pointer
            .interact_pos()
            .or_else(|| input.pointer.hover_pos())
    });
    let pointer_inside = pointer_position_logical.is_some_and(|pos| rect.contains(pos));
    let pointer_position_physical = pointer_position_logical.map(|pos| {
        [
            (pos.x - rect.min.x) * pixels_per_point,
            (pos.y - rect.min.y) * pixels_per_point,
        ]
    });
    let pointer_delta_logical = ui.input(|input| input.pointer.delta());
    let primary_pressed = pointer_inside
        && ui.input(|input| input.pointer.button_pressed(egui::PointerButton::Primary));
    let primary_released = ui
        .input(|input| input.pointer.button_released(egui::PointerButton::Primary))
        && (pointer_inside || interaction_state.primary_press_origin_physical.is_some());
    let primary_down = ui.input(|input| input.pointer.button_down(egui::PointerButton::Primary))
        && (pointer_inside || interaction_state.primary_press_origin_physical.is_some());
    let secondary_down = response.dragged_by(egui::PointerButton::Secondary)
        || (pointer_inside && ui.input(|input| input.pointer.secondary_down()));
    let middle_down = response.dragged_by(egui::PointerButton::Middle)
        || (pointer_inside && ui.input(|input| input.pointer.middle_down()));

    ViewportInputSnapshot {
        viewport_size_physical: [
            (rect.width().max(1.0) * pixels_per_point) as u32,
            (rect.height().max(1.0) * pixels_per_point) as u32,
        ],
        pixels_per_point,
        now_seconds,
        pointer_inside,
        pointer_position_physical,
        pointer_delta_physical: [
            pointer_delta_logical.x * pixels_per_point,
            pointer_delta_logical.y * pixels_per_point,
        ],
        wheel_delta_logical: ui
            .input(|input| [input.smooth_scroll_delta.x, input.smooth_scroll_delta.y]),
        primary: PointerButtonSnapshot {
            down: primary_down,
            pressed: primary_pressed,
            released: primary_released,
        },
        secondary: PointerButtonSnapshot {
            down: secondary_down,
            pressed: pointer_inside
                && ui.input(|input| input.pointer.button_pressed(egui::PointerButton::Secondary)),
            released: ui.input(|input| {
                input
                    .pointer
                    .button_released(egui::PointerButton::Secondary)
            }),
        },
        middle: PointerButtonSnapshot {
            down: middle_down,
            pressed: pointer_inside
                && ui.input(|input| input.pointer.button_pressed(egui::PointerButton::Middle)),
            released: ui.input(|input| input.pointer.button_released(egui::PointerButton::Middle)),
        },
        modifiers: ui.input(|input| crate::keymap::KeyboardModifiers {
            ctrl: input.modifiers.ctrl,
            shift: input.modifiers.shift,
            alt: input.modifiers.alt,
        }),
        pressure: egui_pointer_pressure(ui),
        double_clicked: response.double_clicked(),
    }
}

fn egui_pointer_pressure(ui: &egui::Ui) -> f32 {
    ui.input(|input| {
        if let Some(touch) = input.multi_touch() {
            if touch.force > 0.0 {
                return touch.force;
            }
        }
        for event in &input.events {
            if let egui::Event::Touch {
                force: Some(force), ..
            } = event
            {
                if *force > 0.0 {
                    return *force;
                }
            }
        }
        0.0
    })
}

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    selected_set: &std::collections::HashSet<NodeId>,
    multi_transform_session: &mut MultiTransformSessionState,
    viewport_interaction: &mut ViewportInteractionState,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    gizmo_visible: bool,
    pivot_offset: &mut glam::Vec3,
    sculpt_state: &mut SculptState,
    active_tool: &ActiveTool,
    time: f32,
    render_config: &crate::settings::RenderConfig,
    sculpt_count: usize,
    fps_info: Option<(f64, f64)>, // (fps, frame_ms)
    actions: &mut ActionSink,
    snap_config: &SnapConfig,
    selection_behavior: &SelectionBehaviorSettings,
    isolation_label: Option<&str>,
    turntable_active: bool,
    last_sculpt_hit: Option<glam::Vec3>,
    hover_world_pos: Option<glam::Vec3>,
    _cursor_over_geometry: bool,
    sculpt_brush_adjust: &mut Option<SculptBrushAdjustState>,
    active_light_ids: &std::collections::HashSet<crate::graph::scene::NodeId>,
    soloed_light: Option<NodeId>,
    solo_label: Option<&str>,
    reference_images: &crate::app::reference_images::ReferenceImageStore,
    reference_image_cache: &crate::ui::reference_image::EguiReferenceImageCache,
    show_distance_readout: &mut bool,
    measurement_mode: &mut bool,
    measurement_points: &mut Vec<glam::Vec3>,
) -> ViewportOutput {
    let rect = ui.available_rect_before_wrap();
    let mut output = ViewportOutput {
        viewport_rect: rect,
        pending_pick: None,
        sculpt_ctrl_held: false,
        sculpt_shift_held: false,
        sculpt_pressure: 0.0,
        brush_radius_delta: 0.0,
        brush_strength_delta: 0.0,
        is_hover_pick: false,
        gizmo_drag_active: false,
    };
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
    let touch_active = ui.input(|i| i.any_touches());
    let multi_touch_active = ui.input(|i| i.multi_touch()).is_some();
    let sculpt_active = sculpt_state.is_active();

    if sculpt_active
        && !touch_active
        && response.hovered()
        && !ui.ctx().wants_keyboard_input()
        && ui.input(|i| i.key_pressed(egui::Key::F))
    {
        let mode = if ui.input(|i| i.modifiers.shift) {
            SculptBrushAdjustMode::Strength
        } else {
            SculptBrushAdjustMode::Radius
        };
        let anchor = response.hover_pos().unwrap_or(rect.center());
        let initial_value = match mode {
            SculptBrushAdjustMode::Radius => sculpt_state.selected_profile().radius,
            SculptBrushAdjustMode::Strength => sculpt_state.selected_profile().strength,
        };
        *sculpt_brush_adjust = Some(SculptBrushAdjustState {
            mode,
            anchor_pos: [anchor.x, anchor.y],
            initial_value,
        });
    }

    if let Some(adjust) = sculpt_brush_adjust.as_ref() {
        if sculpt_state.is_active() {
            let pointer_pos = response.hover_pos().unwrap_or(rect.center());
            let delta_x = pointer_pos.x - adjust.anchor_pos[0];
            let selected_mode = sculpt_state.selected_brush();
            let profile = sculpt_state.selected_profile_mut();
            apply_modal_brush_adjustment(
                profile,
                selected_mode,
                adjust.mode,
                adjust.initial_value,
                delta_x,
                camera.distance,
            );
        }

        let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape))
            || response.clicked_by(egui::PointerButton::Secondary);
        let confirm = response.clicked_by(egui::PointerButton::Primary)
            || ui.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space));

        if cancel {
            if sculpt_state.is_active() {
                let selected_mode = sculpt_state.selected_brush();
                let profile = sculpt_state.selected_profile_mut();
                match adjust.mode {
                    SculptBrushAdjustMode::Radius => profile.radius = adjust.initial_value,
                    SculptBrushAdjustMode::Strength => {
                        profile.strength = adjust.initial_value;
                        profile.clamp_strength_for_mode(selected_mode);
                    }
                }
            }
            *sculpt_brush_adjust = None;
        } else if confirm {
            *sculpt_brush_adjust = None;
        } else {
            ui.ctx().request_repaint();
        }
    }
    let sculpt_adjust_active = sculpt_brush_adjust.is_some();

    // --- Paint the SDF viewport (WGPU callback) ---
    let pixels_per_point = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.width() * pixels_per_point,
        rect.height() * pixels_per_point,
    ];
    // Interaction detection
    let camera_dragging = if sculpt_adjust_active {
        false
    } else if sculpt_active {
        response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
            || (touch_active && response.dragged_by(egui::PointerButton::Primary))
    } else {
        response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    };
    let sculpt_brushing = sculpt_active
        && !touch_active
        && !sculpt_adjust_active
        && response.dragged_by(egui::PointerButton::Primary);
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
            let render_target_id = resolve_presented_object(scene, id)
                .map(|presented| presented.render_highlight_id())
                .unwrap_or(id);
            let order = scene.visible_topo_order();
            order.iter().position(|&nid| nid == render_target_id)
        })
        .map(|i| i as f32)
        .unwrap_or(-1.0);
    let shading_mode_val = render_config.shading_mode.gpu_value();

    // Compute brush_pos for 3D brush preview: [x, y, z, radius] or [0,0,0,0] when inactive.
    // Zero-latency tracking: project cursor through camera at last known surface depth.
    // GPU hover picks refine the depth each frame, but projection gives instant feedback.
    let brush_pos = if sculpt_state.is_active() {
        let preview_radius = ui.input(|i| {
            sculpt_state.preview_radius_with_modifiers(i.modifiers.ctrl, i.modifiers.shift)
        });
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
                [approx.x, approx.y, approx.z, preview_radius]
            }
            (_, Some(hit)) => [hit.x, hit.y, hit.z, preview_radius],
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
    let volumetric_count = scene_light_list
        .iter()
        .filter(|l| l.volumetric[0] > 0.5)
        .count() as f32;
    let volumetric_steps = render_config.volumetric_steps as f32;
    let scene_light_info = [
        scene_light_count as f32,
        volumetric_count,
        volumetric_steps,
        0.0,
    ];
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
    let ambient_luminance = scene_ambient
        .color
        .dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    let effective_ambient = if ambient_luminance > 0.0 {
        ambient_luminance
    } else {
        render_config.ambient
    };
    let ambient_info = [
        effective_ambient,
        render_config.environment_specular_intensity(),
        0.0,
        render_config.ao_mode.gpu_value(),
    ];
    let background_info = [
        match render_config.environment_background_mode {
            EnvironmentBackgroundMode::Procedural => 0.0,
            EnvironmentBackgroundMode::Environment => 1.0,
        },
        render_config.environment_background_blur.clamp(0.0, 1.0),
        match render_config.background_mode {
            crate::settings::BackgroundMode::SkyGradient => 0.0,
            crate::settings::BackgroundMode::SolidColor => 1.0,
        },
        0.0,
    ];
    let background_secondary = [
        render_config.sky_zenith[0],
        render_config.sky_zenith[1],
        render_config.sky_zenith[2],
        0.0,
    ];
    let environment_flags = (if render_config.specular_aa_enabled {
        1_u32
    } else {
        0_u32
    }) | render_config.local_reflection_mode.flag_bit();
    let environment_info = [
        render_config.environment_rotation_degrees.to_radians(),
        render_config.environment_exposure.exp2(),
        0.0,
        environment_flags as f32,
    ];

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
        ambient_info,
        background_info,
        [
            render_config.sky_horizon[0],
            render_config.sky_horizon[1],
            render_config.sky_horizon[2],
            0.0,
        ],
        background_secondary,
        [
            render_config.bg_solid_color[0],
            render_config.bg_solid_color[1],
            render_config.bg_solid_color[2],
            0.0,
        ],
        environment_info,
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

    // --- Reference image overlay ---
    crate::ui::reference_image::draw_overlay(
        ui.painter(),
        camera,
        rect,
        reference_images,
        reference_image_cache,
    );

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
        selected_set,
        gizmo_state,
        gizmo_mode,
        gizmo_space,
        pivot_offset,
        rect,
        snap_config,
        selection_behavior,
        gizmo_visible,
    );
    output.gizmo_drag_active = !matches!(gizmo_state, GizmoState::Idle);
    sync_multi_transform_session_from_scene(
        scene,
        *selected,
        selected_set,
        gizmo_space,
        selection_behavior,
        multi_transform_session,
    );

    draw_selection_behavior_panel(
        ui,
        *selected,
        selected_set,
        active_tool,
        selection_behavior,
        actions,
    );

    let mut measurement_click_consumed = false;
    if !sculpt_active && !gizmo_consumed && response.clicked() && *measurement_mode {
        if let Some(pos) = response.interact_pointer_pos() {
            let (ray_origin, ray_dir) = cursor_world_ray(camera, rect, pos);
            if let Some(hit) = raycast_scene_surface(scene, ray_origin, ray_dir) {
                if measurement_points.len() >= 2 {
                    measurement_points.clear();
                }
                measurement_points.push(hit);
                measurement_click_consumed = true;
            }
        }
    }

    let light_gizmo_click_consumed =
        !sculpt_active && !gizmo_consumed && response.clicked() && !measurement_click_consumed;
    if light_gizmo_click_consumed {
        if let Some(transform_id) = light_gizmo_result.clicked_transform_id {
            actions.push(Action::Select(Some(transform_id)));
        }
    }

    let mut interaction_feedback = ViewportUiFeedback::default();
    if !gizmo_consumed && !sculpt_adjust_active && !touch_active && !multi_touch_active {
        let viewport_input = capture_egui_viewport_input(
            ui,
            &response,
            rect,
            pixels_per_point,
            time as f64,
            viewport_interaction,
        );
        interaction_feedback = run_viewport_interaction_core(
            ViewportInteractionContext {
                state: viewport_interaction,
                camera,
                scene,
                sculpt_state,
                last_sculpt_hit,
                render_config,
                allow_selection_pick: !(measurement_click_consumed
                    || light_gizmo_click_consumed
                        && light_gizmo_result.clicked_transform_id.is_some()),
            },
            &viewport_input,
            actions,
        );
    } else if ui.input(|input| input.pointer.button_released(egui::PointerButton::Primary)) {
        viewport_interaction.primary_drag_mode = crate::app::state::ViewportPrimaryDragMode::None;
        viewport_interaction.primary_drag_distance = 0.0;
        viewport_interaction.primary_press_origin_physical = None;
    }

    output.pending_pick = interaction_feedback.pending_pick;
    output.sculpt_ctrl_held = interaction_feedback.sculpt_ctrl_held;
    output.sculpt_shift_held = interaction_feedback.sculpt_shift_held;
    output.sculpt_pressure = interaction_feedback.sculpt_pressure;
    output.brush_radius_delta = interaction_feedback.brush_radius_delta;
    output.brush_strength_delta = interaction_feedback.brush_strength_delta;
    output.is_hover_pick = interaction_feedback.is_hover_pick;

    if sculpt_state.is_active() {
        if let Some(hover_pos) = response.hover_pos() {
            let modifiers = ui.input(|i| i.modifiers);
            let effective_mode = sculpt_state.effective_brush_mode(modifiers.ctrl, modifiers.shift);
            let mode_color = brush_cursor_color(&effective_mode);

            ui.painter()
                .circle_stroke(hover_pos, 8.0, egui::Stroke::new(1.0, mode_color));
            ui.painter()
                .circle_filled(hover_pos, 3.0, egui::Color32::from_white_alpha(220));
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

    if sculpt_active && render_config.sculpt_safety_border > 0.0 {
        let border_px = rect.width().min(rect.height()) * render_config.sculpt_safety_border;
        let inner_rect = rect.shrink(border_px);
        ui.painter().rect_stroke(
            inner_rect,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(20)),
            egui::StrokeKind::Outside,
        );
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

    if sculpt_state.is_active() {
        let modifiers = ui.input(|i| i.modifiers);
        let effective_mode = sculpt_state.effective_brush_mode(modifiers.ctrl, modifiers.shift);
        let preview_radius =
            sculpt_state.preview_radius_with_modifiers(modifiers.ctrl, modifiers.shift);
        let effective_strength = sculpt_state.profile(effective_mode).strength;
        let detail_size = active_sculpt_detail_size(scene, sculpt_state).unwrap_or(0.0);
        let detail_state = sculpt_state.detail_state();
        let symmetry = sculpt_state
            .symmetry_axis()
            .map(|axis| match axis {
                0 => "X",
                1 => "Y",
                _ => "Z",
            })
            .unwrap_or("Off");
        let label = match effective_mode {
            BrushMode::Add => "Add",
            BrushMode::Carve => "Carve",
            BrushMode::Smooth => "Smooth",
            BrushMode::Flatten => "Flatten",
            BrushMode::Inflate => "Inflate",
            BrushMode::Grab => "Grab",
        };
        let detail_status = if detail_state.detail_limited_after_growth {
            "  Detail limited"
        } else if detail_state.last_pre_expand_detail_size.is_some() {
            "  Detail coarser"
        } else {
            ""
        };
        let text = if let Some(adjust) = sculpt_brush_adjust.as_ref() {
            match adjust.mode {
                SculptBrushAdjustMode::Radius => {
                    format!(
                        "Adjust Radius  {:.2}  LMB/Enter confirm  Esc/RMB cancel",
                        preview_radius
                    )
                }
                SculptBrushAdjustMode::Strength => format!(
                    "Adjust Strength  {:.3}  LMB/Enter confirm  Esc/RMB cancel",
                    sculpt_state.selected_profile().strength
                ),
            }
        } else {
            format!(
                "{label}  Radius {:.2}  Strength {:.3}  Detail {:.4}{detail_status}  Sym {}",
                preview_radius, effective_strength, detail_size, symmetry
            )
        };
        let font = egui::FontId::monospace(11.0);
        let pos = egui::pos2(rect.min.x + 6.0, rect.min.y + 22.0);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::LEFT_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::LEFT_TOP,
            &text,
            font,
            egui::Color32::from_rgb(220, 220, 220),
        );
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
        if isolation_label.is_some() {
            y_offset += 16.0;
        }
        if solo_label.is_some() {
            y_offset += 16.0;
        }
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

    // --- SDF distance readout near cursor ---
    if *show_distance_readout {
        if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
            if rect.contains(cursor) {
                let (ray_origin, ray_dir) = cursor_world_ray(camera, rect, cursor);
                let hit = raycast_scene_surface(scene, ray_origin, ray_dir);
                let text = if hit.is_some() {
                    "Surface".to_string()
                } else {
                    let sample_point = ray_origin + ray_dir * 2.0;
                    let d = crate::graph::voxel::evaluate_scene_sdf_at_point(scene, sample_point);
                    format!("D: {:.3}", d)
                };
                let pos = cursor + egui::vec2(12.0, 12.0);
                let font = egui::FontId::monospace(11.0);
                ui.painter().text(
                    pos + egui::vec2(1.0, 1.0),
                    egui::Align2::LEFT_TOP,
                    &text,
                    font.clone(),
                    egui::Color32::from_black_alpha(180),
                );
                ui.painter().text(
                    pos,
                    egui::Align2::LEFT_TOP,
                    &text,
                    font,
                    egui::Color32::from_rgb(210, 220, 235),
                );
            }
        }
    }

    // --- Measurement overlay ---
    if *measurement_mode {
        let vp = camera.projection_matrix((rect.width() / rect.height().max(1.0)).max(1e-5))
            * camera.view_matrix();
        if measurement_points.len() == 1 {
            let text = "Measure: click second point";
            let pos = egui::pos2(rect.center().x, rect.max.y - 24.0);
            let font = egui::FontId::proportional(12.0);
            ui.painter().text(
                pos + egui::vec2(1.0, 1.0),
                egui::Align2::CENTER_CENTER,
                text,
                font.clone(),
                egui::Color32::from_black_alpha(180),
            );
            ui.painter().text(
                pos,
                egui::Align2::CENTER_CENTER,
                text,
                font,
                egui::Color32::from_rgb(255, 220, 120),
            );
        } else if measurement_points.len() >= 2 {
            let a = measurement_points[0];
            let b = measurement_points[1];
            let a_screen = gizmo::world_to_screen(a, &vp, rect);
            let b_screen = gizmo::world_to_screen(b, &vp, rect);
            if let (Some(a_screen), Some(b_screen)) = (a_screen, b_screen) {
                ui.painter().line_segment(
                    [a_screen, b_screen],
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 90)),
                );
                ui.painter()
                    .circle_filled(a_screen, 4.0, egui::Color32::from_rgb(255, 220, 90));
                ui.painter()
                    .circle_filled(b_screen, 4.0, egui::Color32::from_rgb(255, 220, 90));

                let dist = a.distance(b);
                let mid = egui::pos2(
                    (a_screen.x + b_screen.x) * 0.5,
                    (a_screen.y + b_screen.y) * 0.5,
                );
                let label = format!("{:.3} units", dist);
                let font = egui::FontId::monospace(11.0);
                ui.painter().text(
                    mid + egui::vec2(1.0, 1.0),
                    egui::Align2::CENTER_BOTTOM,
                    &label,
                    font.clone(),
                    egui::Color32::from_black_alpha(180),
                );
                ui.painter().text(
                    mid,
                    egui::Align2::CENTER_BOTTOM,
                    &label,
                    font,
                    egui::Color32::from_rgb(255, 235, 140),
                );
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sculpt::{BrushMode, SculptBrushProfile};

    #[test]
    fn modal_radius_adjustment_updates_radius_with_camera_scale() {
        let mut profile = SculptBrushProfile::default_for_mode(BrushMode::Add);
        apply_modal_brush_adjustment(
            &mut profile,
            BrushMode::Add,
            SculptBrushAdjustMode::Radius,
            0.5,
            20.0,
            4.0,
        );
        assert!((profile.radius - 0.9).abs() < 1e-5);
    }

    #[test]
    fn modal_strength_adjustment_clamps_to_brush_limits() {
        let mut profile = SculptBrushProfile::default_for_mode(BrushMode::Grab);
        apply_modal_brush_adjustment(
            &mut profile,
            BrushMode::Grab,
            SculptBrushAdjustMode::Strength,
            1.0,
            2000.0,
            1.0,
        );
        assert_eq!(profile.strength, 3.0);
    }
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
