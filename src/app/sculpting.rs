use glam::{Mat4, Vec3, Vec4};

use crate::gpu::picking::{PendingPick, PickResult};
use crate::graph::presented_object::{resolve_presented_object, PresentedObjectKind};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::sculpt::{self, ActiveTool, BrushMode, FalloffMode, SculptState};
use crate::viewport::{BrushDispatch, BrushGpuParams};

use super::{state::SculptRuntimeCache, PickRayInputs, PickState, SdfApp};

fn resolve_sculpt_target_for_selection(scene: &Scene, selected: Option<NodeId>) -> Option<NodeId> {
    let selected_id = selected?;
    if let Some(presented) = resolve_presented_object(scene, selected_id) {
        if let Some(sculpt_id) = presented.attached_sculpt_id {
            return Some(sculpt_id);
        }
        if matches!(presented.kind, PresentedObjectKind::Voxel) {
            return Some(presented.host_id);
        }
    }

    scene
        .nodes
        .get(&selected_id)
        .and_then(|node| matches!(node.data, NodeData::Sculpt { .. }).then_some(selected_id))
}

impl SdfApp {
    // -----------------------------------------------------------------------
    // Async sculpt pick (1-frame delay, eliminates GPU stall)
    // -----------------------------------------------------------------------

    /// Build a world ray from submitted pick data.
    fn pick_ray_from_pending(ray_inputs: PickRayInputs) -> Option<(Vec3, Vec3)> {
        let viewport_w = ray_inputs.viewport_size[0].max(1.0);
        let viewport_h = ray_inputs.viewport_size[1].max(1.0);
        let uv_x = ray_inputs.mouse_pos[0] / viewport_w * 2.0 - 1.0;
        let uv_y = ray_inputs.mouse_pos[1] / viewport_h * 2.0 - 1.0;

        let ndc = Vec4::new(uv_x, -uv_y, 1.0, 1.0);
        let inv_vp = Mat4::from_cols_array(&ray_inputs.inv_view_proj);
        let world = inv_vp * ndc;
        if world.w.abs() <= 1e-6 {
            return None;
        }

        let eye = Vec3::new(ray_inputs.eye[0], ray_inputs.eye[1], ray_inputs.eye[2]);
        let world_pos = world.truncate() / world.w;
        let ray_dir = (world_pos - eye).normalize_or_zero();
        if ray_dir.length_squared() <= 1e-8 {
            return None;
        }

        Some((eye, ray_dir))
    }

    /// Convert a pending pick payload into lightweight ray inputs.
    fn ray_inputs_from_pending(pending: &PendingPick) -> PickRayInputs {
        PickRayInputs {
            mouse_pos: pending.mouse_pos,
            inv_view_proj: pending.camera_uniform.inv_view_proj,
            eye: [
                pending.camera_uniform.eye[0],
                pending.camera_uniform.eye[1],
                pending.camera_uniform.eye[2],
            ],
            viewport_size: [
                pending.camera_uniform.viewport[2],
                pending.camera_uniform.viewport[3],
            ],
        }
    }
    pub(super) fn clear_sculpt_runtime_cache(&mut self) {
        self.async_state.sculpt_runtime_cache = None;
    }

    fn sculpt_stroke_label(
        base_mode: &BrushMode,
        ctrl_held: bool,
        shift_held: bool,
    ) -> &'static str {
        let effective_mode = if shift_held {
            BrushMode::Smooth
        } else if ctrl_held {
            match base_mode {
                BrushMode::Add => BrushMode::Carve,
                BrushMode::Carve => BrushMode::Add,
                BrushMode::Inflate => BrushMode::Carve,
                other => *other,
            }
        } else {
            *base_mode
        };
        match effective_mode {
            BrushMode::Add => "Sculpt Add",
            BrushMode::Carve => "Sculpt Carve",
            BrushMode::Smooth => "Sculpt Smooth",
            BrushMode::Flatten => "Sculpt Flatten",
            BrushMode::Inflate => "Sculpt Inflate",
            BrushMode::Grab => "Sculpt Grab",
        }
    }

    fn world_to_grid_from_bounds(
        local_pos: Vec3,
        bounds_min: Vec3,
        bounds_max: Vec3,
        resolution: u32,
    ) -> Vec3 {
        let extent = (bounds_max - bounds_min).max(Vec3::splat(1e-6));
        let norm = (local_pos - bounds_min) / extent;
        norm * (resolution.saturating_sub(1)) as f32
    }

    fn sculpt_runtime_cache(&mut self, node_id: NodeId) -> Option<SculptRuntimeCache> {
        let structure_key = self.doc.scene.structure_key();
        if let Some(cache) = self.async_state.sculpt_runtime_cache {
            if cache.node_id == node_id && cache.structure_key == structure_key {
                return Some(cache);
            }
        }

        let node = self.doc.scene.nodes.get(&node_id)?;
        let NodeData::Sculpt {
            position,
            rotation,
            ref voxel_grid,
            ..
        } = node.data
        else {
            return None;
        };

        let material_id = self
            .doc
            .scene
            .visible_topo_order()
            .iter()
            .position(|&id| id == node_id)
            .map(|i| i as i32)?;

        let cache = SculptRuntimeCache {
            node_id,
            structure_key,
            material_id,
            position,
            rotation,
            gpu_offset: self.gpu.voxel_gpu_offsets.get(&node_id).copied(),
            grid_resolution: voxel_grid.resolution,
            bounds_min: voxel_grid.bounds_min,
            bounds_max: voxel_grid.bounds_max,
        };
        self.async_state.sculpt_runtime_cache = Some(cache);
        Some(cache)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_gpu_brush_dispatch_from_local(
        cache: SculptRuntimeCache,
        local_hit: Vec3,
        local_view_dir: Vec3,
        brush_mode: &BrushMode,
        radius: f32,
        strength: f32,
        falloff_mode: &FalloffMode,
        flatten_ref: f32,
        surface_constraint: f32,
    ) -> Option<BrushDispatch> {
        let gpu_offset = cache.gpu_offset?;
        let res = cache.grid_resolution;
        if res == 0 {
            return None;
        }

        let brush_min = local_hit - Vec3::splat(radius);
        let brush_max = local_hit + Vec3::splat(radius);
        let g_min =
            Self::world_to_grid_from_bounds(brush_min, cache.bounds_min, cache.bounds_max, res);
        let g_max =
            Self::world_to_grid_from_bounds(brush_max, cache.bounds_min, cache.bounds_max, res);

        let x0 = (g_min.x.floor().max(0.0) as u32).min(res - 1);
        let y0 = (g_min.y.floor().max(0.0) as u32).min(res - 1);
        let z0 = (g_min.z.floor().max(0.0) as u32).min(res - 1);
        let x1 = (g_max.x.ceil().max(0.0) as u32).min(res - 1);
        let y1 = (g_max.y.ceil().max(0.0) as u32).min(res - 1);
        let z1 = (g_max.z.ceil().max(0.0) as u32).min(res - 1);

        Some(BrushDispatch {
            params: BrushGpuParams {
                center_local: local_hit.to_array(),
                radius,
                strength,
                sign_val: brush_mode.sign(),
                grid_offset: gpu_offset,
                grid_resolution: res,
                bounds_min: cache.bounds_min.to_array(),
                _pad0: 0.0,
                bounds_max: cache.bounds_max.to_array(),
                _pad1: 0.0,
                min_voxel: [x0, y0, z0],
                _pad2: 0,
                brush_mode: brush_mode.gpu_mode(),
                falloff_mode: falloff_mode.gpu_mode(),
                smooth_iterations: 0,
                flatten_ref,
                surface_constraint,
                _pad3: [0.0; 3],
                view_dir_local: local_view_dir.to_array(),
                _pad4: 0.0,
            },
            workgroups: [(x1 - x0 + 4) / 4, (y1 - y0 + 4) / 4, (z1 - z0 + 4) / 4],
        })
    }

    /// Predict and apply a sculpt hit from the latest cursor ray when async
    /// pick readback is still pending.
    pub(super) fn predict_sculpt_from_pending_pick(&mut self, pending: &PendingPick) -> bool {
        if !self.async_state.sculpt_dragging {
            return false;
        }

        let active_node = match self.doc.sculpt_state {
            SculptState::Active { node_id, .. } => node_id,
            _ => return false,
        };

        let mut anchor = match self.async_state.last_sculpt_hit {
            Some(hit) => hit,
            None => return false,
        };
        if let SculptState::Active {
            ref session,
            ref stroke_state,
            ..
        } = self.doc.sculpt_state
        {
            if session.selected_brush == BrushMode::Grab {
                if let Some(origin) = stroke_state.grab_start {
                    anchor = origin;
                }
            }
        }

        let Some(cache) = self.sculpt_runtime_cache(active_node) else {
            return false;
        };
        let material_id = cache.material_id;

        let Some((ray_origin, ray_dir)) =
            Self::pick_ray_from_pending(Self::ray_inputs_from_pending(pending))
        else {
            return false;
        };

        let eye = self.doc.camera.eye();
        let forward = (self.doc.camera.target - eye).normalize_or_zero();
        if forward.length_squared() <= 1e-8 {
            return false;
        }
        let denom = ray_dir.dot(forward);
        if denom.abs() <= 1e-6 {
            return false;
        }

        let t = (anchor - ray_origin).dot(forward) / denom;
        if !t.is_finite() || t <= 0.0 {
            return false;
        }

        let projected = ray_origin + ray_dir * t;
        self.async_state.hover_world_pos = Some(projected);
        self.async_state.cursor_over_geometry = true;
        self.handle_sculpt_hit(PickResult {
            material_id,
            distance: 0.0,
            world_pos: projected.to_array(),
        });
        true
    }

    /// Continue Grab strokes when the cursor leaves geometry by projecting the
    /// current mouse ray to the grab plane.
    fn continue_grab_on_pick_miss(&mut self, ray_inputs: PickRayInputs) -> bool {
        let (active_node, grab_origin) = match self.doc.sculpt_state {
            SculptState::Active {
                node_id,
                ref session,
                ref stroke_state,
                ..
            } if session.selected_brush == BrushMode::Grab && stroke_state.grab_start.is_some() => {
                (node_id, stroke_state.grab_start.unwrap_or(Vec3::ZERO))
            }
            _ => return false,
        };

        let Some(cache) = self.sculpt_runtime_cache(active_node) else {
            return false;
        };
        let material_id = cache.material_id;

        let Some((ray_origin, ray_dir)) = Self::pick_ray_from_pending(ray_inputs) else {
            return false;
        };

        let eye = self.doc.camera.eye();
        let forward = (self.doc.camera.target - eye).normalize_or_zero();
        if forward.length_squared() <= 1e-8 {
            return false;
        }
        let denom = ray_dir.dot(forward);
        if denom.abs() <= 1e-6 {
            return false;
        }

        let t = (grab_origin - ray_origin).dot(forward) / denom;
        let projected = ray_origin + ray_dir * t;

        self.async_state.hover_world_pos = Some(projected);
        self.async_state.cursor_over_geometry = true;
        self.handle_sculpt_hit(PickResult {
            material_id,
            distance: 0.0,
            world_pos: projected.to_array(),
        });
        true
    }

    /// Poll for a previously submitted async sculpt pick result.
    /// If ready: apply brush at the hit point (CPU + GPU).
    pub(super) fn poll_sculpt_pick(&mut self) {
        if !matches!(self.async_state.pick_state, PickState::Pending { .. }) {
            return;
        }

        // Non-blocking GPU poll to advance async map
        let _ = self.gpu.render_context.device.poll(wgpu::PollType::Poll);

        // Try to read the result
        let (ready, pending_ray_inputs, submitted_at) = {
            let res = self.gpu.viewport_resources.read();
            let PickState::Pending {
                ref receiver,
                ray_inputs,
                submitted_at,
            } = self.async_state.pick_state
            else {
                return;
            };
            (res.try_read_pick_result(receiver), ray_inputs, submitted_at)
        };

        let Some(pick_result) = ready else {
            return; // Not ready yet - try again next frame
        };
        let pick_latency_ms = submitted_at.elapsed().as_secs_f64() * 1000.0;
        self.perf
            .timings
            .record_sculpt_pick_latency(pick_latency_ms);
        self.async_state.pick_state = PickState::Idle;

        if self.async_state.sculpt_dragging {
            let live_grab_drag = matches!(
                self.doc.sculpt_state,
                SculptState::Active {
                    ref session,
                    ref stroke_state,
                    ..
                } if session.selected_brush == BrushMode::Grab && stroke_state.grab_start.is_some()
            );

            if live_grab_drag {
                // Once Grab has an anchor, drive it from the current cursor ray
                // and ignore delayed pick hits to avoid visible jitter/rubber-banding.
                if !self.continue_grab_on_pick_miss(pending_ray_inputs) {
                    self.async_state.cursor_over_geometry = false;
                    self.async_state.hover_world_pos = None;
                }
            } else {
                match pick_result {
                    Some(result) => {
                        // Drag on geometry: apply brush + update preview.
                        self.async_state.hover_world_pos = Some(Vec3::new(
                            result.world_pos[0],
                            result.world_pos[1],
                            result.world_pos[2],
                        ));
                        self.async_state.cursor_over_geometry = true;
                        self.handle_sculpt_hit(result);
                    }
                    None => {
                        // Miss while dragging: keep Grab/Move alive from cursor ray.
                        if !self.continue_grab_on_pick_miss(pending_ray_inputs) {
                            // Non-grab brushes stop when leaving geometry.
                            self.async_state.cursor_over_geometry = false;
                            self.async_state.hover_world_pos = None;
                        }
                    }
                }
            }
        } else if let Some(result) = pick_result {
            // Hover pick: only update preview position.
            self.handle_hover_pick(result);
        } else {
            // Hover miss: clear preview.
            self.async_state.hover_world_pos = None;
            self.async_state.cursor_over_geometry = false;
        }
    }

    /// Handle a hover-only pick result: update 3D brush preview position
    /// without applying any brush strokes.
    pub(super) fn handle_hover_pick(&mut self, result: PickResult) {
        if result.material_id >= 0 {
            self.async_state.hover_world_pos = Some(Vec3::new(
                result.world_pos[0],
                result.world_pos[1],
                result.world_pos[2],
            ));
            self.async_state.cursor_over_geometry = true;
        } else {
            self.async_state.hover_world_pos = None;
            self.async_state.cursor_over_geometry = false;
        }
    }

    /// Handle a sculpt pick result: apply brush with interpolation.
    pub(super) fn handle_sculpt_hit(&mut self, result: PickResult) {
        let topo_order = self.doc.scene.visible_topo_order();
        let idx = result.material_id as usize;
        if idx >= topo_order.len() {
            return;
        }
        let hit_node_id = topo_order[idx];

        // Per-stroke undo: snapshot grid data at the start of each stroke
        if self.async_state.last_sculpt_hit.is_none() {
            if let Some(active_id) = self.doc.sculpt_state.active_node() {
                let selected_brush = self.doc.sculpt_state.selected_brush();
                self.doc.history.begin_sculpt_stroke(
                    &self.doc.scene,
                    active_id,
                    self.ui.selection.selected,
                    Self::sculpt_stroke_label(
                        &selected_brush,
                        self.async_state.sculpt_ctrl_held,
                        self.async_state.sculpt_shift_held,
                    ),
                );
            }
        }
        let Some(node_id) = self.doc.sculpt_state.active_node() else {
            return;
        };
        let cached_ctx = self.sculpt_runtime_cache(node_id);
        let mut resolved_brush = self.doc.sculpt_state.resolved_brush_for_stroke(
            self.async_state.sculpt_ctrl_held,
            self.async_state.sculpt_shift_held,
        );
        if self.settings.render.pressure_sensitivity && self.async_state.sculpt_pressure > 0.0 {
            resolved_brush.profile.strength *= self.async_state.sculpt_pressure;
        }
        resolved_brush
            .profile
            .clamp_strength_for_mode(resolved_brush.mode);
        let brush_mode = resolved_brush.mode;
        let brush_radius = resolved_brush.profile.radius;
        let brush_strength = resolved_brush.profile.strength;
        let falloff_mode = resolved_brush.profile.falloff_mode;
        let brush_shape = resolved_brush.profile.brush_shape;
        let smooth_iterations = resolved_brush.profile.smooth_iterations;
        let lazy_radius = resolved_brush.profile.lazy_radius;
        let stroke_spacing = resolved_brush.profile.stroke_spacing;
        let surface_constraint = resolved_brush.profile.surface_constraint;
        let front_faces_only = resolved_brush.profile.front_faces_only;
        let symmetry_axis = self.doc.sculpt_state.symmetry_axis();

        if hit_node_id != node_id {
            // Differential sculpt: material ID is the child primitive's, not the sculpt's.
            // If the hit node is a descendant of the active sculpt, treat it as a hit
            // on the sculpt itself and fall through to apply the brush.
            let is_child_of_active_sculpt = self.doc.scene.is_descendant(hit_node_id, node_id);
            if !is_child_of_active_sculpt {
                // Hit a truly different node - handle navigation/conversion
                if let Some(hit_node) = self.doc.scene.nodes.get(&hit_node_id) {
                    if matches!(hit_node.data, NodeData::Sculpt { .. }) {
                        // Hit another sculpt node - switch to it directly
                        self.doc
                            .sculpt_state
                            .activate_preserving_session(hit_node_id, None);
                        self.ui.selection.select_single(hit_node_id);
                        self.async_state.last_sculpt_hit = None;
                        self.async_state.lazy_brush_pos = None;
                        self.clear_sculpt_runtime_cache();
                    } else {
                        // Check if the hit node has a sculpt parent
                        let parent_map = self.doc.scene.build_parent_map();
                        if let Some(sculpt_id) =
                            self.doc.scene.find_sculpt_parent(hit_node_id, &parent_map)
                        {
                            // Switch to the sculpt parent
                            self.doc
                                .sculpt_state
                                .activate_preserving_session(sculpt_id, None);
                            self.ui.selection.select_single(sculpt_id);
                            self.async_state.last_sculpt_hit = None;
                            self.async_state.lazy_brush_pos = None;
                            self.clear_sculpt_runtime_cache();
                        } else {
                            // Non-sculpt node with no sculpt parent - show convert dialog
                            self.ui.sculpt_convert_dialog =
                                Some(crate::app::state::SculptConvertDialog::new(hit_node_id));
                        }
                    }
                }
                return;
            }
            // Fall through: hit_node_id is a child of the active sculpt - apply brush normally
        }

        let hit_world = Vec3::new(
            result.world_pos[0],
            result.world_pos[1],
            result.world_pos[2],
        );
        let Some(cache) = cached_ctx else {
            return;
        };

        // Grab brush: initialize snapshot and start position on first hit.
        // Differential sculpts store displacement only; analytical SDF is sampled on demand.
        let is_grab = brush_mode == BrushMode::Grab;
        let needs_grab_snapshot = self
            .doc
            .sculpt_state
            .stroke_state()
            .is_some_and(|stroke_state| stroke_state.grab_snapshot.is_none());
        if is_grab && needs_grab_snapshot {
            let mut snapshot = None;
            let mut analytical_snapshot = None;
            let mut child_input = None;
            if let Some(node) = self.doc.scene.nodes.get(&node_id) {
                if let NodeData::Sculpt {
                    input,
                    ref voxel_grid,
                    ..
                } = node.data
                {
                    snapshot = Some(voxel_grid.data.clone().into());
                    child_input = input;
                    analytical_snapshot = input.map(|child_id| {
                        sculpt::build_analytical_snapshot(
                            voxel_grid,
                            &self.doc.scene,
                            child_id,
                            cache.position,
                            cache.rotation,
                        )
                        .into()
                    });
                }
            }
            if let Some(stroke_state) = self.doc.sculpt_state.stroke_state_mut() {
                stroke_state.grab_snapshot = snapshot;
                stroke_state.grab_child_input = child_input;
                stroke_state.grab_start = Some(hit_world);
                stroke_state.grab_analytical_snapshot = analytical_snapshot;
            }
        }
        let (grab_snap, grab_analytic_snap, grab_origin, grab_child) = self
            .doc
            .sculpt_state
            .stroke_state()
            .map(|stroke_state| {
                (
                    stroke_state.grab_snapshot.clone(),
                    stroke_state.grab_analytical_snapshot.clone(),
                    stroke_state.grab_start,
                    stroke_state.grab_child_input,
                )
            })
            .unwrap_or((None, None, None, None));

        // Lazy brush: smooth cursor with elastic dead zone
        let effective_hit = if lazy_radius > 0.0 {
            if let Some(ref mut lazy_pos) = self.async_state.lazy_brush_pos {
                let delta = hit_world - *lazy_pos;
                let dist = delta.length();
                if dist <= lazy_radius {
                    return; // Dead zone - don't apply brush
                }
                let factor = 1.0 - lazy_radius / dist;
                *lazy_pos += delta * factor;
                *lazy_pos
            } else {
                self.async_state.lazy_brush_pos = Some(hit_world);
                hit_world
            }
        } else {
            hit_world
        };

        // Capture flatten reference on first hit of a Flatten stroke
        let needs_flatten_reference = brush_mode == BrushMode::Flatten
            && self
                .doc
                .sculpt_state
                .stroke_state()
                .is_some_and(|stroke_state| stroke_state.flatten_reference.is_none());
        if needs_flatten_reference {
            let mut flatten_reference = None;
            if let Some(node) = self.doc.scene.nodes.get(&node_id) {
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    let local_hit = sculpt::inverse_rotate_euler(
                        effective_hit - cache.position,
                        cache.rotation,
                    );
                    flatten_reference = Some(voxel_grid.sample(local_hit));
                }
            }
            if let Some(stroke_state) = self.doc.sculpt_state.stroke_state_mut() {
                stroke_state.flatten_reference = flatten_reference;
            }
        }
        let flatten_ref_val = self
            .doc
            .sculpt_state
            .stroke_state()
            .and_then(|stroke_state| stroke_state.flatten_reference)
            .unwrap_or(0.0);

        // Grab brush: single application per frame, centered at grab start
        if is_grab {
            if let (Some(snap), Some(origin)) = (grab_snap.as_deref(), grab_origin) {
                // Project mouse ray onto camera-facing plane at grab depth (like Blender).
                // This gives 1:1 screen-to-world mapping regardless of surface curvature.
                let eye = self.doc.camera.eye();
                let forward = (self.doc.camera.target - eye).normalize_or_zero();
                let ray_dir = (hit_world - eye).normalize_or_zero();
                let denom = ray_dir.dot(forward);
                let grab_delta = if denom.abs() > 1e-6 {
                    let t = (origin - eye).dot(forward) / denom;
                    (eye + ray_dir * t) - origin
                } else {
                    hit_world - origin
                };

                // Center at grab start position, not current mouse
                let local_center =
                    sculpt::inverse_rotate_euler(origin - cache.position, cache.rotation);
                let local_delta = sculpt::inverse_rotate_euler(grab_delta, cache.rotation);
                let local_view_dir = if front_faces_only {
                    sculpt::inverse_rotate_euler((origin - eye).normalize_or_zero(), cache.rotation)
                        .normalize_or_zero()
                } else {
                    Vec3::ZERO
                };

                let dirty = if let Some(child_id) = grab_child {
                    // Differential sculpt: reconstruct total SDF on demand, then write displacement.
                    sculpt::apply_grab_to_grid_differential_scene(
                        &mut self.doc.scene,
                        node_id,
                        snap,
                        grab_analytic_snap.as_deref(),
                        local_center,
                        brush_radius,
                        brush_strength,
                        local_delta,
                        &falloff_mode,
                        surface_constraint,
                        local_view_dir,
                        child_id,
                        cache.position,
                        cache.rotation,
                    )
                } else if let Some(node) = self.doc.scene.nodes.get_mut(&node_id) {
                    if let NodeData::Sculpt {
                        ref mut voxel_grid, ..
                    } = node.data
                    {
                        Some(sculpt::apply_grab_to_grid(
                            voxel_grid,
                            snap,
                            local_center,
                            brush_radius,
                            brush_strength,
                            local_delta,
                            &falloff_mode,
                            surface_constraint,
                            local_view_dir,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(region) = dirty {
                    if let Some(stroke_state) = self.doc.sculpt_state.stroke_state_mut() {
                        stroke_state.pending_grab_repair_region =
                            Some(match stroke_state.pending_grab_repair_region {
                                Some(existing) => existing.merge(region),
                                None => region,
                            });
                    }
                    let (z0, z1) = region.z_range();
                    self.try_incremental_voxel_upload(node_id, z0, z1);
                    self.upload_voxel_texture_region(node_id, z0, z1);
                }
            }
            self.perf.timings.record_sculpt_brush_batch(1, 0, 0);
            self.async_state.last_sculpt_hit = Some(effective_hit);
            return;
        }

        // Brush stroke interpolation: fill gaps during fast mouse movement
        let hits = self.interpolate_brush_hits(effective_hit, brush_radius, stroke_spacing);

        // Generate mirrored hits if symmetry enabled
        let all_hits = if let Some(axis) = symmetry_axis {
            let mut mirrored = Vec::with_capacity(hits.len() * 2);
            for &h in &hits {
                mirrored.push(h);
                let mut m = h;
                match axis {
                    0 => m.x = -m.x,
                    1 => m.y = -m.y,
                    _ => m.z = -m.z,
                }
                mirrored.push(m);
            }
            mirrored
        } else {
            hits
        };

        let eye = self.doc.camera.eye();
        let mut gpu_dispatches: Vec<BrushDispatch> = Vec::new();
        let mut dirty_region: Option<sculpt::VoxelEditRegion> = None;

        {
            let Some(node) = self.doc.scene.nodes.get_mut(&node_id) else {
                return;
            };
            let NodeData::Sculpt {
                ref mut voxel_grid, ..
            } = node.data
            else {
                return;
            };

            for &pos in &all_hits {
                let view_dir_world = if front_faces_only {
                    (pos - eye).normalize_or_zero()
                } else {
                    Vec3::ZERO
                };

                let local_hit = sculpt::inverse_rotate_euler(pos - cache.position, cache.rotation);
                let local_view_dir = if front_faces_only {
                    sculpt::inverse_rotate_euler(view_dir_world, cache.rotation).normalize_or_zero()
                } else {
                    Vec3::ZERO
                };

                let region = sculpt::apply_brush_local(
                    voxel_grid,
                    local_hit,
                    local_view_dir,
                    &brush_mode,
                    brush_radius,
                    brush_strength,
                    &falloff_mode,
                    &brush_shape,
                    smooth_iterations,
                    flatten_ref_val,
                    surface_constraint,
                );

                dirty_region = Some(match dirty_region {
                    Some(existing) => existing.merge(region),
                    None => region,
                });

                if brush_mode != BrushMode::Smooth {
                    if let Some(dispatch) = Self::build_gpu_brush_dispatch_from_local(
                        cache,
                        local_hit,
                        local_view_dir,
                        &brush_mode,
                        brush_radius,
                        brush_strength,
                        &falloff_mode,
                        flatten_ref_val,
                        surface_constraint,
                    ) {
                        gpu_dispatches.push(dispatch);
                    }
                }
            }
        }

        let dispatch_count = gpu_dispatches.len() as u32;
        let mut submit_count = 0;
        if !gpu_dispatches.is_empty() {
            let mut viewport_resources = self.gpu.viewport_resources.write();
            if viewport_resources.dispatch_brush_batch(
                &self.gpu.render_context.device,
                &self.gpu.render_context.queue,
                &gpu_dispatches,
            ) {
                submit_count = 1;
            }
        }

        if let Some(region) = dirty_region {
            let repaired = sculpt::repair_sdf_region_scene(&mut self.doc.scene, node_id, region, 2)
                .unwrap_or(region);
            let (z0, z1) = repaired.z_range();
            self.try_incremental_voxel_upload(node_id, z0, z1);
            self.upload_voxel_texture_region(node_id, z0, z1);
        }
        self.perf.timings.record_sculpt_brush_batch(
            all_hits.len() as u32,
            dispatch_count,
            submit_count,
        );

        // Incremental composite volume update for the brush-affected region
        if self.settings.render.composite_volume_enabled && !all_hits.is_empty() {
            self.dispatch_composite_region(effective_hit, brush_radius);
        }

        self.async_state.last_sculpt_hit = Some(effective_hit);
    }

    /// Interpolate between last and current hit to prevent gaps.
    pub(super) fn interpolate_brush_hits(
        &self,
        current: Vec3,
        brush_radius: f32,
        stroke_spacing: f32,
    ) -> Vec<Vec3> {
        let Some(last) = self.async_state.last_sculpt_hit else {
            return vec![current];
        };
        let dist = (current - last).length();
        let spacing = stroke_spacing.clamp(0.05, 1.0);
        let mut step = (brush_radius * spacing).max(0.005);

        // Keep stroke sampling dense enough relative to voxel size to reduce
        // staircase/rubber-banding artifacts during fast drags.
        if let SculptState::Active { node_id, .. } = self.doc.sculpt_state {
            if let Some(NodeData::Sculpt { ref voxel_grid, .. }) =
                self.doc.scene.nodes.get(&node_id).map(|n| &n.data)
            {
                let extent = voxel_grid.bounds_max - voxel_grid.bounds_min;
                let denom = voxel_grid.resolution.saturating_sub(1).max(1) as f32;
                let voxel_step = (extent.max_element() / denom).max(0.001);
                step = step.min((voxel_step * 0.75).max(0.002));
            }
        }

        if dist <= step {
            return vec![current];
        }
        let n = ((dist / step).ceil() as usize).clamp(1, 96);
        let mut hits = Vec::with_capacity(n);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            hits.push(last + (current - last) * t);
        }
        hits
    }

    /// Submit an async pick for sculpt mode (non-blocking).
    pub(super) fn submit_sculpt_pick(&mut self) {
        if !self.doc.sculpt_state.is_active() {
            return;
        }
        if !matches!(self.async_state.pick_state, PickState::Idle) {
            return;
        }
        let Some(pending) = self.async_state.pending_pick.take() else {
            return;
        };
        let viewport_resources = self.gpu.viewport_resources.read();
        let rx = viewport_resources.submit_pick(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            &pending,
        );

        self.async_state.pick_state = PickState::Pending {
            receiver: rx,
            ray_inputs: Self::ray_inputs_from_pending(&pending),
            submitted_at: crate::compat::Instant::now(),
        };
    }

    /// Reset stroke interpolation and flatten/grab references when the pointer
    /// is released during sculpting (no pending pick, pick state idle).
    pub(super) fn reset_sculpt_stroke_if_idle(&mut self) {
        if self.doc.sculpt_state.is_active()
            && self.async_state.pending_pick.is_none()
            && matches!(self.async_state.pick_state, PickState::Idle)
        {
            // End stroke only if a drag was in progress
            if self.async_state.last_sculpt_hit.is_some() {
                self.doc.history.end_sculpt_stroke(&self.doc.scene);
            }

            self.async_state.last_sculpt_hit = None;
            self.async_state.lazy_brush_pos = None;
            self.clear_sculpt_runtime_cache();
            self.async_state.sculpt_dragging = false;
            // Clear cursor_over_geometry so the next LMB drag defaults to orbit
            // until a hover pick re-confirms geometry. hover_world_pos is kept
            // so the 3D brush ring stays visible during hover.
            self.async_state.cursor_over_geometry = false;
            self.finalize_pending_grab_repair();
            self.doc.sculpt_state.clear_stroke_state();
        }
    }

    pub(super) fn sync_sculpt_state(&mut self) {
        match self.doc.active_tool {
            ActiveTool::Select => {
                // Original behavior: always deactivate sculpt in Select mode
                if self.doc.sculpt_state.is_active() {
                    self.finalize_pending_grab_repair();
                    self.doc.sculpt_state.deactivate();
                    self.doc.history.discard_pending_sculpt_stroke();
                    self.async_state.last_sculpt_hit = None;
                    self.async_state.lazy_brush_pos = None;
                    self.clear_sculpt_runtime_cache();
                    self.async_state.hover_world_pos = None;
                    self.async_state.cursor_over_geometry = false;
                    self.async_state.sculpt_dragging = false;
                    self.cancel_pending_pick_state();
                }
            }
            ActiveTool::Sculpt => {
                let sel = resolve_sculpt_target_for_selection(
                    &self.doc.scene,
                    self.ui.selection.selected,
                );
                let active = self.doc.sculpt_state.active_node();
                if sel != active {
                    // Selection changed - try to activate on new node
                    if let Some(id) = sel {
                        self.finalize_pending_grab_repair();
                        self.doc
                            .sculpt_state
                            .activate_preserving_session(id, Some(self.scene_avg_extent()));
                    } else {
                        self.finalize_pending_grab_repair();
                        self.doc.sculpt_state.deactivate();
                        self.doc.history.discard_pending_sculpt_stroke();
                    }
                    self.async_state.last_sculpt_hit = None;
                    self.async_state.lazy_brush_pos = None;
                    self.clear_sculpt_runtime_cache();
                    self.cancel_pending_pick_state();
                }
            }
        }
    }

    /// Cancel any in-flight async pick, unmapping the staging buffer so
    /// subsequent `queue.submit` calls don't panic.
    pub(super) fn cancel_pending_pick_state(&mut self) {
        if matches!(self.async_state.pick_state, PickState::Pending { .. }) {
            // Wait for the pending map_async to complete before unmapping -
            // wgpu 22 panics if you unmap a buffer that's still in "mapping" state.
            let _ = self.gpu.render_context.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let viewport_resources = self.gpu.viewport_resources.read();
            viewport_resources.cancel_pending_pick();
        }
        self.async_state.pick_state = PickState::Idle;
    }

    pub(super) fn finalize_pending_grab_repair(&mut self) {
        let Some(node_id) = self.doc.sculpt_state.active_node() else {
            return;
        };
        let pending_region = self
            .doc
            .sculpt_state
            .stroke_state_mut()
            .and_then(|stroke_state| stroke_state.pending_grab_repair_region.take());
        let Some(region) = pending_region else {
            return;
        };

        let repaired = sculpt::repair_sdf_region_scene(&mut self.doc.scene, node_id, region, 2)
            .unwrap_or(region);
        let (z0, z1) = repaired.z_range();
        self.try_incremental_voxel_upload(node_id, z0, z1);
        self.upload_voxel_texture_region(node_id, z0, z1);
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_sculpt_target_for_selection;
    use crate::graph::scene::{Scene, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
    use glam::Vec3;

    fn sculpt_grid() -> VoxelGrid {
        VoxelGrid::new_displacement(16, Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    #[test]
    fn host_selection_resolves_to_attached_sculpt_target() {
        let mut scene = Scene::new();
        let host = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt_id = scene.insert_sculpt_above(
            host,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::splat(0.8),
            sculpt_grid(),
        );

        assert_eq!(
            resolve_sculpt_target_for_selection(&scene, Some(host)),
            Some(sculpt_id)
        );
    }

    #[test]
    fn attached_sculpt_selection_resolves_to_self() {
        let mut scene = Scene::new();
        let host = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt_id = scene.create_sculpt(
            host,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::splat(0.8),
            sculpt_grid(),
        );

        assert_eq!(
            resolve_sculpt_target_for_selection(&scene, Some(sculpt_id)),
            Some(sculpt_id)
        );
    }

    #[test]
    fn non_sculpt_selection_has_no_sculpt_target() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Box);

        assert_eq!(
            resolve_sculpt_target_for_selection(&scene, Some(primitive_id)),
            None
        );
    }
}
