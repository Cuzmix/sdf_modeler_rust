use eframe::wgpu;
use glam::Vec3;

use crate::gpu::picking::PickResult;
use crate::graph::scene::{NodeData, NodeId};
use crate::graph::voxel;
use crate::sculpt::{self, ActiveTool, BrushMode, FalloffMode, SculptState};
use crate::ui::viewport::{BrushDispatch, BrushGpuParams, ViewportResources};

use super::{PickState, SdfApp};

impl SdfApp {
    // -----------------------------------------------------------------------
    // Async sculpt pick (1-frame delay, eliminates GPU stall)
    // -----------------------------------------------------------------------

    /// Poll for a previously submitted async sculpt pick result.
    /// If ready: apply brush at the hit point (CPU + GPU).
    pub(super) fn poll_sculpt_pick(&mut self) {
        if !matches!(self.pick_state, PickState::Pending { .. }) {
            return;
        }

        // Non-blocking GPU poll to advance async map
        self.render_state.device.poll(wgpu::Maintain::Poll);

        // Try to read the result
        let ready = {
            let renderer = self.render_state.renderer.read();
            let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
                return;
            };
            let PickState::Pending { ref receiver } = self.pick_state else {
                return;
            };
            res.try_read_pick_result(receiver)
        };

        let Some(pick_result) = ready else {
            return; // Not ready yet — try again next frame
        };

        self.pick_state = PickState::Idle;

        if let Some(result) = pick_result {
            self.handle_sculpt_hit(result);
        }
    }

    /// Handle a sculpt pick result: apply brush with interpolation.
    /// Handle a sculpt pick result: apply brush with interpolation.
    pub(super) fn handle_sculpt_hit(&mut self, result: PickResult) {
        let topo_order = self.scene.visible_topo_order();
        let idx = result.material_id as usize;
        if idx >= topo_order.len() {
            return;
        }
        let hit_node_id = topo_order[idx];

        let SculptState::Active {
            node_id,
            ref brush_mode,
            brush_radius,
            brush_strength,
            ref falloff_mode,
            smooth_iterations,
            ref mut flatten_reference,
            lazy_radius,
            surface_constraint,
            symmetry_axis,
            ref mut grab_snapshot,
            ref mut grab_start,
            ref mut grab_child_input,
        } = self.sculpt_state
        else {
            return;
        };

        if hit_node_id != node_id {
            return;
        }

        let hit_world = Vec3::new(result.world_pos[0], result.world_pos[1], result.world_pos[2]);
        let brush_mode = brush_mode.clone();
        let falloff_mode = falloff_mode.clone();

        // Grab brush: initialize snapshot and start position on first hit
        let is_grab = brush_mode == BrushMode::Grab;
        if is_grab && grab_snapshot.is_none() {
            if let Some(node) = self.scene.nodes.get(&node_id) {
                if let NodeData::Sculpt { input, ref voxel_grid, position, rotation, .. } = node.data {
                    if let Some(child_id) = input {
                        // Differential sculpt: build total-SDF snapshot (analytical + displacement)
                        let res = voxel_grid.resolution;
                        let mut total_snap = voxel_grid.data.clone();
                        for z in 0..res {
                            for y in 0..res {
                                for x in 0..res {
                                    let local_pos = voxel_grid.grid_to_world(x as f32, y as f32, z as f32);
                                    let world_pos = position + sculpt::inverse_rotate_euler(local_pos, rotation);
                                    let analytical = voxel::evaluate_sdf_tree(&self.scene, child_id, world_pos);
                                    let idx = voxel::VoxelGrid::index(x, y, z, res);
                                    total_snap[idx] += analytical;
                                }
                            }
                        }
                        *grab_snapshot = Some(total_snap);
                        *grab_child_input = Some(child_id);
                    } else {
                        // Standalone sculpt: grid is already total SDF
                        *grab_snapshot = Some(voxel_grid.data.clone());
                        *grab_child_input = None;
                    }
                    *grab_start = Some(hit_world);
                }
            }
        }
        let grab_snap = grab_snapshot.clone();
        let grab_origin = *grab_start;
        let grab_child = *grab_child_input;

        // Lazy brush: smooth cursor with elastic dead zone
        let effective_hit = if lazy_radius > 0.0 {
            if let Some(ref mut lazy_pos) = self.lazy_brush_pos {
                let delta = hit_world - *lazy_pos;
                let dist = delta.length();
                if dist <= lazy_radius {
                    return; // Dead zone — don't apply brush
                }
                let factor = 1.0 - lazy_radius / dist;
                *lazy_pos = *lazy_pos + delta * factor;
                *lazy_pos
            } else {
                self.lazy_brush_pos = Some(hit_world);
                hit_world
            }
        } else {
            hit_world
        };

        // Capture flatten reference on first hit of a Flatten stroke
        if brush_mode == BrushMode::Flatten && flatten_reference.is_none() {
            if let Some(node) = self.scene.nodes.get(&node_id) {
                if let NodeData::Sculpt {
                    ref voxel_grid,
                    position,
                    rotation,
                    ..
                } = node.data
                {
                    let local_hit =
                        sculpt::inverse_rotate_euler(effective_hit - position, rotation);
                    *flatten_reference = Some(voxel_grid.sample(local_hit));
                }
            }
        }
        let flatten_ref_val = flatten_reference.unwrap_or(0.0);

        // Grab brush: single application per frame, centered at grab start
        if is_grab {
            if let (Some(ref snap), Some(origin)) = (&grab_snap, grab_origin) {
                // Project mouse ray onto camera-facing plane at grab depth (like Blender).
                // This gives 1:1 screen-to-world mapping regardless of surface curvature.
                let eye = self.camera.eye();
                let forward = (self.camera.target - eye).normalize();
                let ray_dir = (hit_world - eye).normalize();
                let denom = ray_dir.dot(forward);
                let grab_delta = if denom.abs() > 1e-6 {
                    let t = (origin - eye).dot(forward) / denom;
                    (eye + ray_dir * t) - origin
                } else {
                    hit_world - origin
                };
                let (spos, srot) = match self.scene.nodes.get(&node_id).map(|n| &n.data) {
                    Some(NodeData::Sculpt { position, rotation, .. }) => (*position, *rotation),
                    _ => return,
                };
                // Center at grab start position, not current mouse
                let local_center = sculpt::inverse_rotate_euler(origin - spos, srot);
                let local_delta = sculpt::inverse_rotate_euler(grab_delta, srot);

                let dirty = if let Some(child_id) = grab_child {
                    // Differential sculpt: snapshot is total SDF, write back displacement
                    sculpt::apply_grab_to_grid_differential_scene(
                        &mut self.scene,
                        node_id,
                        snap,
                        local_center,
                        brush_radius,
                        brush_strength,
                        local_delta,
                        &falloff_mode,
                        child_id,
                        spos,
                        srot,
                    )
                } else {
                    // Standalone sculpt: grid is total SDF, write directly
                    if let Some(node) = self.scene.nodes.get_mut(&node_id) {
                        if let NodeData::Sculpt { ref mut voxel_grid, .. } = node.data {
                            Some(sculpt::apply_grab_to_grid(
                                voxel_grid,
                                snap,
                                local_center,
                                brush_radius,
                                brush_strength,
                                local_delta,
                                &falloff_mode,
                            ))
                        } else { None }
                    } else { None }
                };

                if let Some((z0, z1)) = dirty {
                    self.try_incremental_voxel_upload(node_id, z0, z1);
                    self.upload_voxel_texture_region(node_id, z0, z1);
                }
            }
            self.last_sculpt_hit = Some(effective_hit);
            return;
        }

        // Brush stroke interpolation: fill gaps during fast mouse movement
        let hits = self.interpolate_brush_hits(effective_hit, brush_radius);

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

        for &pos in &all_hits {
            // Standard brush modes (Add/Carve/Smooth/Flatten/Inflate)
                let dirty_range = sculpt::apply_brush(
                    &mut self.scene,
                    node_id,
                    pos,
                    &brush_mode,
                    brush_radius,
                    brush_strength,
                    &falloff_mode,
                    smooth_iterations,
                    flatten_ref_val,
                    surface_constraint,
                );

                // GPU brush for non-Smooth modes (instant visual update on storage buffer)
                if brush_mode != BrushMode::Smooth {
                    self.dispatch_gpu_brush(
                        node_id,
                        pos,
                        &brush_mode,
                        brush_radius,
                        brush_strength,
                        &falloff_mode,
                        flatten_ref_val,
                        surface_constraint,
                    );
                }

                if let Some((z0, z1)) = dirty_range {
                    // Smooth mode: upload CPU data to GPU storage buffer (no GPU compute)
                    if brush_mode == BrushMode::Smooth {
                        self.try_incremental_voxel_upload(node_id, z0, z1);
                    }
                    // Update voxel texture region (keeps texture3D in sync)
                    self.upload_voxel_texture_region(node_id, z0, z1);
                }
        }

        // Incremental composite volume update for the brush-affected region
        if self.settings.render.composite_volume_enabled && !all_hits.is_empty() {
            self.dispatch_composite_region(effective_hit, brush_radius);
        }

        self.last_sculpt_hit = Some(effective_hit);
    }

    /// Interpolate between last and current hit to prevent gaps.
    /// Interpolate between last and current hit to prevent gaps.
    pub(super) fn interpolate_brush_hits(&self, current: Vec3, brush_radius: f32) -> Vec<Vec3> {
        let Some(last) = self.last_sculpt_hit else {
            return vec![current];
        };
        let dist = (current - last).length();
        let step = brush_radius * 0.3;
        if dist <= step {
            return vec![current];
        }
        let n = (dist / step).ceil() as usize;
        let mut hits = Vec::with_capacity(n);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            hits.push(last + (current - last) * t);
        }
        hits
    }

    /// Dispatch GPU compute brush to modify voxel_buffer directly on the GPU.
    /// Dispatch GPU compute brush to modify voxel_buffer directly on the GPU.
    pub(super) fn dispatch_gpu_brush(
        &self,
        node_id: NodeId,
        hit_world: Vec3,
        brush_mode: &BrushMode,
        radius: f32,
        strength: f32,
        falloff_mode: &FalloffMode,
        flatten_ref: f32,
        surface_constraint: f32,
    ) {
        let Some(&gpu_offset) = self.voxel_gpu_offsets.get(&node_id) else {
            return;
        };
        let Some(node) = self.scene.nodes.get(&node_id) else {
            return;
        };
        let NodeData::Sculpt {
            position,
            rotation,
            ref voxel_grid,
            ..
        } = node.data
        else {
            return;
        };

        let local_hit = sculpt::inverse_rotate_euler(hit_world - position, rotation);
        let res = voxel_grid.resolution;

        // Compute grid-space AABB of the brush
        let brush_min = local_hit - Vec3::splat(radius);
        let brush_max = local_hit + Vec3::splat(radius);
        let g_min = voxel_grid.world_to_grid(brush_min);
        let g_max = voxel_grid.world_to_grid(brush_max);

        let x0 = (g_min.x.floor().max(0.0) as u32).min(res - 1);
        let y0 = (g_min.y.floor().max(0.0) as u32).min(res - 1);
        let z0 = (g_min.z.floor().max(0.0) as u32).min(res - 1);
        let x1 = (g_max.x.ceil().max(0.0) as u32).min(res - 1);
        let y1 = (g_max.y.ceil().max(0.0) as u32).min(res - 1);
        let z1 = (g_max.z.ceil().max(0.0) as u32).min(res - 1);

        let dispatch = BrushDispatch {
            params: BrushGpuParams {
                center_local: local_hit.to_array(),
                radius,
                strength,
                sign_val: brush_mode.sign(),
                grid_offset: gpu_offset,
                grid_resolution: res,
                bounds_min: voxel_grid.bounds_min.to_array(),
                _pad0: 0.0,
                bounds_max: voxel_grid.bounds_max.to_array(),
                _pad1: 0.0,
                min_voxel: [x0, y0, z0],
                _pad2: 0,
                brush_mode: brush_mode.gpu_mode(),
                falloff_mode: falloff_mode.gpu_mode(),
                smooth_iterations: 0,
                flatten_ref,
                surface_constraint,
                _pad3: [0.0; 3],
            },
            workgroups: [
                (x1 - x0 + 4) / 4,
                (y1 - y0 + 4) / 4,
                (z1 - z0 + 4) / 4,
            ],
        };

        let renderer = self.render_state.renderer.read();
        if let Some(vr) = renderer.callback_resources.get::<ViewportResources>() {
            vr.dispatch_brush(&self.render_state.device, &self.render_state.queue, &dispatch);
        }
    }

    /// Submit an async pick for sculpt mode (non-blocking).
    /// Submit an async pick for sculpt mode (non-blocking).
    pub(super) fn submit_sculpt_pick(&mut self) {
        if !self.sculpt_state.is_active() {
            return;
        }
        if !matches!(self.pick_state, PickState::Idle) {
            return;
        }
        let Some(pending) = self.pending_pick.take() else {
            return;
        };
        let renderer = self.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        let rx = res.submit_pick(
            &self.render_state.device,
            &self.render_state.queue,
            &pending,
        );
        drop(renderer);

        self.pick_state = PickState::Pending { receiver: rx };
    }

    pub(super) fn sync_sculpt_state(&mut self) {
        match self.active_tool {
            ActiveTool::Select => {
                // Original behavior: always deactivate sculpt in Select mode
                if self.sculpt_state.is_active() {
                    self.sculpt_state = SculptState::Inactive;
                    self.last_sculpt_hit = None;
                    self.lazy_brush_pos = None;
                    self.pick_state = PickState::Idle;
                }
            }
            ActiveTool::Sculpt => {
                let sel = self.node_graph_state.selected;
                let active = self.sculpt_state.active_node();
                if sel != active {
                    // Selection changed — try to activate on new node
                    if let Some(id) = sel {
                        if self.scene.nodes.get(&id).map_or(false, |n| matches!(n.data, NodeData::Sculpt { .. })) {
                            self.sculpt_state = SculptState::new_active(id);
                        } else {
                            self.sculpt_state = SculptState::Inactive;
                        }
                    } else {
                        self.sculpt_state = SculptState::Inactive;
                    }
                    self.last_sculpt_hit = None;
                    self.lazy_brush_pos = None;
                    self.pick_state = PickState::Idle;
                }
            }
        }
    }

}
