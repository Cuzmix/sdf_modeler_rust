use glam::Vec3;
use crate::compat::Instant;

use crate::gpu::buffers;
use crate::gpu::codegen;
use crate::graph::scene::{NodeData, NodeId};
use crate::ui::viewport::ViewportResources;

use super::pipeline_compile::{self, PipelineCompileInputs};
use super::SdfApp;

impl SdfApp {
    pub(super) fn sync_gpu_pipeline(&mut self) {
        let new_environment_key = self.settings.render.environment_fingerprint();
        if new_environment_key != self.gpu.last_environment_fingerprint {
            let mut renderer = self.gpu.render_state.renderer.write();
            if let Some(res) = renderer.callback_resources.get_mut::<ViewportResources>() {
                res.rebuild_environment(
                    &self.gpu.render_state.device,
                    &self.gpu.render_state.queue,
                    &self.settings.render,
                );
            }
            self.gpu.last_environment_fingerprint = new_environment_key;
        }

        self.perf.timings.structure_key_s = 0.0;
        self.perf.timings.shader_codegen_s = 0.0;
        self.perf.timings.pipeline_rebuild_s = 0.0;
        let structure_key_start = Instant::now();
        let structure_version = self.doc.scene.structure_version();
        self.perf.timings.structure_key_s = structure_key_start.elapsed().as_secs_f64();

        // First, pick up any pipeline that finished compiling in the background.
        // Applying a newer pipeline here means subsequent checks below will see
        // `last_structure_version` already advanced, so we won't redundantly
        // spawn another compile for the same version.
        self.try_apply_compiled_pipeline();

        let need_rebuild =
            self.gpu.force_pipeline_resync || structure_version != self.gpu.last_structure_version;
        if !need_rebuild {
            return;
        }

        // Async compilation is only safe when the voxel texture bind-group
        // layout doesn't need to change and we're not rebuilding the composite
        // volume pipelines alongside the render pipeline (those require
        // main-thread `ViewportResources` mutation tied to the same version).
        let sculpt_count = buffers::collect_sculpt_tex_info(&self.doc.scene).len();
        let current_sculpt_count = {
            let renderer = self.gpu.render_state.renderer.read();
            renderer
                .callback_resources
                .get::<ViewportResources>()
                .map(|res| res.voxel_textures.len())
                .unwrap_or(0)
        };
        let want_composite = !cfg!(target_arch = "wasm32")
            && self.settings.render.composite_volume_enabled
            && sculpt_count > 0;
        let must_sync = sculpt_count != current_sculpt_count || want_composite;

        if must_sync {
            // Drop any in-flight async compile — its result would be stale
            // compared to the synchronous rebuild we're about to perform.
            self.gpu.pipeline_compile = None;
            self.rebuild_pipeline_sync(structure_version, sculpt_count, want_composite);
            return;
        }

        // If an async compile is already in flight, wait for it. When it
        // lands we update `last_structure_version` to the compiled version;
        // if the scene has advanced further the next frame's check picks
        // that up and spawns a fresh compile.
        if self.gpu.pipeline_compile.is_some() {
            return;
        }

        self.spawn_async_pipeline_compile(structure_version, sculpt_count);
    }

    /// Synchronous pipeline rebuild — the fallback path used when the voxel
    /// bind-group layout must change (sculpt added/removed) or the composite
    /// volume cache needs rebuilding alongside the render pipeline.
    fn rebuild_pipeline_sync(
        &mut self,
        structure_version: u64,
        sculpt_count: usize,
        want_composite: bool,
    ) {
        let shader_codegen_start = Instant::now();
        let shader_src = codegen::generate_shader(&self.doc.scene, &self.settings.render);
        let pick_shader_src =
            codegen::generate_pick_shader(&self.doc.scene, &self.settings.render);
        self.perf.timings.shader_codegen_s = shader_codegen_start.elapsed().as_secs_f64();

        let pipeline_rebuild_start = Instant::now();
        let mut renderer = self.gpu.render_state.renderer.write();
        if let Some(res) = renderer.callback_resources.get_mut::<ViewportResources>() {
            res.rebuild_pipeline(
                &self.gpu.render_state.device,
                &shader_src,
                &pick_shader_src,
                sculpt_count,
            );

            if want_composite {
                let bounds = self.doc.scene.compute_bounds();
                let padding = 1.5;
                let bounds_min = [
                    bounds.0[0] - padding,
                    bounds.0[1] - padding,
                    bounds.0[2] - padding,
                ];
                let bounds_max = [
                    bounds.1[0] + padding,
                    bounds.1[1] + padding,
                    bounds.1[2] + padding,
                ];
                let resolution = self.settings.render.composite_volume_resolution;

                log::info!(
                    "Composite: building pipelines ({}^3, bounds=[{:.2},{:.2},{:.2}]-[{:.2},{:.2},{:.2}])",
                    resolution,
                    bounds_min[0], bounds_min[1], bounds_min[2],
                    bounds_max[0], bounds_max[1], bounds_max[2],
                );

                let comp_compute_src =
                    codegen::generate_composite_shader(&self.doc.scene, &self.settings.render);
                let comp_render_src = codegen::generate_composite_render_shader(
                    &self.settings.render,
                    bounds_min,
                    bounds_max,
                );

                res.rebuild_composite(
                    &self.gpu.render_state.device,
                    &comp_compute_src,
                    &comp_render_src,
                    resolution,
                    bounds_min,
                    bounds_max,
                );
                log::info!(
                    "Composite: pipelines built, use_composite={}",
                    res.use_composite
                );
                self.perf.composite_full_update_needed = true;
            } else {
                res.use_composite = false;
                res.composite = None;
            }
        }
        self.perf.timings.pipeline_rebuild_s = pipeline_rebuild_start.elapsed().as_secs_f64();
        self.gpu.last_structure_version = structure_version;
        self.gpu.force_pipeline_resync = false;
        self.gpu.buffer_dirty = true; // new pipeline needs fresh buffer data
    }

    /// Prepare compile inputs on the main thread (codegen + BGL cloning) and
    /// hand them off to a worker thread to perform the expensive `wgpu`
    /// calls. The old pipeline keeps rendering until the new one arrives.
    fn spawn_async_pipeline_compile(&mut self, structure_version: u64, sculpt_count: usize) {
        let shader_codegen_start = Instant::now();
        let shader_src = codegen::generate_shader(&self.doc.scene, &self.settings.render);
        let pick_shader_src =
            codegen::generate_pick_shader(&self.doc.scene, &self.settings.render);
        self.perf.timings.shader_codegen_s = shader_codegen_start.elapsed().as_secs_f64();

        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };

        let inputs = PipelineCompileInputs {
            device: self.gpu.render_state.device.clone(),
            render_shader_src: shader_src,
            pick_shader_src,
            target_format: res.target_format,
            camera_bgl: std::sync::Arc::clone(&res.camera_bgl),
            scene_bgl: std::sync::Arc::clone(&res.scene_bgl),
            voxel_tex_bgl: std::sync::Arc::clone(&res.voxel_tex_bgl),
            environment_bgl: std::sync::Arc::clone(&res.environment.bind_group_layout),
            pick_bgl: std::sync::Arc::clone(&res.pick_bgl),
            structure_version,
            sculpt_count,
        };
        drop(renderer);

        self.gpu.pipeline_compile = Some(pipeline_compile::spawn_pipeline_compile(inputs));
    }

    /// Poll the in-flight compile. If the worker has produced a new pipeline
    /// pair, swap it into `ViewportResources` and queue a fresh compile if
    /// the scene has already advanced past the version we just applied.
    fn try_apply_compiled_pipeline(&mut self) {
        let Some(handle) = self.gpu.pipeline_compile.as_ref() else {
            return;
        };
        let compiled = match handle.receiver.try_recv() {
            Ok(compiled) => compiled,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Worker panicked or was dropped; abandon this attempt.
                self.gpu.pipeline_compile = None;
                self.gpu.force_pipeline_resync = true;
                return;
            }
        };
        self.gpu.pipeline_compile = None;

        // If the sculpt count drifted while compiling, the pipeline's
        // voxel_tex_bgl no longer matches the live one and the pipeline is
        // unusable. Drop it — the next sync call will take the sync path and
        // rebuild everything together.
        let current_sculpt_count = {
            let renderer = self.gpu.render_state.renderer.read();
            renderer
                .callback_resources
                .get::<ViewportResources>()
                .map(|res| res.voxel_textures.len())
                .unwrap_or(0)
        };
        if compiled.sculpt_count != current_sculpt_count {
            self.gpu.force_pipeline_resync = true;
            return;
        }

        {
            let mut renderer = self.gpu.render_state.renderer.write();
            if let Some(res) = renderer.callback_resources.get_mut::<ViewportResources>() {
                res.pipeline = compiled.render_pipeline;
                res.pick_pipeline = compiled.pick_pipeline;
                // Composite is handled by the sync path only; disable it here
                // to keep state consistent.
                res.use_composite = false;
                res.composite = None;
            }
        }

        self.gpu.last_structure_version = compiled.structure_version;
        self.gpu.force_pipeline_resync = false;
        self.gpu.buffer_dirty = true;

        // Surface the worker's compile time as `pipeline_rebuild_s` so the
        // profiler keeps reporting how long the pipeline took to build —
        // even though this cost was borne off the main thread.
        self.perf.timings.pipeline_rebuild_s = compiled.worker_wall_ms / 1000.0;
        log::debug!(
            "Async pipeline compile applied: struct_version={}, worker_ms={:.2}",
            compiled.structure_version,
            compiled.worker_wall_ms,
        );
    }

    pub(super) fn upload_scene_buffer(&mut self) {
        let buffer_build_start = Instant::now();
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&self.doc.scene);
        let node_data = buffers::build_node_buffer(
            &self.doc.scene,
            &self.ui.node_graph_state.selected_set,
            &voxel_offsets,
        );
        let sculpt_infos = buffers::collect_sculpt_tex_info(&self.doc.scene);
        self.perf.timings.scene_buffer_build_s = buffer_build_start.elapsed().as_secs_f64();
        self.gpu.voxel_gpu_offsets = voxel_offsets;
        self.gpu.sculpt_tex_indices = sculpt_infos
            .iter()
            .map(|i| (i.node_id, i.tex_idx))
            .collect();
        let buffer_write_start = Instant::now();
        let mut renderer = self.gpu.render_state.renderer.write();
        if let Some(res) = renderer.callback_resources.get_mut::<ViewportResources>() {
            res.update_scene_buffer(
                &self.gpu.render_state.device,
                &self.gpu.render_state.queue,
                &node_data,
            );
            res.update_voxel_buffer(
                &self.gpu.render_state.device,
                &self.gpu.render_state.queue,
                &voxel_data,
            );
            // Upload voxel textures for render shader
            for info in &sculpt_infos {
                if let Some(node) = self.doc.scene.nodes.get(&info.node_id) {
                    if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                        res.upload_voxel_texture(
                            &self.gpu.render_state.device,
                            &self.gpu.render_state.queue,
                            info.tex_idx,
                            voxel_grid.resolution,
                            &voxel_grid.data,
                        );
                    }
                }
            }
        }
        self.perf.timings.scene_buffer_write_s = buffer_write_start.elapsed().as_secs_f64();
    }

    /// Dispatch a full composite volume update (all voxels).
    pub(super) fn try_incremental_voxel_upload(&self, node_id: NodeId, z0: u32, z1: u32) -> bool {
        let Some(&gpu_offset) = self.gpu.voxel_gpu_offsets.get(&node_id) else {
            return false;
        };
        let Some(node) = self.doc.scene.nodes.get(&node_id) else {
            return false;
        };
        let NodeData::Sculpt { ref voxel_grid, .. } = node.data else {
            return false;
        };
        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return false;
        };
        res.update_voxel_region(
            &self.gpu.render_state.queue,
            gpu_offset,
            voxel_grid.resolution,
            z0,
            z1,
            &voxel_grid.data,
        );
        true
    }

    /// Upload dirty z-slab region of a sculpt node's voxel data to its texture3D.
    pub(super) fn upload_voxel_texture_region(&self, node_id: NodeId, z0: u32, z1: u32) {
        let Some(&tex_idx) = self.gpu.sculpt_tex_indices.get(&node_id) else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get(&node_id) else {
            return;
        };
        let NodeData::Sculpt { ref voxel_grid, .. } = node.data else {
            return;
        };
        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        res.upload_voxel_texture_region(
            &self.gpu.render_state.queue,
            tex_idx,
            voxel_grid.resolution,
            z0,
            z1,
            &voxel_grid.data,
        );
    }

    // -----------------------------------------------------------------------
    // Async sculpt pick (1-frame delay, eliminates GPU stall)
    // -----------------------------------------------------------------------

    /// Dispatch a full composite volume update (all voxels).
    pub(super) fn dispatch_composite_full(&self) {
        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        let Some(ref comp) = res.composite else {
            return;
        };
        let r = comp.resolution;
        res.dispatch_composite(
            &self.gpu.render_state.device,
            &self.gpu.render_state.queue,
            [0, 0, 0],
            [r - 1, r - 1, r - 1],
        );
    }

    /// Dispatch an incremental composite update for the brush-affected region.
    pub(super) fn dispatch_composite_region(&self, center: Vec3, radius: f32) {
        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        let Some(ref comp) = res.composite else {
            return;
        };

        let pad = 3i32;
        let r = comp.resolution;
        let rf = r as f32;
        let bmin = comp.bounds_min;
        let bmax = comp.bounds_max;
        let size = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];

        let brush_min = [center.x - radius, center.y - radius, center.z - radius];
        let brush_max = [center.x + radius, center.y + radius, center.z + radius];

        let umin = [
            (((brush_min[0] - bmin[0]) / size[0] * rf).floor() as i32 - pad).max(0) as u32,
            (((brush_min[1] - bmin[1]) / size[1] * rf).floor() as i32 - pad).max(0) as u32,
            (((brush_min[2] - bmin[2]) / size[2] * rf).floor() as i32 - pad).max(0) as u32,
        ];
        let umax = [
            (((brush_max[0] - bmin[0]) / size[0] * rf).ceil() as i32 + pad).min(r as i32 - 1)
                as u32,
            (((brush_max[1] - bmin[1]) / size[1] * rf).ceil() as i32 + pad).min(r as i32 - 1)
                as u32,
            (((brush_max[2] - bmin[2]) / size[2] * rf).ceil() as i32 + pad).min(r as i32 - 1)
                as u32,
        ];

        res.dispatch_composite(
            &self.gpu.render_state.device,
            &self.gpu.render_state.queue,
            umin,
            umax,
        );
    }

    pub(super) fn process_pending_pick(&mut self) {
        // Sculpt mode uses async pick path (poll_sculpt_pick / submit_sculpt_pick)
        if self.doc.sculpt_state.is_active() {
            return;
        }
        let Some(pending) = self.async_state.pending_pick.take() else {
            return;
        };
        // Cancel any in-flight async pick (hover pick) so the staging buffer
        // is unmapped before execute_pick calls queue.submit().
        self.cancel_pending_pick_state();
        let topo_order = self.doc.scene.visible_topo_order();
        let renderer = self.gpu.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        if let Some(result) = res.execute_pick(
            &self.gpu.render_state.device,
            &self.gpu.render_state.queue,
            &pending,
        ) {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                let hit_node_id = topo_order[idx];
                // Shift+click toggles viewport multi-selection.
                if pending.additive_select_held {
                    self.ui.node_graph_state.toggle_select(hit_node_id);
                } else {
                    self.ui.node_graph_state.select_single(hit_node_id);
                }
                self.gpu.buffer_dirty = true;
            }
        } else if !pending.additive_select_held {
            // Clicked empty space — deselect
            self.ui.node_graph_state.clear_selection();
            self.gpu.buffer_dirty = true;
        }
    }
}
