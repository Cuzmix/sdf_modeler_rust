use glam::Vec3;

use crate::gpu::buffers;
use crate::gpu::codegen;
use crate::graph::scene::{NodeData, NodeId};
use crate::ui::viewport::ViewportResources;

use super::SdfApp;

impl SdfApp {
    pub(super) fn sync_gpu_pipeline(&mut self) {
        let new_key = self.doc.scene.structure_key();
        if new_key != self.gpu.current_structure_key {
            let shader_src = codegen::generate_shader(&self.doc.scene, &self.settings.render);
            let pick_shader_src = codegen::generate_pick_shader(&self.doc.scene, &self.settings.render);
            let sculpt_count = buffers::collect_sculpt_tex_info(&self.doc.scene).len();
            let mut renderer = self.gpu.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                // Always rebuild the normal + pick pipelines
                res.rebuild_pipeline(
                    &self.gpu.render_state.device, &shader_src, &pick_shader_src, sculpt_count,
                );

                // Rebuild composite pipelines if enabled and there are sculpt nodes.
                // Composite uses 3D storage textures — not supported on WebGPU,
                // so always disable on WASM.
                let want_composite = !cfg!(target_arch = "wasm32")
                    && self.settings.render.composite_volume_enabled
                    && sculpt_count > 0;

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

                    let comp_compute_src = codegen::generate_composite_shader(
                        &self.doc.scene, &self.settings.render,
                    );
                    let comp_render_src = codegen::generate_composite_render_shader(
                        &self.settings.render, bounds_min, bounds_max,
                    );

                    res.rebuild_composite(
                        &self.gpu.render_state.device,
                        &comp_compute_src,
                        &comp_render_src,
                        resolution,
                        bounds_min,
                        bounds_max,
                    );
                    log::info!("Composite: pipelines built, use_composite={}", res.use_composite);
                    self.perf.composite_full_update_needed = true;
                } else {
                    res.use_composite = false;
                    res.composite = None;
                }
            }
            self.gpu.current_structure_key = new_key;
            self.gpu.buffer_dirty = true; // new pipeline needs fresh buffer data
        }
    }

    pub(super) fn upload_scene_buffer(&mut self) {
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&self.doc.scene);
        let node_data =
            buffers::build_node_buffer(&self.doc.scene, self.ui.node_graph_state.selected, &voxel_offsets);
        let sculpt_infos = buffers::collect_sculpt_tex_info(&self.doc.scene);
        self.gpu.voxel_gpu_offsets = voxel_offsets;
        self.gpu.sculpt_tex_indices = sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();
        let mut renderer = self.gpu.render_state.renderer.write();
        if let Some(res) = renderer
            .callback_resources
            .get_mut::<ViewportResources>()
        {
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
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else { return; };
        let Some(ref comp) = res.composite else { return; };
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
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else { return; };
        let Some(ref comp) = res.composite else { return; };

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
            (((brush_max[0] - bmin[0]) / size[0] * rf).ceil() as i32 + pad).min(r as i32 - 1) as u32,
            (((brush_max[1] - bmin[1]) / size[1] * rf).ceil() as i32 + pad).min(r as i32 - 1) as u32,
            (((brush_max[2] - bmin[2]) / size[2] * rf).ceil() as i32 + pad).min(r as i32 - 1) as u32,
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
                self.ui.node_graph_state.selected = Some(hit_node_id);
                self.gpu.buffer_dirty = true;
            }
        } else {
            // Clicked empty space — deselect
            self.ui.node_graph_state.selected = None;
            self.gpu.buffer_dirty = true;
        }
    }

}
