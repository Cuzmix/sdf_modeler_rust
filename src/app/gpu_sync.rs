use glam::Vec3;

use crate::gpu::buffers;
use crate::gpu::codegen;
use crate::graph::presented_object::{collect_render_highlight_ids, resolve_host_selection};
use crate::graph::scene::{NodeData, NodeId};

use super::SdfApp;

impl SdfApp {
    pub(super) fn sync_gpu_pipeline(&mut self) {
        let new_environment_key = self.settings.render.environment_fingerprint();
        if new_environment_key != self.gpu.last_environment_fingerprint {
            let mut resources = self.gpu.viewport_resources.write();
            resources.rebuild_environment(
                &self.gpu.render_context.device,
                &self.gpu.render_context.queue,
                &self.settings.render,
            );
            self.gpu.last_environment_fingerprint = new_environment_key;
        }

        let new_key = self.doc.scene.structure_key();
        if new_key != self.gpu.current_structure_key {
            let shader_src = codegen::generate_shader(&self.doc.scene, &self.settings.render);
            let pick_shader_src =
                codegen::generate_pick_shader(&self.doc.scene, &self.settings.render);
            let sculpt_count = buffers::collect_sculpt_tex_info(&self.doc.scene).len();
            let mut resources = self.gpu.viewport_resources.write();

            resources.rebuild_pipeline(
                &self.gpu.render_context.device,
                &shader_src,
                &pick_shader_src,
                sculpt_count,
            );

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
                    bounds_min[0],
                    bounds_min[1],
                    bounds_min[2],
                    bounds_max[0],
                    bounds_max[1],
                    bounds_max[2],
                );

                let comp_compute_src =
                    codegen::generate_composite_shader(&self.doc.scene, &self.settings.render);
                let comp_render_src = codegen::generate_composite_render_shader(
                    &self.settings.render,
                    bounds_min,
                    bounds_max,
                );

                resources.rebuild_composite(
                    &self.gpu.render_context.device,
                    &comp_compute_src,
                    &comp_render_src,
                    resolution,
                    bounds_min,
                    bounds_max,
                );
                log::info!(
                    "Composite: pipelines built, use_composite={}",
                    resources.use_composite
                );
                self.perf.composite_full_update_needed = true;
            } else {
                resources.use_composite = false;
                resources.composite = None;
            }

            self.gpu.current_structure_key = new_key;
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn upload_scene_buffer(&mut self) {
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&self.doc.scene);
        let render_highlight_ids = collect_render_highlight_ids(
            &self.doc.scene,
            self.ui.selection.selected,
            &self.ui.selection.selected_set,
        );
        let node_data =
            buffers::build_node_buffer(&self.doc.scene, &render_highlight_ids, &voxel_offsets);
        let sculpt_infos = buffers::collect_sculpt_tex_info(&self.doc.scene);
        self.gpu.voxel_gpu_offsets = voxel_offsets;
        self.gpu.sculpt_tex_indices = sculpt_infos
            .iter()
            .map(|info| (info.node_id, info.tex_idx))
            .collect();

        let mut resources = self.gpu.viewport_resources.write();
        resources.update_scene_buffer(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            &node_data,
        );
        resources.update_voxel_buffer(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            &voxel_data,
        );

        for info in &sculpt_infos {
            if let Some(node) = self.doc.scene.nodes.get(&info.node_id) {
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    resources.upload_voxel_texture(
                        &self.gpu.render_context.device,
                        &self.gpu.render_context.queue,
                        info.tex_idx,
                        voxel_grid.resolution,
                        &voxel_grid.data,
                    );
                }
            }
        }
    }

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

        let resources = self.gpu.viewport_resources.read();
        resources.update_voxel_region(
            &self.gpu.render_context.queue,
            gpu_offset,
            voxel_grid.resolution,
            z0,
            z1,
            &voxel_grid.data,
        );
        true
    }

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

        let resources = self.gpu.viewport_resources.read();
        resources.upload_voxel_texture_region(
            &self.gpu.render_context.queue,
            tex_idx,
            voxel_grid.resolution,
            z0,
            z1,
            &voxel_grid.data,
        );
    }

    pub(super) fn dispatch_composite_full(&self) {
        let resources = self.gpu.viewport_resources.read();
        let Some(ref composite) = resources.composite else {
            return;
        };
        let resolution = composite.resolution;
        resources.dispatch_composite(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            [0, 0, 0],
            [resolution - 1, resolution - 1, resolution - 1],
        );
    }

    pub(super) fn dispatch_composite_region(&self, center: Vec3, radius: f32) {
        let resources = self.gpu.viewport_resources.read();
        let Some(ref composite) = resources.composite else {
            return;
        };

        let pad = 3i32;
        let resolution = composite.resolution;
        let resolution_f32 = resolution as f32;
        let bounds_min = composite.bounds_min;
        let bounds_max = composite.bounds_max;
        let size = [
            bounds_max[0] - bounds_min[0],
            bounds_max[1] - bounds_min[1],
            bounds_max[2] - bounds_min[2],
        ];

        let brush_min = [center.x - radius, center.y - radius, center.z - radius];
        let brush_max = [center.x + radius, center.y + radius, center.z + radius];

        let min_index = [
            (((brush_min[0] - bounds_min[0]) / size[0] * resolution_f32).floor() as i32 - pad)
                .max(0) as u32,
            (((brush_min[1] - bounds_min[1]) / size[1] * resolution_f32).floor() as i32 - pad)
                .max(0) as u32,
            (((brush_min[2] - bounds_min[2]) / size[2] * resolution_f32).floor() as i32 - pad)
                .max(0) as u32,
        ];
        let max_index = [
            (((brush_max[0] - bounds_min[0]) / size[0] * resolution_f32).ceil() as i32 + pad)
                .min(resolution as i32 - 1) as u32,
            (((brush_max[1] - bounds_min[1]) / size[1] * resolution_f32).ceil() as i32 + pad)
                .min(resolution as i32 - 1) as u32,
            (((brush_max[2] - bounds_min[2]) / size[2] * resolution_f32).ceil() as i32 + pad)
                .min(resolution as i32 - 1) as u32,
        ];

        resources.dispatch_composite(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            min_index,
            max_index,
        );
    }

    pub(super) fn process_pending_pick(&mut self) {
        if self.doc.sculpt_state.is_active() {
            return;
        }
        let Some(pending) = self.async_state.pending_pick.take() else {
            return;
        };

        self.cancel_pending_pick_state();
        let topo_order = self.doc.scene.visible_topo_order();
        let resources = self.gpu.viewport_resources.read();
        if let Some(result) = resources.execute_pick(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            &pending,
        ) {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                let hit_node_id = topo_order[idx];
                let selected_host = resolve_host_selection(&self.doc.scene, Some(hit_node_id))
                    .unwrap_or(hit_node_id);
                if pending.additive_select_held {
                    self.ui.selection.toggle_select(selected_host);
                } else {
                    self.ui.selection.select_single(selected_host);
                }
                self.gpu.buffer_dirty = true;
            }
        } else if !pending.additive_select_held {
            self.ui.selection.clear_selection();
            self.gpu.buffer_dirty = true;
        }
    }
}
