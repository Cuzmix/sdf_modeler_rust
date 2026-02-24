use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use eframe::egui;
use glam::Vec3;

use crate::graph::scene::NodeData;
use crate::graph::voxel;
use crate::sculpt::SculptState;
use crate::ui::viewport::ViewportResources;

use super::{BakeRequest, BakeStatus, ExportStatus, SdfApp};

impl SdfApp {
    pub(super) fn start_async_bake(&mut self, req: BakeRequest, ctx: &egui::Context) {
        let scene_clone = self.scene.clone();
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx_clone = ctx.clone();
        let resolution = req.resolution;
        let subtree_root = req.subtree_root;

        std::thread::spawn(move || {
            let result = voxel::bake_subtree_with_progress(
                &scene_clone,
                subtree_root,
                resolution,
                progress_clone,
            );
            let _ = tx.send(result);
            ctx_clone.request_repaint();
        });

        self.bake_status = BakeStatus::InProgress {
            existing_sculpt: req.existing_sculpt,
            subtree_root: req.subtree_root,
            color: req.color,
            flatten: req.flatten,
            progress,
            total: resolution,
            receiver: rx,
        };
    }

    /// Instantly create a displacement grid for a non-flatten bake request.
    /// No async thread needed — displacement grids start at 0.0 (O(1)).
    /// Instantly create a displacement grid for a non-flatten bake request.
    /// No async thread needed — displacement grids start at 0.0 (O(1)).
    pub(super) fn apply_instant_displacement_bake(&mut self, req: BakeRequest) {
        let (grid, center) = voxel::create_displacement_grid_for_subtree(
            &self.scene, req.subtree_root, req.resolution,
        );

        if let Some(sculpt_id) = req.existing_sculpt {
            // Re-bake: reset existing sculpt's displacement to zero
            if let Some(node) = self.scene.nodes.get_mut(&sculpt_id) {
                if let NodeData::Sculpt {
                    voxel_grid: ref mut vg,
                    position: ref mut p,
                    ..
                } = node.data
                {
                    *vg = grid;
                    *p = center;
                }
            }
        } else {
            // New sculpt: create above subtree_root
            let sculpt_id = self.scene.insert_sculpt_above(
                req.subtree_root, center, Vec3::ZERO, req.color, grid,
            );
            self.node_graph_state.selected = Some(sculpt_id);
            self.sculpt_state = SculptState::new_active(sculpt_id);
        }
        self.buffer_dirty = true;
    }

    pub(super) fn poll_async_bake(&mut self) {
        let completed = if let BakeStatus::InProgress { ref receiver, .. } = self.bake_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some((grid, center)) = completed {
            // Extract fields before replacing status
            let (existing_sculpt, subtree_root, color, flatten) = match &self.bake_status {
                BakeStatus::InProgress {
                    existing_sculpt, subtree_root, color, flatten, ..
                } => (*existing_sculpt, *subtree_root, *color, *flatten),
                _ => unreachable!(),
            };

            if flatten {
                // Flatten: replace entire subtree with standalone Sculpt
                let new_id = self.scene.flatten_subtree(subtree_root, grid, center, color);
                self.node_graph_state.selected = Some(new_id);
                self.node_graph_state.layout_dirty = true;
                self.sculpt_state = SculptState::new_active(new_id);
            } else if let Some(sculpt_id) = existing_sculpt {
                // Re-bake: update existing sculpt node
                if let Some(node) = self.scene.nodes.get_mut(&sculpt_id) {
                    if let NodeData::Sculpt {
                        voxel_grid: ref mut vg,
                        position: ref mut p,
                        ..
                    } = node.data
                    {
                        *vg = grid;
                        *p = center;
                    }
                }
            } else {
                // New sculpt: create above subtree_root
                let sculpt_id = self.scene.insert_sculpt_above(
                    subtree_root, center, Vec3::ZERO, color, grid,
                );
                self.node_graph_state.selected = Some(sculpt_id);
                self.sculpt_state = SculptState::new_active(sculpt_id);
            }

            self.buffer_dirty = true;
            self.bake_status = BakeStatus::Idle;
        }
    }

    pub(super) fn take_screenshot(&self) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save Screenshot")
            .add_filter("PNG Image", &["png"])
            .save_file()
        else {
            return;
        };

        let renderer = self.render_state.renderer.read();
        let resources = renderer
            .callback_resources
            .get::<ViewportResources>()
            .unwrap();

        // Use a reasonable default size; actual viewport size isn't easily accessible here
        let width = 1920u32;
        let height = 1080u32;
        let scene_bounds = self.scene.compute_bounds();
        let viewport = [0.0, 0.0, width as f32, height as f32];
        let uniform = self.camera.to_uniform(viewport, 0.0, 0.0, false, scene_bounds, -1.0);

        let pixels = resources.screenshot(
            &self.render_state.device,
            &self.render_state.queue,
            &uniform,
            width,
            height,
        );

        if let Err(e) = image::save_buffer(
            &path,
            &pixels,
            width,
            height,
            image::ColorType::Rgba8,
        ) {
            log::error!("Failed to save screenshot: {}", e);
        } else {
            log::info!("Screenshot saved to {:?}", path);
        }
    }

    pub(super) fn start_export(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Export OBJ Mesh")
            .add_filter("Wavefront OBJ", &["obj"])
            .save_file()
        else {
            return;
        };

        let scene_clone = self.scene.clone();
        let bounds = self.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let resolution = 128u32; // Default export resolution
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let mesh = crate::export::marching_cubes(
                &scene_clone,
                resolution,
                bounds_min,
                bounds_max,
                &progress_clone,
            );
            let _ = tx.send(mesh);
            ctx_clone.request_repaint();
        });

        // Total progress = (resolution+1) sampling slices + resolution cell slices
        let total = (resolution + 1) + resolution;
        self.export_status = ExportStatus::InProgress {
            progress,
            total,
            receiver: rx,
            path,
        };
    }

    pub(super) fn poll_export(&mut self) {
        let completed = if let ExportStatus::InProgress { ref receiver, .. } = self.export_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some(mesh) = completed {
            let path = if let ExportStatus::InProgress { ref path, .. } = self.export_status {
                path.clone()
            } else {
                unreachable!()
            };

            match crate::export::write_obj(&mesh, &path) {
                Ok(()) => log::info!(
                    "Exported {} vertices, {} triangles to {:?}",
                    mesh.vertices.len(),
                    mesh.triangles.len(),
                    path,
                ),
                Err(e) => log::error!("Failed to write OBJ: {}", e),
            }

            self.export_status = ExportStatus::Idle;
        }
    }

}
