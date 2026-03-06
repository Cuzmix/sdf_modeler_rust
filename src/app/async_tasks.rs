use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use eframe::egui;
use glam::Vec3;

use crate::graph::scene::NodeData;
use crate::graph::voxel;
use crate::sculpt::SculptState;

#[cfg(not(target_arch = "wasm32"))]
use super::{BakeStatus, ExportStatus, ImportStatus};
use super::{BakeRequest, SdfApp};

impl SdfApp {
    // ── Bake ─────────────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_async_bake(&mut self, req: BakeRequest, ctx: &egui::Context) {
        let scene_clone = self.doc.scene.clone();
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

        self.async_state.bake_status = BakeStatus::InProgress {
            existing_sculpt: req.existing_sculpt,
            subtree_root: req.subtree_root,
            color: req.color,
            flatten: req.flatten,
            progress,
            total: resolution,
            receiver: rx,
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn start_async_bake(&mut self, req: BakeRequest, _ctx: &egui::Context) {
        let progress = Arc::new(AtomicU32::new(0));
        let (grid, center) = voxel::bake_subtree_with_progress(
            &self.doc.scene,
            req.subtree_root,
            req.resolution,
            progress,
        );
        self.apply_bake_result(grid, center, req.existing_sculpt, req.subtree_root, req.color, req.flatten);
    }

    /// Instantly create a displacement grid for a non-flatten bake request.
    /// No async thread needed — displacement grids start at 0.0 (O(1)).
    pub(super) fn apply_instant_displacement_bake(&mut self, req: BakeRequest) {
        let (grid, center) = voxel::create_displacement_grid_for_subtree(
            &self.doc.scene, req.subtree_root, req.resolution,
        );

        if let Some(sculpt_id) = req.existing_sculpt {
            if let Some(node) = self.doc.scene.nodes.get_mut(&sculpt_id) {
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
            let sculpt_id = self.doc.scene.insert_sculpt_above(
                req.subtree_root, center, Vec3::ZERO, req.color, grid,
            );
            self.ui.node_graph_state.selected = Some(sculpt_id);
            self.doc.sculpt_state = SculptState::new_active(sculpt_id);
        }
        self.gpu.buffer_dirty = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_async_bake(&mut self) {
        let completed = if let BakeStatus::InProgress { ref receiver, .. } = self.async_state.bake_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some((grid, center)) = completed {
            let (existing_sculpt, subtree_root, color, flatten) = match &self.async_state.bake_status {
                BakeStatus::InProgress {
                    existing_sculpt, subtree_root, color, flatten, ..
                } => (*existing_sculpt, *subtree_root, *color, *flatten),
                _ => unreachable!(),
            };

            self.apply_bake_result(grid, center, existing_sculpt, subtree_root, color, flatten);
            self.async_state.bake_status = BakeStatus::Idle;
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_async_bake(&mut self) {
        // Bake runs synchronously on WASM — nothing to poll.
    }

    fn apply_bake_result(
        &mut self,
        grid: voxel::VoxelGrid,
        center: Vec3,
        existing_sculpt: Option<crate::graph::scene::NodeId>,
        subtree_root: crate::graph::scene::NodeId,
        color: Vec3,
        flatten: bool,
    ) {
        if flatten {
            let new_id = self.doc.scene.flatten_subtree(subtree_root, grid, center, color);
            self.ui.node_graph_state.selected = Some(new_id);
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.doc.sculpt_state = SculptState::new_active(new_id);
        } else if let Some(sculpt_id) = existing_sculpt {
            if let Some(node) = self.doc.scene.nodes.get_mut(&sculpt_id) {
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
            let sculpt_id = self.doc.scene.insert_sculpt_above(
                subtree_root, center, Vec3::ZERO, color, grid,
            );
            self.ui.node_graph_state.selected = Some(sculpt_id);
            self.doc.sculpt_state = SculptState::new_active(sculpt_id);
        }
        self.gpu.buffer_dirty = true;
    }

    // ── Screenshot ───────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn take_screenshot(&self) {
        use crate::ui::viewport::ViewportResources;

        let Some(path) = rfd::FileDialog::new()
            .set_title("Save Screenshot")
            .add_filter("PNG Image", &["png"])
            .save_file()
        else {
            return;
        };

        let renderer = self.gpu.render_state.renderer.read();
        let resources = renderer
            .callback_resources
            .get::<ViewportResources>()
            .unwrap();

        let width = 1920u32;
        let height = 1080u32;
        let scene_bounds = self.doc.scene.compute_bounds();
        let viewport = [0.0, 0.0, width as f32, height as f32];
        let uniform = self.doc.camera.to_uniform(viewport, 0.0, 0.0, false, scene_bounds, -1.0, 0.0, [0.0; 4], [0.0; 4]);

        let pixels = resources.screenshot(
            &self.gpu.render_state.device,
            &self.gpu.render_state.queue,
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

    #[cfg(target_arch = "wasm32")]
    pub(super) fn take_screenshot(&self) {
        log::warn!("Screenshot is not supported on web");
    }

    // ── Export ────────────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_export(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Export Mesh")
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("STL Binary", &["stl"])
            .add_filter("Stanford PLY", &["ply"])
            .add_filter("glTF Binary", &["glb"])
            .add_filter("USD ASCII", &["usda"])
            .save_file()
        else {
            return;
        };

        let scene_clone = self.doc.scene.clone();
        let bounds = self.doc.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let resolution = self.settings.export_resolution.clamp(32, 512);
        let adaptive = self.settings.adaptive_export;
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
                adaptive,
            );
            let _ = tx.send(mesh);
            ctx_clone.request_repaint();
        });

        let total = (resolution + 1) + resolution;
        self.async_state.export_status = ExportStatus::InProgress {
            progress,
            total,
            receiver: rx,
            path,
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn start_export(&mut self, _ctx: &egui::Context) {
        let bounds = self.doc.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let resolution = self.settings.export_resolution.clamp(32, 512);
        let progress = Arc::new(AtomicU32::new(0));

        let mesh = crate::export::marching_cubes(
            &self.doc.scene,
            resolution,
            bounds_min,
            bounds_max,
            &progress,
            self.settings.adaptive_export,
        );

        let mut buf = std::io::Cursor::new(Vec::new());
        if let Err(e) = crate::export::write_obj_to(&mesh, &mut buf) {
            log::error!("Export failed: {}", e);
            return;
        }

        let msg = format!(
            "Exported OBJ ({} verts, {} tris)",
            mesh.vertices.len(), mesh.triangles.len(),
        );
        crate::io::web_download("export.obj", &buf.into_inner(), "model/obj");
        self.ui.toasts.push(super::Toast {
            message: msg,
            is_error: false,
            created: crate::compat::Instant::now(),
            duration: crate::compat::Duration::from_secs(4),
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_export(&mut self) {
        let completed = if let ExportStatus::InProgress { ref receiver, .. } = self.async_state.export_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some(mesh) = completed {
            let path = if let ExportStatus::InProgress { ref path, .. } = self.async_state.export_status {
                path.clone()
            } else {
                unreachable!()
            };

            match crate::export::write_mesh(&mesh, &path) {
                Ok(()) => {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("obj").to_uppercase();
                    let msg = format!(
                        "Exported {} ({} verts, {} tris)",
                        ext, mesh.vertices.len(), mesh.triangles.len(),
                    );
                    log::info!("{} to {:?}", msg, path);
                    self.ui.toasts.push(super::Toast {
                        message: msg,
                        is_error: false,
                        created: crate::compat::Instant::now(),
                        duration: crate::compat::Duration::from_secs(4),
                    });
                }
                Err(e) => {
                    log::error!("Failed to write mesh: {}", e);
                    self.ui.toasts.push(super::Toast {
                        message: format!("Export failed: {}", e),
                        is_error: true,
                        created: crate::compat::Instant::now(),
                        duration: crate::compat::Duration::from_secs(6),
                    });
                }
            }

            self.async_state.export_status = ExportStatus::Idle;
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_export(&mut self) {
        // Export runs synchronously on WASM — nothing to poll.
    }

    // ── Import Mesh ──────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_import(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Import Mesh")
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("STL Binary", &["stl"])
            .add_filter("All Mesh Files", &["obj", "stl"])
            .pick_file()
        else {
            return;
        };

        let mesh = match crate::mesh_import::load_mesh(&path) {
            Ok(m) => m,
            Err(e) => {
                log::error!("Failed to load mesh: {}", e);
                self.ui.toasts.push(super::Toast {
                    message: format!("Import failed: {}", e),
                    is_error: true,
                    created: crate::compat::Instant::now(),
                    duration: crate::compat::Duration::from_secs(6),
                });
                return;
            }
        };

        let resolution = 64u32;
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let result = crate::mesh_import::mesh_to_sdf(&mesh, resolution, &progress_clone);
            let _ = tx.send(result);
            ctx_clone.request_repaint();
        });

        self.async_state.import_status = ImportStatus::InProgress {
            progress,
            total: resolution,
            receiver: rx,
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn start_import(&mut self, _ctx: &egui::Context) {
        log::warn!("Mesh import is not supported on web");
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_import(&mut self) {
        let completed = if let ImportStatus::InProgress { ref receiver, .. } = self.async_state.import_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some((grid, center)) = completed {
            let desired_resolution = grid.resolution;
            let name = self.doc.scene.next_name("Import");
            let sculpt_id = self.doc.scene.add_node(
                name,
                NodeData::Sculpt {
                    input: None,
                    position: center,
                    rotation: Vec3::ZERO,
                    color: Vec3::new(0.7, 0.7, 0.7),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    fresnel: 0.04,
                    layer_intensity: 1.0,
                    voxel_grid: grid,
                    desired_resolution,
                },
            );
            self.ui.node_graph_state.selected = Some(sculpt_id);
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.doc.sculpt_state = SculptState::new_active(sculpt_id);
            self.gpu.buffer_dirty = true;
            self.ui.toasts.push(super::Toast {
                message: "Mesh imported successfully".into(),
                is_error: false,
                created: crate::compat::Instant::now(),
                duration: crate::compat::Duration::from_secs(4),
            });
            self.async_state.import_status = ImportStatus::Idle;
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_import(&mut self) {
        // Import not supported on WASM.
    }
}
