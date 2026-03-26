use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Arc;

use glam::Vec3;

use crate::desktop_dialogs::FileDialogSelection;
use crate::gpu::camera::CameraUniform;
use crate::graph::presented_object::resolve_presented_object;
use crate::graph::scene::{MaterialParams, NodeData};
use crate::graph::voxel;
use crate::viewport::ViewportResources;

use super::{BakeRequest, SdfApp};
#[cfg(not(target_arch = "wasm32"))]
use super::{BakeStatus, ExportStatus, ImportStatus};

fn push_platform_unsupported_toast(app: &mut SdfApp, feature: &str) {
    app.ui.toasts.push(super::Toast {
        message: format!("{feature} is not supported on this platform yet"),
        is_error: true,
        created: crate::compat::Instant::now(),
        duration: crate::compat::Duration::from_secs(4),
    });
}

impl SdfApp {
    fn build_viewport_uniform(
        &self,
        resources: &ViewportResources,
        width: u32,
        height: u32,
    ) -> CameraUniform {
        use crate::settings::{BackgroundMode, EnvironmentBackgroundMode};

        let scene_bounds = self.doc.scene.compute_bounds();
        let viewport = [0.0, 0.0, width as f32, height as f32];
        let (scene_light_count, scene_light_list, scene_ambient) =
            crate::gpu::buffers::collect_scene_lights(
                &self.doc.scene,
                self.doc.camera.eye(),
                self.doc.soloed_light,
                self.last_time as f32,
            );
        let volumetric_count = scene_light_list
            .iter()
            .filter(|light| light.volumetric[0] > 0.5)
            .count() as f32;
        let scene_light_info = [
            scene_light_count as f32,
            volumetric_count,
            self.settings.render.volumetric_steps as f32,
            0.0,
        ];

        let mut scene_lights_flat = [[0.0_f32; 4]; 32];
        let mut scene_light_vol = [[0.0_f32; 4]; 8];
        for (index, light) in scene_light_list.iter().enumerate() {
            scene_lights_flat[index * 4] = light.position_type;
            scene_lights_flat[index * 4 + 1] = light.direction_intensity;
            scene_lights_flat[index * 4 + 2] = light.color_range;
            scene_lights_flat[index * 4 + 3] = light.params;
            scene_light_vol[index] = light.volumetric;
        }

        let ambient_luminance = scene_ambient
            .color
            .dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
        let effective_ambient = if ambient_luminance > 0.0 {
            ambient_luminance
        } else {
            self.settings.render.ambient
        };
        let ambient_info = [
            effective_ambient,
            self.settings.render.environment_specular_intensity(),
            resources
                .environment
                .prefiltered_mip_count
                .saturating_sub(1) as f32,
            self.settings.render.ao_mode.gpu_value(),
        ];
        let background_info = [
            match self.settings.render.environment_background_mode {
                EnvironmentBackgroundMode::Procedural => 0.0,
                EnvironmentBackgroundMode::Environment => 1.0,
            },
            self.settings
                .render
                .environment_background_blur
                .clamp(0.0, 1.0),
            match self.settings.render.background_mode {
                BackgroundMode::SkyGradient => 0.0,
                BackgroundMode::SolidColor => 1.0,
            },
            0.0,
        ];
        let environment_flags = (if self.settings.render.specular_aa_enabled {
            1_u32
        } else {
            0_u32
        }) | self.settings.render.local_reflection_mode.flag_bit();
        let environment_info = [
            self.settings
                .render
                .environment_rotation_degrees
                .to_radians(),
            self.settings.render.environment_exposure.exp2(),
            if resources.environment.has_hdri_background_texture() {
                1.0
            } else {
                0.0
            },
            environment_flags as f32,
        ];

        self.doc.camera.to_uniform(
            viewport,
            0.0,
            0.0,
            false,
            scene_bounds,
            -1.0,
            0.0,
            [0.0; 4],
            [0.0; 4],
            ambient_info,
            background_info,
            [
                self.settings.render.sky_horizon[0],
                self.settings.render.sky_horizon[1],
                self.settings.render.sky_horizon[2],
                0.0,
            ],
            [
                self.settings.render.sky_zenith[0],
                self.settings.render.sky_zenith[1],
                self.settings.render.sky_zenith[2],
                0.0,
            ],
            [
                self.settings.render.bg_solid_color[0],
                self.settings.render.bg_solid_color[1],
                self.settings.render.bg_solid_color[2],
                0.0,
            ],
            environment_info,
            scene_light_info,
            scene_lights_flat,
            scene_light_vol,
        )
    }

    pub(super) fn viewport_frame_image(
        &self,
        width: u32,
        height: u32,
        generation: u64,
    ) -> super::frontend_models::ViewportFrameImage {
        let mut resources = self.gpu.viewport_resources.write();
        let uniform = self.build_viewport_uniform(&resources, width, height);
        let post_process = crate::viewport::ViewportPostProcessOptions {
            outline_color: self.settings.render.outline_color,
            outline_width: self.settings.render.outline_thickness,
            bloom_params: if self.settings.render.bloom_enabled {
                [
                    self.settings.render.bloom_threshold,
                    self.settings.render.bloom_intensity,
                    self.settings.render.bloom_radius,
                    1.0,
                ]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            },
        };
        let rgba8 = resources.screenshot(
            &self.gpu.render_context.device,
            &self.gpu.render_context.queue,
            &uniform,
            width,
            height,
            post_process,
        );

        super::frontend_models::ViewportFrameImage {
            width,
            height,
            rgba8,
            generation,
        }
    }

    pub(super) fn render_viewport_texture(
        &self,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let mut resources = self.gpu.viewport_resources.write();
        let uniform = self.build_viewport_uniform(&resources, width, height);
        let post_process = crate::viewport::ViewportPostProcessOptions {
            outline_color: self.settings.render.outline_color,
            outline_width: self.settings.render.outline_thickness,
            bloom_params: if self.settings.render.bloom_enabled {
                [
                    self.settings.render.bloom_threshold,
                    self.settings.render.bloom_intensity,
                    self.settings.render.bloom_radius,
                    1.0,
                ]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            },
        };
        resources.render_to_view(crate::viewport::ViewportRenderRequest {
            device: &self.gpu.render_context.device,
            queue: &self.gpu.render_context.queue,
            uniform: &uniform,
            view: target_view,
            camera: &self.doc.camera,
            reference_images: &self.ui.reference_images,
            viewport_size: (width, height),
            post_process,
        });
    }
    // ── Bake ─────────────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_async_bake(&mut self, req: BakeRequest) {
        let scene_clone = self.doc.scene.clone();
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let (tx, rx) = std::sync::mpsc::channel();
        let wake = self.wake.clone();
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
            wake.wake();
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
    pub(super) fn start_async_bake(&mut self, req: BakeRequest) {
        let progress = Arc::new(AtomicU32::new(0));
        let (grid, center) = voxel::bake_subtree_with_progress(
            &self.doc.scene,
            req.subtree_root,
            req.resolution,
            progress,
        );
        self.apply_bake_result(
            grid,
            center,
            req.existing_sculpt,
            req.subtree_root,
            req.color,
            req.flatten,
        );
    }

    /// Instantly create a displacement grid for a non-flatten bake request.
    /// No async thread needed — displacement grids start at 0.0 (O(1)).
    pub(super) fn apply_instant_displacement_bake(&mut self, req: BakeRequest) {
        let (grid, center) = voxel::create_displacement_grid_for_subtree(
            &self.doc.scene,
            req.subtree_root,
            req.resolution,
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
            let (_, sculpt_id) = self.doc.scene.insert_sculpt_layer_above(
                req.subtree_root,
                center,
                Vec3::ZERO,
                req.color,
                grid,
            );
            let selected_host = resolve_presented_object(&self.doc.scene, req.subtree_root)
                .map(|presented| presented.host_id)
                .unwrap_or(req.subtree_root);
            self.ui.selection.select_single(selected_host);
            let extent = self.scene_avg_extent();
            self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
            self.doc
                .sculpt_state
                .activate_preserving_session(sculpt_id, Some(extent));
            self.ui.primary_shell.interaction_mode =
                crate::app::state::InteractionMode::Sculpt(self.doc.sculpt_state.selected_brush());
            self.ui.measurement_mode = false;
            self.ui.measurement_points.clear();
        }
        self.gpu.buffer_dirty = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_async_bake(&mut self) {
        let completed =
            if let BakeStatus::InProgress { ref receiver, .. } = self.async_state.bake_status {
                receiver.try_recv().ok()
            } else {
                None
            };

        if let Some((grid, center)) = completed {
            let (existing_sculpt, subtree_root, color, flatten) =
                match &self.async_state.bake_status {
                    BakeStatus::InProgress {
                        existing_sculpt,
                        subtree_root,
                        color,
                        flatten,
                        ..
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
            let new_id = self
                .doc
                .scene
                .flatten_subtree(subtree_root, grid, center, color);
            self.ui.selection.select_single(new_id);
            self.ui.scene_graph_view.needs_initial_rebuild = true;
            let extent = self.scene_avg_extent();
            self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
            self.doc
                .sculpt_state
                .activate_preserving_session(new_id, Some(extent));
            self.ui.primary_shell.interaction_mode =
                crate::app::state::InteractionMode::Sculpt(self.doc.sculpt_state.selected_brush());
            self.ui.measurement_mode = false;
            self.ui.measurement_points.clear();
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
            let (_, sculpt_id) = self.doc.scene.insert_sculpt_layer_above(
                subtree_root,
                center,
                Vec3::ZERO,
                color,
                grid,
            );
            let selected_host = resolve_presented_object(&self.doc.scene, subtree_root)
                .map(|presented| presented.host_id)
                .unwrap_or(subtree_root);
            self.ui.selection.select_single(selected_host);
            let extent = self.scene_avg_extent();
            self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
            self.doc
                .sculpt_state
                .activate_preserving_session(sculpt_id, Some(extent));
            self.ui.primary_shell.interaction_mode =
                crate::app::state::InteractionMode::Sculpt(self.doc.sculpt_state.selected_brush());
            self.ui.measurement_mode = false;
            self.ui.measurement_points.clear();
        }
        self.gpu.buffer_dirty = true;
    }

    // ── Screenshot ───────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn take_screenshot(&mut self) {
        let path = match crate::desktop_dialogs::screenshot_dialog() {
            FileDialogSelection::Selected(path) => path,
            FileDialogSelection::Unsupported => {
                push_platform_unsupported_toast(self, "Saving screenshots");
                return;
            }
            FileDialogSelection::Cancelled => {
                return;
            }
        };

        let frame = self.viewport_frame_image(1920, 1080, 0);

        if let Err(e) = image::save_buffer(
            &path,
            &frame.rgba8,
            frame.width,
            frame.height,
            image::ColorType::Rgba8,
        ) {
            log::error!("Failed to save screenshot: {}", e);
        } else {
            log::info!("Screenshot saved to {:?}", path);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn take_screenshot(&mut self) {
        log::warn!("Screenshot is not supported on web");
    }

    // ── Export ────────────────────────────────────────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_export(&mut self) {
        let path = match crate::desktop_dialogs::mesh_export_dialog() {
            FileDialogSelection::Selected(path) => path,
            FileDialogSelection::Unsupported => {
                push_platform_unsupported_toast(self, "Mesh export dialogs");
                return;
            }
            FileDialogSelection::Cancelled => {
                return;
            }
        };

        let scene_clone = self.doc.scene.clone();
        let bounds = self.doc.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let max_res = self.settings.max_export_resolution.max(16);
        let resolution = self.settings.export_resolution.clamp(16, max_res);
        let adaptive = self.settings.adaptive_export;
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);
        let (tx, rx) = std::sync::mpsc::channel();
        let wake = self.wake.clone();

        std::thread::spawn(move || {
            let mesh = crate::export::marching_cubes(
                &scene_clone,
                resolution,
                bounds_min,
                bounds_max,
                &progress_clone,
                adaptive,
                &cancelled_clone,
            );
            let _ = tx.send(mesh);
            wake.wake();
        });

        let total = (resolution + 1) + resolution;
        self.async_state.export_status = ExportStatus::InProgress {
            progress,
            total,
            resolution,
            receiver: rx,
            path,
            cancelled,
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn start_export(&mut self) {
        let bounds = self.doc.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let max_res = self.settings.max_export_resolution.max(16);
        let resolution = self.settings.export_resolution.clamp(16, max_res);
        let progress = Arc::new(AtomicU32::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));

        let mesh = crate::export::marching_cubes(
            &self.doc.scene,
            resolution,
            bounds_min,
            bounds_max,
            &progress,
            self.settings.adaptive_export,
            &cancelled,
        );

        let Some(mesh) = mesh else {
            return;
        };

        let mut buf = std::io::Cursor::new(Vec::new());
        if let Err(e) = crate::export::write_obj_to(&mesh, &mut buf) {
            log::error!("Export failed: {}", e);
            return;
        }

        let msg = format!(
            "Exported OBJ ({} verts, {} tris)",
            mesh.vertices.len(),
            mesh.triangles.len(),
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
        let result =
            if let ExportStatus::InProgress { ref receiver, .. } = self.async_state.export_status {
                receiver.try_recv().ok()
            } else {
                None
            };

        let Some(maybe_mesh) = result else {
            return;
        };

        let path = if let ExportStatus::InProgress { ref path, .. } = self.async_state.export_status
        {
            path.clone()
        } else {
            unreachable!()
        };

        match maybe_mesh {
            Some(mesh) => match crate::export::write_mesh(&mesh, &path) {
                Ok(()) => {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("obj")
                        .to_uppercase();
                    let msg = format!(
                        "Exported {} ({} verts, {} tris)",
                        ext,
                        mesh.vertices.len(),
                        mesh.triangles.len(),
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
            },
            None => {
                // Export was cancelled
                self.ui.toasts.push(super::Toast {
                    message: "Export cancelled".into(),
                    is_error: false,
                    created: crate::compat::Instant::now(),
                    duration: crate::compat::Duration::from_secs(3),
                });
            }
        }

        self.async_state.export_status = ExportStatus::Idle;
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn poll_export(&mut self) {
        // Export runs synchronously on WASM — nothing to poll.
    }

    // ── Import Mesh ──────────────────────────────────────────────────────

    /// Open a file picker, load the mesh, and show the import settings dialog.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn open_import_dialog(&mut self) {
        let path = match crate::desktop_dialogs::mesh_import_dialog() {
            FileDialogSelection::Selected(path) => path,
            FileDialogSelection::Unsupported => {
                push_platform_unsupported_toast(self, "Mesh import dialogs");
                return;
            }
            FileDialogSelection::Cancelled => {
                return;
            }
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

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("mesh")
            .to_string();

        let max_res = self.settings.max_sculpt_resolution;
        self.ui.import_dialog = Some(super::state::ImportDialog::new(mesh, filename, max_res));
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn open_import_dialog(&mut self) {
        log::warn!("Mesh import is not supported on web");
    }

    /// Start the voxelization thread with the user-chosen resolution.
    /// Called after the user confirms the import dialog.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn start_import_voxelize(&mut self, resolution: u32) {
        let Some(dialog) = self.ui.import_dialog.take() else {
            return;
        };

        let mesh = dialog.mesh;
        let filename = dialog.filename;
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);
        let (tx, rx) = std::sync::mpsc::channel();
        let wake = self.wake.clone();

        std::thread::spawn(move || {
            let result = crate::mesh_import::mesh_to_sdf(&mesh, resolution, &progress_clone);
            if cancelled_clone.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            let _ = tx.send(result);
            wake.wake();
        });

        self.async_state.import_status = ImportStatus::InProgress {
            progress,
            total: resolution,
            filename,
            receiver: rx,
            cancelled,
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn start_import_voxelize(&mut self, _resolution: u32) {
        log::warn!("Mesh import is not supported on web");
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn poll_import(&mut self) {
        // Check if cancelled — if so, reset immediately without waiting for result
        if let ImportStatus::InProgress {
            ref cancelled,
            ref receiver,
            ..
        } = self.async_state.import_status
        {
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                // Try to drain the channel (thread may have already sent)
                let _ = receiver.try_recv();
                self.ui.toasts.push(super::Toast {
                    message: "Import cancelled".into(),
                    is_error: false,
                    created: crate::compat::Instant::now(),
                    duration: crate::compat::Duration::from_secs(3),
                });
                self.async_state.import_status = ImportStatus::Idle;
                return;
            }
        }

        let completed =
            if let ImportStatus::InProgress { ref receiver, .. } = self.async_state.import_status {
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
                    material: MaterialParams::with_base_color(Vec3::new(0.7, 0.7, 0.7)),
                    layer_intensity: 1.0,
                    voxel_grid: grid,
                    desired_resolution,
                },
            );
            self.ui.selection.select_single(sculpt_id);
            self.ui.scene_graph_view.needs_initial_rebuild = true;
            self.doc
                .sculpt_state
                .activate_preserving_session(sculpt_id, Some(self.scene_avg_extent()));
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
