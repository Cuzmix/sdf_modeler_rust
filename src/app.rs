use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use egui_dock::DockState;
use glam::Vec3;

use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::{PendingPick, PickResult};
use crate::graph::history::History;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel;
use crate::sculpt::{self, BrushMode, FalloffMode, SculptState, DEFAULT_BRUSH_STRENGTH};
use crate::settings::Settings;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::{BrushDispatch, BrushGpuParams, ViewportResources};

// ---------------------------------------------------------------------------
// Frame timing / profiling
// ---------------------------------------------------------------------------

const TIMING_HISTORY_LEN: usize = 120;

pub struct FrameTimings {
    /// Per-phase CPU timings (current frame), in seconds.
    pub pipeline_sync_s: f64,
    pub buffer_upload_s: f64,
    pub composite_dispatch_s: f64,
    pub ui_draw_s: f64,
    pub total_cpu_s: f64,

    /// Smoothed (EMA) values for display stability.
    pub avg_frame_ms: f64,
    pub avg_fps: f64,

    /// Rolling history for sparkline (frame time in ms).
    pub history: Vec<f32>,
    pub history_idx: usize,
}

impl FrameTimings {
    fn new() -> Self {
        Self {
            pipeline_sync_s: 0.0,
            buffer_upload_s: 0.0,
            composite_dispatch_s: 0.0,
            ui_draw_s: 0.0,
            total_cpu_s: 0.0,
            avg_frame_ms: 16.0,
            avg_fps: 60.0,
            history: vec![0.0; TIMING_HISTORY_LEN],
            history_idx: 0,
        }
    }

    fn push_frame(&mut self, dt_s: f64) {
        let dt_ms = dt_s * 1000.0;
        // EMA smoothing (alpha ~0.1 for ~10-frame averaging)
        let alpha = 0.1;
        self.avg_frame_ms = self.avg_frame_ms * (1.0 - alpha) + dt_ms * alpha;
        self.avg_fps = if self.avg_frame_ms > 0.0 {
            1000.0 / self.avg_frame_ms
        } else {
            0.0
        };

        // Ring buffer history
        self.history[self.history_idx] = dt_ms as f32;
        self.history_idx = (self.history_idx + 1) % TIMING_HISTORY_LEN;
    }
}

// ---------------------------------------------------------------------------
// Async bake types
// ---------------------------------------------------------------------------

/// Request emitted by UI to start an async bake.
pub struct BakeRequest {
    pub subtree_root: NodeId,
    pub resolution: u32,
    pub color: Vec3,
    /// If Some, update this existing sculpt node. If None, create a new one above subtree_root.
    pub existing_sculpt: Option<NodeId>,
    /// If true, replace the entire subtree with a standalone Sculpt node (destructive flatten).
    pub flatten: bool,
}

enum BakeStatus {
    Idle,
    InProgress {
        /// Existing sculpt node to update, or None to create new above subtree_root.
        existing_sculpt: Option<NodeId>,
        /// The subtree root (used when creating a new sculpt node).
        subtree_root: NodeId,
        color: Vec3,
        /// If true, replace entire subtree with standalone Sculpt (destructive flatten).
        flatten: bool,
        progress: Arc<AtomicU32>,
        total: u32,
        receiver: std::sync::mpsc::Receiver<(voxel::VoxelGrid, Vec3)>,
    },
}

enum ExportStatus {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        /// Total progress steps = (resolution+1) sample slices + resolution cell slices
        total: u32,
        receiver: std::sync::mpsc::Receiver<crate::export::ExportMesh>,
        path: std::path::PathBuf,
    },
}

/// Async pick state for sculpt mode (1-frame delay, eliminates GPU stall).
enum PickState {
    Idle,
    Pending {
        receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

pub struct SdfApp {
    camera: Camera,
    scene: Scene,
    dock_state: DockState<Tab>,
    node_graph_state: NodeGraphState,
    render_state: RenderState,
    current_structure_key: u64,
    history: History,
    last_time: f64,
    show_debug: bool,
    pending_pick: Option<PendingPick>,
    gizmo_state: GizmoState,
    gizmo_mode: GizmoMode,
    gizmo_space: GizmoSpace,
    pivot_offset: Vec3,
    last_gizmo_selection: Option<NodeId>,
    sculpt_state: SculptState,
    settings: Settings,
    initial_vsync: bool,
    buffer_dirty: bool,
    last_data_fingerprint: u64,
    bake_status: BakeStatus,
    voxel_gpu_offsets: HashMap<NodeId, u32>,
    /// Maps sculpt NodeId → texture index for voxel texture3D uploads.
    sculpt_tex_indices: HashMap<NodeId, usize>,
    /// Async pick state for sculpt mode.
    pick_state: PickState,
    /// Last brush hit for stroke interpolation.
    last_sculpt_hit: Option<Vec3>,
    /// Lazy brush smoothed position (None = first hit of stroke).
    lazy_brush_pos: Option<Vec3>,
    /// Async mesh export state.
    export_status: ExportStatus,
    /// When true, request one more repaint at full resolution after interaction stops.
    resolution_upgrade_pending: bool,
    /// When true, dispatch a full composite volume update next frame.
    composite_full_update_needed: bool,
    /// Frame profiling data.
    timings: FrameTimings,
    /// Node currently being renamed in scene tree.
    renaming_node: Option<NodeId>,
    /// Rename text buffer.
    rename_buf: String,
    /// Show keyboard shortcuts help window.
    show_help: bool,
    /// Scene has unsaved modifications.
    scene_dirty: bool,
    /// Fingerprint at last save/load (to detect unsaved changes).
    saved_fingerprint: u64,
}

impl SdfApp {
    pub fn new(cc: &eframe::CreationContext<'_>, settings: Settings) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");

        let scene = Scene::new();
        let shader_src = codegen::generate_shader(&scene, &settings.render);
        let pick_shader_src = codegen::generate_pick_shader(&scene, &settings.render);
        let structure_key = scene.structure_key();

        let resources = ViewportResources::new(
            &render_state.device,
            render_state.target_format,
            &shader_src,
            &pick_shader_src,
        );

        // Upload initial scene buffer
        let (voxel_data, voxel_offsets) = codegen::build_voxel_buffer(&scene);
        let node_data = codegen::build_node_buffer(&scene, None, &voxel_offsets);
        {
            let mut renderer = render_state.renderer.write();
            renderer.callback_resources.insert(resources);
        }
        {
            let mut renderer = render_state.renderer.write();
            let res = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
                .unwrap();
            res.update_scene_buffer(&render_state.device, &render_state.queue, &node_data);
            res.update_voxel_buffer(&render_state.device, &render_state.queue, &voxel_data);
        }

        Self {
            camera: Camera::default(),
            scene,
            dock_state: dock::create_dock_state(),
            node_graph_state: NodeGraphState::new(),
            render_state,
            current_structure_key: structure_key,
            history: History::new(),
            last_time: 0.0,
            show_debug: true,
            pending_pick: None,
            gizmo_state: GizmoState::Idle,
            gizmo_mode: GizmoMode::Translate,
            gizmo_space: GizmoSpace::Local,
            pivot_offset: Vec3::ZERO,
            last_gizmo_selection: None,
            sculpt_state: SculptState::Inactive,
            initial_vsync: settings.vsync_enabled,
            settings,
            buffer_dirty: false, // initial upload already done above
            last_data_fingerprint: 0,
            bake_status: BakeStatus::Idle,
            voxel_gpu_offsets: voxel_offsets,
            sculpt_tex_indices: HashMap::new(),
            pick_state: PickState::Idle,
            last_sculpt_hit: None,
            lazy_brush_pos: None,
            export_status: ExportStatus::Idle,
            resolution_upgrade_pending: false,
            composite_full_update_needed: false,
            timings: FrameTimings::new(),
            renaming_node: None,
            rename_buf: String::new(),
            show_help: false,
            scene_dirty: false,
            saved_fingerprint: 0,
        }
    }

    fn delete_selected(&mut self) {
        if let Some(sel) = self.node_graph_state.selected {
            self.scene.remove_node(sel);
            self.node_graph_state.selected = None;
            self.node_graph_state.layout_dirty = true;
            self.sculpt_state = SculptState::Inactive;
            self.buffer_dirty = true;
        }
    }

    fn handle_keyboard_input(&mut self, ctx: &egui::Context) {
        // Help
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_help = !self.show_help;
        }
        // Camera presets
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.camera.set_front();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F6)) {
            self.camera.set_top();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F7)) {
            self.camera.set_right();
        }

        // Debug toggle
        if ctx.input(|i| i.key_pressed(egui::Key::F4)) {
            self.show_debug = !self.show_debug;
        }

        // Delete selected node
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            self.delete_selected();
        }

        // Undo / Redo
        let undo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z));
        let redo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y));

        if undo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .undo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
                self.buffer_dirty = true;
            }
        } else if redo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .redo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
                self.buffer_dirty = true;
            }
        }

        // Save / Load
        let save_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S));
        let open_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O));

        if save_pressed {
            if let Some(path) = crate::io::save_dialog() {
                if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                    log::error!("Failed to save project: {}", e);
                }
            }
        } else if open_pressed {
            if let Some(path) = crate::io::open_dialog() {
                match crate::io::load_project(&path) {
                    Ok(project) => {
                        self.scene = project.scene;
                        self.camera = project.camera;
                        self.history = History::new();
                        self.node_graph_state.selected = None;
                        self.node_graph_state.layout_dirty = true;
                        self.sculpt_state = SculptState::Inactive;
                        self.current_structure_key = 0; // Force pipeline rebuild
                        self.buffer_dirty = true;
                    }
                    Err(e) => {
                        log::error!("Failed to load project: {}", e);
                    }
                }
            }
        }

        // Screenshot
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::P)) {
            self.take_screenshot();
        }

        // Export OBJ
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::E)) {
            if matches!(self.export_status, ExportStatus::Idle) {
                self.start_export(ctx);
            }
        }

        // Gizmo mode
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            self.gizmo_mode = GizmoMode::Translate;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::E) && !i.modifiers.ctrl) {
            self.gizmo_mode = GizmoMode::Rotate;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.gizmo_mode = GizmoMode::Scale;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::G)) {
            self.gizmo_space = match self.gizmo_space {
                GizmoSpace::Local => GizmoSpace::World,
                GizmoSpace::World => GizmoSpace::Local,
            };
        }
        if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::C)) {
            self.pivot_offset = Vec3::ZERO;
        }

        // Sculpt brush mode shortcuts (when sculpt is active)
        if self.sculpt_state.is_active() {
            if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Add;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Carve;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Smooth;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num4)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Flatten;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num5)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Inflate;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num6)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    *brush_mode = BrushMode::Grab;
                    if *brush_strength < 0.5 {
                        *brush_strength = 1.0;
                    }
                }
            }
            // Symmetry toggles: X/Y/Z
            if ctx.input(|i| i.key_pressed(egui::Key::X)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(0) { None } else { Some(0) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Y)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(1) { None } else { Some(1) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Z)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(2) { None } else { Some(2) };
                }
            }
        }
    }

    fn start_async_bake(&mut self, req: BakeRequest, ctx: &egui::Context) {
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
    fn apply_instant_displacement_bake(&mut self, req: BakeRequest) {
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

    fn take_screenshot(&self) {
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
        let uniform = self.camera.to_uniform(viewport, 0.0, 0.0, false, scene_bounds);

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

    fn start_export(&mut self, ctx: &egui::Context) {
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

    fn poll_export(&mut self) {
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

    fn poll_async_bake(&mut self) {
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

    fn try_incremental_voxel_upload(&self, node_id: NodeId, z0: u32, z1: u32) -> bool {
        let Some(&gpu_offset) = self.voxel_gpu_offsets.get(&node_id) else {
            return false;
        };
        let Some(node) = self.scene.nodes.get(&node_id) else {
            return false;
        };
        let NodeData::Sculpt { ref voxel_grid, .. } = node.data else {
            return false;
        };
        let renderer = self.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return false;
        };
        res.update_voxel_region(
            &self.render_state.queue,
            gpu_offset,
            voxel_grid.resolution,
            z0,
            z1,
            &voxel_grid.data,
        );
        true
    }

    /// Upload dirty z-slab region of a sculpt node's voxel data to its texture3D.
    fn upload_voxel_texture_region(&self, node_id: NodeId, z0: u32, z1: u32) {
        let Some(&tex_idx) = self.sculpt_tex_indices.get(&node_id) else {
            return;
        };
        let Some(node) = self.scene.nodes.get(&node_id) else {
            return;
        };
        let NodeData::Sculpt { ref voxel_grid, .. } = node.data else {
            return;
        };
        let renderer = self.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        res.upload_voxel_texture_region(
            &self.render_state.queue,
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

    /// Poll for a previously submitted async sculpt pick result.
    /// If ready: apply brush at the hit point (CPU + GPU).
    fn poll_sculpt_pick(&mut self) {
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
    fn handle_sculpt_hit(&mut self, result: PickResult) {
        let topo_order = self.scene.topo_order();
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
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    *grab_snapshot = Some(voxel_grid.data.clone());
                    *grab_start = Some(hit_world);
                }
            }
        }
        let grab_snap = grab_snapshot.clone();
        let grab_origin = *grab_start;

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

                if let Some(node) = self.scene.nodes.get_mut(&node_id) {
                    if let NodeData::Sculpt { ref mut voxel_grid, .. } = node.data {
                        let (z0, z1) = sculpt::apply_grab_to_grid(
                            voxel_grid,
                            snap,
                            local_center,
                            brush_radius,
                            brush_strength,
                            local_delta,
                            &falloff_mode,
                        );
                        self.try_incremental_voxel_upload(node_id, z0, z1);
                        self.upload_voxel_texture_region(node_id, z0, z1);
                    }
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
    fn interpolate_brush_hits(&self, current: Vec3, brush_radius: f32) -> Vec<Vec3> {
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
    fn dispatch_gpu_brush(
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
    fn submit_sculpt_pick(&mut self) {
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

    fn sync_gpu_pipeline(&mut self) {
        let new_key = self.scene.structure_key();
        if new_key != self.current_structure_key {
            let shader_src = codegen::generate_shader(&self.scene, &self.settings.render);
            let pick_shader_src = codegen::generate_pick_shader(&self.scene, &self.settings.render);
            let sculpt_count = codegen::collect_sculpt_tex_info(&self.scene).len();
            let mut renderer = self.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                // Always rebuild the normal + pick pipelines
                res.rebuild_pipeline(
                    &self.render_state.device, &shader_src, &pick_shader_src, sculpt_count,
                );

                // Rebuild composite pipelines if enabled and there are sculpt nodes
                if self.settings.render.composite_volume_enabled && sculpt_count > 0 {
                    let bounds = self.scene.compute_bounds();
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
                        &self.scene, &self.settings.render,
                    );
                    let comp_render_src = codegen::generate_composite_render_shader(
                        &self.settings.render, bounds_min, bounds_max,
                    );

                    res.rebuild_composite(
                        &self.render_state.device,
                        &comp_compute_src,
                        &comp_render_src,
                        resolution,
                        bounds_min,
                        bounds_max,
                    );
                    log::info!("Composite: pipelines built, use_composite={}", res.use_composite);
                    self.composite_full_update_needed = true;
                } else {
                    res.use_composite = false;
                    res.composite = None;
                }
            }
            self.current_structure_key = new_key;
            self.buffer_dirty = true; // new pipeline needs fresh buffer data
        }
    }

    fn upload_scene_buffer(&mut self) {
        let (voxel_data, voxel_offsets) = codegen::build_voxel_buffer(&self.scene);
        let node_data =
            codegen::build_node_buffer(&self.scene, self.node_graph_state.selected, &voxel_offsets);
        let sculpt_infos = codegen::collect_sculpt_tex_info(&self.scene);
        self.voxel_gpu_offsets = voxel_offsets;
        self.sculpt_tex_indices = sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();
        let mut renderer = self.render_state.renderer.write();
        if let Some(res) = renderer
            .callback_resources
            .get_mut::<ViewportResources>()
        {
            res.update_scene_buffer(
                &self.render_state.device,
                &self.render_state.queue,
                &node_data,
            );
            res.update_voxel_buffer(
                &self.render_state.device,
                &self.render_state.queue,
                &voxel_data,
            );
            // Upload voxel textures for render shader
            for info in &sculpt_infos {
                if let Some(node) = self.scene.nodes.get(&info.node_id) {
                    if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                        res.upload_voxel_texture(
                            &self.render_state.device,
                            &self.render_state.queue,
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
    fn dispatch_composite_full(&self) {
        let renderer = self.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else { return; };
        let Some(ref comp) = res.composite else { return; };
        let r = comp.resolution;
        res.dispatch_composite(
            &self.render_state.device,
            &self.render_state.queue,
            [0, 0, 0],
            [r - 1, r - 1, r - 1],
        );
    }

    /// Dispatch an incremental composite update for the brush-affected region.
    fn dispatch_composite_region(&self, center: Vec3, radius: f32) {
        let renderer = self.render_state.renderer.read();
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
            &self.render_state.device,
            &self.render_state.queue,
            umin,
            umax,
        );
    }

    fn process_pending_pick(&mut self) {
        // Sculpt mode uses async pick path (poll_sculpt_pick / submit_sculpt_pick)
        if self.sculpt_state.is_active() {
            return;
        }
        let Some(pending) = self.pending_pick.take() else {
            return;
        };
        let topo_order = self.scene.topo_order();
        let renderer = self.render_state.renderer.read();
        let Some(res) = renderer.callback_resources.get::<ViewportResources>() else {
            return;
        };
        if let Some(result) = res.execute_pick(
            &self.render_state.device,
            &self.render_state.queue,
            &pending,
        ) {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                let hit_node_id = topo_order[idx];
                self.node_graph_state.selected = Some(hit_node_id);
                self.buffer_dirty = true;
            }
        } else {
            // Clicked empty space — deselect
            self.node_graph_state.selected = None;
            self.buffer_dirty = true;
        }
    }

    fn sync_sculpt_state(&mut self) {
        // Auto-deactivate sculpt when selection changes away from sculpted node
        if let Some(active_node) = self.sculpt_state.active_node() {
            let mut deactivate = false;
            if self.node_graph_state.selected != Some(active_node) {
                deactivate = true;
            }
            // Also deactivate if the node is no longer a Sculpt node
            if let Some(node) = self.scene.nodes.get(&active_node) {
                if !matches!(node.data, NodeData::Sculpt { .. }) {
                    deactivate = true;
                }
            }
            if deactivate {
                self.sculpt_state = SculptState::Inactive;
                self.last_sculpt_hit = None;
                self.lazy_brush_pos = None;
                self.pick_state = PickState::Idle;
            }
        }
    }

    fn show_debug_window(&self, ctx: &egui::Context) {
        if !self.show_debug {
            return;
        }
        let t = &self.timings;
        egui::Window::new("Profiler")
            .default_pos([10.0, 10.0])
            .default_size([280.0, 320.0])
            .show(ctx, |ui| {
                // --- FPS / Frame time ---
                let color = if t.avg_fps >= 55.0 {
                    egui::Color32::from_rgb(100, 255, 100)
                } else if t.avg_fps >= 30.0 {
                    egui::Color32::from_rgb(255, 255, 100)
                } else {
                    egui::Color32::from_rgb(255, 100, 100)
                };
                ui.colored_label(color, format!(
                    "FPS: {:.0}  ({:.2} ms)", t.avg_fps, t.avg_frame_ms
                ));

                // --- Frame time sparkline ---
                let history_ordered: Vec<f32> = {
                    let idx = t.history_idx;
                    let mut v = Vec::with_capacity(TIMING_HISTORY_LEN);
                    v.extend_from_slice(&t.history[idx..]);
                    v.extend_from_slice(&t.history[..idx]);
                    v
                };
                let max_ms = history_ordered.iter().cloned().fold(1.0_f32, f32::max);
                let target_ms = 16.67_f32; // 60 FPS target line

                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 50.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));

                // Draw 60fps target line
                let target_y = rect.bottom() - (target_ms / max_ms) * rect.height();
                if target_y > rect.top() {
                    painter.line_segment(
                        [egui::pos2(rect.left(), target_y), egui::pos2(rect.right(), target_y)],
                        egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(100, 100, 255, 80)),
                    );
                }

                // Draw bars
                let bar_w = rect.width() / TIMING_HISTORY_LEN as f32;
                for (i, &ms) in history_ordered.iter().enumerate() {
                    let h = (ms / max_ms) * rect.height();
                    let x = rect.left() + i as f32 * bar_w;
                    let bar_color = if ms <= 16.67 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else if ms <= 33.33 {
                        egui::Color32::from_rgb(200, 200, 80)
                    } else {
                        egui::Color32::from_rgb(200, 80, 80)
                    };
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, rect.bottom() - h),
                            egui::vec2(bar_w.max(1.0), h),
                        ),
                        0.0,
                        bar_color,
                    );
                }

                ui.add_space(2.0);

                // --- CPU phase breakdown ---
                egui::CollapsingHeader::new("CPU Phases")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.monospace(format!("Pipeline sync:  {:6.2} ms", t.pipeline_sync_s * 1000.0));
                        ui.monospace(format!("Buffer upload:  {:6.2} ms", t.buffer_upload_s * 1000.0));
                        ui.monospace(format!("Comp dispatch:  {:6.2} ms", t.composite_dispatch_s * 1000.0));
                        ui.monospace(format!("UI draw:        {:6.2} ms", t.ui_draw_s * 1000.0));
                        ui.monospace(format!("Total CPU:      {:6.2} ms", t.total_cpu_s * 1000.0));
                    });

                ui.separator();

                // --- Scene stats ---
                egui::CollapsingHeader::new("Scene")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label(format!("Nodes: {}", self.scene.nodes.len()));
                        ui.label(format!("Top-level: {}", self.scene.top_level_nodes().len()));
                        ui.label(format!("Sculpt textures: {}", self.sculpt_tex_indices.len()));
                        ui.label(format!(
                            "Composite: {}",
                            if self.settings.render.composite_volume_enabled { "ON" } else { "OFF" }
                        ));
                    });

                // --- Render state ---
                egui::CollapsingHeader::new("Render State")
                    .default_open(true)
                    .show(ui, |ui| {
                        let renderer = self.render_state.renderer.read();
                        if let Some(res) = renderer.callback_resources.get::<ViewportResources>() {
                            ui.label(format!(
                                "Render size: {}x{}", res.render_width, res.render_height
                            ));
                            ui.label(format!("Composite active: {}", res.use_composite));
                        }
                    });

                // --- Camera ---
                egui::CollapsingHeader::new("Camera")
                    .default_open(false)
                    .show(ui, |ui| {
                        let eye = self.camera.eye();
                        ui.label(format!("Eye: ({:.2}, {:.2}, {:.2})", eye.x, eye.y, eye.z));
                        ui.label(format!("Distance: {:.2}", self.camera.distance));
                        ui.label(format!(
                            "Yaw: {:.1} Pitch: {:.1}",
                            self.camera.yaw.to_degrees(),
                            self.camera.pitch.to_degrees(),
                        ));
                    });
            });
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        let mut action_new = false;
        let mut action_open = false;
        let mut action_save = false;
        let mut action_screenshot = false;
        let mut action_export = false;
        let mut action_undo = false;
        let mut action_redo = false;
        let mut action_delete = false;

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // --- File ---
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene").clicked() {
                        action_new = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                        action_open = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Save As...").shortcut_text("Ctrl+S")).clicked() {
                        action_save = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Screenshot...").shortcut_text("Ctrl+P")).clicked() {
                        action_screenshot = true;
                        ui.close_menu();
                    }
                    let export_idle = matches!(self.export_status, ExportStatus::Idle);
                    if ui.add_enabled(export_idle, egui::Button::new("Export OBJ...").shortcut_text("Ctrl+E")).clicked() {
                        action_export = true;
                        ui.close_menu();
                    }
                });

                // --- Edit ---
                ui.menu_button("Edit", |ui| {
                    if ui.add(egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                        action_undo = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                        action_redo = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    let has_sel = self.node_graph_state.selected.is_some();
                    if ui.add_enabled(has_sel, egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                        action_delete = true;
                        ui.close_menu();
                    }
                });

                // --- View ---
                ui.menu_button("View", |ui| {
                    let profiler_label = if self.show_debug { "Hide Profiler" } else { "Show Profiler" };
                    if ui.add(egui::Button::new(profiler_label).shortcut_text("F4")).clicked() {
                        self.show_debug = !self.show_debug;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label("Camera Presets");
                    if ui.add(egui::Button::new("Front").shortcut_text("F5")).clicked() {
                        self.camera.set_front();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Top").shortcut_text("F6")).clicked() {
                        self.camera.set_top();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Right").shortcut_text("F7")).clicked() {
                        self.camera.set_right();
                        ui.close_menu();
                    }
                });

                // --- Help ---
                ui.menu_button("Help", |ui| {
                    if ui.add(egui::Button::new("Keyboard Shortcuts").shortcut_text("F1")).clicked() {
                        self.show_help = !self.show_help;
                        ui.close_menu();
                    }
                });

                // --- Settings ---
                ui.menu_button("Settings", |ui| {
                    let mut vsync = self.settings.vsync_enabled;
                    if ui.checkbox(&mut vsync, "VSync").changed() {
                        self.settings.vsync_enabled = vsync;
                        self.settings.save();
                    }
                    if self.settings.vsync_enabled != self.initial_vsync {
                        ui.weak("(restart required)");
                    }
                });

                // Progress indicators (right-aligned)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let ExportStatus::InProgress { ref progress, total, .. } = self.export_status {
                        let done = progress.load(Ordering::Relaxed);
                        let frac = done as f32 / total.max(1) as f32;
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("Exporting... {:.0}%", frac * 100.0))
                                .desired_width(200.0),
                        );
                    }
                    if let BakeStatus::InProgress { ref progress, total, .. } = self.bake_status {
                        let done = progress.load(Ordering::Relaxed);
                        let frac = done as f32 / total.max(1) as f32;
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("Baking... {:.0}%", frac * 100.0))
                                .desired_width(200.0),
                        );
                    }
                });
            });
        });

        // Process deferred menu actions
        if action_new {
            self.scene = Scene::new();
            self.history = History::new();
            self.node_graph_state.selected = None;
            self.node_graph_state.layout_dirty = true;
            self.sculpt_state = SculptState::Inactive;
            self.current_structure_key = 0;
            self.buffer_dirty = true;
            self.saved_fingerprint = self.scene.data_fingerprint();
            self.scene_dirty = false;
        }
        if action_open {
            if let Some(path) = crate::io::open_dialog() {
                match crate::io::load_project(&path) {
                    Ok(project) => {
                        self.scene = project.scene;
                        self.camera = project.camera;
                        self.history = History::new();
                        self.node_graph_state.selected = None;
                        self.node_graph_state.layout_dirty = true;
                        self.sculpt_state = SculptState::Inactive;
                        self.current_structure_key = 0;
                        self.buffer_dirty = true;
                        self.saved_fingerprint = self.scene.data_fingerprint();
                        self.scene_dirty = false;
                    }
                    Err(e) => log::error!("Failed to load project: {}", e),
                }
            }
        }
        if action_save {
            if let Some(path) = crate::io::save_dialog() {
                if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                    log::error!("Failed to save project: {}", e);
                } else {
                    self.saved_fingerprint = self.scene.data_fingerprint();
                    self.scene_dirty = false;
                }
            }
        }
        if action_screenshot {
            self.take_screenshot();
        }
        if action_export {
            self.start_export(ctx);
        }
        if action_undo {
            if let Some((restored_scene, restored_sel)) =
                self.history.undo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
                self.buffer_dirty = true;
            }
        }
        if action_redo {
            if let Some((restored_scene, restored_sel)) =
                self.history.redo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
                self.buffer_dirty = true;
            }
        }
        if action_delete {
            self.delete_selected();
        }
    }

    fn show_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help { return; }
        egui::Window::new("Keyboard Shortcuts")
            .open(&mut self.show_help)
            .resizable(false)
            .default_width(380.0)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let section = |ui: &mut egui::Ui, title: &str| {
                            ui.colored_label(egui::Color32::from_rgb(180, 200, 255), title);
                            ui.end_row();
                        };
                        let row = |ui: &mut egui::Ui, key: &str, desc: &str| {
                            ui.monospace(key);
                            ui.label(desc);
                            ui.end_row();
                        };

                        section(ui, "General");
                        row(ui, "Ctrl+O", "Open project");
                        row(ui, "Ctrl+S", "Save project");
                        row(ui, "Ctrl+Z", "Undo");
                        row(ui, "Ctrl+Y", "Redo");
                        row(ui, "Delete", "Delete selected node");
                        row(ui, "Ctrl+P", "Screenshot");
                        row(ui, "Ctrl+E", "Export OBJ");
                        row(ui, "F1", "Toggle this help");
                        row(ui, "F4", "Toggle profiler");

                        ui.separator(); ui.end_row();
                        section(ui, "Camera");
                        row(ui, "LMB drag", "Orbit");
                        row(ui, "RMB drag", "Pan");
                        row(ui, "Scroll", "Zoom");
                        row(ui, "F5", "Front view");
                        row(ui, "F6", "Top view");
                        row(ui, "F7", "Right view");

                        ui.separator(); ui.end_row();
                        section(ui, "Gizmo");
                        row(ui, "W", "Move tool");
                        row(ui, "E", "Rotate tool");
                        row(ui, "R", "Scale tool");
                        row(ui, "G", "Toggle Local / World");
                        row(ui, "Alt+Drag", "Move pivot");
                        row(ui, "Alt+C", "Reset pivot");

                        ui.separator(); ui.end_row();
                        section(ui, "Sculpt Mode");
                        row(ui, "LMB drag", "Paint brush");
                        row(ui, "RMB drag", "Pan camera");
                        row(ui, "MMB drag", "Orbit camera");
                        row(ui, "1", "Add brush");
                        row(ui, "2", "Carve brush");
                        row(ui, "3", "Smooth brush");
                        row(ui, "4", "Flatten brush");
                        row(ui, "5", "Inflate brush");
                        row(ui, "6", "Grab brush");
                        row(ui, "X / Y / Z", "Toggle symmetry axis");

                        ui.separator(); ui.end_row();
                        section(ui, "Scene Tree");
                        row(ui, "Double-click", "Rename node");
                        row(ui, "Right-click", "Context menu");
                    });
            });
    }

    fn show_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Mode indicator
                    match &self.sculpt_state {
                        SculptState::Active { brush_mode, symmetry_axis, .. } => {
                            let mode_name = match brush_mode {
                                BrushMode::Add => "Add",
                                BrushMode::Carve => "Carve",
                                BrushMode::Smooth => "Smooth",
                                BrushMode::Flatten => "Flatten",
                                BrushMode::Inflate => "Inflate",
                                BrushMode::Grab => "Grab",
                            };
                            ui.colored_label(
                                egui::Color32::from_rgb(180, 130, 255),
                                format!("Sculpt: {}", mode_name),
                            );
                            if let Some(axis) = symmetry_axis {
                                let axis_name = match axis {
                                    0 => "X",
                                    1 => "Y",
                                    _ => "Z",
                                };
                                ui.separator();
                                ui.label(format!("Sym: {}", axis_name));
                            }
                        }
                        SculptState::Inactive => {
                            ui.colored_label(
                                egui::Color32::from_rgb(130, 200, 255),
                                self.gizmo_mode.label(),
                            );
                            ui.separator();
                            ui.weak(self.gizmo_space.label());
                        }
                    }

                    ui.separator();

                    // Selection info
                    if let Some(sel) = self.node_graph_state.selected {
                        if let Some(node) = self.scene.nodes.get(&sel) {
                            ui.weak(&node.name);
                        }
                    }

                    // Right-aligned control hints
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.sculpt_state.is_active() {
                            ui.weak("LMB: Paint | RMB: Pan | MMB: Orbit | 1-6: Brush | X/Y/Z: Sym");
                        } else {
                            ui.weak("LMB: Orbit | RMB: Pan | Scroll: Zoom | Del: Delete");
                        }
                    });
                });
            });
    }
}
impl eframe::App for SdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();

        let now = ctx.input(|i| i.time);
        let dt = now - self.last_time;
        self.last_time = now;
        self.timings.push_frame(dt);

        self.history
            .begin_frame(&self.scene, self.node_graph_state.selected);

        self.handle_keyboard_input(ctx);

        // Reset pivot when selection changes
        let current_sel = self.node_graph_state.selected;
        if current_sel != self.last_gizmo_selection {
            self.pivot_offset = Vec3::ZERO;
            self.last_gizmo_selection = current_sel;
        }

        self.sync_sculpt_state();
        self.poll_async_bake();
        self.poll_export();
        self.poll_sculpt_pick(); // Read async pick result from previous frame

        let t0 = Instant::now();
        self.sync_gpu_pipeline();
        self.timings.pipeline_sync_s = t0.elapsed().as_secs_f64();

        self.process_pending_pick(); // Synchronous pick for normal (non-sculpt) mode

        // --- UI ---
        self.show_menu_bar(ctx);
        self.show_status_bar(ctx);
        self.show_help_window(ctx);
        self.show_debug_window(ctx);

        let t_ui = Instant::now();
        let baking = !matches!(self.bake_status, BakeStatus::Idle);
        let bake_progress = match &self.bake_status {
            BakeStatus::InProgress { progress, total, .. } => {
                Some((progress.load(Ordering::Relaxed), *total))
            }
            BakeStatus::Idle => None,
        };

        let mut pending_pick = None;
        let mut settings_dirty = false;
        let mut bake_request: Option<BakeRequest> = None;
        let sculpt_count = self.sculpt_tex_indices.len();
        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.camera,
            scene: &mut self.scene,
            node_graph_state: &mut self.node_graph_state,
            gizmo_state: &mut self.gizmo_state,
            gizmo_mode: &self.gizmo_mode,
            gizmo_space: &self.gizmo_space,
            pivot_offset: &mut self.pivot_offset,
            sculpt_state: &mut self.sculpt_state,
            settings: &mut self.settings,
            settings_dirty: &mut settings_dirty,
            time: now as f32,
            pending_pick: &mut pending_pick,
            bake_request: &mut bake_request,
            bake_progress,
            sculpt_count,
            renaming_node: &mut self.renaming_node,
            rename_buf: &mut self.rename_buf,
            fps_info: Some((self.timings.avg_fps, self.timings.avg_frame_ms)),
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        self.timings.ui_draw_s = t_ui.elapsed().as_secs_f64();

        // Defensive: if a node was deleted via any UI panel, clean up state
        if let Some(sel) = self.node_graph_state.selected {
            if !self.scene.nodes.contains_key(&sel) {
                self.node_graph_state.selected = None;
                self.node_graph_state.layout_dirty = true;
                self.sculpt_state = SculptState::Inactive;
                self.buffer_dirty = true;
            }
        }

        if pending_pick.is_some() {
            self.pending_pick = pending_pick;
        }

        // Sculpt mode: submit async pick for next frame's poll_sculpt_pick
        self.submit_sculpt_pick();

        // Reset stroke interpolation and flatten reference when mouse is released during sculpting
        if self.sculpt_state.is_active()
            && self.pending_pick.is_none()
            && matches!(self.pick_state, PickState::Idle)
        {
            self.last_sculpt_hit = None;
            self.lazy_brush_pos = None;
            if let SculptState::Active {
                ref mut flatten_reference,
                ref mut grab_snapshot,
                ref mut grab_start,
                ..
            } = self.sculpt_state
            {
                *flatten_reference = None;
                *grab_snapshot = None;
                *grab_start = None;
            }
        }

        // Start bake if requested by UI
        if let Some(req) = bake_request {
            if !baking {
                if req.flatten {
                    // Flatten: needs full SDF evaluation (async bake)
                    self.start_async_bake(req, ctx);
                } else {
                    // Differential SDF: instant displacement grid (no async needed)
                    self.apply_instant_displacement_bake(req);
                }
            }
        }

        if settings_dirty {
            self.current_structure_key = 0; // Force pipeline rebuild
            self.buffer_dirty = true;
        }

        // Detect UI-driven scene data changes via lightweight fingerprint
        let fp = self.scene.data_fingerprint();
        if fp != self.last_data_fingerprint {
            self.last_data_fingerprint = fp;
            self.buffer_dirty = true;
        }

        // Track unsaved changes and update window title
        let now_dirty = fp != self.saved_fingerprint
            || self.scene.structure_key() != 0 && self.saved_fingerprint == 0 && !self.scene.nodes.is_empty();
        if now_dirty != self.scene_dirty {
            self.scene_dirty = now_dirty;
            let title = if now_dirty { "SDF Modeler *" } else { "SDF Modeler" };
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.into()));
        }

        // Upload GPU buffers only when scene data actually changed
        let t_upload = Instant::now();
        if self.buffer_dirty {
            self.upload_scene_buffer();
            self.buffer_dirty = false;
            // Composite volume needs full update when scene buffer changes
            if self.settings.render.composite_volume_enabled {
                self.composite_full_update_needed = true;
            }
        }
        self.timings.buffer_upload_s = t_upload.elapsed().as_secs_f64();

        // Dispatch composite volume update after buffers are uploaded
        let t_comp = Instant::now();
        if self.composite_full_update_needed {
            self.dispatch_composite_full();
            self.composite_full_update_needed = false;
        }
        self.timings.composite_dispatch_s = t_comp.elapsed().as_secs_f64();

        // Undo/Redo: end-of-frame commit
        let is_dragging = ctx.dragged_id().is_some();
        self.history.end_frame(
            &self.scene,
            self.node_graph_state.selected,
            is_dragging,
        );

        // Resolution upgrade: when interaction stops, request one more frame at full res
        if is_dragging || self.sculpt_state.is_active() {
            self.resolution_upgrade_pending = true;
        } else if self.resolution_upgrade_pending {
            self.resolution_upgrade_pending = false;
            ctx.request_repaint();
        }

        // Only repaint when something needs updating (saves GPU when idle)
        let needs_repaint = is_dragging
            || self.sculpt_state.is_active()
            || !matches!(self.bake_status, BakeStatus::Idle)
            || !matches!(self.export_status, ExportStatus::Idle)
            || !matches!(self.pick_state, PickState::Idle)
            || self.pending_pick.is_some()
            || settings_dirty;
        if needs_repaint {
            ctx.request_repaint();
        }

        self.timings.total_cpu_s = frame_start.elapsed().as_secs_f64();
    }
}
