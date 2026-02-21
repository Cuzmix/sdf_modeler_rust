use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui_dock::DockState;
use glam::Vec3;

use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::graph::voxel;
use crate::sculpt::{self, BrushMode, SculptState};
use crate::settings::Settings;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::gizmo::{GizmoMode, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::ViewportResources;

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
}

enum BakeStatus {
    Idle,
    InProgress {
        /// Existing sculpt node to update, or None to create new above subtree_root.
        existing_sculpt: Option<NodeId>,
        /// The subtree root (used when creating a new sculpt node).
        subtree_root: NodeId,
        color: Vec3,
        progress: Arc<AtomicU32>,
        total: u32,
        receiver: std::sync::mpsc::Receiver<(voxel::VoxelGrid, Vec3)>,
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
    sculpt_state: SculptState,
    settings: Settings,
    initial_vsync: bool,
    buffer_dirty: bool,
    last_data_fingerprint: u64,
    bake_status: BakeStatus,
    voxel_gpu_offsets: HashMap<NodeId, u32>,
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
            sculpt_state: SculptState::Inactive,
            initial_vsync: settings.vsync_enabled,
            settings,
            buffer_dirty: false, // initial upload already done above
            last_data_fingerprint: 0,
            bake_status: BakeStatus::Idle,
            voxel_gpu_offsets: voxel_offsets,
        }
    }

    fn handle_keyboard_input(&mut self, ctx: &egui::Context) {
        // Debug toggle
        if ctx.input(|i| i.key_pressed(egui::Key::F3)) {
            self.show_debug = !self.show_debug;
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

        // Gizmo mode
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            self.gizmo_mode = GizmoMode::Translate;
        }

        // Sculpt brush mode shortcuts (when sculpt is active)
        if self.sculpt_state.is_active() {
            if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                if let SculptState::Active { ref mut brush_mode, .. } = self.sculpt_state {
                    *brush_mode = BrushMode::Add;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                if let SculptState::Active { ref mut brush_mode, .. } = self.sculpt_state {
                    *brush_mode = BrushMode::Carve;
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
            progress,
            total: resolution,
            receiver: rx,
        };
    }

    fn poll_async_bake(&mut self) {
        let completed = if let BakeStatus::InProgress { ref receiver, .. } = self.bake_status {
            receiver.try_recv().ok()
        } else {
            None
        };

        if let Some((grid, center)) = completed {
            // Extract fields before replacing status
            let (existing_sculpt, subtree_root, color) = match &self.bake_status {
                BakeStatus::InProgress {
                    existing_sculpt, subtree_root, color, ..
                } => (*existing_sculpt, *subtree_root, *color),
                _ => unreachable!(),
            };

            if let Some(sculpt_id) = existing_sculpt {
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
                self.sculpt_state = SculptState::Active {
                    node_id: sculpt_id,
                    brush_mode: BrushMode::Add,
                    brush_radius: sculpt::DEFAULT_BRUSH_RADIUS,
                    brush_strength: sculpt::DEFAULT_BRUSH_STRENGTH,
                };
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

    fn sync_gpu_pipeline(&mut self) {
        let new_key = self.scene.structure_key();
        if new_key != self.current_structure_key {
            let shader_src = codegen::generate_shader(&self.scene, &self.settings.render);
            let pick_shader_src = codegen::generate_pick_shader(&self.scene, &self.settings.render);
            let mut renderer = self.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                res.rebuild_pipeline(&self.render_state.device, &shader_src, &pick_shader_src);
            }
            self.current_structure_key = new_key;
            self.buffer_dirty = true; // new pipeline needs fresh buffer data
        }
    }

    fn upload_scene_buffer(&mut self) {
        let (voxel_data, voxel_offsets) = codegen::build_voxel_buffer(&self.scene);
        let node_data =
            codegen::build_node_buffer(&self.scene, self.node_graph_state.selected, &voxel_offsets);
        self.voxel_gpu_offsets = voxel_offsets;
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
        }
    }

    fn process_pending_pick(&mut self) {
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

                // Sculpt mode: apply brush if we hit the sculpted node
                if let SculptState::Active {
                    node_id,
                    ref brush_mode,
                    brush_radius,
                    brush_strength,
                } = self.sculpt_state
                {
                    if hit_node_id == node_id {
                        let hit_world = Vec3::new(
                            result.world_pos[0],
                            result.world_pos[1],
                            result.world_pos[2],
                        );
                        let dirty_range = sculpt::apply_brush(
                            &mut self.scene,
                            node_id,
                            hit_world,
                            brush_mode,
                            brush_radius,
                            brush_strength,
                        );
                        // Try incremental upload; fall back to full upload
                        let did_incremental = if let Some((z0, z1)) = dirty_range {
                            self.try_incremental_voxel_upload(node_id, z0, z1)
                        } else {
                            false
                        };
                        if !did_incremental {
                            self.buffer_dirty = true;
                        }
                    }
                    // Don't change selection in sculpt mode
                } else {
                    self.node_graph_state.selected = Some(hit_node_id);
                    self.buffer_dirty = true; // selection state encoded in GPU buffer
                }
            }
        } else if !self.sculpt_state.is_active() {
            // Clicked empty space — deselect (only in normal mode)
            self.node_graph_state.selected = None;
            self.buffer_dirty = true;
        }
    }

    fn sync_sculpt_state(&mut self) {
        // Auto-deactivate sculpt when selection changes away from sculpted node
        if let Some(active_node) = self.sculpt_state.active_node() {
            if self.node_graph_state.selected != Some(active_node) {
                self.sculpt_state = SculptState::Inactive;
            }
            // Also deactivate if the node is no longer a Sculpt node
            if let Some(node) = self.scene.nodes.get(&active_node) {
                if !matches!(node.data, NodeData::Sculpt { .. }) {
                    self.sculpt_state = SculptState::Inactive;
                }
            }
        }
    }

    fn show_debug_window(&self, ctx: &egui::Context, dt: f64) {
        if !self.show_debug {
            return;
        }
        egui::Window::new("Debug")
            .default_pos([10.0, 10.0])
            .default_size([220.0, 160.0])
            .show(ctx, |ui| {
                let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                ui.label(format!("FPS: {:.0}", fps));
                ui.separator();
                let eye = self.camera.eye();
                ui.label(format!("Eye: ({:.2}, {:.2}, {:.2})", eye.x, eye.y, eye.z));
                ui.label(format!(
                    "Target: ({:.2}, {:.2}, {:.2})",
                    self.camera.target.x, self.camera.target.y, self.camera.target.z
                ));
                ui.label(format!("Distance: {:.2}", self.camera.distance));
                ui.label(format!(
                    "Yaw: {:.1}\u{00B0}  Pitch: {:.1}\u{00B0}",
                    self.camera.yaw.to_degrees(),
                    self.camera.pitch.to_degrees(),
                ));
                ui.separator();
                ui.label(format!("Nodes: {}", self.scene.nodes.len()));
                ui.label(format!("Top-level: {}", self.scene.top_level_nodes().len()));
            });
    }

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
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
            });
        });
    }
}

impl eframe::App for SdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = ctx.input(|i| i.time);
        let dt = now - self.last_time;
        self.last_time = now;

        self.history
            .begin_frame(&self.scene, self.node_graph_state.selected);

        self.handle_keyboard_input(ctx);
        self.sync_sculpt_state();
        self.poll_async_bake();
        self.sync_gpu_pipeline();
        self.process_pending_pick();

        // --- UI ---
        self.show_debug_window(ctx, dt);
        self.show_menu_bar(ctx);

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
        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.camera,
            scene: &mut self.scene,
            node_graph_state: &mut self.node_graph_state,
            gizmo_state: &mut self.gizmo_state,
            gizmo_mode: &self.gizmo_mode,
            sculpt_state: &mut self.sculpt_state,
            settings: &mut self.settings,
            settings_dirty: &mut settings_dirty,
            time: now as f32,
            pending_pick: &mut pending_pick,
            bake_request: &mut bake_request,
            bake_progress,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        if pending_pick.is_some() {
            self.pending_pick = pending_pick;
        }

        // Start async bake if requested by UI
        if let Some(req) = bake_request {
            if !baking {
                self.start_async_bake(req, ctx);
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

        // Upload GPU buffers only when scene data actually changed
        if self.buffer_dirty {
            self.upload_scene_buffer();
            self.buffer_dirty = false;
        }

        // Undo/Redo: end-of-frame commit
        let is_dragging = ctx.dragged_id().is_some();
        self.history.end_frame(
            &self.scene,
            self.node_graph_state.selected,
            is_dragging,
        );

        ctx.request_repaint();
    }
}
