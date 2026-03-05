mod action_handler;
pub(crate) mod actions;
mod async_tasks;
mod gpu_sync;
mod input;
mod sculpting;
pub(crate) mod state;
mod ui_panels;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use crate::compat::{Duration, Instant};

use eframe::egui;
use eframe::wgpu;
use glam::Vec3;

use crate::gpu::buffers;
use crate::gpu::codegen;
use crate::graph::scene::NodeId;
use crate::graph::voxel;
use crate::sculpt::{ActiveTool, SculptState};
use crate::settings::Settings;
use crate::ui::dock::{self, SceneTreeContext, SdfTabViewer, ViewportContext};
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::ViewportResources;

use state::{
    AsyncState, DocumentState, GizmoContext, GpuSyncState, PerfState, PersistenceState, UiState,
};

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
#[derive(Debug, Clone)]
pub struct BakeRequest {
    pub subtree_root: NodeId,
    pub resolution: u32,
    pub color: Vec3,
    /// If Some, update this existing sculpt node. If None, create a new one above subtree_root.
    pub existing_sculpt: Option<NodeId>,
    /// If true, replace the entire subtree with a standalone Sculpt node (destructive flatten).
    pub flatten: bool,
}

pub(super) enum BakeStatus {
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

pub enum ExportStatus {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        /// Total progress steps = (resolution+1) sample slices + resolution cell slices
        total: u32,
        receiver: std::sync::mpsc::Receiver<crate::export::ExportMesh>,
        path: std::path::PathBuf,
    },
}

pub(super) enum ImportStatus {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        total: u32,
        receiver: std::sync::mpsc::Receiver<(crate::graph::voxel::VoxelGrid, glam::Vec3)>,
    },
}

pub struct Toast {
    pub message: String,
    pub is_error: bool,
    pub created: Instant,
    pub duration: Duration,
}

/// Async pick state for sculpt mode (1-frame delay, eliminates GPU stall).
pub(super) enum PickState {
    Idle,
    Pending {
        receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

// ---------------------------------------------------------------------------
// Main app struct — decomposed into cohesive sub-structs.
// ---------------------------------------------------------------------------

pub struct SdfApp {
    /// Core document state: scene, camera, history, tools.
    pub(super) doc: DocumentState,
    /// Gizmo interaction state.
    pub(super) gizmo: GizmoContext,
    /// GPU synchronization state.
    pub(super) gpu: GpuSyncState,
    /// Async task tracking (bake, export, pick).
    pub(super) async_state: AsyncState,
    /// UI-only state: layout, dialogs, toasts.
    pub(super) ui: UiState,
    /// File persistence state.
    pub(super) persistence: PersistenceState,
    /// Performance / profiling state.
    pub(super) perf: PerfState,
    /// Application settings (render quality, export, etc.).
    pub(super) settings: Settings,
    /// Initial vsync state (preserved across settings changes).
    pub(super) initial_vsync: bool,
    /// Last frame timestamp for delta calculation.
    pub(super) last_time: f64,
}

impl SdfApp {
    pub fn new(cc: &eframe::CreationContext<'_>, settings: Settings) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");

        let scene = crate::graph::scene::Scene::new();
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
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&scene);
        let node_data = buffers::build_node_buffer(&scene, None, &voxel_offsets);
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

        let initial_vsync = settings.vsync_enabled;

        Self {
            doc: DocumentState {
                camera: crate::gpu::camera::Camera::default(),
                scene,
                history: crate::graph::history::History::new(),
                active_tool: ActiveTool::default(),
                sculpt_state: SculptState::Inactive,
                sculpt_history: crate::sculpt_history::SculptHistory::new(),
                clipboard_node: None,
            },
            gizmo: GizmoContext {
                state: GizmoState::Idle,
                mode: GizmoMode::Translate,
                space: GizmoSpace::Local,
                pivot_offset: Vec3::ZERO,
                last_selection: None,
            },
            gpu: GpuSyncState {
                render_state,
                current_structure_key: structure_key,
                buffer_dirty: false, // initial upload already done above
                last_data_fingerprint: 0,
                voxel_gpu_offsets: voxel_offsets,
                sculpt_tex_indices: HashMap::new(),
            },
            async_state: AsyncState {
                bake_status: BakeStatus::Idle,
                export_status: ExportStatus::Idle,
                import_status: ImportStatus::Idle,
                pick_state: PickState::Idle,
                pending_pick: None,
                last_sculpt_hit: None,
                lazy_brush_pos: None,
                sculpt_ctrl_held: false,
                sculpt_shift_held: false,
                sculpt_pressure: 0.0,
                hover_world_pos: None,
                cursor_over_geometry: false,
                sculpt_dragging: false,
            },
            ui: UiState {
                dock_state: dock::create_dock_state(),
                node_graph_state: NodeGraphState::new(),
                show_debug: false,
                show_help: false,
                show_export_dialog: false,
                show_settings: false,
                renaming_node: None,
                rename_buf: String::new(),
                scene_tree_drag: None,
                scene_tree_search: String::new(),
                isolation_state: None,
                toasts: Vec::new(),
                turntable_active: false,
                property_clipboard: None,
                command_palette_open: false,
                command_palette_query: String::new(),
                command_palette_selected: 0,
                sculpt_convert_dialog: None,
            },
            persistence: PersistenceState {
                current_file_path: None,
                scene_dirty: false,
                saved_fingerprint: 0,
                last_auto_save: Instant::now(),
            },
            perf: PerfState {
                timings: FrameTimings::new(),
                resolution_upgrade_pending: false,
                composite_full_update_needed: false,
            },
            settings,
            initial_vsync,
            last_time: 0.0,
        }
    }
}

impl eframe::App for SdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── 1. Frame setup ─────────────────────────────────────────────
        let frame_start = Instant::now();

        let now = ctx.input(|i| i.time);
        let dt = now - self.last_time;
        self.last_time = now;
        self.perf.timings.push_frame(dt);

        // Tick camera view transition animation
        let camera_animating = self.doc.camera.tick_transition(dt);

        // Turntable auto-rotate
        if self.ui.turntable_active {
            self.doc.camera.yaw += dt as f32 * 0.5; // ~0.5 rad/s
        }

        self.doc.history
            .begin_frame(&self.doc.scene, self.ui.node_graph_state.selected);

        // ── 2. Async polling ───────────────────────────────────────────
        self.sync_sculpt_state();
        self.poll_async_bake();
        self.poll_export();
        self.poll_import();
        self.poll_sculpt_pick();

        // Detect sculpt drag end: LMB released while sculpt_dragging was true.
        // Must happen here (before draw) because hover picks immediately fill
        // pending_pick, preventing reset_sculpt_stroke_if_idle from ever firing.
        if self.async_state.sculpt_dragging
            && !ctx.input(|i| i.pointer.primary_down())
        {
            if self.async_state.last_sculpt_hit.is_some() {
                self.doc.sculpt_history.end_stroke();
            }
            self.async_state.last_sculpt_hit = None;
            self.async_state.lazy_brush_pos = None;
            self.async_state.sculpt_dragging = false;
            self.async_state.cursor_over_geometry = false;
            if let SculptState::Active {
                ref mut flatten_reference,
                ref mut grab_snapshot,
                ref mut grab_start,
                ref mut grab_child_input,
                ..
            } = self.doc.sculpt_state
            {
                *flatten_reference = None;
                *grab_snapshot = None;
                *grab_start = None;
                *grab_child_input = None;
            }
        }

        // Reset pivot when selection changes
        let current_sel = self.ui.node_graph_state.selected;
        if current_sel != self.gizmo.last_selection {
            self.gizmo.pivot_offset = Vec3::ZERO;
            self.gizmo.last_selection = current_sel;
        }

        // ── 3. GPU pipeline sync ───────────────────────────────────────
        let t0 = Instant::now();
        self.sync_gpu_pipeline();
        self.perf.timings.pipeline_sync_s = t0.elapsed().as_secs_f64();

        self.process_pending_pick();

        // ── 4. Collect actions from keyboard + UI drawing ──────────────
        let mut action_sink = actions::ActionSink::new();
        self.collect_keyboard_actions(ctx, &mut action_sink);

        self.show_menu_bar(ctx, &mut action_sink);
        self.show_status_bar(ctx);
        crate::ui::help::draw(ctx, &mut self.ui.show_help);
        crate::ui::profiler::draw(
            ctx,
            self.ui.show_debug,
            &self.perf.timings,
            &self.doc.scene,
            &self.gpu,
            &self.settings,
            &self.doc.camera,
        );
        crate::ui::toasts::draw(ctx, &mut self.ui.toasts);

        // Sculpt convert dialog
        crate::ui::sculpt_convert_dialog::draw(
            ctx,
            &mut self.ui.sculpt_convert_dialog,
            &mut action_sink,
        );

        // Export dialog — acts on result
        match crate::ui::export_dialog::draw(
            ctx,
            &mut self.ui.show_export_dialog,
            &mut self.settings,
            &self.async_state.export_status,
        ) {
            crate::ui::export_dialog::ExportDialogResult::Export => {
                self.settings.save();
                self.start_export(ctx);
            }
            _ => {}
        }

        // Unified settings window
        crate::ui::settings_window::draw(
            ctx,
            &mut self.ui.show_settings,
            &mut self.settings,
            &mut self.ui.show_debug,
            self.initial_vsync,
            &mut action_sink,
        );

        let t_ui = Instant::now();
        let bake_progress = match &self.async_state.bake_status {
            BakeStatus::InProgress { progress, total, .. } => {
                Some((progress.load(Ordering::Relaxed), *total))
            }
            BakeStatus::Idle => None,
        };

        let mut pending_pick = None;
        let mut sculpt_ctrl_held = false;
        let mut sculpt_shift_held = false;
        let mut sculpt_pressure: f32 = 0.0;
        let mut is_hover_pick = false;
        let sculpt_count = self.gpu.sculpt_tex_indices.len();
        let isolation_label: Option<String> = self.ui.isolation_state.as_ref().and_then(|iso| {
            self.doc.scene.nodes.get(&iso.isolated_node).map(|n| n.name.clone())
        });
        let fps_info = if self.settings.show_fps_overlay {
            Some((self.perf.timings.avg_fps, self.perf.timings.avg_frame_ms))
        } else {
            None
        };
        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.doc.camera,
            scene: &mut self.doc.scene,
            node_graph_state: &mut self.ui.node_graph_state,
            active_tool: &self.doc.active_tool,
            sculpt_state: &mut self.doc.sculpt_state,
            settings: &mut self.settings,
            time: now as f32,
            bake_progress,
            viewport: ViewportContext {
                gizmo_state: &mut self.gizmo.state,
                gizmo_mode: &self.gizmo.mode,
                gizmo_space: &self.gizmo.space,
                pivot_offset: &mut self.gizmo.pivot_offset,
                pending_pick: &mut pending_pick,
                sculpt_count,
                fps_info,
                sculpt_ctrl_held: &mut sculpt_ctrl_held,
                sculpt_shift_held: &mut sculpt_shift_held,
                sculpt_pressure: &mut sculpt_pressure,
                last_sculpt_hit: self.async_state.last_sculpt_hit,
                isolation_label: isolation_label.clone(),
                turntable_active: self.ui.turntable_active,
                is_hover_pick: &mut is_hover_pick,
                hover_world_pos: self.async_state.hover_world_pos,
                cursor_over_geometry: self.async_state.cursor_over_geometry,
            },
            scene_tree: SceneTreeContext {
                renaming_node: &mut self.ui.renaming_node,
                rename_buf: &mut self.ui.rename_buf,
                drag_state: &mut self.ui.scene_tree_drag,
                search_filter: &mut self.ui.scene_tree_search,
            },
            actions: &mut action_sink,
            history: &self.doc.history,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.ui.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        // Command palette (drawn after dock, on top of everything)
        crate::ui::command_palette::draw(
            ctx,
            &mut self.ui.command_palette_open,
            &mut self.ui.command_palette_query,
            &mut self.ui.command_palette_selected,
            &self.doc.scene,
            &mut action_sink,
        );

        self.perf.timings.ui_draw_s = t_ui.elapsed().as_secs_f64();

        // ── 5. Process all collected actions (single mutation point) ───
        self.process_actions(action_sink, ctx);

        // ── 6. Post-action cleanup ─────────────────────────────────────

        // Defensive: if a node was deleted via any UI panel, clean up state
        if let Some(sel) = self.ui.node_graph_state.selected {
            if !self.doc.scene.nodes.contains_key(&sel) {
                self.ui.node_graph_state.selected = None;
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.doc.sculpt_state = SculptState::Inactive;
                self.gpu.buffer_dirty = true;
            }
        }

        if pending_pick.is_some() {
            self.async_state.pending_pick = pending_pick;
            self.async_state.sculpt_dragging = !is_hover_pick;
            if !is_hover_pick {
                // Only copy modifier keys for sculpt drag picks
                self.async_state.sculpt_ctrl_held = sculpt_ctrl_held;
                self.async_state.sculpt_shift_held = sculpt_shift_held;
                self.async_state.sculpt_pressure = sculpt_pressure;
            }
        }

        // Sculpt mode: submit async pick for next frame's poll_sculpt_pick
        self.submit_sculpt_pick();

        // Reset stroke interpolation and flatten reference when mouse is released during sculpting
        self.reset_sculpt_stroke_if_idle();

        // ── 7. GPU upload + dirty tracking ─────────────────────────────

        // Detect UI-driven scene data changes via lightweight fingerprint
        let fp = self.doc.scene.data_fingerprint();
        if fp != self.gpu.last_data_fingerprint {
            self.gpu.last_data_fingerprint = fp;
            self.gpu.buffer_dirty = true;
        }

        // Track unsaved changes and update window title
        let now_dirty = fp != self.persistence.saved_fingerprint
            || self.doc.scene.structure_key() != 0 && self.persistence.saved_fingerprint == 0 && !self.doc.scene.nodes.is_empty();
        if now_dirty != self.persistence.scene_dirty {
            self.persistence.scene_dirty = now_dirty;
            let title = if now_dirty { "SDF Modeler *" } else { "SDF Modeler" };
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.into()));
        }

        // Auto-save
        #[cfg(not(target_arch = "wasm32"))]
        if self.settings.auto_save_enabled
            && self.persistence.scene_dirty
            && self.persistence.last_auto_save.elapsed() >= Duration::from_secs(self.settings.auto_save_interval_secs as u64)
        {
            self.persistence.last_auto_save = Instant::now();
            let path = self.persistence.current_file_path.clone().unwrap_or_else(crate::io::auto_save_path);
            if let Err(e) = crate::io::save_project(&self.doc.scene, &self.doc.camera, &path) {
                log::error!("Auto-save failed: {}", e);
            } else {
                log::info!("Auto-saved to {}", path.display());
            }
        }

        // Upload GPU buffers only when scene data actually changed
        let t_upload = Instant::now();
        if self.gpu.buffer_dirty {
            self.upload_scene_buffer();
            self.gpu.buffer_dirty = false;
            // Composite volume needs full update when scene buffer changes
            if self.settings.render.composite_volume_enabled {
                self.perf.composite_full_update_needed = true;
            }
        }
        self.perf.timings.buffer_upload_s = t_upload.elapsed().as_secs_f64();

        // Dispatch composite volume update after buffers are uploaded
        let t_comp = Instant::now();
        if self.perf.composite_full_update_needed {
            self.dispatch_composite_full();
            self.perf.composite_full_update_needed = false;
        }
        self.perf.timings.composite_dispatch_s = t_comp.elapsed().as_secs_f64();

        // ── 8. Finalize ────────────────────────────────────────────────
        let is_dragging = ctx.dragged_id().is_some();
        self.doc.history.end_frame(
            &self.doc.scene,
            self.ui.node_graph_state.selected,
            is_dragging,
        );

        // Resolution upgrade: when interaction stops, request one more frame at full res
        if is_dragging || self.doc.sculpt_state.is_active() {
            self.perf.resolution_upgrade_pending = true;
        } else if self.perf.resolution_upgrade_pending {
            self.perf.resolution_upgrade_pending = false;
            ctx.request_repaint();
        }

        // Only repaint when something needs updating (saves GPU when idle)
        let needs_repaint = is_dragging
            || camera_animating
            || self.ui.turntable_active
            || self.doc.sculpt_state.is_active()
            || !matches!(self.async_state.bake_status, BakeStatus::Idle)
            || !matches!(self.async_state.export_status, ExportStatus::Idle)
            || !matches!(self.async_state.import_status, ImportStatus::Idle)
            || !matches!(self.async_state.pick_state, PickState::Idle)
            || self.async_state.pending_pick.is_some()
            || self.gpu.buffer_dirty
            || self.settings.continuous_repaint;
        if needs_repaint {
            ctx.request_repaint();
        }

        self.perf.timings.total_cpu_s = frame_start.elapsed().as_secs_f64();
    }
}
