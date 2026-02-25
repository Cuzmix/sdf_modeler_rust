mod async_tasks;
mod gpu_sync;
mod input;
mod sculpting;
mod ui_panels;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use egui_dock::DockState;
use glam::Vec3;

use crate::gpu::buffers;
use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeId, Scene};
use crate::graph::voxel;
use crate::sculpt::SculptState;
use crate::settings::Settings;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::ViewportResources;

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

pub(super) enum ExportStatus {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        /// Total progress steps = (resolution+1) sample slices + resolution cell slices
        total: u32,
        receiver: std::sync::mpsc::Receiver<crate::export::ExportMesh>,
        path: std::path::PathBuf,
    },
}

pub(super) struct Toast {
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

pub struct SdfApp {
    pub(super) camera: Camera,
    pub(super) scene: Scene,
    pub(super) dock_state: DockState<Tab>,
    pub(super) node_graph_state: NodeGraphState,
    pub(super) render_state: RenderState,
    pub(super) current_structure_key: u64,
    pub(super) history: History,
    pub(super) last_time: f64,
    pub(super) show_debug: bool,
    pub(super) pending_pick: Option<PendingPick>,
    pub(super) gizmo_state: GizmoState,
    pub(super) gizmo_mode: GizmoMode,
    pub(super) gizmo_space: GizmoSpace,
    pub(super) pivot_offset: Vec3,
    pub(super) last_gizmo_selection: Option<NodeId>,
    pub(super) sculpt_state: SculptState,
    pub(super) settings: Settings,
    pub(super) initial_vsync: bool,
    pub(super) buffer_dirty: bool,
    pub(super) last_data_fingerprint: u64,
    pub(super) bake_status: BakeStatus,
    pub(super) voxel_gpu_offsets: HashMap<NodeId, u32>,
    /// Maps sculpt NodeId -> texture index for voxel texture3D uploads.
    pub(super) sculpt_tex_indices: HashMap<NodeId, usize>,
    /// Async pick state for sculpt mode.
    pub(super) pick_state: PickState,
    /// Last brush hit for stroke interpolation.
    pub(super) last_sculpt_hit: Option<Vec3>,
    /// Lazy brush smoothed position (None = first hit of stroke).
    pub(super) lazy_brush_pos: Option<Vec3>,
    /// Async mesh export state.
    pub(super) export_status: ExportStatus,
    /// When true, request one more repaint at full resolution after interaction stops.
    pub(super) resolution_upgrade_pending: bool,
    /// When true, dispatch a full composite volume update next frame.
    pub(super) composite_full_update_needed: bool,
    /// Frame profiling data.
    pub(super) timings: FrameTimings,
    /// Node currently being renamed in scene tree.
    pub(super) renaming_node: Option<NodeId>,
    /// Rename text buffer.
    pub(super) rename_buf: String,
    /// Show keyboard shortcuts help window.
    pub(super) show_help: bool,
    /// Scene has unsaved modifications.
    pub(super) scene_dirty: bool,
    /// Fingerprint at last save/load (to detect unsaved changes).
    pub(super) saved_fingerprint: u64,
    /// Toast notifications (success/error messages).
    pub(super) toasts: Vec<Toast>,
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
            toasts: Vec::new(),
        }
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
        self.show_toasts(ctx);

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
        let fps_info = if self.settings.show_fps_overlay {
            Some((self.timings.avg_fps, self.timings.avg_frame_ms))
        } else {
            None
        };
        let initial_vsync = self.initial_vsync;
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
            fps_info,
            show_debug: &mut self.show_debug,
            initial_vsync,
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
                ref mut grab_child_input,
                ..
            } = self.sculpt_state
            {
                *flatten_reference = None;
                *grab_snapshot = None;
                *grab_start = None;
                *grab_child_input = None;
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
            || settings_dirty
            || self.settings.continuous_repaint;
        if needs_repaint {
            ctx.request_repaint();
        }

        self.timings.total_cpu_s = frame_start.elapsed().as_secs_f64();
    }
}
