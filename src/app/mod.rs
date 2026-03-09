mod action_handler;
pub(crate) mod actions;
mod async_tasks;
mod gpu_sync;
mod input;
mod sculpting;
pub(crate) mod state;
mod ui_panels;

use crate::compat::{Duration, Instant};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

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

    /// Sculpt interaction counters (updated every frame).
    pub sculpt_brush_samples: u32,
    pub sculpt_gpu_dispatches: u32,
    pub sculpt_gpu_submits: u32,
    pub sculpt_pick_latency_ms: f64,
    pub sculpt_pick_latency_avg_ms: f64,

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
            sculpt_brush_samples: 0,
            sculpt_gpu_dispatches: 0,
            sculpt_gpu_submits: 0,
            sculpt_pick_latency_ms: 0.0,
            sculpt_pick_latency_avg_ms: 0.0,
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

    fn begin_frame(&mut self) {
        self.sculpt_brush_samples = 0;
        self.sculpt_gpu_dispatches = 0;
        self.sculpt_gpu_submits = 0;
    }

    fn record_sculpt_brush_batch(
        &mut self,
        sample_count: u32,
        dispatch_count: u32,
        submit_count: u32,
    ) {
        self.sculpt_brush_samples = self.sculpt_brush_samples.saturating_add(sample_count);
        self.sculpt_gpu_dispatches = self.sculpt_gpu_dispatches.saturating_add(dispatch_count);
        self.sculpt_gpu_submits = self.sculpt_gpu_submits.saturating_add(submit_count);
    }

    fn record_sculpt_pick_latency(&mut self, latency_ms: f64) {
        self.sculpt_pick_latency_ms = latency_ms;
        let alpha = 0.2;
        self.sculpt_pick_latency_avg_ms =
            self.sculpt_pick_latency_avg_ms * (1.0 - alpha) + latency_ms * alpha;
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
        resolution: u32,
        receiver: std::sync::mpsc::Receiver<Option<crate::export::ExportMesh>>,
        path: std::path::PathBuf,
        cancelled: Arc<AtomicBool>,
    },
}

pub enum ImportStatus {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        total: u32,
        filename: String,
        receiver: std::sync::mpsc::Receiver<(crate::graph::voxel::VoxelGrid, glam::Vec3)>,
        cancelled: Arc<AtomicBool>,
    },
}

pub struct Toast {
    pub message: String,
    pub is_error: bool,
    pub created: Instant,
    pub duration: Duration,
}

/// Minimal data needed to reconstruct a cursor ray for sculpt interaction.
#[derive(Clone, Copy)]
pub(super) struct PickRayInputs {
    pub mouse_pos: [f32; 2],
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 3],
    pub viewport_size: [f32; 2],
}

/// Async pick state for sculpt mode (1-frame delay, eliminates GPU stall).
pub(super) enum PickState {
    Idle,
    Pending {
        receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        ray_inputs: PickRayInputs,
        submitted_at: Instant,
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
    /// Material preset library (built-in + user-saved).
    pub(super) material_library: crate::material_preset::MaterialLibrary,
    /// Initial vsync state (preserved across settings changes).
    pub(super) initial_vsync: bool,
    /// Last frame timestamp for delta calculation.
    pub(super) last_time: f64,
}

impl SdfApp {
    #[cfg(not(target_arch = "wasm32"))]
    fn recovery_summary_from_meta(meta: Option<&crate::io::RecoveryMeta>) -> String {
        let timestamp = meta.map(|m| m.autosave_unix_secs).unwrap_or(0);
        let project_hint = meta
            .and_then(|m| m.project_path.as_deref())
            .map(|path| format!("\nSource project: {path}"))
            .unwrap_or_default();
        format!("Recovered unsaved work found from UNIX timestamp {timestamp}.{project_hint}")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn recover_from_autosave(&mut self) -> bool {
        let autosave_path = crate::io::auto_save_path();
        match crate::io::load_project(&autosave_path) {
            Ok(project) => {
                self.doc.scene = project.scene;
                self.doc.camera = project.camera;
                self.doc.history = crate::graph::history::History::new();
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.doc.sculpt_state = SculptState::Inactive;
                self.gpu.current_structure_key = 0;
                self.gpu.buffer_dirty = true;
                self.persistence.current_file_path = None;
                self.persistence.saved_fingerprint = 0;
                self.persistence.scene_dirty = true;
                true
            }
            Err(error) => {
                log::error!("Failed to recover autosave: {}", error);
                self.ui.toasts.push(Toast {
                    message: "Failed to recover autosave".into(),
                    is_error: true,
                    created: crate::compat::Instant::now(),
                    duration: crate::compat::Duration::from_secs(5),
                });
                false
            }
        }
    }

    pub fn new(cc: &eframe::CreationContext<'_>, settings: Settings) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");

        #[cfg(not(target_arch = "wasm32"))]
        let recovery_meta = crate::io::read_recovery_meta();
        #[cfg(not(target_arch = "wasm32"))]
        let show_recovery_dialog = !settings.last_clean_exit && crate::io::has_recovery_file();
        #[cfg(not(target_arch = "wasm32"))]
        let recovery_summary = Self::recovery_summary_from_meta(recovery_meta.as_ref());
        #[cfg(target_arch = "wasm32")]
        let show_recovery_dialog = false;
        #[cfg(target_arch = "wasm32")]
        let recovery_summary = String::new();

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
        let empty_selection = std::collections::HashSet::new();
        let node_data = buffers::build_node_buffer(&scene, &empty_selection, &voxel_offsets);
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

        let mut settings = settings;
        let initial_vsync = settings.vsync_enabled;
        #[cfg(not(target_arch = "wasm32"))]
        {
            settings.last_clean_exit = false;
            settings.save();
        }

        Self {
            doc: DocumentState {
                camera: crate::gpu::camera::Camera::default(),
                scene,
                history: crate::graph::history::History::new(),
                active_tool: ActiveTool::default(),
                sculpt_state: SculptState::Inactive,
                sculpt_history: crate::sculpt_history::SculptHistory::new(),
                clipboard_node: None,
                soloed_light: None,
            },
            gizmo: GizmoContext {
                state: GizmoState::Idle,
                mode: GizmoMode::Translate,
                space: GizmoSpace::Local,
                pivot_offset: Vec3::ZERO,
                last_selection: None,
                gizmo_visible: true,
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
                sculpt_runtime_cache: None,
            },
            ui: UiState {
                dock_state: dock::create_dock_state(),
                node_graph_state: NodeGraphState::new(),
                light_graph_state: NodeGraphState::new(),
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
                import_dialog: None,
                show_quick_toolbar: false,
                rebinding_action: None,
                active_light_ids: std::collections::HashSet::new(),
                total_light_count: 0,
                last_light_warning_count: None,
                show_recovery_dialog,
                recovery_summary,
                reference_images: crate::ui::reference_image::ReferenceImageManager::default(),
                show_distance_readout: false,
                measurement_mode: false,
                measurement_points: Vec::new(),
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
            material_library: crate::material_preset::MaterialLibrary::load(),
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
        self.perf.timings.begin_frame();

        // Tick camera view transition animation
        let camera_animating = self.doc.camera.tick_transition(dt);

        // Turntable auto-rotate
        if self.ui.turntable_active {
            self.doc.camera.yaw += dt as f32 * 0.5; // ~0.5 rad/s
        }

        self.doc
            .history
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
        if self.async_state.sculpt_dragging && !ctx.input(|i| i.pointer.primary_down()) {
            if self.async_state.last_sculpt_hit.is_some() {
                self.doc.sculpt_history.end_stroke();
            }
            self.async_state.last_sculpt_hit = None;
            self.async_state.lazy_brush_pos = None;
            self.async_state.sculpt_runtime_cache = None;
            self.async_state.sculpt_dragging = false;
            self.async_state.cursor_over_geometry = false;
            if let SculptState::Active {
                ref mut flatten_reference,
                ref mut grab_snapshot,
                ref mut grab_analytical_snapshot,
                ref mut grab_start,
                ref mut grab_child_input,
                ..
            } = self.doc.sculpt_state
            {
                *flatten_reference = None;
                *grab_snapshot = None;
                *grab_analytical_snapshot = None;
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

        if self.ui.show_recovery_dialog {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(action) = crate::ui::recovery_dialog::draw(
                ctx,
                &mut self.ui.show_recovery_dialog,
                &self.ui.recovery_summary,
                !self.settings.recent_files.is_empty(),
            ) {
                match action {
                    crate::ui::recovery_dialog::RecoveryDialogAction::Recover => {
                        let _ = self.recover_from_autosave();
                    }
                    crate::ui::recovery_dialog::RecoveryDialogAction::Discard => {
                        if let Err(error) = crate::io::remove_recovery_files() {
                            log::error!("Failed to remove recovery files: {}", error);
                        }
                    }
                    crate::ui::recovery_dialog::RecoveryDialogAction::OpenLastProject => {
                        if let Some(path) = self.settings.recent_files.first() {
                            let open_path = std::path::PathBuf::from(path);
                            let _ = self.load_project_from_path(&open_path);
                        }
                    }
                }
            }
            ctx.request_repaint();
            return;
        }

        self.show_menu_bar(ctx, &mut action_sink);
        self.show_status_bar(ctx);
        crate::ui::help::draw(ctx, &mut self.ui.show_help, &self.settings.keymap);
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
        crate::ui::quick_toolbar::draw(ctx, &mut self.ui.show_quick_toolbar, &mut action_sink);

        // Sculpt convert dialog
        crate::ui::sculpt_convert_dialog::draw(
            ctx,
            &mut self.ui.sculpt_convert_dialog,
            &mut action_sink,
            self.settings.max_sculpt_resolution,
        );

        // Import mesh settings dialog
        crate::ui::import_dialog::draw(
            ctx,
            &mut self.ui.import_dialog,
            &mut action_sink,
            self.settings.max_sculpt_resolution,
        );

        // Export dialog — acts on result
        if let crate::ui::export_dialog::ExportDialogResult::Export = crate::ui::export_dialog::draw(
            ctx,
            &mut self.ui.show_export_dialog,
            &mut self.settings,
            &self.async_state.export_status,
        ) {
            self.settings.save();
            self.start_export(ctx);
        }

        // Export/Import progress modals (shown during active operations)
        crate::ui::export_progress::draw_export(ctx, &self.async_state.export_status);
        crate::ui::export_progress::draw_import(ctx, &self.async_state.import_status);

        // Unified settings window
        crate::ui::settings_window::draw(
            ctx,
            &mut self.ui.show_settings,
            &mut self.settings,
            &mut self.ui.show_debug,
            self.initial_vsync,
            &mut action_sink,
            &mut self.ui.rebinding_action,
        );

        let t_ui = Instant::now();
        let bake_progress = match &self.async_state.bake_status {
            BakeStatus::InProgress {
                progress, total, ..
            } => Some((progress.load(Ordering::Relaxed), *total)),
            BakeStatus::Idle => None,
        };

        let mut pending_pick = None;
        let mut sculpt_ctrl_held = false;
        let mut sculpt_shift_held = false;
        let mut sculpt_pressure: f32 = 0.0;
        let mut is_hover_pick = false;
        let sculpt_count = self.gpu.sculpt_tex_indices.len();
        let isolation_label: Option<String> = self.ui.isolation_state.as_ref().and_then(|iso| {
            self.doc
                .scene
                .nodes
                .get(&iso.isolated_node)
                .map(|n| n.name.clone())
        });
        let solo_label: Option<String> = self
            .doc
            .soloed_light
            .and_then(|id| self.doc.scene.nodes.get(&id).map(|n| n.name.clone()));
        let fps_info = if self.settings.show_fps_overlay {
            Some((self.perf.timings.avg_fps, self.perf.timings.avg_frame_ms))
        } else {
            None
        };
        // Compute active light set for this frame (used by scene tree + properties)
        {
            let (active_ids, total_count) =
                crate::gpu::buffers::identify_active_lights(&self.doc.scene, self.doc.camera.eye());
            self.ui.active_light_ids = active_ids;
            self.ui.total_light_count = total_count;

            // Toast warning when scene exceeds the light limit
            if total_count > crate::graph::scene::MAX_SCENE_LIGHTS {
                if self.ui.last_light_warning_count != Some(total_count) {
                    self.ui.last_light_warning_count = Some(total_count);
                    self.ui.toasts.push(Toast {
                        message: format!(
                            "Scene has {} lights — only the {} nearest to camera are active.",
                            total_count,
                            crate::graph::scene::MAX_SCENE_LIGHTS,
                        ),
                        is_error: false,
                        created: crate::compat::Instant::now(),
                        duration: std::time::Duration::from_secs(5),
                    });
                }
            } else {
                // Reset warning state when count drops below limit
                self.ui.last_light_warning_count = None;
            }
        }

        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.doc.camera,
            scene: &mut self.doc.scene,
            node_graph_state: &mut self.ui.node_graph_state,
            light_graph_state: &mut self.ui.light_graph_state,
            active_tool: &self.doc.active_tool,
            sculpt_state: &mut self.doc.sculpt_state,
            settings: &mut self.settings,
            time: now as f32,
            bake_progress,
            viewport: ViewportContext {
                gizmo_state: &mut self.gizmo.state,
                gizmo_mode: &self.gizmo.mode,
                gizmo_space: &self.gizmo.space,
                gizmo_visible: self.gizmo.gizmo_visible,
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
                soloed_light: self.doc.soloed_light,
                solo_label: solo_label.clone(),
                show_distance_readout: &mut self.ui.show_distance_readout,
                measurement_mode: &mut self.ui.measurement_mode,
                measurement_points: &mut self.ui.measurement_points,
            },
            scene_tree: SceneTreeContext {
                renaming_node: &mut self.ui.renaming_node,
                rename_buf: &mut self.ui.rename_buf,
                drag_state: &mut self.ui.scene_tree_drag,
                search_filter: &mut self.ui.scene_tree_search,
            },
            actions: &mut action_sink,
            history: &self.doc.history,
            active_light_ids: &self.ui.active_light_ids,
            material_library: &mut self.material_library,
            reference_images: &mut self.ui.reference_images,
            timings: &self.perf.timings,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.ui.dock_state).show_inside(ui, &mut tab_viewer);
            });

        // Command palette (drawn after dock, on top of everything)
        crate::ui::command_palette::draw(
            ctx,
            &mut self.ui.command_palette_open,
            &mut self.ui.command_palette_query,
            &mut self.ui.command_palette_selected,
            &self.doc.scene,
            &self.settings.keymap,
            &mut action_sink,
        );

        self.perf.timings.ui_draw_s = t_ui.elapsed().as_secs_f64();

        // ── 5. Process all collected actions (single mutation point) ───
        self.process_actions(action_sink, ctx);

        // ── 6. Post-action cleanup ─────────────────────────────────────

        // Defensive: if a node was deleted via any UI panel, clean up state
        if let Some(sel) = self.ui.node_graph_state.selected {
            if !self.doc.scene.nodes.contains_key(&sel) {
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.doc.sculpt_state = SculptState::Inactive;
                self.gpu.buffer_dirty = true;
            }
        }

        if let Some(ref pending) = pending_pick {
            self.async_state.sculpt_dragging = !is_hover_pick;
            if !is_hover_pick {
                // Only copy modifier keys for sculpt drag picks
                self.async_state.sculpt_ctrl_held = sculpt_ctrl_held;
                self.async_state.sculpt_shift_held = sculpt_shift_held;
                self.async_state.sculpt_pressure = sculpt_pressure;

                // Keep strokes smooth while async pick readback is still in flight.
                if matches!(self.async_state.pick_state, PickState::Pending { .. }) {
                    let _ = self.predict_sculpt_from_pending_pick(pending);
                }
            }
        }

        if pending_pick.is_some() {
            self.async_state.pending_pick = pending_pick;
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
            || self.doc.scene.structure_key() != 0
                && self.persistence.saved_fingerprint == 0
                && !self.doc.scene.nodes.is_empty();
        if now_dirty != self.persistence.scene_dirty {
            self.persistence.scene_dirty = now_dirty;
            let title = if now_dirty {
                "SDF Modeler *"
            } else {
                "SDF Modeler"
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.into()));
        }

        // Auto-save
        #[cfg(not(target_arch = "wasm32"))]
        if self.settings.auto_save_enabled
            && self.persistence.scene_dirty
            && self.persistence.last_auto_save.elapsed()
                >= Duration::from_secs(self.settings.auto_save_interval_secs as u64)
        {
            self.persistence.last_auto_save = Instant::now();
            let path = crate::io::auto_save_path();
            if let Err(e) = crate::io::save_project(&self.doc.scene, &self.doc.camera, &path) {
                log::error!("Auto-save failed: {}", e);
            } else {
                if let Err(e) =
                    crate::io::write_recovery_meta(self.persistence.current_file_path.as_deref())
                {
                    log::error!("Auto-save metadata write failed: {}", e);
                }
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
            || self.settings.continuous_repaint
            || self.doc.scene.has_light_expressions();
        if needs_repaint {
            ctx.request_repaint();
        }

        self.perf.timings.total_cpu_s = frame_start.elapsed().as_secs_f64();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn on_exit(&mut self) {
        self.settings.last_clean_exit = true;
        self.settings.save();
    }
}
