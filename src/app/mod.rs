mod action_handler;
pub(crate) mod actions;
mod async_tasks;
mod backend_frame;
mod egui_frontend;
mod frontend_bridge;
mod gpu_sync;
mod input;
mod sculpt_detail;
mod sculpting;
pub(crate) mod state;
mod ui_panels;

use crate::compat::{Duration, Instant};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32};
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
use crate::ui::dock::{self};
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
// Main app struct - decomposed into cohesive sub-structs.
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
    fn format_recovery_timestamp_in_time_zone(
        unix_secs: u64,
        time_zone: &jiff::tz::TimeZone,
    ) -> Option<String> {
        let unix_secs = i64::try_from(unix_secs).ok()?;
        let timestamp = jiff::Timestamp::from_second(unix_secs).ok()?;
        let zoned = timestamp.to_zoned(time_zone.clone());
        Some(zoned.strftime("%A, %B %d, %Y at %-I:%M %p %Z").to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn format_recovery_timestamp(unix_secs: u64) -> String {
        Self::format_recovery_timestamp_in_time_zone(unix_secs, &jiff::tz::TimeZone::system())
            .unwrap_or_else(|| "an unknown time".to_string())
    }

    #[cfg(all(not(target_arch = "wasm32"), test))]
    fn recovery_summary_from_meta_in_time_zone(
        meta: Option<&crate::io::RecoveryMeta>,
        time_zone: &jiff::tz::TimeZone,
    ) -> String {
        let recovered_at = meta
            .and_then(|m| {
                Self::format_recovery_timestamp_in_time_zone(m.autosave_unix_secs, time_zone)
            })
            .unwrap_or_else(|| "an unknown time".to_string());
        let project_hint = meta
            .and_then(|m| m.project_path.as_deref())
            .map(|path| format!("\nSource project: {path}"))
            .unwrap_or_default();
        format!("Recovered unsaved work found from {recovered_at}.{project_hint}")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn recovery_summary_from_meta(meta: Option<&crate::io::RecoveryMeta>) -> String {
        let recovered_at = meta
            .map(|m| Self::format_recovery_timestamp(m.autosave_unix_secs))
            .unwrap_or_else(|| "an unknown time".to_string());
        let project_hint = meta
            .and_then(|m| m.project_path.as_deref())
            .map(|path| format!("\nSource project: {path}"))
            .unwrap_or_default();
        format!("Recovered unsaved work found from {recovered_at}.{project_hint}")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn recover_from_autosave(&mut self) -> bool {
        let autosave_path = crate::io::auto_save_path();
        match crate::io::load_project(&autosave_path) {
            Ok(project) => {
                self.doc.scene = project.scene;
                self.doc.camera = project.camera;
                self.doc.sculpt_state = project
                    .sculpt_state
                    .map(|persisted| SculptState::from_persisted(persisted, &self.doc.scene))
                    .unwrap_or_else(SculptState::new_inactive);
                self.doc.active_tool = if self.doc.sculpt_state.is_active() {
                    ActiveTool::Sculpt
                } else {
                    ActiveTool::Select
                };
                if let Some(render_config) = project.render_config {
                    self.settings.render = render_config;
                    self.gpu.last_environment_fingerprint = 0;
                }
                self.doc.history = crate::graph::history::History::new();
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
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
            &render_state.adapter,
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
            res.rebuild_environment(&render_state.device, &render_state.queue, &settings.render);
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
                sculpt_state: SculptState::new_inactive(),
                clipboard_node: None,
                soloed_light: None,
            },
            gizmo: GizmoContext {
                state: GizmoState::Idle,
                mode: GizmoMode::Translate,
                space: GizmoSpace::Local,
                pivot_offset: Vec3::ZERO,
                last_selection_ids: Vec::new(),
                gizmo_visible: true,
            },
            gpu: GpuSyncState {
                render_state,
                current_structure_key: structure_key,
                buffer_dirty: false, // initial upload already done above
                last_data_fingerprint: 0,
                last_environment_fingerprint: settings.render.environment_fingerprint(),
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
                sculpt_brush_adjust: None,
                show_distance_readout: false,
                measurement_mode: false,
                measurement_points: Vec::new(),
                multi_transform_edit: crate::app::state::MultiTransformSessionState::default(),
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
        let frame_start = Instant::now();
        let frame_input = frontend_bridge::capture_frame_input(ctx);
        let camera_animating = self.run_backend_pre_ui(&frame_input);

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

        let ui_feedback = self.draw_egui_frontend(ctx, frame_input.now_seconds, &mut action_sink);

        self.process_actions(action_sink, ctx);

        let frame_commands = self.run_backend_post_ui(&frame_input, camera_animating, ui_feedback);
        frontend_bridge::apply_frame_commands(ctx, &frame_commands);

        self.perf.timings.total_cpu_s = frame_start.elapsed().as_secs_f64();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn on_exit(&mut self) {
        self.settings.last_clean_exit = true;
        self.settings.save();
    }
}

#[cfg(test)]
mod tests {
    use super::SdfApp;

    #[test]
    fn format_recovery_timestamp_uses_human_readable_local_style() {
        let time_zone = jiff::tz::TimeZone::get("US/Eastern").unwrap();
        assert_eq!(
            SdfApp::format_recovery_timestamp_in_time_zone(1_777_386_018, &time_zone).as_deref(),
            Some("Tuesday, April 28, 2026 at 10:20 AM EDT")
        );
    }

    #[test]
    fn recovery_summary_uses_human_readable_time() {
        let time_zone = jiff::tz::TimeZone::get("US/Eastern").unwrap();
        let meta = crate::io::RecoveryMeta {
            autosave_unix_secs: 1_777_386_018,
            project_path: Some("C:\\projects\\dragon.sdf".to_string()),
            last_project_save_unix_secs: None,
        };

        assert_eq!(
            SdfApp::recovery_summary_from_meta_in_time_zone(Some(&meta), &time_zone),
            "Recovered unsaved work found from Tuesday, April 28, 2026 at 10:20 AM EDT.\nSource project: C:\\projects\\dragon.sdf"
        );
    }
}
