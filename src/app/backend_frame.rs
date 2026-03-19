use crate::compat::{Duration, Instant};
use crate::gpu::picking::PendingPick;
use crate::sculpt::{BrushMode, SculptState};
use glam::Vec3;

use super::{BakeStatus, ExportStatus, ImportStatus, PickState, SdfApp};

#[derive(Clone, Copy, Debug)]
pub(super) struct FrameInputSnapshot {
    pub now_seconds: f64,
    pub pointer_primary_down: bool,
    pub is_dragging_ui: bool,
}

pub(super) struct UiFrameFeedback {
    pub pending_pick: Option<PendingPick>,
    pub sculpt_ctrl_held: bool,
    pub sculpt_shift_held: bool,
    pub sculpt_pressure: f32,
    pub is_hover_pick: bool,
}

#[derive(Debug, Default)]
pub(super) struct FrameCommands {
    pub request_repaint: bool,
    pub window_title: Option<String>,
}

impl SdfApp {
    pub(super) fn run_backend_pre_ui(&mut self, frame_input: &FrameInputSnapshot) -> bool {
        let dt = frame_input.now_seconds - self.last_time;
        self.last_time = frame_input.now_seconds;
        self.perf.timings.push_frame(dt);
        self.perf.timings.begin_frame();

        // Tick camera view transition animation.
        let camera_animating = self.doc.camera.tick_transition(dt);

        // Turntable auto-rotate.
        if self.ui.turntable_active {
            self.doc.camera.yaw += dt as f32 * 0.5;
        }

        self.doc
            .history
            .begin_frame(&self.doc.scene, self.ui.node_graph_state.selected);

        self.sync_sculpt_state();
        self.poll_async_bake();
        self.poll_export();
        self.poll_import();
        self.poll_sculpt_pick();

        // Detect sculpt drag end before the draw phase fills pending picks again.
        if self.async_state.sculpt_dragging && !frame_input.pointer_primary_down {
            if self.async_state.last_sculpt_hit.is_some() {
                self.doc.history.end_sculpt_stroke(&self.doc.scene);
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

        // Reset pivot when selection changes.
        let current_selection = self.ui.node_graph_state.selected;
        if current_selection != self.gizmo.last_selection {
            self.gizmo.pivot_offset = Vec3::ZERO;
            self.gizmo.last_selection = current_selection;
        }

        let pipeline_start = Instant::now();
        self.sync_gpu_pipeline();
        self.perf.timings.pipeline_sync_s = pipeline_start.elapsed().as_secs_f64();

        self.process_pending_pick();

        camera_animating
    }

    pub(super) fn run_backend_post_ui(
        &mut self,
        frame_input: &FrameInputSnapshot,
        camera_animating: bool,
        ui_feedback: UiFrameFeedback,
    ) -> FrameCommands {
        let mut commands = FrameCommands::default();

        // Defensive cleanup: clear stale selection after deletion.
        if let Some(selected_node) = self.ui.node_graph_state.selected {
            if !self.doc.scene.nodes.contains_key(&selected_node) {
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.doc.sculpt_state = SculptState::Inactive;
                self.gpu.buffer_dirty = true;
            }
        }

        if ui_feedback.pending_pick.is_some() {
            self.async_state.sculpt_dragging = !ui_feedback.is_hover_pick;
            if !ui_feedback.is_hover_pick {
                self.async_state.sculpt_ctrl_held = ui_feedback.sculpt_ctrl_held;
                self.async_state.sculpt_shift_held = ui_feedback.sculpt_shift_held;
                self.async_state.sculpt_pressure = ui_feedback.sculpt_pressure;
            }
        }

        let live_grab_drag = self.async_state.sculpt_dragging
            && matches!(
                self.doc.sculpt_state,
                SculptState::Active {
                    ref brush_mode,
                    grab_start: Some(_),
                    ..
                } if *brush_mode == BrushMode::Grab
            );

        if let Some(ref pending_pick) = ui_feedback.pending_pick {
            if !ui_feedback.is_hover_pick {
                // During active Grab drags, keep sculpting from live cursor rays.
                if live_grab_drag
                    || matches!(self.async_state.pick_state, PickState::Pending { .. })
                {
                    let _ = self.predict_sculpt_from_pending_pick(pending_pick);
                }
            }
        }

        if ui_feedback.pending_pick.is_some() && !live_grab_drag {
            self.async_state.pending_pick = ui_feedback.pending_pick;
        }

        if !live_grab_drag {
            self.submit_sculpt_pick();
        }

        self.reset_sculpt_stroke_if_idle();

        // Detect scene data changes and schedule buffer upload.
        let data_fingerprint = self.doc.scene.data_fingerprint();
        if data_fingerprint != self.gpu.last_data_fingerprint {
            self.gpu.last_data_fingerprint = data_fingerprint;
            self.gpu.buffer_dirty = true;
        }

        let scene_dirty = data_fingerprint != self.persistence.saved_fingerprint
            || self.doc.scene.structure_key() != 0
                && self.persistence.saved_fingerprint == 0
                && !self.doc.scene.nodes.is_empty();
        if scene_dirty != self.persistence.scene_dirty {
            self.persistence.scene_dirty = scene_dirty;
            commands.window_title = Some(if scene_dirty {
                "SDF Modeler *".to_string()
            } else {
                "SDF Modeler".to_string()
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.settings.auto_save_enabled
            && self.persistence.scene_dirty
            && self.persistence.last_auto_save.elapsed()
                >= Duration::from_secs(self.settings.auto_save_interval_secs as u64)
        {
            self.persistence.last_auto_save = Instant::now();
            let path = crate::io::auto_save_path();
            if let Err(error) = crate::io::save_project(
                &self.doc.scene,
                &self.doc.camera,
                &self.settings.render,
                &path,
            ) {
                log::error!("Auto-save failed: {}", error);
            } else {
                if let Err(error) =
                    crate::io::write_recovery_meta(self.persistence.current_file_path.as_deref())
                {
                    log::error!("Auto-save metadata write failed: {}", error);
                }
                log::info!("Auto-saved to {}", path.display());
            }
        }

        let upload_start = Instant::now();
        if self.gpu.buffer_dirty {
            self.upload_scene_buffer();
            self.gpu.buffer_dirty = false;
            if self.settings.render.composite_volume_enabled {
                self.perf.composite_full_update_needed = true;
            }
        }
        self.perf.timings.buffer_upload_s = upload_start.elapsed().as_secs_f64();

        let composite_start = Instant::now();
        if self.perf.composite_full_update_needed {
            self.dispatch_composite_full();
            self.perf.composite_full_update_needed = false;
        }
        self.perf.timings.composite_dispatch_s = composite_start.elapsed().as_secs_f64();

        let is_anything_dragged = frame_input.is_dragging_ui || self.async_state.sculpt_dragging;
        self.doc.history.end_frame(
            &self.doc.scene,
            self.ui.node_graph_state.selected,
            is_anything_dragged,
        );

        if is_anything_dragged || self.doc.sculpt_state.is_active() {
            self.perf.resolution_upgrade_pending = true;
        } else if self.perf.resolution_upgrade_pending {
            self.perf.resolution_upgrade_pending = false;
            commands.request_repaint = true;
        }

        let needs_repaint = is_anything_dragged
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
            commands.request_repaint = true;
        }

        commands
    }
}
