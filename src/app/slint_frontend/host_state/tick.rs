use std::sync::atomic::Ordering;

use slint::ComponentHandle;

use crate::app::backend_frame::{FrameCommands, ViewportUiFeedback};
use crate::app::frontend_models::{build_shell_snapshot, ShellSnapshotInputs};
use crate::app::slint_bridge::capture_frame_input;
use crate::app::{BakeStatus, ExportStatus, ImportStatus, PickState};
use crate::gizmo::{build_viewport_gizmo_overlay, GizmoInputSnapshot};

use super::{SlintHostState, TickOutcome};
use crate::app::slint_frontend::bindings::{
    apply_gizmo_overlay, apply_runtime_ui_state, apply_shell_snapshot,
};
use crate::app::slint_frontend::SlintHostWindow;

impl SlintHostState {
    pub(in crate::app::slint_frontend) fn tick(&mut self, window: &SlintHostWindow) -> TickOutcome {
        self.viewport_input.set_viewport_geometry(
            window.get_viewport_width(),
            window.get_viewport_height(),
            window.window().scale_factor(),
        );
        let next_viewport_size = (
            (window.get_viewport_width().max(1.0) * window.window().scale_factor()) as u32,
            (window.get_viewport_height().max(1.0) * window.window().scale_factor()) as u32,
        );
        let viewport_size_changed = next_viewport_size != self.viewport_size;
        self.viewport_size = next_viewport_size;
        if viewport_size_changed {
            self.viewport_dirty = true;
        }
        self.ensure_viewport_texture(window);

        let frame_now = self.frame_started_at.elapsed().as_secs_f64();
        let frame_input = capture_frame_input(frame_now, &self.viewport_input);
        let camera_animating = self.app.run_backend_pre_ui(&frame_input);
        let mut actions = std::mem::take(&mut self.queued_actions);
        let had_actions = !actions.is_empty();
        let viewport_input_snapshot = self.viewport_input.take_snapshot(frame_now);
        let viewport_feedback = self
            .app
            .run_viewport_interaction(&viewport_input_snapshot, &mut actions);
        let camera_changed = viewport_feedback.camera_changed;
        apply_brush_adjust_feedback(&mut self.app, &viewport_feedback);
        let had_interaction_actions = !actions.is_empty();
        self.app.process_actions(actions);
        let commands =
            self.app
                .run_backend_post_ui(&frame_input, camera_animating, viewport_feedback);

        if let Some(ref title) = commands.window_title {
            self.last_window_title = title.clone();
            window.set_window_title(self.last_window_title.clone().into());
        }

        let file_actions_enabled = !matches!(
            self.app.async_state.bake_status,
            BakeStatus::InProgress { .. }
        ) && !matches!(
            self.app.async_state.export_status,
            ExportStatus::InProgress { .. }
        ) && !matches!(
            self.app.async_state.import_status,
            ImportStatus::InProgress { .. }
        );

        let snapshot = build_shell_snapshot(ShellSnapshotInputs {
            scene: &self.app.doc.scene,
            selection: &self.app.ui.selection,
            scene_panel_ui: &self.app.ui.scene_panel,
            primary_shell: &self.app.ui.primary_shell,
            menu_ui: &self.app.ui.menu,
            panel_framework: &self.app.ui.panel_framework,
            node_graph_view: &self.app.ui.node_graph_view,
            viewport_size_logical: [window.get_viewport_width(), window.get_viewport_height()],
            safe_area_insets_logical: [
                window.get_safe_area_left(),
                window.get_safe_area_top(),
                window.get_safe_area_right(),
                window.get_safe_area_bottom(),
            ],
            virtual_keyboard_position_logical: [
                window.get_virtual_keyboard_x(),
                window.get_virtual_keyboard_y(),
            ],
            virtual_keyboard_size_logical: [
                window.get_virtual_keyboard_width(),
                window.get_virtual_keyboard_height(),
            ],
            workspace: &self.app.ui.workspace,
            history: &self.app.doc.history,
            reference_images: &self.app.ui.reference_images,
            expert_panels: &self.app.ui.expert_panels,
            file_actions_enabled,
            settings: &self.app.settings,
            sculpt_state: &self.app.doc.sculpt_state,
            interaction_mode: self.app.ui.primary_shell.interaction_mode,
            gizmo_mode: self.app.gizmo.mode.clone(),
            gizmo_space: self.app.gizmo.space.clone(),
            camera_is_ortho: self.app.doc.camera.orthographic,
            measurement_mode_active: self.app.ui.measurement_mode,
            turntable_active: self.app.ui.turntable_active,
            help_visible: self.app.ui.show_help,
            command_palette_visible: self.app.ui.command_palette_open,
            can_undo: self.app.doc.history.undo_count() > 0,
            can_redo: self.app.doc.history.redo_count() > 0,
            has_selection: !self.app.ui.selection.selected_set.is_empty(),
            has_clipboard_node: self.app.doc.clipboard_node.is_some(),
        });
        apply_runtime_ui_state(window, &self.app);

        let snapshot_changed = self.last_snapshot.as_ref() != Some(&snapshot);
        if snapshot_changed {
            apply_shell_snapshot(window, &snapshot);
            self.last_snapshot = Some(snapshot);
        }

        let gizmo_overlay = build_viewport_gizmo_overlay(
            &self.app.doc.camera,
            &self.app.doc.scene,
            self.app.ui.selection.selected,
            &self.app.ui.selection.selected_set,
            &self.app.gizmo.state,
            &self.app.gizmo.mode,
            &self.app.gizmo.space,
            self.app.gizmo.pivot_offset,
            viewport_input_snapshot.viewport_size_physical,
            &self.app.settings.selection_behavior,
            self.app.gizmo.gizmo_visible
                && !self.app.ui.measurement_mode
                && !self.app.doc.sculpt_state.is_active(),
            Some(&GizmoInputSnapshot {
                viewport_size_physical: viewport_input_snapshot.viewport_size_physical,
                pointer_inside: viewport_input_snapshot.pointer_inside,
                pointer_position_physical: viewport_input_snapshot.pointer_position_physical,
                pointer_delta_physical: viewport_input_snapshot.pointer_delta_physical,
                primary_down: viewport_input_snapshot.primary.down,
                primary_pressed: viewport_input_snapshot.primary.pressed,
                primary_released: viewport_input_snapshot.primary.released,
                modifiers: viewport_input_snapshot.modifiers,
            }),
        );
        let overlay_changed = self.last_gizmo_overlay.as_ref() != gizmo_overlay.as_ref();
        if overlay_changed {
            apply_gizmo_overlay(window, gizmo_overlay.as_ref());
            self.last_gizmo_overlay = gizmo_overlay;
        }

        let woke = self.wake_flag.swap(false, Ordering::Relaxed);
        self.viewport_dirty |= viewport_size_changed
            || had_actions
            || had_interaction_actions
            || camera_changed
            || commands.request_repaint
            || snapshot_changed
            || woke;

        TickOutcome {
            request_redraw: self.viewport_dirty || snapshot_changed || overlay_changed || woke,
            needs_continuous_ticks: self.needs_continuous_ticks(camera_animating, &commands),
        }
    }

    fn needs_continuous_ticks(&self, camera_animating: bool, commands: &FrameCommands) -> bool {
        camera_animating
            || commands.request_repaint
            || self.viewport_input.needs_continuous_ticks()
            || self.app.ui.turntable_active
            || self.app.async_state.sculpt_dragging
            || self.app.async_state.pending_pick.is_some()
            || matches!(self.app.async_state.pick_state, PickState::Pending { .. })
            || matches!(
                self.app.async_state.bake_status,
                BakeStatus::InProgress { .. }
            )
            || matches!(
                self.app.async_state.export_status,
                ExportStatus::InProgress { .. }
            )
            || matches!(
                self.app.async_state.import_status,
                ImportStatus::InProgress { .. }
            )
    }
}

fn apply_brush_adjust_feedback(app: &mut crate::app::SdfApp, feedback: &ViewportUiFeedback) {
    if (feedback.brush_radius_delta != 0.0 || feedback.brush_strength_delta != 0.0)
        && app.doc.sculpt_state.is_active()
    {
        let selected_mode = app.doc.sculpt_state.selected_brush();
        let profile = app.doc.sculpt_state.selected_profile_mut();
        profile.radius = (profile.radius + feedback.brush_radius_delta).clamp(0.05, 2.0);
        profile.strength += feedback.brush_strength_delta;
        profile.clamp_strength_for_mode(selected_mode);
    }
}
