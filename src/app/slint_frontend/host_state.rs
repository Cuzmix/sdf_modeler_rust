use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use slint::ComponentHandle;

use crate::app::actions::{Action, ActionSink};
use crate::app::backend_frame::{FrameCommands, ViewportUiFeedback};
use crate::app::frontend_models::{build_shell_snapshot, ShellSnapshot, ShellSnapshotInputs};
use crate::app::slint_bridge::{capture_frame_input, SlintViewportInputState};
use crate::app::{BakeStatus, ExportStatus, ImportStatus, PickState, SdfApp};
use crate::gizmo::{build_viewport_gizmo_overlay, GizmoInputSnapshot, ViewportGizmoOverlay};

use super::bindings::{apply_gizmo_overlay, apply_runtime_ui_state, apply_shell_snapshot};
use super::SlintHostWindow;

pub(super) struct NativeWgpuContext {
    pub(super) instance: wgpu::Instance,
    pub(super) render_context: crate::app::runtime::AppRenderContext,
}

pub(super) struct SlintViewportTexture {
    view: wgpu::TextureView,
    size: (u32, u32),
}

pub(super) struct TickOutcome {
    pub(super) request_redraw: bool,
    pub(super) needs_continuous_ticks: bool,
}

pub(super) struct SlintHostState {
    pub(super) app: SdfApp,
    frame_started_at: Instant,
    pub(super) queued_actions: ActionSink,
    pub(super) wake_flag: Arc<AtomicBool>,
    viewport_size: (u32, u32),
    pub(super) viewport_input: SlintViewportInputState,
    viewport_texture: Option<SlintViewportTexture>,
    pub(super) viewport_dirty: bool,
    pub(super) continuous_tick_active: bool,
    pub(super) last_snapshot: Option<ShellSnapshot>,
    last_gizmo_overlay: Option<ViewportGizmoOverlay>,
    last_window_title: String,
}

impl SlintHostState {
    pub(super) fn new(app: SdfApp, wake_flag: Arc<AtomicBool>) -> Self {
        Self {
            app,
            frame_started_at: Instant::now(),
            queued_actions: Vec::new(),
            wake_flag,
            viewport_size: (960, 540),
            viewport_input: SlintViewportInputState::default(),
            viewport_texture: None,
            viewport_dirty: true,
            continuous_tick_active: false,
            last_snapshot: None,
            last_gizmo_overlay: None,
            last_window_title: "SDF Modeler".to_string(),
        }
    }

    pub(super) fn queue_action(&mut self, action: Action) {
        self.queued_actions.push(action);
        self.viewport_dirty = true;
    }

    pub(super) fn tick(&mut self, window: &SlintHostWindow) -> TickOutcome {
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

        let snapshot = build_shell_snapshot(ShellSnapshotInputs {
            scene: &self.app.doc.scene,
            selection: &self.app.ui.selection,
            scene_filter_query: &self.app.ui.scene_tree_search,
            history: &self.app.doc.history,
            reference_images: &self.app.ui.reference_images,
            expert_panels: &self.app.ui.expert_panels,
            settings: &self.app.settings,
            sculpt_state: &self.app.doc.sculpt_state,
            interaction_mode: self.app.ui.primary_shell.interaction_mode,
            gizmo_mode: self.app.gizmo.mode.clone(),
            gizmo_space: self.app.gizmo.space.clone(),
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

    pub(super) fn render_viewport_if_needed(&mut self) {
        if !self.viewport_dirty {
            return;
        }
        let Some(texture) = self.viewport_texture.as_ref() else {
            return;
        };
        self.app.render_viewport_texture(
            &texture.view,
            self.viewport_size.0.max(1),
            self.viewport_size.1.max(1),
        );
        self.viewport_dirty = false;
    }

    pub(super) fn release_viewport_texture(&mut self) {
        self.viewport_texture = None;
        self.viewport_dirty = true;
    }

    fn ensure_viewport_texture(&mut self, window: &SlintHostWindow) {
        let width = self.viewport_size.0.max(1);
        let height = self.viewport_size.1.max(1);
        if self
            .viewport_texture
            .as_ref()
            .is_some_and(|texture| texture.size == (width, height))
        {
            return;
        }

        let texture = self
            .app
            .gpu
            .render_context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Slint Viewport Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.app.gpu.render_context.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image = match slint::Image::try_from(texture.clone()) {
            Ok(image) => image,
            Err(error) => {
                log::error!("Failed to import viewport texture into Slint: {error}");
                return;
            }
        };
        window.set_viewport_image(image.clone());
        self.viewport_texture = Some(SlintViewportTexture {
            view,
            size: (width, height),
        });
        self.viewport_dirty = true;
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

fn apply_brush_adjust_feedback(app: &mut SdfApp, feedback: &ViewportUiFeedback) {
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
