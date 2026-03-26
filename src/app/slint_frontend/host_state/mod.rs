use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use crate::app::actions::{Action, ActionSink};
use crate::app::frontend_models::ShellSnapshot;
use crate::app::slint_bridge::SlintViewportInputState;
use crate::app::SdfApp;
use crate::gizmo::ViewportGizmoOverlay;

mod tick;
mod viewport_texture;

pub(super) struct NativeWgpuContext {
    pub(super) instance: wgpu::Instance,
    pub(super) render_context: crate::app::runtime::AppRenderContext,
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
    viewport_texture: Option<viewport_texture::SlintViewportTexture>,
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
}
