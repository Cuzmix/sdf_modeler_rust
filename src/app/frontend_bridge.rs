use eframe::egui;

use super::backend_frame::{FrameCommands, FrameInputSnapshot};

pub(super) fn capture_frame_input(ctx: &egui::Context) -> FrameInputSnapshot {
    FrameInputSnapshot {
        now_seconds: ctx.input(|input_state| input_state.time),
        pointer_primary_down: ctx.input(|input_state| input_state.pointer.primary_down()),
        is_dragging_ui: ctx.dragged_id().is_some(),
    }
}

pub(super) fn apply_frame_commands(ctx: &egui::Context, commands: &FrameCommands) {
    if let Some(title) = commands.window_title.as_deref() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.into()));
    }
    if commands.request_repaint {
        ctx.request_repaint();
    }
}
