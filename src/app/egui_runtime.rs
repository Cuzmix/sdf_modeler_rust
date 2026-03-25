use super::runtime::{AppRenderContext, WakeHandle};

pub(super) fn app_render_context_from_egui(
    render_state: &egui_wgpu::RenderState,
) -> AppRenderContext {
    AppRenderContext::new(
        render_state.device.clone(),
        render_state.queue.clone(),
        render_state.adapter.clone(),
        render_state.target_format,
    )
}

pub(super) fn wake_handle_from_egui(ctx: &egui::Context) -> WakeHandle {
    let ctx = ctx.clone();
    WakeHandle::new(move || ctx.request_repaint())
}
