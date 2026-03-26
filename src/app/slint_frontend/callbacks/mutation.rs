use super::context::CallbackContext;

pub(super) fn mutate_host_and_tick<F>(context: &CallbackContext, mutate: F)
where
    F: FnOnce(&mut crate::app::slint_frontend::host_state::SlintHostState),
{
    {
        let mut host_state = context.host.borrow_mut();
        mutate(&mut host_state);
        host_state.viewport_dirty = true;
    }

    if let Some(window) = context.window_weak.upgrade() {
        super::super::drive_host_tick(&window, &context.host, &context.active_timer);
    }
}
