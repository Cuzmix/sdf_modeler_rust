use std::cell::RefCell;
use std::rc::Rc;

use slint::Timer;

use super::host_state::SlintHostState;
use super::SlintHostWindow;

mod context;
mod inspector;
mod mutation;
mod scene;
mod scene_lookup;
mod utilities;
mod vector_axes;
mod viewport;

pub(super) fn install_callbacks(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    let context = context::CallbackContext::new(window, host, active_timer);
    scene::install(window, &context);
    inspector::install(window, &context);
    utilities::install(window, &context);
    viewport::install(window, &context);
}
