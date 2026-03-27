use std::cell::RefCell;
use std::rc::Rc;

use slint::Timer;

use super::host_state::SlintHostState;
use super::SlintHostWindow;

mod context;
mod inspector;
mod menu_strip;
mod mutation;
mod panels;
mod scene;
mod scene_lookup;
mod tool_palette;
mod utilities;
mod vector_axes;
mod viewport;
mod workspace;

pub(super) fn install_callbacks(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    let context = context::CallbackContext::new(window, host, active_timer);
    menu_strip::install(window, &context);
    panels::install(window, &context);
    scene::install(window, &context);
    tool_palette::install(window, &context);
    inspector::install(window, &context);
    utilities::install(window, &context);
    workspace::install(window, &context);
    viewport::install(window, &context);
}
