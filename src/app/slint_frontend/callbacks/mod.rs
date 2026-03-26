use std::cell::RefCell;
use std::rc::Rc;

use slint::{ComponentHandle, Timer};

use crate::app::frontend_models::ShellSnapshot;

use super::host_state::SlintHostState;
use super::SlintHostWindow;

mod inspector;
mod scene;
mod utilities;
mod viewport;

#[derive(Clone)]
pub(super) struct CallbackContext {
    pub(super) host: Rc<RefCell<SlintHostState>>,
    pub(super) active_timer: Rc<Timer>,
    pub(super) window_weak: slint::Weak<SlintHostWindow>,
}

pub(super) fn install_callbacks(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    let context = CallbackContext {
        host: host.clone(),
        active_timer: active_timer.clone(),
        window_weak: window.as_weak(),
    };
    scene::install(window, &context);
    inspector::install(window, &context);
    utilities::install(window, &context);
    viewport::install(window, &context);
}

pub(super) fn mutate_host_and_tick<F>(context: &CallbackContext, mutate: F)
where
    F: FnOnce(&mut SlintHostState),
{
    {
        let mut host_state = context.host.borrow_mut();
        mutate(&mut host_state);
        host_state.viewport_dirty = true;
    }

    if let Some(window) = context.window_weak.upgrade() {
        super::drive_host_tick(&window, &context.host, &context.active_timer);
    }
}

fn scene_row_at(
    snapshot: Option<&ShellSnapshot>,
    index: i32,
) -> Option<&crate::app::frontend_models::ScenePanelRow> {
    if index < 0 {
        return None;
    }
    snapshot?.scene_panel.rows.get(index as usize)
}

fn axis_value(axis: i32, values: [f32; 3]) -> f32 {
    match axis {
        1 => values[1],
        2 => values[2],
        _ => values[0],
    }
}
