use std::cell::RefCell;
use std::rc::Rc;

use slint::{ComponentHandle, Timer};

use super::super::host_state::SlintHostState;
use super::super::SlintHostWindow;

#[derive(Clone)]
pub(super) struct CallbackContext {
    pub(super) host: Rc<RefCell<SlintHostState>>,
    pub(super) active_timer: Rc<Timer>,
    pub(super) window_weak: slint::Weak<SlintHostWindow>,
}

impl CallbackContext {
    pub(super) fn new(
        window: &SlintHostWindow,
        host: &Rc<RefCell<SlintHostState>>,
        active_timer: &Rc<Timer>,
    ) -> Self {
        Self {
            host: host.clone(),
            active_timer: active_timer.clone(),
            window_weak: window.as_weak(),
        }
    }
}
