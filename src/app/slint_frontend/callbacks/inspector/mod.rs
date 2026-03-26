use super::context::CallbackContext;
use crate::app::slint_frontend::SlintHostWindow;

mod edit_helpers;
mod light;
mod material;
mod operation;
mod sculpt;
mod transform;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    transform::install(window, context);
    material::install(window, context);
    operation::install(window, context);
    sculpt::install(window, context);
    light::install(window, context);
}
