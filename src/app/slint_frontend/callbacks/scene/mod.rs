use super::CallbackContext;
use crate::app::slint_frontend::SlintHostWindow;

mod rows;
mod text;
mod toolbar;
mod top_bar;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    top_bar::install(window, context);
    toolbar::install(window, context);
    rows::install(window, context);
    text::install(window, context);
}
