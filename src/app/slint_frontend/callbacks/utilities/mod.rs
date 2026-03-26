use super::context::CallbackContext;
use crate::app::slint_frontend::SlintHostWindow;

mod history;
mod import_dialog;
mod reference_images;
mod render_settings;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    history::install(window, context);
    render_settings::install(window, context);
    reference_images::install(window, context);
    import_dialog::install(window, context);
}
