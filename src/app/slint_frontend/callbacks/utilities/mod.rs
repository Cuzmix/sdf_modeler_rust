use super::CallbackContext;
use crate::app::slint_frontend::SlintHostWindow;

mod import_dialog;
mod reference_images;
mod render_settings;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    render_settings::install(window, context);
    reference_images::install(window, context);
    import_dialog::install(window, context);
}
