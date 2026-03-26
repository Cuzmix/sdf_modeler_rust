use crate::settings::Settings;

fn init_logging() {
    let _ = env_logger::builder().is_test(false).try_init();
}

pub(crate) fn run_desktop() -> Result<(), String> {
    init_logging();
    let settings = Settings::load();
    crate::app::slint_frontend::run_slint_host(settings)
}

#[cfg(target_os = "android")]
pub(crate) fn run_android(app: slint::android::AndroidApp) {
    init_logging();
    let settings = Settings::load();
    if let Err(error) = crate::app::slint_frontend::run_slint_host_android(app, settings) {
        log::error!("{error}");
    }
}
