use std::sync::{Mutex, OnceLock};

use sdf_modeler::app_bridge::AppBridge;

pub(crate) fn app_bridge() -> &'static Mutex<AppBridge> {
    static APP_BRIDGE: OnceLock<Mutex<AppBridge>> = OnceLock::new();
    APP_BRIDGE.get_or_init(|| Mutex::new(AppBridge::new()))
}
