use std::sync::{Mutex, OnceLock};

use sdf_modeler::app_bridge::AppBridge;

pub(crate) fn app_bridge() -> &'static Mutex<AppBridge> {
    static APP_BRIDGE: OnceLock<Mutex<AppBridge>> = OnceLock::new();
    APP_BRIDGE.get_or_init(|| Mutex::new(AppBridge::new()))
}

pub(crate) fn snapshot_json(bridge: &AppBridge) -> String {
    serde_json::to_string(&bridge.scene_snapshot()).expect("scene snapshot json")
}

pub(crate) fn viewport_feedback_json(bridge: &AppBridge) -> String {
    serde_json::to_string(&bridge.viewport_feedback()).expect("viewport feedback json")
}
