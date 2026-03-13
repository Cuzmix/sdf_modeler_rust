use std::sync::{Mutex, OnceLock};

use sdf_modeler::app_bridge::AppBridge;

fn app_bridge() -> &'static Mutex<AppBridge> {
    static APP_BRIDGE: OnceLock<Mutex<AppBridge>> = OnceLock::new();
    APP_BRIDGE.get_or_init(|| Mutex::new(AppBridge::new()))
}

fn snapshot_json(bridge: &AppBridge) -> String {
    serde_json::to_string(&bridge.scene_snapshot()).expect("scene snapshot json")
}

#[flutter_rust_bridge::frb(sync)]
pub fn ping() -> String {
    "pong".to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn bridge_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn scene_snapshot_json() -> String {
    let bridge = app_bridge().lock().expect("app bridge mutex");
    snapshot_json(&bridge)
}

pub fn render_preview_frame(width: u32, height: u32, time_seconds: f32) -> Vec<u8> {
    app_bridge()
        .lock()
        .expect("app bridge mutex")
        .render_viewport_frame(width, height, time_seconds)
}

#[flutter_rust_bridge::frb(sync)]
pub fn orbit_camera(delta_x: f32, delta_y: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.orbit_camera(delta_x, delta_y);
}

#[flutter_rust_bridge::frb(sync)]
pub fn pan_camera(delta_x: f32, delta_y: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.pan_camera(delta_x, delta_y);
}

#[flutter_rust_bridge::frb(sync)]
pub fn zoom_camera(delta: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.zoom_camera(delta);
}

#[flutter_rust_bridge::frb(sync)]
pub fn select_node_at_viewport(
    mouse_x: f32,
    mouse_y: f32,
    width: u32,
    height: u32,
    time_seconds: f32,
) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node_at_viewport(mouse_x, mouse_y, width, height, time_seconds);
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn frame_all() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.frame_all();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_sphere() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_sphere();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_scene() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_scene();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
    let _ = app_bridge();
}
