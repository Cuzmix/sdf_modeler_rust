use crate::bridge_state::{app_bridge, snapshot_json};

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
        .pixels
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
pub fn select_node(node_id: Option<u64>) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node(node_id);
    snapshot_json(&bridge)
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
pub fn toggle_node_visibility(node_id: u64) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_visibility(node_id);
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_node_lock(node_id: u64) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_lock(node_id);
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn delete_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.delete_selected();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn duplicate_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.duplicate_selected();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn rename_node(node_id: u64, name: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.rename_node(node_id, &name);
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn undo() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.undo();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn redo() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.redo();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn focus_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.focus_selected();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn frame_all() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.frame_all();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_front() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_front();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_top() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_top();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_right() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_right();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_back() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_back();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_left() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_left();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_bottom() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_bottom();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_orthographic() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_orthographic();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_sphere() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_sphere();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_box() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_box();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_cylinder() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_cylinder();
    snapshot_json(&bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_torus() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_torus();
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
