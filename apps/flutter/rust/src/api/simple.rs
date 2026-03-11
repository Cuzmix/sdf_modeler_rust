#[flutter_rust_bridge::frb(sync)]
pub fn ping() -> String {
    "pong".to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn bridge_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Keep default utilities enabled for logging/error integration.
    flutter_rust_bridge::setup_default_user_utils();
}
