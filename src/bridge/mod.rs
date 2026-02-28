/// Flutter bridge API — thin FFI layer between Dart and the Rust core.
/// All public functions here are exposed to Dart via flutter_rust_bridge codegen.

/// Simple hello function to verify the bridge is working end-to-end.
pub fn hello_from_rust() -> String {
    "Hello from SDF Modeler Rust core!".to_string()
}

/// Returns the wgpu backend name for the default adapter (useful for diagnostics).
pub fn gpu_backend_name() -> String {
    // This is a lightweight check — no device creation needed
    format!("wgpu {}", env!("CARGO_PKG_VERSION"))
}
