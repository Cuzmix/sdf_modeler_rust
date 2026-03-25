#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn main() -> Result<(), String> {
    sdf_modeler_lib::run_native()
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
fn main() {}
