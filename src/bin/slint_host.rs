#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn main() {
    if let Err(error) = sdf_modeler_lib::run_slint_native() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
fn main() {
    eprintln!("The `slint_host` binary is only supported on native desktop.");
    std::process::exit(1);
}
