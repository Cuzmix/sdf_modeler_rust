#[cfg(not(target_os = "android"))]
fn main() -> eframe::Result<()> {
    sdf_modeler_lib::run_native()
}

#[cfg(target_os = "android")]
fn main() {}
