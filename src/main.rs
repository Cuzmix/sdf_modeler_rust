#[cfg(feature = "egui_ui")]
fn main() -> eframe::Result<()> {
    sdf_modeler::run_native()
}

#[cfg(not(feature = "egui_ui"))]
fn main() {
    println!("SDF Modeler: no UI feature enabled. Use --features egui_ui or flutter_ui.");
}
