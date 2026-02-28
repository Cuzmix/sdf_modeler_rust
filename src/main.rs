#[cfg(feature = "egui_ui")]
fn main() -> eframe::Result<()> {
    sdf_modeler::run_native()
}

#[cfg(not(feature = "egui_ui"))]
fn main() {
    sdf_modeler::run_native()
}
