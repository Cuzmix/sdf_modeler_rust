mod app;
mod gpu;
mod graph;
mod ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("SDF Modeler"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "SDF Modeler",
        options,
        Box::new(|cc| Ok(Box::new(app::SdfApp::new(cc)))),
    )
}
