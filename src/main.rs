mod app;
mod gpu;
mod graph;
mod io;
mod sculpt;
mod settings;
mod ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let settings = settings::Settings::load();

    let present_mode = if settings.vsync_enabled {
        eframe::wgpu::PresentMode::AutoVsync
    } else {
        eframe::wgpu::PresentMode::AutoNoVsync
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("SDF Modeler"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode,
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "SDF Modeler",
        options,
        Box::new(move |cc| Ok(Box::new(app::SdfApp::new(cc, settings)))),
    )
}
