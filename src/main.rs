mod app;
mod gpu;
mod graph;
mod io;
mod sculpt;
mod settings;
mod ui;

use std::sync::Arc;

use eframe::egui;
use eframe::wgpu;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let settings = settings::Settings::load();

    let present_mode = if settings.vsync_enabled {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("SDF Modeler"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode,
            device_descriptor: Arc::new(|adapter| {
                let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                };
                wgpu::DeviceDescriptor {
                    label: Some("SDF Modeler device"),
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 8192,
                        max_storage_buffers_per_shader_stage: 4,
                        max_storage_buffer_binding_size: 1 << 27, // 128MB
                        ..base_limits
                    },
                    memory_hints: wgpu::MemoryHints::default(),
                }
            }),
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
