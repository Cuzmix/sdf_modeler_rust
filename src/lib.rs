mod app;
mod compat;
pub mod core;
pub mod expression;
mod export;
mod gpu;
mod graph;
mod io;
pub mod keymap;
mod material_preset;
mod mesh_import;
mod sculpt;
mod sculpt_history;
mod settings;
mod ui;
#[cfg(not(target_arch = "wasm32"))]
mod ui_slint;

#[cfg(not(target_arch = "wasm32"))]
fn parse_frontend_from_args() -> Option<settings::FrontendKind> {
    let args: Vec<String> = std::env::args().collect();

    for (idx, arg) in args.iter().enumerate() {
        if let Some(raw) = arg.strip_prefix("--frontend=") {
            if let Some(frontend) = settings::FrontendKind::from_str(raw) {
                return Some(frontend);
            }
        }
        if arg == "--frontend" {
            if let Some(next) = args.get(idx + 1) {
                if let Some(frontend) = settings::FrontendKind::from_str(next) {
                    return Some(frontend);
                }
            }
        }
    }

    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_selected_frontend() -> Result<(), Box<dyn std::error::Error>> {
    let settings = settings::Settings::load();

    let selected_frontend = parse_frontend_from_args()
        .or_else(|| {
            std::env::var("SDF_FRONTEND")
                .ok()
                .and_then(|raw| settings::FrontendKind::from_str(&raw))
        })
        .unwrap_or(settings.preferred_frontend);

    match selected_frontend {
        settings::FrontendKind::Egui => {
            #[cfg(feature = "legacy-egui-ui")]
            {
                run_native()
                    .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })
            }
            #[cfg(not(feature = "legacy-egui-ui"))]
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "legacy-egui-ui feature is not enabled",
                )
                .into());
            }
        }
        settings::FrontendKind::Slint => ui_slint::run_slint_shell(&settings),
    }
}

// --- Native entry point ------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> eframe::Result<()> {
    use std::sync::Arc;
    use eframe::egui;
    use eframe::wgpu;

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
                        max_storage_textures_per_shader_stage: 4,
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

// --- WASM entry point --------------------------------------------------------

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        eframe::WebLogger::init(log::LevelFilter::Info).ok();
        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    #[wasm_bindgen]
    pub async fn start(&self, canvas_id: &str) -> Result<(), JsValue> {
        use std::sync::Arc;
        use eframe::wgpu;
        use wasm_bindgen::JsCast;

        let document = web_sys::window()
            .and_then(|w| w.document())
            .ok_or_else(|| JsValue::from_str("No document"))?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("Canvas not found"))?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("Element is not a canvas"))?;

        let settings = settings::Settings::load();

        let web_options = eframe::WebOptions {
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                device_descriptor: Arc::new(|_adapter| {
                    wgpu::DeviceDescriptor {
                        label: Some("SDF Modeler device (web)"),
                        required_features: wgpu::Features::FLOAT32_FILTERABLE,
                        required_limits: wgpu::Limits {
                            max_texture_dimension_2d: 4096,
                            max_storage_buffers_per_shader_stage: 4,
                            max_storage_buffer_binding_size: 1 << 25, // 32MB
                            max_storage_textures_per_shader_stage: 0,
                            ..wgpu::Limits::downlevel_defaults()
                        },
                        memory_hints: wgpu::MemoryHints::default(),
                    }
                }),
                ..Default::default()
            },
            ..Default::default()
        };

        self.runner
            .start(
                canvas,
                web_options,
                Box::new(move |cc| Ok(Box::new(app::SdfApp::new(cc, settings)))),
            )
            .await
    }

    #[wasm_bindgen]
    pub fn destroy(&self) {
        self.runner.destroy();
    }
}
