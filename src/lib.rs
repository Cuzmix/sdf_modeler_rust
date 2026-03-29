mod app;
mod compat;
mod desktop_dialogs;
mod dock_style;
mod egui_theme;
mod export;
pub mod expression;
mod gpu;
mod graph;
mod io;
pub mod keymap;
mod material_preset;
mod mesh_import;
mod native_paths;
mod sculpt;
mod settings;
mod ui;

// ── Native entry point ──────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn build_native_options(settings: &settings::Settings) -> eframe::NativeOptions {
    use eframe::egui;
    use eframe::wgpu;
    use std::sync::Arc;

    let present_mode = if settings.vsync_enabled {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    };

    eframe::NativeOptions {
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
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_native_with_options(
    native_options: eframe::NativeOptions,
    settings: settings::Settings,
) -> eframe::Result<()> {
    eframe::run_native(
        "SDF Modeler",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::SdfApp::new(cc, settings)))),
    )
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
pub fn run_native() -> eframe::Result<()> {
    let _ = env_logger::builder().is_test(false).try_init();

    let settings = settings::Settings::load();
    let options = build_native_options(&settings);
    run_native_with_options(options, settings)
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let _ = env_logger::builder().is_test(false).try_init();

    let settings = settings::Settings::load();
    let mut options = build_native_options(&settings);
    options.event_loop_builder = Some(Box::new(move |event_loop_builder| {
        event_loop_builder.with_android_app(app);
    }));

    if let Err(error) = run_native_with_options(options, settings) {
        log::error!("Android startup failed: {}", error);
    }
}

// ── WASM entry point ────────────────────────────────────────────────────────

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
        use eframe::wgpu;
        use std::sync::Arc;
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
