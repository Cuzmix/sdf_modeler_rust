mod app;
mod compat;
mod desktop_dialogs;
mod egui_keymap;
mod export;
pub mod expression;
mod gpu;
mod graph;
mod io;
pub mod keymap;
mod material_preset;
mod mesh_import;
mod native_paths;
#[cfg(not(target_arch = "wasm32"))]
mod native_wgpu;
mod sculpt;
mod settings;
mod ui;

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
pub fn run_native() -> Result<(), String> {
    let _ = env_logger::builder().is_test(false).try_init();

    let settings = settings::Settings::load();
    app::slint_frontend::run_slint_host(settings)
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
pub fn run_slint_native() -> Result<(), String> {
    run_native()
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
struct EframeHostAdapter {
    app: app::SdfApp,
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
impl EframeHostAdapter {
    fn new(cc: &eframe::CreationContext<'_>, settings: settings::Settings) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");
        Self {
            app: app::SdfApp::new_from_egui(&render_state, &cc.egui_ctx, settings),
        }
    }
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
impl eframe::App for EframeHostAdapter {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.app.run_egui_frame(ctx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn on_exit(&mut self) {
        self.app.mark_clean_exit();
    }
}

#[cfg(target_os = "android")]
fn build_android_options(settings: &settings::Settings) -> eframe::NativeOptions {
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
            device_descriptor: std::sync::Arc::new(native_wgpu::native_device_descriptor),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let _ = env_logger::builder().is_test(false).try_init();

    let settings = settings::Settings::load();
    let mut options = build_android_options(&settings);
    options.event_loop_builder = Some(Box::new(move |event_loop_builder| {
        event_loop_builder.with_android_app(app);
    }));

    if let Err(error) = eframe::run_native(
        "SDF Modeler",
        options,
        Box::new(move |cc| Ok(Box::new(EframeHostAdapter::new(cc, settings.clone())))),
    ) {
        log::error!("Android startup failed: {}", error);
    }
}

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
                device_descriptor: Arc::new(|_adapter| wgpu::DeviceDescriptor {
                    label: Some("SDF Modeler device (web)"),
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 4096,
                        max_storage_buffers_per_shader_stage: 4,
                        max_storage_buffer_binding_size: 1 << 25,
                        max_storage_textures_per_shader_stage: 0,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    memory_hints: wgpu::MemoryHints::default(),
                }),
                ..Default::default()
            },
            ..Default::default()
        };

        self.runner
            .start(
                canvas,
                web_options,
                Box::new(move |cc| Ok(Box::new(EframeHostAdapter::new(cc, settings)))),
            )
            .await
    }

    #[wasm_bindgen]
    pub fn destroy(&self) {
        self.runner.destroy();
    }
}
