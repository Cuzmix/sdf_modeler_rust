use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use slint::{ComponentHandle, Timer, TimerMode};

use crate::app::runtime::{AppRenderContext, WakeHandle};
use crate::settings::Settings;

mod bindings;
mod callbacks;
mod host_state;

slint::include_modules!();

use host_state::{NativeWgpuContext, SlintHostState};

pub(crate) fn run_slint_host(settings: Settings) -> Result<(), String> {
    run_slint_host_internal(settings)
}

#[cfg(target_os = "android")]
pub(crate) fn run_slint_host_android(
    app: slint::android::AndroidApp,
    settings: Settings,
) -> Result<(), String> {
    slint::android::init(app).map_err(|error| error.to_string())?;
    run_slint_host_internal(settings)
}

fn run_slint_host_internal(settings: Settings) -> Result<(), String> {
    let native_wgpu = create_render_context()?;
    slint::BackendSelector::new()
        .require_wgpu_27(slint::wgpu_27::WGPUConfiguration::Manual {
            instance: native_wgpu.instance.clone(),
            adapter: native_wgpu.render_context.adapter.as_ref().clone(),
            device: native_wgpu.render_context.device.as_ref().clone(),
            queue: native_wgpu.render_context.queue.as_ref().clone(),
        })
        .select()
        .map_err(|error| error.to_string())?;

    let window = SlintHostWindow::new().map_err(|error| error.to_string())?;
    window.set_window_title("SDF Modeler".into());

    let wake_flag = Arc::new(AtomicBool::new(false));
    let wake = WakeHandle::new({
        let wake_flag = wake_flag.clone();
        let window_weak = window.as_weak();
        move || {
            wake_flag.store(true, Ordering::Relaxed);
            let _ = window_weak.upgrade_in_event_loop(|window| {
                window.window().request_redraw();
            });
        }
    });

    let app = super::SdfApp::new_from_runtime(native_wgpu.render_context, wake, settings);
    let host = Rc::new(RefCell::new(SlintHostState::new(app, wake_flag)));
    let active_timer = Rc::new(Timer::default());

    callbacks::install_callbacks(&window, &host, &active_timer);
    install_rendering_notifier(&window, &host, &active_timer)?;
    drive_host_tick(&window, &host, &active_timer);

    window.run().map_err(|error| error.to_string())?;
    active_timer.stop();
    host.borrow_mut().app.mark_clean_exit();

    Ok(())
}

fn create_render_context() -> Result<NativeWgpuContext, String> {
    let instance = wgpu::Instance::new(&crate::native_wgpu::native_instance_descriptor());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|error| format!("Failed to acquire a WGPU adapter for the Slint host: {error}"))?;
    let descriptor = crate::native_wgpu::native_device_descriptor(&adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
        .map_err(|error| format!("Failed to create a WGPU device for the Slint host: {error}"))?;

    Ok(NativeWgpuContext {
        instance,
        render_context: AppRenderContext::new(
            Arc::new(device),
            Arc::new(queue),
            Arc::new(adapter),
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ),
    })
}

fn install_rendering_notifier(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) -> Result<(), String> {
    let window_weak = window.as_weak();
    let host = host.clone();
    let active_timer = active_timer.clone();
    window
        .window()
        .set_rendering_notifier(move |state, graphics_api| {
            let Some(window) = window_weak.upgrade() else {
                return;
            };
            match state {
                slint::RenderingState::RenderingSetup => {
                    if !matches!(graphics_api, slint::GraphicsAPI::WGPU27 { .. }) {
                        log::error!("Slint did not initialize the WGPU 27 renderer");
                    }
                    host.borrow_mut().viewport_dirty = true;
                }
                slint::RenderingState::BeforeRendering => {
                    if host.borrow().wake_flag.load(Ordering::Relaxed) {
                        drive_host_tick(&window, &host, &active_timer);
                    }
                    host.borrow_mut().render_viewport_if_needed();
                }
                slint::RenderingState::RenderingTeardown => {
                    host.borrow_mut().release_viewport_texture();
                }
                slint::RenderingState::AfterRendering => {}
                _ => {}
            }
        })
        .map_err(|error| error.to_string())
}

fn drive_host_tick(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    let outcome = host.borrow_mut().tick(window);
    sync_continuous_timer(
        window.as_weak(),
        host,
        active_timer,
        outcome.needs_continuous_ticks,
    );
    if outcome.request_redraw {
        window.window().request_redraw();
    }
}

fn sync_continuous_timer(
    window_weak: slint::Weak<SlintHostWindow>,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
    active: bool,
) {
    if host.borrow().continuous_tick_active == active {
        return;
    }
    host.borrow_mut().continuous_tick_active = active;
    if active {
        let window_weak = window_weak.clone();
        let host = host.clone();
        let active_timer = active_timer.clone();
        let timer_for_callback = active_timer.clone();
        active_timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
            let Some(window) = window_weak.upgrade() else {
                return;
            };
            drive_host_tick(&window, &host, &timer_for_callback);
        });
    } else {
        active_timer.stop();
    }
}
