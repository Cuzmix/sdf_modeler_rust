use std::cell::RefCell;
use std::rc::Rc;

mod gpu;

slint::include_modules!();

/// Tracks mouse drag state and dirty flag for rendering.
struct InputState {
    dragging: bool,
    drag_button: i32,
    last_x: f32,
    last_y: f32,
    needs_redraw: bool,
    last_vp_w: u32,
    last_vp_h: u32,
}

impl InputState {
    fn new() -> Self {
        Self {
            dragging: false,
            drag_button: -1,
            last_x: 0.0,
            last_y: 0.0,
            needs_redraw: true, // draw the initial frame
            last_vp_w: 0,
            last_vp_h: 0,
        }
    }
}

fn main() {
    env_logger::init();

    // Require wgpu 28 backend so we can share the Device
    slint::BackendSelector::new()
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::default())
        .select()
        .expect("Failed to select wgpu 28 backend");

    let app = MainWindow::new().expect("Failed to create MainWindow");
    let app_weak = app.as_weak();

    // Shared state
    let gpu_state: Rc<RefCell<Option<gpu::state::GpuState>>> = Rc::new(RefCell::new(None));
    let camera: Rc<RefCell<gpu::camera::OrbitCamera>> =
        Rc::new(RefCell::new(gpu::camera::OrbitCamera::new()));
    let input: Rc<RefCell<InputState>> = Rc::new(RefCell::new(InputState::new()));
    let start_time = std::time::Instant::now();

    // Grab device+queue from Slint's wgpu backend
    let gpu_for_notifier = gpu_state.clone();
    app.window()
        .set_rendering_notifier(move |state, graphics_api| {
            match (state, graphics_api) {
                (
                    slint::RenderingState::RenderingSetup,
                    slint::GraphicsAPI::WGPU28 {
                        device, queue, ..
                    },
                ) => {
                    log::info!("wgpu RenderingSetup — creating GpuState");
                    let gs = gpu::state::GpuState::new(device.clone(), queue.clone());
                    *gpu_for_notifier.borrow_mut() = Some(gs);
                }
                (slint::RenderingState::RenderingTeardown, _) => {
                    log::info!("wgpu RenderingTeardown — dropping GpuState");
                    *gpu_for_notifier.borrow_mut() = None;
                }
                _ => {}
            }
        })
        .expect("Failed to set rendering notifier");

    // ── Pointer Events ──────────────────────────────────────────
    {
        let input_down = input.clone();
        app.on_on_pointer_down(move |x, y, button| {
            let mut s = input_down.borrow_mut();
            s.dragging = true;
            s.drag_button = button;
            s.last_x = x;
            s.last_y = y;
        });
    }
    {
        let input_move = input.clone();
        let cam_move = camera.clone();
        app.on_on_pointer_move(move |x, y| {
            let mut s = input_move.borrow_mut();
            if !s.dragging {
                return;
            }
            let dx = x - s.last_x;
            let dy = y - s.last_y;
            s.last_x = x;
            s.last_y = y;

            let mut cam = cam_move.borrow_mut();
            match s.drag_button {
                0 => cam.orbit(dx, dy),          // left = orbit
                1 | 2 => cam.pan(dx, dy),        // right/middle = pan
                _ => {}
            }
            s.needs_redraw = true;
        });
    }
    {
        let input_up = input.clone();
        app.on_on_pointer_up(move || {
            let mut s = input_up.borrow_mut();
            s.dragging = false;
            s.drag_button = -1;
        });
    }
    {
        let input_scroll = input.clone();
        let cam_scroll = camera.clone();
        app.on_on_scroll(move |delta| {
            cam_scroll.borrow_mut().zoom(delta);
            input_scroll.borrow_mut().needs_redraw = true;
        });
    }

    // ── Render Loop (~60 fps timer) ─────────────────────────────
    let gpu_for_timer = gpu_state.clone();
    let cam_for_timer = camera.clone();
    let input_for_timer = input.clone();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16),
        move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };

            let vp_w = app.get_viewport_width() as u32;
            let vp_h = app.get_viewport_height() as u32;

            if vp_w == 0 || vp_h == 0 {
                return;
            }

            let mut inp = input_for_timer.borrow_mut();

            // Mark dirty on viewport resize
            if vp_w != inp.last_vp_w || vp_h != inp.last_vp_h {
                inp.last_vp_w = vp_w;
                inp.last_vp_h = vp_h;
                inp.needs_redraw = true;
            }

            // Skip GPU work when nothing changed
            if !inp.needs_redraw {
                return;
            }
            inp.needs_redraw = false;
            drop(inp); // release borrow before GPU work

            let elapsed = start_time.elapsed().as_secs_f32();
            let uniforms = cam_for_timer.borrow().uniforms(vp_w, vp_h, elapsed);

            let mut gpu_ref = gpu_for_timer.borrow_mut();
            if let Some(gs) = gpu_ref.as_mut() {
                gs.update_camera(&uniforms);
                if let Some(image) = gs.render_frame(vp_w, vp_h) {
                    app.set_viewport_texture(image);
                }
            }
        },
    );

    app.run().expect("Failed to run application");
}
