use std::cell::RefCell;
use std::rc::Rc;

mod gpu;

use gpu::scene::{GizmoAxis, PickResult, Scene, SdfPrimitive};

slint::include_modules!();

/// Tracks mouse drag state, dirty flag, click detection, and gizmo interaction.
struct InputState {
    dragging: bool,
    drag_button: i32,
    last_x: f32,
    last_y: f32,
    down_x: f32,
    down_y: f32,
    needs_redraw: bool,
    scene_dirty: bool,
    last_vp_w: u32,
    last_vp_h: u32,
    // FPS tracking
    frame_count: u32,
    fps_accum_start: std::time::Instant,
    // Gizmo drag state
    gizmo_axis: Option<GizmoAxis>,
    gizmo_node_start: glam::Vec3,
    gizmo_drag_origin: glam::Vec3,
}

impl InputState {
    fn new() -> Self {
        Self {
            dragging: false,
            drag_button: -1,
            last_x: 0.0,
            last_y: 0.0,
            down_x: 0.0,
            down_y: 0.0,
            needs_redraw: true,
            scene_dirty: true,
            last_vp_w: 0,
            last_vp_h: 0,
            frame_count: 0,
            fps_accum_start: std::time::Instant::now(),
            gizmo_axis: None,
            gizmo_node_start: glam::Vec3::ZERO,
            gizmo_drag_origin: glam::Vec3::ZERO,
        }
    }
}

/// Generate a world-space ray from a pixel position.
fn screen_to_ray(
    x: f32,
    y: f32,
    vp_w: u32,
    vp_h: u32,
    camera: &gpu::camera::OrbitCamera,
) -> (glam::Vec3, glam::Vec3) {
    let aspect = vp_w as f32 / vp_h.max(1) as f32;
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let inv_vp = (proj * view).inverse();

    let ndc_x = (x / vp_w as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (y / vp_h as f32) * 2.0;

    let near = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let origin = near.truncate() / near.w;
    let target = far.truncate() / far.w;
    let dir = (target - origin).normalize();

    (origin, dir)
}

/// Find closest point on an axis line to a camera ray (line-line closest point).
fn closest_point_on_axis(
    axis_origin: glam::Vec3,
    axis_dir: glam::Vec3,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> glam::Vec3 {
    let w = axis_origin - ray_origin;
    let a = axis_dir.dot(axis_dir);
    let b = axis_dir.dot(ray_dir);
    let c = ray_dir.dot(ray_dir);
    let d = axis_dir.dot(w);
    let e = ray_dir.dot(w);
    let denom = a * c - b * b;
    let t = if denom.abs() < 1e-6 {
        0.0
    } else {
        (b * e - c * d) / denom
    };
    axis_origin + axis_dir * t
}

/// Push the selected node's properties to the Slint UI.
fn push_selection_to_ui(app: &MainWindow, scene: &Scene) {
    if let Some(idx) = scene.selected {
        if let Some(node) = scene.nodes.get(idx) {
            app.set_has_selection(true);
            app.set_selected_name(slint::format!("{}", node.name));
            app.set_prop_pos_x(node.position.x);
            app.set_prop_pos_y(node.position.y);
            app.set_prop_pos_z(node.position.z);
            app.set_prop_scale_x(node.scale.x);
            app.set_prop_scale_y(node.scale.y);
            app.set_prop_scale_z(node.scale.z);
            return;
        }
    }
    app.set_has_selection(false);
}

fn main() {
    env_logger::init();

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
    let scene: Rc<RefCell<Scene>> = Rc::new(RefCell::new(Scene::default_scene()));
    let start_time = std::time::Instant::now();

    // Push initial selection
    push_selection_to_ui(&app, &scene.borrow());

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

    // ── Add Primitive ───────────────────────────────────────────
    {
        let scene_add = scene.clone();
        let input_add = input.clone();
        let app_add = app.as_weak();
        app.on_on_add_primitive(move |prim_type| {
            let prim = match prim_type {
                0 => SdfPrimitive::Sphere,
                1 => SdfPrimitive::Box,
                2 => SdfPrimitive::Cylinder,
                3 => SdfPrimitive::Torus,
                _ => SdfPrimitive::Sphere,
            };
            let mut s = scene_add.borrow_mut();
            s.add_node(prim);
            let mut inp = input_add.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
            if let Some(app) = app_add.upgrade() {
                push_selection_to_ui(&app, &s);
            }
        });
    }

    // ── Delete Selected ─────────────────────────────────────────
    {
        let scene_del = scene.clone();
        let input_del = input.clone();
        let app_del = app.as_weak();
        app.on_on_delete_selected(move || {
            let mut s = scene_del.borrow_mut();
            s.remove_selected();
            let mut inp = input_del.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
            if let Some(app) = app_del.upgrade() {
                push_selection_to_ui(&app, &s);
            }
        });
    }

    // ── Property Changed (from sliders) ─────────────────────────
    {
        let scene_prop = scene.clone();
        let input_prop = input.clone();
        let app_prop = app.as_weak();
        app.on_on_property_changed(move || {
            let Some(app) = app_prop.upgrade() else { return };
            let mut s = scene_prop.borrow_mut();
            if let Some(idx) = s.selected {
                if let Some(node) = s.nodes.get_mut(idx) {
                    node.position.x = app.get_prop_pos_x();
                    node.position.y = app.get_prop_pos_y();
                    node.position.z = app.get_prop_pos_z();
                    node.scale.x = app.get_prop_scale_x();
                    node.scale.y = app.get_prop_scale_y();
                    node.scale.z = app.get_prop_scale_z();
                }
            }
            let mut inp = input_prop.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
        });
    }

    // ── Pointer Events ──────────────────────────────────────────
    {
        let input_down = input.clone();
        let gpu_down = gpu_state.clone();
        let scene_down = scene.clone();
        let cam_down = camera.clone();
        app.on_on_pointer_down(move |x, y, button| {
            let mut s = input_down.borrow_mut();
            s.dragging = false;
            s.drag_button = button;
            s.last_x = x;
            s.last_y = y;
            s.down_x = x;
            s.down_y = y;
            s.gizmo_axis = None;

            // On left-click, check if a gizmo axis was hit
            if button == 0 && s.last_vp_w > 0 && s.last_vp_h > 0 {
                let ndc_x = (x / s.last_vp_w as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (y / s.last_vp_h as f32) * 2.0;
                let vp_w = s.last_vp_w;
                let vp_h = s.last_vp_h;
                drop(s);

                let gpu_ref = gpu_down.borrow();
                let pick = if let Some(gs) = gpu_ref.as_ref() {
                    gs.pick_at(ndc_x, ndc_y)
                } else {
                    PickResult::Background
                };
                drop(gpu_ref);

                if let PickResult::GizmoAxis(axis) = pick {
                    let sc = scene_down.borrow();
                    if let Some(idx) = sc.selected {
                        if let Some(node) = sc.nodes.get(idx) {
                            let cam = cam_down.borrow();
                            let (ro, rd) = screen_to_ray(x, y, vp_w, vp_h, &cam);
                            drop(cam);

                            let axis_dir = match axis {
                                GizmoAxis::X => glam::Vec3::X,
                                GizmoAxis::Y => glam::Vec3::Y,
                                GizmoAxis::Z => glam::Vec3::Z,
                            };

                            let drag_origin =
                                closest_point_on_axis(node.position, axis_dir, ro, rd);

                            let mut s = input_down.borrow_mut();
                            s.gizmo_axis = Some(axis);
                            s.gizmo_node_start = node.position;
                            s.gizmo_drag_origin = drag_origin;
                            s.dragging = true; // immediately start drag
                        }
                    }
                }
            }
        });
    }
    {
        let input_move = input.clone();
        let cam_move = camera.clone();
        let scene_move = scene.clone();
        let app_move = app.as_weak();
        app.on_on_pointer_move(move |x, y| {
            let mut s = input_move.borrow_mut();
            let dx = x - s.last_x;
            let dy = y - s.last_y;
            s.last_x = x;
            s.last_y = y;

            // Detect drag (> 3px from down position)
            let total_dx = x - s.down_x;
            let total_dy = y - s.down_y;
            if total_dx * total_dx + total_dy * total_dy > 9.0 {
                s.dragging = true;
            }

            if !s.dragging {
                return;
            }

            // Gizmo drag mode
            if let Some(axis) = s.gizmo_axis {
                let vp_w = s.last_vp_w;
                let vp_h = s.last_vp_h;
                let node_start = s.gizmo_node_start;
                let drag_origin = s.gizmo_drag_origin;
                s.needs_redraw = true;
                s.scene_dirty = true;
                drop(s);

                let axis_dir = match axis {
                    GizmoAxis::X => glam::Vec3::X,
                    GizmoAxis::Y => glam::Vec3::Y,
                    GizmoAxis::Z => glam::Vec3::Z,
                };

                let cam = cam_move.borrow();
                let (ro, rd) = screen_to_ray(x, y, vp_w, vp_h, &cam);
                drop(cam);

                let current = closest_point_on_axis(node_start, axis_dir, ro, rd);
                let displacement = current - drag_origin;
                // Only take the component along the axis
                let delta = axis_dir * displacement.dot(axis_dir);
                let new_pos = node_start + delta;

                let mut sc = scene_move.borrow_mut();
                if let Some(idx) = sc.selected {
                    if let Some(node) = sc.nodes.get_mut(idx) {
                        node.position = new_pos;
                    }
                }
                drop(sc);

                if let Some(app) = app_move.upgrade() {
                    let sc = scene_move.borrow();
                    push_selection_to_ui(&app, &sc);
                }

                return;
            }

            // Camera control mode
            let mut cam = cam_move.borrow_mut();
            match s.drag_button {
                0 => cam.orbit(dx, dy),
                1 | 2 => cam.pan(dx, dy),
                _ => {}
            }
            s.needs_redraw = true;
        });
    }
    {
        let input_up = input.clone();
        let scene_up = scene.clone();
        let gpu_up = gpu_state.clone();
        let app_up = app.as_weak();
        app.on_on_pointer_up(move || {
            let s = input_up.borrow();
            let was_click = !s.dragging && s.drag_button == 0;
            let click_x = s.down_x;
            let click_y = s.down_y;
            let vp_w = s.last_vp_w;
            let vp_h = s.last_vp_h;
            drop(s);

            if was_click && vp_w > 0 && vp_h > 0 {
                // Pixel → NDC
                let ndc_x = (click_x / vp_w as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (click_y / vp_h as f32) * 2.0;

                // GPU pick
                let gpu_ref = gpu_up.borrow();
                let pick_result = if let Some(gs) = gpu_ref.as_ref() {
                    gs.pick_at(ndc_x, ndc_y)
                } else {
                    PickResult::Background
                };
                drop(gpu_ref);

                let mut sc = scene_up.borrow_mut();
                match pick_result {
                    PickResult::Node(i) => {
                        sc.selected = Some(i);
                    }
                    _ => {
                        sc.selected = None;
                    }
                }

                let mut inp = input_up.borrow_mut();
                inp.scene_dirty = true;
                inp.needs_redraw = true;
                drop(inp);

                if let Some(app) = app_up.upgrade() {
                    push_selection_to_ui(&app, &sc);
                }
            }

            let mut s = input_up.borrow_mut();
            s.dragging = false;
            s.drag_button = -1;
            s.gizmo_axis = None;
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
    let scene_for_timer = scene.clone();
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

            // Track viewport size for click-to-select
            if vp_w != inp.last_vp_w || vp_h != inp.last_vp_h {
                inp.last_vp_w = vp_w;
                inp.last_vp_h = vp_h;
                inp.needs_redraw = true;
            }

            if !inp.needs_redraw {
                return;
            }
            inp.needs_redraw = false;
            let scene_dirty = inp.scene_dirty;
            inp.scene_dirty = false;
            drop(inp);

            let elapsed = start_time.elapsed().as_secs_f32();
            let uniforms = cam_for_timer.borrow().uniforms(vp_w, vp_h, elapsed);

            let mut gpu_ref = gpu_for_timer.borrow_mut();
            if let Some(gs) = gpu_ref.as_mut() {
                gs.update_camera(&uniforms);

                // Upload scene data when changed
                if scene_dirty {
                    let sc = scene_for_timer.borrow();
                    let gpu_nodes = sc.pack_nodes();
                    let gpu_info = sc.pack_info();
                    gs.update_scene(&gpu_nodes, &gpu_info);
                }

                if let Some(image) = gs.render_frame(vp_w, vp_h) {
                    app.set_viewport_texture(image);
                }
            }
            drop(gpu_ref);

            // Update FPS counter every ~500ms
            let mut inp = input_for_timer.borrow_mut();
            inp.frame_count += 1;
            let fps_elapsed = inp.fps_accum_start.elapsed().as_secs_f32();
            if fps_elapsed >= 0.5 {
                let fps = inp.frame_count as f32 / fps_elapsed;
                app.set_fps_text(slint::format!("{:.0} fps", fps));
                inp.frame_count = 0;
                inp.fps_accum_start = std::time::Instant::now();
            }
        },
    );

    app.run().expect("Failed to run application");
}
