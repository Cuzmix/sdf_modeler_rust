// src/app_slint.rs — Phase 3: Slint + wgpu 28 integration with full UI panels
//
// Slint owns the window and event loop. We create wgpu resources manually and
// render SDF to an offscreen texture that Slint displays as an Image element.
// UI panels (scene tree, properties, render settings) sync via ui_dirty flag.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use slint::{ModelRc, SharedString, VecModel};
use slint::wgpu_28::wgpu;

use crate::app_slint_sync;
use crate::compat::Instant;
use crate::gpu::buffers;
use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};
use crate::settings::{RenderConfig, Settings};
use crate::ui::viewport::ViewportResources;

slint::include_modules!();

// ---------------------------------------------------------------------------
// Mouse state tracking
// ---------------------------------------------------------------------------

struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    middle_pressed: bool,
    last_position: Option<(f32, f32)>,
    click_start: Option<(f32, f32)>,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            left_pressed: false,
            right_pressed: false,
            middle_pressed: false,
            last_position: None,
            click_start: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared application state (Rc<RefCell> for Slint callback sharing)
// ---------------------------------------------------------------------------

struct AppSharedState {
    camera: Camera,
    scene: Scene,
    selected: Option<NodeId>,
    history: History,
    settings: Settings,

    // Sync
    current_structure_key: u64,
    buffer_dirty: bool,
    ui_dirty: bool,
    last_data_fingerprint: u64,
    voxel_gpu_offsets: HashMap<NodeId, u32>,
    sculpt_tex_indices: HashMap<NodeId, usize>,

    // Pick
    pending_pick: Option<PendingPick>,

    // Input
    mouse: MouseState,

    // Timing
    start_time: Instant,
    #[allow(dead_code)]
    last_frame_time: Instant,

    // File
    current_file_path: Option<PathBuf>,

    // Viewport dimensions (updated each render frame)
    viewport_w: u32,
    viewport_h: u32,
}

// ---------------------------------------------------------------------------
// GPU renderer (lives inside rendering notifier closure)
// ---------------------------------------------------------------------------

struct SdfRenderer {
    viewport: ViewportResources,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(settings: Settings, scene: Scene, camera: Camera, file_path: Option<PathBuf>) {
    // 1. Create wgpu resources manually (we need specific features/limits)
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, // Slint configures its own surface
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");

    log::info!("Using adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("SDF Modeler device"),
            required_features: wgpu::Features::FLOAT32_FILTERABLE,
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 8192,
                max_storage_buffers_per_shader_stage: 4,
                max_storage_buffer_binding_size: 1 << 27, // 128MB
                max_storage_textures_per_shader_stage: 4,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        },
    ))
    .expect("Failed to create device");

    // 2. Configure Slint backend with our wgpu resources
    slint::BackendSelector::new()
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::Manual {
            instance,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
        })
        .with_winit_window_attributes_hook(|attrs| {
            attrs
                .with_title("SDF Modeler")
                .with_inner_size(slint::winit_030::winit::dpi::LogicalSize::new(
                    1280.0, 800.0,
                ))
        })
        .select()
        .expect("Failed to select Slint backend");

    // 3. Create Slint main window
    let app = MainWindow::new().expect("Failed to create MainWindow");

    // 4. Load project file if provided
    let mut scene = scene;
    let mut camera = camera;
    let mut current_file_path = None;
    if let Some(ref path) = file_path {
        match crate::io::load_project(path) {
            Ok(project) => {
                scene = project.scene;
                camera = project.camera;
                current_file_path = Some(path.clone());
                log::info!("Loaded project: {}", path.display());
            }
            Err(e) => {
                log::error!("Failed to load {}: {}", path.display(), e);
            }
        }
    }

    // 5. Create shared state
    let now = Instant::now();
    let shared = Rc::new(RefCell::new(AppSharedState {
        camera,
        scene,
        selected: None,
        history: History::new(),
        settings,
        current_structure_key: 0,
        buffer_dirty: true,
        ui_dirty: true,
        last_data_fingerprint: 0,
        voxel_gpu_offsets: HashMap::new(),
        sculpt_tex_indices: HashMap::new(),
        pending_pick: None,
        mouse: MouseState::default(),
        start_time: now,
        last_frame_time: now,
        current_file_path,
        viewport_w: 1280,
        viewport_h: 800,
    }));

    // 6. Set static dropdown models (once at startup)
    {
        let prim_names: Vec<SharedString> = SdfPrimitive::ALL.iter()
            .map(|k| SharedString::from(k.base_name()))
            .collect();
        app.set_primitive_types(ModelRc::new(VecModel::from(prim_names)));

        let op_names: Vec<SharedString> = CsgOp::ALL.iter()
            .map(|k| SharedString::from(k.base_name()))
            .collect();
        app.set_csg_op_types(ModelRc::new(VecModel::from(op_names)));

        let xform_names: Vec<SharedString> = TransformKind::ALL.iter()
            .map(|k| SharedString::from(k.base_name()))
            .collect();
        app.set_transform_types(ModelRc::new(VecModel::from(xform_names)));

        let mod_names: Vec<SharedString> = ModifierKind::ALL.iter()
            .map(|k| SharedString::from(k.base_name()))
            .collect();
        app.set_modifier_types(ModelRc::new(VecModel::from(mod_names)));
    }

    // 7. Wire Slint callbacks (use as_weak() to avoid borrow issues)

    // --- Viewport callbacks (from Phase 2) ---
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_viewport_pointer_event(move |event| {
            let mut state = s.borrow_mut();
            match event.button {
                0 => {
                    if event.pressed {
                        state.mouse.click_start = Some((event.x, event.y));
                    } else {
                        if let (Some(start), Some(end)) =
                            (state.mouse.click_start, state.mouse.last_position)
                        {
                            let dx = (end.0 - start.0).abs();
                            let dy = (end.1 - start.1).abs();
                            if dx < 5.0 && dy < 5.0 {
                                queue_pick(&mut state, end.0, end.1);
                            }
                        }
                        state.mouse.click_start = None;
                    }
                    state.mouse.left_pressed = event.pressed;
                }
                1 => state.mouse.right_pressed = event.pressed,
                2 => state.mouse.middle_pressed = event.pressed,
                _ => {}
            }
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_viewport_mouse_moved(move |x, y, _any_pressed| {
            let mut state = s.borrow_mut();
            if let Some((lx, ly)) = state.mouse.last_position {
                let dx = x - lx;
                let dy = y - ly;
                let mut needs_redraw = false;
                if state.mouse.left_pressed {
                    state.camera.orbit(dx, dy);
                    needs_redraw = true;
                }
                if state.mouse.right_pressed {
                    state.camera.pan(dx, dy);
                    needs_redraw = true;
                }
                if state.mouse.middle_pressed {
                    state.camera.orbit(dx, dy);
                    needs_redraw = true;
                }
                if needs_redraw {
                    if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
                }
            }
            state.mouse.last_position = Some((x, y));
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_viewport_scroll(move |event| {
            let mut state = s.borrow_mut();
            if event.delta_y != 0.0 {
                state.camera.zoom(event.delta_y);
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_viewport_key_pressed(move |key_text, ctrl, _shift| {
            let key = key_text.as_str();
            let mut needs_redraw = false;

            match key {
                "z" | "Z" if ctrl => {
                    let mut state = s.borrow_mut();
                    let st = &mut *state;
                    if let Some((rs, rsel)) = st.history.undo(&st.scene, st.selected) {
                        st.scene = rs;
                        st.selected = rsel;
                        st.current_structure_key = 0;
                        st.buffer_dirty = true;
                        st.ui_dirty = true;
                        needs_redraw = true;
                    }
                }
                "y" | "Y" if ctrl => {
                    let mut state = s.borrow_mut();
                    let st = &mut *state;
                    if let Some((rs, rsel)) = st.history.redo(&st.scene, st.selected) {
                        st.scene = rs;
                        st.selected = rsel;
                        st.current_structure_key = 0;
                        st.buffer_dirty = true;
                        st.ui_dirty = true;
                        needs_redraw = true;
                    }
                }
                "s" | "S" if ctrl => {
                    let existing = s.borrow().current_file_path.clone();
                    let path = existing.or_else(crate::io::save_dialog);
                    if let Some(path) = path {
                        let mut state = s.borrow_mut();
                        match crate::io::save_project(&state.scene, &state.camera, &path) {
                            Ok(()) => {
                                state.current_file_path = Some(path.clone());
                                log::info!("Saved to {}", path.display());
                            }
                            Err(e) => log::error!("Save failed: {}", e),
                        }
                    }
                }
                "o" | "O" if ctrl => {
                    if let Some(path) = crate::io::open_dialog() {
                        match crate::io::load_project(&path) {
                            Ok(project) => {
                                let mut state = s.borrow_mut();
                                state.scene = project.scene;
                                state.camera = project.camera;
                                state.selected = None;
                                state.current_file_path = Some(path.clone());
                                state.current_structure_key = 0;
                                state.buffer_dirty = true;
                                state.ui_dirty = true;
                                needs_redraw = true;
                                log::info!("Loaded {}", path.display());
                            }
                            Err(e) => log::error!("Load failed: {}", e),
                        }
                    }
                }
                "\u{007f}" | "\u{0008}" => {
                    let mut state = s.borrow_mut();
                    if let Some(sel) = state.selected.take() {
                        state.scene.remove_node(sel);
                        state.buffer_dirty = true;
                        state.ui_dirty = true;
                        needs_redraw = true;
                    }
                }
                "f" | "F" if !ctrl => {
                    let mut state = s.borrow_mut();
                    if let Some(sel) = state.selected {
                        let parent_map = state.scene.build_parent_map();
                        let (center, radius) =
                            state.scene.compute_subtree_sphere(sel, &parent_map);
                        state.camera.focus_on(glam::Vec3::from(center), radius);
                        needs_redraw = true;
                    }
                }
                _ => {}
            }

            if needs_redraw {
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }

    // --- Toolbar callbacks ---
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_toolbar_undo(move || {
            let mut state = s.borrow_mut();
            let st = &mut *state;
            if let Some((rs, rsel)) = st.history.undo(&st.scene, st.selected) {
                st.scene = rs;
                st.selected = rsel;
                st.current_structure_key = 0;
                st.buffer_dirty = true;
                st.ui_dirty = true;
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_toolbar_redo(move || {
            let mut state = s.borrow_mut();
            let st = &mut *state;
            if let Some((rs, rsel)) = st.history.redo(&st.scene, st.selected) {
                st.scene = rs;
                st.selected = rsel;
                st.current_structure_key = 0;
                st.buffer_dirty = true;
                st.ui_dirty = true;
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_toolbar_save(move || {
            let existing = s.borrow().current_file_path.clone();
            let path = existing.or_else(crate::io::save_dialog);
            if let Some(path) = path {
                let mut state = s.borrow_mut();
                match crate::io::save_project(&state.scene, &state.camera, &path) {
                    Ok(()) => {
                        state.current_file_path = Some(path.clone());
                        log::info!("Saved to {}", path.display());
                    }
                    Err(e) => log::error!("Save failed: {}", e),
                }
            }
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_toolbar_open(move || {
            if let Some(path) = crate::io::open_dialog() {
                match crate::io::load_project(&path) {
                    Ok(project) => {
                        let mut state = s.borrow_mut();
                        state.scene = project.scene;
                        state.camera = project.camera;
                        state.selected = None;
                        state.current_file_path = Some(path.clone());
                        state.current_structure_key = 0;
                        state.buffer_dirty = true;
                        state.ui_dirty = true;
                        log::info!("Loaded {}", path.display());
                    }
                    Err(e) => log::error!("Load failed: {}", e),
                }
            }
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_toolbar_new(move || {
            let mut state = s.borrow_mut();
            state.scene = Scene::new();
            state.selected = None;
            state.current_file_path = None;
            state.current_structure_key = 0;
            state.buffer_dirty = true;
            state.ui_dirty = true;
            state.history = History::new();
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_add_primitive(move |kind_idx| {
            let mut state = s.borrow_mut();
            if let Some(kind) = SdfPrimitive::ALL.get(kind_idx as usize) {
                let id = state.scene.create_primitive(kind.clone());
                state.selected = Some(id);
                state.buffer_dirty = true;
                state.ui_dirty = true;
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_add_operation(move |op_idx| {
            let mut state = s.borrow_mut();
            if let Some(op) = CsgOp::ALL.get(op_idx as usize) {
                let id = state.scene.create_operation(op.clone(), None, None);
                state.selected = Some(id);
                state.buffer_dirty = true;
                state.ui_dirty = true;
                if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
            }
        });
    }

    // --- Scene tree callbacks ---
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_tree_select_node(move |node_id| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            if state.scene.nodes.contains_key(&id) {
                state.selected = Some(id);
            } else {
                state.selected = None;
            }
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_tree_toggle_visibility(move |node_id| {
            let mut state = s.borrow_mut();
            state.scene.toggle_visibility(node_id as NodeId);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }

    // --- Properties callbacks ---
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_update_name(move |node_id, name| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            if let Some(node) = state.scene.nodes.get_mut(&id) {
                node.name = name.to_string();
            }
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_update_kind(move |node_id, kind_idx| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            app_slint_sync::apply_kind_update(&mut state.scene, id, kind_idx);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_update_float(move |node_id, param, value| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            app_slint_sync::apply_float_update(&mut state.scene, id, param, value);
            state.buffer_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_update_vec3(move |node_id, param_group, axis, text| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            app_slint_sync::apply_vec3_update(&mut state.scene, id, param_group, axis, text.as_str());
            state.buffer_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_delete_node(move |node_id| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            state.scene.remove_node(id);
            if state.selected == Some(id) {
                state.selected = None;
            }
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_props_swap_inputs(move |node_id| {
            let mut state = s.borrow_mut();
            let id = node_id as NodeId;
            state.scene.swap_children(id);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }

    // --- Render settings callbacks ---
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_update_bool(move |param, value| {
            let mut state = s.borrow_mut();
            app_slint_sync::apply_render_bool(&mut state.settings.render, param, value);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_update_float(move |param, value| {
            let mut state = s.borrow_mut();
            app_slint_sync::apply_render_float(&mut state.settings.render, param, value);
            state.buffer_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_update_int(move |param, value| {
            let mut state = s.borrow_mut();
            app_slint_sync::apply_render_int(&mut state.settings.render, param, value);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_apply_preset(move |preset| {
            let mut state = s.borrow_mut();
            app_slint_sync::apply_render_preset(&mut state.settings.render, preset);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_reset_section(move |section| {
            let mut state = s.borrow_mut();
            app_slint_sync::apply_render_reset_section(&mut state.settings.render, section);
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }
    {
        let aw = app.as_weak();
        let s = shared.clone();
        app.on_render_reset_all(move || {
            let mut state = s.borrow_mut();
            state.settings.render = RenderConfig::default();
            state.buffer_dirty = true;
            state.ui_dirty = true;
            if let Some(a) = aw.upgrade() { a.window().request_redraw(); }
        });
    }

    // 8. Rendering notifier — renders SDF to texture each frame
    let shared_render = shared.clone();
    let app_weak = app.as_weak();
    let device_for_render = device.clone();
    let queue_for_render = queue.clone();
    let mut renderer: Option<SdfRenderer> = None;

    app.window()
        .set_rendering_notifier(move |state, _graphics_api| match state {
            slint::RenderingState::RenderingSetup => {
                let s = shared_render.borrow();
                let shader_src = codegen::generate_shader(&s.scene, &s.settings.render);
                let pick_shader_src =
                    codegen::generate_pick_shader(&s.scene, &s.settings.render);

                let mut viewport = ViewportResources::new(
                    &device_for_render,
                    wgpu::TextureFormat::Rgba8Unorm,
                    &shader_src,
                    &pick_shader_src,
                );

                // Initial buffer upload
                let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&s.scene);
                let node_data = buffers::build_node_buffer(&s.scene, None, &voxel_offsets);
                viewport.update_scene_buffer(&device_for_render, &queue_for_render, &node_data);
                viewport.update_voxel_buffer(&device_for_render, &queue_for_render, &voxel_data);

                let sculpt_infos = buffers::collect_sculpt_tex_info(&s.scene);
                for info in &sculpt_infos {
                    if let Some(node) = s.scene.nodes.get(&info.node_id) {
                        if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                            viewport.upload_voxel_texture(
                                &device_for_render,
                                &queue_for_render,
                                info.tex_idx,
                                voxel_grid.resolution,
                                &voxel_grid.data,
                            );
                        }
                    }
                }

                drop(s);
                let mut s = shared_render.borrow_mut();
                s.voxel_gpu_offsets = voxel_offsets;
                s.sculpt_tex_indices =
                    sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();
                s.current_structure_key = s.scene.structure_key();
                s.buffer_dirty = false;

                renderer = Some(SdfRenderer { viewport });
                log::info!("SDF renderer initialized (Rgba8Unorm)");
            }
            slint::RenderingState::BeforeRendering => {
                let Some(ref mut rend) = renderer else {
                    return;
                };
                render_frame(
                    &device_for_render,
                    &queue_for_render,
                    rend,
                    &shared_render,
                    &app_weak,
                );
            }
            slint::RenderingState::RenderingTeardown => {
                renderer = None;
                log::info!("SDF renderer torn down");
            }
            _ => {}
        })
        .expect("Failed to set rendering notifier");

    // Request initial draw
    app.window().request_redraw();

    // 9. Run Slint event loop
    app.run().expect("Slint event loop error");
}

// ---------------------------------------------------------------------------
// Render frame (called from BeforeRendering)
// ---------------------------------------------------------------------------

fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rend: &mut SdfRenderer,
    shared: &Rc<RefCell<AppSharedState>>,
    app_weak: &slint::Weak<MainWindow>,
) {
    let Some(app) = app_weak.upgrade() else {
        return;
    };

    let mut s = shared.borrow_mut();

    // Read viewport dimensions from Slint
    let scale = app.window().scale_factor();
    let vp_w = (app.get_viewport_width() * scale).max(1.0) as u32;
    let vp_h = (app.get_viewport_height() * scale).max(1.0) as u32;
    s.viewport_w = vp_w;
    s.viewport_h = vp_h;

    let now = Instant::now();
    s.last_frame_time = now;

    // Undo/redo frame bookkeeping (destructure for disjoint borrows)
    {
        let st = &mut *s;
        st.history.begin_frame(&st.scene, st.selected);
    }

    // Sync GPU pipeline if scene topology changed
    let new_key = s.scene.structure_key();
    if new_key != s.current_structure_key {
        let shader_src = codegen::generate_shader(&s.scene, &s.settings.render);
        let pick_shader_src = codegen::generate_pick_shader(&s.scene, &s.settings.render);
        let sculpt_count = buffers::collect_sculpt_tex_info(&s.scene).len();
        rend.viewport
            .rebuild_pipeline(device, &shader_src, &pick_shader_src, sculpt_count);
        s.current_structure_key = new_key;
        s.buffer_dirty = true;
    }

    // Process pending pick
    if let Some(pending) = s.pending_pick.take() {
        let topo_order = s.scene.visible_topo_order();
        if let Some(result) = rend.viewport.execute_pick(device, queue, &pending) {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                s.selected = Some(topo_order[idx]);
                s.buffer_dirty = true;
                s.ui_dirty = true;
            }
        } else {
            s.selected = None;
            s.buffer_dirty = true;
            s.ui_dirty = true;
        }
    }

    // Detect data changes
    let fp = s.scene.data_fingerprint();
    if fp != s.last_data_fingerprint {
        s.last_data_fingerprint = fp;
        s.buffer_dirty = true;
    }

    // Upload scene buffer if dirty
    if s.buffer_dirty {
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&s.scene);
        let node_data = buffers::build_node_buffer(&s.scene, s.selected, &voxel_offsets);
        let sculpt_infos = buffers::collect_sculpt_tex_info(&s.scene);
        s.voxel_gpu_offsets = voxel_offsets;
        s.sculpt_tex_indices =
            sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();

        rend.viewport
            .update_scene_buffer(device, queue, &node_data);
        rend.viewport
            .update_voxel_buffer(device, queue, &voxel_data);

        for info in &sculpt_infos {
            if let Some(node) = s.scene.nodes.get(&info.node_id) {
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    rend.viewport.upload_voxel_texture(
                        device,
                        queue,
                        info.tex_idx,
                        voxel_grid.resolution,
                        &voxel_grid.data,
                    );
                }
            }
        }
        s.buffer_dirty = false;
    }

    // Camera uniform
    let viewport_rect = [0.0, 0.0, vp_w as f32, vp_h as f32];
    let scene_bounds = s.scene.compute_bounds();
    let selected_idx = s
        .selected
        .and_then(|id| {
            let order = s.scene.visible_topo_order();
            order.iter().position(|&nid| nid == id)
        })
        .map(|i| i as f32)
        .unwrap_or(-1.0);
    let elapsed = s.start_time.elapsed().as_secs_f32();
    let render_uniform = s.camera.to_uniform(
        viewport_rect,
        elapsed,
        0.0,
        s.settings.render.show_grid,
        scene_bounds,
        selected_idx,
    );

    // End undo/redo frame (destructure for disjoint borrows)
    {
        let st = &mut *s;
        let is_dragging =
            st.mouse.left_pressed || st.mouse.right_pressed || st.mouse.middle_pressed;
        st.history.end_frame(&st.scene, st.selected, is_dragging);
    }

    // Ensure offscreen texture (Rgba8Unorm)
    rend.viewport.ensure_offscreen_texture(device, vp_w, vp_h);

    // Write camera uniform
    queue.write_buffer(
        &rend.viewport.camera_buffer,
        0,
        bytemuck::bytes_of(&render_uniform),
    );

    // Encode SDF render pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SDF Frame Encoder"),
    });

    {
        let offscreen_view = rend.viewport.offscreen_view.as_ref().unwrap();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SDF Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: offscreen_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if rend.viewport.use_composite {
            if let Some(ref comp) = rend.viewport.composite {
                pass.set_pipeline(&comp.render_pipeline);
                pass.set_bind_group(0, &rend.viewport.camera_bind_group, &[]);
                pass.set_bind_group(1, &rend.viewport.scene_bind_group, &[]);
                pass.set_bind_group(2, &comp.render_bg, &[]);
            }
        } else {
            pass.set_pipeline(&rend.viewport.pipeline);
            pass.set_bind_group(0, &rend.viewport.camera_bind_group, &[]);
            pass.set_bind_group(1, &rend.viewport.scene_bind_group, &[]);
            pass.set_bind_group(2, &rend.viewport.voxel_tex_bind_group, &[]);
        }
        pass.draw(0..3, 0..1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Convert offscreen texture to Slint Image
    let ui_dirty = s.ui_dirty;
    let selected = s.selected;
    if let Some(ref texture) = rend.viewport.offscreen_texture {
        match slint::Image::try_from(texture.clone()) {
            Ok(image) => {
                // Build UI data while we still hold the borrow
                let tree_data = if ui_dirty {
                    Some(app_slint_sync::build_scene_tree_flat(&s.scene, selected))
                } else {
                    None
                };
                let props_data = if ui_dirty {
                    Some(app_slint_sync::build_selected_props(&s.scene, selected))
                } else {
                    None
                };
                let render_data = if ui_dirty {
                    Some(app_slint_sync::build_render_settings(&s.settings.render))
                } else {
                    None
                };
                s.ui_dirty = false;
                drop(s); // release borrow before setting Slint properties

                app.set_viewport_image(image);

                // Push UI data to Slint
                if let Some(tree) = tree_data {
                    sync_scene_tree(&app, &tree);
                }
                if let Some(props) = props_data {
                    sync_selected_props(&app, &props);
                }
                if let Some(rs) = render_data {
                    sync_render_settings(&app, &rs);
                }
                app.set_has_selection(selected.is_some());
            }
            Err(e) => {
                log::error!("Failed to import texture to Slint: {:?}", e);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn queue_pick(state: &mut AppSharedState, x: f32, y: f32) {
    let w = state.viewport_w.max(1) as f32;
    let h = state.viewport_h.max(1) as f32;
    let viewport = [0.0, 0.0, w, h];
    let scene_bounds = state.scene.compute_bounds();
    let elapsed = state.start_time.elapsed().as_secs_f32();
    let pick_uniform =
        state
            .camera
            .to_uniform(viewport, elapsed, 0.0, false, scene_bounds, -1.0);
    state.pending_pick = Some(PendingPick {
        mouse_pos: [x, y],
        camera_uniform: pick_uniform,
    });
}

// ---------------------------------------------------------------------------
// Slint UI sync helpers (push Rust state → Slint properties)
// ---------------------------------------------------------------------------

fn sync_scene_tree(app: &MainWindow, data: &[app_slint_sync::SceneNodeInfoData]) {
    let items: Vec<SceneNodeInfo> = data
        .iter()
        .map(|d| SceneNodeInfo {
            id: d.id,
            name: SharedString::from(&d.name),
            node_type: SharedString::from(&d.node_type),
            badge: SharedString::from(&d.badge),
            depth: d.depth,
            is_selected: d.is_selected,
            is_hidden: d.is_hidden,
            has_children: d.has_children,
        })
        .collect();
    app.set_scene_tree_nodes(ModelRc::new(VecModel::from(items)));
}

fn sync_selected_props(app: &MainWindow, p: &app_slint_sync::SelectedNodePropsData) {
    app.set_selected_props(SelectedNodeProps {
        id: p.id,
        name: SharedString::from(&p.name),
        node_type: p.node_type,
        primitive_kind: p.primitive_kind,
        csg_op: p.csg_op,
        smooth_k: p.smooth_k,
        left_name: SharedString::from(&p.left_name),
        right_name: SharedString::from(&p.right_name),
        has_left: p.has_left,
        has_right: p.has_right,
        transform_kind: p.transform_kind,
        modifier_kind: p.modifier_kind,
        pos_x: p.pos_x,
        pos_y: p.pos_y,
        pos_z: p.pos_z,
        rot_x: p.rot_x,
        rot_y: p.rot_y,
        rot_z: p.rot_z,
        scale_x: p.scale_x,
        scale_y: p.scale_y,
        scale_z: p.scale_z,
        color_r: p.color_r,
        color_g: p.color_g,
        color_b: p.color_b,
        roughness: p.roughness,
        metallic: p.metallic,
        fresnel: p.fresnel,
        emissive_r: p.emissive_r,
        emissive_g: p.emissive_g,
        emissive_b: p.emissive_b,
        emissive_intensity: p.emissive_intensity,
        value_x: p.value_x,
        value_y: p.value_y,
        value_z: p.value_z,
        extra_x: p.extra_x,
        extra_y: p.extra_y,
        extra_z: p.extra_z,
        scale_label_x: SharedString::from(&p.scale_label_x),
        scale_label_y: SharedString::from(&p.scale_label_y),
        scale_label_z: SharedString::from(&p.scale_label_z),
        scale_count: p.scale_count,
    });
}

fn sync_render_settings(app: &MainWindow, r: &app_slint_sync::RenderSettingsDataRust) {
    app.set_render_settings(RenderSettingsData {
        shadows_enabled: r.shadows_enabled,
        shadow_steps: r.shadow_steps,
        shadow_penumbra_k: r.shadow_penumbra_k,
        shadow_bias: r.shadow_bias,
        shadow_mint: r.shadow_mint,
        shadow_maxt: r.shadow_maxt,
        ao_enabled: r.ao_enabled,
        ao_samples: r.ao_samples,
        ao_step: r.ao_step,
        ao_decay: r.ao_decay,
        ao_intensity: r.ao_intensity,
        march_max_steps: r.march_max_steps,
        march_epsilon: r.march_epsilon,
        march_step_multiplier: r.march_step_multiplier,
        march_max_distance: r.march_max_distance,
        key_light_dir_x: r.key_light_dir_x,
        key_light_dir_y: r.key_light_dir_y,
        key_light_dir_z: r.key_light_dir_z,
        key_diffuse: r.key_diffuse,
        key_spec_power: r.key_spec_power,
        key_spec_intensity: r.key_spec_intensity,
        fill_light_dir_x: r.fill_light_dir_x,
        fill_light_dir_y: r.fill_light_dir_y,
        fill_light_dir_z: r.fill_light_dir_z,
        fill_intensity: r.fill_intensity,
        ambient: r.ambient,
        sky_horizon_r: r.sky_horizon_r,
        sky_horizon_g: r.sky_horizon_g,
        sky_horizon_b: r.sky_horizon_b,
        sky_zenith_r: r.sky_zenith_r,
        sky_zenith_g: r.sky_zenith_g,
        sky_zenith_b: r.sky_zenith_b,
        fog_enabled: r.fog_enabled,
        fog_density: r.fog_density,
        fog_color_r: r.fog_color_r,
        fog_color_g: r.fog_color_g,
        fog_color_b: r.fog_color_b,
        gamma: r.gamma,
        tonemapping_aces: r.tonemapping_aces,
        outline_color_r: r.outline_color_r,
        outline_color_g: r.outline_color_g,
        outline_color_b: r.outline_color_b,
        outline_thickness: r.outline_thickness,
        show_grid: r.show_grid,
    });
}
