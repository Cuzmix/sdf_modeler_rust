use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::compat::Instant;
use crate::gpu::buffers;
use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::Settings;
use crate::ui::viewport::ViewportResources;

// ---------------------------------------------------------------------------
// Mouse state tracking
// ---------------------------------------------------------------------------

struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    middle_pressed: bool,
    last_position: Option<(f64, f64)>,
    /// Where the mouse was when button was first pressed (for click vs drag).
    click_start: Option<(f64, f64)>,
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
// Two-phase app state (PreInit → Running)
// ---------------------------------------------------------------------------

pub enum AppState {
    PreInit {
        settings: Settings,
        scene: Scene,
        camera: Camera,
        file_path: Option<PathBuf>,
    },
    Running(SdfWgpuApp),
    /// Temporary state during transition.
    Transitioning,
}

// ---------------------------------------------------------------------------
// Main application struct
// ---------------------------------------------------------------------------

pub struct SdfWgpuApp {
    // GPU
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Rendering
    viewport: ViewportResources,

    // Data
    camera: Camera,
    scene: Scene,
    selected: Option<NodeId>,
    history: History,
    settings: Settings,

    // Sync
    current_structure_key: u64,
    buffer_dirty: bool,
    last_data_fingerprint: u64,
    voxel_gpu_offsets: HashMap<NodeId, u32>,
    sculpt_tex_indices: HashMap<NodeId, usize>,

    // Pick
    pending_pick: Option<PendingPick>,

    // Input
    mouse: MouseState,
    modifiers: ModifiersState,

    // Timing
    start_time: Instant,
    last_frame_time: Instant,

    // File
    current_file_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// ApplicationHandler — delegates to SdfWgpuApp
// ---------------------------------------------------------------------------

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Only initialize once (PreInit → Running)
        let AppState::PreInit { .. } = self else {
            return;
        };

        // Take ownership of PreInit data
        let old = std::mem::replace(self, AppState::Transitioning);
        let AppState::PreInit {
            settings,
            mut scene,
            mut camera,
            file_path,
        } = old
        else {
            unreachable!();
        };

        // Load project file if provided
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

        // Create window
        let attrs = WindowAttributes::default()
            .with_title("SDF Modeler")
            .with_inner_size(PhysicalSize::new(1280u32, 800u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));

        // Create wgpu instance + adapter + device
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
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

        // Configure surface
        let caps = surface.get_capabilities(&adapter);
        let target_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: target_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if settings.vsync_enabled {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        // Generate initial shaders and create viewport resources
        let shader_src = codegen::generate_shader(&scene, &settings.render);
        let pick_shader_src = codegen::generate_pick_shader(&scene, &settings.render);
        let structure_key = scene.structure_key();

        let mut viewport = ViewportResources::new(
            &device,
            target_format,
            &shader_src,
            &pick_shader_src,
        );

        // Upload initial scene buffers
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&scene);
        let node_data = buffers::build_node_buffer(&scene, None, &voxel_offsets);
        viewport.update_scene_buffer(&device, &queue, &node_data);
        viewport.update_voxel_buffer(&device, &queue, &voxel_data);

        let sculpt_infos = buffers::collect_sculpt_tex_info(&scene);
        let sculpt_tex_indices: HashMap<NodeId, usize> =
            sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();

        // Upload voxel textures
        for info in &sculpt_infos {
            if let Some(node) = scene.nodes.get(&info.node_id) {
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    viewport.upload_voxel_texture(
                        &device,
                        &queue,
                        info.tex_idx,
                        voxel_grid.resolution,
                        &voxel_grid.data,
                    );
                }
            }
        }

        let now = Instant::now();

        *self = AppState::Running(SdfWgpuApp {
            window,
            device,
            queue,
            surface,
            surface_config,
            viewport,
            camera,
            scene,
            selected: None,
            history: History::new(),
            settings,
            current_structure_key: structure_key,
            buffer_dirty: false,
            last_data_fingerprint: 0,
            voxel_gpu_offsets: voxel_offsets,
            sculpt_tex_indices,
            pending_pick: None,
            mouse: MouseState::default(),
            modifiers: ModifiersState::empty(),
            start_time: now,
            last_frame_time: now,
            current_file_path,
        });

        // Request initial draw
        if let AppState::Running(ref app) = self {
            app.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running(ref mut app) = self else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                app.handle_resize(new_size);
                app.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                app.render();
            }
            WindowEvent::MouseInput { button, state, .. } => {
                app.handle_mouse_input(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                app.handle_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                app.handle_mouse_wheel(delta);
            }
            WindowEvent::ModifiersChanged(mods) => {
                app.modifiers = mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    app.handle_key_input(&event, event_loop);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// SdfWgpuApp implementation
// ---------------------------------------------------------------------------

impl SdfWgpuApp {
    fn handle_resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => {
                if pressed {
                    self.mouse.click_start = self.mouse.last_position;
                } else {
                    // Detect click (small movement)
                    if let (Some(start), Some(end)) =
                        (self.mouse.click_start, self.mouse.last_position)
                    {
                        let dx = (end.0 - start.0).abs();
                        let dy = (end.1 - start.1).abs();
                        if dx < 5.0 && dy < 5.0 {
                            self.queue_pick(end.0, end.1);
                        }
                    }
                    self.mouse.click_start = None;
                }
                self.mouse.left_pressed = pressed;
            }
            MouseButton::Right => {
                self.mouse.right_pressed = pressed;
            }
            MouseButton::Middle => {
                self.mouse.middle_pressed = pressed;
            }
            _ => {}
        }
    }

    fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        if let Some((lx, ly)) = self.mouse.last_position {
            let dx = (x - lx) as f32;
            let dy = (y - ly) as f32;

            if self.mouse.left_pressed {
                self.camera.orbit(dx, dy);
                self.window.request_redraw();
            }
            if self.mouse.right_pressed {
                self.camera.pan(dx, dy);
                self.window.request_redraw();
            }
            if self.mouse.middle_pressed {
                self.camera.orbit(dx, dy);
                self.window.request_redraw();
            }
        }
        self.mouse.last_position = Some((x, y));
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 40.0,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        if scroll != 0.0 {
            self.camera.zoom(scroll);
            self.window.request_redraw();
        }
    }

    fn handle_key_input(
        &mut self,
        event: &winit::event::KeyEvent,
        event_loop: &ActiveEventLoop,
    ) {
        let ctrl = self.modifiers.control_key();
        let PhysicalKey::Code(key) = event.physical_key else {
            return;
        };

        match key {
            // Undo
            KeyCode::KeyZ if ctrl => {
                if let Some((restored_scene, restored_sel)) =
                    self.history.undo(&self.scene, self.selected)
                {
                    self.scene = restored_scene;
                    self.selected = restored_sel;
                    self.current_structure_key = 0; // Force pipeline rebuild
                    self.buffer_dirty = true;
                    self.window.request_redraw();
                }
            }
            // Redo
            KeyCode::KeyY if ctrl => {
                if let Some((restored_scene, restored_sel)) =
                    self.history.redo(&self.scene, self.selected)
                {
                    self.scene = restored_scene;
                    self.selected = restored_sel;
                    self.current_structure_key = 0;
                    self.buffer_dirty = true;
                    self.window.request_redraw();
                }
            }
            // Save
            KeyCode::KeyS if ctrl => {
                let path = self
                    .current_file_path
                    .clone()
                    .or_else(crate::io::save_dialog);
                if let Some(path) = path {
                    match crate::io::save_project(&self.scene, &self.camera, &path) {
                        Ok(()) => {
                            self.current_file_path = Some(path.clone());
                            log::info!("Saved to {}", path.display());
                        }
                        Err(e) => log::error!("Save failed: {}", e),
                    }
                }
            }
            // Open
            KeyCode::KeyO if ctrl => {
                if let Some(path) = crate::io::open_dialog() {
                    match crate::io::load_project(&path) {
                        Ok(project) => {
                            self.scene = project.scene;
                            self.camera = project.camera;
                            self.selected = None;
                            self.current_file_path = Some(path.clone());
                            self.current_structure_key = 0;
                            self.buffer_dirty = true;
                            self.window.request_redraw();
                            log::info!("Loaded {}", path.display());
                        }
                        Err(e) => log::error!("Load failed: {}", e),
                    }
                }
            }
            // Delete selected node
            KeyCode::Delete | KeyCode::Backspace => {
                if let Some(sel) = self.selected.take() {
                    self.scene.remove_node(sel);
                    self.buffer_dirty = true;
                    self.window.request_redraw();
                }
            }
            // Camera presets
            KeyCode::F5 => {
                self.camera.set_front();
                self.window.request_redraw();
            }
            KeyCode::F6 => {
                self.camera.set_top();
                self.window.request_redraw();
            }
            KeyCode::F7 => {
                self.camera.set_right();
                self.window.request_redraw();
            }
            // Focus on selected
            KeyCode::KeyF if !ctrl => {
                if let Some(sel) = self.selected {
                    let parent_map = self.scene.build_parent_map();
                    let (center, radius) = self.scene.compute_subtree_sphere(sel, &parent_map);
                    self.camera.focus_on(glam::Vec3::from(center), radius);
                    self.window.request_redraw();
                }
            }
            // Escape exits
            KeyCode::Escape => {
                event_loop.exit();
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // GPU sync (ported from app/gpu_sync.rs)
    // -----------------------------------------------------------------------

    fn sync_gpu_pipeline(&mut self) {
        let new_key = self.scene.structure_key();
        if new_key != self.current_structure_key {
            let shader_src = codegen::generate_shader(&self.scene, &self.settings.render);
            let pick_shader_src =
                codegen::generate_pick_shader(&self.scene, &self.settings.render);
            let sculpt_count = buffers::collect_sculpt_tex_info(&self.scene).len();
            self.viewport.rebuild_pipeline(
                &self.device,
                &shader_src,
                &pick_shader_src,
                sculpt_count,
            );
            self.current_structure_key = new_key;
            self.buffer_dirty = true;
        }
    }

    fn upload_scene_buffer(&mut self) {
        let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&self.scene);
        let node_data =
            buffers::build_node_buffer(&self.scene, self.selected, &voxel_offsets);
        let sculpt_infos = buffers::collect_sculpt_tex_info(&self.scene);
        self.voxel_gpu_offsets = voxel_offsets;
        self.sculpt_tex_indices =
            sculpt_infos.iter().map(|i| (i.node_id, i.tex_idx)).collect();

        self.viewport
            .update_scene_buffer(&self.device, &self.queue, &node_data);
        self.viewport
            .update_voxel_buffer(&self.device, &self.queue, &voxel_data);

        // Upload voxel textures for render shader
        for info in &sculpt_infos {
            if let Some(node) = self.scene.nodes.get(&info.node_id) {
                if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                    self.viewport.upload_voxel_texture(
                        &self.device,
                        &self.queue,
                        info.tex_idx,
                        voxel_grid.resolution,
                        &voxel_grid.data,
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pick (ported from app/gpu_sync.rs)
    // -----------------------------------------------------------------------

    fn queue_pick(&mut self, x: f64, y: f64) {
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let viewport = [0.0, 0.0, w, h];
        let scene_bounds = self.scene.compute_bounds();
        let pick_uniform =
            self.camera
                .to_uniform(viewport, self.elapsed_time(), 0.0, false, scene_bounds, -1.0);
        self.pending_pick = Some(PendingPick {
            mouse_pos: [x as f32, y as f32],
            camera_uniform: pick_uniform,
        });
        self.window.request_redraw();
    }

    fn process_pending_pick(&mut self) {
        let Some(pending) = self.pending_pick.take() else {
            return;
        };
        let topo_order = self.scene.visible_topo_order();
        if let Some(result) =
            self.viewport
                .execute_pick(&self.device, &self.queue, &pending)
        {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                self.selected = Some(topo_order[idx]);
                self.buffer_dirty = true;
            }
        } else {
            self.selected = None;
            self.buffer_dirty = true;
        }
    }

    // -----------------------------------------------------------------------
    // Render frame
    // -----------------------------------------------------------------------

    fn render(&mut self) {
        let now = Instant::now();
        let _dt = now.duration_since(self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;

        // Undo/redo frame bookkeeping
        self.history.begin_frame(&self.scene, self.selected);

        // Sync GPU pipeline if scene topology changed
        self.sync_gpu_pipeline();

        // Process any pending pick
        self.process_pending_pick();

        // Detect data changes
        let fp = self.scene.data_fingerprint();
        if fp != self.last_data_fingerprint {
            self.last_data_fingerprint = fp;
            self.buffer_dirty = true;
        }

        // Upload scene buffer if dirty
        if self.buffer_dirty {
            self.upload_scene_buffer();
            self.buffer_dirty = false;
        }

        // Acquire swapchain texture
        let surface_texture = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {:?}", e);
                return;
            }
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let w = self.surface_config.width;
        let h = self.surface_config.height;

        // Camera uniform
        let viewport = [0.0, 0.0, w as f32, h as f32];
        let scene_bounds = self.scene.compute_bounds();
        let selected_idx = self
            .selected
            .and_then(|id| {
                let order = self.scene.visible_topo_order();
                order.iter().position(|&nid| nid == id)
            })
            .map(|i| i as f32)
            .unwrap_or(-1.0);
        let render_uniform = self.camera.to_uniform(
            viewport,
            self.elapsed_time(),
            0.0, // quality_mode: full
            self.settings.render.show_grid,
            scene_bounds,
            selected_idx,
        );

        // Ensure offscreen texture matches window size
        self.viewport.ensure_offscreen_texture(&self.device, w, h);

        // Write camera uniform
        self.queue.write_buffer(
            &self.viewport.camera_buffer,
            0,
            bytemuck::bytes_of(&render_uniform),
        );

        // Write blit params (display viewport)
        self.queue.write_buffer(
            &self.viewport.blit_params_buffer,
            0,
            bytemuck::cast_slice(&viewport),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        // Pass 1: Render SDF to offscreen texture
        {
            let offscreen_view = self.viewport.offscreen_view.as_ref().unwrap();
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

            if self.viewport.use_composite {
                if let Some(ref comp) = self.viewport.composite {
                    pass.set_pipeline(&comp.render_pipeline);
                    pass.set_bind_group(0, &self.viewport.camera_bind_group, &[]);
                    pass.set_bind_group(1, &self.viewport.scene_bind_group, &[]);
                    pass.set_bind_group(2, &comp.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&self.viewport.pipeline);
                pass.set_bind_group(0, &self.viewport.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.viewport.scene_bind_group, &[]);
                pass.set_bind_group(2, &self.viewport.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Blit offscreen to swapchain
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
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

            if let Some(ref blit_bg) = self.viewport.blit_bind_group {
                pass.set_pipeline(&self.viewport.blit_pipeline);
                pass.set_bind_group(0, blit_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        // End undo/redo frame
        let is_dragging =
            self.mouse.left_pressed || self.mouse.right_pressed || self.mouse.middle_pressed;
        self.history
            .end_frame(&self.scene, self.selected, is_dragging);
    }

    fn elapsed_time(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }
}
