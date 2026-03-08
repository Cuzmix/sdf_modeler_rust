use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::wgpu;
use pollster::block_on;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::core::{AppCore, AppCoreInit, CoreAsyncState, CoreCommand, CoreSelection};
use crate::gpu::buffers;
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeId, Scene, SdfPrimitive};
use crate::sculpt::{ActiveTool, SculptState};
use crate::settings::Settings;
use crate::ui::viewport::ViewportResources;

const VIEWPORT_TOP_BAR_PX: f32 = 52.0;
const VIEWPORT_STATUS_BAR_PX: f32 = 34.0;
const VIEWPORT_RIGHT_PANEL_PX: f32 = 336.0;
const CLICK_DRAG_THRESHOLD_PX: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ViewportRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl ViewportRect {
    fn contains(self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px <= self.x + self.width && py <= self.y + self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ViewportPointer {
    u: f32,
    v: f32,
    local_x: f32,
    local_y: f32,
}

fn compute_viewport_rect(surface_width: u32, surface_height: u32) -> ViewportRect {
    let total_w = surface_width as f32;
    let total_h = surface_height as f32;
    let x = 0.0;
    let y = VIEWPORT_TOP_BAR_PX.min(total_h - 1.0).max(0.0);
    let width = (total_w - VIEWPORT_RIGHT_PANEL_PX).max(1.0);
    let height = (total_h - VIEWPORT_TOP_BAR_PX - VIEWPORT_STATUS_BAR_PX).max(1.0);
    ViewportRect {
        x,
        y,
        width,
        height,
    }
}

fn map_pointer_to_viewport(px: f32, py: f32, viewport: ViewportRect) -> Option<ViewportPointer> {
    if !viewport.contains(px, py) || viewport.width <= 0.0 || viewport.height <= 0.0 {
        return None;
    }

    let local_x = px - viewport.x;
    let local_y = py - viewport.y;
    Some(ViewportPointer {
        u: (local_x / viewport.width).clamp(0.0, 1.0),
        v: (local_y / viewport.height).clamp(0.0, 1.0),
        local_x,
        local_y,
    })
}

fn benchmark_seconds_from_args(default_seconds: u64) -> u64 {
    let args: Vec<String> = std::env::args().collect();
    for (idx, arg) in args.iter().enumerate() {
        if let Some(raw) = arg.strip_prefix("--benchmark-seconds=") {
            if let Ok(seconds) = raw.parse::<u64>() {
                return seconds.max(1);
            }
        }
        if arg == "--benchmark-seconds" {
            if let Some(next) = args.get(idx + 1) {
                if let Ok(seconds) = next.parse::<u64>() {
                    return seconds.max(1);
                }
            }
        }
    }
    default_seconds
}

#[derive(Debug, Clone)]
struct FramePacingTracker {
    start_time: Instant,
    last_frame_time: Option<Instant>,
    busy_time: Duration,
    frame_intervals_ms: Vec<f32>,
    frame_count: u64,
}

impl Default for FramePacingTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FramePacingTracker {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            last_frame_time: None,
            busy_time: Duration::ZERO,
            frame_intervals_ms: Vec::new(),
            frame_count: 0,
        }
    }

    fn reset(&mut self) {
        self.start_time = Instant::now();
        self.last_frame_time = None;
        self.busy_time = Duration::ZERO;
        self.frame_intervals_ms.clear();
        self.frame_count = 0;
    }

    fn record_frame(&mut self, work_time: Duration) {
        let now = Instant::now();
        if let Some(previous) = self.last_frame_time {
            self.frame_intervals_ms.push((now - previous).as_secs_f32() * 1000.0);
        }
        self.last_frame_time = Some(now);
        self.busy_time += work_time;
        self.frame_count += 1;
    }

    fn build_report(&self, model_name: &str) -> String {
        let elapsed_s = self.start_time.elapsed().as_secs_f32().max(0.0001);
        let fps_avg = self.frame_count as f32 / elapsed_s;
        let idle_ratio = (1.0 - (self.busy_time.as_secs_f32() / elapsed_s)).clamp(0.0, 1.0) * 100.0;

        let mut sorted = self.frame_intervals_ms.clone();
        sorted.sort_by(|left, right| left.total_cmp(right));

        let frame_ms_avg = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().sum::<f32>() / sorted.len() as f32
        };

        let frame_ms_p95 = percentile(&sorted, 0.95);
        let frame_ms_p99 = percentile(&sorted, 0.99);

        format!(
            "model={model_name} frames={} elapsed_s={:.2} fps_avg={:.2} frame_ms_avg={:.2} frame_ms_p95={:.2} frame_ms_p99={:.2} idle_cpu_estimate_pct={:.2}",
            self.frame_count, elapsed_s, fps_avg, frame_ms_avg, frame_ms_p95, frame_ms_p99, idle_ratio
        )
    }
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let clamped = pct.clamp(0.0, 1.0);
    let max_index = (sorted.len() - 1) as f32;
    let idx = (max_index * clamped).round() as usize;
    sorted[idx]
}
fn frame_stats_title(fps: f32, core: &AppCore, pointer_hint: Option<ViewportPointer>) -> String {
    let selected = core
        .selection
        .primary
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let pointer = pointer_hint
        .map(|p| format!("u={:.3} v={:.3}", p.u, p.v))
        .unwrap_or_else(|| "outside viewport".to_string());
    format!(
        "SDF Modeler (Slint host) | FPS: {:.1} | Nodes: {} | Sel: {} | {}",
        fps,
        core.scene.nodes.len(),
        selected,
        pointer
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragKind {
    Orbit,
    Pan,
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    kind: DragKind,
    start: (f32, f32),
    last: (f32, f32),
    moved: bool,
}

impl DragState {
    fn new(kind: DragKind, px: f32, py: f32) -> Self {
        Self {
            kind,
            start: (px, py),
            last: (px, py),
            moved: false,
        }
    }
}

struct RuntimeState {
    current_structure_key: u64,
    last_data_fingerprint: u64,
    buffer_dirty: bool,
    voxel_gpu_offsets: HashMap<NodeId, u32>,
    sculpt_tex_indices: HashMap<NodeId, u32>,
    modifiers: ModifiersState,
    cursor_pos: Option<(f32, f32)>,
    drag_state: Option<DragState>,
    last_fps_instant: Instant,
    frame_counter: u32,
    fps_smoothed: f32,
    last_pointer_hint: Option<ViewportPointer>,
    benchmark_duration: Duration,
    benchmark_start: Instant,
    benchmark_reported: bool,
    pacing: FramePacingTracker,
}

impl RuntimeState {
    fn new(scene: &Scene, benchmark_duration: Duration) -> Self {
        Self {
            current_structure_key: scene.structure_key(),
            last_data_fingerprint: scene.data_fingerprint(),
            buffer_dirty: true,
            voxel_gpu_offsets: HashMap::new(),
            sculpt_tex_indices: HashMap::new(),
            modifiers: ModifiersState::default(),
            cursor_pos: None,
            drag_state: None,
            last_fps_instant: Instant::now(),
            frame_counter: 0,
            fps_smoothed: 60.0,
            last_pointer_hint: None,
            benchmark_duration,
            benchmark_start: Instant::now(),
            benchmark_reported: false,
            pacing: FramePacingTracker::new(),
        }
    }

    fn tick_fps(&mut self) -> f32 {
        self.frame_counter += 1;
        let elapsed = self.last_fps_instant.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let fps = self.frame_counter as f32 / elapsed.as_secs_f32().max(0.001);
            self.fps_smoothed = self.fps_smoothed * 0.7 + fps * 0.3;
            self.frame_counter = 0;
            self.last_fps_instant = Instant::now();
        }
        self.fps_smoothed
    }
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    resources: ViewportResources,
}

impl GpuState {
    async fn new(window: Arc<Window>, core: &AppCore, settings: &Settings) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|err| format!("surface creation failed: {err}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "request_adapter failed".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("sdf-modeler-slint-host-device"),
                    required_features: wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|err| format!("request_device failed: {err}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|fmt| fmt.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = if settings.vsync_enabled {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        let present_mode = if caps.present_modes.contains(&present_mode) {
            present_mode
        } else {
            caps.present_modes[0]
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader_src = codegen::generate_shader(&core.scene, &settings.render);
        let pick_shader_src = codegen::generate_pick_shader(&core.scene, &settings.render);
        let resources = ViewportResources::new(&device, config.format, &shader_src, &pick_shader_src);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            resources,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn viewport_rect(&self) -> ViewportRect {
        compute_viewport_rect(self.config.width, self.config.height)
    }
}

struct SlintViewportApp {
    settings: Settings,
    core: AppCore,
    runtime: RuntimeState,
    gpu: Option<GpuState>,
}

impl SlintViewportApp {
    fn new(settings: &Settings) -> Self {
        let core = AppCore::from_init(AppCoreInit {
            scene: Scene::new(),
            history: History::new(),
            camera: Camera::default(),
            selection: CoreSelection::default(),
            active_tool: ActiveTool::Select,
            sculpt_state: SculptState::Inactive,
            async_state: CoreAsyncState::default(),
            soloed_light: None,
            show_debug: false,
            show_settings: false,
        });
        let benchmark_duration = Duration::from_secs(benchmark_seconds_from_args(0));
        Self {
            settings: settings.clone(),
            runtime: RuntimeState::new(&core.scene, benchmark_duration),
            core,
            gpu: None,
        }
    }

    fn apply_core_command(&mut self, command: CoreCommand) {
        let result = self.core.apply_command(command);
        if result.buffer_dirty || result.needs_graph_rebuild {
            self.runtime.buffer_dirty = true;
        }
    }

    fn sync_pipeline_and_buffers(&mut self, gpu: &mut GpuState) {
        let structure_key = self.core.scene.structure_key();
        if structure_key != self.runtime.current_structure_key {
            let shader_src = codegen::generate_shader(&self.core.scene, &self.settings.render);
            let pick_shader_src = codegen::generate_pick_shader(&self.core.scene, &self.settings.render);
            let sculpt_count = buffers::collect_sculpt_tex_info(&self.core.scene).len();
            gpu.resources
                .rebuild_pipeline(&gpu.device, &shader_src, &pick_shader_src, sculpt_count);
            self.runtime.current_structure_key = structure_key;
            self.runtime.buffer_dirty = true;
        }

        let data_fp = self.core.scene.data_fingerprint();
        if data_fp != self.runtime.last_data_fingerprint {
            self.runtime.last_data_fingerprint = data_fp;
            self.runtime.buffer_dirty = true;
        }

        if self.runtime.buffer_dirty {
            let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&self.core.scene);
            let node_data =
                buffers::build_node_buffer(&self.core.scene, &self.core.selection.set, &voxel_offsets);
            let sculpt_infos = buffers::collect_sculpt_tex_info(&self.core.scene);

            self.runtime.voxel_gpu_offsets = voxel_offsets;
            self.runtime.sculpt_tex_indices =
                sculpt_infos.iter().map(|info| (info.node_id, info.tex_idx as u32)).collect();

            gpu.resources
                .update_scene_buffer(&gpu.device, &gpu.queue, &node_data);
            gpu.resources
                .update_voxel_buffer(&gpu.device, &gpu.queue, &voxel_data);

            for info in &sculpt_infos {
                if let Some(node) = self.core.scene.nodes.get(&info.node_id) {
                    if let crate::graph::scene::NodeData::Sculpt { voxel_grid, .. } = &node.data {
                        gpu.resources.upload_voxel_texture(
                            &gpu.device,
                            &gpu.queue,
                            info.tex_idx,
                            voxel_grid.resolution,
                            &voxel_grid.data,
                        );
                    }
                }
            }
            self.runtime.buffer_dirty = false;
        }
    }

    fn build_render_uniform(&self, viewport: ViewportRect, render_scale: f32, time_s: f32) -> CameraUniform {
        let render_w = (viewport.width * render_scale).max(1.0);
        let render_h = (viewport.height * render_scale).max(1.0);
        let render_viewport = [0.0, 0.0, render_w, render_h];
        let scene_bounds = self.core.scene.compute_bounds();
        let selected_idx = self
            .core
            .selection
            .primary
            .and_then(|id| {
                let order = self.core.scene.visible_topo_order();
                order.iter().position(|&nid| nid == id)
            })
            .map(|idx| idx as f32)
            .unwrap_or(-1.0);
        let shading_mode = self.settings.render.shading_mode.gpu_value();
        let cross_section = [
            self.settings.render.cross_section_axis as f32,
            self.settings.render.cross_section_position,
            0.0,
            0.0,
        ];
        let (light_count, scene_lights, scene_ambient) = buffers::collect_scene_lights(
            &self.core.scene,
            self.core.camera.eye(),
            self.core.soloed_light,
            time_s,
        );
        let volumetric_count =
            scene_lights.iter().filter(|light| light.volumetric[0] > 0.5).count() as f32;
        let mut scene_lights_flat = [[0.0_f32; 4]; 32];
        let mut scene_light_vol = [[0.0_f32; 4]; 8];
        for (idx, light) in scene_lights.iter().enumerate() {
            scene_lights_flat[idx * 4] = light.position_type;
            scene_lights_flat[idx * 4 + 1] = light.direction_intensity;
            scene_lights_flat[idx * 4 + 2] = light.color_range;
            scene_lights_flat[idx * 4 + 3] = light.params;
            scene_light_vol[idx] = light.volumetric;
        }
        let ambient_luminance = scene_ambient
            .color
            .dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
        let effective_ambient = if ambient_luminance > 0.0 {
            ambient_luminance
        } else {
            self.settings.render.ambient
        };
        let scene_light_info = [
            light_count as f32,
            volumetric_count,
            self.settings.render.volumetric_steps as f32,
            0.0,
        ];
        self.core.camera.to_uniform(
            render_viewport,
            time_s,
            0.0,
            self.settings.render.show_grid,
            scene_bounds,
            selected_idx,
            shading_mode,
            [0.0; 4],
            cross_section,
            effective_ambient,
            scene_light_info,
            scene_lights_flat,
            scene_light_vol,
        )
    }

    fn picked_node_at(
        &self,
        gpu: &GpuState,
        viewport: ViewportRect,
        pointer: ViewportPointer,
    ) -> Option<NodeId> {
        let scene_bounds = self.core.scene.compute_bounds();
        let pick_uniform = self.core.camera.to_uniform(
            [0.0, 0.0, viewport.width.max(1.0), viewport.height.max(1.0)],
            0.0,
            0.0,
            false,
            scene_bounds,
            -1.0,
            0.0,
            [0.0; 4],
            [0.0; 4],
            0.0,
            [0.0; 4],
            [[0.0; 4]; 32],
            [[0.0; 4]; 8],
        );

        let pending = PendingPick {
            mouse_pos: [pointer.local_x, pointer.local_y],
            camera_uniform: pick_uniform,
            ctrl_held: self.runtime.modifiers.control_key(),
        };
        let topo_order = self.core.scene.visible_topo_order();
        if let Some(result) = gpu
            .resources
            .execute_pick(&gpu.device, &gpu.queue, &pending)
        {
            let idx = result.material_id as usize;
            if idx < topo_order.len() {
                return Some(topo_order[idx]);
            }
        }
        None
    }

    fn handle_hotkey(&mut self, key: KeyCode) -> bool {
        let ctrl = self.runtime.modifiers.control_key();
        match (ctrl, key) {
            (true, KeyCode::KeyN) => {
                self.apply_core_command(CoreCommand::NewScene);
                true
            }
            (true, KeyCode::KeyZ) => {
                self.apply_core_command(CoreCommand::Undo);
                true
            }
            (true, KeyCode::KeyY) => {
                self.apply_core_command(CoreCommand::Redo);
                true
            }
            (false, KeyCode::Digit1) => {
                self.apply_core_command(CoreCommand::SetActiveTool(ActiveTool::Select));
                true
            }
            (false, KeyCode::Digit2) => {
                self.apply_core_command(CoreCommand::SetActiveTool(ActiveTool::Sculpt));
                true
            }
            (false, KeyCode::KeyP) => {
                self.apply_core_command(CoreCommand::CreatePrimitive(SdfPrimitive::Sphere));
                true
            }
            (false, KeyCode::Delete) => {
                self.apply_core_command(CoreCommand::DeleteSelected);
                true
            }
            (false, KeyCode::KeyF) => {
                let (bmin, bmax) = self.core.scene.compute_bounds();
                let min = glam::Vec3::from_array(bmin);
                let max = glam::Vec3::from_array(bmax);
                let center = (min + max) * 0.5;
                let radius = (max - min).length().max(0.25) * 0.5;
                self.core.camera.focus_on(center, radius);
                true
            }
            _ => false,
        }
    }

    fn render_frame(&mut self, gpu: &mut GpuState) -> Result<(), wgpu::SurfaceError> {
        self.sync_pipeline_and_buffers(gpu);

        let viewport = gpu.viewport_rect();
        let interacting = self.runtime.drag_state.is_some();
        let render_scale = if interacting {
            self.settings
                .render
                .interaction_render_scale
                .clamp(0.25, 1.0)
        } else {
            self.settings.render.rest_render_scale.clamp(0.25, 2.0)
        };
        let uniform = self.build_render_uniform(
            viewport,
            render_scale,
            self.runtime.benchmark_start.elapsed().as_secs_f32(),
        );
        gpu.queue
            .write_buffer(&gpu.resources.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        gpu.resources.ensure_offscreen_texture(
            &gpu.device,
            (viewport.width * render_scale).max(1.0) as u32,
            (viewport.height * render_scale).max(1.0) as u32,
        );

        let bloom = if self.settings.render.bloom_enabled {
            [
                self.settings.render.bloom_threshold,
                self.settings.render.bloom_intensity,
                self.settings.render.bloom_radius,
                1.0,
            ]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        let blit_params: [f32; 12] = [
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            self.settings.render.outline_color[0],
            self.settings.render.outline_color[1],
            self.settings.render.outline_color[2],
            self.settings.render.outline_thickness,
            bloom[0],
            bloom[1],
            bloom[2],
            bloom[3],
        ];
        gpu.queue.write_buffer(
            &gpu.resources.blit_params_buffer,
            0,
            bytemuck::cast_slice(&blit_params),
        );

        let frame = gpu.surface.get_current_texture()?;
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slint-host-frame-encoder"),
            });

        {
            let offscreen_view = gpu
                .resources
                .offscreen_view
                .as_ref()
                .expect("offscreen view must exist");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slint-host-offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if gpu.resources.use_composite {
                if let Some(ref comp) = gpu.resources.composite {
                    pass.set_pipeline(&comp.render_pipeline);
                    pass.set_bind_group(0, &gpu.resources.camera_bind_group, &[]);
                    pass.set_bind_group(1, &gpu.resources.scene_bind_group, &[]);
                    pass.set_bind_group(2, &comp.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&gpu.resources.pipeline);
                pass.set_bind_group(0, &gpu.resources.camera_bind_group, &[]);
                pass.set_bind_group(1, &gpu.resources.scene_bind_group, &[]);
                pass.set_bind_group(2, &gpu.resources.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slint-host-blit-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.09,
                            g: 0.10,
                            b: 0.13,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(ref blit_bg) = gpu.resources.blit_bind_group {
                pass.set_scissor_rect(
                    viewport.x as u32,
                    viewport.y as u32,
                    viewport.width.max(1.0) as u32,
                    viewport.height.max(1.0) as u32,
                );
                pass.set_pipeline(&gpu.resources.blit_pipeline);
                pass.set_bind_group(0, blit_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

impl ApplicationHandler for SlintViewportApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("SDF Modeler (Slint host)")
            .with_inner_size(PhysicalSize::new(1360, 860));

        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                eprintln!("window creation failed: {err}");
                event_loop.exit();
                return;
            }
        };

        match block_on(GpuState::new(window.clone(), &self.core, &self.settings)) {
            Ok(mut gpu) => {
                self.sync_pipeline_and_buffers(&mut gpu);
                self.gpu = Some(gpu);
                self.runtime.benchmark_start = Instant::now();
                self.runtime.benchmark_reported = false;
                self.runtime.pacing.reset();
                window.request_redraw();
            }
            Err(err) => {
                eprintln!("GPU init failed: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    let size = gpu.window.inner_size();
                    gpu.resize(size);
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.runtime.modifiers = mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        if code == KeyCode::Escape {
                            event_loop.exit();
                            return;
                        }
                        if self.handle_hotkey(code) {
                            if let Some(gpu) = self.gpu.as_ref() {
                                gpu.window.request_redraw();
                            }
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;
                self.runtime.cursor_pos = Some((px, py));
                let Some(viewport) = self.gpu.as_ref().map(|gpu| gpu.viewport_rect()) else {
                    self.runtime.last_pointer_hint = None;
                    return;
                };
                self.runtime.last_pointer_hint = map_pointer_to_viewport(px, py, viewport);

                if let Some(mut drag) = self.runtime.drag_state {
                    let dx = px - drag.last.0;
                    let dy = py - drag.last.1;
                    if (px - drag.start.0).abs() > CLICK_DRAG_THRESHOLD_PX
                        || (py - drag.start.1).abs() > CLICK_DRAG_THRESHOLD_PX
                    {
                        drag.moved = true;
                    }
                    match drag.kind {
                        DragKind::Orbit => {
                            self.core.camera.orbit(dx, dy);
                            if self.settings.render.clamp_orbit_pitch {
                                self.core.camera.clamp_pitch();
                            }
                        }
                        DragKind::Pan => self.core.camera.pan(dx, dy),
                    }
                    drag.last = (px, py);
                    self.runtime.drag_state = Some(drag);
                    if let Some(gpu) = self.gpu.as_ref() {
                        gpu.window.request_redraw();
                    }
                } else {
                    let fps = self.runtime.tick_fps();
                    if let Some(gpu) = self.gpu.as_ref() {
                        gpu.window
                            .set_title(&frame_stats_title(fps, &self.core, self.runtime.last_pointer_hint));
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some((px, py)) = self.runtime.cursor_pos else {
                    return;
                };
                let Some(viewport) = self.gpu.as_ref().map(|gpu| gpu.viewport_rect()) else {
                    return;
                };
                match (state, button) {
                    (winit::event::ElementState::Pressed, MouseButton::Left) => {
                        if viewport.contains(px, py) {
                            self.runtime.drag_state = Some(DragState::new(DragKind::Orbit, px, py));
                        }
                    }
                    (winit::event::ElementState::Released, MouseButton::Left) => {
                        let drag = self.runtime.drag_state.take();
                        if let Some(drag) = drag {
                            if drag.kind == DragKind::Orbit && !drag.moved {
                                if let Some(pointer) = map_pointer_to_viewport(px, py, viewport) {
                                    let picked = self
                                        .gpu
                                        .as_ref()
                                        .and_then(|gpu| self.picked_node_at(gpu, viewport, pointer));
                                    if self.runtime.modifiers.control_key() {
                                        if let Some(id) = picked {
                                            self.apply_core_command(CoreCommand::ToggleSelect(id));
                                        }
                                    } else {
                                        self.apply_core_command(CoreCommand::Select(picked));
                                    }
                                }
                            }
                        }
                        if let Some(gpu) = self.gpu.as_ref() {
                            gpu.window.request_redraw();
                        }
                    }
                    (winit::event::ElementState::Pressed, MouseButton::Right) => {
                        if viewport.contains(px, py) {
                            self.runtime.drag_state = Some(DragState::new(DragKind::Pan, px, py));
                        }
                    }
                    (winit::event::ElementState::Released, MouseButton::Right) => {
                        self.runtime.drag_state = None;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let Some((px, py)) = self.runtime.cursor_pos else {
                    return;
                };
                let Some(viewport) = self.gpu.as_ref().map(|gpu| gpu.viewport_rect()) else {
                    return;
                };
                if !viewport.contains(px, py) {
                    return;
                }
                let zoom_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                self.core.camera.zoom(zoom_delta);
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(mut gpu) = self.gpu.take() else {
                    return;
                };
                let _ = self.core.camera.tick_transition(1.0 / 60.0);
                let frame_work_start = Instant::now();
                let render_result = self.render_frame(&mut gpu);
                self.runtime.pacing.record_frame(frame_work_start.elapsed());
                match render_result {
                    Ok(()) => {
                        let fps = self.runtime.tick_fps();
                        gpu.window
                            .set_title(&frame_stats_title(fps, &self.core, self.runtime.last_pointer_hint));
                    }
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = gpu.window.inner_size();
                        gpu.resize(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        eprintln!("Out of GPU memory");
                        self.gpu = Some(gpu);
                        event_loop.exit();
                        return;
                    }
                    Err(wgpu::SurfaceError::Timeout) => {}
                }

                if self.runtime.benchmark_duration > Duration::ZERO
                    && self.runtime.benchmark_start.elapsed() >= self.runtime.benchmark_duration
                {
                    if !self.runtime.benchmark_reported {
                        eprintln!(
                            "{}",
                            self.runtime
                                .pacing
                                .build_report("main_slint_winit_host")
                        );
                        self.runtime.benchmark_reported = true;
                    }
                    self.gpu = Some(gpu);
                    event_loop.exit();
                } else {
                    gpu.window.request_redraw();
                    self.gpu = Some(gpu);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

pub fn run(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = SlintViewportApp::new(settings);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_viewport_rect, map_pointer_to_viewport, ViewportRect};

    #[test]
    fn viewport_rect_has_expected_panel_offsets() {
        let rect = compute_viewport_rect(1360, 860);
        assert!(rect.x >= 0.0);
        assert!(rect.y > 0.0);
        assert!(rect.width > 900.0);
        assert!(rect.height > 700.0);
    }

    #[test]
    fn pointer_mapping_normalizes_inside_viewport() {
        let viewport = ViewportRect {
            x: 10.0,
            y: 20.0,
            width: 200.0,
            height: 100.0,
        };
        let mapped = map_pointer_to_viewport(110.0, 70.0, viewport).expect("inside viewport");
        assert!((mapped.u - 0.5).abs() < 0.0001);
        assert!((mapped.v - 0.5).abs() < 0.0001);
        assert!((mapped.local_x - 100.0).abs() < 0.0001);
        assert!((mapped.local_y - 50.0).abs() < 0.0001);
    }

    #[test]
    fn pointer_mapping_rejects_outside() {
        let viewport = ViewportRect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        assert!(map_pointer_to_viewport(130.0, 50.0, viewport).is_none());
        assert!(map_pointer_to_viewport(50.0, 130.0, viewport).is_none());
    }
}










