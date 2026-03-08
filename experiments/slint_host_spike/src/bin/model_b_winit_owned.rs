use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pollster::block_on;
use slint::wgpu_28::wgpu;
use slint::winit_030::winit;
use slint_host_spike::{benchmark_seconds_from_args, map_pointer_to_viewport, FramePacingTracker, ViewportRect};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
}

impl GpuState {
    async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

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
            .map_err(|err| format!("request_adapter failed: {err}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("slint-spike-model-b-device"),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|err| format!("request_device failed: {err}"))?;

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|fmt| fmt.is_srgb())
            .unwrap_or(capabilities.formats[0]);
        let present_mode = capabilities
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::AutoVsync)
            .unwrap_or(capabilities.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slint-spike-model-b-shader"),
            source: wgpu::ShaderSource::Wgsl(TRIANGLE_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slint-spike-model-b-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slint-spike-model-b-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self { window, surface, device, queue, config, pipeline })
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
        let panel_width = 320.0;
        let width = (self.config.width as f32 - panel_width).max(1.0);
        ViewportRect { x: 0.0, y: 0.0, width, height: self.config.height as f32 }
    }

    fn render(&mut self, elapsed_s: f32) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slint-spike-model-b-encoder"),
        });

        let viewport_rect = self.viewport_rect();
        let blue = (0.18 + 0.16 * (elapsed_s * 0.75).sin()).clamp(0.0, 1.0) as f64;

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slint-spike-model-b-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.09,
                            b: 0.11,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_viewport(viewport_rect.x, viewport_rect.y, viewport_rect.width, viewport_rect.height, 0.0, 1.0);
            pass.set_scissor_rect(
                viewport_rect.x as u32,
                viewport_rect.y as u32,
                viewport_rect.width as u32,
                viewport_rect.height as u32,
            );
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);

            // Draw a fake "UI panel" region by clearing only the panel slice via scissor + second pass.
            pass.set_viewport(
                viewport_rect.width,
                0.0,
                (self.config.width as f32 - viewport_rect.width).max(1.0),
                self.config.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                viewport_rect.width as u32,
                0,
                (self.config.width as f32 - viewport_rect.width).max(1.0) as u32,
                self.config.height,
            );
            // Re-use same pipeline draw to keep spike simple; color variation provides visual split.
            let _ = blue;
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

struct ModelBApp {
    gpu: Option<GpuState>,
    benchmark_duration: Duration,
    benchmark_start: Instant,
    tracker: FramePacingTracker,
    printed_report: bool,
}

impl ModelBApp {
    fn new(benchmark_duration: Duration) -> Self {
        Self {
            gpu: None,
            benchmark_duration,
            benchmark_start: Instant::now(),
            tracker: FramePacingTracker::new(),
            printed_report: false,
        }
    }

    fn print_report_once(&mut self) {
        if self.printed_report {
            return;
        }
        self.printed_report = true;
        println!("{}", self.tracker.build_report("model_b_winit_owned"));
    }
}

impl ApplicationHandler for ModelBApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Slint Host Spike (Model B: Winit-owned baseline)")
            .with_inner_size(PhysicalSize::new(1280, 760));

        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                eprintln!("window creation failed: {err}");
                event_loop.exit();
                return;
            }
        };

        match block_on(GpuState::new(window.clone())) {
            Ok(gpu_state) => {
                self.gpu = Some(gpu_state);
                self.benchmark_start = Instant::now();
                self.tracker = FramePacingTracker::new();
                window.request_redraw();
            }
            Err(err) => {
                eprintln!("GPU initialization failed: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                self.print_report_once();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let mapped = map_pointer_to_viewport(position.x as f32, position.y as f32, gpu.viewport_rect());
                let title = match mapped {
                    Some(value) => format!(
                        "Slint Host Spike (Model B) pointer=({:.3}, {:.3})",
                        value.u, value.v
                    ),
                    None => "Slint Host Spike (Model B) pointer=outside viewport".to_string(),
                };
                gpu.window.set_title(&title);
            }
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                match gpu.render(self.benchmark_start.elapsed().as_secs_f32()) {
                    Ok(()) => {
                        self.tracker.record_frame(frame_start.elapsed());
                    }
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let current_size = gpu.window.inner_size();
                        gpu.resize(current_size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        eprintln!("Out of GPU memory during spike render");
                        self.print_report_once();
                        event_loop.exit();
                        return;
                    }
                    Err(wgpu::SurfaceError::Timeout) => {}
                    Err(wgpu::SurfaceError::Other) => {}
                }

                if self.benchmark_start.elapsed() >= self.benchmark_duration {
                    self.print_report_once();
                    event_loop.exit();
                } else {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let benchmark_seconds = benchmark_seconds_from_args(12);
    let event_loop = EventLoop::new()?;
    let mut app = ModelBApp::new(Duration::from_secs(benchmark_seconds));
    event_loop.run_app(&mut app)?;
    Ok(())
}

const TRIANGLE_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.75, -0.60),
        vec2<f32>( 0.75, -0.60),
        vec2<f32>( 0.00,  0.70),
    );
    return vec4<f32>(positions[vid], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.80, 0.68, 0.22, 1.0);
}
"#;