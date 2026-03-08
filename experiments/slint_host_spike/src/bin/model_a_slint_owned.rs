use std::cell::RefCell;
use std::error::Error;
use std::rc::Rc;
use std::time::{Duration, Instant};

use slint::wgpu_28::wgpu;
use slint::{GraphicsAPI, RenderingState};
use slint_host_spike::{benchmark_seconds_from_args, map_pointer_to_viewport, FramePacingTracker, ViewportRect};

slint::slint! {
    export component SpikeWindow inherits Window {
        in-out property <image> viewport-image;
        in-out property <string> pointer-text: "pointer=outside";
        callback viewport-pointer(px: float, py: float, viewport-w: float, viewport-h: float);

        preferred-width: 1280px;
        preferred-height: 760px;
        title: "Slint Host Spike (Model A: Slint-owned)";

        VerticalLayout {
            spacing: 8px;
            padding: 8px;

            Rectangle {
                border-color: #3a3a3a;
                border-width: 1px;
                background: #0f1116;
                min-height: 620px;

                Image {
                    source: root.viewport-image;
                    width: parent.width;
                    height: parent.height;
                    image-fit: fill;
                }

                TouchArea {
                    moved => {
                        root.viewport-pointer(self.mouse-x / 1px, self.mouse-y / 1px, parent.width / 1px, parent.height / 1px);
                    }
                    clicked => {
                        root.viewport-pointer(self.mouse-x / 1px, self.mouse-y / 1px, parent.width / 1px, parent.height / 1px);
                    }
                }
            }

            Text {
                text: root.pointer-text;
                color: #d5d9e0;
                font-size: 14px;
            }
        }
    }
}

struct OffscreenTriangleRenderer {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
}

impl OffscreenTriangleRenderer {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("slint-spike-viewport"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slint-spike-triangle-shader"),
            source: wgpu::ShaderSource::Wgsl(TRIANGLE_WGSL.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slint-spike-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slint-spike-triangle-pipeline"),
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

        Self { texture, texture_view, pipeline }
    }

    fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, elapsed_s: f32) {
        let blue = (0.20 + 0.15 * (elapsed_s * 0.8).sin()).clamp(0.0, 1.0) as f64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slint-spike-encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slint-spike-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.07,
                            g: 0.08,
                            b: blue,
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
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    fn image(&self) -> slint::Image {
        slint::Image::try_from(self.texture.clone()).expect("wgpu texture import into Slint image must succeed")
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let benchmark_seconds = benchmark_seconds_from_args(12);
    let benchmark_duration = Duration::from_secs(benchmark_seconds);

    slint::BackendSelector::new()
        .backend_name("winit".into())
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::default())
        .select()?;

    let app = SpikeWindow::new()?;
    let tracker = Rc::new(RefCell::new(FramePacingTracker::new()));
    let renderer = Rc::new(RefCell::new(None::<OffscreenTriangleRenderer>));
    let start = Instant::now();
    let benchmark_printed = Rc::new(RefCell::new(false));

    let redraw_timer = slint::Timer::default();
    {
        let redraw_app = app.as_weak();
        redraw_timer.start(slint::TimerMode::Repeated, Duration::from_millis(16), move || {
            if let Some(app_instance) = redraw_app.upgrade() {
                app_instance.window().request_redraw();
            }
        });
    }

    {
        let pointer_app = app.as_weak();
        app.on_viewport_pointer(move |px, py, viewport_w, viewport_h| {
            let viewport = ViewportRect { x: 0.0, y: 0.0, width: viewport_w, height: viewport_h };
            let label = match map_pointer_to_viewport(px, py, viewport) {
                Some(mapped) => format!(
                    "pointer raw=({:.1}, {:.1}) viewport=({:.0}x{:.0}) normalized=({:.3}, {:.3})",
                    px, py, viewport_w, viewport_h, mapped.u, mapped.v
                ),
                None => format!(
                    "pointer raw=({:.1}, {:.1}) viewport=({:.0}x{:.0}) normalized=outside",
                    px, py, viewport_w, viewport_h
                ),
            };

            if let Some(app_instance) = pointer_app.upgrade() {
                app_instance.set_pointer_text(label.into());
            }
        });
    }

    {
        let app_weak = app.as_weak();
        let tracker = tracker.clone();
        let renderer = renderer.clone();
        let benchmark_printed = benchmark_printed.clone();

        app.window().set_rendering_notifier(move |state, graphics_api| {
            let (Some(app_instance), GraphicsAPI::WGPU28 { device, queue, .. }) =
                (app_weak.upgrade(), graphics_api)
            else {
                return;
            };

            match state {
                RenderingState::RenderingSetup => {
                    let offscreen_renderer = OffscreenTriangleRenderer::new(device, 1200, 620);
                    offscreen_renderer.render(device, queue, 0.0);
                    app_instance.set_viewport_image(offscreen_renderer.image());
                    *renderer.borrow_mut() = Some(offscreen_renderer);
                }
                RenderingState::BeforeRendering => {
                    if let Some(current_renderer) = renderer.borrow().as_ref() {
                        let frame_start = Instant::now();
                        current_renderer.render(device, queue, start.elapsed().as_secs_f32());
                        tracker.borrow_mut().record_frame(frame_start.elapsed());
                    }

                    if start.elapsed() >= benchmark_duration {
                        if !*benchmark_printed.borrow() {
                            *benchmark_printed.borrow_mut() = true;
                            println!("{}", tracker.borrow().build_report("model_a_slint_owned"));
                        }
                        let _ = slint::quit_event_loop();
                    } else {
                        app_instance.window().request_redraw();
                    }
                }
                RenderingState::RenderingTeardown => {
                    renderer.borrow_mut().take();
                }
                RenderingState::AfterRendering => {}
                _ => {}
            }
        })?;
    }

    app.run()?;
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
    return vec4<f32>(0.89, 0.55, 0.25, 1.0);
}
"#;