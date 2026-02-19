use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen::SdfNodeGpu;

pub struct ViewportResources {
    pub pipeline: wgpu::RenderPipeline,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bgl: wgpu::BindGroupLayout,
    pub scene_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bgl: wgpu::BindGroupLayout,
    pub scene_buffer_capacity: usize,
    pub target_format: wgpu::TextureFormat,
}

impl ViewportResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        shader_src: &str,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Scene storage buffer
        let initial_capacity = 16usize;
        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Storage"),
            size: (initial_capacity * std::mem::size_of::<SdfNodeGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene BG"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl, &scene_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            camera_bgl,
            scene_buffer,
            scene_bind_group,
            scene_bgl,
            scene_buffer_capacity: initial_capacity,
            target_format,
        }
    }

    pub fn rebuild_pipeline(&mut self, device: &wgpu::Device, shader_src: &str) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader (rebuilt)"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[&self.camera_bgl, &self.scene_bgl],
            push_constant_ranges: &[],
        });

        self.pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
    }

    pub fn update_scene_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        node_data: &[SdfNodeGpu],
    ) {
        let needed = node_data.len().max(1);
        if needed > self.scene_buffer_capacity {
            self.scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Scene Storage"),
                size: (needed * std::mem::size_of::<SdfNodeGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Scene BG"),
                layout: &self.scene_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scene_buffer.as_entire_binding(),
                }],
            });
            self.scene_buffer_capacity = needed;
        }
        if !node_data.is_empty() {
            queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(node_data));
        }
    }
}

// ---------------------------------------------------------------------------
// Paint callback
// ---------------------------------------------------------------------------

struct ViewportCallback {
    uniform: CameraUniform,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&self.uniform),
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &resources.scene_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

pub fn draw(ui: &mut egui::Ui, camera: &mut Camera, time: f32) {
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

    if response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        camera.orbit(delta.x, delta.y);
    }
    if response.dragged_by(egui::PointerButton::Secondary) {
        let delta = response.drag_delta();
        camera.pan(delta.x, delta.y);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }

    let ppp = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * ppp,
        rect.min.y * ppp,
        rect.width() * ppp,
        rect.height() * ppp,
    ];
    let uniform = camera.to_uniform(viewport, time);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback { uniform },
    ));
}
