use slint::wgpu_28::wgpu;

use super::camera::CameraUniforms;
use super::scene::{GizmoAxis, PickResult, Scene, SceneInfoGpu, SdfNodeGpu};

/// Initial storage buffer size (enough for ~50 nodes).
const INITIAL_SCENE_BUFFER_SIZE: u64 = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PickInfoGpu {
    click_ndc: [f32; 2],
    _pad: [f32; 2],
}

pub struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    // Camera (group 0)
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    // Scene (group 1) — storage buffer for nodes, uniform for info
    scene_buffer: wgpu::Buffer,
    scene_buffer_capacity: u64,
    scene_info_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    // Pick pass (group 2)
    pick_pipeline: wgpu::RenderPipeline,
    pick_texture: wgpu::Texture,
    pick_staging: wgpu::Buffer,
    pick_info_buffer: wgpu::Buffer,
    pick_bind_group: wgpu::BindGroup,
    pick_bind_group_layout: wgpu::BindGroupLayout,
    // Offscreen texture
    render_texture: Option<wgpu::Texture>,
    tex_width: u32,
    tex_height: u32,
}

impl GpuState {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, scene: &Scene) -> Self {
        let shader_src = super::codegen::compose_shader(scene);
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Group 0: Camera ─────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform_buffer"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── Group 1: Scene (storage buffer for nodes) ────────────
        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_storage_buffer"),
            size: INITIAL_SCENE_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_info_uniform_buffer"),
            size: std::mem::size_of::<SceneInfoGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_bind_group_layout"),
                entries: &[
                    // @binding(0): storage buffer (nodes array)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1): uniform buffer (scene info)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene_info_buffer.as_entire_binding(),
                },
            ],
        });

        // ── Group 2: Pick Info ──────────────────────────────────
        let pick_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_info_buffer"),
            size: std::mem::size_of::<PickInfoGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pick_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pick_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_bind_group"),
            layout: &pick_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_info_buffer.as_entire_binding(),
            }],
        });

        // ── Main Pipeline (groups 0, 1) ────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_pipeline_layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &scene_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sdf_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Pick Pipeline (groups 0, 1, 2) ─────────────────────
        let pick_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pick_pipeline_layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &scene_bind_group_layout,
                    &pick_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick_pipeline"),
            layout: Some(&pick_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("pick_fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Pick Texture + Staging ─────────────────────────────
        let pick_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 256 bytes minimum due to COPY_BYTES_PER_ROW_ALIGNMENT
        let pick_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_staging"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            scene_buffer,
            scene_buffer_capacity: INITIAL_SCENE_BUFFER_SIZE,
            scene_info_buffer,
            scene_bind_group,
            scene_bind_group_layout,
            pick_pipeline,
            pick_texture,
            pick_staging,
            pick_info_buffer,
            pick_bind_group,
            pick_bind_group_layout,
            render_texture: None,
            tex_width: 0,
            tex_height: 0,
        }
    }

    /// Rebuild both main and pick pipelines with a new shader source.
    /// Called on the slow path when graph topology changes.
    pub fn rebuild_pipelines(&mut self, shader_src: &str) {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Main pipeline (groups 0, 1)
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("sdf_pipeline_layout"),
                    bind_group_layouts: &[
                        &self.camera_bind_group_layout,
                        &self.scene_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

        self.pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sdf_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Pick pipeline (groups 0, 1, 2)
        let pick_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pick_pipeline_layout"),
                    bind_group_layouts: &[
                        &self.camera_bind_group_layout,
                        &self.scene_bind_group_layout,
                        &self.pick_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

        self.pick_pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("pick_pipeline"),
                    layout: Some(&pick_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader_module,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader_module,
                        entry_point: Some("pick_fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

        log::debug!("Pipelines rebuilt with new shader");
    }

    /// Write camera uniform data to the GPU buffer.
    pub fn update_camera(&self, uniforms: &CameraUniforms) {
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Write scene node data and info to the GPU buffers.
    /// Dynamically resizes the storage buffer if needed.
    pub fn update_scene(&mut self, nodes: &[SdfNodeGpu], info: &SceneInfoGpu) {
        if !nodes.is_empty() {
            let required_size = (nodes.len() * std::mem::size_of::<SdfNodeGpu>()) as u64;

            // Grow buffer if needed
            if required_size > self.scene_buffer_capacity {
                let new_capacity = required_size.next_power_of_two();
                self.scene_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("scene_storage_buffer"),
                    size: new_capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.scene_buffer_capacity = new_capacity;

                // Recreate bind group with new buffer
                self.scene_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scene_bind_group"),
                        layout: &self.scene_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.scene_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.scene_info_buffer.as_entire_binding(),
                            },
                        ],
                    });
            }

            self.queue
                .write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(nodes));
        }
        self.queue
            .write_buffer(&self.scene_info_buffer, 0, bytemuck::bytes_of(info));
    }

    /// GPU pick: render a single pixel at the click position and read back the node ID.
    pub fn pick_at(&self, ndc_x: f32, ndc_y: f32) -> PickResult {
        // Upload click NDC
        let info = PickInfoGpu {
            click_ndc: [ndc_x, ndc_y],
            _pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.pick_info_buffer, 0, bytemuck::bytes_of(&info));

        // Render pick pass to 1x1 texture
        let view = self
            .pick_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pick_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_bind_group(2, &self.pick_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy texture to staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pick_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back the pixel
        let buffer_slice = self.pick_staging.slice(..4);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        if receiver.recv().ok().and_then(|r| r.ok()).is_none() {
            return PickResult::Background;
        }

        let data = buffer_slice.get_mapped_range();
        let id = data[0];
        drop(data);
        self.pick_staging.unmap();

        match id {
            0 => PickResult::Background,
            1 => PickResult::Floor,
            253 => PickResult::GizmoAxis(GizmoAxis::X),
            254 => PickResult::GizmoAxis(GizmoAxis::Y),
            255 => PickResult::GizmoAxis(GizmoAxis::Z),
            n => PickResult::Node((n - 2) as usize),
        }
    }

    /// Ensure the offscreen render texture matches the requested size.
    fn ensure_texture(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);

        if self.tex_width == w && self.tex_height == h && self.render_texture.is_some() {
            return;
        }

        self.render_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        self.tex_width = w;
        self.tex_height = h;
    }

    /// Render one frame: run SDF raymarching, return as Slint Image.
    pub fn render_frame(&mut self, width: u32, height: u32) -> Option<slint::Image> {
        self.ensure_texture(width, height);

        let texture = self.render_texture.as_ref()?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sdf_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sdf_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
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
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        slint::Image::try_from(texture.clone()).ok()
    }
}
