use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen::SdfNodeGpu;
use crate::gpu::picking::{PendingPick, PickResult};
use crate::graph::scene::NodeId;

const INITIAL_SCENE_CAPACITY: usize = 16;
const INITIAL_VOXEL_CAPACITY: usize = 4; // in f32 elements (minimum valid buffer)

/// GPU-side brush parameters. Matches the WGSL `BrushParams` struct layout (80 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BrushGpuParams {
    // vec3f center_local + f32 radius
    pub center_local: [f32; 3],
    pub radius: f32,
    // f32 strength, f32 sign_val, u32 grid_offset, u32 grid_resolution
    pub strength: f32,
    pub sign_val: f32,
    pub grid_offset: u32,
    pub grid_resolution: u32,
    // vec3f bounds_min + pad
    pub bounds_min: [f32; 3],
    pub _pad0: f32,
    // vec3f bounds_max + pad
    pub bounds_max: [f32; 3],
    pub _pad1: f32,
    // uvec3 min_voxel + pad
    pub min_voxel: [u32; 3],
    pub _pad2: u32,
}

/// CPU-side brush dispatch info (wraps GPU params + workgroup counts).
pub struct BrushDispatch {
    pub params: BrushGpuParams,
    pub workgroups: [u32; 3],
}

pub struct ViewportResources {
    // --- Render pipeline ---
    pub pipeline: wgpu::RenderPipeline,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bgl: wgpu::BindGroupLayout,
    pub scene_buffer: wgpu::Buffer,
    pub voxel_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bgl: wgpu::BindGroupLayout,
    pub scene_buffer_capacity: usize,
    pub voxel_buffer_capacity: usize, // in f32 elements
    pub target_format: wgpu::TextureFormat,

    // --- Voxel texture3D resources (render shader only) ---
    pub voxel_textures: Vec<wgpu::Texture>,
    pub voxel_texture_views: Vec<wgpu::TextureView>,
    pub voxel_sampler: wgpu::Sampler,
    pub voxel_tex_bgl: wgpu::BindGroupLayout,
    pub voxel_tex_bind_group: wgpu::BindGroup,

    // --- Pick compute pipeline ---
    pub pick_pipeline: wgpu::ComputePipeline,
    pub pick_input_buffer: wgpu::Buffer,
    pub pick_output_buffer: wgpu::Buffer,
    pub pick_staging_buffer: wgpu::Buffer,
    pub pick_bind_group: wgpu::BindGroup,
    pub pick_bgl: wgpu::BindGroupLayout,

    // --- Brush compute pipeline ---
    pub brush_pipeline: wgpu::ComputePipeline,
    pub brush_uniform_buffer: wgpu::Buffer,
    pub brush_bind_group: wgpu::BindGroup,
    pub brush_bgl: wgpu::BindGroupLayout,
}

impl ViewportResources {
    fn create_render_pipeline(
        device: &wgpu::Device,
        shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        voxel_tex_bgl: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, voxel_tex_bgl],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&layout),
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
        })
    }

    fn create_pick_pipeline(
        device: &wgpu::Device,
        pick_shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        pick_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pick Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(pick_shader_src.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pick Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, pick_bgl],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pick Compute Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_pick",
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_scene_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_brush_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brush BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_brush_pipeline(
        device: &wgpu::Device,
        brush_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brush Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::codegen::BRUSH_COMPUTE_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Brush Pipeline Layout"),
            bind_group_layouts: &[brush_bgl],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Brush Compute Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_brush",
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Create the bind group layout for voxel textures: binding 0 = sampler, then N texture_3d bindings.
    fn create_voxel_tex_bgl(device: &wgpu::Device, tex_count: usize) -> wgpu::BindGroupLayout {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }];
        for i in 0..tex_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            });
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel Tex BGL"),
            entries: &entries,
        })
    }

    /// Create the bind group for voxel textures.
    fn create_voxel_tex_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        views: &[wgpu::TextureView],
    ) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(sampler),
        }];
        for (i, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Tex BG"),
            layout: bgl,
            entries: &entries,
        })
    }

    fn rebuild_brush_bind_group(&mut self, device: &wgpu::Device) {
        self.brush_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brush BG"),
            layout: &self.brush_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.brush_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
            ],
        });
    }

    fn rebuild_scene_bind_group(&mut self, device: &wgpu::Device) {
        self.scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene BG"),
            layout: &self.scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
            ],
        });
        // Brush bind group also references voxel_buffer
        self.rebuild_brush_bind_group(device);
    }

    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        shader_src: &str,
        pick_shader_src: &str,
    ) -> Self {
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
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

        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Storage"),
            size: (INITIAL_SCENE_CAPACITY * std::mem::size_of::<SdfNodeGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let voxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Storage"),
            size: (INITIAL_VOXEL_CAPACITY * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_bgl = Self::create_scene_bgl(device);

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene BG"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Voxel texture3D resources (render shader only) ---
        let voxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Voxel Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let voxel_tex_bgl = Self::create_voxel_tex_bgl(device, 0);
        let voxel_tex_bind_group = Self::create_voxel_tex_bind_group(
            device, &voxel_tex_bgl, &voxel_sampler, &[],
        );

        let pipeline = Self::create_render_pipeline(
            device, shader_src, &camera_bgl, &scene_bgl, &voxel_tex_bgl, target_format,
        );

        // --- Pick compute resources ---
        let pick_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pick Input"),
            size: 16, // vec2f mouse_pos + vec2f pad
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pick_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pick Output"),
            size: 32, // 8 x f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pick_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pick Staging"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pick BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pick BG"),
            layout: &pick_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pick_output_buffer.as_entire_binding(),
                },
            ],
        });

        let pick_pipeline = Self::create_pick_pipeline(
            device, pick_shader_src, &camera_bgl, &scene_bgl, &pick_bgl,
        );

        // --- Brush compute resources ---
        let brush_bgl = Self::create_brush_bgl(device);
        let brush_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brush Uniform"),
            size: std::mem::size_of::<BrushGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let brush_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brush BG"),
            layout: &brush_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brush_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_buffer.as_entire_binding(),
                },
            ],
        });
        let brush_pipeline = Self::create_brush_pipeline(device, &brush_bgl);

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            camera_bgl,
            scene_buffer,
            voxel_buffer,
            scene_bind_group,
            scene_bgl,
            scene_buffer_capacity: INITIAL_SCENE_CAPACITY,
            voxel_buffer_capacity: INITIAL_VOXEL_CAPACITY,
            target_format,
            voxel_textures: Vec::new(),
            voxel_texture_views: Vec::new(),
            voxel_sampler,
            voxel_tex_bgl,
            voxel_tex_bind_group,
            pick_pipeline,
            pick_input_buffer,
            pick_output_buffer,
            pick_staging_buffer,
            pick_bind_group,
            pick_bgl,
            brush_pipeline,
            brush_uniform_buffer,
            brush_bind_group,
            brush_bgl,
        }
    }

    pub fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader_src: &str,
        pick_shader_src: &str,
        sculpt_count: usize,
    ) {
        // Rebuild voxel texture BGL if sculpt count changed
        if sculpt_count != self.voxel_textures.len() {
            self.rebuild_voxel_textures(device, sculpt_count);
        }
        self.pipeline = Self::create_render_pipeline(
            device, shader_src, &self.camera_bgl, &self.scene_bgl,
            &self.voxel_tex_bgl, self.target_format,
        );
        self.pick_pipeline = Self::create_pick_pipeline(
            device, pick_shader_src, &self.camera_bgl, &self.scene_bgl, &self.pick_bgl,
        );
    }

    /// Rebuild voxel texture3D resources for the given number of sculpt nodes.
    /// Creates 1x1x1 placeholder textures — actual data uploaded later.
    fn rebuild_voxel_textures(&mut self, device: &wgpu::Device, count: usize) {
        let mut textures = Vec::with_capacity(count);
        let mut views = Vec::with_capacity(count);
        for i in 0..count {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Voxel Tex {i}")),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            textures.push(tex);
        }
        self.voxel_textures = textures;
        self.voxel_texture_views = views;
        self.voxel_tex_bgl = Self::create_voxel_tex_bgl(device, count);
        self.voxel_tex_bind_group = Self::create_voxel_tex_bind_group(
            device, &self.voxel_tex_bgl, &self.voxel_sampler, &self.voxel_texture_views,
        );
    }

    /// Upload full voxel data to a specific texture (recreates if resolution changed).
    pub fn upload_voxel_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex_idx: usize,
        resolution: u32,
        data: &[f32],
    ) {
        if tex_idx >= self.voxel_textures.len() {
            return;
        }
        let current_size = self.voxel_textures[tex_idx].size();
        if current_size.width != resolution {
            // Recreate texture at correct resolution
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Voxel Tex {tex_idx}")),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: resolution,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.voxel_texture_views[tex_idx] = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.voxel_textures[tex_idx] = tex;
            // Rebuild bind group with updated view
            self.voxel_tex_bind_group = Self::create_voxel_tex_bind_group(
                device, &self.voxel_tex_bgl, &self.voxel_sampler, &self.voxel_texture_views,
            );
        }
        // Upload data
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.voxel_textures[tex_idx],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(resolution * 4), // R32Float = 4 bytes per texel
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
        );
    }

    /// Upload only the modified z-slab range of a voxel texture.
    pub fn upload_voxel_texture_region(
        &self,
        queue: &wgpu::Queue,
        tex_idx: usize,
        resolution: u32,
        z0: u32,
        z1: u32,
        data: &[f32],
    ) {
        if tex_idx >= self.voxel_textures.len() {
            return;
        }
        let slab_size = (resolution * resolution) as usize;
        let start_index = z0 as usize * slab_size;
        let end_index = ((z1 as usize) + 1) * slab_size;
        let sub_data = &data[start_index..end_index];
        let depth = z1 - z0 + 1;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.voxel_textures[tex_idx],
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: z0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(sub_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(resolution * 4),
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: depth,
            },
        );
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
            self.scene_buffer_capacity = needed;
            self.rebuild_scene_bind_group(device);
        }
        if !node_data.is_empty() {
            queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(node_data));
        }
    }

    pub fn update_voxel_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        voxel_data: &[f32],
    ) {
        let needed = voxel_data.len().max(INITIAL_VOXEL_CAPACITY);
        if needed > self.voxel_buffer_capacity {
            self.voxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Voxel Storage"),
                size: (needed * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.voxel_buffer_capacity = needed;
            self.rebuild_scene_bind_group(device);
        }
        if !voxel_data.is_empty() {
            queue.write_buffer(&self.voxel_buffer, 0, bytemuck::cast_slice(voxel_data));
        }
    }

    /// Upload only the modified z-slab range of a single voxel grid to GPU.
    pub fn update_voxel_region(
        &self,
        queue: &wgpu::Queue,
        grid_gpu_offset: u32,
        resolution: u32,
        z0: u32,
        z1: u32,
        grid_data: &[f32],
    ) {
        let slab_size = (resolution * resolution) as usize;
        let start_index = z0 as usize * slab_size;
        let end_index = ((z1 as usize) + 1) * slab_size;
        let sub_data = &grid_data[start_index..end_index];
        let byte_offset = ((grid_gpu_offset as usize) + start_index) * std::mem::size_of::<f32>();
        queue.write_buffer(
            &self.voxel_buffer,
            byte_offset as u64,
            bytemuck::cast_slice(sub_data),
        );
    }

    /// Dispatch pick compute shader and synchronously read back the result.
    /// Used for non-sculpt clicks (selection) where latency matters less.
    pub fn execute_pick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending: &PendingPick,
    ) -> Option<PickResult> {
        let rx = self.submit_pick(device, queue, pending);
        device.poll(wgpu::Maintain::Wait);
        Self::read_pick_from_receiver(&self.pick_staging_buffer, rx)
    }

    /// Submit pick compute shader (non-blocking). Returns a channel receiver
    /// that will signal when the staging buffer is mapped and ready to read.
    pub fn submit_pick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending: &PendingPick,
    ) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        // Write camera uniform
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&pending.camera_uniform),
        );

        // Write pick input (mouse_pos + padding)
        let pick_input: [f32; 4] = [pending.mouse_pos[0], pending.mouse_pos[1], 0.0, 0.0];
        queue.write_buffer(&self.pick_input_buffer, 0, bytemuck::cast_slice(&pick_input));

        // Encode compute dispatch + copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pick Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pick Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_bind_group(2, &self.pick_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.pick_output_buffer, 0, &self.pick_staging_buffer, 0, 32);

        queue.submit(std::iter::once(encoder.finish()));

        // Request async map — caller polls with device.poll(Poll)
        let buffer_slice = self.pick_staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        rx
    }

    /// Try to read a pick result from a previously submitted async pick.
    /// Returns None if the map hasn't completed yet or the result is invalid.
    pub fn try_read_pick_result(
        &self,
        rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Option<Option<PickResult>> {
        match rx.try_recv() {
            Ok(Ok(())) => {
                let result = Self::read_pick_from_staging(&self.pick_staging_buffer);
                Some(result)
            }
            Ok(Err(_)) => Some(None), // Map failed
            Err(std::sync::mpsc::TryRecvError::Empty) => None, // Not ready yet
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(None),
        }
    }

    fn read_pick_from_receiver(
        staging: &wgpu::Buffer,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Option<PickResult> {
        if rx.recv().ok()?.ok().is_none() {
            return None;
        }
        Self::read_pick_from_staging(staging)
    }

    fn read_pick_from_staging(staging: &wgpu::Buffer) -> Option<PickResult> {
        let buffer_slice = staging.slice(..);
        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = PickResult {
            material_id: floats[0] as i32,
            distance: floats[1],
            world_pos: [floats[2], floats[3], floats[4]],
        };
        drop(data);
        staging.unmap();

        // material_id < 0 means no hit
        if result.material_id < 0 || result.distance > 49.0 {
            return None;
        }

        Some(result)
    }

    /// Dispatch brush compute shader to modify voxel data directly on GPU.
    pub fn dispatch_brush(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dispatch: &BrushDispatch,
    ) {
        queue.write_buffer(
            &self.brush_uniform_buffer,
            0,
            bytemuck::bytes_of(&dispatch.params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Brush Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Brush Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brush_pipeline);
            pass.set_bind_group(0, &self.brush_bind_group, &[]);
            pass.dispatch_workgroups(
                dispatch.workgroups[0],
                dispatch.workgroups[1],
                dispatch.workgroups[2],
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
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
        render_pass.set_bind_group(2, &resources.voxel_tex_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

use crate::graph::scene::Scene;
use crate::sculpt::SculptState;
use crate::ui::gizmo::{self, GizmoMode, GizmoState};

const BRUSH_CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 200, 200, 128);

/// Returns an optional PendingPick if the user clicked/dragged in the viewport.
pub fn draw(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    sculpt_state: &SculptState,
    time: f32,
    render_config: &crate::settings::RenderConfig,
    sculpt_count: usize,
) -> Option<PendingPick> {
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

    // --- Paint the SDF viewport (WGPU callback) ---
    let pixels_per_point = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.width() * pixels_per_point,
        rect.height() * pixels_per_point,
    ];
    // Fast quality mode: half steps + skip AO/shadows
    let sculpt_active = sculpt_state.is_active();
    let camera_dragging = if sculpt_active {
        response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    } else {
        response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    };
    let sculpt_brushing = sculpt_active
        && response.dragged_by(egui::PointerButton::Primary)
        && render_config.sculpt_fast_mode;
    let multi_sculpt_reduce = render_config.auto_reduce_steps && sculpt_count >= 2;
    let quality_mode = if camera_dragging || sculpt_brushing || multi_sculpt_reduce {
        1.0
    } else {
        0.0
    };
    let scene_bounds = scene.compute_bounds();
    let uniform = camera.to_uniform(viewport, time, quality_mode, scene_bounds);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback { uniform },
    ));

    // --- Gizmo overlay (drawn on top of WGPU content) ---

    let gizmo_consumed = if sculpt_active {
        false // Gizmo is disabled during sculpt mode
    } else {
        gizmo::draw_and_interact(
            ui.painter(),
            &response,
            camera,
            scene,
            selected,
            gizmo_state,
            gizmo_mode,
            rect,
        )
    };

    // --- Interaction priority: sculpt > gizmo > pick > orbit ---
    let mut pending_pick = None;

    if sculpt_active {
        // Sculpt mode: drag applies brush continuously via pick
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(viewport, time, 0.0, scene_bounds),
                    });
                }
            }
        }

        // Brush cursor preview
        if let SculptState::Active { brush_radius, .. } = sculpt_state {
            if let Some(hover_pos) = response.hover_pos() {
                let screen_radius = brush_radius / camera.distance * rect.height() * 0.5;
                ui.painter().circle_stroke(
                    hover_pos,
                    screen_radius,
                    egui::Stroke::new(1.5, BRUSH_CURSOR_COLOR),
                );
            }
        }

        // Right-click still orbits in sculpt mode, secondary drag pans
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
        }
    } else if !gizmo_consumed {
        // Normal mode: click to pick
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let mouse_px = [
                    (pos.x - rect.min.x) * pixels_per_point,
                    (pos.y - rect.min.y) * pixels_per_point,
                ];
                let pick_uniform = camera.to_uniform(viewport, time, 0.0, scene_bounds);
                pending_pick = Some(PendingPick {
                    mouse_pos: mouse_px,
                    camera_uniform: pick_uniform,
                });
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }

    pending_pick
}
