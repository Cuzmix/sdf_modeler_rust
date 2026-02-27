use bytemuck::{Pod, Zeroable};
use eframe::wgpu;

use super::ViewportResources;

/// GPU-side composite volume parameters. Matches WGSL `CompositeParams` (64 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CompositeParams {
    pub bounds_min: [f32; 4], // xyz + pad
    pub bounds_max: [f32; 4], // xyz + pad
    pub resolution: u32,
    pub update_min: [u32; 3],
    pub update_max: [u32; 3],
    pub _pad: u32,
}

// Compile-time layout assertion: CompositeParams must be exactly 64 bytes.
const _: () = assert!(std::mem::size_of::<CompositeParams>() == 64);

/// GPU resources for the composite scene volume cache.
pub struct CompositeResources {
    pub sdf_texture: wgpu::Texture,
    pub sdf_view: wgpu::TextureView,
    pub mat_texture: wgpu::Texture,
    pub mat_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bgl: wgpu::BindGroupLayout,
    pub compute_bg: wgpu::BindGroup,
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bgl: wgpu::BindGroupLayout,
    pub render_bg: wgpu::BindGroup,
    pub params_buffer: wgpu::Buffer,
    pub resolution: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

impl ViewportResources {
    /// Create the composite compute bind group layout (@group(3) for compute shader).
    pub(super) fn create_comp_compute_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Comp Compute BGL"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Snorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the composite render bind group layout (@group(2) for composite render shader).
    pub(super) fn create_comp_render_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Comp Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Build or rebuild all composite volume resources (textures, pipelines, bind groups).
    pub fn rebuild_composite(
        &mut self,
        device: &wgpu::Device,
        comp_compute_src: &str,
        comp_render_src: &str,
        resolution: u32,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) {
        let res = resolution;

        // Create 3D textures (STORAGE_BINDING for compute write + TEXTURE_BINDING for render read)
        let sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Comp SDF Tex"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mat_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Comp Mat Tex"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Comp Normal Tex"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sdf_view = sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mat_view = mat_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Comp Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Comp Params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute pipeline: groups 0-2 = existing (camera, scene, voxel_tex), group 3 = composite
        let compute_bgl = Self::create_comp_compute_bgl(device);
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Comp Compute BG"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sdf_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mat_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal_view) },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Comp Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(comp_compute_src.into()),
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Comp Compute Layout"),
            bind_group_layouts: &[&self.camera_bgl, &self.scene_bgl, &self.voxel_tex_bgl, &compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Comp Compute Pipeline"),
            layout: Some(&compute_layout),
            module: &compute_shader,
            entry_point: "cs_composite",
            compilation_options: Default::default(),
            cache: None,
        });

        // Render pipeline: groups 0-1 = existing (camera, scene), group 2 = composite textures
        let render_bgl = Self::create_comp_render_bgl(device);
        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Comp Render BG"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sdf_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mat_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal_view) },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Comp Render Shader"),
            source: wgpu::ShaderSource::Wgsl(comp_render_src.into()),
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Comp Render Layout"),
            bind_group_layouts: &[&self.camera_bgl, &self.scene_bgl, &render_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Comp Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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

        self.composite = Some(CompositeResources {
            sdf_texture,
            sdf_view,
            mat_texture,
            mat_view,
            normal_texture,
            normal_view,
            sampler,
            compute_pipeline,
            compute_bgl,
            compute_bg,
            render_pipeline,
            render_bgl,
            render_bg,
            params_buffer,
            resolution,
            bounds_min,
            bounds_max,
        });
        self.use_composite = true;
    }

    /// Dispatch the composite compute shader over a sub-region of the volume.
    pub fn dispatch_composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        update_min: [u32; 3],
        update_max: [u32; 3],
    ) {
        let Some(ref comp) = self.composite else { return; };

        let params = CompositeParams {
            bounds_min: [comp.bounds_min[0], comp.bounds_min[1], comp.bounds_min[2], 0.0],
            bounds_max: [comp.bounds_max[0], comp.bounds_max[1], comp.bounds_max[2], 0.0],
            resolution: comp.resolution,
            update_min,
            update_max,
            _pad: 0,
        };
        queue.write_buffer(&comp.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Comp Dispatch Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Comp Dispatch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&comp.compute_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_bind_group(2, &self.voxel_tex_bind_group, &[]);
            pass.set_bind_group(3, &comp.compute_bg, &[]);

            // Workgroups: ceil((update_max - update_min + 1) / 4) per axis
            let wx = ((update_max[0] - update_min[0] + 4) / 4).max(1);
            let wy = ((update_max[1] - update_min[1] + 4) / 4).max(1);
            let wz = ((update_max[2] - update_min[2] + 4) / 4).max(1);
            pass.dispatch_workgroups(wx, wy, wz);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}
