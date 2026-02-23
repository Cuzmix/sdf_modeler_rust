use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen::SdfNodeGpu;
use crate::gpu::picking::{PendingPick, PickResult};
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::SculptState;
use crate::ui::gizmo::{self, GizmoMode, GizmoState};

const INITIAL_SCENE_CAPACITY: usize = 16;
const INITIAL_VOXEL_CAPACITY: usize = 4; // in f32 elements (minimum valid buffer)

/// GPU-side brush parameters. Matches the WGSL `BrushParams` struct layout (112 bytes).
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
    // f32 brush_mode, f32 falloff_mode, u32 smooth_iterations, f32 flatten_ref
    pub brush_mode: f32,
    pub falloff_mode: f32,
    pub smooth_iterations: u32,
    pub flatten_ref: f32,
    // f32 surface_constraint + padding to 112 bytes (16-byte aligned)
    pub surface_constraint: f32,
    pub _pad3: [f32; 3],
}

/// CPU-side brush dispatch info (wraps GPU params + workgroup counts).
pub struct BrushDispatch {
    pub params: BrushGpuParams,
    pub workgroups: [u32; 3],
}

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

    // --- Resolution scaling (offscreen render + blit) ---
    pub offscreen_texture: Option<wgpu::Texture>,
    pub offscreen_view: Option<wgpu::TextureView>,
    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bgl: wgpu::BindGroupLayout,
    pub blit_sampler: wgpu::Sampler,
    pub blit_bind_group: Option<wgpu::BindGroup>,
    pub blit_params_buffer: wgpu::Buffer,
    pub render_width: u32,
    pub render_height: u32,

    // --- Composite scene volume cache ---
    pub composite: Option<CompositeResources>,
    pub use_composite: bool,
}

const BLIT_SHADER_SRC: &str = r#"
struct BlitParams {
    viewport: vec4f,
}

@group(0) @binding(0) var<uniform> params: BlitParams;
@group(0) @binding(1) var blit_sampler: sampler;
@group(0) @binding(2) var blit_texture: texture_2d<f32>;

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_blit(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let uv = (frag_coord.xy - params.viewport.xy) / params.viewport.zw;
    return textureSampleLevel(blit_texture, blit_sampler, uv, 0.0);
}
"#;

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

    fn create_blit_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_blit_pipeline(
        device: &wgpu::Device,
        blit_bgl: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER_SRC.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[blit_bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_blit",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_blit",
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

    /// Ensure the offscreen render texture matches the given dimensions.
    /// Recreates it and the blit bind group when the size changes.
    pub fn ensure_offscreen_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.render_width == width && self.render_height == height
            && self.offscreen_texture.is_some()
        {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen RT"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.blit_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blit_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        }));
        self.offscreen_view = Some(view);
        self.offscreen_texture = Some(tex);
        self.render_width = width;
        self.render_height = height;
    }

    /// Create the bind group layout for voxel textures: binding 0 = sampler, then N texture_3d bindings.
    fn create_voxel_tex_bgl(device: &wgpu::Device, tex_count: usize) -> wgpu::BindGroupLayout {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }];
        for i in 0..tex_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
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

        // --- Blit (resolution scaling) resources ---
        let blit_bgl = Self::create_blit_bgl(device);
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blit Params"),
            size: 16, // vec4f viewport
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blit_pipeline = Self::create_blit_pipeline(device, &blit_bgl, target_format);

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
            offscreen_texture: None,
            offscreen_view: None,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            blit_bind_group: None,
            blit_params_buffer,
            render_width: 0,
            render_height: 0,
            composite: None,
            use_composite: false,
        }
    }

    /// Create the composite compute bind group layout (@group(3) for compute shader).
    fn create_comp_compute_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_comp_render_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

    /// Render the viewport to an offscreen texture and return RGBA pixel data.
    pub fn screenshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform: &CameraUniform,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // Write camera uniform
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniform));

        // Create offscreen texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Screenshot Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Row alignment for buffer copy
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * height) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        // Render pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screenshot Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            pass.set_bind_group(2, &self.voxel_tex_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy texture to staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_row) as usize;
            let end = start + unpadded_row as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        staging.unmap();

        pixels
    }
}

// ---------------------------------------------------------------------------
// Paint callback
// ---------------------------------------------------------------------------

struct ViewportCallback {
    /// Camera uniform with viewport set to render dimensions (offscreen texture size).
    render_uniform: CameraUniform,
    /// Display viewport in physical pixels: [x, y, width, height].
    display_viewport: [f32; 4],
    /// Render scale factor (0.25 - 1.0).
    render_scale: f32,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources
            .get_mut::<ViewportResources>()
            .unwrap();

        let display_w = self.display_viewport[2] as u32;
        let display_h = self.display_viewport[3] as u32;
        let render_w = ((display_w as f32) * self.render_scale).max(1.0) as u32;
        let render_h = ((display_h as f32) * self.render_scale).max(1.0) as u32;

        // Ensure offscreen texture + blit bind group are the right size
        resources.ensure_offscreen_texture(device, render_w, render_h);

        // Write camera uniform (viewport = render dimensions for the SDF shader)
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&self.render_uniform),
        );

        // Write blit params (display viewport for the blit shader)
        queue.write_buffer(
            &resources.blit_params_buffer,
            0,
            bytemuck::cast_slice(&self.display_viewport),
        );

        // Render SDF scene to offscreen texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Offscreen Encoder"),
        });
        {
            let view = resources.offscreen_view.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Offscreen SDF Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            // Use composite render pipeline when enabled, otherwise direct render
            if resources.use_composite {
                if let Some(ref comp) = resources.composite {
                    pass.set_pipeline(&comp.render_pipeline);
                    pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                    pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                    pass.set_bind_group(2, &comp.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&resources.pipeline);
                pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                pass.set_bind_group(2, &resources.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        if let Some(ref blit_bg) = resources.blit_bind_group {
            render_pass.set_pipeline(&resources.blit_pipeline);
            render_pass.set_bind_group(0, blit_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

const BRUSH_CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 200, 200, 128);

/// Draw a semi-transparent symmetry plane overlay at the mirror axis.
fn draw_symmetry_plane(painter: &egui::Painter, camera: &Camera, rect: egui::Rect, axis: u8) {
    let aspect = rect.width() / rect.height();
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    let extent = 5.0_f32;
    let corners: [glam::Vec3; 4] = match axis {
        0 => [
            glam::Vec3::new(0.0, -extent, -extent),
            glam::Vec3::new(0.0, -extent, extent),
            glam::Vec3::new(0.0, extent, extent),
            glam::Vec3::new(0.0, extent, -extent),
        ],
        1 => [
            glam::Vec3::new(-extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, extent),
            glam::Vec3::new(-extent, 0.0, extent),
        ],
        _ => [
            glam::Vec3::new(-extent, -extent, 0.0),
            glam::Vec3::new(extent, -extent, 0.0),
            glam::Vec3::new(extent, extent, 0.0),
            glam::Vec3::new(-extent, extent, 0.0),
        ],
    };

    let screen_pts: Vec<egui::Pos2> = corners
        .iter()
        .filter_map(|&c| gizmo::world_to_screen(c, &view_proj, rect))
        .collect();

    if screen_pts.len() < 3 {
        return;
    }

    let (fill, border) = match axis {
        0 => (
            egui::Color32::from_rgba_premultiplied(255, 50, 50, 20),
            egui::Color32::from_rgba_premultiplied(255, 80, 80, 80),
        ),
        1 => (
            egui::Color32::from_rgba_premultiplied(50, 255, 50, 20),
            egui::Color32::from_rgba_premultiplied(80, 255, 80, 80),
        ),
        _ => (
            egui::Color32::from_rgba_premultiplied(50, 50, 255, 20),
            egui::Color32::from_rgba_premultiplied(80, 80, 255, 80),
        ),
    };

    painter.add(egui::Shape::convex_polygon(
        screen_pts,
        fill,
        egui::Stroke::new(1.0, border),
    ));
}

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
    fps_info: Option<(f64, f64)>, // (fps, frame_ms)
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
    // Interaction detection
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
        && response.dragged_by(egui::PointerButton::Primary);
    let multi_sculpt_reduce = render_config.auto_reduce_steps && sculpt_count >= 2;
    let is_interacting = camera_dragging || sculpt_brushing;
    // Fast quality mode: half steps + skip AO/shadows
    let quality_mode = if (is_interacting && render_config.sculpt_fast_mode) || multi_sculpt_reduce {
        1.0
    } else {
        0.0
    };

    // Resolution scaling: reduced resolution during interaction
    let render_scale = if is_interacting {
        render_config.interaction_render_scale.clamp(0.25, 1.0)
    } else {
        render_config.rest_render_scale.clamp(0.25, 1.0)
    };

    let scene_bounds = scene.compute_bounds();

    // Render uniform uses the RENDER viewport (offscreen texture dimensions)
    let render_w = (viewport[2] * render_scale).max(1.0);
    let render_h = (viewport[3] * render_scale).max(1.0);
    let render_viewport = [0.0, 0.0, render_w, render_h];
    let render_uniform = camera.to_uniform(render_viewport, time, quality_mode, render_config.show_grid, scene_bounds);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback {
            render_uniform,
            display_viewport: viewport,
            render_scale,
        },
    ));

    // --- Symmetry plane overlay ---
    if let Some(axis) = sculpt_state.symmetry_axis() {
        draw_symmetry_plane(ui.painter(), camera, rect, axis);
    }

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
                        camera_uniform: camera.to_uniform(viewport, time, 0.0, false, scene_bounds),
                    });
                }
            }
        }

        // Enhanced brush cursor preview
        if let SculptState::Active {
            brush_radius,
            brush_strength,
            symmetry_axis,
            ..
        } = sculpt_state
        {
            if let Some(hover_pos) = response.hover_pos() {
                let screen_radius = brush_radius / camera.distance * rect.height() * 0.5;

                // Outer ring: brush extent
                ui.painter().circle_stroke(
                    hover_pos,
                    screen_radius,
                    egui::Stroke::new(1.5, BRUSH_CURSOR_COLOR),
                );

                // Inner fill: strength indicator (opacity proportional to strength)
                let strength_alpha = (brush_strength / 0.5 * 60.0).clamp(8.0, 60.0) as u8;
                ui.painter().circle_filled(
                    hover_pos,
                    screen_radius * 0.6,
                    egui::Color32::from_rgba_premultiplied(200, 200, 200, strength_alpha),
                );

                // Crosshair center dot
                ui.painter().circle_filled(
                    hover_pos,
                    2.0,
                    egui::Color32::from_rgba_premultiplied(255, 255, 255, 160),
                );

                // Symmetry mirror cursor
                if let Some(axis) = symmetry_axis {
                    let mirror_color = match axis {
                        0 => egui::Color32::from_rgba_premultiplied(255, 100, 100, 100),
                        1 => egui::Color32::from_rgba_premultiplied(100, 255, 100, 100),
                        _ => egui::Color32::from_rgba_premultiplied(100, 100, 255, 100),
                    };
                    // Mirror the hover position through the symmetry plane in screen space
                    // Project the origin and the mirrored point to get screen-space mirror
                    let aspect = rect.width() / rect.height();
                    let vp = camera.projection_matrix(aspect) * camera.view_matrix();
                    let origin = gizmo::world_to_screen(glam::Vec3::ZERO, &vp, rect);
                    if let Some(origin_screen) = origin {
                        // Mirror hover_pos around the axis line through origin
                        let mirror_pos = match axis {
                            0 => {
                                // X symmetry: mirror horizontally around origin.x
                                egui::pos2(
                                    2.0 * origin_screen.x - hover_pos.x,
                                    hover_pos.y,
                                )
                            }
                            1 => {
                                // Y symmetry: mirror vertically around origin.y
                                egui::pos2(
                                    hover_pos.x,
                                    2.0 * origin_screen.y - hover_pos.y,
                                )
                            }
                            _ => {
                                // Z symmetry: approximate mirror horizontally
                                egui::pos2(
                                    2.0 * origin_screen.x - hover_pos.x,
                                    hover_pos.y,
                                )
                            }
                        };
                        if rect.contains(mirror_pos) {
                            ui.painter().circle_stroke(
                                mirror_pos,
                                screen_radius,
                                egui::Stroke::new(1.0, mirror_color),
                            );
                            ui.painter().circle_filled(
                                mirror_pos,
                                2.0,
                                mirror_color,
                            );
                        }
                    }
                }
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
                let pick_uniform = camera.to_uniform(viewport, time, 0.0, false, scene_bounds);
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

    // --- FPS counter overlay (top-left of viewport) ---
    if let Some((fps, frame_ms)) = fps_info {
        let text = format!("{:.0} FPS ({:.1} ms)", fps, frame_ms);
        let font = egui::FontId::monospace(11.0);
        let pos = rect.min + egui::vec2(6.0, 4.0);
        // Shadow for readability
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::LEFT_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        let color = if fps >= 55.0 {
            egui::Color32::from_rgb(120, 220, 120) // green
        } else if fps >= 30.0 {
            egui::Color32::from_rgb(220, 200, 80) // yellow
        } else {
            egui::Color32::from_rgb(220, 100, 100) // red
        };
        ui.painter().text(
            pos,
            egui::Align2::LEFT_TOP,
            &text,
            font,
            color,
        );
    }

    pending_pick
}
