mod composite;
mod draw;
mod environment;
mod environment_bake;
mod gpu_ops;
mod pipelines;
mod textures;

pub use composite::CompositeResources;
pub use draw::draw;
pub use environment::EnvironmentResources;

use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;

use crate::gpu::buffers::SdfNodeGpu;
use crate::gpu::camera::CameraUniform;

const INITIAL_SCENE_CAPACITY: usize = 16;
const INITIAL_VOXEL_CAPACITY: usize = 4; // in f32 elements (minimum valid buffer)

/// GPU-side brush parameters. Matches the WGSL `BrushParams` struct layout (128 bytes).
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
    // f32 surface_constraint + pad + vec3f view_dir_local + pad (128-byte struct)
    pub surface_constraint: f32,
    pub _pad3: [f32; 3],
    pub view_dir_local: [f32; 3],
    pub _pad4: f32,
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
    // BGLs are `Arc`-wrapped so the async pipeline compile worker can hold
    // its own reference while the main thread keeps rendering.
    pub camera_bgl: Arc<wgpu::BindGroupLayout>,
    pub scene_buffer: wgpu::Buffer,
    pub voxel_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bgl: Arc<wgpu::BindGroupLayout>,
    pub scene_buffer_capacity: usize,
    pub voxel_buffer_capacity: usize, // in f32 elements
    pub target_format: wgpu::TextureFormat,

    // --- Voxel texture3D resources (render shader only) ---
    pub voxel_textures: Vec<wgpu::Texture>,
    pub voxel_texture_views: Vec<wgpu::TextureView>,
    pub voxel_sampler: wgpu::Sampler,
    pub voxel_tex_bgl: Arc<wgpu::BindGroupLayout>,
    pub voxel_tex_bind_group: wgpu::BindGroup,

    // --- Environment resources (IBL cubemaps + BRDF LUT) ---
    pub environment: EnvironmentResources,

    // --- Pick compute pipeline ---
    pub pick_pipeline: wgpu::ComputePipeline,
    pub pick_input_buffer: wgpu::Buffer,
    pub pick_output_buffer: wgpu::Buffer,
    pub pick_staging_buffer: wgpu::Buffer,
    pub pick_bind_group: wgpu::BindGroup,
    pub pick_bgl: Arc<wgpu::BindGroupLayout>,

    // --- Brush compute pipeline ---
    pub brush_pipeline: wgpu::ComputePipeline,
    pub brush_uniform_buffer: wgpu::Buffer,
    pub brush_uniform_stride: u64,
    pub brush_uniform_capacity: u32,
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

pub(crate) const BLIT_SHADER_SRC: &str = include_str!("../../shaders/blit.wgsl");

impl ViewportResources {
    fn brush_param_size() -> u64 {
        std::mem::size_of::<BrushGpuParams>() as u64
    }

    fn brush_binding_size() -> NonZeroU64 {
        NonZeroU64::new(Self::brush_param_size()).expect("BrushGpuParams size must be non-zero")
    }

    fn brush_uniform_stride(device: &wgpu::Device) -> u64 {
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1) as u64;
        Self::brush_param_size().div_ceil(align) * align
    }

    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
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

        let camera_bgl = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        }));

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

        let scene_bgl = Arc::new(Self::create_scene_bgl(device));

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
        let voxel_tex_bgl = Arc::new(Self::create_voxel_tex_bgl(device, 0));
        let voxel_tex_bind_group =
            Self::create_voxel_tex_bind_group(device, &voxel_tex_bgl, &voxel_sampler, &[]);
        let environment = EnvironmentResources::new(device, adapter);

        let pipeline = Self::create_render_pipeline(
            device,
            shader_src,
            &camera_bgl,
            &scene_bgl,
            &voxel_tex_bgl,
            &environment.bind_group_layout,
            target_format,
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

        let pick_bgl = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        }));

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

        let pick_pipeline =
            Self::create_pick_pipeline(device, pick_shader_src, &camera_bgl, &scene_bgl, &pick_bgl);

        // --- Brush compute resources ---
        let brush_bgl = Self::create_brush_bgl(device);
        let brush_uniform_stride = Self::brush_uniform_stride(device);
        let brush_uniform_capacity = 1;
        let brush_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brush Uniform"),
            size: brush_uniform_stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let brush_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brush BG"),
            layout: &brush_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &brush_uniform_buffer,
                        offset: 0,
                        size: Some(Self::brush_binding_size()),
                    }),
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
            size: 48, // vec4f viewport + vec4f outline_color_and_width + vec4f bloom_params
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
            environment,
            pick_pipeline,
            pick_input_buffer,
            pick_output_buffer,
            pick_staging_buffer,
            pick_bind_group,
            pick_bgl,
            brush_pipeline,
            brush_uniform_buffer,
            brush_uniform_stride,
            brush_uniform_capacity,
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
            device,
            shader_src,
            &self.camera_bgl,
            &self.scene_bgl,
            &self.voxel_tex_bgl,
            &self.environment.bind_group_layout,
            self.target_format,
        );
        self.pick_pipeline = Self::create_pick_pipeline(
            device,
            pick_shader_src,
            &self.camera_bgl,
            &self.scene_bgl,
            &self.pick_bgl,
        );
    }

    /// Ensure the offscreen render texture matches the given dimensions.
    /// Recreates it and the blit bind group when the size changes.
    pub fn ensure_offscreen_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.render_width == width
            && self.render_height == height
            && self.offscreen_texture.is_some()
        {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen RT"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

    pub(super) fn rebuild_brush_bind_group(&mut self, device: &wgpu::Device) {
        self.brush_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brush BG"),
            layout: &self.brush_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.brush_uniform_buffer,
                        offset: 0,
                        size: Some(Self::brush_binding_size()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub(super) fn ensure_brush_uniform_capacity(&mut self, device: &wgpu::Device, needed: u32) {
        let needed = needed.max(1);
        if needed <= self.brush_uniform_capacity {
            return;
        }

        let new_capacity = needed.next_power_of_two();
        self.brush_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brush Uniform"),
            size: self.brush_uniform_stride * new_capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.brush_uniform_capacity = new_capacity;
        self.rebuild_brush_bind_group(device);
    }

    pub(super) fn rebuild_scene_bind_group(&mut self, device: &wgpu::Device) {
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
}
