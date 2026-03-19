use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;

use crate::settings::{BackgroundMode, EnvironmentSource, RenderConfig};

const ENVIRONMENT_BAKE_SHADER_SRC: &str = include_str!("../../shaders/environment_bake.wgsl");
const MAX_BAKE_PASS_COUNT: u32 = 64;

pub struct EnvironmentBakeInputs<'a> {
    pub source_texture: &'a wgpu::Texture,
    pub irradiance_texture: &'a wgpu::Texture,
    pub prefiltered_texture: &'a wgpu::Texture,
    pub brdf_lut_texture: &'a wgpu::Texture,
    pub prefiltered_mip_count: u32,
    pub hdri_view: Option<&'a wgpu::TextureView>,
}

pub struct EnvironmentBakeGpu {
    uniform_buffer: wgpu::Buffer,
    uniform_stride: u64,
    uniform_bind_group: wgpu::BindGroup,
    source_bgl: wgpu::BindGroupLayout,
    source_sampler: wgpu::Sampler,
    _fallback_source_texture: wgpu::Texture,
    fallback_source_view: wgpu::TextureView,
    source_pipeline: wgpu::RenderPipeline,
    irradiance_pipeline: wgpu::RenderPipeline,
    prefiltered_pipeline: wgpu::RenderPipeline,
    brdf_lut_pipeline: wgpu::RenderPipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EnvironmentBakeUniform {
    face_and_flags: [u32; 4],
    params: [f32; 4],
    sky_horizon: [f32; 4],
    sky_zenith: [f32; 4],
    solid_color: [f32; 4],
}

#[derive(Copy, Clone)]
enum BakePassKind {
    Source,
    Irradiance,
    Prefiltered,
    BrdfLut,
}

#[derive(Copy, Clone)]
struct BakePassDescriptor {
    kind: BakePassKind,
    face: u32,
    mip_level: u32,
    resolution: u32,
    roughness: f32,
}

impl EnvironmentBakeGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_stride = Self::uniform_stride(device);
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Environment Bake Uniform"),
            size: uniform_stride * MAX_BAKE_PASS_COUNT as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment Bake Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(Self::uniform_binding_size()),
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Environment Bake Uniform BG"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: Some(Self::uniform_binding_size()),
                }),
            }],
        });
        let source_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment Bake Source BGL"),
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
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let source_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Bake Source Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let fallback_source_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Bake Fallback Source"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_source_view =
            fallback_source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Environment Bake Shader"),
            source: wgpu::ShaderSource::Wgsl(ENVIRONMENT_BAKE_SHADER_SRC.into()),
        });
        let cube_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Environment Cube Bake Layout"),
            bind_group_layouts: &[&uniform_bgl, &source_bgl],
            push_constant_ranges: &[],
        });
        let brdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Environment BRDF Bake Layout"),
            bind_group_layouts: &[&uniform_bgl, &source_bgl],
            push_constant_ranges: &[],
        });

        Self {
            uniform_buffer,
            uniform_stride,
            uniform_bind_group,
            source_bgl,
            source_sampler,
            _fallback_source_texture: fallback_source_texture,
            fallback_source_view,
            source_pipeline: Self::create_cube_pipeline(
                device,
                &shader,
                &cube_layout,
                "fs_source_cube",
                "Environment Source Bake Pipeline",
            ),
            irradiance_pipeline: Self::create_cube_pipeline(
                device,
                &shader,
                &cube_layout,
                "fs_irradiance_cube",
                "Environment Irradiance Bake Pipeline",
            ),
            prefiltered_pipeline: Self::create_cube_pipeline(
                device,
                &shader,
                &cube_layout,
                "fs_prefiltered_cube",
                "Environment Prefiltered Bake Pipeline",
            ),
            brdf_lut_pipeline: Self::create_brdf_pipeline(
                device,
                &shader,
                &brdf_layout,
                "Environment BRDF LUT Bake Pipeline",
            ),
        }
    }

    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &RenderConfig,
        inputs: EnvironmentBakeInputs<'_>,
    ) {
        let bake_passes = Self::build_bake_passes(inputs.prefiltered_mip_count);
        assert!(
            bake_passes.len() <= MAX_BAKE_PASS_COUNT as usize,
            "Environment bake pass capacity exceeded"
        );

        let uniform_bytes = self.encode_uniforms(config, &bake_passes);
        queue.write_buffer(&self.uniform_buffer, 0, &uniform_bytes);

        let hdri_view = inputs.hdri_view.unwrap_or(&self.fallback_source_view);
        let source_bind_group = self.create_source_bind_group(device, hdri_view);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Environment Bake Encoder"),
        });

        for (pass_index, pass) in bake_passes.iter().enumerate() {
            let dynamic_offset = (pass_index as u64 * self.uniform_stride)
                .try_into()
                .expect("Environment bake offset must fit in u32");

            let target_view = match pass.kind {
                BakePassKind::Source => {
                    Self::create_cube_face_view(inputs.source_texture, pass.face, pass.mip_level)
                }
                BakePassKind::Irradiance => Self::create_cube_face_view(
                    inputs.irradiance_texture,
                    pass.face,
                    pass.mip_level,
                ),
                BakePassKind::Prefiltered => Self::create_cube_face_view(
                    inputs.prefiltered_texture,
                    pass.face,
                    pass.mip_level,
                ),
                BakePassKind::BrdfLut => inputs
                    .brdf_lut_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            };
            let pipeline = match pass.kind {
                BakePassKind::Source => &self.source_pipeline,
                BakePassKind::Irradiance => &self.irradiance_pipeline,
                BakePassKind::Prefiltered => &self.prefiltered_pipeline,
                BakePassKind::BrdfLut => &self.brdf_lut_pipeline,
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Environment Bake Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[dynamic_offset]);
            render_pass.set_bind_group(1, &source_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    fn create_cube_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        fragment_entry_point: &str,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_environment_bake",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: fragment_entry_point,
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_brdf_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_environment_bake",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_brdf_lut",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_source_bind_group(
        &self,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Environment Bake Source BG"),
            layout: &self.source_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.source_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
            ],
        })
    }

    fn create_cube_face_view(
        texture: &wgpu::Texture,
        face: u32,
        mip_level: u32,
    ) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Environment Cube Face View"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: face,
            array_layer_count: Some(1),
        })
    }

    fn build_bake_passes(prefiltered_mip_count: u32) -> Vec<BakePassDescriptor> {
        let mut passes = Vec::with_capacity((6 * (2 + prefiltered_mip_count) + 1) as usize);
        for face in 0..6 {
            passes.push(BakePassDescriptor {
                kind: BakePassKind::Source,
                face,
                mip_level: 0,
                resolution: super::environment::SOURCE_ENV_RESOLUTION,
                roughness: 0.0,
            });
            passes.push(BakePassDescriptor {
                kind: BakePassKind::Irradiance,
                face,
                mip_level: 0,
                resolution: super::environment::IRRADIANCE_ENV_RESOLUTION,
                roughness: 0.0,
            });
            for mip_level in 0..prefiltered_mip_count {
                let resolution =
                    (super::environment::PREFILTERED_ENV_RESOLUTION >> mip_level).max(1);
                let roughness = if prefiltered_mip_count <= 1 {
                    0.0
                } else {
                    mip_level as f32 / (prefiltered_mip_count - 1) as f32
                };
                passes.push(BakePassDescriptor {
                    kind: BakePassKind::Prefiltered,
                    face,
                    mip_level,
                    resolution,
                    roughness,
                });
            }
        }
        passes.push(BakePassDescriptor {
            kind: BakePassKind::BrdfLut,
            face: 0,
            mip_level: 0,
            resolution: super::environment::BRDF_LUT_RESOLUTION,
            roughness: 0.0,
        });
        passes
    }

    fn encode_uniforms(
        &self,
        config: &RenderConfig,
        bake_passes: &[BakePassDescriptor],
    ) -> Vec<u8> {
        let mut bytes = vec![0; self.uniform_stride as usize * bake_passes.len()];
        for (index, pass) in bake_passes.iter().enumerate() {
            let uniform = Self::build_uniform(config, *pass);
            let dst = index * self.uniform_stride as usize;
            let src = bytemuck::bytes_of(&uniform);
            bytes[dst..dst + src.len()].copy_from_slice(src);
        }
        bytes
    }

    fn build_uniform(config: &RenderConfig, pass: BakePassDescriptor) -> EnvironmentBakeUniform {
        EnvironmentBakeUniform {
            face_and_flags: [
                pass.face,
                match config.environment_source {
                    EnvironmentSource::ProceduralSky => 0,
                    EnvironmentSource::Hdri => 1,
                },
                u32::from(config.environment_source == EnvironmentSource::Hdri),
                match config.background_mode {
                    BackgroundMode::SkyGradient => 0,
                    BackgroundMode::SolidColor => 1,
                },
            ],
            params: [
                pass.roughness,
                config.environment_rotation_degrees.to_radians(),
                config.environment_exposure.exp2(),
                pass.resolution as f32,
            ],
            sky_horizon: [
                config.sky_horizon[0],
                config.sky_horizon[1],
                config.sky_horizon[2],
                0.0,
            ],
            sky_zenith: [
                config.sky_zenith[0],
                config.sky_zenith[1],
                config.sky_zenith[2],
                0.0,
            ],
            solid_color: [
                config.bg_solid_color[0],
                config.bg_solid_color[1],
                config.bg_solid_color[2],
                0.0,
            ],
        }
    }

    fn uniform_binding_size() -> NonZeroU64 {
        NonZeroU64::new(std::mem::size_of::<EnvironmentBakeUniform>() as u64)
            .expect("Environment bake uniform size must be non-zero")
    }

    fn uniform_stride(device: &wgpu::Device) -> u64 {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1) as u64;
        (std::mem::size_of::<EnvironmentBakeUniform>() as u64).div_ceil(alignment) * alignment
    }
}

#[cfg(test)]
mod tests {
    use naga::front::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::{EnvironmentBakeGpu, ENVIRONMENT_BAKE_SHADER_SRC};

    #[test]
    fn environment_bake_shader_validates_with_naga() {
        let module = wgsl::parse_str(ENVIRONMENT_BAKE_SHADER_SRC)
            .expect("environment bake shader should parse");
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .expect("environment bake shader should validate");
    }

    #[test]
    fn environment_bake_pass_count_matches_capacity_budget() {
        let mip_count = super::super::environment::PREFILTERED_ENV_RESOLUTION.ilog2() + 1;
        let pass_count = EnvironmentBakeGpu::build_bake_passes(mip_count).len();
        assert!(pass_count <= 64);
        assert_eq!(pass_count, 61);
    }
}
