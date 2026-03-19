use std::f32::consts::PI;

use eframe::wgpu;
use glam::{Vec2, Vec3};
use half::f16;

use crate::settings::{BackgroundMode, RenderConfig};

use super::ViewportResources;

const SOURCE_ENV_RESOLUTION: u32 = 64;
const IRRADIANCE_ENV_RESOLUTION: u32 = 16;
const PREFILTERED_ENV_RESOLUTION: u32 = 32;
const BRDF_LUT_RESOLUTION: u32 = 128;
const IRRADIANCE_SAMPLE_COUNT: u32 = 64;
const PREFILTER_SAMPLE_COUNT: u32 = 64;
const BRDF_SAMPLE_COUNT: u32 = 128;

pub struct EnvironmentResources {
    pub sampler: wgpu::Sampler,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub source_texture: wgpu::Texture,
    pub source_view: wgpu::TextureView,
    pub irradiance_texture: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    pub prefiltered_texture: wgpu::Texture,
    pub prefiltered_view: wgpu::TextureView,
    pub brdf_lut_texture: wgpu::Texture,
    pub brdf_lut_view: wgpu::TextureView,
    pub prefiltered_mip_count: u32,
}

impl EnvironmentResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind_group_layout = Self::create_bind_group_layout(device);
        let (source_texture, source_view) = Self::create_cube_texture(device, 1, 1, "Env Source");
        let (irradiance_texture, irradiance_view) =
            Self::create_cube_texture(device, 1, 1, "Env Irradiance");
        let (prefiltered_texture, prefiltered_view) =
            Self::create_cube_texture(device, 1, 1, "Env Prefiltered");
        let (brdf_lut_texture, brdf_lut_view) =
            Self::create_brdf_lut_texture(device, 1, "Env BRDF LUT");
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &sampler,
            &source_view,
            &irradiance_view,
            &prefiltered_view,
            &brdf_lut_view,
        );

        Self {
            sampler,
            bind_group_layout,
            bind_group,
            source_texture,
            source_view,
            irradiance_texture,
            irradiance_view,
            prefiltered_texture,
            prefiltered_view,
            brdf_lut_texture,
            brdf_lut_view,
            prefiltered_mip_count: 1,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment BGL"),
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
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

    fn create_cube_texture(
        device: &wgpu::Device,
        resolution: u32,
        mip_level_count: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        (texture, view)
    }

    fn create_brdf_lut_texture(
        device: &wgpu::Device,
        resolution: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        source_view: &wgpu::TextureView,
        irradiance_view: &wgpu::TextureView,
        prefiltered_view: &wgpu::TextureView,
        brdf_lut_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Environment BG"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(brdf_lut_view),
                },
            ],
        })
    }

    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, config: &RenderConfig) {
        let prefiltered_mip_count = PREFILTERED_ENV_RESOLUTION.ilog2() + 1;
        let (source_texture, source_view) =
            Self::create_cube_texture(device, SOURCE_ENV_RESOLUTION, 1, "Env Source");
        let (irradiance_texture, irradiance_view) =
            Self::create_cube_texture(device, IRRADIANCE_ENV_RESOLUTION, 1, "Env Irradiance");
        let (prefiltered_texture, prefiltered_view) = Self::create_cube_texture(
            device,
            PREFILTERED_ENV_RESOLUTION,
            prefiltered_mip_count,
            "Env Prefiltered",
        );
        let (brdf_lut_texture, brdf_lut_view) =
            Self::create_brdf_lut_texture(device, BRDF_LUT_RESOLUTION, "Env BRDF LUT");

        for face in 0..6 {
            let source_face = bake_environment_cube_face(
                SOURCE_ENV_RESOLUTION,
                face,
                0.0,
                CubeBakeMode::Source,
                config,
            );
            write_cube_face(
                queue,
                &source_texture,
                SOURCE_ENV_RESOLUTION,
                face,
                0,
                &source_face,
            );

            let irradiance_face = bake_environment_cube_face(
                IRRADIANCE_ENV_RESOLUTION,
                face,
                0.0,
                CubeBakeMode::Irradiance,
                config,
            );
            write_cube_face(
                queue,
                &irradiance_texture,
                IRRADIANCE_ENV_RESOLUTION,
                face,
                0,
                &irradiance_face,
            );

            for mip in 0..prefiltered_mip_count {
                let mip_resolution = (PREFILTERED_ENV_RESOLUTION >> mip).max(1);
                let roughness = if prefiltered_mip_count <= 1 {
                    0.0
                } else {
                    mip as f32 / (prefiltered_mip_count - 1) as f32
                };
                let prefiltered_face = bake_environment_cube_face(
                    mip_resolution,
                    face,
                    roughness,
                    CubeBakeMode::Prefiltered,
                    config,
                );
                write_cube_face(
                    queue,
                    &prefiltered_texture,
                    mip_resolution,
                    face,
                    mip,
                    &prefiltered_face,
                );
            }
        }

        let brdf_lut = bake_brdf_lut(BRDF_LUT_RESOLUTION);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &brdf_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&brdf_lut),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(BRDF_LUT_RESOLUTION * 2 * std::mem::size_of::<f16>() as u32),
                rows_per_image: Some(BRDF_LUT_RESOLUTION),
            },
            wgpu::Extent3d {
                width: BRDF_LUT_RESOLUTION,
                height: BRDF_LUT_RESOLUTION,
                depth_or_array_layers: 1,
            },
        );

        let bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.sampler,
            &source_view,
            &irradiance_view,
            &prefiltered_view,
            &brdf_lut_view,
        );

        self.source_texture = source_texture;
        self.source_view = source_view;
        self.irradiance_texture = irradiance_texture;
        self.irradiance_view = irradiance_view;
        self.prefiltered_texture = prefiltered_texture;
        self.prefiltered_view = prefiltered_view;
        self.brdf_lut_texture = brdf_lut_texture;
        self.brdf_lut_view = brdf_lut_view;
        self.bind_group = bind_group;
        self.prefiltered_mip_count = prefiltered_mip_count;
    }
}

enum CubeBakeMode {
    Source,
    Irradiance,
    Prefiltered,
}

impl ViewportResources {
    pub fn rebuild_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &RenderConfig,
    ) {
        self.environment.rebuild(device, queue, config);
    }
}

fn write_cube_face(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    resolution: u32,
    face: u32,
    mip_level: u32,
    data: &[[f16; 4]],
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(data),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(resolution * 4 * std::mem::size_of::<f16>() as u32),
            rows_per_image: Some(resolution),
        },
        wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
    );
}

fn bake_environment_cube_face(
    resolution: u32,
    face: u32,
    roughness: f32,
    mode: CubeBakeMode,
    config: &RenderConfig,
) -> Vec<[f16; 4]> {
    let mut pixels = Vec::with_capacity((resolution * resolution) as usize);
    for y in 0..resolution {
        for x in 0..resolution {
            let direction = cubemap_direction(face, x, y, resolution);
            let color = match mode {
                CubeBakeMode::Source => sample_background_color(config, direction),
                CubeBakeMode::Irradiance => integrate_irradiance(config, direction),
                CubeBakeMode::Prefiltered => prefilter_environment(config, direction, roughness),
            };
            pixels.push([
                f16::from_f32(color.x),
                f16::from_f32(color.y),
                f16::from_f32(color.z),
                f16::from_f32(1.0),
            ]);
        }
    }
    pixels
}

fn cubemap_direction(face: u32, x: u32, y: u32, resolution: u32) -> Vec3 {
    let u = (2.0 * ((x as f32 + 0.5) / resolution as f32)) - 1.0;
    let v = (2.0 * ((y as f32 + 0.5) / resolution as f32)) - 1.0;

    match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

fn sample_background_color(config: &RenderConfig, direction: Vec3) -> Vec3 {
    match config.background_mode {
        BackgroundMode::SolidColor => Vec3::from_array(config.bg_solid_color),
        BackgroundMode::SkyGradient => {
            let t = (direction.y * 0.5 + 0.5).clamp(0.0, 1.0);
            Vec3::from_array(config.sky_horizon).lerp(Vec3::from_array(config.sky_zenith), t)
        }
    }
}

fn integrate_irradiance(config: &RenderConfig, normal: Vec3) -> Vec3 {
    let (tangent, bitangent) = tangent_frame(normal);
    let mut irradiance = Vec3::ZERO;
    let mut weight = 0.0;

    for sample_index in 0..IRRADIANCE_SAMPLE_COUNT {
        let xi = hammersley(sample_index, IRRADIANCE_SAMPLE_COUNT);
        let hemisphere = cosine_sample_hemisphere(xi);
        let sample_dir =
            (tangent * hemisphere.x + bitangent * hemisphere.y + normal * hemisphere.z).normalize();
        let no_l = normal.dot(sample_dir).max(0.0);
        if no_l <= 0.0 {
            continue;
        }
        irradiance += sample_background_color(config, sample_dir) * no_l;
        weight += no_l;
    }

    if weight > 0.0 {
        irradiance / weight
    } else {
        sample_background_color(config, normal)
    }
}

fn prefilter_environment(config: &RenderConfig, reflection_dir: Vec3, roughness: f32) -> Vec3 {
    let normal = reflection_dir.normalize();
    let view_dir = normal;
    let mut prefiltered = Vec3::ZERO;
    let mut weight = 0.0;

    for sample_index in 0..PREFILTER_SAMPLE_COUNT {
        let xi = hammersley(sample_index, PREFILTER_SAMPLE_COUNT);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        let light_dir = (2.0 * view_dir.dot(half_vector) * half_vector - view_dir).normalize();
        let no_l = normal.dot(light_dir).max(0.0);
        if no_l <= 0.0 {
            continue;
        }
        prefiltered += sample_background_color(config, light_dir) * no_l;
        weight += no_l;
    }

    if weight > 0.0 {
        prefiltered / weight
    } else {
        sample_background_color(config, reflection_dir)
    }
}

fn bake_brdf_lut(resolution: u32) -> Vec<[f16; 2]> {
    let mut data = Vec::with_capacity((resolution * resolution) as usize);
    for y in 0..resolution {
        let roughness = (y as f32 + 0.5) / resolution as f32;
        for x in 0..resolution {
            let no_v = (x as f32 + 0.5) / resolution as f32;
            let integrated = integrate_brdf(no_v, roughness);
            data.push([f16::from_f32(integrated.x), f16::from_f32(integrated.y)]);
        }
    }
    data
}

fn integrate_brdf(no_v: f32, roughness: f32) -> Vec2 {
    let view_dir = Vec3::new((1.0 - no_v * no_v).sqrt(), 0.0, no_v);
    let normal = Vec3::Z;
    let mut scale = 0.0;
    let mut bias = 0.0;

    for sample_index in 0..BRDF_SAMPLE_COUNT {
        let xi = hammersley(sample_index, BRDF_SAMPLE_COUNT);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        let light_dir = (2.0 * view_dir.dot(half_vector) * half_vector - view_dir).normalize();
        let no_l = light_dir.z.max(0.0);
        let no_h = half_vector.z.max(0.0);
        let vo_h = view_dir.dot(half_vector).max(0.0);
        if no_l <= 0.0 {
            continue;
        }

        let geometry = geometry_smith_ibl(no_v, no_l, roughness);
        let visibility = (geometry * vo_h) / (no_h * no_v).max(0.0001);
        let fresnel = (1.0 - vo_h).powi(5);
        scale += (1.0 - fresnel) * visibility;
        bias += fresnel * visibility;
    }

    Vec2::new(
        scale / BRDF_SAMPLE_COUNT as f32,
        bias / BRDF_SAMPLE_COUNT as f32,
    )
}

fn geometry_smith_ibl(no_v: f32, no_l: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx(no_v, roughness) * geometry_schlick_ggx(no_l, roughness)
}

fn geometry_schlick_ggx(no_x: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) * 0.5;
    no_x / (no_x * (1.0 - k) + k).max(0.0001)
}

fn importance_sample_ggx(xi: Vec2, normal: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let a2 = a * a;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    let half_vector_tangent = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let (tangent, bitangent) = tangent_frame(normal);
    (tangent * half_vector_tangent.x
        + bitangent * half_vector_tangent.y
        + normal * half_vector_tangent.z)
        .normalize()
}

fn cosine_sample_hemisphere(xi: Vec2) -> Vec3 {
    let r = xi.x.sqrt();
    let phi = 2.0 * PI * xi.y;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - xi.x).max(0.0).sqrt();
    Vec3::new(x, y, z)
}

fn tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.z.abs() < 0.999 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    (tangent, bitangent)
}

fn hammersley(index: u32, count: u32) -> Vec2 {
    Vec2::new(index as f32 / count as f32, radical_inverse_vdc(index))
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}
