use std::path::{Path, PathBuf};

use glam::Vec3;
use half::f16;

#[cfg(not(target_arch = "wasm32"))]
use crate::native_paths;
use crate::settings::{BackgroundMode, EnvironmentSource, RenderConfig};

use super::environment_bake::{EnvironmentBakeGpu, EnvironmentBakeInputs};
use super::ViewportResources;

pub(crate) const DEFAULT_PROCEDURAL_ENV_RESOLUTION: u32 = 512;
pub(crate) const MIN_IRRADIANCE_ENV_RESOLUTION: u32 = 16;
pub(crate) const MAX_IRRADIANCE_ENV_RESOLUTION: u32 = 64;
pub(crate) const BRDF_LUT_RESOLUTION: u32 = 256;

#[derive(Copy, Clone)]
pub(crate) struct EnvironmentBakeResolutionSet {
    pub source_resolution: u32,
    pub irradiance_resolution: u32,
    pub prefiltered_resolution: u32,
    pub brdf_lut_resolution: u32,
}

struct EnvironmentImageMap {
    width: u32,
    height: u32,
    pixels: Vec<Vec3>,
}

struct EnvironmentSourceTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

struct EnvironmentBindViews<'a> {
    source_view: &'a wgpu::TextureView,
    irradiance_view: &'a wgpu::TextureView,
    prefiltered_view: &'a wgpu::TextureView,
    brdf_lut_view: &'a wgpu::TextureView,
    background_view: &'a wgpu::TextureView,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum EnvironmentPipelineMode {
    FullFloatBaked,
    CompatibleFallback,
}

#[derive(Copy, Clone)]
struct EnvironmentTextureFormats {
    cube: wgpu::TextureFormat,
    brdf_lut: wgpu::TextureFormat,
}

impl EnvironmentPipelineMode {
    fn formats(self) -> EnvironmentTextureFormats {
        match self {
            Self::FullFloatBaked => EnvironmentTextureFormats {
                cube: wgpu::TextureFormat::Rgba16Float,
                brdf_lut: wgpu::TextureFormat::Rg16Float,
            },
            Self::CompatibleFallback => EnvironmentTextureFormats {
                cube: wgpu::TextureFormat::Rgba8Unorm,
                brdf_lut: wgpu::TextureFormat::Rg8Unorm,
            },
        }
    }

    fn supports_baked_environment(self) -> bool {
        matches!(self, Self::FullFloatBaked)
    }
}

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
    _fallback_background_texture: wgpu::Texture,
    fallback_background_view: wgpu::TextureView,
    pipeline_mode: EnvironmentPipelineMode,
    texture_formats: EnvironmentTextureFormats,
    bake_gpu: Option<EnvironmentBakeGpu>,
    cached_hdri_path: Option<String>,
    cached_hdri_texture: Option<EnvironmentSourceTexture>,
}

impl EnvironmentImageMap {
    #[cfg(not(target_arch = "wasm32"))]
    fn load(path: &Path) -> Result<Self, String> {
        let reader = image::ImageReader::open(path)
            .map_err(|error| format!("failed to open environment map: {error}"))?;
        let reader = reader
            .with_guessed_format()
            .map_err(|error| format!("failed to detect environment format: {error}"))?;
        let decoded = reader
            .decode()
            .map_err(|error| format!("failed to decode environment map: {error}"))?;
        let rgb = decoded.into_rgb32f();
        let (width, height) = rgb.dimensions();
        let pixels = rgb
            .pixels()
            .map(|pixel| Vec3::new(pixel.0[0], pixel.0[1], pixel.0[2]))
            .collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn load(path: &Path) -> Result<Self, String> {
        let _ = path;
        Err("HDR and EXR environment loading is unsupported on this platform".into())
    }

    fn rgba16f_pixels(&self) -> Vec<[f16; 4]> {
        self.pixels
            .iter()
            .map(|pixel| {
                [
                    f16::from_f32(pixel.x),
                    f16::from_f32(pixel.y),
                    f16::from_f32(pixel.z),
                    f16::from_f32(1.0),
                ]
            })
            .collect()
    }
}

impl EnvironmentResources {
    pub fn new(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Self {
        let pipeline_mode = Self::select_pipeline_mode(adapter);
        let texture_formats = pipeline_mode.formats();
        let cube_usage = if pipeline_mode.supports_baked_environment() {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        };
        let brdf_usage = if pipeline_mode.supports_baked_environment() {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind_group_layout = Self::create_bind_group_layout(device);
        let bake_gpu = pipeline_mode.supports_baked_environment().then(|| {
            EnvironmentBakeGpu::new(device, texture_formats.cube, texture_formats.brdf_lut)
        });
        let (source_texture, source_view) =
            Self::create_cube_texture(device, 1, 1, "Env Source", texture_formats.cube, cube_usage);
        let (irradiance_texture, irradiance_view) = Self::create_cube_texture(
            device,
            1,
            1,
            "Env Irradiance",
            texture_formats.cube,
            cube_usage,
        );
        let (prefiltered_texture, prefiltered_view) = Self::create_cube_texture(
            device,
            1,
            1,
            "Env Prefiltered",
            texture_formats.cube,
            cube_usage,
        );
        let (brdf_lut_texture, brdf_lut_view) = Self::create_brdf_lut_texture(
            device,
            1,
            "Env BRDF LUT",
            texture_formats.brdf_lut,
            brdf_usage,
        );
        let (fallback_background_texture, fallback_background_view) = Self::create_2d_texture(
            device,
            1,
            1,
            "Env Background Fallback",
            texture_formats.cube,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &sampler,
            EnvironmentBindViews {
                source_view: &source_view,
                irradiance_view: &irradiance_view,
                prefiltered_view: &prefiltered_view,
                brdf_lut_view: &brdf_lut_view,
                background_view: &fallback_background_view,
            },
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
            _fallback_background_texture: fallback_background_texture,
            fallback_background_view,
            pipeline_mode,
            texture_formats,
            bake_gpu,
            cached_hdri_path: None,
            cached_hdri_texture: None,
        }
    }

    fn select_pipeline_mode(adapter: &wgpu::Adapter) -> EnvironmentPipelineMode {
        if cfg!(target_os = "android") {
            log::warn!(
                "Using Android-compatible environment fallback; skipping baked float IBL pipelines"
            );
            return EnvironmentPipelineMode::CompatibleFallback;
        }

        let required_usages =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let cube_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let brdf_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rg16Float);
        let supports_float_bake =
            Self::supports_baked_environment_format(cube_features, required_usages)
                && Self::supports_baked_environment_format(brdf_features, required_usages);

        if supports_float_bake {
            EnvironmentPipelineMode::FullFloatBaked
        } else {
            let adapter_info = adapter.get_info();
            log::warn!(
                "Adapter '{}' lacks baked float environment support; using compatibility fallback",
                adapter_info.name
            );
            EnvironmentPipelineMode::CompatibleFallback
        }
    }

    fn supports_baked_environment_format(
        features: wgpu::TextureFormatFeatures,
        required_usages: wgpu::TextureUsages,
    ) -> bool {
        features.allowed_usages.contains(required_usages)
            && features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
    }

    pub fn uses_compatibility_fallback(&self) -> bool {
        matches!(
            self.pipeline_mode,
            EnvironmentPipelineMode::CompatibleFallback
        )
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
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
            format,
            usage,
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
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
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
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_2d_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        views: EnvironmentBindViews<'_>,
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
                    resource: wgpu::BindingResource::TextureView(views.source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(views.irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(views.prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(views.brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(views.background_view),
                },
            ],
        })
    }

    fn create_hdri_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        map: &EnvironmentImageMap,
    ) -> EnvironmentSourceTexture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDRI Source Texture"),
            size: wgpu::Extent3d {
                width: map.width.max(1),
                height: map.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pixels = map.rgba16f_pixels();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(map.width * 4 * std::mem::size_of::<f16>() as u32),
                rows_per_image: Some(map.height),
            },
            wgpu::Extent3d {
                width: map.width,
                height: map.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        EnvironmentSourceTexture {
            _texture: texture,
            view,
            width: map.width.max(1),
            height: map.height.max(1),
        }
    }

    fn fallback_environment_color(config: &RenderConfig) -> [f32; 4] {
        let base_color = match config.background_mode {
            BackgroundMode::SolidColor => Vec3::from_array(config.bg_solid_color),
            BackgroundMode::SkyGradient => {
                let horizon = Vec3::from_array(config.sky_horizon);
                let zenith = Vec3::from_array(config.sky_zenith);
                horizon.lerp(zenith, 0.35)
            }
        };
        let exposed = base_color * config.environment_exposure.exp2();
        [
            exposed.x.clamp(0.0, 1.0),
            exposed.y.clamp(0.0, 1.0),
            exposed.z.clamp(0.0, 1.0),
            1.0,
        ]
    }

    fn write_rgba_texture(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        color: [f32; 4],
    ) {
        match format {
            wgpu::TextureFormat::Rgba16Float => {
                let pixel = [
                    f16::from_f32(color[0]),
                    f16::from_f32(color[1]),
                    f16::from_f32(color[2]),
                    f16::from_f32(color[3]),
                ];
                let pixels = vec![pixel; (width * height * depth_or_array_layers) as usize];
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&pixels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4 * std::mem::size_of::<f16>() as u32),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers,
                    },
                );
            }
            wgpu::TextureFormat::Rgba8Unorm => {
                let pixel = [
                    (color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (color[3] * 255.0).round().clamp(0.0, 255.0) as u8,
                ];
                let pixels = vec![pixel; (width * height * depth_or_array_layers) as usize];
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&pixels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers,
                    },
                );
            }
            _ => unreachable!("unsupported RGBA environment format"),
        }
    }

    fn write_rg_texture(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        color: [f32; 2],
    ) {
        match format {
            wgpu::TextureFormat::Rg16Float => {
                let pixel = [f16::from_f32(color[0]), f16::from_f32(color[1])];
                let pixels = vec![pixel; (width * height) as usize];
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&pixels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 2 * std::mem::size_of::<f16>() as u32),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            wgpu::TextureFormat::Rg8Unorm => {
                let pixel = [
                    (color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                ];
                let pixels = vec![pixel; (width * height) as usize];
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&pixels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 2),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            _ => unreachable!("unsupported RG environment format"),
        }
    }

    fn clear_hdri_cache(&mut self) {
        self.cached_hdri_path = None;
        self.cached_hdri_texture = None;
    }

    fn resolve_hdri_path(path: &str) -> Option<PathBuf> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            native_paths::resolve_bundled_asset_path(path)
        }

        #[cfg(target_arch = "wasm32")]
        {
            if path.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(path))
            }
        }
    }

    fn runtime_environment_config(&self, config: &RenderConfig) -> RenderConfig {
        let mut runtime_config = config.clone();
        if runtime_config.environment_source == EnvironmentSource::Hdri
            && self.cached_hdri_texture.is_none()
        {
            runtime_config.environment_source = EnvironmentSource::ProceduralSky;
        }
        runtime_config
    }

    fn ensure_hdri_texture_loaded(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &RenderConfig,
    ) {
        if config.environment_source != EnvironmentSource::Hdri {
            self.clear_hdri_cache();
            return;
        }

        let path = config
            .hdri_path
            .as_deref()
            .map(str::trim)
            .unwrap_or_default();
        if path.is_empty() {
            self.clear_hdri_cache();
            return;
        }

        let should_reload = self.cached_hdri_path.as_deref() != Some(path);
        if should_reload {
            match Self::resolve_hdri_path(path) {
                Some(resolved_path) => match EnvironmentImageMap::load(&resolved_path) {
                    Ok(map) => {
                        self.cached_hdri_texture =
                            Some(Self::create_hdri_texture(device, queue, &map));
                        self.cached_hdri_path = Some(path.to_string());
                    }
                    Err(error) => {
                        log::warn!(
                            "Failed to load environment map '{}': {}. Falling back to procedural sky.",
                            path,
                            error
                        );
                        self.cached_hdri_path = Some(path.to_string());
                        self.cached_hdri_texture = None;
                    }
                },
                None => {
                    log::warn!(
                        "Failed to resolve environment map '{}'. Falling back to procedural sky.",
                        path
                    );
                    self.cached_hdri_path = Some(path.to_string());
                    self.cached_hdri_texture = None;
                }
            }
        }
    }

    pub fn has_hdri_background_texture(&self) -> bool {
        self.cached_hdri_texture.is_some()
    }

    fn resolved_bake_resolutions(
        &self,
        config: &RenderConfig,
        device_limits: &wgpu::Limits,
    ) -> EnvironmentBakeResolutionSet {
        let hdri_dimensions = self
            .cached_hdri_texture
            .as_ref()
            .map(|texture| (texture.width, texture.height));
        resolve_environment_bake_resolutions(
            config.environment_bake_resolution,
            hdri_dimensions,
            device_limits.max_texture_dimension_2d,
        )
    }

    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, config: &RenderConfig) {
        if !self.pipeline_mode.supports_baked_environment() {
            self.clear_hdri_cache();
            self.rebuild_compatible_fallback(device, queue, config);
            return;
        }

        self.ensure_hdri_texture_loaded(device, queue, config);
        let runtime_config = self.runtime_environment_config(config);
        let bake_resolutions = self.resolved_bake_resolutions(&runtime_config, &device.limits());
        let prefiltered_mip_count = bake_resolutions.prefiltered_resolution.ilog2() + 1;
        let (source_texture, source_view) = Self::create_cube_texture(
            device,
            bake_resolutions.source_resolution,
            1,
            "Env Source",
            self.texture_formats.cube,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (irradiance_texture, irradiance_view) = Self::create_cube_texture(
            device,
            bake_resolutions.irradiance_resolution,
            1,
            "Env Irradiance",
            self.texture_formats.cube,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (prefiltered_texture, prefiltered_view) = Self::create_cube_texture(
            device,
            bake_resolutions.prefiltered_resolution,
            prefiltered_mip_count,
            "Env Prefiltered",
            self.texture_formats.cube,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let (brdf_lut_texture, brdf_lut_view) = Self::create_brdf_lut_texture(
            device,
            bake_resolutions.brdf_lut_resolution,
            "Env BRDF LUT",
            self.texture_formats.brdf_lut,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let hdri_view = self
            .cached_hdri_texture
            .as_ref()
            .map(|texture| &texture.view);
        let background_view = hdri_view.unwrap_or(&self.fallback_background_view);

        self.bake_gpu
            .as_mut()
            .expect("baked environment pipeline should exist")
            .bake(
                device,
                queue,
                &runtime_config,
                EnvironmentBakeInputs {
                    resolutions: bake_resolutions,
                    source_texture: &source_texture,
                    irradiance_texture: &irradiance_texture,
                    prefiltered_texture: &prefiltered_texture,
                    brdf_lut_texture: &brdf_lut_texture,
                    prefiltered_mip_count,
                    hdri_view,
                },
            );

        let bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.sampler,
            EnvironmentBindViews {
                source_view: &source_view,
                irradiance_view: &irradiance_view,
                prefiltered_view: &prefiltered_view,
                brdf_lut_view: &brdf_lut_view,
                background_view,
            },
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

    fn rebuild_compatible_fallback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &RenderConfig,
    ) {
        let texture_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let fallback_color = Self::fallback_environment_color(config);
        let (source_texture, source_view) = Self::create_cube_texture(
            device,
            1,
            1,
            "Env Source",
            self.texture_formats.cube,
            texture_usage,
        );
        let (irradiance_texture, irradiance_view) = Self::create_cube_texture(
            device,
            1,
            1,
            "Env Irradiance",
            self.texture_formats.cube,
            texture_usage,
        );
        let (prefiltered_texture, prefiltered_view) = Self::create_cube_texture(
            device,
            1,
            1,
            "Env Prefiltered",
            self.texture_formats.cube,
            texture_usage,
        );
        let (brdf_lut_texture, brdf_lut_view) = Self::create_brdf_lut_texture(
            device,
            1,
            "Env BRDF LUT",
            self.texture_formats.brdf_lut,
            texture_usage,
        );
        let (fallback_background_texture, fallback_background_view) = Self::create_2d_texture(
            device,
            1,
            1,
            "Env Background Fallback",
            self.texture_formats.cube,
            texture_usage,
        );

        Self::write_rgba_texture(
            queue,
            &source_texture,
            self.texture_formats.cube,
            1,
            1,
            6,
            fallback_color,
        );
        Self::write_rgba_texture(
            queue,
            &irradiance_texture,
            self.texture_formats.cube,
            1,
            1,
            6,
            fallback_color,
        );
        Self::write_rgba_texture(
            queue,
            &prefiltered_texture,
            self.texture_formats.cube,
            1,
            1,
            6,
            fallback_color,
        );
        Self::write_rgba_texture(
            queue,
            &fallback_background_texture,
            self.texture_formats.cube,
            1,
            1,
            1,
            fallback_color,
        );
        Self::write_rg_texture(
            queue,
            &brdf_lut_texture,
            self.texture_formats.brdf_lut,
            1,
            1,
            [0.0, 0.0],
        );

        let bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.sampler,
            EnvironmentBindViews {
                source_view: &source_view,
                irradiance_view: &irradiance_view,
                prefiltered_view: &prefiltered_view,
                brdf_lut_view: &brdf_lut_view,
                background_view: &fallback_background_view,
            },
        );

        self.source_texture = source_texture;
        self.source_view = source_view;
        self.irradiance_texture = irradiance_texture;
        self.irradiance_view = irradiance_view;
        self.prefiltered_texture = prefiltered_texture;
        self.prefiltered_view = prefiltered_view;
        self.brdf_lut_texture = brdf_lut_texture;
        self.brdf_lut_view = brdf_lut_view;
        self._fallback_background_texture = fallback_background_texture;
        self.fallback_background_view = fallback_background_view;
        self.bind_group = bind_group;
        self.prefiltered_mip_count = 1;
    }
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

fn resolve_environment_bake_resolutions(
    requested_source_resolution: u32,
    hdri_dimensions: Option<(u32, u32)>,
    max_texture_dimension: u32,
) -> EnvironmentBakeResolutionSet {
    let source_resolution = resolve_source_environment_resolution(
        requested_source_resolution,
        hdri_dimensions,
        max_texture_dimension,
    );
    let irradiance_resolution = (source_resolution / 16)
        .clamp(MIN_IRRADIANCE_ENV_RESOLUTION, MAX_IRRADIANCE_ENV_RESOLUTION);

    EnvironmentBakeResolutionSet {
        source_resolution,
        irradiance_resolution,
        prefiltered_resolution: source_resolution,
        brdf_lut_resolution: BRDF_LUT_RESOLUTION.min(max_texture_dimension.max(1)),
    }
}

fn resolve_source_environment_resolution(
    requested_source_resolution: u32,
    hdri_dimensions: Option<(u32, u32)>,
    max_texture_dimension: u32,
) -> u32 {
    let base_resolution = if requested_source_resolution > 0 {
        requested_source_resolution
    } else if let Some((width, height)) = hdri_dimensions {
        cube_face_resolution_from_equirect(width, height)
    } else {
        DEFAULT_PROCEDURAL_ENV_RESOLUTION
    };

    base_resolution.clamp(1, max_texture_dimension.max(1))
}

pub(crate) fn cube_face_resolution_from_equirect(width: u32, height: u32) -> u32 {
    width.div_ceil(4).max(height.div_ceil(2)).max(1)
}

#[cfg(test)]
mod tests {
    use super::{
        cube_face_resolution_from_equirect, resolve_environment_bake_resolutions,
        resolve_source_environment_resolution, EnvironmentImageMap,
    };

    #[test]
    fn rgba16f_pixels_keep_rgb_and_write_opaque_alpha() {
        let map = EnvironmentImageMap {
            width: 1,
            height: 2,
            pixels: vec![
                glam::Vec3::new(1.0, 0.5, 0.25),
                glam::Vec3::new(0.1, 0.2, 0.3),
            ],
        };

        let pixels = map.rgba16f_pixels();
        assert_eq!(pixels.len(), 2);
        assert!((pixels[0][0].to_f32() - 1.0).abs() < 0.0001);
        assert!((pixels[0][1].to_f32() - 0.5).abs() < 0.0001);
        assert!((pixels[0][2].to_f32() - 0.25).abs() < 0.0001);
        assert!((pixels[0][3].to_f32() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn cube_face_resolution_matches_equirect_layout() {
        assert_eq!(cube_face_resolution_from_equirect(4096, 2048), 1024);
        assert_eq!(cube_face_resolution_from_equirect(8192, 4096), 2048);
        assert_eq!(cube_face_resolution_from_equirect(1536, 1024), 512);
    }

    #[test]
    fn source_environment_resolution_prefers_explicit_override() {
        let resolution = resolve_source_environment_resolution(256, Some((4096, 2048)), 8192);
        assert_eq!(resolution, 256);
    }

    #[test]
    fn source_environment_resolution_uses_hdri_face_equivalent_in_auto_mode() {
        let resolution = resolve_source_environment_resolution(0, Some((4096, 2048)), 8192);
        assert_eq!(resolution, 1024);
    }

    #[test]
    fn source_environment_resolution_clamps_to_device_limit() {
        let resolution = resolve_source_environment_resolution(0, Some((16384, 8192)), 2048);
        assert_eq!(resolution, 2048);
    }

    #[test]
    fn derived_environment_bake_resolutions_scale_with_source_face_size() {
        let resolutions = resolve_environment_bake_resolutions(0, Some((4096, 2048)), 8192);
        assert_eq!(resolutions.source_resolution, 1024);
        assert_eq!(resolutions.prefiltered_resolution, 1024);
        assert_eq!(resolutions.irradiance_resolution, 64);
        assert_eq!(resolutions.brdf_lut_resolution, 256);
    }
}
