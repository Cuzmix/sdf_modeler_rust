use crate::gpu::codegen;
use crate::graph::scene::Scene;
use crate::settings::RenderConfig;
use crate::viewport::ViewportResources;

/// Opaque GPU context for Flutter — owns the wgpu device, queue,
/// and all viewport rendering resources (pipelines, buffers, textures).
#[cfg(feature = "flutter_ui")]
#[flutter_rust_bridge::frb(opaque)]
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) resources: ViewportResources,
}

#[cfg(feature = "flutter_ui")]
impl GpuContext {
    /// Create a headless wgpu device + ViewportResources with a default scene.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("No suitable GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SDF Modeler device (Flutter)"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    max_storage_buffers_per_shader_stage: 4,
                    max_storage_buffer_binding_size: 1 << 27, // 128MB
                    max_storage_textures_per_shader_stage: 4,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        // Generate initial shaders from a default scene
        let scene = Scene::new();
        let config = RenderConfig::default();
        let shader_src = codegen::generate_shader(&scene, &config);
        let pick_shader_src = codegen::generate_pick_shader(&scene, &config);

        // RGBA8 for Flutter's decodeImageFromPixels (PixelFormat.rgba8888)
        let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let resources = ViewportResources::new(&device, target_format, &shader_src, &pick_shader_src);

        Ok(Self { device, queue, resources })
    }
}
