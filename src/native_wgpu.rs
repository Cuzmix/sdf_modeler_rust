#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_instance_descriptor() -> wgpu::InstanceDescriptor {
    wgpu::InstanceDescriptor {
        backends: platform_backends(),
        ..Default::default()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn platform_backends() -> wgpu::Backends {
    #[cfg(target_os = "windows")]
    {
        wgpu::Backends::DX12
    }

    #[cfg(target_os = "android")]
    {
        wgpu::Backends::VULKAN
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        wgpu::Backends::METAL
    }

    #[cfg(not(any(
        target_os = "windows",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        wgpu::Backends::PRIMARY
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn native_device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };

    wgpu::DeviceDescriptor {
        label: Some("SDF Modeler device"),
        required_features: wgpu::Features::FLOAT32_FILTERABLE,
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 8192,
            max_storage_buffers_per_shader_stage: 4,
            max_storage_buffer_binding_size: 1 << 27,
            max_storage_textures_per_shader_stage: 4,
            ..base_limits
        },
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }
}
