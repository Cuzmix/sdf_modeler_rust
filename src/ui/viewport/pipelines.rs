use std::num::NonZeroU64;

use eframe::wgpu;

use super::{ViewportResources, BLIT_SHADER_SRC};

impl ViewportResources {
    pub(crate) fn create_render_pipeline(
        device: &wgpu::Device,
        shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        voxel_tex_bgl: &wgpu::BindGroupLayout,
        environment_bgl: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, voxel_tex_bgl, environment_bgl],
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

    pub(crate) fn create_pick_pipeline(
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

    /// Bind group layout for the tape buffer used by the universal fallback
    /// render and pick pipelines. Separate from `scene_bgl` so the unrolled
    /// pipelines don't have to declare an unused binding — adding the tape
    /// to the shared layout would force every existing pipeline to be
    /// rebuilt against a layout it doesn't read from.
    pub(super) fn create_tape_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tape BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    /// Build the universal fallback render pipeline. Same fragment shader
    /// shape as `create_render_pipeline` but its `scene_sdf` walks the
    /// tape buffer at @group(2) instead of inlining scene topology.
    ///
    /// The fallback layout drops `voxel_tex_bgl` to stay inside wgpu's
    /// default `max_bind_groups = 4` limit. Voxel textures are only
    /// needed for sculpt nodes, and the tape encoder rejects sculpt
    /// scenes (F2 contract — see `gpu/tape.rs`), so the fallback never
    /// has anything to sample from the texture array. Compiled once at
    /// startup and reused for every F2-compatible scene.
    pub(crate) fn create_fallback_render_pipeline(
        device: &wgpu::Device,
        shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        tape_bgl: &wgpu::BindGroupLayout,
        environment_bgl: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fallback Render Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fallback Render Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, tape_bgl, environment_bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fallback Render Pipeline"),
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

    /// Build the universal fallback pick pipeline. Same compute shape as
    /// `create_pick_pipeline` but reads the tape buffer.
    pub(crate) fn create_fallback_pick_pipeline(
        device: &wgpu::Device,
        pick_shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        pick_bgl: &wgpu::BindGroupLayout,
        tape_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fallback Pick Shader"),
            source: wgpu::ShaderSource::Wgsl(pick_shader_src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fallback Pick Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl, pick_bgl, tape_bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fallback Pick Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "cs_pick",
            compilation_options: Default::default(),
            cache: None,
        })
    }

    pub(super) fn create_scene_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

    pub(super) fn create_brush_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brush BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            NonZeroU64::new(Self::brush_param_size())
                                .expect("BrushGpuParams size must be non-zero"),
                        ),
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

    pub(super) fn create_brush_pipeline(
        device: &wgpu::Device,
        brush_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brush Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::gpu::shader_templates::BRUSH_COMPUTE_SHADER.into(),
            ),
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

    pub(super) fn create_blit_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

    pub(super) fn create_blit_pipeline(
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
}
