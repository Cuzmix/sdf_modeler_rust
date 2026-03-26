use std::collections::{HashMap, HashSet};
use std::mem;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use crate::app::reference_images::{RefPlane, ReferenceImageEntry, ReferenceImageStore};
use crate::gpu::camera::Camera;

const REFERENCE_OVERLAY_TESSELLATION: usize = 20;
const INITIAL_OVERLAY_VERTEX_CAPACITY: usize = 256;
const INITIAL_OVERLAY_INDEX_CAPACITY: usize = 384;
const REFERENCE_OVERLAY_SHADER_SRC: &str = include_str!("../shaders/reference_overlay.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OverlayVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub struct ReferenceOverlayRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    index_buffer: wgpu::Buffer,
    index_capacity: usize,
    textures: HashMap<String, ReferenceOverlayTexture>,
    failed_paths: HashSet<String>,
}

pub struct ReferenceOverlayPass<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target_view: &'a wgpu::TextureView,
    pub camera: &'a Camera,
    pub viewport_size: (u32, u32),
    pub store: &'a ReferenceImageStore,
}

struct ReferenceOverlayTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

struct PreparedReferenceOverlay {
    path: String,
    vertices: Vec<OverlayVertex>,
    indices: Vec<u32>,
}

impl ReferenceOverlayRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reference Overlay BGL"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Reference Overlay Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reference Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(REFERENCE_OVERLAY_SHADER_SRC.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reference Overlay Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Reference Overlay Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<OverlayVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reference Overlay Vertices"),
            size: (INITIAL_OVERLAY_VERTEX_CAPACITY * mem::size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reference Overlay Indices"),
            size: (INITIAL_OVERLAY_INDEX_CAPACITY * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            vertex_capacity: INITIAL_OVERLAY_VERTEX_CAPACITY,
            index_buffer,
            index_capacity: INITIAL_OVERLAY_INDEX_CAPACITY,
            textures: HashMap::new(),
            failed_paths: HashSet::new(),
        }
    }

    pub fn render(&mut self, pass: ReferenceOverlayPass<'_>) {
        let overlays = prepare_reference_overlays(pass.camera, pass.viewport_size, pass.store);
        for overlay in overlays {
            if overlay.vertices.is_empty() || overlay.indices.is_empty() {
                continue;
            }
            if !self.ensure_texture(pass.device, pass.queue, &overlay.path) {
                continue;
            }
            self.ensure_overlay_buffer_capacity(
                pass.device,
                overlay.vertices.len(),
                overlay.indices.len(),
            );
            pass.queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&overlay.vertices),
            );
            pass.queue.write_buffer(
                &self.index_buffer,
                0,
                bytemuck::cast_slice(&overlay.indices),
            );

            let Some(texture) = self.textures.get(&overlay.path) else {
                continue;
            };
            let mut render_pass = pass.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Reference Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pass.target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &texture.bind_group, &[]);
            render_pass.set_vertex_buffer(
                0,
                self.vertex_buffer
                    .slice(..(overlay.vertices.len() * mem::size_of::<OverlayVertex>()) as u64),
            );
            render_pass.set_index_buffer(
                self.index_buffer
                    .slice(..(overlay.indices.len() * mem::size_of::<u32>()) as u64),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..overlay.indices.len() as u32, 0, 0..1);
        }
    }

    fn ensure_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> bool {
        if self.textures.contains_key(path) {
            return true;
        }
        if self.failed_paths.contains(path) {
            return false;
        }
        let loaded = match load_reference_texture(path) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.failed_paths.insert(path.to_string());
                log::error!("{error}");
                return false;
            }
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Reference Overlay Texture"),
            size: wgpu::Extent3d {
                width: loaded.width,
                height: loaded.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &loaded.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(loaded.width * 4),
                rows_per_image: Some(loaded.height),
            },
            wgpu::Extent3d {
                width: loaded.width,
                height: loaded.height,
                depth_or_array_layers: 1,
            },
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reference Overlay BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });
        self.textures.insert(
            path.to_string(),
            ReferenceOverlayTexture {
                _texture: texture,
                bind_group,
            },
        );
        true
    }

    fn ensure_overlay_buffer_capacity(
        &mut self,
        device: &wgpu::Device,
        vertex_count: usize,
        index_count: usize,
    ) {
        if vertex_count > self.vertex_capacity {
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reference Overlay Vertices"),
                size: (vertex_count * mem::size_of::<OverlayVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = vertex_count;
        }
        if index_count > self.index_capacity {
            self.index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reference Overlay Indices"),
                size: (index_count * mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = index_count;
        }
    }
}

struct LoadedReferenceTexture {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

fn load_reference_texture(path: &str) -> Result<LoadedReferenceTexture, String> {
    let image_path = Path::new(path);
    let dynamic_image = image::open(image_path).map_err(|error| {
        format!(
            "Failed to open reference image '{}': {error}",
            image_path.display()
        )
    })?;
    let rgba = dynamic_image.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(format!(
            "Reference image '{}' has invalid dimensions",
            image_path.display()
        ));
    }
    Ok(LoadedReferenceTexture {
        width,
        height,
        rgba8: rgba.into_raw(),
    })
}

fn prepare_reference_overlays(
    camera: &Camera,
    viewport_size: (u32, u32),
    store: &ReferenceImageStore,
) -> Vec<PreparedReferenceOverlay> {
    let aspect = viewport_size.0.max(1) as f32 / viewport_size.1.max(1) as f32;
    let view_proj = camera.projection_matrix(aspect.max(1e-5)) * camera.view_matrix();
    store
        .images
        .iter()
        .filter(|image| image.visible)
        .map(|image| {
            let (vertices, indices) = build_reference_mesh(image, &view_proj);
            PreparedReferenceOverlay {
                path: image.path.clone(),
                vertices,
                indices,
            }
        })
        .collect()
}

fn build_reference_mesh(
    image: &ReferenceImageEntry,
    view_proj: &Mat4,
) -> (Vec<OverlayVertex>, Vec<u32>) {
    let corners = quad_world_corners(image);
    let tint = [1.0, 1.0, 1.0, image.opacity.clamp(0.0, 1.0)];
    let mut vertices =
        Vec::with_capacity(REFERENCE_OVERLAY_TESSELLATION * REFERENCE_OVERLAY_TESSELLATION * 4);
    let mut indices =
        Vec::with_capacity(REFERENCE_OVERLAY_TESSELLATION * REFERENCE_OVERLAY_TESSELLATION * 6);
    let inv_tess = 1.0 / REFERENCE_OVERLAY_TESSELLATION as f32;

    for y in 0..REFERENCE_OVERLAY_TESSELLATION {
        let v0 = y as f32 * inv_tess;
        let v1 = (y + 1) as f32 * inv_tess;
        for x in 0..REFERENCE_OVERLAY_TESSELLATION {
            let u0 = x as f32 * inv_tess;
            let u1 = (x + 1) as f32 * inv_tess;

            let p00 = bilerp_quad(corners, u0, v0);
            let p10 = bilerp_quad(corners, u1, v0);
            let p11 = bilerp_quad(corners, u1, v1);
            let p01 = bilerp_quad(corners, u0, v1);

            let (Some(s00), Some(s10), Some(s11), Some(s01)) = (
                world_to_ndc(p00, view_proj),
                world_to_ndc(p10, view_proj),
                world_to_ndc(p11, view_proj),
                world_to_ndc(p01, view_proj),
            ) else {
                continue;
            };

            let base = vertices.len() as u32;
            vertices.push(OverlayVertex {
                position: s00,
                uv: [u0, v0],
                color: tint,
            });
            vertices.push(OverlayVertex {
                position: s10,
                uv: [u1, v0],
                color: tint,
            });
            vertices.push(OverlayVertex {
                position: s11,
                uv: [u1, v1],
                color: tint,
            });
            vertices.push(OverlayVertex {
                position: s01,
                uv: [u0, v1],
                color: tint,
            });
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    (vertices, indices)
}

fn quad_world_corners(image: &ReferenceImageEntry) -> [Vec3; 4] {
    let aspect = if image.size_px[1] > 0.0 {
        image.size_px[0] / image.size_px[1]
    } else {
        1.0
    };
    let half_h = image.scale * 0.5;
    let half_w = half_h * aspect.max(1e-4);
    let center = image.offset;

    match image.plane {
        RefPlane::Front | RefPlane::Back => [
            Vec3::new(center.x - half_w, center.y + half_h, center.z),
            Vec3::new(center.x + half_w, center.y + half_h, center.z),
            Vec3::new(center.x + half_w, center.y - half_h, center.z),
            Vec3::new(center.x - half_w, center.y - half_h, center.z),
        ],
        RefPlane::Left | RefPlane::Right => [
            Vec3::new(center.x, center.y + half_h, center.z + half_w),
            Vec3::new(center.x, center.y + half_h, center.z - half_w),
            Vec3::new(center.x, center.y - half_h, center.z - half_w),
            Vec3::new(center.x, center.y - half_h, center.z + half_w),
        ],
        RefPlane::Top | RefPlane::Bottom => [
            Vec3::new(center.x - half_w, center.y, center.z - half_h),
            Vec3::new(center.x + half_w, center.y, center.z - half_h),
            Vec3::new(center.x + half_w, center.y, center.z + half_h),
            Vec3::new(center.x - half_w, center.y, center.z + half_h),
        ],
    }
}

fn bilerp_quad(corners: [Vec3; 4], u: f32, v: f32) -> Vec3 {
    let top = corners[0].lerp(corners[1], u);
    let bottom = corners[3].lerp(corners[2], u);
    top.lerp(bottom, v)
}

fn world_to_ndc(world_pos: Vec3, view_proj: &Mat4) -> Option<[f32; 2]> {
    let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some([ndc.x, -ndc.y])
}
