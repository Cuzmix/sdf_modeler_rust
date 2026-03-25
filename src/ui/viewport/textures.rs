use super::ViewportResources;

impl ViewportResources {
    /// Create the bind group layout for voxel textures: binding 0 = sampler, then N texture_3d bindings.
    pub(super) fn create_voxel_tex_bgl(
        device: &wgpu::Device,
        tex_count: usize,
    ) -> wgpu::BindGroupLayout {
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
    pub(super) fn create_voxel_tex_bind_group(
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

    /// Rebuild voxel texture3D resources for the given number of sculpt nodes.
    /// Creates 1x1x1 placeholder textures Ã¢â‚¬â€ actual data uploaded later.
    pub(super) fn rebuild_voxel_textures(&mut self, device: &wgpu::Device, count: usize) {
        let mut textures = Vec::with_capacity(count);
        let mut views = Vec::with_capacity(count);
        for i in 0..count {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Voxel Tex {i}")),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
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
            device,
            &self.voxel_tex_bgl,
            &self.voxel_sampler,
            &self.voxel_texture_views,
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
            self.voxel_texture_views[tex_idx] =
                tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.voxel_textures[tex_idx] = tex;
            // Rebuild bind group with updated view
            self.voxel_tex_bind_group = Self::create_voxel_tex_bind_group(
                device,
                &self.voxel_tex_bgl,
                &self.voxel_sampler,
                &self.voxel_texture_views,
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
}
