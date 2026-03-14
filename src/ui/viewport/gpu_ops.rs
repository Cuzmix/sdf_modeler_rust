use eframe::wgpu;

use crate::gpu::buffers::SdfNodeGpu;
use crate::gpu::camera::CameraUniform;
use crate::gpu::picking::{PendingPick, PickResult};
use crate::settings::RenderConfig;

use super::{BrushDispatch, ViewportResources, INITIAL_VOXEL_CAPACITY};

impl ViewportResources {
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

    /// Cancel any in-flight pick mapping. Safe to call even if nothing is mapped.
    pub fn cancel_pending_pick(&self) {
        self.pick_staging_buffer.unmap();
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
        queue.write_buffer(
            &self.pick_input_buffer,
            0,
            bytemuck::cast_slice(&pick_input),
        );

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

        encoder.copy_buffer_to_buffer(
            &self.pick_output_buffer,
            0,
            &self.pick_staging_buffer,
            0,
            32,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Request async map ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â caller polls with device.poll(Poll)
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
            Ok(Err(_)) => Some(None),                          // Map failed
            Err(std::sync::mpsc::TryRecvError::Empty) => None, // Not ready yet
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(None),
        }
    }

    fn read_pick_from_receiver(
        staging: &wgpu::Buffer,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Option<PickResult> {
        rx.recv().ok()?.ok()?;
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

    /// Dispatch brush compute shaders in one submit to reduce CPU overhead.
    /// Returns true when a submit was issued.
    pub fn dispatch_brush_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dispatches: &[BrushDispatch],
    ) -> bool {
        if dispatches.is_empty() {
            return false;
        }

        self.ensure_brush_uniform_capacity(device, dispatches.len() as u32);

        for (index, dispatch) in dispatches.iter().enumerate() {
            let offset = self.brush_uniform_stride * index as u64;
            queue.write_buffer(
                &self.brush_uniform_buffer,
                offset,
                bytemuck::bytes_of(&dispatch.params),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Brush Batch Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Brush Batch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brush_pipeline);

            for (index, dispatch) in dispatches.iter().enumerate() {
                let dynamic_offset = (self.brush_uniform_stride * index as u64) as u32;
                pass.set_bind_group(0, &self.brush_bind_group, &[dynamic_offset]);
                pass.dispatch_workgroups(
                    dispatch.workgroups[0],
                    dispatch.workgroups[1],
                    dispatch.workgroups[2],
                );
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        true
    }

    /// Render the viewport to an offscreen texture and return RGBA pixel data.
    pub fn screenshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform: &CameraUniform,
        render_config: &RenderConfig,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // Write camera uniform
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniform));

        // Render to an intermediate scene texture so the shared blit pipeline can
        // apply outline and bloom before readback.
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Screenshot Scene Texture"),
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
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Screenshot Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Row alignment for buffer copy
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let buffer_size = (padded_row * height) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blit_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Blit Params"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blit_data: [f32; 12] = [
            0.0,
            0.0,
            width as f32,
            height as f32,
            render_config.outline_color[0],
            render_config.outline_color[1],
            render_config.outline_color[2],
            render_config.outline_thickness,
            render_config.bloom_threshold,
            render_config.bloom_intensity,
            render_config.bloom_radius,
            if render_config.bloom_enabled {
                1.0
            } else {
                0.0
            },
        ];
        queue.write_buffer(&blit_params_buffer, 0, bytemuck::cast_slice(&blit_data));

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screenshot Blit BG"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blit_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        // Scene render pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screenshot Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scene_view,
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
            if self.use_composite {
                if let Some(ref composite) = self.composite {
                    pass.set_pipeline(&composite.render_pipeline);
                    pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    pass.set_bind_group(1, &self.scene_bind_group, &[]);
                    pass.set_bind_group(2, &composite.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.scene_bind_group, &[]);
                pass.set_bind_group(2, &self.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        // Shared post-process blit pass.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Screenshot Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
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
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy texture to staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &output_texture,
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
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
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
