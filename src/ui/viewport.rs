use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen::SdfNodeGpu;
use crate::gpu::picking::{PendingPick, PickResult};
use crate::graph::scene::NodeId;

const INITIAL_SCENE_CAPACITY: usize = 16;
const INITIAL_VOXEL_CAPACITY: usize = 4; // in f32 elements (minimum valid buffer)

pub struct ViewportResources {
    // --- Render pipeline ---
    pub pipeline: wgpu::RenderPipeline,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bgl: wgpu::BindGroupLayout,
    pub scene_buffer: wgpu::Buffer,
    pub voxel_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bgl: wgpu::BindGroupLayout,
    pub scene_buffer_capacity: usize,
    pub voxel_buffer_capacity: usize, // in f32 elements
    pub target_format: wgpu::TextureFormat,

    // --- Pick compute pipeline ---
    pub pick_pipeline: wgpu::ComputePipeline,
    pub pick_input_buffer: wgpu::Buffer,
    pub pick_output_buffer: wgpu::Buffer,
    pub pick_staging_buffer: wgpu::Buffer,
    pub pick_bind_group: wgpu::BindGroup,
    pub pick_bgl: wgpu::BindGroupLayout,
}

impl ViewportResources {
    fn create_render_pipeline(
        device: &wgpu::Device,
        shader_src: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        scene_bgl: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, scene_bgl],
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

    fn create_pick_pipeline(
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

    fn create_scene_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

    fn rebuild_scene_bind_group(&mut self, device: &wgpu::Device) {
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
    }

    pub fn new(
        device: &wgpu::Device,
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

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

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

        let scene_bgl = Self::create_scene_bgl(device);

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

        let pipeline = Self::create_render_pipeline(
            device, shader_src, &camera_bgl, &scene_bgl, target_format,
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

        let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

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

        let pick_pipeline = Self::create_pick_pipeline(
            device, pick_shader_src, &camera_bgl, &scene_bgl, &pick_bgl,
        );

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
            pick_pipeline,
            pick_input_buffer,
            pick_output_buffer,
            pick_staging_buffer,
            pick_bind_group,
            pick_bgl,
        }
    }

    pub fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader_src: &str,
        pick_shader_src: &str,
    ) {
        self.pipeline = Self::create_render_pipeline(
            device, shader_src, &self.camera_bgl, &self.scene_bgl, self.target_format,
        );
        self.pick_pipeline = Self::create_pick_pipeline(
            device, pick_shader_src, &self.camera_bgl, &self.scene_bgl, &self.pick_bgl,
        );
    }

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

    /// Dispatch pick compute shader and synchronously read back the result.
    pub fn execute_pick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending: &PendingPick,
    ) -> Option<PickResult> {
        // Write camera uniform
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&pending.camera_uniform),
        );

        // Write pick input (mouse_pos + padding)
        let pick_input: [f32; 4] = [pending.mouse_pos[0], pending.mouse_pos[1], 0.0, 0.0];
        queue.write_buffer(&self.pick_input_buffer, 0, bytemuck::cast_slice(&pick_input));

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

        encoder.copy_buffer_to_buffer(&self.pick_output_buffer, 0, &self.pick_staging_buffer, 0, 32);

        queue.submit(std::iter::once(encoder.finish()));

        // Synchronous readback
        let buffer_slice = self.pick_staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok()?.ok().is_none() {
            return None;
        }

        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = PickResult {
            material_id: floats[0] as i32,
            distance: floats[1],
            world_pos: [floats[2], floats[3], floats[4]],
        };
        drop(data);
        self.pick_staging_buffer.unmap();

        // material_id < 0 means no hit
        if result.material_id < 0 || result.distance > 49.0 {
            return None;
        }

        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Paint callback
// ---------------------------------------------------------------------------

struct ViewportCallback {
    uniform: CameraUniform,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&self.uniform),
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &resources.scene_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

use crate::graph::scene::Scene;
use crate::sculpt::SculptState;
use crate::ui::gizmo::{self, GizmoMode, GizmoState};

const BRUSH_CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 200, 200, 128);

/// Returns an optional PendingPick if the user clicked/dragged in the viewport.
pub fn draw(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    sculpt_state: &SculptState,
    time: f32,
) -> Option<PendingPick> {
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

    // --- Paint the SDF viewport (WGPU callback) ---
    let pixels_per_point = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.width() * pixels_per_point,
        rect.height() * pixels_per_point,
    ];
    let uniform = camera.to_uniform(viewport, time);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback { uniform },
    ));

    // --- Gizmo overlay (drawn on top of WGPU content) ---
    let sculpt_active = sculpt_state.is_active();

    let gizmo_consumed = if sculpt_active {
        false // Gizmo is disabled during sculpt mode
    } else {
        gizmo::draw_and_interact(
            ui.painter(),
            &response,
            camera,
            scene,
            selected,
            gizmo_state,
            gizmo_mode,
            rect,
        )
    };

    // --- Interaction priority: sculpt > gizmo > pick > orbit ---
    let mut pending_pick = None;

    if sculpt_active {
        // Sculpt mode: drag applies brush continuously via pick
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(viewport, time),
                    });
                }
            }
        }

        // Brush cursor preview
        if let SculptState::Active { brush_radius, .. } = sculpt_state {
            if let Some(hover_pos) = response.hover_pos() {
                let screen_radius = brush_radius / camera.distance * rect.height() * 0.5;
                ui.painter().circle_stroke(
                    hover_pos,
                    screen_radius,
                    egui::Stroke::new(1.5, BRUSH_CURSOR_COLOR),
                );
            }
        }

        // Right-click still orbits in sculpt mode, secondary drag pans
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
        }
    } else if !gizmo_consumed {
        // Normal mode: click to pick
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let mouse_px = [
                    (pos.x - rect.min.x) * pixels_per_point,
                    (pos.y - rect.min.y) * pixels_per_point,
                ];
                let pick_uniform = camera.to_uniform(viewport, time);
                pending_pick = Some(PendingPick {
                    mouse_pos: mouse_px,
                    camera_uniform: pick_uniform,
                });
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }

    pending_pick
}
