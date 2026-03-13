use std::collections::HashSet;

use eframe::wgpu;

use crate::gpu::buffers::{
    build_node_buffer, build_voxel_buffer, collect_scene_lights, collect_sculpt_tex_info,
};
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::codegen::{generate_pick_shader, generate_shader};
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::RenderConfig;
use crate::ui::viewport::ViewportResources;

const HEADLESS_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct HeadlessRenderRequest<'a> {
    pub scene: &'a Scene,
    pub camera: &'a Camera,
    pub render_config: &'a RenderConfig,
    pub selected_node: Option<NodeId>,
    pub time_seconds: f32,
    pub width: u32,
    pub height: u32,
}

pub struct HeadlessPickRequest<'a> {
    pub scene: &'a Scene,
    pub camera: &'a Camera,
    pub render_config: &'a RenderConfig,
    pub time_seconds: f32,
    pub width: u32,
    pub height: u32,
    pub mouse_x: f32,
    pub mouse_y: f32,
}

pub struct HeadlessViewportRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    viewport_resources: ViewportResources,
    current_structure_key: Option<u64>,
    last_data_fingerprint: Option<u64>,
    last_selected_node: Option<NodeId>,
    last_render_config: RenderConfig,
}

impl HeadlessViewportRenderer {
    pub fn new(scene: &Scene, render_config: &RenderConfig) -> Self {
        let (device, queue) = create_headless_device();
        let shader_src = generate_shader(scene, render_config);
        let pick_shader_src = generate_pick_shader(scene, render_config);
        let sculpt_count = collect_sculpt_tex_info(scene).len();
        let viewport_resources = ViewportResources::new(
            &device,
            HEADLESS_TEXTURE_FORMAT,
            &shader_src,
            &pick_shader_src,
        );

        let mut renderer = Self {
            device,
            queue,
            viewport_resources,
            current_structure_key: None,
            last_data_fingerprint: None,
            last_selected_node: None,
            last_render_config: render_config.clone(),
        };
        renderer.sync_scene(scene, render_config, None, sculpt_count);
        renderer
    }

    pub fn render_scene(&mut self, request: HeadlessRenderRequest<'_>) -> Vec<u8> {
        let sculpt_count = collect_sculpt_tex_info(request.scene).len();
        self.sync_scene(
            request.scene,
            request.render_config,
            request.selected_node,
            sculpt_count,
        );

        let render_uniform = build_render_uniform(
            request.scene,
            request.camera,
            request.render_config,
            request.selected_node,
            request.time_seconds,
            request.width.max(1),
            request.height.max(1),
        );

        self.viewport_resources.screenshot(
            &self.device,
            &self.queue,
            &render_uniform,
            request.width.max(1),
            request.height.max(1),
        )
    }

    pub fn pick_node(&mut self, request: HeadlessPickRequest<'_>) -> Option<NodeId> {
        let sculpt_count = collect_sculpt_tex_info(request.scene).len();
        self.sync_scene(request.scene, request.render_config, None, sculpt_count);

        let pending_pick = PendingPick {
            mouse_pos: [request.mouse_x, request.mouse_y],
            camera_uniform: build_render_uniform(
                request.scene,
                request.camera,
                request.render_config,
                None,
                request.time_seconds,
                request.width.max(1),
                request.height.max(1),
            ),
            ctrl_held: false,
        };

        let pick_result = self.viewport_resources.execute_pick(
            &self.device,
            &self.queue,
            &pending_pick,
        )?;

        let visible_topo_order = request.scene.visible_topo_order();
        visible_topo_order.get(pick_result.material_id as usize).copied()
    }

    fn sync_scene(
        &mut self,
        scene: &Scene,
        render_config: &RenderConfig,
        selected_node: Option<NodeId>,
        sculpt_count: usize,
    ) {
        let structure_key = scene.structure_key();
        let structure_changed = self.current_structure_key != Some(structure_key);
        let shader_config_changed = self.last_render_config.needs_shader_rebuild(render_config);

        if structure_changed || shader_config_changed {
            let shader_src = generate_shader(scene, render_config);
            let pick_shader_src = generate_pick_shader(scene, render_config);
            self.viewport_resources.rebuild_pipeline(
                &self.device,
                &shader_src,
                &pick_shader_src,
                sculpt_count,
            );
            self.current_structure_key = Some(structure_key);
            self.last_render_config = render_config.clone();
        }

        let data_fingerprint = scene.data_fingerprint();
        let scene_data_changed = self.last_data_fingerprint != Some(data_fingerprint);
        let selection_changed = self.last_selected_node != selected_node;

        let (voxel_data, voxel_offsets) = build_voxel_buffer(scene);

        if structure_changed || scene_data_changed || selection_changed {
            let selected_set = selected_node.into_iter().collect::<HashSet<_>>();
            let node_data = build_node_buffer(scene, &selected_set, &voxel_offsets);
            self.viewport_resources
                .update_scene_buffer(&self.device, &self.queue, &node_data);
            self.last_selected_node = selected_node;
        }

        if structure_changed || scene_data_changed {
            self.viewport_resources
                .update_voxel_buffer(&self.device, &self.queue, &voxel_data);

            for sculpt_info in collect_sculpt_tex_info(scene) {
                let Some(node) = scene.nodes.get(&sculpt_info.node_id) else {
                    continue;
                };

                if let NodeData::Sculpt { voxel_grid, .. } = &node.data {
                    self.viewport_resources.upload_voxel_texture(
                        &self.device,
                        &self.queue,
                        sculpt_info.tex_idx,
                        voxel_grid.resolution,
                        &voxel_grid.data,
                    );
                }
            }

            self.last_data_fingerprint = Some(data_fingerprint);
        }
    }
}

fn create_headless_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("headless viewport adapter");

    let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };

    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("SDF Modeler headless renderer"),
            required_features: wgpu::Features::FLOAT32_FILTERABLE,
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 8192,
                max_storage_buffers_per_shader_stage: 4,
                max_storage_buffer_binding_size: 1 << 27,
                max_storage_textures_per_shader_stage: 4,
                ..base_limits
            },
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("headless viewport device")
}

fn build_render_uniform(
    scene: &Scene,
    camera: &Camera,
    render_config: &RenderConfig,
    selected_node: Option<NodeId>,
    time_seconds: f32,
    width: u32,
    height: u32,
) -> CameraUniform {
    let viewport = [0.0, 0.0, width as f32, height as f32];
    let scene_bounds = scene.compute_bounds();
    let selected_idx = selected_node
        .and_then(|node_id| {
            let order = scene.visible_topo_order();
            order
                .iter()
                .position(|&visible_node_id| visible_node_id == node_id)
        })
        .map(|index| index as f32)
        .unwrap_or(-1.0);

    let (scene_light_count, scene_light_list, scene_ambient) =
        collect_scene_lights(scene, camera.eye(), None, time_seconds);
    let volumetric_count = scene_light_list
        .iter()
        .filter(|light| light.volumetric[0] > 0.5)
        .count() as f32;
    let scene_light_info = [
        scene_light_count as f32,
        volumetric_count,
        render_config.volumetric_steps as f32,
        0.0,
    ];

    let mut scene_lights_flat = [[0.0_f32; 4]; 32];
    let mut scene_light_vol = [[0.0_f32; 4]; 8];
    for (index, light) in scene_light_list.iter().enumerate() {
        scene_lights_flat[index * 4] = light.position_type;
        scene_lights_flat[index * 4 + 1] = light.direction_intensity;
        scene_lights_flat[index * 4 + 2] = light.color_range;
        scene_lights_flat[index * 4 + 3] = light.params;
        scene_light_vol[index] = light.volumetric;
    }

    let ambient_luminance = scene_ambient
        .color
        .dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    let effective_ambient = if ambient_luminance > 0.0 {
        ambient_luminance
    } else {
        render_config.ambient
    };

    let cross_section = [
        render_config.cross_section_axis as f32,
        render_config.cross_section_position,
        0.0,
        0.0,
    ];

    camera.to_uniform(
        viewport,
        time_seconds,
        0.0,
        render_config.show_grid,
        scene_bounds,
        selected_idx,
        render_config.shading_mode.gpu_value(),
        [0.0; 4],
        cross_section,
        effective_ambient,
        scene_light_info,
        scene_lights_flat,
        scene_light_vol,
    )
}
