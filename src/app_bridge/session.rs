use glam::Vec3;

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode, SdfPrimitive};
use crate::settings::RenderConfig;

use super::dto::{
    AppCameraSnapshot, AppNodeSnapshot, AppSceneSnapshot, AppSceneStatsSnapshot, AppToolSnapshot,
    AppVec3, AppViewportFeedbackSnapshot,
};
use super::renderer::{HeadlessPickRequest, HeadlessRenderRequest, HeadlessViewportRenderer};

pub struct RenderedViewportFrame {
    pub pixels: Vec<u8>,
    pub camera_animating: bool,
}

pub struct AppBridge {
    scene: Scene,
    camera: Camera,
    render_config: RenderConfig,
    selected_node: Option<NodeId>,
    hovered_node: Option<NodeId>,
    active_tool_label: String,
    renderer: HeadlessViewportRenderer,
    last_viewport_time_seconds: Option<f32>,
}

impl Default for AppBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBridge {
    pub fn new() -> Self {
        let scene = Scene::new();
        let camera = Camera::default();
        let render_config = RenderConfig::default();
        let renderer = HeadlessViewportRenderer::new(&scene, &render_config);

        Self {
            scene,
            camera,
            render_config,
            selected_node: None,
            hovered_node: None,
            active_tool_label: "Select".to_string(),
            renderer,
            last_viewport_time_seconds: None,
        }
    }

    pub fn scene_snapshot(&self) -> AppSceneSnapshot {
        let top_level_nodes = self
            .scene
            .top_level_nodes()
            .into_iter()
            .filter_map(|node_id| self.scene.nodes.get(&node_id))
            .map(|node| self.node_snapshot(node))
            .collect();

        let node_counts = self.scene.node_type_counts();
        let bounds = self.scene.compute_bounds();

        AppSceneSnapshot {
            selected_node: self.node_snapshot_by_id(self.selected_node),
            top_level_nodes,
            camera: self.camera_snapshot(),
            stats: AppSceneStatsSnapshot {
                total_nodes: node_counts.total as u32,
                visible_nodes: node_counts.visible as u32,
                top_level_nodes: self.scene.top_level_nodes().len() as u32,
                primitive_nodes: node_counts.primitives as u32,
                operation_nodes: node_counts.operations as u32,
                transform_nodes: node_counts.transforms as u32,
                modifier_nodes: node_counts.modifiers as u32,
                sculpt_nodes: node_counts.sculpts as u32,
                light_nodes: node_counts.lights as u32,
                voxel_memory_bytes: self.scene.voxel_memory_bytes() as u64,
                sdf_eval_complexity: self.scene.sdf_eval_complexity() as u32,
                structure_key: self.scene.structure_key(),
                data_fingerprint: self.scene.data_fingerprint(),
                bounds_min: AppVec3::new(bounds.0[0], bounds.0[1], bounds.0[2]),
                bounds_max: AppVec3::new(bounds.1[0], bounds.1[1], bounds.1[2]),
            },
            tool: AppToolSnapshot {
                active_tool_label: self.active_tool_label.clone(),
                shading_mode_label: self.render_config.shading_mode.label().to_string(),
                grid_enabled: self.render_config.show_grid,
            },
        }
    }

    pub fn viewport_feedback(&self) -> AppViewportFeedbackSnapshot {
        AppViewportFeedbackSnapshot {
            camera: self.camera_snapshot(),
            selected_node: self.node_snapshot_by_id(self.selected_node),
            hovered_node: self.node_snapshot_by_id(self.hovered_node),
        }
    }

    pub fn render_viewport_frame(
        &mut self,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> RenderedViewportFrame {
        let camera_animating = self.tick_camera_animation(time_seconds);
        let pixels = self.renderer.render_scene(HeadlessRenderRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            selected_node: self.selected_node,
            time_seconds,
            width,
            height,
        });

        RenderedViewportFrame {
            pixels,
            camera_animating,
        }
    }

    pub fn orbit_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.cancel_camera_transition();
        self.camera.orbit(delta_x, delta_y);
        self.camera.clamp_pitch();
    }

    pub fn pan_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.cancel_camera_transition();
        self.camera.pan(delta_x, delta_y);
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.cancel_camera_transition();
        self.camera.zoom(delta);
    }

    pub fn hover_node_at_viewport(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> Option<u64> {
        let next_hovered_node = self.renderer.pick_node(HeadlessPickRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            time_seconds,
            width,
            height,
            mouse_x,
            mouse_y,
        });
        self.hovered_node = next_hovered_node;
        next_hovered_node
    }

    pub fn clear_hovered_node(&mut self) {
        self.hovered_node = None;
    }

    pub fn select_node_at_viewport(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> Option<u64> {
        let next_selected_node = self.renderer.pick_node(HeadlessPickRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            time_seconds,
            width,
            height,
            mouse_x,
            mouse_y,
        });
        self.selected_node = next_selected_node;
        self.hovered_node = next_selected_node;
        next_selected_node
    }

    pub fn focus_selected(&mut self) {
        let Some(selected_node) = self.selected_node else {
            return;
        };

        self.cancel_camera_transition();
        let parent_map = self.scene.build_parent_map();
        let (center, radius) = self.scene.compute_subtree_sphere(selected_node, &parent_map);
        self.camera
            .focus_on(Vec3::new(center[0], center[1], center[2]), radius.max(0.5));
    }

    pub fn frame_all(&mut self) {
        self.cancel_camera_transition();
        let bounds = self.scene.compute_bounds();
        let bounds_min = Vec3::new(bounds.0[0], bounds.0[1], bounds.0[2]);
        let bounds_max = Vec3::new(bounds.1[0], bounds.1[1], bounds.1[2]);
        let center = (bounds_min + bounds_max) * 0.5;
        let radius = ((bounds_max - bounds_min) * 0.5).length().max(0.5);
        self.camera.focus_on(center, radius);
    }

    pub fn camera_front(&mut self) {
        self.start_camera_view_transition(0.0, 0.0, 0.0);
    }

    pub fn camera_top(&mut self) {
        self.start_camera_view_transition(0.0, std::f32::consts::FRAC_PI_2, 0.0);
    }

    pub fn camera_right(&mut self) {
        self.start_camera_view_transition(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    }

    pub fn camera_back(&mut self) {
        self.start_camera_view_transition(std::f32::consts::PI, 0.0, 0.0);
    }

    pub fn camera_left(&mut self) {
        self.start_camera_view_transition(-std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    }

    pub fn camera_bottom(&mut self) {
        self.start_camera_view_transition(0.0, -std::f32::consts::FRAC_PI_2, 0.0);
    }

    pub fn toggle_orthographic(&mut self) {
        self.cancel_camera_transition();
        self.camera.toggle_ortho();
    }

    pub fn add_primitive(&mut self, kind: SdfPrimitive) -> u64 {
        let new_node_id = self.scene.create_primitive(kind);
        self.selected_node = Some(new_node_id);
        self.hovered_node = Some(new_node_id);
        new_node_id
    }

    pub fn add_sphere(&mut self) -> u64 {
        self.add_primitive(SdfPrimitive::Sphere)
    }

    pub fn select_node(&mut self, node_id: Option<u64>) {
        self.selected_node = node_id.filter(|id| self.scene.nodes.contains_key(id));
        self.hovered_node = self.selected_node;
    }

    pub fn reset_scene(&mut self) {
        self.scene = Scene::new();
        self.camera = Camera::default();
        self.selected_node = None;
        self.hovered_node = None;
        self.last_viewport_time_seconds = None;
    }

    fn cancel_camera_transition(&mut self) {
        self.camera.transition = None;
    }

    fn start_camera_view_transition(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.camera.start_transition(yaw, pitch, roll);
    }

    fn tick_camera_animation(&mut self, time_seconds: f32) -> bool {
        let delta_seconds = self
            .last_viewport_time_seconds
            .map(|last_time_seconds| (time_seconds - last_time_seconds).clamp(0.0, 0.1))
            .unwrap_or(0.0);
        self.last_viewport_time_seconds = Some(time_seconds);
        self.camera.tick_transition(delta_seconds as f64)
    }

    fn camera_snapshot(&self) -> AppCameraSnapshot {
        let eye = self.camera.eye();
        AppCameraSnapshot {
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            roll: self.camera.roll,
            distance: self.camera.distance,
            fov_degrees: self.camera.fov.to_degrees(),
            orthographic: self.camera.orthographic,
            target: app_vec3(self.camera.target),
            eye: AppVec3::new(eye.x, eye.y, eye.z),
        }
    }

    fn node_snapshot_by_id(&self, node_id: Option<NodeId>) -> Option<AppNodeSnapshot> {
        node_id
            .and_then(|resolved_node_id| self.scene.nodes.get(&resolved_node_id))
            .map(|node| self.node_snapshot(node))
    }

    fn node_snapshot(&self, node: &SceneNode) -> AppNodeSnapshot {
        AppNodeSnapshot {
            id: node.id,
            name: node.name.clone(),
            kind_label: node_kind_label(node),
            visible: !self.scene.is_hidden(node.id),
            locked: node.locked,
        }
    }
}

fn node_kind_label(node: &SceneNode) -> String {
    match &node.data {
        NodeData::Primitive { kind, .. } => kind.base_name().to_string(),
        NodeData::Operation { op, .. } => op.base_name().to_string(),
        NodeData::Sculpt { .. } => "Sculpt".to_string(),
        NodeData::Transform { .. } => "Transform".to_string(),
        NodeData::Modifier { kind, .. } => kind.base_name().to_string(),
        NodeData::Light { light_type, .. } => light_type.label().to_string(),
    }
}

fn app_vec3(value: Vec3) -> AppVec3 {
    AppVec3::new(value.x, value.y, value.z)
}
