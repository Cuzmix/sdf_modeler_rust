use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui_dock::DockState;

use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::graph::history::History;
use crate::graph::scene::Scene;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::ViewportResources;

pub struct SdfApp {
    camera: Camera,
    scene: Scene,
    dock_state: DockState<Tab>,
    node_graph_state: NodeGraphState,
    render_state: RenderState,
    current_structure_key: u64,
    history: History,
    last_time: f64,
    show_debug: bool,
}

impl SdfApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");

        let scene = Scene::new();
        let shader_src = codegen::generate_shader(&scene);
        let structure_key = scene.structure_key();

        let resources = ViewportResources::new(
            &render_state.device,
            render_state.target_format,
            &shader_src,
        );

        // Upload initial scene buffer
        let node_data = codegen::build_node_buffer(&scene, None);
        {
            let mut renderer = render_state.renderer.write();
            renderer.callback_resources.insert(resources);
        }
        {
            let mut renderer = render_state.renderer.write();
            let res = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
                .unwrap();
            res.update_scene_buffer(&render_state.device, &render_state.queue, &node_data);
        }

        Self {
            camera: Camera::default(),
            scene,
            dock_state: dock::create_dock_state(),
            node_graph_state: NodeGraphState::new(),
            render_state,
            current_structure_key: structure_key,
            history: History::new(),
            last_time: 0.0,
            show_debug: true,
        }
    }

    fn show_debug_window(&self, ctx: &egui::Context, dt: f64) {
        if !self.show_debug {
            return;
        }
        egui::Window::new("Debug")
            .default_pos([10.0, 10.0])
            .default_size([220.0, 160.0])
            .show(ctx, |ui| {
                let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                ui.label(format!("FPS: {:.0}", fps));
                ui.separator();
                let eye = self.camera.eye();
                ui.label(format!("Eye: ({:.2}, {:.2}, {:.2})", eye.x, eye.y, eye.z));
                ui.label(format!(
                    "Target: ({:.2}, {:.2}, {:.2})",
                    self.camera.target.x, self.camera.target.y, self.camera.target.z
                ));
                ui.label(format!("Distance: {:.2}", self.camera.distance));
                ui.label(format!(
                    "Yaw: {:.1}°  Pitch: {:.1}°",
                    self.camera.yaw.to_degrees(),
                    self.camera.pitch.to_degrees(),
                ));
                ui.separator();
                ui.label(format!("Nodes: {}", self.scene.nodes.len()));
                if let Some(root) = self.scene.root {
                    ui.label(format!("Root: {}", root));
                } else {
                    ui.label("Root: none");
                }
            });
    }
}

impl eframe::App for SdfApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = ctx.input(|i| i.time);
        let dt = now - self.last_time;
        self.last_time = now;

        if ctx.input(|i| i.key_pressed(egui::Key::F3)) {
            self.show_debug = !self.show_debug;
        }

        // --- Undo/Redo: capture "before" snapshot ---
        self.history
            .begin_frame(&self.scene, self.node_graph_state.selected);

        // --- Undo/Redo keyboard shortcuts ---
        let undo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z));
        let redo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y));

        if undo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .undo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
            }
        } else if redo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .redo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.layout_dirty = true;
            }
        }

        // --- Frame-start: check for topology changes and update GPU ---
        let new_key = self.scene.structure_key();
        if new_key != self.current_structure_key {
            let shader_src = codegen::generate_shader(&self.scene);
            let mut renderer = self.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                res.rebuild_pipeline(&self.render_state.device, &shader_src);
            }
            self.current_structure_key = new_key;
        }

        // Update scene buffer every frame (parameters may have changed)
        let node_data =
            codegen::build_node_buffer(&self.scene, self.node_graph_state.selected);
        {
            let mut renderer = self.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                res.update_scene_buffer(
                    &self.render_state.device,
                    &self.render_state.queue,
                    &node_data,
                );
            }
        }

        // --- UI ---
        self.show_debug_window(ctx, dt);

        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.camera,
            scene: &mut self.scene,
            node_graph_state: &mut self.node_graph_state,
            time: now as f32,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        // --- Undo/Redo: end-of-frame commit ---
        let is_dragging = ctx.dragged_id().is_some();
        self.history.end_frame(
            &self.scene,
            self.node_graph_state.selected,
            is_dragging,
        );

        ctx.request_repaint();
    }
}
