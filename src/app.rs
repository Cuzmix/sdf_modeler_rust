use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui_dock::DockState;

use crate::gpu::camera::Camera;
use crate::gpu::codegen;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::Scene;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::gizmo::{GizmoMode, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::viewport::ViewportResources;
use crate::settings::Settings;

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
    pending_pick: Option<PendingPick>,
    gizmo_state: GizmoState,
    gizmo_mode: GizmoMode,
    settings: Settings,
    initial_vsync: bool,
}

impl SdfApp {
    pub fn new(cc: &eframe::CreationContext<'_>, settings: Settings) -> Self {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .expect("WGPU render state required");

        let scene = Scene::new();
        let shader_src = codegen::generate_shader(&scene);
        let pick_shader_src = codegen::generate_pick_shader(&scene);
        let structure_key = scene.structure_key();

        let resources = ViewportResources::new(
            &render_state.device,
            render_state.target_format,
            &shader_src,
            &pick_shader_src,
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
            pending_pick: None,
            gizmo_state: GizmoState::Idle,
            gizmo_mode: GizmoMode::Translate,
            initial_vsync: settings.vsync_enabled,
            settings,
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

        // --- Save/Load keyboard shortcuts ---
        let save_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S));
        let open_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O));

        if save_pressed {
            if let Some(path) = crate::io::save_dialog() {
                if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                    log::error!("Failed to save project: {}", e);
                }
            }
        } else if open_pressed {
            if let Some(path) = crate::io::open_dialog() {
                match crate::io::load_project(&path) {
                    Ok(project) => {
                        self.scene = project.scene;
                        self.camera = project.camera;
                        self.history = History::new();
                        self.node_graph_state.selected = None;
                        self.node_graph_state.layout_dirty = true;
                        // Force pipeline rebuild
                        self.current_structure_key = 0;
                    }
                    Err(e) => {
                        log::error!("Failed to load project: {}", e);
                    }
                }
            }
        }

        // --- Frame-start: check for topology changes and update GPU ---
        let new_key = self.scene.structure_key();
        if new_key != self.current_structure_key {
            let shader_src = codegen::generate_shader(&self.scene);
            let pick_shader_src = codegen::generate_pick_shader(&self.scene);
            let mut renderer = self.render_state.renderer.write();
            if let Some(res) = renderer
                .callback_resources
                .get_mut::<ViewportResources>()
            {
                res.rebuild_pipeline(&self.render_state.device, &shader_src, &pick_shader_src);
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

        // --- Process pending GPU pick from previous frame ---
        if let Some(pending) = self.pending_pick.take() {
            let topo_order = self.scene.topo_order();
            let renderer = self.render_state.renderer.read();
            if let Some(res) = renderer
                .callback_resources
                .get::<ViewportResources>()
            {
                if let Some(result) = res.execute_pick(
                    &self.render_state.device,
                    &self.render_state.queue,
                    &pending,
                ) {
                    // Map material_id (topo_order index) → NodeId
                    let idx = result.material_id as usize;
                    if idx < topo_order.len() {
                        self.node_graph_state.selected = Some(topo_order[idx]);
                    }
                } else {
                    // Clicked empty space → deselect
                    self.node_graph_state.selected = None;
                }
            }
        }

        // --- Gizmo mode shortcuts ---
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            self.gizmo_mode = GizmoMode::Translate;
        }
        // Future: E = Rotate, R = Scale

        // --- UI ---
        self.show_debug_window(ctx, dt);

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Settings", |ui| {
                    let mut vsync = self.settings.vsync_enabled;
                    if ui.checkbox(&mut vsync, "VSync").changed() {
                        self.settings.vsync_enabled = vsync;
                        self.settings.save();
                    }
                    if self.settings.vsync_enabled != self.initial_vsync {
                        ui.weak("(restart required)");
                    }
                });
            });
        });

        let mut pending_pick = None;
        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.camera,
            scene: &mut self.scene,
            node_graph_state: &mut self.node_graph_state,
            gizmo_state: &mut self.gizmo_state,
            gizmo_mode: &self.gizmo_mode,
            time: now as f32,
            pending_pick: &mut pending_pick,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        // Store pending pick for next frame processing
        if pending_pick.is_some() {
            self.pending_pick = pending_pick;
        }

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
