use eframe::egui;
use egui_dock::DockState;

use crate::gpu::camera::Camera;
use crate::graph::scene::Scene;
use crate::ui::dock::{self, SdfTabViewer, Tab};
use crate::ui::viewport::ViewportResources;

pub struct SdfApp {
    camera: Camera,
    scene: Scene,
    dock_state: DockState<Tab>,
    last_time: f64,
    show_debug: bool,
}

impl SdfApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(wgpu_state) = &cc.wgpu_render_state {
            let resources =
                ViewportResources::new(&wgpu_state.device, wgpu_state.target_format);
            wgpu_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);
        }

        Self {
            camera: Camera::default(),
            scene: Scene::new(),
            dock_state: dock::create_dock_state(),
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

        self.show_debug_window(ctx, dt);

        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.camera,
            time: now as f32,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.dock_state)
                    .show_inside(ui, &mut tab_viewer);
            });

        ctx.request_repaint();
    }
}
