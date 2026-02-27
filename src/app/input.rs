use eframe::egui;
use glam::Vec3;

use crate::graph::history::History;
use crate::sculpt::{BrushMode, SculptState, DEFAULT_BRUSH_STRENGTH};
use crate::ui::gizmo::{GizmoMode, GizmoSpace};

use super::{ExportStatus, SdfApp};

impl SdfApp {
    pub(super) fn delete_selected(&mut self) {
        if let Some(sel) = self.node_graph_state.selected {
            self.scene.remove_node(sel);
            self.node_graph_state.selected = None;
            self.node_graph_state.needs_initial_rebuild = true;
            self.sculpt_state = SculptState::Inactive;
            self.buffer_dirty = true;
        }
    }

    pub(super) fn handle_keyboard_input(&mut self, ctx: &egui::Context) {
        // Help
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_help = !self.show_help;
        }
        // Camera presets
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.camera.set_front();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F6)) {
            self.camera.set_top();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F7)) {
            self.camera.set_right();
        }

        // Debug toggle
        if ctx.input(|i| i.key_pressed(egui::Key::F4)) {
            self.show_debug = !self.show_debug;
        }

        // Delete selected node
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            self.delete_selected();
        }

        // Undo / Redo
        let undo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z));
        let redo_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y));

        if undo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .undo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.needs_initial_rebuild = true;
                self.buffer_dirty = true;
            }
        } else if redo_pressed {
            if let Some((restored_scene, restored_sel)) = self
                .history
                .redo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.needs_initial_rebuild = true;
                self.buffer_dirty = true;
            }
        }

        // Save / Load
        let save_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S));
        let open_pressed = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O));

        if save_pressed {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(path) = crate::io::save_dialog() {
                if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                    log::error!("Failed to save project: {}", e);
                }
            }
            #[cfg(target_arch = "wasm32")]
            crate::io::web_save_project(&self.scene, &self.camera);
        } else if open_pressed {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(path) = crate::io::open_dialog() {
                match crate::io::load_project(&path) {
                    Ok(project) => {
                        self.scene = project.scene;
                        self.camera = project.camera;
                        self.history = History::new();
                        self.node_graph_state.selected = None;
                        self.node_graph_state.needs_initial_rebuild = true;
                        self.sculpt_state = SculptState::Inactive;
                        self.current_structure_key = 0; // Force pipeline rebuild
                        self.buffer_dirty = true;
                    }
                    Err(e) => {
                        log::error!("Failed to load project: {}", e);
                    }
                }
            }
        }

        // Screenshot
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::P)) {
            self.take_screenshot();
        }

        // Export OBJ
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::E)) {
            if matches!(self.export_status, ExportStatus::Idle) {
                self.start_export(ctx);
            }
        }

        // Gizmo mode
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            self.gizmo_mode = GizmoMode::Translate;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::E) && !i.modifiers.ctrl) {
            self.gizmo_mode = GizmoMode::Rotate;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.gizmo_mode = GizmoMode::Scale;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::G)) {
            self.gizmo_space = match self.gizmo_space {
                GizmoSpace::Local => GizmoSpace::World,
                GizmoSpace::World => GizmoSpace::Local,
            };
        }
        if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::C)) {
            self.pivot_offset = Vec3::ZERO;
        }

        // Sculpt brush mode shortcuts (when sculpt is active)
        if self.sculpt_state.is_active() {
            if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Add;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Carve;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Smooth;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num4)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Flatten;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num5)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Inflate;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num6)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.sculpt_state {
                    *brush_mode = BrushMode::Grab;
                    if *brush_strength < 0.5 {
                        *brush_strength = 1.0;
                    }
                }
            }
            // Symmetry toggles: X/Y/Z
            if ctx.input(|i| i.key_pressed(egui::Key::X)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(0) { None } else { Some(0) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Y)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(1) { None } else { Some(1) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Z)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(2) { None } else { Some(2) };
                }
            }
        }
    }

}
