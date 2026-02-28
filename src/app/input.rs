use eframe::egui;
use glam::Vec3;

use crate::graph::history::History;
use crate::graph::scene::NodeData;
use crate::sculpt::{ActiveTool, BrushMode, SculptState, DEFAULT_BRUSH_STRENGTH};
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

        // Focus on selected node
        if ctx.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.ctrl) {
            if let Some(sel) = self.node_graph_state.selected {
                let parent_map = self.scene.build_parent_map();
                let (center, radius) = self.scene.compute_subtree_sphere(sel, &parent_map);
                self.camera.focus_on(
                    Vec3::new(center[0], center[1], center[2]),
                    radius.max(0.5),
                );
            }
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
            {
                // Quick save if we already have a file path, otherwise show dialog
                let path = if let Some(ref p) = self.current_file_path {
                    Some(p.clone())
                } else {
                    crate::io::save_dialog()
                };
                if let Some(path) = path {
                    if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                        log::error!("Failed to save project: {}", e);
                    } else {
                        self.current_file_path = Some(path.clone());
                        self.saved_fingerprint = self.scene.data_fingerprint();
                        self.settings.add_recent_file(&path.to_string_lossy());
                    }
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
                        self.current_file_path = Some(path.clone());
                        self.saved_fingerprint = self.scene.data_fingerprint();
                        self.settings.add_recent_file(&path.to_string_lossy());
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

        // Copy / Paste / Duplicate (consume_key prevents egui clipboard noise)
        let ctrl = egui::Modifiers::CTRL;
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::C)) {
            self.clipboard_node = self.node_graph_state.selected;
        }
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::V)) {
            if let Some(src) = self.clipboard_node {
                if self.scene.nodes.contains_key(&src) {
                    if let Some(new_id) = self.scene.duplicate_subtree(src) {
                        // Offset the new root's position
                        if let Some(node) = self.scene.nodes.get_mut(&new_id) {
                            match &mut node.data {
                                NodeData::Primitive { ref mut position, .. }
                                | NodeData::Sculpt { ref mut position, .. } => {
                                    position.x += 1.0;
                                }
                                _ => {}
                            }
                        }
                        self.node_graph_state.selected = Some(new_id);
                        self.node_graph_state.needs_initial_rebuild = true;
                        self.buffer_dirty = true;
                    }
                }
            }
        }
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::D)) {
            if let Some(sel) = self.node_graph_state.selected {
                if let Some(new_id) = self.scene.duplicate_subtree(sel) {
                    if let Some(node) = self.scene.nodes.get_mut(&new_id) {
                        match &mut node.data {
                            NodeData::Primitive { ref mut position, .. }
                            | NodeData::Sculpt { ref mut position, .. } => {
                                position.x += 1.0;
                            }
                            _ => {}
                        }
                    }
                    self.node_graph_state.selected = Some(new_id);
                    self.node_graph_state.needs_initial_rebuild = true;
                    self.buffer_dirty = true;
                }
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

        // Tool switching shortcuts
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && self.active_tool == ActiveTool::Sculpt {
            self.active_tool = ActiveTool::Select;
            self.sculpt_state = SculptState::Inactive;
            self.last_sculpt_hit = None;
            self.lazy_brush_pos = None;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::S) && !i.modifiers.ctrl) && self.active_tool == ActiveTool::Select {
            self.active_tool = ActiveTool::Sculpt;
            // Auto-activate sculpt if a Sculpt node is selected
            if let Some(sel) = self.node_graph_state.selected {
                if self.scene.nodes.get(&sel).map_or(false, |n| matches!(n.data, NodeData::Sculpt { .. })) {
                    self.sculpt_state = SculptState::new_active(sel);
                }
            }
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
