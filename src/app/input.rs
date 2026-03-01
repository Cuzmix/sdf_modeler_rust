use eframe::egui;

use crate::sculpt::{ActiveTool, BrushMode, SculptState, DEFAULT_BRUSH_STRENGTH};
use crate::ui::gizmo::GizmoMode;

use super::actions::{Action, ActionSink};
use super::{ExportStatus, SdfApp};

impl SdfApp {
    pub(super) fn delete_selected(&mut self) {
        if let Some(sel) = self.ui.node_graph_state.selected {
            self.doc.scene.remove_node(sel);
            self.ui.node_graph_state.selected = None;
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.doc.sculpt_state = SculptState::Inactive;
            self.gpu.buffer_dirty = true;
        }
    }

    /// Collect keyboard-triggered actions into the action sink.
    /// Reads state immutably to decide what actions to emit; does not
    /// mutate app state directly (except for egui input consumption).
    pub(super) fn collect_keyboard_actions(&mut self, ctx: &egui::Context, actions: &mut ActionSink) {
        // Help
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            actions.push(Action::ToggleHelp);
        }
        // Camera presets
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            actions.push(Action::CameraFront);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F6)) {
            actions.push(Action::CameraTop);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F7)) {
            actions.push(Action::CameraRight);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F8)) {
            actions.push(Action::CameraBack);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F9)) {
            actions.push(Action::CameraLeft);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F10)) {
            actions.push(Action::CameraBottom);
        }
        // Orthographic toggle
        if ctx.input(|i| i.key_pressed(egui::Key::O) && !i.modifiers.ctrl) {
            actions.push(Action::ToggleOrtho);
        }

        // Focus on selected node / Frame all
        if ctx.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.ctrl) {
            actions.push(Action::FocusSelected);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Home)) {
            actions.push(Action::FrameAll);
        }

        // Debug toggle
        if ctx.input(|i| i.key_pressed(egui::Key::F4)) {
            actions.push(Action::ToggleDebug);
        }

        // Delete selected node
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            actions.push(Action::DeleteSelected);
        }

        // Undo / Redo
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z)) {
            actions.push(Action::Undo);
        } else if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y)) {
            actions.push(Action::Redo);
        }

        // Save / Load
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S)) {
            actions.push(Action::SaveProject);
        } else if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O)) {
            actions.push(Action::OpenProject);
        }

        // Screenshot
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::P)) {
            actions.push(Action::TakeScreenshot);
        }

        // Export
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::E)) {
            if matches!(self.async_state.export_status, ExportStatus::Idle) {
                actions.push(Action::ShowExportDialog);
            }
        }

        // Copy / Paste / Duplicate (consume_key prevents egui clipboard noise)
        let ctrl = egui::Modifiers::CTRL;
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::C)) {
            actions.push(Action::Copy);
        }
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::V)) {
            actions.push(Action::Paste);
        }
        if ctx.input_mut(|i| i.consume_key(ctrl, egui::Key::D)) {
            actions.push(Action::Duplicate);
        }

        // Gizmo mode
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            actions.push(Action::SetGizmoMode(GizmoMode::Translate));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::E) && !i.modifiers.ctrl) {
            actions.push(Action::SetGizmoMode(GizmoMode::Rotate));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            actions.push(Action::SetGizmoMode(GizmoMode::Scale));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::G)) {
            actions.push(Action::ToggleGizmoSpace);
        }
        if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::C)) {
            actions.push(Action::ResetPivot);
        }

        // Isolation mode
        if ctx.input(|i| i.key_pressed(egui::Key::Slash)) {
            actions.push(Action::ToggleIsolation);
        }

        // Shading mode cycle (Z key, but not in sculpt mode — Z is symmetry toggle there)
        if ctx.input(|i| i.key_pressed(egui::Key::Z) && !i.modifiers.ctrl) && !self.doc.sculpt_state.is_active() {
            actions.push(Action::CycleShadingMode);
        }

        // Turntable toggle (Space, only when no text field focused)
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) && !ctx.wants_keyboard_input() {
            actions.push(Action::ToggleTurntable);
        }

        // Property copy/paste (Ctrl+Shift+C/V)
        if ctx.input(|i| i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::C)) {
            actions.push(Action::CopyProperties);
        }
        if ctx.input(|i| i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::V)) {
            actions.push(Action::PasteProperties);
        }

        // Camera bookmarks: Ctrl+1-9 to save
        if !self.doc.sculpt_state.is_active() {
            for (idx, key) in [
                egui::Key::Num1, egui::Key::Num2, egui::Key::Num3,
                egui::Key::Num4, egui::Key::Num5, egui::Key::Num6,
                egui::Key::Num7, egui::Key::Num8, egui::Key::Num9,
            ].iter().enumerate() {
                if ctx.input(|i| i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(*key)) {
                    actions.push(Action::SaveBookmark(idx));
                }
            }
        }

        // Command palette (Ctrl+K)
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::K)) {
            actions.push(Action::ToggleCommandPalette);
        }

        // Tool switching shortcuts
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && self.doc.active_tool == ActiveTool::Sculpt {
            actions.push(Action::SetTool(ActiveTool::Select));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::S) && !i.modifiers.ctrl) && self.doc.active_tool == ActiveTool::Select {
            actions.push(Action::SetTool(ActiveTool::Sculpt));
        }

        // Sculpt brush mode shortcuts (when sculpt is active)
        // These are direct mutations on sculpt_state fields — kept inline
        // because they are data-level edits (like slider drags), not structural.
        if self.doc.sculpt_state.is_active() {
            if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Add;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Carve;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Smooth;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num4)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Flatten;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num5)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    if *brush_mode == BrushMode::Grab && *brush_strength > 0.5 { *brush_strength = DEFAULT_BRUSH_STRENGTH; }
                    *brush_mode = BrushMode::Inflate;
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Num6)) {
                if let SculptState::Active { ref mut brush_mode, ref mut brush_strength, .. } = self.doc.sculpt_state {
                    *brush_mode = BrushMode::Grab;
                    if *brush_strength < 0.5 {
                        *brush_strength = 1.0;
                    }
                }
            }
            // Bracket keys: resize brush
            if ctx.input(|i| i.key_pressed(egui::Key::OpenBracket)) {
                if let SculptState::Active { ref mut brush_radius, .. } = self.doc.sculpt_state {
                    *brush_radius = (*brush_radius - 0.05).max(0.05);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::CloseBracket)) {
                if let SculptState::Active { ref mut brush_radius, .. } = self.doc.sculpt_state {
                    *brush_radius = (*brush_radius + 0.05).min(2.0);
                }
            }
            // Symmetry toggles: X/Y/Z
            if ctx.input(|i| i.key_pressed(egui::Key::X)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.doc.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(0) { None } else { Some(0) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Y)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.doc.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(1) { None } else { Some(1) };
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Z)) {
                if let SculptState::Active { ref mut symmetry_axis, .. } = self.doc.sculpt_state {
                    *symmetry_axis = if *symmetry_axis == Some(2) { None } else { Some(2) };
                }
            }
        }
    }
}
