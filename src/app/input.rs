use eframe::egui;

use crate::keymap::ActionBinding;
use crate::sculpt::{ActiveTool, BrushMode};
use crate::ui::gizmo::GizmoMode;

use super::actions::{Action, ActionSink};
use super::state::{InteractionMode, ShellPanelKind};
use super::{ExportStatus, SdfApp};

impl SdfApp {
    pub(super) fn delete_selected(&mut self) {
        if let Some(sel) = self.ui.node_graph_state.selected {
            self.doc.scene.remove_node(sel);
            self.ui.node_graph_state.clear_selection();
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.doc.sculpt_state = crate::sculpt::SculptState::new_inactive();
            self.sync_interaction_mode_after_sculpt_exit();
            self.gpu.buffer_dirty = true;
        }
    }

    /// Collect keyboard-triggered actions into the action sink.
    /// Iterates over the configurable keymap instead of hardcoded if-chains.
    pub(super) fn collect_keyboard_actions(
        &mut self,
        ctx: &egui::Context,
        actions: &mut ActionSink,
    ) {
        let is_sculpt = self.doc.sculpt_state.is_active();

        // Phase 1: detect which bindings were pressed (immutable self access).
        // We collect into a Vec so that the dispatch phase can mutate self freely.
        let mut triggered: Vec<ActionBinding> = Vec::new();

        for (&binding, combo) in self.settings.keymap.bindings() {
            // --- Context filtering ---

            // Sculpt-only bindings require active sculpt mode
            if binding.is_sculpt_only() && !is_sculpt {
                continue;
            }
            // CycleShadingMode (Z) conflicts with SculptSymmetryZ in sculpt mode
            if is_sculpt && matches!(binding, ActionBinding::CycleShadingMode) {
                continue;
            }
            // In sculpt mode, F is repurposed for Blender-style brush radius modal editing.
            if is_sculpt && matches!(binding, ActionBinding::FocusSelected) {
                continue;
            }
            // Turntable toggle should not fire when a text field is focused
            if matches!(binding, ActionBinding::ToggleTurntable) && ctx.wants_keyboard_input() {
                continue;
            }
            // Export requires idle status
            if matches!(binding, ActionBinding::ShowExportDialog)
                && !matches!(self.async_state.export_status, ExportStatus::Idle)
            {
                continue;
            }
            // Quick sculpt only from select tool
            if matches!(binding, ActionBinding::QuickSculptMode)
                && self.doc.active_tool != ActiveTool::Select
            {
                continue;
            }
            // Exit sculpt only when in sculpt tool
            if matches!(binding, ActionBinding::ExitSculptMode)
                && self.doc.active_tool != ActiveTool::Sculpt
            {
                continue;
            }

            // --- Key press detection ---

            let pressed = match binding {
                // Clipboard and command palette use consume_key to prevent egui interference
                ActionBinding::Copy
                | ActionBinding::Paste
                | ActionBinding::Duplicate
                | ActionBinding::ToggleCommandPalette => {
                    ctx.input_mut(|i| i.consume_key(combo.egui_modifiers(), combo.egui_key()))
                }
                _ => ctx
                    .input(|i| i.key_pressed(combo.egui_key()) && combo.matches_egui(&i.modifiers)),
            };

            if pressed {
                triggered.push(binding);
            }
        }

        // Phase 2: dispatch triggered bindings (mutable self access)
        for binding in triggered {
            self.dispatch_binding(binding, actions);
        }

        // Camera bookmarks (Ctrl+1-9) — not in the keymap, kept inline
        if !is_sculpt {
            for (idx, key) in [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
                egui::Key::Num5,
                egui::Key::Num6,
                egui::Key::Num7,
                egui::Key::Num8,
                egui::Key::Num9,
            ]
            .iter()
            .enumerate()
            {
                if ctx.input(|i| i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(*key)) {
                    actions.push(Action::SaveBookmark(idx));
                }
            }
        }
    }

    /// Map an ActionBinding to the corresponding app action or direct state mutation.
    fn dispatch_binding(&mut self, binding: ActionBinding, actions: &mut ActionSink) {
        match binding {
            // --- Direct action mappings ---
            ActionBinding::ToggleHelp => actions.push(Action::ToggleHelp),
            ActionBinding::ToggleDebug => actions.push(Action::ToggleDebug),
            ActionBinding::NewScene => actions.push(Action::NewScene),
            ActionBinding::OpenProject => actions.push(Action::OpenProject),
            ActionBinding::SaveProject => actions.push(Action::SaveProject),
            ActionBinding::DeleteSelected => actions.push(Action::DeleteSelected),
            ActionBinding::TakeScreenshot => actions.push(Action::TakeScreenshot),
            ActionBinding::Copy => actions.push(Action::Copy),
            ActionBinding::Paste => actions.push(Action::Paste),
            ActionBinding::Duplicate => actions.push(Action::Duplicate),
            ActionBinding::CopyProperties => actions.push(Action::CopyProperties),
            ActionBinding::PasteProperties => actions.push(Action::PasteProperties),
            ActionBinding::ToggleCommandPalette => actions.push(Action::ToggleCommandPalette),
            ActionBinding::CameraFront => actions.push(Action::CameraFront),
            ActionBinding::CameraTop => actions.push(Action::CameraTop),
            ActionBinding::CameraRight => actions.push(Action::CameraRight),
            ActionBinding::CameraBack => actions.push(Action::CameraBack),
            ActionBinding::CameraLeft => actions.push(Action::CameraLeft),
            ActionBinding::CameraBottom => actions.push(Action::CameraBottom),
            ActionBinding::ToggleOrtho => actions.push(Action::ToggleOrtho),
            ActionBinding::FocusSelected => actions.push(Action::FocusSelected),
            ActionBinding::FrameAll => actions.push(Action::FrameAll),
            ActionBinding::GizmoTranslate => {
                actions.push(Action::SetGizmoMode(GizmoMode::Translate));
            }
            ActionBinding::GizmoRotate => {
                actions.push(Action::SetGizmoMode(GizmoMode::Rotate));
            }
            ActionBinding::GizmoScale => {
                actions.push(Action::SetGizmoMode(GizmoMode::Scale));
            }
            ActionBinding::ToggleGizmoSpace => actions.push(Action::ToggleGizmoSpace),
            ActionBinding::ResetPivot => actions.push(Action::ResetPivot),
            ActionBinding::EnterSculptMode => actions.push(Action::SetInteractionMode(
                InteractionMode::Sculpt(self.doc.sculpt_state.selected_brush()),
            )),
            ActionBinding::ToggleIsolation => actions.push(Action::ToggleIsolation),
            ActionBinding::CycleShadingMode => actions.push(Action::CycleShadingMode),
            ActionBinding::ToggleTurntable => actions.push(Action::ToggleTurntable),
            ActionBinding::ToggleMeasurementTool => actions.push(Action::SetInteractionMode(
                if self.ui.primary_shell.interaction_mode == InteractionMode::Measure {
                    InteractionMode::Select
                } else {
                    InteractionMode::Measure
                },
            )),
            ActionBinding::ShowExportDialog => actions.push(Action::ShowExportDialog),

            // --- Context-sensitive actions ---
            ActionBinding::Undo => actions.push(Action::Undo),
            ActionBinding::Redo => actions.push(Action::Redo),
            ActionBinding::ShowQuickToolbar => {
                if self.ui.primary_shell.tool_panel.is_docked() {
                    actions.push(Action::HideShellPanel(ShellPanelKind::Tool));
                } else {
                    self.ui.primary_shell.toggle_tool_panel();
                }
            }
            ActionBinding::ToggleReferenceImages => {
                actions.push(Action::ToggleAllReferenceImages);
            }
            ActionBinding::QuickSculptMode => {
                // Context already checked in phase 1
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    self.doc.sculpt_state.selected_brush(),
                )));
            }
            ActionBinding::ExitSculptMode => {
                // Context already checked in phase 1
                actions.push(Action::SetInteractionMode(InteractionMode::Select));
            }

            // --- Interaction rail shortcuts ---
            ActionBinding::SculptBrushAdd => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Add,
                )));
            }
            ActionBinding::SculptBrushCarve => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Carve,
                )));
            }
            ActionBinding::SculptBrushSmooth => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Smooth,
                )));
            }
            ActionBinding::SculptBrushFlatten => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Flatten,
                )));
            }
            ActionBinding::SculptBrushInflate => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Inflate,
                )));
            }
            ActionBinding::SculptBrushGrab => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    BrushMode::Grab,
                )));
            }
            ActionBinding::SculptBrushShrink => {
                if self.doc.sculpt_state.is_active() {
                    let brush = self.doc.sculpt_state.selected_profile_mut();
                    brush.radius = (brush.radius - 0.05).max(0.05);
                }
            }
            ActionBinding::SculptBrushGrow => {
                if self.doc.sculpt_state.is_active() {
                    let brush = self.doc.sculpt_state.selected_profile_mut();
                    brush.radius = (brush.radius + 0.05).min(2.0);
                }
            }

            // --- Sculpt symmetry toggles ---
            ActionBinding::SculptSymmetryX => {
                if self.doc.sculpt_state.is_active() {
                    let axis = self.doc.sculpt_state.symmetry_axis();
                    self.doc.sculpt_state.set_symmetry_axis(if axis == Some(0) {
                        None
                    } else {
                        Some(0)
                    });
                }
            }
            ActionBinding::SculptSymmetryY => {
                if self.doc.sculpt_state.is_active() {
                    let axis = self.doc.sculpt_state.symmetry_axis();
                    self.doc.sculpt_state.set_symmetry_axis(if axis == Some(1) {
                        None
                    } else {
                        Some(1)
                    });
                }
            }
            ActionBinding::SculptSymmetryZ => {
                if self.doc.sculpt_state.is_active() {
                    let axis = self.doc.sculpt_state.symmetry_axis();
                    self.doc.sculpt_state.set_symmetry_axis(if axis == Some(2) {
                        None
                    } else {
                        Some(2)
                    });
                }
            }
        }
    }
}
