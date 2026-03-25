use crate::keymap::ActionBinding;
use crate::sculpt::BrushMode;
use crate::ui::gizmo::GizmoMode;

use super::actions::{Action, ActionSink};
use super::state::InteractionMode;
use super::SdfApp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TriggeredInput {
    Binding(ActionBinding),
    SaveBookmark(usize),
}

impl SdfApp {
    pub(super) fn delete_selected(&mut self) {
        if let Some(sel) = self.ui.selection.selected {
            self.doc.scene.remove_node(sel);
            self.ui.selection.clear_selection();
            self.ui.scene_graph_view.needs_initial_rebuild = true;
            self.doc.sculpt_state = crate::sculpt::SculptState::new_inactive();
            self.sync_interaction_mode_after_sculpt_exit();
            self.gpu.buffer_dirty = true;
        }
    }

    /// Collect keyboard-triggered actions into the action sink.
    /// Frontend bridges decode toolkit input into `TriggeredInput` values first.
    pub(super) fn collect_keyboard_actions(
        &mut self,
        triggered_inputs: Vec<TriggeredInput>,
        actions: &mut ActionSink,
    ) {
        for triggered_input in triggered_inputs {
            match triggered_input {
                TriggeredInput::Binding(binding) => self.dispatch_binding(binding, actions),
                TriggeredInput::SaveBookmark(index) => actions.push(Action::SaveBookmark(index)),
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
                self.ui.primary_shell.toggle_tool_rail();
            }
            ActionBinding::ToggleReferenceImages => {
                actions.push(Action::ToggleAllReferenceImages);
            }
            ActionBinding::QuickSculptMode => {
                actions.push(Action::SetInteractionMode(InteractionMode::Sculpt(
                    self.doc.sculpt_state.selected_brush(),
                )));
            }
            ActionBinding::ExitSculptMode => {
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
