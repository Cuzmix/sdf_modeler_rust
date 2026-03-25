use crate::egui_keymap::{key_to_egui, modifiers_from_egui, modifiers_to_egui};
use crate::keymap::{ActionBinding, KeyboardModifiers, KeymapConfig, SerializableKey};
use crate::sculpt::ActiveTool;

use super::backend_frame::{FrameCommands, FrameInputSnapshot};
use super::input::TriggeredInput;

#[derive(Clone, Copy, Debug)]
pub(super) struct KeyboardActionContext {
    pub is_sculpt: bool,
    pub active_tool: ActiveTool,
    pub export_idle: bool,
}

pub(super) fn capture_frame_input(ctx: &egui::Context) -> FrameInputSnapshot {
    FrameInputSnapshot {
        now_seconds: ctx.input(|input_state| input_state.time),
        pointer_primary_down: ctx.input(|input_state| input_state.pointer.primary_down()),
        is_dragging_ui: ctx.dragged_id().is_some(),
    }
}

pub(super) fn collect_triggered_inputs(
    ctx: &egui::Context,
    keymap: &KeymapConfig,
    keyboard_context: KeyboardActionContext,
) -> Vec<TriggeredInput> {
    let mut triggered_inputs = Vec::new();

    for (&binding, combo) in keymap.bindings() {
        if binding.is_sculpt_only() && !keyboard_context.is_sculpt {
            continue;
        }
        if keyboard_context.is_sculpt && matches!(binding, ActionBinding::CycleShadingMode) {
            continue;
        }
        if keyboard_context.is_sculpt && matches!(binding, ActionBinding::FocusSelected) {
            continue;
        }
        if matches!(binding, ActionBinding::ToggleTurntable) && ctx.wants_keyboard_input() {
            continue;
        }
        if matches!(binding, ActionBinding::ShowExportDialog) && !keyboard_context.export_idle {
            continue;
        }
        if matches!(binding, ActionBinding::QuickSculptMode)
            && keyboard_context.active_tool != ActiveTool::Select
        {
            continue;
        }
        if matches!(binding, ActionBinding::ExitSculptMode)
            && keyboard_context.active_tool != ActiveTool::Sculpt
        {
            continue;
        }

        let key = key_to_egui(combo.key);
        let modifiers = KeyboardModifiers {
            ctrl: combo.ctrl,
            shift: combo.shift,
            alt: combo.alt,
        };
        let pressed = match binding {
            ActionBinding::Copy
            | ActionBinding::Paste
            | ActionBinding::Duplicate
            | ActionBinding::ToggleCommandPalette => {
                ctx.input_mut(|input| input.consume_key(modifiers_to_egui(modifiers), key))
            }
            _ => ctx.input(|input| {
                input.key_pressed(key)
                    && combo.matches_modifiers(modifiers_from_egui(&input.modifiers))
            }),
        };

        if pressed {
            triggered_inputs.push(TriggeredInput::Binding(binding));
        }
    }

    if !keyboard_context.is_sculpt {
        for (index, bookmark_key) in [
            SerializableKey::Num1,
            SerializableKey::Num2,
            SerializableKey::Num3,
            SerializableKey::Num4,
            SerializableKey::Num5,
            SerializableKey::Num6,
            SerializableKey::Num7,
            SerializableKey::Num8,
            SerializableKey::Num9,
        ]
        .into_iter()
        .enumerate()
        {
            let pressed = ctx.input(|input| {
                let modifiers = modifiers_from_egui(&input.modifiers);
                modifiers.ctrl
                    && !modifiers.shift
                    && !modifiers.alt
                    && input.key_pressed(key_to_egui(bookmark_key))
            });
            if pressed {
                triggered_inputs.push(TriggeredInput::SaveBookmark(index));
            }
        }
    }

    triggered_inputs
}

pub(super) fn apply_frame_commands(ctx: &egui::Context, commands: &FrameCommands) {
    if let Some(title) = commands.window_title.as_deref() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.into()));
    }
    if commands.request_repaint {
        ctx.request_repaint();
    }
}
