use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{SlintHostWindow, ToolPaletteAction};
use crate::app::state::InteractionMode;
use crate::sculpt::BrushMode;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_tool_palette_action(move |action| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_tool_palette_action(host_state, action);
        });
    });
}

fn handle_tool_palette_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    action: ToolPaletteAction,
) {
    let next_mode = match action {
        ToolPaletteAction::Select => InteractionMode::Select,
        ToolPaletteAction::BrushAdd => InteractionMode::Sculpt(BrushMode::Add),
        ToolPaletteAction::BrushCarve => InteractionMode::Sculpt(BrushMode::Carve),
        ToolPaletteAction::BrushSmooth => InteractionMode::Sculpt(BrushMode::Smooth),
        ToolPaletteAction::BrushFlatten => InteractionMode::Sculpt(BrushMode::Flatten),
        ToolPaletteAction::BrushInflate => InteractionMode::Sculpt(BrushMode::Inflate),
        ToolPaletteAction::BrushGrab => InteractionMode::Sculpt(BrushMode::Grab),
    };

    if host_state.app.ui.primary_shell.interaction_mode != next_mode {
        host_state.queue_action(Action::SetInteractionMode(next_mode));
    }
}
