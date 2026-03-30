use super::context::CallbackContext;
use crate::app::slint_frontend::{
    SlintHostWindow, ViewportPointerAction, ViewportPointerButton, ViewportPointerPhase,
};
use crate::keymap::KeyboardModifiers;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_viewport_pointer_action(move |action| {
            apply_viewport_input(&context, |viewport_input| {
                handle_pointer_action(viewport_input, action);
            });
        });
    }

    {
        let context = context.clone();
        window.on_viewport_scroll_action(move |action| {
            apply_viewport_input(&context, |viewport_input| {
                viewport_input.handle_scroll(
                    action.delta_x,
                    action.delta_y,
                    keyboard_modifiers(
                        action.modifiers.ctrl,
                        action.modifiers.shift,
                        action.modifiers.alt,
                    ),
                );
            });
        });
    }

    {
        let context = context.clone();
        window.on_viewport_double_click(move |action| {
            apply_viewport_input(&context, |viewport_input| {
                viewport_input.handle_double_click(action.x, action.y);
            });
        });
    }
}

fn apply_viewport_input<F>(context: &CallbackContext, mutate: F)
where
    F: FnOnce(&mut crate::app::slint_bridge::SlintViewportInputState),
{
    {
        let mut host_state = context.host.borrow_mut();
        mutate(&mut host_state.viewport_input);
        host_state.app.ui.node_graph_view.graph_keyboard_focus = false;
    }
    if let Some(window) = context.window_weak.upgrade() {
        super::super::drive_host_tick(&window, &context.host, &context.active_timer);
    }
}

fn handle_pointer_action(
    viewport_input: &mut crate::app::slint_bridge::SlintViewportInputState,
    action: ViewportPointerAction,
) {
    let modifiers = keyboard_modifiers(
        action.modifiers.ctrl,
        action.modifiers.shift,
        action.modifiers.alt,
    );
    match action.phase {
        ViewportPointerPhase::Down => viewport_input.handle_pointer_down(
            action.x,
            action.y,
            pointer_button_to_code(action.button, action.is_touch),
            modifiers,
            action.is_touch,
        ),
        ViewportPointerPhase::Up => viewport_input.handle_pointer_up(
            action.x,
            action.y,
            pointer_button_to_code(action.button, action.is_touch),
            modifiers,
            action.is_touch,
        ),
        ViewportPointerPhase::Cancel => viewport_input.handle_pointer_cancel(),
        ViewportPointerPhase::Move => {
            viewport_input.handle_pointer_move(action.x, action.y, modifiers, action.is_touch);
        }
    }
}

fn keyboard_modifiers(ctrl: bool, shift: bool, alt: bool) -> KeyboardModifiers {
    KeyboardModifiers { ctrl, shift, alt }
}

fn pointer_button_to_code(button: ViewportPointerButton, is_touch: bool) -> i32 {
    match button {
        ViewportPointerButton::Primary => crate::app::slint_bridge::POINTER_BUTTON_PRIMARY,
        ViewportPointerButton::Secondary => crate::app::slint_bridge::POINTER_BUTTON_SECONDARY,
        ViewportPointerButton::Middle => crate::app::slint_bridge::POINTER_BUTTON_MIDDLE,
        ViewportPointerButton::Other if is_touch => {
            crate::app::slint_bridge::POINTER_BUTTON_PRIMARY
        }
        ViewportPointerButton::Other => crate::app::slint_bridge::POINTER_BUTTON_OTHER,
    }
}

#[cfg(test)]
mod tests {
    use super::pointer_button_to_code;
    use crate::app::slint_frontend::ViewportPointerButton;

    #[test]
    fn touch_other_button_maps_to_primary_code() {
        assert_eq!(
            pointer_button_to_code(ViewportPointerButton::Other, true),
            crate::app::slint_bridge::POINTER_BUTTON_PRIMARY
        );
        assert_eq!(
            pointer_button_to_code(ViewportPointerButton::Other, false),
            crate::app::slint_bridge::POINTER_BUTTON_OTHER
        );
    }
}
