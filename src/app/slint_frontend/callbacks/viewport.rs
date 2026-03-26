use super::CallbackContext;
use crate::app::slint_frontend::SlintHostWindow;
use crate::keymap::KeyboardModifiers;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    {
        let context = context.clone();
        window.on_viewport_pointer_event(move |event, x, y| {
            let modifiers = KeyboardModifiers {
                ctrl: event.modifiers.control,
                shift: event.modifiers.shift,
                alt: event.modifiers.alt,
            };
            match format!("{:?}", event.kind).as_str() {
                "Down" => context
                    .host
                    .borrow_mut()
                    .viewport_input
                    .handle_pointer_down(
                        x,
                        y,
                        pointer_button_to_code(&event.button),
                        modifiers,
                        event.is_touch,
                    ),
                "Up" => context.host.borrow_mut().viewport_input.handle_pointer_up(
                    x,
                    y,
                    pointer_button_to_code(&event.button),
                    modifiers,
                    event.is_touch,
                ),
                "Cancel" => {
                    context
                        .host
                        .borrow_mut()
                        .viewport_input
                        .handle_pointer_cancel();
                }
                "Move" => context
                    .host
                    .borrow_mut()
                    .viewport_input
                    .handle_pointer_move(x, y, modifiers, event.is_touch),
                _ => {}
            }
            if let Some(window) = context.window_weak.upgrade() {
                super::super::drive_host_tick(&window, &context.host, &context.active_timer);
            }
        });
    }

    {
        let context = context.clone();
        window.on_viewport_scroll_event(move |event| {
            context.host.borrow_mut().viewport_input.handle_scroll(
                event.delta_x,
                event.delta_y,
                KeyboardModifiers {
                    ctrl: event.modifiers.control,
                    shift: event.modifiers.shift,
                    alt: event.modifiers.alt,
                },
            );
            if let Some(window) = context.window_weak.upgrade() {
                super::super::drive_host_tick(&window, &context.host, &context.active_timer);
            }
        });
    }

    {
        let context = context.clone();
        window.on_viewport_double_clicked(move |x, y| {
            context
                .host
                .borrow_mut()
                .viewport_input
                .handle_double_click(x, y);
            if let Some(window) = context.window_weak.upgrade() {
                super::super::drive_host_tick(&window, &context.host, &context.active_timer);
            }
        });
    }
}

fn pointer_button_to_code(button: &impl std::fmt::Debug) -> i32 {
    match format!("{button:?}").as_str() {
        "Left" => crate::app::slint_bridge::POINTER_BUTTON_PRIMARY,
        "Right" => crate::app::slint_bridge::POINTER_BUTTON_SECONDARY,
        "Middle" => crate::app::slint_bridge::POINTER_BUTTON_MIDDLE,
        _ => crate::app::slint_bridge::POINTER_BUTTON_OTHER,
    }
}
