use slint::SharedString;

use super::context::CallbackContext;
use crate::app::actions::Action;
use crate::app::slint_frontend::SlintHostWindow;
use crate::keymap::{ActionBinding, KeyCombo, KeymapConfig, SerializableKey};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_keyboard_shortcut(move |text, ctrl, shift, alt| {
        handle_keyboard_shortcut(&context, text, ctrl, shift, alt)
    });
}

fn handle_keyboard_shortcut(
    context: &CallbackContext,
    text: SharedString,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> bool {
    let handled = {
        let mut host_state = context.host.borrow_mut();
        let handled = dispatch_shortcut_binding(&mut host_state, text.as_str(), ctrl, shift, alt);
        if handled {
            host_state.viewport_dirty = true;
        }
        handled
    };

    if handled {
        if let Some(window) = context.window_weak.upgrade() {
            super::super::drive_host_tick(&window, &context.host, &context.active_timer);
        }
    }
    handled
}

fn dispatch_shortcut_binding(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    key_text: &str,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> bool {
    let Some(binding) =
        resolve_binding_for_shortcut(key_text, ctrl, shift, alt, &host_state.app.settings.keymap)
    else {
        return false;
    };

    let mut actions = Vec::new();
    host_state.app.collect_binding_action(binding, &mut actions);
    for action in actions {
        host_state.queue_action(action);
    }

    if host_state.app.ui.menu.has_open_surface() {
        host_state.queue_action(Action::DismissMenuSurfaces);
    }

    true
}

fn resolve_binding_for_shortcut(
    key_text: &str,
    ctrl: bool,
    shift: bool,
    alt: bool,
    keymap: &KeymapConfig,
) -> Option<ActionBinding> {
    let key = parse_serializable_key(key_text)?;
    let combo = KeyCombo {
        key,
        ctrl,
        shift,
        alt,
    };
    keymap.find_action_for_combo(&combo)
}

fn parse_serializable_key(key_text: &str) -> Option<SerializableKey> {
    match key_text {
        " " => return Some(SerializableKey::Space),
        "\n" | "\r" | "\r\n" => return Some(SerializableKey::Enter),
        "\u{1b}" => return Some(SerializableKey::Escape),
        "\t" => return Some(SerializableKey::Tab),
        _ => {}
    }

    let normalized = key_text.trim();
    if normalized.is_empty() {
        return None;
    }

    match normalized {
        "Space" => return Some(SerializableKey::Space),
        "Enter" | "Return" => return Some(SerializableKey::Enter),
        "Escape" | "Esc" => return Some(SerializableKey::Escape),
        "Tab" => return Some(SerializableKey::Tab),
        "Delete" | "Del" => return Some(SerializableKey::Delete),
        "Home" => return Some(SerializableKey::Home),
        "End" => return Some(SerializableKey::End),
        "ArrowUp" | "UpArrow" | "Up" => return Some(SerializableKey::ArrowUp),
        "ArrowDown" | "DownArrow" | "Down" => return Some(SerializableKey::ArrowDown),
        "ArrowLeft" | "LeftArrow" | "Left" => return Some(SerializableKey::ArrowLeft),
        "ArrowRight" | "RightArrow" | "Right" => return Some(SerializableKey::ArrowRight),
        "[" | "{" | "OpenBracket" | "BracketLeft" => {
            return Some(SerializableKey::OpenBracket);
        }
        "]" | "}" | "CloseBracket" | "BracketRight" => {
            return Some(SerializableKey::CloseBracket);
        }
        "/" | "?" | "Slash" => return Some(SerializableKey::Slash),
        _ => {}
    }

    if let Some(function_key) = parse_function_key(normalized) {
        return Some(function_key);
    }

    if normalized.len() == 1 {
        let character = normalized.chars().next()?;
        if character.is_ascii_alphabetic() {
            return parse_letter_key(character);
        }
        if character.is_ascii_digit() {
            return parse_digit_key(character);
        }
    }

    None
}

fn parse_function_key(text: &str) -> Option<SerializableKey> {
    match text {
        "F1" => Some(SerializableKey::F1),
        "F2" => Some(SerializableKey::F2),
        "F3" => Some(SerializableKey::F3),
        "F4" => Some(SerializableKey::F4),
        "F5" => Some(SerializableKey::F5),
        "F6" => Some(SerializableKey::F6),
        "F7" => Some(SerializableKey::F7),
        "F8" => Some(SerializableKey::F8),
        "F9" => Some(SerializableKey::F9),
        "F10" => Some(SerializableKey::F10),
        "F11" => Some(SerializableKey::F11),
        "F12" => Some(SerializableKey::F12),
        _ => None,
    }
}

fn parse_letter_key(character: char) -> Option<SerializableKey> {
    match character.to_ascii_uppercase() {
        'A' => Some(SerializableKey::A),
        'B' => Some(SerializableKey::B),
        'C' => Some(SerializableKey::C),
        'D' => Some(SerializableKey::D),
        'E' => Some(SerializableKey::E),
        'F' => Some(SerializableKey::F),
        'G' => Some(SerializableKey::G),
        'H' => Some(SerializableKey::H),
        'I' => Some(SerializableKey::I),
        'J' => Some(SerializableKey::J),
        'K' => Some(SerializableKey::K),
        'L' => Some(SerializableKey::L),
        'M' => Some(SerializableKey::M),
        'N' => Some(SerializableKey::N),
        'O' => Some(SerializableKey::O),
        'P' => Some(SerializableKey::P),
        'Q' => Some(SerializableKey::Q),
        'R' => Some(SerializableKey::R),
        'S' => Some(SerializableKey::S),
        'T' => Some(SerializableKey::T),
        'U' => Some(SerializableKey::U),
        'V' => Some(SerializableKey::V),
        'W' => Some(SerializableKey::W),
        'X' => Some(SerializableKey::X),
        'Y' => Some(SerializableKey::Y),
        'Z' => Some(SerializableKey::Z),
        _ => None,
    }
}

fn parse_digit_key(character: char) -> Option<SerializableKey> {
    match character {
        '0' => Some(SerializableKey::Num0),
        '1' => Some(SerializableKey::Num1),
        '2' => Some(SerializableKey::Num2),
        '3' => Some(SerializableKey::Num3),
        '4' => Some(SerializableKey::Num4),
        '5' => Some(SerializableKey::Num5),
        '6' => Some(SerializableKey::Num6),
        '7' => Some(SerializableKey::Num7),
        '8' => Some(SerializableKey::Num8),
        '9' => Some(SerializableKey::Num9),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_serializable_key, resolve_binding_for_shortcut};
    use crate::keymap::{ActionBinding, KeymapConfig, SerializableKey};

    #[test]
    fn parse_serializable_key_maps_expected_aliases() {
        assert_eq!(parse_serializable_key("a"), Some(SerializableKey::A));
        assert_eq!(parse_serializable_key("F10"), Some(SerializableKey::F10));
        assert_eq!(
            parse_serializable_key("UpArrow"),
            Some(SerializableKey::ArrowUp)
        );
        assert_eq!(parse_serializable_key("?"), Some(SerializableKey::Slash));
        assert_eq!(parse_serializable_key("{"), Some(SerializableKey::OpenBracket));
        assert_eq!(parse_serializable_key(" "), Some(SerializableKey::Space));
        assert_eq!(parse_serializable_key("\u{1b}"), Some(SerializableKey::Escape));
    }

    #[test]
    fn resolve_binding_for_shortcut_uses_keymap_combo_lookup() {
        let keymap = KeymapConfig::default();
        assert_eq!(
            resolve_binding_for_shortcut("s", true, false, false, &keymap),
            Some(ActionBinding::SaveProject)
        );
        assert_eq!(
            resolve_binding_for_shortcut("C", true, true, false, &keymap),
            Some(ActionBinding::CopyProperties)
        );
    }

    #[test]
    fn resolve_binding_for_shortcut_returns_none_for_unbound_combo() {
        let keymap = KeymapConfig::default();
        assert_eq!(
            resolve_binding_for_shortcut("q", false, false, false, &keymap),
            None
        );
        assert_eq!(
            resolve_binding_for_shortcut("UnknownKey", false, false, false, &keymap),
            None
        );
    }
}
