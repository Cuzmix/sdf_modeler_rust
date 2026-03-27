use slint::SharedString;

use super::context::CallbackContext;
use crate::app::actions::Action;
use crate::app::slint_frontend::SlintHostWindow;
use crate::app::state::MenuDropdownKind;
use crate::keymap::{ActionBinding, KeyCombo, KeymapConfig, SerializableKey};

enum OpenMenuShortcutIntent {
    Dismiss,
    Launch(Action),
    Consume,
}

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
    if host_state.app.ui.menu.has_open_surface() {
        match resolve_open_menu_shortcut_intent(key_text, ctrl, shift, alt) {
            OpenMenuShortcutIntent::Dismiss => host_state.queue_action(Action::DismissMenuSurfaces),
            OpenMenuShortcutIntent::Launch(action) => host_state.queue_action(action),
            OpenMenuShortcutIntent::Consume => {}
        }
        return true;
    }

    if let Some(binding) =
        resolve_binding_for_shortcut(key_text, ctrl, shift, alt, &host_state.app.settings.keymap)
    {
        let mut actions = Vec::new();
        host_state.app.collect_binding_action(binding, &mut actions);
        for action in actions {
            host_state.queue_action(action);
        }

        if host_state.app.ui.menu.has_open_surface() {
            host_state.queue_action(Action::DismissMenuSurfaces);
        }
        return true;
    }

    if let Some(action) = resolve_menu_launcher_shortcut(key_text, ctrl, shift, alt) {
        host_state.queue_action(action);
        return true;
    }

    false
}

fn resolve_open_menu_shortcut_intent(
    key_text: &str,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> OpenMenuShortcutIntent {
    if parse_serializable_key(key_text) == Some(SerializableKey::Escape) {
        return OpenMenuShortcutIntent::Dismiss;
    }
    if let Some(action) = resolve_menu_launcher_shortcut(key_text, ctrl, shift, alt) {
        return OpenMenuShortcutIntent::Launch(action);
    }
    OpenMenuShortcutIntent::Consume
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

fn resolve_menu_launcher_shortcut(
    key_text: &str,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> Option<Action> {
    if ctrl || shift || !alt {
        return None;
    }

    let normalized = key_text.trim();
    if normalized.len() != 1 {
        return None;
    }

    let key = normalized.chars().next()?.to_ascii_uppercase();
    match key {
        'F' => Some(Action::ToggleMenuDropdown(MenuDropdownKind::File)),
        'E' => Some(Action::ToggleMenuDropdown(MenuDropdownKind::Edit)),
        'V' => Some(Action::ToggleMenuDropdown(MenuDropdownKind::View)),
        'S' => Some(Action::ToggleSettingsCard),
        'H' => Some(Action::ToggleMenuDropdown(MenuDropdownKind::Help)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_serializable_key, resolve_binding_for_shortcut, resolve_menu_launcher_shortcut,
        resolve_open_menu_shortcut_intent, OpenMenuShortcutIntent,
    };
    use crate::app::actions::Action;
    use crate::app::state::MenuDropdownKind;
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
        assert_eq!(
            parse_serializable_key("{"),
            Some(SerializableKey::OpenBracket)
        );
        assert_eq!(parse_serializable_key(" "), Some(SerializableKey::Space));
        assert_eq!(
            parse_serializable_key("\u{1b}"),
            Some(SerializableKey::Escape)
        );
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

    #[test]
    fn resolve_menu_launcher_shortcut_maps_alt_letter_combos() {
        assert!(matches!(
            resolve_menu_launcher_shortcut("f", false, false, true),
            Some(Action::ToggleMenuDropdown(MenuDropdownKind::File))
        ));
        assert!(matches!(
            resolve_menu_launcher_shortcut("S", false, false, true),
            Some(Action::ToggleSettingsCard)
        ));
        assert!(matches!(
            resolve_menu_launcher_shortcut("h", false, false, true),
            Some(Action::ToggleMenuDropdown(MenuDropdownKind::Help))
        ));
    }

    #[test]
    fn resolve_menu_launcher_shortcut_ignores_non_alt_or_modified_combos() {
        assert!(resolve_menu_launcher_shortcut("f", false, false, false).is_none());
        assert!(resolve_menu_launcher_shortcut("f", true, false, true).is_none());
        assert!(resolve_menu_launcher_shortcut("F10", false, false, true).is_none());
    }

    #[test]
    fn resolve_open_menu_shortcut_intent_prioritizes_escape() {
        assert!(matches!(
            resolve_open_menu_shortcut_intent("\u{1b}", false, false, false),
            OpenMenuShortcutIntent::Dismiss
        ));
    }

    #[test]
    fn resolve_open_menu_shortcut_intent_allows_alt_launcher_shortcuts() {
        assert!(matches!(
            resolve_open_menu_shortcut_intent("h", false, false, true),
            OpenMenuShortcutIntent::Launch(Action::ToggleMenuDropdown(MenuDropdownKind::Help))
        ));
    }

    #[test]
    fn resolve_open_menu_shortcut_intent_consumes_unhandled_shortcuts() {
        assert!(matches!(
            resolve_open_menu_shortcut_intent("z", true, false, false),
            OpenMenuShortcutIntent::Consume
        ));
    }
}
