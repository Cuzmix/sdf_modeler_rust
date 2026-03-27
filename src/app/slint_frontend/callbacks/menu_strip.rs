use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{
    MenuCommandAction, MenuKindView, SettingsCardAction, SlintHostWindow,
};
use crate::app::state::MenuDropdownKind;

enum MenuCommandDispatch {
    Queue(Action),
    StartExport,
}

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let launcher_context = context.clone();
    window.on_menu_launcher_action(move |kind| {
        mutate_host_and_tick(&launcher_context, move |host_state| {
            handle_menu_launcher_action(host_state, kind);
        });
    });

    let command_context = context.clone();
    window.on_menu_command_action(move |command| {
        mutate_host_and_tick(&command_context, move |host_state| {
            handle_menu_command_action(host_state, command);
        });
    });

    let settings_context = context.clone();
    window.on_settings_card_action(move |action, value| {
        mutate_host_and_tick(&settings_context, move |host_state| {
            handle_settings_card_action(host_state, action, value);
        });
    });

    let dismiss_context = context.clone();
    window.on_dismiss_menu_surfaces(move || {
        mutate_host_and_tick(&dismiss_context, move |host_state| {
            host_state.queue_action(Action::DismissMenuSurfaces);
        });
    });
}

fn handle_menu_launcher_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    kind: MenuKindView,
) {
    match kind {
        MenuKindView::Settings => {
            host_state.queue_action(Action::ToggleSettingsCard);
        }
        MenuKindView::File | MenuKindView::Edit | MenuKindView::View | MenuKindView::Help => {
            if let Some(menu_kind) = menu_dropdown_kind(kind) {
                host_state.queue_action(Action::ToggleMenuDropdown(menu_kind));
            }
        }
        MenuKindView::None => {}
    }
}

fn handle_menu_command_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    command: MenuCommandAction,
) {
    match map_menu_command(command) {
        MenuCommandDispatch::Queue(action) => host_state.queue_action(action),
        MenuCommandDispatch::StartExport => host_state.app.start_export(),
    }
    host_state.queue_action(Action::DismissMenuSurfaces);
}

fn handle_settings_card_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    action: SettingsCardAction,
    value: f32,
) {
    match action {
        SettingsCardAction::SetMultiAxisOrientation => {
            let mut next = host_state.app.settings.selection_behavior;
            next.multi_axis_orientation = if value >= 0.5 {
                crate::settings::MultiAxisOrientation::ActiveObject
            } else {
                crate::settings::MultiAxisOrientation::WorldZero
            };
            host_state.queue_action(Action::SetSelectionBehavior(next));
        }
        SettingsCardAction::SetGroupRotateDirection => {
            let mut next = host_state.app.settings.selection_behavior;
            next.group_rotate_direction = if value >= 0.5 {
                crate::settings::GroupRotateDirection::Inverted
            } else {
                crate::settings::GroupRotateDirection::Standard
            };
            host_state.queue_action(Action::SetSelectionBehavior(next));
        }
        SettingsCardAction::SetMultiPivotMode => {
            let mut next = host_state.app.settings.selection_behavior;
            next.multi_pivot_mode = if value >= 0.5 {
                crate::settings::MultiPivotMode::ActiveObject
            } else {
                crate::settings::MultiPivotMode::SelectionCenter
            };
            host_state.queue_action(Action::SetSelectionBehavior(next));
        }
        SettingsCardAction::SetAutoSaveEnabled => {
            host_state.queue_action(Action::SetAutoSaveEnabled(value >= 0.5));
        }
        SettingsCardAction::SetShowFpsOverlay => {
            host_state.queue_action(Action::SetShowFpsOverlay(value >= 0.5));
        }
        SettingsCardAction::SetContinuousRepaint => {
            host_state.queue_action(Action::SetContinuousRepaint(value >= 0.5));
        }
        SettingsCardAction::ExportSettings => {
            host_state.queue_action(Action::ExportSettings);
        }
        SettingsCardAction::ImportSettings => {
            host_state.queue_action(Action::ImportSettings);
        }
        SettingsCardAction::ResetPrimaryShellLayout => {
            host_state.queue_action(Action::ResetPrimaryShellLayout);
        }
    }
}

fn menu_dropdown_kind(kind: MenuKindView) -> Option<MenuDropdownKind> {
    match kind {
        MenuKindView::File => Some(MenuDropdownKind::File),
        MenuKindView::Edit => Some(MenuDropdownKind::Edit),
        MenuKindView::View => Some(MenuDropdownKind::View),
        MenuKindView::Help => Some(MenuDropdownKind::Help),
        MenuKindView::Settings | MenuKindView::None => None,
    }
}

fn map_menu_command(command: MenuCommandAction) -> MenuCommandDispatch {
    match command {
        MenuCommandAction::NewScene => MenuCommandDispatch::Queue(Action::NewScene),
        MenuCommandAction::OpenProject => MenuCommandDispatch::Queue(Action::OpenProject),
        MenuCommandAction::SaveProject => MenuCommandDispatch::Queue(Action::SaveProject),
        MenuCommandAction::ImportMesh => MenuCommandDispatch::Queue(Action::ImportMesh),
        MenuCommandAction::ExportMesh => MenuCommandDispatch::StartExport,
        MenuCommandAction::TakeScreenshot => MenuCommandDispatch::Queue(Action::TakeScreenshot),
        MenuCommandAction::AddReferenceImage => {
            MenuCommandDispatch::Queue(Action::AddReferenceImage)
        }
        MenuCommandAction::Undo => MenuCommandDispatch::Queue(Action::Undo),
        MenuCommandAction::Redo => MenuCommandDispatch::Queue(Action::Redo),
        MenuCommandAction::Copy => MenuCommandDispatch::Queue(Action::Copy),
        MenuCommandAction::Paste => MenuCommandDispatch::Queue(Action::Paste),
        MenuCommandAction::Duplicate => MenuCommandDispatch::Queue(Action::Duplicate),
        MenuCommandAction::DeleteSelected => MenuCommandDispatch::Queue(Action::DeleteSelected),
        MenuCommandAction::FrameAll => MenuCommandDispatch::Queue(Action::FrameAll),
        MenuCommandAction::FocusSelected => MenuCommandDispatch::Queue(Action::FocusSelected),
        MenuCommandAction::CameraFront => MenuCommandDispatch::Queue(Action::CameraFront),
        MenuCommandAction::CameraTop => MenuCommandDispatch::Queue(Action::CameraTop),
        MenuCommandAction::CameraRight => MenuCommandDispatch::Queue(Action::CameraRight),
        MenuCommandAction::ToggleOrtho => MenuCommandDispatch::Queue(Action::ToggleOrtho),
        MenuCommandAction::ToggleMeasurement => {
            MenuCommandDispatch::Queue(Action::ToggleMeasurementTool)
        }
        MenuCommandAction::ToggleTurntable => MenuCommandDispatch::Queue(Action::ToggleTurntable),
        MenuCommandAction::ToggleHelp => MenuCommandDispatch::Queue(Action::ToggleHelp),
        MenuCommandAction::ToggleCommandPalette => {
            MenuCommandDispatch::Queue(Action::ToggleCommandPalette)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{map_menu_command, menu_dropdown_kind, MenuCommandDispatch};
    use crate::app::actions::Action;
    use crate::app::slint_frontend::{MenuCommandAction, MenuKindView};
    use crate::app::state::MenuDropdownKind;

    #[test]
    fn dropdown_kind_mapping_is_defined_for_dropdown_launchers_only() {
        assert_eq!(
            menu_dropdown_kind(MenuKindView::File),
            Some(MenuDropdownKind::File)
        );
        assert_eq!(
            menu_dropdown_kind(MenuKindView::Edit),
            Some(MenuDropdownKind::Edit)
        );
        assert_eq!(
            menu_dropdown_kind(MenuKindView::View),
            Some(MenuDropdownKind::View)
        );
        assert_eq!(
            menu_dropdown_kind(MenuKindView::Help),
            Some(MenuDropdownKind::Help)
        );
        assert_eq!(menu_dropdown_kind(MenuKindView::Settings), None);
        assert_eq!(menu_dropdown_kind(MenuKindView::None), None);
    }

    #[test]
    fn menu_command_mapping_routes_export_as_host_start_export() {
        assert!(matches!(
            map_menu_command(MenuCommandAction::ExportMesh),
            MenuCommandDispatch::StartExport
        ));
    }

    #[test]
    fn menu_command_mapping_routes_standard_commands_to_actions() {
        assert!(matches!(
            map_menu_command(MenuCommandAction::SaveProject),
            MenuCommandDispatch::Queue(Action::SaveProject)
        ));
        assert!(matches!(
            map_menu_command(MenuCommandAction::ToggleCommandPalette),
            MenuCommandDispatch::Queue(Action::ToggleCommandPalette)
        ));
    }
}
