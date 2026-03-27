use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::frontend_models::{
    menu_commands_for_kind, MenuCommandAvailability, MenuCommandCheckState, MenuCommandKind,
    MenuCommandModel,
};
use crate::app::slint_frontend::{
    MenuCommandAction, MenuKindView, MenuNavigationAction, SettingsCardAction, SlintHostWindow,
};
use crate::app::{BakeStatus, ExportStatus, ImportStatus};
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

    let navigation_context = context.clone();
    window.on_menu_navigation_action(move |action| {
        mutate_host_and_tick(&navigation_context, move |host_state| {
            handle_menu_navigation_action(host_state, action);
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
    dispatch_menu_command(host_state, map_menu_command(command));
    host_state.queue_action(Action::DismissMenuSurfaces);
}

fn handle_menu_navigation_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    action: MenuNavigationAction,
) {
    if matches!(action, MenuNavigationAction::Dismiss) {
        host_state.queue_action(Action::DismissMenuSurfaces);
        return;
    }

    let Some(kind) = host_state.app.ui.menu.active_dropdown else {
        return;
    };
    let items = menu_commands_for_kind(
        kind,
        file_actions_enabled(host_state),
        &host_state.app.settings,
        menu_command_checks(host_state),
        menu_command_availability(host_state),
    );
    let current_index = host_state.app.ui.menu.highlighted_command_index;

    match action {
        MenuNavigationAction::Dismiss => {}
        MenuNavigationAction::SelectNext => {
            let next = step_enabled_index(&items, current_index, true);
            host_state.queue_action(Action::SetMenuHighlightedIndex(next));
        }
        MenuNavigationAction::SelectPrevious => {
            let previous = step_enabled_index(&items, current_index, false);
            host_state.queue_action(Action::SetMenuHighlightedIndex(previous));
        }
        MenuNavigationAction::SelectNextMenu => {
            host_state.queue_action(Action::OpenMenuDropdown(step_menu_kind(kind, true)));
        }
        MenuNavigationAction::SelectPreviousMenu => {
            host_state.queue_action(Action::OpenMenuDropdown(step_menu_kind(kind, false)));
        }
        MenuNavigationAction::ActivateSelected => {
            if let Some(command) = active_enabled_command(&items, current_index) {
                dispatch_menu_command(host_state, map_menu_command_kind(command));
                host_state.queue_action(Action::DismissMenuSurfaces);
            }
        }
    }
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

fn file_actions_enabled(host_state: &crate::app::slint_frontend::host_state::SlintHostState) -> bool {
    !matches!(
        host_state.app.async_state.bake_status,
        BakeStatus::InProgress { .. }
    ) && !matches!(
        host_state.app.async_state.export_status,
        ExportStatus::InProgress { .. }
    ) && !matches!(
        host_state.app.async_state.import_status,
        ImportStatus::InProgress { .. }
    )
}

fn dispatch_menu_command(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    dispatch: MenuCommandDispatch,
) {
    match dispatch {
        MenuCommandDispatch::Queue(action) => host_state.queue_action(action),
        MenuCommandDispatch::StartExport => host_state.app.start_export(),
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

fn map_menu_command_kind(command: MenuCommandKind) -> MenuCommandDispatch {
    match command {
        MenuCommandKind::NewScene => MenuCommandDispatch::Queue(Action::NewScene),
        MenuCommandKind::OpenProject => MenuCommandDispatch::Queue(Action::OpenProject),
        MenuCommandKind::SaveProject => MenuCommandDispatch::Queue(Action::SaveProject),
        MenuCommandKind::ImportMesh => MenuCommandDispatch::Queue(Action::ImportMesh),
        MenuCommandKind::ExportMesh => MenuCommandDispatch::StartExport,
        MenuCommandKind::TakeScreenshot => MenuCommandDispatch::Queue(Action::TakeScreenshot),
        MenuCommandKind::AddReferenceImage => MenuCommandDispatch::Queue(Action::AddReferenceImage),
        MenuCommandKind::Undo => MenuCommandDispatch::Queue(Action::Undo),
        MenuCommandKind::Redo => MenuCommandDispatch::Queue(Action::Redo),
        MenuCommandKind::Copy => MenuCommandDispatch::Queue(Action::Copy),
        MenuCommandKind::Paste => MenuCommandDispatch::Queue(Action::Paste),
        MenuCommandKind::Duplicate => MenuCommandDispatch::Queue(Action::Duplicate),
        MenuCommandKind::DeleteSelected => MenuCommandDispatch::Queue(Action::DeleteSelected),
        MenuCommandKind::FrameAll => MenuCommandDispatch::Queue(Action::FrameAll),
        MenuCommandKind::FocusSelected => MenuCommandDispatch::Queue(Action::FocusSelected),
        MenuCommandKind::CameraFront => MenuCommandDispatch::Queue(Action::CameraFront),
        MenuCommandKind::CameraTop => MenuCommandDispatch::Queue(Action::CameraTop),
        MenuCommandKind::CameraRight => MenuCommandDispatch::Queue(Action::CameraRight),
        MenuCommandKind::ToggleOrtho => MenuCommandDispatch::Queue(Action::ToggleOrtho),
        MenuCommandKind::ToggleMeasure => MenuCommandDispatch::Queue(Action::ToggleMeasurementTool),
        MenuCommandKind::ToggleTurntable => MenuCommandDispatch::Queue(Action::ToggleTurntable),
        MenuCommandKind::ToggleHelp => MenuCommandDispatch::Queue(Action::ToggleHelp),
        MenuCommandKind::ToggleCommandPalette => {
            MenuCommandDispatch::Queue(Action::ToggleCommandPalette)
        }
    }
}

fn step_enabled_index(
    items: &[MenuCommandModel],
    current_index: Option<usize>,
    forward: bool,
) -> Option<usize> {
    let enabled_indices = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| item.enabled.then_some(index))
        .collect::<Vec<_>>();
    if enabled_indices.is_empty() {
        return None;
    }

    let current_slot = current_index
        .and_then(|index| enabled_indices.iter().position(|candidate| *candidate == index));
    let next_slot = match current_slot {
        Some(slot) if forward => (slot + 1) % enabled_indices.len(),
        Some(0) => enabled_indices.len() - 1,
        Some(slot) => slot - 1,
        None if forward => 0,
        None => enabled_indices.len() - 1,
    };
    Some(enabled_indices[next_slot])
}

fn active_enabled_command(
    items: &[MenuCommandModel],
    current_index: Option<usize>,
) -> Option<MenuCommandKind> {
    let active_index = current_index
        .filter(|index| items.get(*index).is_some_and(|item| item.enabled))
        .or_else(|| items.iter().position(|item| item.enabled))?;
    items.get(active_index).map(|item| item.command)
}

fn menu_command_checks(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
) -> MenuCommandCheckState {
    MenuCommandCheckState {
        ortho_enabled: host_state.app.doc.camera.orthographic,
        measurement_enabled: host_state.app.ui.measurement_mode,
        turntable_enabled: host_state.app.ui.turntable_active,
        help_visible: host_state.app.ui.show_help,
        command_palette_visible: host_state.app.ui.command_palette_open,
    }
}

fn menu_command_availability(
    host_state: &crate::app::slint_frontend::host_state::SlintHostState,
) -> MenuCommandAvailability {
    let has_selection = !host_state.app.ui.selection.selected_set.is_empty();
    MenuCommandAvailability {
        undo_enabled: host_state.app.doc.history.undo_count() > 0,
        redo_enabled: host_state.app.doc.history.redo_count() > 0,
        copy_enabled: has_selection,
        paste_enabled: host_state.app.doc.clipboard_node.is_some(),
        duplicate_enabled: has_selection,
        delete_enabled: has_selection,
        focus_selected_enabled: has_selection,
    }
}

fn step_menu_kind(kind: MenuDropdownKind, forward: bool) -> MenuDropdownKind {
    const ORDER: [MenuDropdownKind; 4] = [
        MenuDropdownKind::File,
        MenuDropdownKind::Edit,
        MenuDropdownKind::View,
        MenuDropdownKind::Help,
    ];
    let current_index = ORDER.iter().position(|candidate| *candidate == kind).unwrap_or(0);
    let offset = if forward { 1 } else { ORDER.len() - 1 };
    ORDER[(current_index + offset) % ORDER.len()]
}

#[cfg(test)]
mod tests {
    use super::{
        active_enabled_command, map_menu_command, map_menu_command_kind, menu_dropdown_kind,
        step_enabled_index, step_menu_kind, MenuCommandDispatch,
    };
    use crate::app::actions::Action;
    use crate::app::frontend_models::MenuCommandKind;
    use crate::app::slint_frontend::{MenuCommandAction, MenuKindView};
    use crate::app::state::MenuDropdownKind;

    fn command_item(command: MenuCommandKind, enabled: bool) -> crate::app::frontend_models::MenuCommandModel {
        crate::app::frontend_models::MenuCommandModel {
            command,
            label: format!("{command:?}"),
            shortcut_label: String::new(),
            enabled,
            checked: false,
        }
    }

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

    #[test]
    fn menu_command_kind_mapping_matches_command_action_mapping() {
        assert!(matches!(
            map_menu_command_kind(MenuCommandKind::SaveProject),
            MenuCommandDispatch::Queue(Action::SaveProject)
        ));
        assert!(matches!(
            map_menu_command_kind(MenuCommandKind::ExportMesh),
            MenuCommandDispatch::StartExport
        ));
    }

    #[test]
    fn step_enabled_index_skips_disabled_and_wraps_forward() {
        let items = vec![
            command_item(MenuCommandKind::Undo, true),
            command_item(MenuCommandKind::Redo, false),
            command_item(MenuCommandKind::Copy, true),
        ];

        assert_eq!(step_enabled_index(&items, Some(0), true), Some(2));
        assert_eq!(step_enabled_index(&items, Some(2), true), Some(0));
    }

    #[test]
    fn step_enabled_index_wraps_backward() {
        let items = vec![
            command_item(MenuCommandKind::Undo, true),
            command_item(MenuCommandKind::Redo, false),
            command_item(MenuCommandKind::Copy, true),
        ];

        assert_eq!(step_enabled_index(&items, Some(0), false), Some(2));
        assert_eq!(step_enabled_index(&items, None, false), Some(2));
    }

    #[test]
    fn active_enabled_command_falls_back_to_first_enabled_item() {
        let items = vec![
            command_item(MenuCommandKind::Undo, false),
            command_item(MenuCommandKind::Redo, true),
            command_item(MenuCommandKind::Copy, true),
        ];

        assert_eq!(
            active_enabled_command(&items, Some(0)),
            Some(MenuCommandKind::Redo)
        );
        assert_eq!(
            active_enabled_command(&items, Some(2)),
            Some(MenuCommandKind::Copy)
        );
    }

    #[test]
    fn step_menu_kind_wraps_in_both_directions() {
        assert_eq!(
            step_menu_kind(MenuDropdownKind::File, false),
            MenuDropdownKind::Help
        );
        assert_eq!(
            step_menu_kind(MenuDropdownKind::Help, true),
            MenuDropdownKind::File
        );
        assert_eq!(
            step_menu_kind(MenuDropdownKind::Edit, true),
            MenuDropdownKind::View
        );
    }
}
