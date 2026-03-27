use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{
    MenuCommandKind, MenuDropdownModel, MenuStripKind, ShellSnapshot,
};
use crate::app::slint_frontend::{
    GroupRotateDirectionView, MenuCommandAction, MenuDropdownItemView, MenuDropdownState,
    MenuKindView, MenuStripItemView, MenuStripState, MultiAxisOrientationView, MultiPivotModeView,
    SettingsCardState,
};

pub(super) fn build_menu_strip_state(snapshot: &ShellSnapshot) -> MenuStripState {
    MenuStripState {
        visible: snapshot.menu_strip.visible,
        items: Rc::new(VecModel::from(
            snapshot
                .menu_strip
                .items
                .iter()
                .map(|item| MenuStripItemView {
                    label: item.label.clone().into(),
                    kind: menu_strip_kind_view(item.kind),
                    active: item.active,
                    focused: item.focused,
                })
                .collect::<Vec<_>>(),
        ))
        .into(),
    }
}

pub(super) fn build_menu_dropdown_state(snapshot: &ShellSnapshot) -> MenuDropdownState {
    menu_dropdown_state(&snapshot.menu_dropdown)
}

pub(super) fn build_settings_card_state(snapshot: &ShellSnapshot) -> SettingsCardState {
    SettingsCardState {
        visible: snapshot.settings_card.visible,
        multi_axis_orientation: match snapshot.settings_card.multi_axis_orientation {
            crate::settings::MultiAxisOrientation::WorldZero => MultiAxisOrientationView::WorldZero,
            crate::settings::MultiAxisOrientation::ActiveObject => {
                MultiAxisOrientationView::ActiveObject
            }
        },
        group_rotate_direction: match snapshot.settings_card.group_rotate_direction {
            crate::settings::GroupRotateDirection::Standard => GroupRotateDirectionView::Standard,
            crate::settings::GroupRotateDirection::Inverted => GroupRotateDirectionView::Inverted,
        },
        multi_pivot_mode: match snapshot.settings_card.multi_pivot_mode {
            crate::settings::MultiPivotMode::SelectionCenter => MultiPivotModeView::SelectionCenter,
            crate::settings::MultiPivotMode::ActiveObject => MultiPivotModeView::ActiveObject,
        },
        auto_save_enabled: snapshot.settings_card.auto_save_enabled,
        show_fps_overlay: snapshot.settings_card.show_fps_overlay,
        continuous_repaint: snapshot.settings_card.continuous_repaint,
    }
}

fn menu_dropdown_state(model: &MenuDropdownModel) -> MenuDropdownState {
    MenuDropdownState {
        visible: model.visible,
        title: model.title.clone().into(),
        menu: model
            .kind
            .map(menu_dropdown_kind_view)
            .unwrap_or(MenuKindView::None),
        anchor_index: model.anchor_index,
        highlighted_index: model.highlighted_index,
        items: Rc::new(VecModel::from(
            model
                .items
                .iter()
                .map(|item| MenuDropdownItemView {
                    label: item.label.clone().into(),
                    shortcut_label: item.shortcut_label.clone().into(),
                    command: menu_command_view(item.command),
                    enabled: item.enabled,
                    checked: item.checked,
                })
                .collect::<Vec<_>>(),
        ))
        .into(),
    }
}

fn menu_strip_kind_view(kind: MenuStripKind) -> MenuKindView {
    match kind {
        MenuStripKind::File => MenuKindView::File,
        MenuStripKind::Edit => MenuKindView::Edit,
        MenuStripKind::View => MenuKindView::View,
        MenuStripKind::Settings => MenuKindView::Settings,
        MenuStripKind::Help => MenuKindView::Help,
    }
}

fn menu_dropdown_kind_view(kind: crate::app::state::MenuDropdownKind) -> MenuKindView {
    match kind {
        crate::app::state::MenuDropdownKind::File => MenuKindView::File,
        crate::app::state::MenuDropdownKind::Edit => MenuKindView::Edit,
        crate::app::state::MenuDropdownKind::View => MenuKindView::View,
        crate::app::state::MenuDropdownKind::Help => MenuKindView::Help,
    }
}

fn menu_command_view(command: MenuCommandKind) -> MenuCommandAction {
    match command {
        MenuCommandKind::NewScene => MenuCommandAction::NewScene,
        MenuCommandKind::OpenProject => MenuCommandAction::OpenProject,
        MenuCommandKind::SaveProject => MenuCommandAction::SaveProject,
        MenuCommandKind::ImportMesh => MenuCommandAction::ImportMesh,
        MenuCommandKind::ExportMesh => MenuCommandAction::ExportMesh,
        MenuCommandKind::TakeScreenshot => MenuCommandAction::TakeScreenshot,
        MenuCommandKind::AddReferenceImage => MenuCommandAction::AddReferenceImage,
        MenuCommandKind::Undo => MenuCommandAction::Undo,
        MenuCommandKind::Redo => MenuCommandAction::Redo,
        MenuCommandKind::Copy => MenuCommandAction::Copy,
        MenuCommandKind::Paste => MenuCommandAction::Paste,
        MenuCommandKind::Duplicate => MenuCommandAction::Duplicate,
        MenuCommandKind::DeleteSelected => MenuCommandAction::DeleteSelected,
        MenuCommandKind::FrameAll => MenuCommandAction::FrameAll,
        MenuCommandKind::FocusSelected => MenuCommandAction::FocusSelected,
        MenuCommandKind::CameraFront => MenuCommandAction::CameraFront,
        MenuCommandKind::CameraTop => MenuCommandAction::CameraTop,
        MenuCommandKind::CameraRight => MenuCommandAction::CameraRight,
        MenuCommandKind::ToggleOrtho => MenuCommandAction::ToggleOrtho,
        MenuCommandKind::ToggleMeasure => MenuCommandAction::ToggleMeasurement,
        MenuCommandKind::ToggleTurntable => MenuCommandAction::ToggleTurntable,
        MenuCommandKind::ToggleHelp => MenuCommandAction::ToggleHelp,
        MenuCommandKind::ToggleCommandPalette => MenuCommandAction::ToggleCommandPalette,
    }
}
