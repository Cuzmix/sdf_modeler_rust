use std::collections::HashSet;

use crate::settings::{
    MenuLauncherPreference, PanelKindPreference, PinnedPanelPreference, ShellChromeSettings,
    ShellFloatingRect,
};

use super::state::{
    MenuLauncherKind, MenuUiState, PanelBarId, PanelFrameworkState, PanelInstanceState, PanelKind,
    PanelPresentation,
};
use super::ui_geometry::FloatingPanelBounds;

pub(crate) fn apply(
    menu: &mut MenuUiState,
    panel_framework: &mut PanelFrameworkState,
    prefs: &ShellChromeSettings,
) {
    menu.strip_visible = prefs.menu_strip_visible;
    menu.focused_launcher = prefs
        .menu_focused_launcher
        .map(menu_launcher_from_preference);
    menu.active_dropdown = None;
    menu.settings_card_open = false;
    menu.highlighted_command_index = None;

    *panel_framework = PanelFrameworkState::default();
    if let Some(bar) = panel_framework.bar_mut(PanelBarId::PrimaryRight) {
        bar.transient_rect = prefs
            .primary_transient_rect
            .and_then(bounds_from_settings_rect);
    }

    let mut next_instance_id = panel_framework.next_instance_id;
    let mut seen_kinds = HashSet::new();
    let mut restored_instances = Vec::new();
    for entry in &prefs.pinned_panels {
        let kind = panel_kind_from_preference(entry.kind);
        if !seen_kinds.insert(kind) {
            continue;
        }
        restored_instances.push(PanelInstanceState {
            id: next_instance_id,
            kind,
            presentation: PanelPresentation::PinnedFloating,
            pinned: true,
            anchor_bar: PanelBarId::PrimaryRight,
            visible: true,
            collapsed: entry.collapsed,
            rect: entry.rect.and_then(bounds_from_settings_rect),
        });
        next_instance_id = next_instance_id.wrapping_add(1);
    }
    panel_framework.focus_order = restored_instances
        .iter()
        .map(|instance| instance.id)
        .collect();
    panel_framework.pinned_instances = restored_instances;
    panel_framework.next_instance_id = next_instance_id;
    panel_framework.panel_interaction = None;
}

pub(crate) fn capture(
    menu: &MenuUiState,
    panel_framework: &PanelFrameworkState,
) -> ShellChromeSettings {
    let mut pinned_instances = panel_framework
        .pinned_instances
        .iter()
        .filter(|instance| instance.visible)
        .collect::<Vec<_>>();
    pinned_instances.sort_by_key(|instance| {
        panel_framework
            .focus_order
            .iter()
            .position(|focused_id| *focused_id == instance.id)
            .unwrap_or(usize::MAX)
    });

    ShellChromeSettings {
        menu_strip_visible: menu.strip_visible,
        menu_focused_launcher: menu.focused_launcher.map(menu_launcher_to_preference),
        primary_transient_rect: panel_framework
            .bar(PanelBarId::PrimaryRight)
            .and_then(|bar| bar.transient_rect)
            .and_then(settings_rect_from_bounds),
        pinned_panels: pinned_instances
            .into_iter()
            .map(|instance| PinnedPanelPreference {
                kind: panel_kind_to_preference(instance.kind),
                collapsed: instance.collapsed,
                rect: instance.rect.and_then(settings_rect_from_bounds),
            })
            .collect(),
    }
}

fn menu_launcher_to_preference(kind: MenuLauncherKind) -> MenuLauncherPreference {
    match kind {
        MenuLauncherKind::File => MenuLauncherPreference::File,
        MenuLauncherKind::Edit => MenuLauncherPreference::Edit,
        MenuLauncherKind::View => MenuLauncherPreference::View,
        MenuLauncherKind::Settings => MenuLauncherPreference::Settings,
        MenuLauncherKind::Help => MenuLauncherPreference::Help,
    }
}

fn menu_launcher_from_preference(kind: MenuLauncherPreference) -> MenuLauncherKind {
    match kind {
        MenuLauncherPreference::File => MenuLauncherKind::File,
        MenuLauncherPreference::Edit => MenuLauncherKind::Edit,
        MenuLauncherPreference::View => MenuLauncherKind::View,
        MenuLauncherPreference::Settings => MenuLauncherKind::Settings,
        MenuLauncherPreference::Help => MenuLauncherKind::Help,
    }
}

fn panel_kind_to_preference(kind: PanelKind) -> PanelKindPreference {
    match kind {
        PanelKind::Tool => PanelKindPreference::Tool,
        PanelKind::ObjectProperties => PanelKindPreference::ObjectProperties,
        PanelKind::RenderSettings => PanelKindPreference::RenderSettings,
        PanelKind::Scene => PanelKindPreference::Scene,
        PanelKind::History => PanelKindPreference::History,
        PanelKind::ReferenceImages => PanelKindPreference::ReferenceImages,
    }
}

fn panel_kind_from_preference(kind: PanelKindPreference) -> PanelKind {
    match kind {
        PanelKindPreference::Tool => PanelKind::Tool,
        PanelKindPreference::ObjectProperties => PanelKind::ObjectProperties,
        PanelKindPreference::RenderSettings => PanelKind::RenderSettings,
        PanelKindPreference::Scene => PanelKind::Scene,
        PanelKindPreference::History => PanelKind::History,
        PanelKindPreference::ReferenceImages => PanelKind::ReferenceImages,
    }
}

fn settings_rect_from_bounds(bounds: FloatingPanelBounds) -> Option<ShellFloatingRect> {
    if !bounds.is_valid() {
        return None;
    }
    Some(ShellFloatingRect {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
    })
}

fn bounds_from_settings_rect(rect: ShellFloatingRect) -> Option<FloatingPanelBounds> {
    let bounds = FloatingPanelBounds::from_min_size(rect.x, rect.y, rect.width, rect.height);
    bounds.is_valid().then_some(bounds)
}

#[cfg(test)]
mod tests {
    use super::{apply, capture};
    use crate::app::state::{
        MenuLauncherKind, MenuUiState, PanelBarId, PanelFrameworkState, PanelKind,
    };
    use crate::app::ui_geometry::FloatingPanelBounds;
    use crate::settings::{
        MenuLauncherPreference, PanelKindPreference, PinnedPanelPreference, ShellChromeSettings,
        ShellFloatingRect,
    };

    #[test]
    fn apply_restores_menu_and_panel_framework_state() {
        let mut menu = MenuUiState::default();
        let mut panel_framework = PanelFrameworkState::default();
        let prefs = ShellChromeSettings {
            menu_strip_visible: false,
            menu_focused_launcher: Some(MenuLauncherPreference::Help),
            primary_transient_rect: Some(ShellFloatingRect {
                x: 120.0,
                y: 80.0,
                width: 360.0,
                height: 300.0,
            }),
            pinned_panels: vec![
                PinnedPanelPreference {
                    kind: PanelKindPreference::Tool,
                    collapsed: true,
                    rect: Some(ShellFloatingRect {
                        x: 20.0,
                        y: 30.0,
                        width: 350.0,
                        height: 240.0,
                    }),
                },
                PinnedPanelPreference {
                    kind: PanelKindPreference::Tool,
                    collapsed: false,
                    rect: Some(ShellFloatingRect {
                        x: 999.0,
                        y: 999.0,
                        width: 0.0,
                        height: 0.0,
                    }),
                },
                PinnedPanelPreference {
                    kind: PanelKindPreference::Scene,
                    collapsed: false,
                    rect: None,
                },
            ],
        };

        apply(&mut menu, &mut panel_framework, &prefs);

        assert!(!menu.strip_visible);
        assert_eq!(menu.focused_launcher, Some(MenuLauncherKind::Help));
        assert_eq!(menu.active_dropdown, None);
        assert!(!menu.settings_card_open);

        let transient_rect = panel_framework
            .bar(PanelBarId::PrimaryRight)
            .and_then(|bar| bar.transient_rect)
            .expect("remembered transient rect");
        assert!((transient_rect.x - 120.0).abs() < 0.01);
        assert_eq!(panel_framework.pinned_instances.len(), 2);
        assert_eq!(panel_framework.pinned_instances[0].kind, PanelKind::Tool);
        assert!(panel_framework.pinned_instances[0].collapsed);
        assert_eq!(panel_framework.pinned_instances[1].kind, PanelKind::Scene);
        assert_eq!(panel_framework.focus_order.len(), 2);
    }

    #[test]
    fn capture_keeps_focus_order_and_serializes_geometry() {
        let mut menu = MenuUiState::default();
        menu.focused_launcher = Some(MenuLauncherKind::Settings);
        let mut panel_framework = PanelFrameworkState::default();
        panel_framework
            .bar_mut(PanelBarId::PrimaryRight)
            .expect("primary bar")
            .transient_rect = Some(FloatingPanelBounds::from_min_size(10.0, 12.0, 350.0, 260.0));
        panel_framework.open_panel(PanelKind::Tool, PanelBarId::PrimaryRight);
        panel_framework.pin_panel(PanelKind::Tool);
        panel_framework.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        panel_framework.pin_panel(PanelKind::RenderSettings);
        panel_framework.focus_panel(PanelKind::RenderSettings);
        panel_framework.focus_panel(PanelKind::Tool);

        let prefs = capture(&menu, &panel_framework);

        assert!(prefs.menu_strip_visible);
        assert_eq!(
            prefs.menu_focused_launcher,
            Some(MenuLauncherPreference::Settings)
        );
        assert_eq!(
            prefs.primary_transient_rect.map(|rect| rect.width),
            Some(350.0)
        );
        assert_eq!(prefs.pinned_panels.len(), 2);
        assert_eq!(
            prefs.pinned_panels[0].kind,
            PanelKindPreference::RenderSettings
        );
        assert_eq!(prefs.pinned_panels[1].kind, PanelKindPreference::Tool);
    }
}
