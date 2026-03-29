use eframe::egui::{self, Color32};
use egui_dock::{LeafHighlighting, OverlayType, TabAddAlign};
use serde::{Deserialize, Serialize};

pub type ColorRgba = [u8; 4];

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DockTabAddAlign {
    Left,
    #[default]
    Right,
}

impl DockTabAddAlign {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Right => "Right",
        }
    }

    pub(crate) fn to_egui(self) -> TabAddAlign {
        match self {
            Self::Left => TabAddAlign::Left,
            Self::Right => TabAddAlign::Right,
        }
    }

    pub(crate) fn from_egui(value: TabAddAlign) -> Self {
        match value {
            TabAddAlign::Left => Self::Left,
            TabAddAlign::Right => Self::Right,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DockOverlayType {
    HighlightedAreas,
    #[default]
    Widgets,
}

impl DockOverlayType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::HighlightedAreas => "Highlighted Areas",
            Self::Widgets => "Widgets",
        }
    }

    pub(crate) fn to_egui(self) -> OverlayType {
        match self {
            Self::HighlightedAreas => OverlayType::HighlightedAreas,
            Self::Widgets => OverlayType::Widgets,
        }
    }

    pub(crate) fn from_egui(value: OverlayType) -> Self {
        match value {
            OverlayType::HighlightedAreas => Self::HighlightedAreas,
            OverlayType::Widgets => Self::Widgets,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
#[serde(default)]
pub struct DockEdgeInsets {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl DockEdgeInsets {
    pub(crate) fn to_egui(self) -> egui::Margin {
        egui::Margin {
            left: self.left,
            right: self.right,
            top: self.top,
            bottom: self.bottom,
        }
    }

    pub(crate) fn from_egui(value: egui::Margin) -> Self {
        Self {
            left: value.left,
            right: value.right,
            top: value.top,
            bottom: value.bottom,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
#[serde(default)]
pub struct DockOptionalEdgeInsets {
    pub enabled: bool,
    pub value: DockEdgeInsets,
}

impl DockOptionalEdgeInsets {
    pub(crate) fn to_egui(self) -> Option<egui::Margin> {
        self.enabled.then_some(self.value.to_egui())
    }

    pub(crate) fn from_egui(value: Option<egui::Margin>) -> Self {
        match value {
            Some(value) => Self {
                enabled: true,
                value: DockEdgeInsets::from_egui(value),
            },
            None => Self::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
#[serde(default)]
pub struct DockRounding {
    pub nw: f32,
    pub ne: f32,
    pub sw: f32,
    pub se: f32,
}

impl DockRounding {
    pub(crate) fn to_egui(self) -> egui::Rounding {
        egui::Rounding {
            nw: self.nw,
            ne: self.ne,
            sw: self.sw,
            se: self.se,
        }
    }

    pub(crate) fn from_egui(value: egui::Rounding) -> Self {
        Self {
            nw: value.nw,
            ne: value.ne,
            sw: value.sw,
            se: value.se,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct DockStroke {
    pub width: f32,
    pub color: ColorRgba,
}

impl Default for DockStroke {
    fn default() -> Self {
        Self {
            width: 0.0,
            color: [0, 0, 0, 0],
        }
    }
}

impl DockStroke {
    pub(crate) fn to_egui(self) -> egui::Stroke {
        egui::Stroke::new(self.width, color_from_rgba(self.color))
    }

    pub(crate) fn from_egui(value: egui::Stroke) -> Self {
        Self {
            width: value.width,
            color: color_to_rgba(value.color),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockButtonsStyleSettings {
    pub close_tab_color: ColorRgba,
    pub close_tab_active_color: ColorRgba,
    pub close_tab_bg_fill: ColorRgba,
    pub add_tab_align: DockTabAddAlign,
    pub add_tab_color: ColorRgba,
    pub add_tab_active_color: ColorRgba,
    pub add_tab_bg_fill: ColorRgba,
    pub add_tab_border_color: ColorRgba,
}

impl Default for DockButtonsStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::ButtonsStyle::default())
    }
}

impl DockButtonsStyleSettings {
    fn to_egui(&self) -> egui_dock::ButtonsStyle {
        egui_dock::ButtonsStyle {
            close_tab_color: color_from_rgba(self.close_tab_color),
            close_tab_active_color: color_from_rgba(self.close_tab_active_color),
            close_tab_bg_fill: color_from_rgba(self.close_tab_bg_fill),
            add_tab_align: self.add_tab_align.to_egui(),
            add_tab_color: color_from_rgba(self.add_tab_color),
            add_tab_active_color: color_from_rgba(self.add_tab_active_color),
            add_tab_bg_fill: color_from_rgba(self.add_tab_bg_fill),
            add_tab_border_color: color_from_rgba(self.add_tab_border_color),
        }
    }

    fn from_egui(value: &egui_dock::ButtonsStyle) -> Self {
        Self {
            close_tab_color: color_to_rgba(value.close_tab_color),
            close_tab_active_color: color_to_rgba(value.close_tab_active_color),
            close_tab_bg_fill: color_to_rgba(value.close_tab_bg_fill),
            add_tab_align: DockTabAddAlign::from_egui(value.add_tab_align),
            add_tab_color: color_to_rgba(value.add_tab_color),
            add_tab_active_color: color_to_rgba(value.add_tab_active_color),
            add_tab_bg_fill: color_to_rgba(value.add_tab_bg_fill),
            add_tab_border_color: color_to_rgba(value.add_tab_border_color),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockSeparatorStyleSettings {
    pub width: f32,
    pub extra_interact_width: f32,
    pub extra: f32,
    pub color_idle: ColorRgba,
    pub color_hovered: ColorRgba,
    pub color_dragged: ColorRgba,
}

impl Default for DockSeparatorStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::SeparatorStyle::default())
    }
}

impl DockSeparatorStyleSettings {
    fn to_egui(&self) -> egui_dock::SeparatorStyle {
        egui_dock::SeparatorStyle {
            width: self.width,
            extra_interact_width: self.extra_interact_width,
            extra: self.extra,
            color_idle: color_from_rgba(self.color_idle),
            color_hovered: color_from_rgba(self.color_hovered),
            color_dragged: color_from_rgba(self.color_dragged),
        }
    }

    fn from_egui(value: &egui_dock::SeparatorStyle) -> Self {
        Self {
            width: value.width,
            extra_interact_width: value.extra_interact_width,
            extra: value.extra,
            color_idle: color_to_rgba(value.color_idle),
            color_hovered: color_to_rgba(value.color_hovered),
            color_dragged: color_to_rgba(value.color_dragged),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockTabBarStyleSettings {
    pub bg_fill: ColorRgba,
    pub height: f32,
    pub show_scroll_bar_on_overflow: bool,
    pub rounding: DockRounding,
    pub hline_color: ColorRgba,
    pub fill_tab_bar: bool,
}

impl Default for DockTabBarStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::TabBarStyle::default())
    }
}

impl DockTabBarStyleSettings {
    fn to_egui(&self) -> egui_dock::TabBarStyle {
        egui_dock::TabBarStyle {
            bg_fill: color_from_rgba(self.bg_fill),
            height: self.height,
            show_scroll_bar_on_overflow: self.show_scroll_bar_on_overflow,
            rounding: self.rounding.to_egui(),
            hline_color: color_from_rgba(self.hline_color),
            fill_tab_bar: self.fill_tab_bar,
        }
    }

    fn from_egui(value: &egui_dock::TabBarStyle) -> Self {
        Self {
            bg_fill: color_to_rgba(value.bg_fill),
            height: value.height,
            show_scroll_bar_on_overflow: value.show_scroll_bar_on_overflow,
            rounding: DockRounding::from_egui(value.rounding),
            hline_color: color_to_rgba(value.hline_color),
            fill_tab_bar: value.fill_tab_bar,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockTabInteractionStyleSettings {
    pub outline_color: ColorRgba,
    pub rounding: DockRounding,
    pub bg_fill: ColorRgba,
    pub text_color: ColorRgba,
}

impl Default for DockTabInteractionStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::TabInteractionStyle::default())
    }
}

impl DockTabInteractionStyleSettings {
    fn to_egui(&self) -> egui_dock::TabInteractionStyle {
        egui_dock::TabInteractionStyle {
            outline_color: color_from_rgba(self.outline_color),
            rounding: self.rounding.to_egui(),
            bg_fill: color_from_rgba(self.bg_fill),
            text_color: color_from_rgba(self.text_color),
        }
    }

    fn from_egui(value: &egui_dock::TabInteractionStyle) -> Self {
        Self {
            outline_color: color_to_rgba(value.outline_color),
            rounding: DockRounding::from_egui(value.rounding),
            bg_fill: color_to_rgba(value.bg_fill),
            text_color: color_to_rgba(value.text_color),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockTabBodyStyleSettings {
    pub inner_margin: DockEdgeInsets,
    pub stroke: DockStroke,
    pub rounding: DockRounding,
    pub bg_fill: ColorRgba,
}

impl Default for DockTabBodyStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::TabBodyStyle::default())
    }
}

impl DockTabBodyStyleSettings {
    fn to_egui(&self) -> egui_dock::TabBodyStyle {
        egui_dock::TabBodyStyle {
            inner_margin: self.inner_margin.to_egui(),
            stroke: self.stroke.to_egui(),
            rounding: self.rounding.to_egui(),
            bg_fill: color_from_rgba(self.bg_fill),
        }
    }

    fn from_egui(value: &egui_dock::TabBodyStyle) -> Self {
        Self {
            inner_margin: DockEdgeInsets::from_egui(value.inner_margin),
            stroke: DockStroke::from_egui(value.stroke),
            rounding: DockRounding::from_egui(value.rounding),
            bg_fill: color_to_rgba(value.bg_fill),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockTabStyleSettings {
    pub active: DockTabInteractionStyleSettings,
    pub inactive: DockTabInteractionStyleSettings,
    pub focused: DockTabInteractionStyleSettings,
    pub hovered: DockTabInteractionStyleSettings,
    pub inactive_with_kb_focus: DockTabInteractionStyleSettings,
    pub active_with_kb_focus: DockTabInteractionStyleSettings,
    pub focused_with_kb_focus: DockTabInteractionStyleSettings,
    pub tab_body: DockTabBodyStyleSettings,
    pub hline_below_active_tab_name: bool,
    pub minimum_width_enabled: bool,
    pub minimum_width: f32,
}

impl Default for DockTabStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::TabStyle::default())
    }
}

impl DockTabStyleSettings {
    fn to_egui(&self) -> egui_dock::TabStyle {
        egui_dock::TabStyle {
            active: self.active.to_egui(),
            inactive: self.inactive.to_egui(),
            focused: self.focused.to_egui(),
            hovered: self.hovered.to_egui(),
            inactive_with_kb_focus: self.inactive_with_kb_focus.to_egui(),
            active_with_kb_focus: self.active_with_kb_focus.to_egui(),
            focused_with_kb_focus: self.focused_with_kb_focus.to_egui(),
            tab_body: self.tab_body.to_egui(),
            hline_below_active_tab_name: self.hline_below_active_tab_name,
            minimum_width: self.minimum_width_enabled.then_some(self.minimum_width),
        }
    }

    fn from_egui(value: &egui_dock::TabStyle) -> Self {
        Self {
            active: DockTabInteractionStyleSettings::from_egui(&value.active),
            inactive: DockTabInteractionStyleSettings::from_egui(&value.inactive),
            focused: DockTabInteractionStyleSettings::from_egui(&value.focused),
            hovered: DockTabInteractionStyleSettings::from_egui(&value.hovered),
            inactive_with_kb_focus: DockTabInteractionStyleSettings::from_egui(
                &value.inactive_with_kb_focus,
            ),
            active_with_kb_focus: DockTabInteractionStyleSettings::from_egui(
                &value.active_with_kb_focus,
            ),
            focused_with_kb_focus: DockTabInteractionStyleSettings::from_egui(
                &value.focused_with_kb_focus,
            ),
            tab_body: DockTabBodyStyleSettings::from_egui(&value.tab_body),
            hline_below_active_tab_name: value.hline_below_active_tab_name,
            minimum_width_enabled: value.minimum_width.is_some(),
            minimum_width: value.minimum_width.unwrap_or(96.0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockOverlayFeelSettings {
    pub window_drop_coverage: f32,
    pub center_drop_coverage: f32,
    pub fade_hold_time: f32,
    pub max_preference_time: f32,
    pub interact_expansion: f32,
}

impl Default for DockOverlayFeelSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::OverlayFeel::default())
    }
}

impl DockOverlayFeelSettings {
    fn to_egui(&self) -> egui_dock::OverlayFeel {
        egui_dock::OverlayFeel {
            window_drop_coverage: self.window_drop_coverage,
            center_drop_coverage: self.center_drop_coverage,
            fade_hold_time: self.fade_hold_time,
            max_preference_time: self.max_preference_time,
            interact_expansion: self.interact_expansion,
        }
    }

    fn from_egui(value: &egui_dock::OverlayFeel) -> Self {
        Self {
            window_drop_coverage: value.window_drop_coverage,
            center_drop_coverage: value.center_drop_coverage,
            fade_hold_time: value.fade_hold_time,
            max_preference_time: value.max_preference_time,
            interact_expansion: value.interact_expansion,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockLeafHighlightingSettings {
    pub color: ColorRgba,
    pub rounding: DockRounding,
    pub stroke: DockStroke,
    pub expansion: f32,
}

impl Default for DockLeafHighlightingSettings {
    fn default() -> Self {
        Self::from_egui(&LeafHighlighting::default())
    }
}

impl DockLeafHighlightingSettings {
    fn to_egui(&self) -> LeafHighlighting {
        LeafHighlighting {
            color: color_from_rgba(self.color),
            rounding: self.rounding.to_egui(),
            stroke: self.stroke.to_egui(),
            expansion: self.expansion,
        }
    }

    fn from_egui(value: &LeafHighlighting) -> Self {
        Self {
            color: color_to_rgba(value.color),
            rounding: DockRounding::from_egui(value.rounding),
            stroke: DockStroke::from_egui(value.stroke),
            expansion: value.expansion,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockOverlayStyleSettings {
    pub selection_color: ColorRgba,
    pub selection_stroke_width: f32,
    pub button_spacing: f32,
    pub max_button_size: f32,
    pub hovered_leaf_highlight: DockLeafHighlightingSettings,
    pub surface_fade_opacity: f32,
    pub button_color: ColorRgba,
    pub button_border_stroke: DockStroke,
    pub overlay_type: DockOverlayType,
    pub feel: DockOverlayFeelSettings,
}

impl Default for DockOverlayStyleSettings {
    fn default() -> Self {
        Self::from_egui(&egui_dock::OverlayStyle::default())
    }
}

impl DockOverlayStyleSettings {
    fn to_egui(&self) -> egui_dock::OverlayStyle {
        egui_dock::OverlayStyle {
            selection_color: color_from_rgba(self.selection_color),
            selection_stroke_width: self.selection_stroke_width,
            button_spacing: self.button_spacing,
            max_button_size: self.max_button_size,
            hovered_leaf_highlight: self.hovered_leaf_highlight.to_egui(),
            surface_fade_opacity: self.surface_fade_opacity,
            button_color: color_from_rgba(self.button_color),
            button_border_stroke: self.button_border_stroke.to_egui(),
            overlay_type: self.overlay_type.to_egui(),
            feel: self.feel.to_egui(),
        }
    }

    fn from_egui(value: &egui_dock::OverlayStyle) -> Self {
        Self {
            selection_color: color_to_rgba(value.selection_color),
            selection_stroke_width: value.selection_stroke_width,
            button_spacing: value.button_spacing,
            max_button_size: value.max_button_size,
            hovered_leaf_highlight: DockLeafHighlightingSettings::from_egui(
                &value.hovered_leaf_highlight,
            ),
            surface_fade_opacity: value.surface_fade_opacity,
            button_color: color_to_rgba(value.button_color),
            button_border_stroke: DockStroke::from_egui(value.button_border_stroke),
            overlay_type: DockOverlayType::from_egui(value.overlay_type),
            feel: DockOverlayFeelSettings::from_egui(&value.feel),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct DockStyleSettings {
    pub enabled: bool,
    pub dock_area_padding: DockOptionalEdgeInsets,
    pub main_surface_border_stroke: DockStroke,
    pub main_surface_border_rounding: DockRounding,
    pub buttons: DockButtonsStyleSettings,
    pub separator: DockSeparatorStyleSettings,
    pub tab_bar: DockTabBarStyleSettings,
    pub tab: DockTabStyleSettings,
    pub overlay: DockOverlayStyleSettings,
}

impl Default for DockStyleSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            dock_area_padding: DockOptionalEdgeInsets::default(),
            main_surface_border_stroke: DockStroke::from_egui(egui::Stroke::NONE),
            main_surface_border_rounding: DockRounding::default(),
            buttons: DockButtonsStyleSettings::default(),
            separator: DockSeparatorStyleSettings::default(),
            tab_bar: DockTabBarStyleSettings::default(),
            tab: DockTabStyleSettings::default(),
            overlay: DockOverlayStyleSettings::default(),
        }
    }
}

impl DockStyleSettings {
    pub fn from_egui_style(style: &egui::Style) -> Self {
        Self::from_dock_style(&egui_dock::Style::from_egui(style))
    }

    pub fn reset_to_current_theme(&mut self, style: &egui::Style) {
        let mut reset = Self::from_egui_style(style);
        reset.enabled = self.enabled;
        *self = reset;
    }

    pub fn enable_from_current_theme(&mut self, style: &egui::Style) {
        *self = Self::from_egui_style(style);
        self.enabled = true;
    }

    pub fn to_egui_dock_style(&self, egui_style: &egui::Style) -> egui_dock::Style {
        if !self.enabled {
            return egui_dock::Style::from_egui(egui_style);
        }

        egui_dock::Style {
            dock_area_padding: self.dock_area_padding.to_egui(),
            main_surface_border_stroke: self.main_surface_border_stroke.to_egui(),
            main_surface_border_rounding: self.main_surface_border_rounding.to_egui(),
            buttons: self.buttons.to_egui(),
            separator: self.separator.to_egui(),
            tab_bar: self.tab_bar.to_egui(),
            tab: self.tab.to_egui(),
            overlay: self.overlay.to_egui(),
        }
    }

    fn from_dock_style(style: &egui_dock::Style) -> Self {
        Self {
            enabled: true,
            dock_area_padding: DockOptionalEdgeInsets::from_egui(style.dock_area_padding),
            main_surface_border_stroke: DockStroke::from_egui(style.main_surface_border_stroke),
            main_surface_border_rounding: DockRounding::from_egui(
                style.main_surface_border_rounding,
            ),
            buttons: DockButtonsStyleSettings::from_egui(&style.buttons),
            separator: DockSeparatorStyleSettings::from_egui(&style.separator),
            tab_bar: DockTabBarStyleSettings::from_egui(&style.tab_bar),
            tab: DockTabStyleSettings::from_egui(&style.tab),
            overlay: DockOverlayStyleSettings::from_egui(&style.overlay),
        }
    }
}

pub(crate) fn color_to_rgba(color: Color32) -> ColorRgba {
    color.to_srgba_unmultiplied()
}

pub(crate) fn color_from_rgba([r, g, b, a]: ColorRgba) -> Color32 {
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

#[cfg(test)]
mod tests {
    use super::DockStyleSettings;

    #[test]
    fn disabled_style_uses_runtime_egui_defaults() {
        let settings = DockStyleSettings::default();
        let egui_style = eframe::egui::Style::default();

        let actual = settings.to_egui_dock_style(&egui_style);
        let expected = egui_dock::Style::from_egui(&egui_style);

        assert_eq!(actual.tab_bar.height, expected.tab_bar.height);
        assert_eq!(actual.tab_bar.bg_fill, expected.tab_bar.bg_fill);
        assert_eq!(actual.tab.active.bg_fill, expected.tab.active.bg_fill);
        assert_eq!(actual.overlay.overlay_type, expected.overlay.overlay_type);
    }

    #[test]
    fn roundtrip_from_egui_style_preserves_tab_and_overlay_values() {
        let egui_style = eframe::egui::Style::default();
        let settings = DockStyleSettings::from_egui_style(&egui_style);
        let actual = settings.to_egui_dock_style(&egui_style);
        let expected = egui_dock::Style::from_egui(&egui_style);

        assert_eq!(actual.tab_bar.height, expected.tab_bar.height);
        assert_eq!(
            actual.tab.inactive.text_color,
            expected.tab.inactive.text_color
        );
        assert_eq!(
            actual.overlay.selection_color,
            expected.overlay.selection_color
        );
        assert_eq!(
            actual.overlay.feel.interact_expansion,
            expected.overlay.feel.interact_expansion
        );
    }
}
