use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Component, Path, PathBuf};

use eframe::egui::{
    self, style::ScrollStyle, Color32, FontData, FontDefinitions, FontFamily, FontId, TextStyle,
};
use serde::{Deserialize, Serialize};

use crate::dock_style::{
    color_from_rgba, color_to_rgba, ColorRgba, DockEdgeInsets, DockRounding, DockStroke,
};

const UI_SANS_FAMILY_NAME: &str = "UiSans";
const UI_HEADING_FAMILY_NAME: &str = "UiHeading";
const UI_MONO_FAMILY_NAME: &str = "UiMono";

const VIEWPORT_HUD_STYLE_NAME: &str = "ViewportHud";
const VIEWPORT_MONO_STYLE_NAME: &str = "ViewportMono";
const SCENE_LABEL_STYLE_NAME: &str = "SceneLabel";

const DEFAULT_VIEWPORT_HUD_POINTS: f32 = 13.0;
const DEFAULT_VIEWPORT_MONO_POINTS: f32 = 11.0;
const DEFAULT_SCENE_LABEL_POINTS: f32 = 11.0;

const DEFAULT_PROPORTIONAL_FONT_NAME: &str = "Ubuntu-Light";
const DEFAULT_MONOSPACE_FONT_NAME: &str = "Hack";

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum ThemePreset {
    #[default]
    StudioDark,
    SlateLight,
    HighContrastDark,
}

impl ThemePreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::StudioDark => "Studio Dark",
            Self::SlateLight => "Slate Light",
            Self::HighContrastDark => "High Contrast Dark",
        }
    }

    pub fn egui_theme(self) -> egui::Theme {
        match self {
            Self::SlateLight => egui::Theme::Light,
            Self::StudioDark | Self::HighContrastDark => egui::Theme::Dark,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum ThemeFontRole {
    UiSans,
    UiHeading,
    UiMono,
}

impl ThemeFontRole {
    pub const ALL: [Self; 3] = [Self::UiSans, Self::UiHeading, Self::UiMono];

    pub fn label(self) -> &'static str {
        match self {
            Self::UiSans => "UI Sans",
            Self::UiHeading => "UI Heading",
            Self::UiMono => "UI Mono",
        }
    }

    fn family_name(self) -> &'static str {
        match self {
            Self::UiSans => UI_SANS_FAMILY_NAME,
            Self::UiHeading => UI_HEADING_FAMILY_NAME,
            Self::UiMono => UI_MONO_FAMILY_NAME,
        }
    }

    fn font_key(self) -> &'static str {
        match self {
            Self::UiSans => "theme_ui_sans_primary",
            Self::UiHeading => "theme_ui_heading_primary",
            Self::UiMono => "theme_ui_mono_primary",
        }
    }

    fn storage_stem(self) -> &'static str {
        match self {
            Self::UiSans => "ui_sans",
            Self::UiHeading => "ui_heading",
            Self::UiMono => "ui_mono",
        }
    }

    fn default_font_name(self) -> &'static str {
        match self {
            Self::UiSans | Self::UiHeading => DEFAULT_PROPORTIONAL_FONT_NAME,
            Self::UiMono => DEFAULT_MONOSPACE_FONT_NAME,
        }
    }

    fn fallback_family(self) -> FontFamily {
        match self {
            Self::UiSans | Self::UiHeading => FontFamily::Proportional,
            Self::UiMono => FontFamily::Monospace,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AppTextRole {
    ViewportHud,
    ViewportMono,
    SceneLabel,
}

impl AppTextRole {
    fn text_style(self) -> TextStyle {
        match self {
            Self::ViewportHud => TextStyle::Name(VIEWPORT_HUD_STYLE_NAME.into()),
            Self::ViewportMono => TextStyle::Name(VIEWPORT_MONO_STYLE_NAME.into()),
            Self::SceneLabel => TextStyle::Name(SCENE_LABEL_STYLE_NAME.into()),
        }
    }

    fn default_points(self) -> f32 {
        match self {
            Self::ViewportHud => DEFAULT_VIEWPORT_HUD_POINTS,
            Self::ViewportMono => DEFAULT_VIEWPORT_MONO_POINTS,
            Self::SceneLabel => DEFAULT_SCENE_LABEL_POINTS,
        }
    }

    fn fallback_text_style(self) -> TextStyle {
        match self {
            Self::ViewportHud | Self::SceneLabel => TextStyle::Body,
            Self::ViewportMono => TextStyle::Monospace,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeOptionalColor {
    pub enabled: bool,
    pub color: ColorRgba,
}

impl Default for ThemeOptionalColor {
    fn default() -> Self {
        Self {
            enabled: false,
            color: [0, 0, 0, 255],
        }
    }
}

impl ThemeOptionalColor {
    fn to_egui(self) -> Option<Color32> {
        self.enabled.then(|| color_from_rgba(self.color))
    }

    fn from_egui(value: Option<Color32>) -> Self {
        match value {
            Some(color) => Self {
                enabled: true,
                color: color_to_rgba(color),
            },
            None => Self::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
#[serde(default)]
pub struct ThemeVec2 {
    pub x: f32,
    pub y: f32,
}

impl ThemeVec2 {
    fn to_egui(self) -> egui::Vec2 {
        egui::vec2(self.x, self.y)
    }

    fn from_egui(value: egui::Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeFontTweakSettings {
    pub scale: f32,
    pub y_offset_factor: f32,
    pub y_offset: f32,
    pub baseline_offset_factor: f32,
}

impl ThemeFontTweakSettings {
    fn to_egui(self) -> egui::epaint::text::FontTweak {
        egui::epaint::text::FontTweak {
            scale: self.scale,
            y_offset_factor: self.y_offset_factor,
            y_offset: self.y_offset,
            baseline_offset_factor: self.baseline_offset_factor,
        }
    }

    fn from_egui(value: egui::epaint::text::FontTweak) -> Self {
        Self {
            scale: value.scale,
            y_offset_factor: value.y_offset_factor,
            y_offset: value.y_offset,
            baseline_offset_factor: value.baseline_offset_factor,
        }
    }
}

impl Default for ThemeFontTweakSettings {
    fn default() -> Self {
        Self::from_egui(Default::default())
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub enum ThemeFontSource {
    #[default]
    BundledDefault,
    Imported {
        relative_path: String,
    },
}

impl ThemeFontSource {
    pub fn label(&self) -> String {
        match self {
            Self::BundledDefault => "Built-in egui defaults".to_string(),
            Self::Imported { relative_path } => format!("Imported ({relative_path})"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(default)]
pub struct ThemeFontRoleSettings {
    pub source: ThemeFontSource,
    pub tweak: ThemeFontTweakSettings,
    pub import_serial: u64,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeTextSizes {
    pub small: f32,
    pub body: f32,
    pub button: f32,
    pub monospace: f32,
    pub heading: f32,
    pub viewport_hud: f32,
    pub viewport_mono: f32,
    pub scene_label: f32,
}

impl Default for ThemeTextSizes {
    fn default() -> Self {
        Self {
            small: 10.0,
            body: 14.0,
            button: 14.0,
            monospace: 13.0,
            heading: 20.0,
            viewport_hud: DEFAULT_VIEWPORT_HUD_POINTS,
            viewport_mono: DEFAULT_VIEWPORT_MONO_POINTS,
            scene_label: DEFAULT_SCENE_LABEL_POINTS,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(default)]
pub struct ThemeBaseColorsSettings {
    pub override_text_color: ThemeOptionalColor,
    pub hyperlink_color: ColorRgba,
    pub faint_bg_color: ColorRgba,
    pub extreme_bg_color: ColorRgba,
    pub code_bg_color: ColorRgba,
    pub warn_fg_color: ColorRgba,
    pub error_fg_color: ColorRgba,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeSelectionSettings {
    pub bg_fill: ColorRgba,
    pub stroke: DockStroke,
}

impl Default for ThemeSelectionSettings {
    fn default() -> Self {
        Self {
            bg_fill: [61, 133, 224, 96],
            stroke: DockStroke {
                width: 1.0,
                color: [120, 190, 255, 255],
            },
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeWidgetVisualsSettings {
    pub bg_fill: ColorRgba,
    pub weak_bg_fill: ColorRgba,
    pub bg_stroke: DockStroke,
    pub rounding: DockRounding,
    pub fg_stroke: DockStroke,
    pub expansion: f32,
}

impl ThemeWidgetVisualsSettings {
    fn from_egui(value: &egui::style::WidgetVisuals) -> Self {
        Self {
            bg_fill: color_to_rgba(value.bg_fill),
            weak_bg_fill: color_to_rgba(value.weak_bg_fill),
            bg_stroke: DockStroke::from_egui(value.bg_stroke),
            rounding: DockRounding::from_egui(value.rounding),
            fg_stroke: DockStroke::from_egui(value.fg_stroke),
            expansion: value.expansion,
        }
    }

    fn to_egui(self) -> egui::style::WidgetVisuals {
        egui::style::WidgetVisuals {
            bg_fill: color_from_rgba(self.bg_fill),
            weak_bg_fill: color_from_rgba(self.weak_bg_fill),
            bg_stroke: self.bg_stroke.to_egui(),
            rounding: self.rounding.to_egui(),
            fg_stroke: self.fg_stroke.to_egui(),
            expansion: self.expansion,
        }
    }
}

impl Default for ThemeWidgetVisualsSettings {
    fn default() -> Self {
        Self::from_egui(&egui::Visuals::dark().widgets.inactive)
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(default)]
pub struct ThemeWidgetStatesSettings {
    pub noninteractive: ThemeWidgetVisualsSettings,
    pub inactive: ThemeWidgetVisualsSettings,
    pub hovered: ThemeWidgetVisualsSettings,
    pub active: ThemeWidgetVisualsSettings,
    pub open: ThemeWidgetVisualsSettings,
}

impl ThemeWidgetStatesSettings {
    fn from_egui(value: &egui::style::Widgets) -> Self {
        Self {
            noninteractive: ThemeWidgetVisualsSettings::from_egui(&value.noninteractive),
            inactive: ThemeWidgetVisualsSettings::from_egui(&value.inactive),
            hovered: ThemeWidgetVisualsSettings::from_egui(&value.hovered),
            active: ThemeWidgetVisualsSettings::from_egui(&value.active),
            open: ThemeWidgetVisualsSettings::from_egui(&value.open),
        }
    }

    fn to_egui(&self) -> egui::style::Widgets {
        egui::style::Widgets {
            noninteractive: self.noninteractive.to_egui(),
            inactive: self.inactive.to_egui(),
            hovered: self.hovered.to_egui(),
            active: self.active.to_egui(),
            open: self.open.to_egui(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeWindowsPanelsSettings {
    pub window_rounding: DockRounding,
    pub window_fill: ColorRgba,
    pub window_stroke: DockStroke,
    pub menu_rounding: DockRounding,
    pub panel_fill: ColorRgba,
    pub resize_corner_size: f32,
    pub clip_rect_margin: f32,
    pub button_frame: bool,
    pub collapsing_header_frame: bool,
    pub indent_has_left_vline: bool,
    pub striped: bool,
    pub slider_trailing_fill: bool,
}

impl Default for ThemeWindowsPanelsSettings {
    fn default() -> Self {
        let visuals = egui::Visuals::dark();
        Self::from_egui(&visuals)
    }
}

impl ThemeWindowsPanelsSettings {
    fn from_egui(value: &egui::Visuals) -> Self {
        Self {
            window_rounding: DockRounding::from_egui(value.window_rounding),
            window_fill: color_to_rgba(value.window_fill),
            window_stroke: DockStroke::from_egui(value.window_stroke),
            menu_rounding: DockRounding::from_egui(value.menu_rounding),
            panel_fill: color_to_rgba(value.panel_fill),
            resize_corner_size: value.resize_corner_size,
            clip_rect_margin: value.clip_rect_margin,
            button_frame: value.button_frame,
            collapsing_header_frame: value.collapsing_header_frame,
            indent_has_left_vline: value.indent_has_left_vline,
            striped: value.striped,
            slider_trailing_fill: value.slider_trailing_fill,
        }
    }

    fn apply_to_visuals(&self, visuals: &mut egui::Visuals) {
        visuals.window_rounding = self.window_rounding.to_egui();
        visuals.window_fill = color_from_rgba(self.window_fill);
        visuals.window_stroke = self.window_stroke.to_egui();
        visuals.menu_rounding = self.menu_rounding.to_egui();
        visuals.panel_fill = color_from_rgba(self.panel_fill);
        visuals.resize_corner_size = self.resize_corner_size;
        visuals.clip_rect_margin = self.clip_rect_margin;
        visuals.button_frame = self.button_frame;
        visuals.collapsing_header_frame = self.collapsing_header_frame;
        visuals.indent_has_left_vline = self.indent_has_left_vline;
        visuals.striped = self.striped;
        visuals.slider_trailing_fill = self.slider_trailing_fill;
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeSpacingSettings {
    pub item_spacing: ThemeVec2,
    pub window_margin: DockEdgeInsets,
    pub button_padding: ThemeVec2,
    pub menu_margin: DockEdgeInsets,
    pub indent: f32,
    pub interact_size: ThemeVec2,
    pub slider_width: f32,
    pub slider_rail_height: f32,
    pub combo_width: f32,
    pub text_edit_width: f32,
    pub tooltip_width: f32,
    pub menu_width: f32,
    pub menu_spacing: f32,
    pub combo_height: f32,
}

impl Default for ThemeSpacingSettings {
    fn default() -> Self {
        Self::from_egui(&egui::Style::default().spacing)
    }
}

impl ThemeSpacingSettings {
    fn from_egui(value: &egui::Spacing) -> Self {
        Self {
            item_spacing: ThemeVec2::from_egui(value.item_spacing),
            window_margin: DockEdgeInsets::from_egui(value.window_margin),
            button_padding: ThemeVec2::from_egui(value.button_padding),
            menu_margin: DockEdgeInsets::from_egui(value.menu_margin),
            indent: value.indent,
            interact_size: ThemeVec2::from_egui(value.interact_size),
            slider_width: value.slider_width,
            slider_rail_height: value.slider_rail_height,
            combo_width: value.combo_width,
            text_edit_width: value.text_edit_width,
            tooltip_width: value.tooltip_width,
            menu_width: value.menu_width,
            menu_spacing: value.menu_spacing,
            combo_height: value.combo_height,
        }
    }

    fn apply_to_spacing(&self, spacing: &mut egui::Spacing) {
        spacing.item_spacing = self.item_spacing.to_egui();
        spacing.window_margin = self.window_margin.to_egui();
        spacing.button_padding = self.button_padding.to_egui();
        spacing.menu_margin = self.menu_margin.to_egui();
        spacing.indent = self.indent;
        spacing.interact_size = self.interact_size.to_egui();
        spacing.slider_width = self.slider_width;
        spacing.slider_rail_height = self.slider_rail_height;
        spacing.combo_width = self.combo_width;
        spacing.text_edit_width = self.text_edit_width;
        spacing.tooltip_width = self.tooltip_width;
        spacing.menu_width = self.menu_width;
        spacing.menu_spacing = self.menu_spacing;
        spacing.combo_height = self.combo_height;
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct ThemeScrollbarsSettings {
    pub floating: bool,
    pub bar_width: f32,
    pub handle_min_length: f32,
    pub bar_inner_margin: f32,
    pub bar_outer_margin: f32,
    pub floating_width: f32,
    pub floating_allocated_width: f32,
    pub foreground_color: bool,
    pub dormant_background_opacity: f32,
    pub active_background_opacity: f32,
    pub interact_background_opacity: f32,
    pub dormant_handle_opacity: f32,
    pub active_handle_opacity: f32,
    pub interact_handle_opacity: f32,
}

impl Default for ThemeScrollbarsSettings {
    fn default() -> Self {
        Self::from_egui(ScrollStyle::default())
    }
}

impl ThemeScrollbarsSettings {
    fn from_egui(value: ScrollStyle) -> Self {
        Self {
            floating: value.floating,
            bar_width: value.bar_width,
            handle_min_length: value.handle_min_length,
            bar_inner_margin: value.bar_inner_margin,
            bar_outer_margin: value.bar_outer_margin,
            floating_width: value.floating_width,
            floating_allocated_width: value.floating_allocated_width,
            foreground_color: value.foreground_color,
            dormant_background_opacity: value.dormant_background_opacity,
            active_background_opacity: value.active_background_opacity,
            interact_background_opacity: value.interact_background_opacity,
            dormant_handle_opacity: value.dormant_handle_opacity,
            active_handle_opacity: value.active_handle_opacity,
            interact_handle_opacity: value.interact_handle_opacity,
        }
    }

    fn to_egui(self) -> ScrollStyle {
        ScrollStyle {
            floating: self.floating,
            bar_width: self.bar_width,
            handle_min_length: self.handle_min_length,
            bar_inner_margin: self.bar_inner_margin,
            bar_outer_margin: self.bar_outer_margin,
            floating_width: self.floating_width,
            floating_allocated_width: self.floating_allocated_width,
            foreground_color: self.foreground_color,
            dormant_background_opacity: self.dormant_background_opacity,
            active_background_opacity: self.active_background_opacity,
            interact_background_opacity: self.interact_background_opacity,
            dormant_handle_opacity: self.dormant_handle_opacity,
            active_handle_opacity: self.active_handle_opacity,
            interact_handle_opacity: self.interact_handle_opacity,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct EguiThemeSettings {
    pub preset: ThemePreset,
    pub ui_sans: ThemeFontRoleSettings,
    pub ui_heading: ThemeFontRoleSettings,
    pub ui_mono: ThemeFontRoleSettings,
    pub text_sizes: ThemeTextSizes,
    pub base_colors: ThemeBaseColorsSettings,
    pub widget_states: ThemeWidgetStatesSettings,
    pub selection: ThemeSelectionSettings,
    pub windows_panels: ThemeWindowsPanelsSettings,
    pub spacing: ThemeSpacingSettings,
    pub scrollbars: ThemeScrollbarsSettings,
    pub motion: UiMotionSettings,
}

impl Default for EguiThemeSettings {
    fn default() -> Self {
        Self::from_preset(ThemePreset::StudioDark)
    }
}

pub struct EguiThemeBuild {
    pub theme: egui::Theme,
    pub style: egui::Style,
    pub fonts: FontDefinitions,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(default)]
pub struct UiMotionSettings {
    pub enabled: bool,
    pub reduced_motion: bool,
    pub surface_duration_s: f32,
    pub micro_duration_s: f32,
    pub toast_duration_s: f32,
    pub dock_duration_s: f32,
    pub surface_slide_px: f32,
    pub overlay_scale_delta: f32,
    pub dock_hover_emphasis: f32,
}

impl Default for UiMotionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reduced_motion: false,
            surface_duration_s: 0.17,
            micro_duration_s: 0.09,
            toast_duration_s: 0.22,
            dock_duration_s: 0.14,
            surface_slide_px: 10.0,
            overlay_scale_delta: 0.02,
            dock_hover_emphasis: 0.65,
        }
    }
}

impl UiMotionSettings {
    const REDUCED_MOTION_DURATION_FACTOR: f32 = 0.4;

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn effective_duration(self, duration_s: f32) -> f32 {
        if !self.enabled {
            0.0
        } else if self.reduced_motion {
            (duration_s.max(0.0)) * Self::REDUCED_MOTION_DURATION_FACTOR
        } else {
            duration_s.max(0.0)
        }
    }

    pub fn surface_duration(self) -> f32 {
        self.effective_duration(self.surface_duration_s)
    }

    pub fn micro_duration(self) -> f32 {
        self.effective_duration(self.micro_duration_s)
    }

    pub fn toast_duration(self) -> f32 {
        self.effective_duration(self.toast_duration_s)
    }

    pub fn dock_duration(self) -> f32 {
        self.effective_duration(self.dock_duration_s)
    }

    pub fn effective_slide_px(self) -> f32 {
        if self.reduced_motion {
            0.0
        } else {
            self.surface_slide_px.max(0.0)
        }
    }

    pub fn effective_overlay_scale_delta(self) -> f32 {
        if self.reduced_motion {
            0.0
        } else {
            self.overlay_scale_delta.clamp(0.0, 0.25)
        }
    }

    pub fn effective_dock_hover_emphasis(self) -> f32 {
        if self.enabled {
            self.dock_hover_emphasis.clamp(0.0, 2.0)
        } else {
            0.0
        }
    }
}

impl EguiThemeSettings {
    pub fn from_preset(preset: ThemePreset) -> Self {
        let style = match preset {
            ThemePreset::StudioDark => build_studio_dark_style(),
            ThemePreset::SlateLight => build_slate_light_style(),
            ThemePreset::HighContrastDark => build_high_contrast_dark_style(),
        };

        Self {
            preset,
            ui_sans: ThemeFontRoleSettings::default(),
            ui_heading: ThemeFontRoleSettings::default(),
            ui_mono: ThemeFontRoleSettings::default(),
            text_sizes: ThemeTextSizes::from_style(&style),
            base_colors: ThemeBaseColorsSettings::from_visuals(&style.visuals),
            widget_states: ThemeWidgetStatesSettings::from_egui(&style.visuals.widgets),
            selection: ThemeSelectionSettings::from_visuals(&style.visuals),
            windows_panels: ThemeWindowsPanelsSettings::from_egui(&style.visuals),
            spacing: ThemeSpacingSettings::from_egui(&style.spacing),
            scrollbars: ThemeScrollbarsSettings::from_egui(style.spacing.scroll),
            motion: UiMotionSettings::default(),
        }
    }

    pub fn reset_to_preset(&mut self) {
        *self = Self::from_preset(self.preset);
    }

    pub fn fingerprint(&self) -> u64 {
        let json = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        json.hash(&mut hasher);
        hasher.finish()
    }

    pub fn font_role(&self, role: ThemeFontRole) -> &ThemeFontRoleSettings {
        match role {
            ThemeFontRole::UiSans => &self.ui_sans,
            ThemeFontRole::UiHeading => &self.ui_heading,
            ThemeFontRole::UiMono => &self.ui_mono,
        }
    }

    pub fn font_role_mut(&mut self, role: ThemeFontRole) -> &mut ThemeFontRoleSettings {
        match role {
            ThemeFontRole::UiSans => &mut self.ui_sans,
            ThemeFontRole::UiHeading => &mut self.ui_heading,
            ThemeFontRole::UiMono => &mut self.ui_mono,
        }
    }

    pub fn use_built_in_font(&mut self, role: ThemeFontRole) {
        self.font_role_mut(role).source = ThemeFontSource::BundledDefault;
    }

    pub fn build(&self) -> EguiThemeBuild {
        let mut style = self.preset.egui_theme().default_style();
        let mut warnings = Vec::new();
        let mut fonts = FontDefinitions::default();

        self.apply_to_style(&mut style);
        self.install_font_role(&mut fonts, ThemeFontRole::UiSans, &mut warnings);
        self.install_font_role(&mut fonts, ThemeFontRole::UiHeading, &mut warnings);
        self.install_font_role(&mut fonts, ThemeFontRole::UiMono, &mut warnings);

        EguiThemeBuild {
            theme: self.preset.egui_theme(),
            style,
            fonts,
            warnings,
        }
    }

    fn apply_to_style(&self, style: &mut egui::Style) {
        self.text_sizes.apply_to_style(style);
        self.base_colors.apply_to_visuals(&mut style.visuals);
        style.visuals.widgets = self.widget_states.to_egui();
        style.visuals.selection = self.selection.to_egui();
        self.windows_panels.apply_to_visuals(&mut style.visuals);
        self.spacing.apply_to_spacing(&mut style.spacing);
        style.spacing.scroll = self.scrollbars.to_egui();
        style.animation_time = self.motion.micro_duration();
    }

    fn install_font_role(
        &self,
        fonts: &mut FontDefinitions,
        role: ThemeFontRole,
        warnings: &mut Vec<String>,
    ) {
        let role_settings = self.font_role(role);
        let fallback_family = role.fallback_family();
        let fallback_names = fonts
            .families
            .get(&fallback_family)
            .cloned()
            .unwrap_or_default();

        let primary_font = match self.load_font_data(fonts, role, warnings) {
            Some(font) => font,
            None => return,
        };

        fonts.font_data.insert(
            role.font_key().to_string(),
            primary_font.tweak(role_settings.tweak.to_egui()),
        );

        let mut family_names = vec![role.font_key().to_string()];
        family_names.extend(fallback_names);
        fonts
            .families
            .insert(FontFamily::Name(role.family_name().into()), family_names);
    }

    fn load_font_data(
        &self,
        fonts: &FontDefinitions,
        role: ThemeFontRole,
        warnings: &mut Vec<String>,
    ) -> Option<FontData> {
        match &self.font_role(role).source {
            ThemeFontSource::BundledDefault => self.default_font_data(fonts, role, warnings),
            ThemeFontSource::Imported { relative_path } => {
                match load_imported_font_bytes(relative_path) {
                    Ok(bytes) => Some(FontData::from_owned(bytes)),
                    Err(error) => {
                        warnings.push(format!(
                            "{} font import failed: {}. Falling back to built-in egui fonts.",
                            role.label(),
                            error
                        ));
                        self.default_font_data(fonts, role, warnings)
                    }
                }
            }
        }
    }

    fn default_font_data(
        &self,
        fonts: &FontDefinitions,
        role: ThemeFontRole,
        warnings: &mut Vec<String>,
    ) -> Option<FontData> {
        match fonts.font_data.get(role.default_font_name()).cloned() {
            Some(font) => Some(font),
            None => {
                warnings.push(format!(
                    "Built-in egui font {:?} was unavailable for {}.",
                    role.default_font_name(),
                    role.label()
                ));
                None
            }
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    pub fn import_font_from_dialog(
        &mut self,
        role: ThemeFontRole,
    ) -> Result<Option<String>, String> {
        match crate::desktop_dialogs::font_import_dialog(role.label()) {
            crate::desktop_dialogs::FileDialogSelection::Selected(path) => {
                self.import_font_from_path(role, &path)?;
                Ok(Some(format!(
                    "{} imported from {}",
                    role.label(),
                    path.display()
                )))
            }
            crate::desktop_dialogs::FileDialogSelection::Cancelled => Ok(None),
            crate::desktop_dialogs::FileDialogSelection::Unsupported => {
                Err("Font import is not available on this platform.".to_string())
            }
        }
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    pub fn import_font_from_dialog(
        &mut self,
        role: ThemeFontRole,
    ) -> Result<Option<String>, String> {
        let _ = role;
        Err("Font import is not available on this platform.".to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn import_font_from_path(
        &mut self,
        role: ThemeFontRole,
        path: &Path,
    ) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("Failed to read {}: {}", path.display(), error))?;
        validate_font_bytes(&bytes).map_err(|error| {
            format!(
                "{} is not a valid .ttf/.otf font: {}",
                path.display(),
                error
            )
        })?;

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .filter(|ext| ext == "ttf" || ext == "otf")
            .unwrap_or_else(|| "ttf".to_string());
        let relative_path = format!("fonts/{}.{}", role.storage_stem(), extension);
        let destination = resolve_storage_relative_path(&relative_path)?;

        crate::native_paths::ensure_parent_dir(&destination)?;
        std::fs::write(&destination, bytes).map_err(|error| {
            format!(
                "Failed to copy font to {}: {}",
                destination.display(),
                error
            )
        })?;

        let role_settings = self.font_role_mut(role);
        role_settings.source = ThemeFontSource::Imported {
            relative_path: relative_path.clone(),
        };
        role_settings.import_serial = role_settings.import_serial.saturating_add(1);

        Ok(relative_path)
    }
}

impl ThemeTextSizes {
    fn from_style(style: &egui::Style) -> Self {
        Self {
            small: font_size(style, TextStyle::Small),
            body: font_size(style, TextStyle::Body),
            button: font_size(style, TextStyle::Button),
            monospace: font_size(style, TextStyle::Monospace),
            heading: font_size(style, TextStyle::Heading),
            viewport_hud: DEFAULT_VIEWPORT_HUD_POINTS,
            viewport_mono: DEFAULT_VIEWPORT_MONO_POINTS,
            scene_label: DEFAULT_SCENE_LABEL_POINTS,
        }
    }

    fn apply_to_style(&self, style: &mut egui::Style) {
        style.text_styles.insert(
            TextStyle::Small,
            FontId::new(self.small, FontFamily::Name(UI_SANS_FAMILY_NAME.into())),
        );
        style.text_styles.insert(
            TextStyle::Body,
            FontId::new(self.body, FontFamily::Name(UI_SANS_FAMILY_NAME.into())),
        );
        style.text_styles.insert(
            TextStyle::Button,
            FontId::new(self.button, FontFamily::Name(UI_SANS_FAMILY_NAME.into())),
        );
        style.text_styles.insert(
            TextStyle::Monospace,
            FontId::new(self.monospace, FontFamily::Name(UI_MONO_FAMILY_NAME.into())),
        );
        style.text_styles.insert(
            TextStyle::Heading,
            FontId::new(
                self.heading,
                FontFamily::Name(UI_HEADING_FAMILY_NAME.into()),
            ),
        );
        style.text_styles.insert(
            TextStyle::Name(VIEWPORT_HUD_STYLE_NAME.into()),
            FontId::new(
                self.viewport_hud,
                FontFamily::Name(UI_SANS_FAMILY_NAME.into()),
            ),
        );
        style.text_styles.insert(
            TextStyle::Name(VIEWPORT_MONO_STYLE_NAME.into()),
            FontId::new(
                self.viewport_mono,
                FontFamily::Name(UI_MONO_FAMILY_NAME.into()),
            ),
        );
        style.text_styles.insert(
            TextStyle::Name(SCENE_LABEL_STYLE_NAME.into()),
            FontId::new(
                self.scene_label,
                FontFamily::Name(UI_SANS_FAMILY_NAME.into()),
            ),
        );
    }
}

impl ThemeBaseColorsSettings {
    fn from_visuals(visuals: &egui::Visuals) -> Self {
        Self {
            override_text_color: ThemeOptionalColor::from_egui(visuals.override_text_color),
            hyperlink_color: color_to_rgba(visuals.hyperlink_color),
            faint_bg_color: color_to_rgba(visuals.faint_bg_color),
            extreme_bg_color: color_to_rgba(visuals.extreme_bg_color),
            code_bg_color: color_to_rgba(visuals.code_bg_color),
            warn_fg_color: color_to_rgba(visuals.warn_fg_color),
            error_fg_color: color_to_rgba(visuals.error_fg_color),
        }
    }

    fn apply_to_visuals(&self, visuals: &mut egui::Visuals) {
        visuals.override_text_color = self.override_text_color.to_egui();
        visuals.hyperlink_color = color_from_rgba(self.hyperlink_color);
        visuals.faint_bg_color = color_from_rgba(self.faint_bg_color);
        visuals.extreme_bg_color = color_from_rgba(self.extreme_bg_color);
        visuals.code_bg_color = color_from_rgba(self.code_bg_color);
        visuals.warn_fg_color = color_from_rgba(self.warn_fg_color);
        visuals.error_fg_color = color_from_rgba(self.error_fg_color);
    }
}

impl ThemeSelectionSettings {
    fn from_visuals(visuals: &egui::Visuals) -> Self {
        Self {
            bg_fill: color_to_rgba(visuals.selection.bg_fill),
            stroke: DockStroke::from_egui(visuals.selection.stroke),
        }
    }

    fn to_egui(self) -> egui::style::Selection {
        egui::style::Selection {
            bg_fill: color_from_rgba(self.bg_fill),
            stroke: self.stroke.to_egui(),
        }
    }
}

pub fn resolve_font_id(style: &egui::Style, role: AppTextRole) -> FontId {
    style
        .text_styles
        .get(&role.text_style())
        .cloned()
        .or_else(|| style.text_styles.get(&role.fallback_text_style()).cloned())
        .unwrap_or_else(|| role.fallback_text_style().resolve(style))
}

pub fn resolve_scaled_font_id(style: &egui::Style, role: AppTextRole, points: f32) -> FontId {
    let mut font = resolve_font_id(style, role);
    let scale = font.size / role.default_points().max(0.001);
    font.size = points * scale;
    font
}

pub fn viewport_hud_size_scale(style: &egui::Style) -> f32 {
    resolve_font_id(style, AppTextRole::ViewportHud).size / DEFAULT_VIEWPORT_HUD_POINTS
}

pub fn viewport_mono_size_scale(style: &egui::Style) -> f32 {
    resolve_font_id(style, AppTextRole::ViewportMono).size / DEFAULT_VIEWPORT_MONO_POINTS
}

pub fn scene_label_size_scale(style: &egui::Style) -> f32 {
    resolve_font_id(style, AppTextRole::SceneLabel).size / DEFAULT_SCENE_LABEL_POINTS
}

fn font_size(style: &egui::Style, text_style: TextStyle) -> f32 {
    text_style.resolve(style).size
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_storage_relative_path(relative_path: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative_path);
    if path.is_absolute() {
        return Err("Imported font paths must stay inside app storage.".to_string());
    }

    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err("Imported font paths must stay inside app storage.".to_string());
            }
        }
    }

    Ok(crate::native_paths::app_storage_file(relative_path))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_imported_font_bytes(relative_path: &str) -> Result<Vec<u8>, String> {
    let path = resolve_storage_relative_path(relative_path)?;
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("Failed to read {}: {}", path.display(), error))?;
    validate_font_bytes(&bytes)
        .map_err(|error| format!("{} is not a valid font: {}", path.display(), error))?;
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn load_imported_font_bytes(relative_path: &str) -> Result<Vec<u8>, String> {
    let _ = relative_path;
    Err("Imported fonts are not supported on this platform.".to_string())
}

fn validate_font_bytes(bytes: &[u8]) -> Result<(), String> {
    ab_glyph::FontRef::try_from_slice(bytes)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn build_studio_dark_style() -> egui::Style {
    let mut style = egui::Theme::Dark.default_style();
    style.spacing.item_spacing = egui::vec2(10.0, 7.0);
    style.spacing.button_padding = egui::vec2(11.0, 7.0);
    style.spacing.window_margin = egui::Margin::same(14.0);
    style.spacing.menu_margin = egui::Margin::same(10.0);
    style.spacing.interact_size = egui::vec2(42.0, 28.0);
    style.spacing.slider_width = 168.0;
    style.spacing.combo_width = 148.0;
    style.spacing.text_edit_width = 196.0;
    style.spacing.tooltip_width = 420.0;
    style.spacing.menu_width = 220.0;
    style.spacing.menu_spacing = 4.0;
    style.spacing.combo_height = 220.0;
    style.spacing.scroll = ScrollStyle::floating();

    style.visuals.override_text_color = Some(Color32::from_rgb(232, 236, 243));
    style.visuals.hyperlink_color = Color32::from_rgb(124, 162, 255);
    style.visuals.faint_bg_color = Color32::from_rgb(28, 33, 43);
    style.visuals.extreme_bg_color = Color32::from_rgb(10, 13, 18);
    style.visuals.code_bg_color = Color32::from_rgb(18, 23, 31);
    style.visuals.warn_fg_color = Color32::from_rgb(251, 191, 36);
    style.visuals.error_fg_color = Color32::from_rgb(248, 113, 113);
    style.visuals.panel_fill = Color32::from_rgb(14, 18, 24);
    style.visuals.window_fill = Color32::from_rgb(17, 22, 30);
    style.visuals.window_stroke = egui::Stroke::new(1.0, Color32::from_rgb(54, 63, 79));
    style.visuals.window_rounding = egui::Rounding::same(14.0);
    style.visuals.menu_rounding = egui::Rounding::same(12.0);
    style.visuals.resize_corner_size = 10.0;
    style.visuals.clip_rect_margin = 4.0;
    style.visuals.button_frame = true;
    style.visuals.collapsing_header_frame = false;
    style.visuals.indent_has_left_vline = false;
    style.visuals.striped = false;
    style.visuals.slider_trailing_fill = true;
    style.visuals.selection = egui::style::Selection {
        bg_fill: Color32::from_rgba_unmultiplied(70, 103, 179, 92),
        stroke: egui::Stroke::new(1.0, Color32::from_rgb(147, 197, 253)),
    };
    style.visuals.widgets = egui::style::Widgets {
        noninteractive: widget_visuals(
            Color32::from_rgb(16, 20, 27),
            Color32::from_rgb(16, 20, 27),
            Color32::from_rgb(53, 61, 76),
            Color32::from_rgb(232, 236, 243),
            12.0,
            0.0,
        ),
        inactive: widget_visuals(
            Color32::from_rgb(27, 33, 43),
            Color32::from_rgb(24, 29, 38),
            Color32::from_rgb(67, 78, 96),
            Color32::from_rgb(233, 237, 244),
            12.0,
            0.0,
        ),
        hovered: widget_visuals(
            Color32::from_rgb(35, 42, 54),
            Color32::from_rgb(31, 37, 48),
            Color32::from_rgb(96, 121, 169),
            Color32::from_rgb(245, 247, 250),
            12.0,
            1.0,
        ),
        active: widget_visuals(
            Color32::from_rgb(41, 49, 63),
            Color32::from_rgb(36, 43, 56),
            Color32::from_rgb(129, 161, 228),
            Color32::from_rgb(255, 255, 255),
            12.0,
            1.0,
        ),
        open: widget_visuals(
            Color32::from_rgb(32, 39, 51),
            Color32::from_rgb(28, 34, 45),
            Color32::from_rgb(102, 134, 194),
            Color32::from_rgb(246, 248, 251),
            12.0,
            0.0,
        ),
    };
    style
}

fn build_slate_light_style() -> egui::Style {
    let mut style = egui::Theme::Light.default_style();
    style.spacing.item_spacing = egui::vec2(10.0, 7.0);
    style.spacing.button_padding = egui::vec2(11.0, 7.0);
    style.spacing.window_margin = egui::Margin::same(14.0);
    style.spacing.menu_margin = egui::Margin::same(10.0);
    style.spacing.interact_size = egui::vec2(42.0, 28.0);
    style.spacing.slider_width = 168.0;
    style.spacing.combo_width = 148.0;
    style.spacing.text_edit_width = 196.0;
    style.spacing.tooltip_width = 420.0;
    style.spacing.menu_width = 220.0;
    style.spacing.menu_spacing = 4.0;
    style.spacing.combo_height = 220.0;
    style.spacing.scroll = ScrollStyle::solid();

    style.visuals.override_text_color = Some(Color32::from_rgb(30, 41, 59));
    style.visuals.hyperlink_color = Color32::from_rgb(59, 130, 246);
    style.visuals.faint_bg_color = Color32::from_rgb(240, 244, 249);
    style.visuals.extreme_bg_color = Color32::from_rgb(226, 232, 240);
    style.visuals.code_bg_color = Color32::from_rgb(232, 237, 244);
    style.visuals.warn_fg_color = Color32::from_rgb(166, 104, 31);
    style.visuals.error_fg_color = Color32::from_rgb(181, 55, 55);
    style.visuals.panel_fill = Color32::from_rgb(248, 250, 252);
    style.visuals.window_fill = Color32::from_rgb(255, 255, 255);
    style.visuals.window_stroke = egui::Stroke::new(1.0, Color32::from_rgb(194, 204, 216));
    style.visuals.window_rounding = egui::Rounding::same(14.0);
    style.visuals.menu_rounding = egui::Rounding::same(12.0);
    style.visuals.resize_corner_size = 10.0;
    style.visuals.clip_rect_margin = 4.0;
    style.visuals.button_frame = true;
    style.visuals.collapsing_header_frame = false;
    style.visuals.indent_has_left_vline = false;
    style.visuals.striped = false;
    style.visuals.slider_trailing_fill = true;
    style.visuals.selection = egui::style::Selection {
        bg_fill: Color32::from_rgba_unmultiplied(96, 165, 250, 56),
        stroke: egui::Stroke::new(1.0, Color32::from_rgb(59, 130, 246)),
    };
    style.visuals.widgets = egui::style::Widgets {
        noninteractive: widget_visuals(
            Color32::from_rgb(255, 255, 255),
            Color32::from_rgb(255, 255, 255),
            Color32::from_rgb(194, 204, 216),
            Color32::from_rgb(30, 41, 59),
            12.0,
            0.0,
        ),
        inactive: widget_visuals(
            Color32::from_rgb(241, 245, 249),
            Color32::from_rgb(237, 242, 247),
            Color32::from_rgb(184, 194, 208),
            Color32::from_rgb(30, 41, 59),
            12.0,
            0.0,
        ),
        hovered: widget_visuals(
            Color32::from_rgb(228, 236, 247),
            Color32::from_rgb(221, 231, 244),
            Color32::from_rgb(99, 127, 189),
            Color32::from_rgb(23, 33, 50),
            12.0,
            1.0,
        ),
        active: widget_visuals(
            Color32::from_rgb(214, 228, 247),
            Color32::from_rgb(205, 221, 242),
            Color32::from_rgb(59, 130, 246),
            Color32::from_rgb(24, 35, 47),
            12.0,
            1.0,
        ),
        open: widget_visuals(
            Color32::from_rgb(224, 233, 245),
            Color32::from_rgb(215, 226, 242),
            Color32::from_rgb(82, 112, 178),
            Color32::from_rgb(26, 38, 49),
            12.0,
            0.0,
        ),
    };
    style
}

fn build_high_contrast_dark_style() -> egui::Style {
    let mut style = egui::Theme::Dark.default_style();
    style.spacing.item_spacing = egui::vec2(12.0, 9.0);
    style.spacing.button_padding = egui::vec2(11.0, 7.0);
    style.spacing.window_margin = egui::Margin::same(12.0);
    style.spacing.menu_margin = egui::Margin::same(10.0);
    style.spacing.interact_size = egui::vec2(42.0, 26.0);
    style.spacing.slider_width = 164.0;
    style.spacing.combo_width = 144.0;
    style.spacing.text_edit_width = 186.0;
    style.spacing.tooltip_width = 440.0;
    style.spacing.menu_width = 188.0;
    style.spacing.menu_spacing = 5.0;
    style.spacing.combo_height = 240.0;
    style.spacing.scroll = ScrollStyle::solid();

    style.visuals.override_text_color = Some(Color32::from_rgb(245, 247, 250));
    style.visuals.hyperlink_color = Color32::from_rgb(102, 191, 255);
    style.visuals.faint_bg_color = Color32::from_rgb(18, 20, 22);
    style.visuals.extreme_bg_color = Color32::from_rgb(8, 9, 10);
    style.visuals.code_bg_color = Color32::from_rgb(12, 13, 15);
    style.visuals.warn_fg_color = Color32::from_rgb(255, 204, 102);
    style.visuals.error_fg_color = Color32::from_rgb(255, 112, 112);
    style.visuals.panel_fill = Color32::from_rgb(10, 11, 13);
    style.visuals.window_fill = Color32::from_rgb(14, 16, 18);
    style.visuals.window_stroke = egui::Stroke::new(1.5, Color32::from_rgb(184, 189, 199));
    style.visuals.window_rounding = egui::Rounding::same(6.0);
    style.visuals.menu_rounding = egui::Rounding::same(6.0);
    style.visuals.resize_corner_size = 11.0;
    style.visuals.clip_rect_margin = 4.0;
    style.visuals.button_frame = true;
    style.visuals.collapsing_header_frame = true;
    style.visuals.indent_has_left_vline = true;
    style.visuals.striped = true;
    style.visuals.slider_trailing_fill = true;
    style.visuals.selection = egui::style::Selection {
        bg_fill: Color32::from_rgba_unmultiplied(84, 144, 255, 128),
        stroke: egui::Stroke::new(1.5, Color32::from_rgb(195, 224, 255)),
    };
    style.visuals.widgets = egui::style::Widgets {
        noninteractive: widget_visuals(
            Color32::from_rgb(14, 16, 18),
            Color32::from_rgb(14, 16, 18),
            Color32::from_rgb(176, 183, 193),
            Color32::from_rgb(245, 247, 250),
            6.0,
            0.0,
        ),
        inactive: widget_visuals(
            Color32::from_rgb(32, 35, 39),
            Color32::from_rgb(24, 27, 31),
            Color32::from_rgb(191, 198, 208),
            Color32::from_rgb(249, 250, 251),
            6.0,
            0.0,
        ),
        hovered: widget_visuals(
            Color32::from_rgb(45, 50, 56),
            Color32::from_rgb(36, 40, 45),
            Color32::from_rgb(136, 188, 255),
            Color32::from_rgb(255, 255, 255),
            6.0,
            1.0,
        ),
        active: widget_visuals(
            Color32::from_rgb(62, 68, 76),
            Color32::from_rgb(52, 58, 65),
            Color32::from_rgb(195, 224, 255),
            Color32::from_rgb(255, 255, 255),
            6.0,
            1.0,
        ),
        open: widget_visuals(
            Color32::from_rgb(52, 57, 63),
            Color32::from_rgb(43, 47, 53),
            Color32::from_rgb(160, 208, 255),
            Color32::from_rgb(255, 255, 255),
            6.0,
            0.0,
        ),
    };
    style
}

fn widget_visuals(
    bg_fill: Color32,
    weak_bg_fill: Color32,
    border_color: Color32,
    fg_color: Color32,
    rounding: f32,
    expansion: f32,
) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill,
        weak_bg_fill,
        bg_stroke: egui::Stroke::new(1.0, border_color),
        rounding: egui::Rounding::same(rounding),
        fg_stroke: egui::Stroke::new(1.0, fg_color),
        expansion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dock_style::DockStyleSettings;

    #[test]
    fn presets_round_trip_through_serde() {
        for preset in [
            ThemePreset::StudioDark,
            ThemePreset::SlateLight,
            ThemePreset::HighContrastDark,
        ] {
            let theme = EguiThemeSettings::from_preset(preset);
            let json = serde_json::to_string(&theme).expect("serialize theme");
            let restored: EguiThemeSettings =
                serde_json::from_str(&json).expect("deserialize theme");
            assert_eq!(restored, theme);
            assert_eq!(restored.preset, preset);
        }
    }

    #[test]
    fn motion_settings_round_trip_through_serde() {
        let motion = UiMotionSettings {
            reduced_motion: true,
            dock_hover_emphasis: 1.4,
            surface_slide_px: 22.0,
            ..Default::default()
        };

        let json = serde_json::to_string(&motion).expect("serialize motion");
        let restored: UiMotionSettings = serde_json::from_str(&json).expect("deserialize motion");
        assert_eq!(restored, motion);
    }

    #[test]
    fn theme_build_registers_custom_text_roles_and_dock_derives_from_style() {
        let theme = EguiThemeSettings::from_preset(ThemePreset::SlateLight);
        let build = theme.build();
        let ctx = egui::Context::default();
        ctx.set_theme(build.theme);
        ctx.set_fonts(build.fonts.clone());
        ctx.set_style(build.style.clone());

        let style = ctx.style();
        assert!(style
            .text_styles
            .contains_key(&TextStyle::Name(VIEWPORT_HUD_STYLE_NAME.into())));
        assert!(style
            .text_styles
            .contains_key(&TextStyle::Name(VIEWPORT_MONO_STYLE_NAME.into())));
        assert!(style
            .text_styles
            .contains_key(&TextStyle::Name(SCENE_LABEL_STYLE_NAME.into())));
        assert_eq!(build.style.animation_time, theme.motion.micro_duration());

        let dock_style = DockStyleSettings::default().to_egui_dock_style(style.as_ref());
        let expected = egui_dock::Style::from_egui(style.as_ref());
        assert_eq!(dock_style.tab_bar.height, expected.tab_bar.height);
        assert_eq!(dock_style.tab_bar.bg_fill, expected.tab_bar.bg_fill);
        assert_eq!(
            dock_style.buttons.close_tab_color,
            expected.buttons.close_tab_color
        );
        assert_eq!(dock_style.separator.width, expected.separator.width);
    }

    #[test]
    fn missing_imported_font_falls_back_without_panicking() {
        let mut theme = EguiThemeSettings::from_preset(ThemePreset::StudioDark);
        theme.ui_sans.source = ThemeFontSource::Imported {
            relative_path: "fonts/missing-font.ttf".to_string(),
        };

        let build = theme.build();
        assert!(!build.warnings.is_empty());
        assert!(build
            .fonts
            .font_data
            .contains_key(ThemeFontRole::UiSans.font_key()));
    }

    #[test]
    fn resolve_font_id_falls_back_when_custom_roles_are_absent() {
        let style = egui::Style::default();

        assert_eq!(
            resolve_font_id(&style, AppTextRole::ViewportHud),
            TextStyle::Body.resolve(&style)
        );
        assert_eq!(
            resolve_font_id(&style, AppTextRole::ViewportMono),
            TextStyle::Monospace.resolve(&style)
        );
    }
}
