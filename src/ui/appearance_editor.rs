use eframe::egui::{self, Color32};

use crate::app::actions::{Action, ActionSink};
use crate::dock_style::ColorRgba;
use crate::dock_style::DockStyleSettings;
use crate::egui_theme::{
    EguiThemeSettings, ThemeFontRole, ThemeFontTweakSettings, ThemeOptionalColor, ThemePreset,
    ThemeScrollbarsSettings, ThemeSpacingSettings, ThemeWidgetStatesSettings,
    ThemeWidgetVisualsSettings, ThemeWindowsPanelsSettings, UiMotionSettings,
};

pub fn draw(
    ui: &mut egui::Ui,
    theme: &mut EguiThemeSettings,
    dock_style: &mut DockStyleSettings,
    actions: &mut ActionSink,
) {
    let mut requested_import = None;

    egui::CollapsingHeader::new("Appearance")
        .default_open(true)
        .show(ui, |ui| {
            draw_preset_section(ui, theme);
            draw_typography_section(ui, theme, &mut requested_import);
            draw_text_sizes_section(ui, theme);
            draw_base_colors_section(ui, theme);
            draw_widget_states_section(ui, &mut theme.widget_states);
            draw_windows_panels_section(ui, &mut theme.windows_panels);
            draw_layout_spacing_section(ui, &mut theme.spacing);
            draw_scrollbars_section(ui, &mut theme.scrollbars);
            draw_motion_section(ui, &mut theme.motion);

            egui::CollapsingHeader::new("Existing Dock Styling")
                .default_open(false)
                .show(ui, |ui| {
                    crate::ui::dock_style_editor::draw_embedded(ui, dock_style)
                });
        });

    if let Some(role) = requested_import {
        match theme.import_font_from_dialog(role) {
            Ok(Some(message)) => actions.push(Action::ShowToast {
                message,
                is_error: false,
            }),
            Ok(None) => {}
            Err(message) => actions.push(Action::ShowToast {
                message,
                is_error: true,
            }),
        }
    }
}

fn draw_preset_section(ui: &mut egui::Ui, theme: &mut EguiThemeSettings) {
    egui::CollapsingHeader::new("Preset")
        .default_open(true)
        .show(ui, |ui| {
            let mut selected = theme.preset;
            egui::ComboBox::from_id_salt("appearance_theme_preset")
                .selected_text(selected.label())
                .show_ui(ui, |ui| {
                    for preset in [
                        ThemePreset::StudioDark,
                        ThemePreset::SlateLight,
                        ThemePreset::HighContrastDark,
                    ] {
                        ui.selectable_value(&mut selected, preset, preset.label());
                    }
                });

            if selected != theme.preset {
                *theme = EguiThemeSettings::from_preset(selected);
            }

            ui.horizontal_wrapped(|ui| {
                if ui.button("Reset to Preset").clicked() {
                    theme.reset_to_preset();
                }
                ui.weak("Presets replace the full editable egui appearance state.");
            });
        });
}

fn draw_typography_section(
    ui: &mut egui::Ui,
    theme: &mut EguiThemeSettings,
    requested_import: &mut Option<ThemeFontRole>,
) {
    egui::CollapsingHeader::new("Typography")
        .default_open(true)
        .show(ui, |ui| {
            ui.weak(
                "Built-in egui fonts are the default. You can optionally import trusted local .ttf/.otf fonts per role.",
            );
            ui.weak(
                "egui 0.29 supports family mapping, sizes, and font tweaks here, but not arbitrary bold/italic styling per widget.",
            );
            ui.separator();

            for role in ThemeFontRole::ALL {
                let source_label = theme.font_role(role).source.label();
                egui::CollapsingHeader::new(role.label())
                    .default_open(role == ThemeFontRole::UiSans)
                    .show(ui, |ui| {
                        ui.label(source_label);
                        ui.horizontal_wrapped(|ui| {
                            #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
                            if ui.button("Import Font...").clicked() {
                                *requested_import = Some(role);
                            }

                            if ui.button("Use Built-In").clicked() {
                                theme.use_built_in_font(role);
                            }
                        });

                        #[cfg(any(target_arch = "wasm32", target_os = "android"))]
                        ui.weak("Font import is disabled on this platform.");

                        draw_font_tweak_editor(ui, &mut theme.font_role_mut(role).tweak);
                    });
            }
        });
}

fn draw_text_sizes_section(ui: &mut egui::Ui, theme: &mut EguiThemeSettings) {
    egui::CollapsingHeader::new("Text Sizes")
        .default_open(false)
        .show(ui, |ui| {
            draw_float_row(ui, "Small", &mut theme.text_sizes.small, 8.0..=20.0);
            draw_float_row(ui, "Body", &mut theme.text_sizes.body, 10.0..=26.0);
            draw_float_row(ui, "Button", &mut theme.text_sizes.button, 10.0..=26.0);
            draw_float_row(
                ui,
                "Monospace",
                &mut theme.text_sizes.monospace,
                10.0..=24.0,
            );
            draw_float_row(ui, "Heading", &mut theme.text_sizes.heading, 14.0..=40.0);
            draw_float_row(
                ui,
                "Viewport HUD",
                &mut theme.text_sizes.viewport_hud,
                9.0..=28.0,
            );
            draw_float_row(
                ui,
                "Viewport Mono",
                &mut theme.text_sizes.viewport_mono,
                8.0..=22.0,
            );
            draw_float_row(
                ui,
                "Scene Label",
                &mut theme.text_sizes.scene_label,
                8.0..=24.0,
            );
        });
}

fn draw_base_colors_section(ui: &mut egui::Ui, theme: &mut EguiThemeSettings) {
    egui::CollapsingHeader::new("Base Colors")
        .default_open(false)
        .show(ui, |ui| {
            draw_optional_color_row(
                ui,
                "Override text color",
                &mut theme.base_colors.override_text_color,
            );
            draw_color_row(ui, "Hyperlink", &mut theme.base_colors.hyperlink_color);
            draw_color_row(
                ui,
                "Faint background",
                &mut theme.base_colors.faint_bg_color,
            );
            draw_color_row(
                ui,
                "Extreme background",
                &mut theme.base_colors.extreme_bg_color,
            );
            draw_color_row(ui, "Code background", &mut theme.base_colors.code_bg_color);
            draw_color_row(
                ui,
                "Warning foreground",
                &mut theme.base_colors.warn_fg_color,
            );
            draw_color_row(
                ui,
                "Error foreground",
                &mut theme.base_colors.error_fg_color,
            );

            ui.separator();
            draw_color_row(ui, "Selection fill", &mut theme.selection.bg_fill);
            draw_stroke_row(
                ui,
                "Selection stroke",
                &mut theme.selection.stroke,
                0.0..=4.0,
            );
        });
}

fn draw_widget_states_section(ui: &mut egui::Ui, widget_states: &mut ThemeWidgetStatesSettings) {
    egui::CollapsingHeader::new("Widget States")
        .default_open(false)
        .show(ui, |ui| {
            draw_widget_visuals_group(ui, "Noninteractive", &mut widget_states.noninteractive);
            draw_widget_visuals_group(ui, "Inactive", &mut widget_states.inactive);
            draw_widget_visuals_group(ui, "Hovered", &mut widget_states.hovered);
            draw_widget_visuals_group(ui, "Active", &mut widget_states.active);
            draw_widget_visuals_group(ui, "Open", &mut widget_states.open);
        });
}

fn draw_windows_panels_section(ui: &mut egui::Ui, settings: &mut ThemeWindowsPanelsSettings) {
    egui::CollapsingHeader::new("Windows & Panels")
        .default_open(false)
        .show(ui, |ui| {
            draw_color_row(ui, "Window fill", &mut settings.window_fill);
            draw_stroke_row(ui, "Window stroke", &mut settings.window_stroke, 0.0..=4.0);
            draw_rounding_row(
                ui,
                "Window rounding",
                &mut settings.window_rounding,
                0.0..=24.0,
            );
            draw_rounding_row(ui, "Menu rounding", &mut settings.menu_rounding, 0.0..=24.0);
            draw_color_row(ui, "Panel fill", &mut settings.panel_fill);
            draw_float_row(
                ui,
                "Resize corner size",
                &mut settings.resize_corner_size,
                4.0..=20.0,
            );
            draw_float_row(
                ui,
                "Clip rect margin",
                &mut settings.clip_rect_margin,
                0.0..=12.0,
            );
            ui.checkbox(&mut settings.button_frame, "Button frames");
            ui.checkbox(
                &mut settings.collapsing_header_frame,
                "Collapsing header frames",
            );
            ui.checkbox(&mut settings.indent_has_left_vline, "Indented guide line");
            ui.checkbox(&mut settings.striped, "Striped grids/tables");
            ui.checkbox(&mut settings.slider_trailing_fill, "Slider trailing fill");
        });
}

fn draw_layout_spacing_section(ui: &mut egui::Ui, spacing: &mut ThemeSpacingSettings) {
    egui::CollapsingHeader::new("Layout & Spacing")
        .default_open(false)
        .show(ui, |ui| {
            draw_vec2_row(ui, "Item spacing", &mut spacing.item_spacing, 0.0..=24.0);
            draw_margin_row(ui, "Window margin", &mut spacing.window_margin, 0.0..=24.0);
            draw_vec2_row(
                ui,
                "Button padding",
                &mut spacing.button_padding,
                0.0..=24.0,
            );
            draw_margin_row(ui, "Menu margin", &mut spacing.menu_margin, 0.0..=24.0);
            draw_float_row(ui, "Indent", &mut spacing.indent, 0.0..=40.0);
            draw_vec2_row(ui, "Interact size", &mut spacing.interact_size, 8.0..=64.0);
            draw_float_row(ui, "Slider width", &mut spacing.slider_width, 60.0..=320.0);
            draw_float_row(
                ui,
                "Slider rail height",
                &mut spacing.slider_rail_height,
                2.0..=20.0,
            );
            draw_float_row(ui, "Combo width", &mut spacing.combo_width, 60.0..=320.0);
            draw_float_row(
                ui,
                "Text edit width",
                &mut spacing.text_edit_width,
                60.0..=420.0,
            );
            draw_float_row(
                ui,
                "Tooltip width",
                &mut spacing.tooltip_width,
                120.0..=640.0,
            );
            draw_float_row(ui, "Menu width", &mut spacing.menu_width, 80.0..=420.0);
            draw_float_row(ui, "Menu spacing", &mut spacing.menu_spacing, 0.0..=20.0);
            draw_float_row(ui, "Combo height", &mut spacing.combo_height, 80.0..=420.0);
        });
}

fn draw_scrollbars_section(ui: &mut egui::Ui, scrollbars: &mut ThemeScrollbarsSettings) {
    egui::CollapsingHeader::new("Scrollbars")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut scrollbars.floating, "Floating scrollbars");
            ui.checkbox(&mut scrollbars.foreground_color, "High contrast handles");
            draw_float_row(ui, "Bar width", &mut scrollbars.bar_width, 2.0..=20.0);
            draw_float_row(
                ui,
                "Handle minimum",
                &mut scrollbars.handle_min_length,
                8.0..=80.0,
            );
            draw_float_row(
                ui,
                "Inner margin",
                &mut scrollbars.bar_inner_margin,
                0.0..=16.0,
            );
            draw_float_row(
                ui,
                "Outer margin",
                &mut scrollbars.bar_outer_margin,
                0.0..=16.0,
            );
            draw_float_row(
                ui,
                "Floating width",
                &mut scrollbars.floating_width,
                1.0..=20.0,
            );
            draw_float_row(
                ui,
                "Floating allocated width",
                &mut scrollbars.floating_allocated_width,
                0.0..=20.0,
            );
            draw_float_row(
                ui,
                "Dormant background opacity",
                &mut scrollbars.dormant_background_opacity,
                0.0..=1.0,
            );
            draw_float_row(
                ui,
                "Active background opacity",
                &mut scrollbars.active_background_opacity,
                0.0..=1.0,
            );
            draw_float_row(
                ui,
                "Interact background opacity",
                &mut scrollbars.interact_background_opacity,
                0.0..=1.0,
            );
            draw_float_row(
                ui,
                "Dormant handle opacity",
                &mut scrollbars.dormant_handle_opacity,
                0.0..=1.0,
            );
            draw_float_row(
                ui,
                "Active handle opacity",
                &mut scrollbars.active_handle_opacity,
                0.0..=1.0,
            );
            draw_float_row(
                ui,
                "Interact handle opacity",
                &mut scrollbars.interact_handle_opacity,
                0.0..=1.0,
            );
        });
}

fn draw_motion_section(ui: &mut egui::Ui, motion: &mut UiMotionSettings) {
    egui::CollapsingHeader::new("Motion")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut motion.enabled, "Enable animations");
            ui.checkbox(&mut motion.reduced_motion, "Reduced motion");
            ui.separator();

            draw_float_row(
                ui,
                "Surface duration",
                &mut motion.surface_duration_s,
                0.0..=0.8,
            );
            draw_float_row(
                ui,
                "Micro duration",
                &mut motion.micro_duration_s,
                0.0..=0.5,
            );
            draw_float_row(
                ui,
                "Toast duration",
                &mut motion.toast_duration_s,
                0.0..=0.8,
            );
            draw_float_row(ui, "Dock duration", &mut motion.dock_duration_s, 0.0..=0.6);
            draw_float_row(
                ui,
                "Surface slide px",
                &mut motion.surface_slide_px,
                0.0..=40.0,
            );
            draw_float_row(
                ui,
                "Overlay scale delta",
                &mut motion.overlay_scale_delta,
                0.0..=0.16,
            );
            draw_float_row(
                ui,
                "Dock hover emphasis",
                &mut motion.dock_hover_emphasis,
                0.0..=2.0,
            );

            ui.horizontal_wrapped(|ui| {
                if ui.button("Reset Motion").clicked() {
                    motion.reset();
                }
                ui.weak("Reduced motion keeps fades and removes slide/scale motion.");
            });
        });
}

fn draw_font_tweak_editor(ui: &mut egui::Ui, tweak: &mut ThemeFontTweakSettings) {
    draw_float_row(ui, "Glyph scale", &mut tweak.scale, 0.6..=1.6);
    draw_float_row(
        ui,
        "Y offset factor",
        &mut tweak.y_offset_factor,
        -0.5..=0.5,
    );
    draw_float_row(ui, "Y offset", &mut tweak.y_offset, -8.0..=8.0);
    draw_float_row(
        ui,
        "Baseline offset factor",
        &mut tweak.baseline_offset_factor,
        -0.5..=0.5,
    );
}

fn draw_widget_visuals_group(
    ui: &mut egui::Ui,
    label: &str,
    visuals: &mut ThemeWidgetVisualsSettings,
) {
    egui::CollapsingHeader::new(label)
        .default_open(label == "Inactive")
        .show(ui, |ui| {
            draw_color_row(ui, "Background", &mut visuals.bg_fill);
            draw_color_row(ui, "Weak background", &mut visuals.weak_bg_fill);
            draw_stroke_row(ui, "Background stroke", &mut visuals.bg_stroke, 0.0..=4.0);
            draw_stroke_row(ui, "Foreground stroke", &mut visuals.fg_stroke, 0.0..=4.0);
            draw_rounding_row(ui, "Rounding", &mut visuals.rounding, 0.0..=24.0);
            draw_float_row(ui, "Expansion", &mut visuals.expansion, -4.0..=8.0);
        });
}

fn draw_color_row(ui: &mut egui::Ui, label: &str, color: &mut ColorRgba) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut egui_color = color32_from_rgba(*color);
        if ui.color_edit_button_srgba(&mut egui_color).changed() {
            *color = rgba_from_color32(egui_color);
        }
    });
}

fn draw_optional_color_row(ui: &mut egui::Ui, label: &str, color: &mut ThemeOptionalColor) {
    ui.horizontal(|ui| {
        ui.checkbox(&mut color.enabled, label);
        if color.enabled {
            let mut egui_color = color32_from_rgba(color.color);
            if ui.color_edit_button_srgba(&mut egui_color).changed() {
                color.color = rgba_from_color32(egui_color);
            }
        }
    });
}

fn draw_float_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(value, range).show_value(true));
    });
}

fn draw_stroke_row(
    ui: &mut egui::Ui,
    label: &str,
    stroke: &mut crate::dock_style::DockStroke,
    width_range: std::ops::RangeInclusive<f32>,
) {
    ui.label(label);
    draw_float_row(ui, "Width", &mut stroke.width, width_range);
    draw_color_row(ui, "Color", &mut stroke.color);
}

fn draw_rounding_row(
    ui: &mut egui::Ui,
    label: &str,
    rounding: &mut crate::dock_style::DockRounding,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.label(label);
    draw_float_row(ui, "NW", &mut rounding.nw, range.clone());
    draw_float_row(ui, "NE", &mut rounding.ne, range.clone());
    draw_float_row(ui, "SW", &mut rounding.sw, range.clone());
    draw_float_row(ui, "SE", &mut rounding.se, range);
}

fn draw_margin_row(
    ui: &mut egui::Ui,
    label: &str,
    margin: &mut crate::dock_style::DockEdgeInsets,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.label(label);
    draw_float_row(ui, "Left", &mut margin.left, range.clone());
    draw_float_row(ui, "Right", &mut margin.right, range.clone());
    draw_float_row(ui, "Top", &mut margin.top, range.clone());
    draw_float_row(ui, "Bottom", &mut margin.bottom, range);
}

fn draw_vec2_row(
    ui: &mut egui::Ui,
    label: &str,
    vec: &mut crate::egui_theme::ThemeVec2,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.label(label);
    draw_float_row(ui, "X", &mut vec.x, range.clone());
    draw_float_row(ui, "Y", &mut vec.y, range);
}

fn color32_from_rgba(color: ColorRgba) -> Color32 {
    Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3])
}

fn rgba_from_color32(color: Color32) -> ColorRgba {
    [color.r(), color.g(), color.b(), color.a()]
}
