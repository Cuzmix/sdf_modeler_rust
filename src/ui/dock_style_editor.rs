use eframe::egui;

use crate::dock_style::{
    ColorRgba, DockEdgeInsets, DockLeafHighlightingSettings, DockOverlayFeelSettings,
    DockOverlayStyleSettings, DockOverlayType, DockRounding, DockStroke, DockStyleSettings,
    DockTabAddAlign, DockTabInteractionStyleSettings, DockTabStyleSettings,
};

#[allow(dead_code)]
pub fn draw(ui: &mut egui::Ui, dock_style: &mut DockStyleSettings) {
    egui::CollapsingHeader::new("Dock Styling")
        .default_open(false)
        .show(ui, |ui| draw_contents(ui, dock_style));
}

pub fn draw_embedded(ui: &mut egui::Ui, dock_style: &mut DockStyleSettings) {
    draw_contents(ui, dock_style);
}

fn draw_contents(ui: &mut egui::Ui, dock_style: &mut DockStyleSettings) {
    let should_seed_current_theme =
        !dock_style.enabled && *dock_style == DockStyleSettings::default();
    let was_enabled = dock_style.enabled;

    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut dock_style.enabled, "Enable custom dock styling");
        if dock_style.enabled && !was_enabled && should_seed_current_theme {
            dock_style.enable_from_current_theme(ui.style().as_ref());
        }

        if ui.button("Load Current Theme").clicked() {
            dock_style.enable_from_current_theme(ui.style().as_ref());
        }

        if ui.button("Reset").clicked() {
            dock_style.reset_to_current_theme(ui.style().as_ref());
        }
    });

    if !dock_style.enabled {
        ui.weak("Using egui_dock styling derived from the current egui theme.");
        return;
    }

    ui.separator();

    egui::CollapsingHeader::new("Global")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(
                &mut dock_style.dock_area_padding.enabled,
                "Use dock area padding",
            );
            if dock_style.dock_area_padding.enabled {
                draw_margin_editor(
                    ui,
                    "Dock area padding",
                    &mut dock_style.dock_area_padding.value,
                    0.0..=32.0,
                );
            }

            draw_stroke_editor(
                ui,
                "Main surface border",
                &mut dock_style.main_surface_border_stroke,
                0.0..=12.0,
            );
            draw_rounding_editor(
                ui,
                "Main surface rounding",
                &mut dock_style.main_surface_border_rounding,
                0.0..=24.0,
            );
        });

    egui::CollapsingHeader::new("Buttons")
        .default_open(false)
        .show(ui, |ui| {
            draw_enum_row(
                ui,
                "Add button align",
                dock_style.buttons.add_tab_align.label(),
                |ui| {
                    ui.selectable_value(
                        &mut dock_style.buttons.add_tab_align,
                        DockTabAddAlign::Left,
                        DockTabAddAlign::Left.label(),
                    );
                    ui.selectable_value(
                        &mut dock_style.buttons.add_tab_align,
                        DockTabAddAlign::Right,
                        DockTabAddAlign::Right.label(),
                    );
                },
            );
            draw_color_row(
                ui,
                "Close button color",
                &mut dock_style.buttons.close_tab_color,
            );
            draw_color_row(
                ui,
                "Close button active",
                &mut dock_style.buttons.close_tab_active_color,
            );
            draw_color_row(
                ui,
                "Close button background",
                &mut dock_style.buttons.close_tab_bg_fill,
            );
            draw_color_row(
                ui,
                "Add button color",
                &mut dock_style.buttons.add_tab_color,
            );
            draw_color_row(
                ui,
                "Add button active",
                &mut dock_style.buttons.add_tab_active_color,
            );
            draw_color_row(
                ui,
                "Add button background",
                &mut dock_style.buttons.add_tab_bg_fill,
            );
            draw_color_row(
                ui,
                "Add button border",
                &mut dock_style.buttons.add_tab_border_color,
            );
        });

    egui::CollapsingHeader::new("Separator")
        .default_open(false)
        .show(ui, |ui| {
            draw_float_row(ui, "Width", &mut dock_style.separator.width, 0.0..=12.0);
            draw_float_row(
                ui,
                "Extra interact width",
                &mut dock_style.separator.extra_interact_width,
                0.0..=24.0,
            );
            draw_float_row(
                ui,
                "Offset clamp",
                &mut dock_style.separator.extra,
                0.0..=400.0,
            );
            draw_color_row(ui, "Idle color", &mut dock_style.separator.color_idle);
            draw_color_row(ui, "Hovered color", &mut dock_style.separator.color_hovered);
            draw_color_row(ui, "Dragged color", &mut dock_style.separator.color_dragged);
        });

    egui::CollapsingHeader::new("Tab Bar")
        .default_open(true)
        .show(ui, |ui| {
            draw_color_row(ui, "Background", &mut dock_style.tab_bar.bg_fill);
            draw_float_row(ui, "Height", &mut dock_style.tab_bar.height, 16.0..=64.0);
            ui.checkbox(
                &mut dock_style.tab_bar.show_scroll_bar_on_overflow,
                "Show scroll bar on overflow",
            );
            ui.checkbox(&mut dock_style.tab_bar.fill_tab_bar, "Fill tab bar width");
            draw_color_row(ui, "Bottom line", &mut dock_style.tab_bar.hline_color);
            draw_rounding_editor(
                ui,
                "Tab bar rounding",
                &mut dock_style.tab_bar.rounding,
                0.0..=24.0,
            );
        });

    egui::CollapsingHeader::new("Tab States")
        .default_open(false)
        .show(ui, |ui| {
            draw_tab_interaction_editor(ui, "Active", &mut dock_style.tab.active);
            draw_tab_interaction_editor(ui, "Inactive", &mut dock_style.tab.inactive);
            draw_tab_interaction_editor(ui, "Focused", &mut dock_style.tab.focused);
            draw_tab_interaction_editor(ui, "Hovered", &mut dock_style.tab.hovered);
            draw_tab_interaction_editor(
                ui,
                "Inactive + keyboard focus",
                &mut dock_style.tab.inactive_with_kb_focus,
            );
            draw_tab_interaction_editor(
                ui,
                "Active + keyboard focus",
                &mut dock_style.tab.active_with_kb_focus,
            );
            draw_tab_interaction_editor(
                ui,
                "Focused + keyboard focus",
                &mut dock_style.tab.focused_with_kb_focus,
            );
        });

    egui::CollapsingHeader::new("Tab Body")
        .default_open(false)
        .show(ui, |ui| {
            draw_tab_body_editor(ui, &mut dock_style.tab);
        });

    egui::CollapsingHeader::new("Overlay")
        .default_open(false)
        .show(ui, |ui| {
            draw_overlay_editor(ui, &mut dock_style.overlay);
        });
}

fn draw_tab_body_editor(ui: &mut egui::Ui, tab_style: &mut DockTabStyleSettings) {
    ui.checkbox(
        &mut tab_style.hline_below_active_tab_name,
        "Show divider under active tab title",
    );
    ui.checkbox(
        &mut tab_style.minimum_width_enabled,
        "Use minimum tab width",
    );
    if tab_style.minimum_width_enabled {
        draw_float_row(
            ui,
            "Minimum tab width",
            &mut tab_style.minimum_width,
            24.0..=280.0,
        );
    }

    ui.separator();
    draw_color_row(ui, "Body background", &mut tab_style.tab_body.bg_fill);
    draw_stroke_editor(
        ui,
        "Body border",
        &mut tab_style.tab_body.stroke,
        0.0..=12.0,
    );
    draw_rounding_editor(
        ui,
        "Body rounding",
        &mut tab_style.tab_body.rounding,
        0.0..=24.0,
    );
    draw_margin_editor(
        ui,
        "Body inner margin",
        &mut tab_style.tab_body.inner_margin,
        0.0..=32.0,
    );
}

fn draw_overlay_editor(ui: &mut egui::Ui, overlay: &mut DockOverlayStyleSettings) {
    draw_enum_row(ui, "Overlay type", overlay.overlay_type.label(), |ui| {
        ui.selectable_value(
            &mut overlay.overlay_type,
            DockOverlayType::Widgets,
            DockOverlayType::Widgets.label(),
        );
        ui.selectable_value(
            &mut overlay.overlay_type,
            DockOverlayType::HighlightedAreas,
            DockOverlayType::HighlightedAreas.label(),
        );
    });

    draw_color_row(ui, "Selection color", &mut overlay.selection_color);
    draw_float_row(
        ui,
        "Selection stroke width",
        &mut overlay.selection_stroke_width,
        0.0..=12.0,
    );
    draw_float_row(
        ui,
        "Button spacing",
        &mut overlay.button_spacing,
        0.0..=40.0,
    );
    draw_float_row(
        ui,
        "Max button size",
        &mut overlay.max_button_size,
        20.0..=240.0,
    );
    draw_float_row(
        ui,
        "Surface fade opacity",
        &mut overlay.surface_fade_opacity,
        0.0..=1.0,
    );
    draw_color_row(ui, "Button color", &mut overlay.button_color);
    draw_stroke_editor(
        ui,
        "Button border",
        &mut overlay.button_border_stroke,
        0.0..=12.0,
    );

    egui::CollapsingHeader::new("Hovered Leaf Highlight")
        .default_open(false)
        .show(ui, |ui| {
            draw_leaf_highlight_editor(ui, &mut overlay.hovered_leaf_highlight);
        });

    egui::CollapsingHeader::new("Overlay Feel")
        .default_open(false)
        .show(ui, |ui| {
            draw_overlay_feel_editor(ui, &mut overlay.feel);
        });
}

fn draw_leaf_highlight_editor(ui: &mut egui::Ui, highlight: &mut DockLeafHighlightingSettings) {
    draw_color_row(ui, "Fill color", &mut highlight.color);
    draw_stroke_editor(ui, "Stroke", &mut highlight.stroke, 0.0..=12.0);
    draw_rounding_editor(ui, "Rounding", &mut highlight.rounding, 0.0..=24.0);
    draw_float_row(ui, "Expansion", &mut highlight.expansion, 0.0..=48.0);
}

fn draw_overlay_feel_editor(ui: &mut egui::Ui, feel: &mut DockOverlayFeelSettings) {
    draw_float_row(
        ui,
        "Window drop coverage",
        &mut feel.window_drop_coverage,
        0.0..=1.0,
    );
    draw_float_row(
        ui,
        "Center drop coverage",
        &mut feel.center_drop_coverage,
        0.0..=1.0,
    );
    draw_float_row(ui, "Fade hold time", &mut feel.fade_hold_time, 0.0..=2.0);
    draw_float_row(
        ui,
        "Max preference time",
        &mut feel.max_preference_time,
        0.0..=2.0,
    );
    draw_float_row(
        ui,
        "Interact expansion",
        &mut feel.interact_expansion,
        0.0..=48.0,
    );
}

fn draw_tab_interaction_editor(
    ui: &mut egui::Ui,
    title: &str,
    style: &mut DockTabInteractionStyleSettings,
) {
    egui::CollapsingHeader::new(title)
        .default_open(false)
        .show(ui, |ui| {
            draw_color_row(ui, "Background", &mut style.bg_fill);
            draw_color_row(ui, "Text color", &mut style.text_color);
            draw_color_row(ui, "Outline color", &mut style.outline_color);
            draw_rounding_editor(ui, "Rounding", &mut style.rounding, 0.0..=24.0);
        });
}

fn draw_margin_editor(
    ui: &mut egui::Ui,
    title: &str,
    margin: &mut DockEdgeInsets,
    range: std::ops::RangeInclusive<f32>,
) {
    egui::CollapsingHeader::new(title)
        .default_open(false)
        .show(ui, |ui| {
            draw_float_row(ui, "Left", &mut margin.left, range.clone());
            draw_float_row(ui, "Right", &mut margin.right, range.clone());
            draw_float_row(ui, "Top", &mut margin.top, range.clone());
            draw_float_row(ui, "Bottom", &mut margin.bottom, range);
        });
}

fn draw_rounding_editor(
    ui: &mut egui::Ui,
    title: &str,
    rounding: &mut DockRounding,
    range: std::ops::RangeInclusive<f32>,
) {
    egui::CollapsingHeader::new(title)
        .default_open(false)
        .show(ui, |ui| {
            draw_float_row(ui, "Top left", &mut rounding.nw, range.clone());
            draw_float_row(ui, "Top right", &mut rounding.ne, range.clone());
            draw_float_row(ui, "Bottom left", &mut rounding.sw, range.clone());
            draw_float_row(ui, "Bottom right", &mut rounding.se, range);
        });
}

fn draw_stroke_editor(
    ui: &mut egui::Ui,
    title: &str,
    stroke: &mut DockStroke,
    width_range: std::ops::RangeInclusive<f32>,
) {
    egui::CollapsingHeader::new(title)
        .default_open(false)
        .show(ui, |ui| {
            draw_float_row(ui, "Width", &mut stroke.width, width_range);
            draw_color_row(ui, "Color", &mut stroke.color);
        });
}

fn draw_color_row(ui: &mut egui::Ui, label: &str, color: &mut ColorRgba) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.color_edit_button_srgba_unmultiplied(color);
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

fn draw_enum_row(
    ui: &mut egui::Ui,
    label: &str,
    selected_text: &str,
    add_options: impl FnOnce(&mut egui::Ui),
) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(label)
            .selected_text(selected_text)
            .show_ui(ui, add_options);
    });
}
