use std::hash::Hash;

use eframe::egui::{self, Color32, Frame, InnerResponse, Margin, Response, RichText, Stroke};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BadgeTone {
    Accent,
    Muted,
    Success,
    Warning,
    Destructive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignTokens {
    pub background: Color32,
    pub card: Color32,
    pub popover: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub border: Color32,
    pub input: Color32,
    pub ring: Color32,
    pub destructive: Color32,
    pub badge: Color32,
    pub kbd_chip: Color32,
    pub text: Color32,
    pub muted_text: Color32,
    pub radius_sm: f32,
    pub radius_md: f32,
    pub radius_lg: f32,
    pub control_height: f32,
    pub dense_spacing: egui::Vec2,
    pub section_spacing: f32,
}

impl DesignTokens {
    pub fn from_style(style: &egui::Style) -> Self {
        let visuals = &style.visuals;
        let background = visuals.extreme_bg_color;
        let card = visuals.window_fill;
        let popover = mix_color(visuals.window_fill, visuals.panel_fill, 0.4);
        let muted = visuals.faint_bg_color;
        let accent = visuals.hyperlink_color;
        let border = visuals.widgets.noninteractive.bg_stroke.color;
        let input = mix_color(
            visuals.widgets.inactive.weak_bg_fill,
            visuals.widgets.inactive.bg_fill,
            0.55,
        );
        let ring = visuals.selection.stroke.color;
        let destructive = visuals.error_fg_color;
        let badge = mix_color(accent, visuals.selection.bg_fill, 0.45);
        let kbd_chip = visuals.code_bg_color;
        let text = visuals.text_color();
        let muted_text = visuals.weak_text_color();
        let radius_lg = visuals.window_rounding.nw.clamp(6.0, 8.0);
        let radius_md = (radius_lg - 2.0).max(6.0);
        let radius_sm = (radius_md - 2.0).max(4.0);

        Self {
            background,
            card,
            popover,
            muted,
            accent,
            border,
            input,
            ring,
            destructive,
            badge,
            kbd_chip,
            text,
            muted_text,
            radius_sm,
            radius_md,
            radius_lg,
            control_height: style.spacing.interact_size.y.max(24.0),
            dense_spacing: egui::vec2(style.spacing.item_spacing.x.min(8.0), 6.0),
            section_spacing: style.spacing.item_spacing.y.max(6.0),
        }
    }
}

pub fn design_tokens(style: &egui::Style) -> DesignTokens {
    DesignTokens::from_style(style)
}

pub fn tokens(ui: &egui::Ui) -> DesignTokens {
    design_tokens(ui.style().as_ref())
}

pub fn app_header_frame(ui: &egui::Ui) -> Frame {
    let tokens = tokens(ui);
    Frame {
        fill: mix_color(tokens.background, tokens.card, 0.35),
        stroke: Stroke::new(1.0, tokens.border),
        inner_margin: Margin::symmetric(8.0, 6.0),
        ..Default::default()
    }
}

pub fn ribbon_frame(ui: &egui::Ui) -> Frame {
    let tokens = tokens(ui);
    Frame {
        fill: mix_color(tokens.background, tokens.muted, 0.45),
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.85)),
        inner_margin: Margin::symmetric(8.0, 4.0),
        ..Default::default()
    }
}

pub fn card_frame(ui: &egui::Ui) -> Frame {
    let tokens = tokens(ui);
    Frame {
        fill: tokens.card,
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.95)),
        inner_margin: Margin::same(8.0),
        rounding: egui::Rounding::same(tokens.radius_md),
        ..Default::default()
    }
}

pub fn inset_frame(ui: &egui::Ui) -> Frame {
    let tokens = tokens(ui);
    Frame {
        fill: mix_color(tokens.card, tokens.input, 0.4),
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.8)),
        inner_margin: Margin::symmetric(8.0, 6.0),
        rounding: egui::Rounding::same(tokens.radius_md),
        ..Default::default()
    }
}

pub fn item_frame(ui: &egui::Ui, selected: bool) -> Frame {
    let tokens = tokens(ui);
    let (fill, stroke) = if selected {
        (
            mix_color(tokens.badge, tokens.card, 0.25),
            Stroke::new(1.0, tokens.ring),
        )
    } else {
        (
            mix_color(tokens.card, tokens.input, 0.3),
            Stroke::new(1.0, tokens.border.gamma_multiply(0.75)),
        )
    };

    Frame {
        fill,
        stroke,
        inner_margin: Margin::symmetric(8.0, 5.0),
        rounding: egui::Rounding::same(tokens.radius_md),
        ..Default::default()
    }
}

pub fn header_group<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> InnerResponse<R> {
    let tokens = tokens(ui);
    inset_frame(ui).show(ui, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, tokens.dense_spacing.y);
        add_contents(ui)
    })
}

pub fn panel_header(ui: &mut egui::Ui, title: &str, description: &str) {
    let tokens = tokens(ui);
    ui.vertical(|ui| {
        ui.label(RichText::new(title).size(16.0).strong().color(tokens.text));
        if !description.is_empty() {
            ui.add_space(2.0);
            ui.label(RichText::new(description).color(tokens.muted_text));
        }
    });
}

pub fn section_card<R>(
    ui: &mut egui::Ui,
    title: &str,
    description: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> InnerResponse<R> {
    let tokens = tokens(ui);
    Frame {
        fill: Color32::TRANSPARENT,
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.6)),
        inner_margin: Margin::symmetric(6.0, 5.0),
        rounding: egui::Rounding::same(tokens.radius_sm),
        ..Default::default()
    }
    .show(ui, |ui| {
        panel_header(ui, title, description);
        ui.add_space(tokens.section_spacing);
        add_contents(ui)
    })
}

pub fn badge(ui: &mut egui::Ui, tone: BadgeTone, text: impl Into<String>) -> Response {
    let tokens = tokens(ui);
    let text = text.into();
    let (fill, stroke_color, text_color) = badge_palette(tokens, tone);

    let inner = Frame {
        fill,
        stroke: Stroke::new(1.0, stroke_color),
        inner_margin: Margin::symmetric(8.0, 4.0),
        rounding: egui::Rounding::same(tokens.radius_sm),
        ..Default::default()
    }
    .show(ui, |ui| {
        ui.label(RichText::new(text).size(11.0).color(text_color).strong())
    });

    inner.response
}

pub fn kbd_chip(ui: &mut egui::Ui, text: impl Into<String>) -> Response {
    let tokens = tokens(ui);
    let text = text.into();
    let inner = Frame {
        fill: tokens.kbd_chip,
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.7)),
        inner_margin: Margin::symmetric(8.0, 4.0),
        rounding: egui::Rounding::same(tokens.radius_sm),
        ..Default::default()
    }
    .show(ui, |ui| {
        ui.label(
            RichText::new(text)
                .monospace()
                .size(11.0)
                .color(tokens.text),
        )
    });

    inner.response
}

pub fn empty_state(ui: &mut egui::Ui, title: &str, description: &str) {
    let tokens = tokens(ui);
    let available_width = ui.available_width().max(220.0);
    ui.add_space(16.0);
    ui.vertical_centered(|ui| {
        card_frame(ui).show(ui, |ui| {
            ui.set_width(available_width.min(340.0));
            ui.vertical_centered(|ui| {
                ui.add_space(4.0);
                ui.label(RichText::new(title).strong().size(16.0).color(tokens.text));
                ui.add_space(6.0);
                ui.label(RichText::new(description).color(tokens.muted_text));
                ui.add_space(4.0);
            });
        });
    });
}

pub fn search_field(
    ui: &mut egui::Ui,
    id_source: impl Hash,
    text: &mut String,
    hint: &str,
) -> Response {
    let tokens = tokens(ui);
    let frame = Frame {
        fill: tokens.input,
        stroke: Stroke::new(1.0, tokens.border.gamma_multiply(0.85)),
        inner_margin: Margin::symmetric(8.0, 6.0),
        rounding: egui::Rounding::same(tokens.radius_md),
        ..Default::default()
    };

    frame
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("\u{1F50D}").color(tokens.muted_text));
                ui.add(
                    egui::TextEdit::singleline(text)
                        .id_source(id_source)
                        .hint_text(hint)
                        .frame(false)
                        .desired_width(ui.available_width() - 30.0),
                )
            })
            .inner
        })
        .inner
}

fn badge_palette(tokens: DesignTokens, tone: BadgeTone) -> (Color32, Color32, Color32) {
    match tone {
        BadgeTone::Accent => (
            mix_color(tokens.badge, tokens.card, 0.18),
            tokens.ring,
            mix_color(tokens.text, tokens.accent, 0.35),
        ),
        BadgeTone::Muted => (
            mix_color(tokens.muted, tokens.card, 0.35),
            tokens.border.gamma_multiply(0.75),
            tokens.muted_text,
        ),
        BadgeTone::Success => {
            let green = Color32::from_rgb(74, 222, 128);
            (
                mix_color(green, tokens.card, 0.18),
                mix_color(green, tokens.border, 0.35),
                mix_color(tokens.text, green, 0.3),
            )
        }
        BadgeTone::Warning => {
            let amber = Color32::from_rgb(251, 191, 36);
            (
                mix_color(amber, tokens.card, 0.16),
                mix_color(amber, tokens.border, 0.28),
                mix_color(tokens.text, amber, 0.3),
            )
        }
        BadgeTone::Destructive => (
            mix_color(tokens.destructive, tokens.card, 0.16),
            mix_color(tokens.destructive, tokens.border, 0.28),
            mix_color(tokens.text, tokens.destructive, 0.32),
        ),
    }
}

fn mix_color(lhs: Color32, rhs: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let blend = |left: u8, right: u8| -> u8 {
        ((left as f32) + ((right as f32) - (left as f32)) * t).round() as u8
    };

    Color32::from_rgba_premultiplied(
        blend(lhs.r(), rhs.r()),
        blend(lhs.g(), rhs.g()),
        blend(lhs.b(), rhs.b()),
        blend(lhs.a(), rhs.a()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egui_theme::{EguiThemeSettings, ThemePreset};

    #[test]
    fn design_tokens_follow_core_style_values() {
        let style = EguiThemeSettings::from_preset(ThemePreset::StudioDark)
            .build()
            .style;
        let tokens = design_tokens(&style);

        assert_eq!(tokens.accent, style.visuals.hyperlink_color);
        assert_eq!(tokens.ring, style.visuals.selection.stroke.color);
        assert_eq!(
            tokens.border,
            style.visuals.widgets.noninteractive.bg_stroke.color
        );
        assert!(tokens.radius_lg >= tokens.radius_md);
        assert!(tokens.radius_md >= tokens.radius_sm);
    }
}
