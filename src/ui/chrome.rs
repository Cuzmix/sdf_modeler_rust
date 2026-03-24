use eframe::egui;

const SURFACE_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(24, 26, 31, 238);
const SURFACE_STROKE: egui::Color32 = egui::Color32::from_rgba_premultiplied(88, 94, 110, 185);
const ACTION_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(44, 48, 58, 220);
const ACTION_FILL_ACTIVE: egui::Color32 = egui::Color32::from_rgba_premultiplied(56, 92, 132, 228);
const ACTION_STROKE: egui::Color32 = egui::Color32::from_rgba_premultiplied(104, 112, 132, 205);
const ACTION_TEXT: egui::Color32 = egui::Color32::from_rgb(224, 230, 242);
const TREE_ROW_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(34, 38, 46, 196);
const TREE_ROW_FILL_SELECTED: egui::Color32 = egui::Color32::from_rgba_premultiplied(46, 72, 104, 232);
const TREE_ROW_FILL_DIMMED: egui::Color32 = egui::Color32::from_rgba_premultiplied(28, 30, 36, 176);
const TREE_ROW_STROKE: egui::Color32 = egui::Color32::from_rgba_premultiplied(76, 82, 96, 176);
const TREE_ROW_STROKE_SELECTED: egui::Color32 = egui::Color32::from_rgb(118, 164, 224);
const CARD_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(30, 34, 42, 214);
const CARD_STROKE: egui::Color32 = egui::Color32::from_rgba_premultiplied(76, 82, 96, 176);

pub const SURFACE_CORNER_RADIUS: f32 = 10.0;
pub const ACTION_CORNER_RADIUS: f32 = 8.0;
pub const TREE_ROW_CORNER_RADIUS: f32 = 8.0;
pub const CARD_CORNER_RADIUS: f32 = 10.0;

pub fn surface_frame(ctx: &egui::Context) -> egui::Frame {
    egui::Frame::window(&ctx.style())
        .fill(SURFACE_FILL)
        .stroke(egui::Stroke::new(1.0, SURFACE_STROKE))
        .rounding(egui::Rounding::same(SURFACE_CORNER_RADIUS))
        .inner_margin(egui::Margin::same(10.0))
        .shadow(egui::epaint::Shadow {
            offset: egui::vec2(0.0, 10.0),
            blur: 28.0,
            spread: 1.0,
            color: egui::Color32::from_black_alpha(72),
        })
}

pub fn action_button(ui: &mut egui::Ui, label: &str, active: bool) -> egui::Response {
    let fill = if active {
        ACTION_FILL_ACTIVE
    } else {
        ACTION_FILL
    };
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .small()
                .color(ACTION_TEXT)
                .strong(),
        )
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, ACTION_STROKE))
        .rounding(egui::Rounding::same(ACTION_CORNER_RADIUS))
        .min_size(egui::vec2(24.0, 22.0)),
    )
}

pub fn tab_pill(ui: &mut egui::Ui, label: &str, active: bool) -> egui::Response {
    let fill = if active {
        ACTION_FILL_ACTIVE
    } else {
        ACTION_FILL
    };
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .small()
                .color(ACTION_TEXT)
                .strong(),
        )
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, ACTION_STROKE))
        .rounding(egui::Rounding::same(ACTION_CORNER_RADIUS))
        .min_size(egui::vec2(76.0, 24.0)),
    )
}

pub fn with_action_button_style<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.scope(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
        let widgets = &mut ui.style_mut().visuals.widgets;
        widgets.inactive.bg_fill = ACTION_FILL;
        widgets.inactive.bg_stroke = egui::Stroke::new(1.0, ACTION_STROKE);
        widgets.inactive.fg_stroke = egui::Stroke::new(1.0, ACTION_TEXT);
        widgets.hovered.bg_fill = ACTION_FILL_ACTIVE;
        widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACTION_STROKE);
        widgets.hovered.fg_stroke = egui::Stroke::new(1.0, ACTION_TEXT);
        widgets.active.bg_fill = ACTION_FILL_ACTIVE;
        widgets.active.bg_stroke = egui::Stroke::new(1.0, ACTION_STROKE);
        widgets.active.fg_stroke = egui::Stroke::new(1.0, ACTION_TEXT);
        add_contents(ui)
    })
    .inner
}

pub fn card_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(CARD_FILL)
        .stroke(egui::Stroke::new(1.0, CARD_STROKE))
        .rounding(egui::Rounding::same(CARD_CORNER_RADIUS))
        .inner_margin(egui::Margin::same(10.0))
}

pub fn section_title(ui: &mut egui::Ui, title: &str, subtitle: Option<&str>) {
    ui.label(egui::RichText::new(title).small().strong());
    if let Some(subtitle) = subtitle {
        ui.small(subtitle);
    }
}

pub fn tree_row_frame(selected: bool, dimmed: bool) -> egui::Frame {
    let fill = if selected {
        TREE_ROW_FILL_SELECTED
    } else if dimmed {
        TREE_ROW_FILL_DIMMED
    } else {
        TREE_ROW_FILL
    };
    let stroke = if selected {
        TREE_ROW_STROKE_SELECTED
    } else {
        TREE_ROW_STROKE
    };

    egui::Frame::none()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke))
        .rounding(egui::Rounding::same(TREE_ROW_CORNER_RADIUS))
        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
}
