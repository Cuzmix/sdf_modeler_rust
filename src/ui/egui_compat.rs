pub fn corner_radius(radius: f32) -> egui::CornerRadius {
    egui::CornerRadius::same(clamp_u8(radius))
}

pub fn margin_same(margin: f32) -> egui::Margin {
    egui::Margin::same(clamp_i8(margin))
}

pub fn margin_symmetric(x: f32, y: f32) -> egui::Margin {
    egui::Margin::symmetric(clamp_i8(x), clamp_i8(y))
}

pub fn shadow(
    offset_x: f32,
    offset_y: f32,
    blur: f32,
    spread: f32,
    color: egui::Color32,
) -> egui::epaint::Shadow {
    egui::epaint::Shadow {
        offset: [clamp_i8(offset_x), clamp_i8(offset_y)],
        blur: clamp_u8(blur),
        spread: clamp_u8(spread),
        color,
    }
}

pub fn outside_stroke() -> egui::StrokeKind {
    egui::StrokeKind::Outside
}

fn clamp_u8(value: f32) -> u8 {
    value.round().clamp(0.0, u8::MAX as f32) as u8
}

fn clamp_i8(value: f32) -> i8 {
    value.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
}
