use eframe::egui::{self, emath::TSTransform};

use crate::egui_theme::UiMotionSettings;

const RUNTIME_MOTION_SETTINGS_ID: &str = "runtime_ui_motion_settings";

pub fn store_runtime_settings(ctx: &egui::Context, settings: UiMotionSettings) {
    ctx.data_mut(|data| {
        data.insert_temp(egui::Id::new(RUNTIME_MOTION_SETTINGS_ID), settings);
    });
}

pub fn settings(ctx: &egui::Context) -> UiMotionSettings {
    ctx.data(|data| {
        data.get_temp::<UiMotionSettings>(egui::Id::new(RUNTIME_MOTION_SETTINGS_ID))
            .unwrap_or_default()
    })
}

pub fn surface_open_t(
    ctx: &egui::Context,
    id: egui::Id,
    open: bool,
    settings: UiMotionSettings,
) -> f32 {
    if settings.enabled {
        ctx.animate_bool_with_time(id.with("surface_open"), open, settings.surface_duration())
    } else if open {
        1.0
    } else {
        0.0
    }
}

pub fn micro_t(ctx: &egui::Context, id: egui::Id, active: bool, settings: UiMotionSettings) -> f32 {
    if settings.enabled {
        ctx.animate_bool_with_time(id.with("micro"), active, settings.micro_duration())
    } else if active {
        1.0
    } else {
        0.0
    }
}

pub fn dock_t(ctx: &egui::Context, id: egui::Id, active: bool, settings: UiMotionSettings) -> f32 {
    if settings.enabled {
        ctx.animate_bool_with_time(id.with("dock"), active, settings.dock_duration())
    } else if active {
        1.0
    } else {
        0.0
    }
}

pub fn fade_alpha(t: f32, _reduced_motion: bool) -> f32 {
    t.clamp(0.0, 1.0)
}

pub fn slide_offset_y(t: f32, slide_px: f32, reduced_motion: bool) -> f32 {
    if reduced_motion {
        0.0
    } else {
        (1.0 - t.clamp(0.0, 1.0)) * slide_px.max(0.0)
    }
}

pub fn scale_factor(t: f32, scale_delta: f32, reduced_motion: bool) -> f32 {
    if reduced_motion {
        1.0
    } else {
        1.0 - (1.0 - t.clamp(0.0, 1.0)) * scale_delta.clamp(0.0, 0.25)
    }
}

pub fn should_draw_surface(open: bool, t: f32) -> bool {
    open || t > 0.0
}

pub fn frame_with_alpha(frame: egui::Frame, t: f32, settings: UiMotionSettings) -> egui::Frame {
    frame.multiply_with_opacity(fade_alpha(t, settings.reduced_motion))
}

pub fn layer_transform(rect: egui::Rect, t: f32, settings: UiMotionSettings) -> TSTransform {
    let slide = slide_offset_y(t, settings.effective_slide_px(), settings.reduced_motion);
    let scale = scale_factor(
        t,
        settings.effective_overlay_scale_delta(),
        settings.reduced_motion,
    );

    if slide == 0.0 && (scale - 1.0).abs() <= f32::EPSILON {
        TSTransform::IDENTITY
    } else {
        let translation = egui::vec2(0.0, slide) + rect.center().to_vec2() * (1.0 - scale);
        TSTransform::new(translation, scale)
    }
}

pub fn apply_surface_transform(
    ctx: &egui::Context,
    response: &egui::Response,
    t: f32,
    settings: UiMotionSettings,
) {
    let transform = layer_transform(response.rect, t, settings);
    ctx.set_transform_layer(response.layer_id, transform);
}

pub fn clear_surface_layers(ctx: &egui::Context, id: egui::Id) {
    for order in [
        egui::Order::Background,
        egui::Order::Middle,
        egui::Order::Foreground,
        egui::Order::Tooltip,
        egui::Order::Debug,
    ] {
        ctx.set_transform_layer(egui::LayerId::new(order, id), TSTransform::IDENTITY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_motion_helpers_are_fade_only() {
        assert_eq!(fade_alpha(0.5, true), 0.5);
        assert_eq!(slide_offset_y(0.5, 16.0, true), 0.0);
        assert_eq!(scale_factor(0.5, 0.04, true), 1.0);
    }

    #[test]
    fn disabled_motion_returns_immediate_end_states() {
        let ctx = egui::Context::default();
        let settings = UiMotionSettings {
            enabled: false,
            ..Default::default()
        };

        assert_eq!(
            surface_open_t(&ctx, egui::Id::new("surface"), true, settings),
            1.0
        );
        assert_eq!(
            surface_open_t(&ctx, egui::Id::new("surface"), false, settings),
            0.0
        );
        assert_eq!(micro_t(&ctx, egui::Id::new("micro"), true, settings), 1.0);
        assert_eq!(dock_t(&ctx, egui::Id::new("dock"), false, settings), 0.0);
    }

    #[test]
    fn dock_animation_ids_stay_stable_across_repeated_toggles() {
        let ctx = egui::Context::default();
        let settings = UiMotionSettings::default();
        let id = egui::Id::new("dock_tab_transition");

        let first = dock_t(&ctx, id, false, settings);
        let second = dock_t(&ctx, id, true, settings);
        let third = dock_t(&ctx, id, false, settings);

        for value in [first, second, third] {
            assert!((0.0..=1.0).contains(&value));
        }
    }
}
