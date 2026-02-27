use eframe::egui;

use crate::settings::Settings;

/// Draw the Developer Settings panel.
/// `show_debug` controls the profiler window visibility (mirrors F4).
/// `initial_vsync` is the vsync state at app startup (to detect restart-required).
pub fn draw(ui: &mut egui::Ui, settings: &mut Settings, show_debug: &mut bool, initial_vsync: bool) {
    ui.heading("Developer Settings");

    ui.separator();

    // --- Display ---
    egui::CollapsingHeader::new("Display")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut settings.show_fps_overlay, "Show FPS Overlay")
                .on_hover_text("Display FPS counter in the top-left of the viewport");

            ui.checkbox(show_debug, "Show Profiler")
                .on_hover_text("Toggle the profiler window (F4)");

            ui.separator();

            ui.checkbox(&mut settings.vsync_enabled, "VSync");
            if settings.vsync_enabled != initial_vsync {
                ui.weak("(restart required)");
            }

            ui.checkbox(&mut settings.continuous_repaint, "Continuous Repaint")
                .on_hover_text("Force repaint every frame (useful for benchmarking)");
        });

    // --- Performance ---
    let config = &mut settings.render;
    egui::CollapsingHeader::new("Performance")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.sculpt_fast_mode, "Fast mode while sculpting")
                .on_hover_text("Half steps + skip AO/shadows during brush strokes");
            ui.checkbox(&mut config.auto_reduce_steps, "Auto-reduce steps (multi-sculpt)")
                .on_hover_text("Halve march steps when 2+ sculpt nodes exist");
            ui.separator();
            labeled_slider(ui, "Interaction Scale", &mut config.interaction_render_scale, 0.25..=1.0, false,
                "Render resolution during orbit/sculpt (0.5 = half res)");
            labeled_slider(ui, "Rest Scale", &mut config.rest_render_scale, 0.25..=1.0, false,
                "Render resolution when idle (1.0 = full res)");
            ui.separator();
            ui.checkbox(&mut config.composite_volume_enabled, "Composite Volume Cache")
                .on_hover_text("Pre-composite all sculpts into a single 3D texture.\nDecouples render cost from sculpt count.");
            ui.add_enabled_ui(config.composite_volume_enabled, |ui| {
                let mut res_i32 = config.composite_volume_resolution as i32;
                ui.horizontal(|ui| {
                    ui.label("Volume Resolution");
                    ui.add(egui::Slider::new(&mut res_i32, 64..=256));
                });
                config.composite_volume_resolution = res_i32 as u32;
                ui.indent("comp_hint_dev", |ui| {
                    let mem_mb = (res_i32 as f32).powi(3) * 12.0 / (1024.0 * 1024.0);
                    ui.weak(format!("~{:.0} MB VRAM ({}^3 x 3 textures)", mem_mb, res_i32));
                });
            });
        });

    settings.save();
}

fn labeled_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    tooltip: &str,
) {
    ui.horizontal(|ui| {
        let lbl = ui.label(label);
        if !tooltip.is_empty() {
            lbl.on_hover_text(tooltip);
        }
        let mut slider = egui::Slider::new(value, range);
        if logarithmic {
            slider = slider.logarithmic(true);
        }
        ui.add(slider);
    });
}
