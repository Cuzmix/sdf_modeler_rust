use eframe::egui;

use crate::settings::Settings;

/// Draw the System Settings window. Returns `true` if a shader-affecting setting changed.
pub fn draw(
    ctx: &egui::Context,
    open: &mut bool,
    settings: &mut Settings,
    show_debug: &mut bool,
    initial_vsync: bool,
) -> bool {
    let before = settings.render.clone();

    egui::Window::new("Settings")
        .open(open)
        .default_width(300.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
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

                // --- Viewport ---
                egui::CollapsingHeader::new("Viewport")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.render.show_grid, "Show Grid")
                            .on_hover_text("Display ground plane grid at Y=0");
                        ui.checkbox(&mut settings.render.clamp_orbit_pitch, "Clamp Orbit Pitch")
                            .on_hover_text("Limit vertical orbit to ±89°. When off, allows full 360° gimbal rotation.");
                        ui.separator();
                        labeled_slider(ui, "Roll Sensitivity", &mut settings.render.roll_sensitivity, 0.001..=0.02, false,
                            "How fast Ctrl+Alt+drag and touch twist roll the camera");
                        ui.checkbox(&mut settings.render.invert_roll, "Invert Roll")
                            .on_hover_text("Reverse the roll direction for both touch twist and Ctrl+Alt+drag");
                    });

                // --- Touch Input ---
                egui::CollapsingHeader::new("Touch Input")
                    .default_open(true)
                    .show(ui, |ui| {
                        labeled_slider(ui, "Zoom Sensitivity", &mut settings.render.touch_zoom_sensitivity, 100.0..=2000.0, false,
                            "How fast pinch-to-zoom responds");
                        ui.checkbox(&mut settings.render.invert_touch_pan, "Invert Touch Pan")
                            .on_hover_text("Reverse two-finger pan direction");
                    });

                // --- Auto-save ---
                egui::CollapsingHeader::new("Auto-save")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.auto_save_enabled, "Enable Auto-save");
                        ui.add_enabled_ui(settings.auto_save_enabled, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Interval:");
                                ui.add(egui::DragValue::new(&mut settings.auto_save_interval_secs)
                                    .range(30..=600).suffix("s").speed(5));
                            });
                        });
                    });

                // --- Performance ---
                let config = &mut settings.render;
                egui::CollapsingHeader::new("Performance")
                    .default_open(false)
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
                            ui.indent("comp_hint", |ui| {
                                let mem_mb = (res_i32 as f32).powi(3) * 12.0 / (1024.0 * 1024.0);
                                ui.weak(format!("~{:.0} MB VRAM ({}^3 x 3 textures)", mem_mb, res_i32));
                            });
                        });
                    });
            });
        });

    let changed = settings.render != before;
    settings.save();
    changed
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
