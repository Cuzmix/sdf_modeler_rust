use eframe::egui;

use crate::settings::Settings;

/// Draw the Render Settings panel. Returns `true` if a shader-affecting setting changed.
pub fn draw(ui: &mut egui::Ui, settings: &mut Settings) -> bool {
    let before = settings.render.clone();

    ui.heading("Render Settings");

    if ui.small_button("Reset All").clicked() {
        settings.render.reset_all();
    }

    ui.separator();

    let config = &mut settings.render;

    // --- Shadows ---
    egui::CollapsingHeader::new("Shadows")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.shadows_enabled, "Enable Shadows");
            ui.add_enabled_ui(config.shadows_enabled, |ui| {
                labeled_slider_i32(ui, "Steps", &mut config.shadow_steps, 8..=128);
                labeled_slider(ui, "Penumbra K", &mut config.shadow_penumbra_k, 1.0..=32.0, false);
                labeled_slider(ui, "Bias", &mut config.shadow_bias, 0.001..=0.2, true);
                labeled_slider(ui, "Min T", &mut config.shadow_mint, 0.01..=0.5, false);
                labeled_slider(ui, "Max T", &mut config.shadow_maxt, 5.0..=100.0, false);
            });
            if ui.small_button("Reset").clicked() {
                config.reset_shadows();
            }
        });

    // --- Ambient Occlusion ---
    egui::CollapsingHeader::new("Ambient Occlusion")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.ao_enabled, "Enable AO");
            ui.add_enabled_ui(config.ao_enabled, |ui| {
                labeled_slider_i32(ui, "Samples", &mut config.ao_samples, 1..=16);
                labeled_slider(ui, "Step Size", &mut config.ao_step, 0.01..=0.5, false);
                labeled_slider(ui, "Decay", &mut config.ao_decay, 0.5..=1.0, false);
                labeled_slider(ui, "Intensity", &mut config.ao_intensity, 0.5..=10.0, false);
            });
            if ui.small_button("Reset").clicked() {
                config.reset_ao();
            }
        });

    // --- Raymarching ---
    egui::CollapsingHeader::new("Raymarching")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider_i32(ui, "Max Steps", &mut config.march_max_steps, 32..=512);
            labeled_slider(ui, "Epsilon", &mut config.march_epsilon, 0.0001..=0.01, true);
            labeled_slider(ui, "Step Multiplier", &mut config.march_step_multiplier, 0.5..=1.0, false);
            labeled_slider(ui, "Max Distance", &mut config.march_max_distance, 10.0..=200.0, false);
            if ui.small_button("Reset").clicked() {
                config.reset_raymarching();
            }
        });

    // --- Lighting ---
    egui::CollapsingHeader::new("Lighting")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Key Light Direction");
            dir_editor(ui, &mut config.key_light_dir);
            labeled_slider(ui, "Diffuse", &mut config.key_diffuse, 0.0..=2.0, false);
            labeled_slider(ui, "Spec Power", &mut config.key_spec_power, 1.0..=128.0, true);
            labeled_slider(ui, "Spec Intensity", &mut config.key_spec_intensity, 0.0..=2.0, false);

            ui.separator();
            ui.label("Fill Light Direction");
            dir_editor(ui, &mut config.fill_light_dir);
            labeled_slider(ui, "Fill Intensity", &mut config.fill_intensity, 0.0..=1.0, false);

            ui.separator();
            labeled_slider(ui, "Ambient", &mut config.ambient, 0.0..=0.5, false);
            if ui.small_button("Reset").clicked() {
                config.reset_lighting();
            }
        });

    // --- Sky ---
    egui::CollapsingHeader::new("Sky")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Horizon:");
                ui.color_edit_button_rgb(&mut config.sky_horizon);
            });
            ui.horizontal(|ui| {
                ui.label("Zenith:");
                ui.color_edit_button_rgb(&mut config.sky_zenith);
            });
            if ui.small_button("Reset").clicked() {
                config.reset_sky();
            }
        });

    // --- Fog ---
    egui::CollapsingHeader::new("Fog")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.fog_enabled, "Enable Fog");
            ui.add_enabled_ui(config.fog_enabled, |ui| {
                labeled_slider(ui, "Density", &mut config.fog_density, 0.001..=0.2, true);
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(&mut config.fog_color);
                });
            });
            if ui.small_button("Reset").clicked() {
                config.reset_fog();
            }
        });

    // --- Gamma ---
    egui::CollapsingHeader::new("Gamma")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider(ui, "Gamma", &mut config.gamma, 1.0..=3.0, false);
            if ui.small_button("Reset").clicked() {
                config.reset_gamma();
            }
        });

    // --- Performance ---
    egui::CollapsingHeader::new("Performance")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.sculpt_fast_mode, "Fast mode while sculpting")
                .on_hover_text("Half steps + skip AO/shadows during brush strokes");
            ui.checkbox(&mut config.auto_reduce_steps, "Auto-reduce steps (multi-sculpt)")
                .on_hover_text("Halve march steps when 2+ sculpt nodes exist");
        });

    let changed = settings.render != before;
    if changed {
        settings.save();
    }
    changed
}

fn labeled_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut slider = egui::Slider::new(value, range);
        if logarithmic {
            slider = slider.logarithmic(true);
        }
        ui.add(slider);
    });
}

fn labeled_slider_i32(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut i32,
    range: std::ops::RangeInclusive<i32>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(value, range));
    });
}

fn dir_editor(ui: &mut egui::Ui, dir: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut dir[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut dir[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut dir[2]).speed(0.05));
    });
}
