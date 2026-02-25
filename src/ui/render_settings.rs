use eframe::egui;

use crate::settings::Settings;

/// Draw the Render Settings panel. Returns `true` if a shader-affecting setting changed.
pub fn draw(ui: &mut egui::Ui, settings: &mut Settings) -> bool {
    let before = settings.render.clone();

    ui.heading("Render Settings");

    ui.horizontal(|ui| {
        if ui.small_button("Reset All").clicked() {
            settings.render.reset_all();
        }
        ui.separator();
        ui.label("Presets:");
        if ui.small_button("Fast").on_hover_text("Low quality, high performance").clicked() {
            apply_preset_fast(&mut settings.render);
        }
        if ui.small_button("Balanced").on_hover_text("Good balance of quality and speed").clicked() {
            apply_preset_balanced(&mut settings.render);
        }
        if ui.small_button("Quality").on_hover_text("Maximum visual quality (slower)").clicked() {
            apply_preset_quality(&mut settings.render);
        }
    });

    ui.separator();

    let config = &mut settings.render;

    // --- Shadows ---
    egui::CollapsingHeader::new("Shadows")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.shadows_enabled, "Enable Shadows")
                .on_hover_text("Soft shadows from key light (expensive)");
            ui.add_enabled_ui(config.shadows_enabled, |ui| {
                labeled_slider_i32_tip(ui, "Steps", &mut config.shadow_steps, 8..=128,
                    "Ray steps for shadow evaluation. More = sharper but slower");
                labeled_slider_tip(ui, "Penumbra K", &mut config.shadow_penumbra_k, 1.0..=32.0, false,
                    "Shadow softness. Higher = harder shadows");
                labeled_slider_tip(ui, "Bias", &mut config.shadow_bias, 0.001..=0.2, true,
                    "Offset to prevent self-shadowing artifacts");
                labeled_slider_tip(ui, "Min T", &mut config.shadow_mint, 0.01..=0.5, false,
                    "Shadow ray start distance (avoids self-intersection)");
                labeled_slider_tip(ui, "Max T", &mut config.shadow_maxt, 5.0..=100.0, false,
                    "Maximum shadow ray distance");
            });
            if ui.small_button("Reset").clicked() {
                config.reset_shadows();
            }
        });

    // --- Ambient Occlusion ---
    egui::CollapsingHeader::new("Ambient Occlusion")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.ao_enabled, "Enable AO")
                .on_hover_text("Ambient occlusion darkens crevices and corners");
            ui.add_enabled_ui(config.ao_enabled, |ui| {
                labeled_slider_i32_tip(ui, "Samples", &mut config.ao_samples, 1..=16,
                    "Number of AO sample steps. More = smoother but slower");
                labeled_slider_tip(ui, "Step Size", &mut config.ao_step, 0.01..=0.5, false,
                    "Distance between AO samples. Larger = wider darkening");
                labeled_slider_tip(ui, "Decay", &mut config.ao_decay, 0.5..=1.0, false,
                    "How quickly AO fades with distance (closer to 1.0 = slower fade)");
                labeled_slider_tip(ui, "Intensity", &mut config.ao_intensity, 0.5..=10.0, false,
                    "Strength of the AO effect. Higher = darker crevices");
            });
            if ui.small_button("Reset").clicked() {
                config.reset_ao();
            }
        });

    // --- Raymarching ---
    egui::CollapsingHeader::new("Raymarching")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider_i32_tip(ui, "Max Steps", &mut config.march_max_steps, 32..=512,
                "Maximum ray march iterations. More = higher quality but slower");
            labeled_slider_tip(ui, "Epsilon", &mut config.march_epsilon, 0.0001..=0.01, true,
                "Surface hit threshold. Smaller = more precise but needs more steps");
            labeled_slider_tip(ui, "Step Multiplier", &mut config.march_step_multiplier, 0.5..=1.0, false,
                "Conservative step factor (<1.0 prevents artifacts on thin features)");
            labeled_slider_tip(ui, "Max Distance", &mut config.march_max_distance, 10.0..=200.0, false,
                "Far plane distance. Rays beyond this are considered misses");
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
            labeled_slider_tip(ui, "Diffuse", &mut config.key_diffuse, 0.0..=2.0, false,
                "Key light diffuse brightness");
            labeled_slider_tip(ui, "Spec Power", &mut config.key_spec_power, 1.0..=128.0, true,
                "Specular highlight sharpness (higher = tighter highlight)");
            labeled_slider_tip(ui, "Spec Intensity", &mut config.key_spec_intensity, 0.0..=2.0, false,
                "Specular highlight brightness");

            ui.separator();
            ui.label("Fill Light Direction");
            dir_editor(ui, &mut config.fill_light_dir);
            labeled_slider_tip(ui, "Fill Intensity", &mut config.fill_intensity, 0.0..=1.0, false,
                "Intensity of the secondary fill light");

            ui.separator();
            labeled_slider_tip(ui, "Ambient", &mut config.ambient, 0.0..=0.5, false,
                "Minimum base lighting (prevents fully black areas)");
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
            ui.checkbox(&mut config.fog_enabled, "Enable Fog")
                .on_hover_text("Distance-based exponential fog with sun scattering");
            ui.add_enabled_ui(config.fog_enabled, |ui| {
                labeled_slider_tip(ui, "Density", &mut config.fog_density, 0.001..=0.2, true,
                    "Fog thickness. Higher = objects fade sooner");
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(&mut config.fog_color);
                });
            });
            if ui.small_button("Reset").clicked() {
                config.reset_fog();
            }
        });

    // --- Gamma / Tonemapping ---
    egui::CollapsingHeader::new("Gamma")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider_tip(ui, "Gamma", &mut config.gamma, 1.0..=3.0, false,
                "Display gamma correction (2.2 = standard sRGB)");
            ui.checkbox(&mut config.tonemapping_aces, "ACES Filmic Tonemapping")
                .on_hover_text("Apply ACES filmic curve before gamma. Better highlight rolloff but shifts hue slightly.");
            if ui.small_button("Reset").clicked() {
                config.reset_gamma();
            }
        });

    // --- Viewport ---
    egui::CollapsingHeader::new("Viewport")
        .default_open(true)
        .show(ui, |ui| {
            ui.checkbox(&mut config.show_grid, "Show Grid")
                .on_hover_text("Display ground plane grid at Y=0");
        });

    // --- Selection Outline ---
    egui::CollapsingHeader::new("Selection Outline")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Color:");
                ui.color_edit_button_rgb(&mut config.outline_color);
            });
            labeled_slider_tip(ui, "Thickness", &mut config.outline_thickness, 1.0..=5.0, false,
                "Outline width in screen pixels");
            if ui.small_button("Reset").clicked() {
                config.reset_outline();
            }
        });

    let changed = settings.render != before;
    if changed {
        settings.save();
    }
    changed
}

fn labeled_slider_tip(
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

fn labeled_slider_i32_tip(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut i32,
    range: std::ops::RangeInclusive<i32>,
    tooltip: &str,
) {
    ui.horizontal(|ui| {
        let lbl = ui.label(label);
        if !tooltip.is_empty() {
            lbl.on_hover_text(tooltip);
        }
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

fn apply_preset_fast(config: &mut crate::settings::RenderConfig) {
    config.shadows_enabled = false;
    config.ao_enabled = false;
    config.march_max_steps = 64;
    config.march_epsilon = 0.005;
    config.fog_enabled = false;
    config.tonemapping_aces = false;
    config.sculpt_fast_mode = true;
    config.auto_reduce_steps = true;
    config.interaction_render_scale = 0.35;
    config.rest_render_scale = 0.75;
}

fn apply_preset_balanced(config: &mut crate::settings::RenderConfig) {
    config.shadows_enabled = false;
    config.ao_enabled = true;
    config.ao_samples = 5;
    config.ao_step = 0.08;
    config.march_max_steps = 128;
    config.march_epsilon = 0.002;
    config.march_step_multiplier = 0.9;
    config.sculpt_fast_mode = false;
    config.auto_reduce_steps = true;
    config.interaction_render_scale = 0.5;
    config.rest_render_scale = 1.0;
}

fn apply_preset_quality(config: &mut crate::settings::RenderConfig) {
    config.shadows_enabled = true;
    config.shadow_steps = 64;
    config.shadow_penumbra_k = 8.0;
    config.ao_enabled = true;
    config.ao_samples = 8;
    config.ao_step = 0.06;
    config.ao_intensity = 4.0;
    config.march_max_steps = 256;
    config.march_epsilon = 0.001;
    config.march_step_multiplier = 0.9;
    config.tonemapping_aces = true;
    config.sculpt_fast_mode = false;
    config.auto_reduce_steps = false;
    config.interaction_render_scale = 0.5;
    config.rest_render_scale = 1.0;
}
