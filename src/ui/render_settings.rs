use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::settings::{BackgroundMode, ShadingMode, Settings};

/// Draw the Render Settings panel. Pushes `Action::SettingsChanged` if a shader-affecting
/// setting changed.
pub fn draw(ui: &mut egui::Ui, settings: &mut Settings, actions: &mut ActionSink) {
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
                labeled_slider_i32(ui, "Steps", &mut config.shadow_steps, 8..=128,
                    "Ray steps for shadow evaluation. More = sharper but slower");
                labeled_slider(ui, "Penumbra K", &mut config.shadow_penumbra_k, 1.0..=32.0, false,
                    "Shadow softness. Higher = harder shadows");
                labeled_slider(ui, "Bias", &mut config.shadow_bias, 0.001..=0.2, true,
                    "Offset to prevent self-shadowing artifacts");
                labeled_slider(ui, "Min T", &mut config.shadow_mint, 0.01..=0.5, false,
                    "Shadow ray start distance (avoids self-intersection)");
                labeled_slider(ui, "Max T", &mut config.shadow_maxt, 5.0..=100.0, false,
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
                labeled_slider_i32(ui, "Samples", &mut config.ao_samples, 1..=16,
                    "Number of AO sample steps. More = smoother but slower");
                labeled_slider(ui, "Step Size", &mut config.ao_step, 0.01..=0.5, false,
                    "Distance between AO samples. Larger = wider darkening");
                labeled_slider(ui, "Decay", &mut config.ao_decay, 0.5..=1.0, false,
                    "How quickly AO fades with distance (closer to 1.0 = slower fade)");
                labeled_slider(ui, "Intensity", &mut config.ao_intensity, 0.5..=10.0, false,
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
            labeled_slider_i32(ui, "Max Steps", &mut config.march_max_steps, 32..=512,
                "Maximum ray march iterations. More = higher quality but slower");
            labeled_slider(ui, "Epsilon", &mut config.march_epsilon, 0.0001..=0.01, true,
                "Surface hit threshold. Smaller = more precise but needs more steps");
            labeled_slider(ui, "Step Multiplier", &mut config.march_step_multiplier, 0.5..=1.0, false,
                "Conservative step factor (<1.0 prevents artifacts on thin features)");
            labeled_slider(ui, "Max Distance", &mut config.march_max_distance, 10.0..=200.0, false,
                "Far plane distance. Rays beyond this are considered misses");
            if ui.small_button("Reset").clicked() {
                config.reset_raymarching();
            }
        });

    // --- Lighting ---
    egui::CollapsingHeader::new("Lighting")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Presets:");
                if ui.small_button("Studio").on_hover_text("Dark studio environment").clicked() {
                    apply_lighting_preset_studio(config);
                }
                if ui.small_button("Outdoor").on_hover_text("Bright outdoor lighting").clicked() {
                    apply_lighting_preset_outdoor(config);
                }
                if ui.small_button("Dramatic").on_hover_text("High-contrast dramatic lighting").clicked() {
                    apply_lighting_preset_dramatic(config);
                }
                if ui.small_button("Flat").on_hover_text("Even, low-contrast lighting").clicked() {
                    apply_lighting_preset_flat(config);
                }
            });
            ui.separator();
            ui.label("Key Light Direction");
            dir_editor(ui, &mut config.key_light_dir);
            labeled_slider(ui, "Diffuse", &mut config.key_diffuse, 0.0..=2.0, false,
                "Key light diffuse brightness");
            labeled_slider(ui, "Spec Power", &mut config.key_spec_power, 1.0..=128.0, true,
                "Specular highlight sharpness (higher = tighter highlight)");
            labeled_slider(ui, "Spec Intensity", &mut config.key_spec_intensity, 0.0..=2.0, false,
                "Specular highlight brightness");

            ui.separator();
            ui.label("Fill Light Direction");
            dir_editor(ui, &mut config.fill_light_dir);
            labeled_slider(ui, "Fill Intensity", &mut config.fill_intensity, 0.0..=1.0, false,
                "Intensity of the secondary fill light");

            ui.separator();
            labeled_slider(ui, "Ambient", &mut config.ambient, 0.0..=0.5, false,
                "Minimum base lighting (prevents fully black areas)");
            if ui.small_button("Reset").clicked() {
                config.reset_lighting();
            }
        });

    // --- Environment Reflection ---
    egui::CollapsingHeader::new("Environment Reflection")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.env_reflection_enabled, "Enable Env Reflection")
                .on_hover_text("Reflect the sky gradient on glossy/metallic surfaces");
            ui.add_enabled_ui(config.env_reflection_enabled, |ui| {
                labeled_slider(ui, "Intensity", &mut config.env_reflection_intensity, 0.0..=2.0, false,
                    "Strength of environment reflections");
            });
        });

    // --- Sky / Background ---
    egui::CollapsingHeader::new("Sky / Background")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut config.background_mode, BackgroundMode::SkyGradient, "Sky Gradient");
                ui.selectable_value(&mut config.background_mode, BackgroundMode::SolidColor, "Solid Color");
            });
            match config.background_mode {
                BackgroundMode::SkyGradient => {
                    ui.horizontal(|ui| {
                        ui.label("Horizon:");
                        ui.color_edit_button_rgb(&mut config.sky_horizon);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Zenith:");
                        ui.color_edit_button_rgb(&mut config.sky_zenith);
                    });
                }
                BackgroundMode::SolidColor => {
                    ui.horizontal(|ui| {
                        ui.label("Color:");
                        ui.color_edit_button_rgb(&mut config.bg_solid_color);
                    });
                }
            }
            if ui.small_button("Reset").clicked() {
                config.reset_sky();
            }
        });

    // --- Subsurface Scattering ---
    egui::CollapsingHeader::new("Subsurface Scattering")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.sss_enabled, "Enable SSS")
                .on_hover_text("Thickness-based subsurface scattering (simulates light passing through thin geometry)");
            ui.add_enabled_ui(config.sss_enabled, |ui| {
                labeled_slider(ui, "Strength", &mut config.sss_strength, 0.5..=20.0, false,
                    "How quickly light attenuates through the object. Lower = more translucent");
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(&mut config.sss_color);
                });
            });
            if ui.small_button("Reset").clicked() {
                config.reset_sss();
            }
        });

    // --- Fog ---
    egui::CollapsingHeader::new("Fog")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.fog_enabled, "Enable Fog")
                .on_hover_text("Distance-based exponential fog with sun scattering");
            ui.add_enabled_ui(config.fog_enabled, |ui| {
                labeled_slider(ui, "Density", &mut config.fog_density, 0.001..=0.2, true,
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

    // --- Bloom ---
    egui::CollapsingHeader::new("Bloom")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.bloom_enabled, "Enable Bloom")
                .on_hover_text("Glow effect on bright areas (post-process star pattern)");
            ui.add_enabled_ui(config.bloom_enabled, |ui| {
                labeled_slider(ui, "Threshold", &mut config.bloom_threshold, 0.1..=2.0, false,
                    "Brightness cutoff for bloom. Lower = more glow");
                labeled_slider(ui, "Intensity", &mut config.bloom_intensity, 0.05..=2.0, false,
                    "Strength of the bloom glow");
                labeled_slider(ui, "Radius", &mut config.bloom_radius, 1.0..=8.0, false,
                    "Spread of the bloom in pixels per step");
            });
        });

    // --- Gamma / Tonemapping ---
    egui::CollapsingHeader::new("Gamma")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider(ui, "Gamma", &mut config.gamma, 1.0..=3.0, false,
                "Display gamma correction (2.2 = standard sRGB)");
            ui.checkbox(&mut config.tonemapping_aces, "ACES Filmic Tonemapping")
                .on_hover_text("Apply ACES filmic curve before gamma. Better highlight rolloff but shifts hue slightly.");
            if ui.small_button("Reset").clicked() {
                config.reset_gamma();
            }
        });

    // --- Selection Outline ---
    egui::CollapsingHeader::new("Selection Outline")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Color:");
                ui.color_edit_button_rgb(&mut config.outline_color);
            });
            labeled_slider(ui, "Thickness", &mut config.outline_thickness, 1.0..=8.0, false,
                "Outline width in screen pixels");
            if ui.small_button("Reset").clicked() {
                config.reset_outline();
            }
        });

    // --- Cross-Section ---
    if config.shading_mode == ShadingMode::CrossSection {
        egui::CollapsingHeader::new("Cross-Section")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Axis:");
                    ui.selectable_value(&mut config.cross_section_axis, 0, "X");
                    ui.selectable_value(&mut config.cross_section_axis, 1, "Y");
                    ui.selectable_value(&mut config.cross_section_axis, 2, "Z");
                });
                labeled_slider(ui, "Position", &mut config.cross_section_position, -5.0..=5.0, false,
                    "Slice plane position along the selected axis");
            });
    }

    egui::CollapsingHeader::new("Sculpt")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider(
                ui,
                "Safety Border",
                &mut config.sculpt_safety_border,
                0.0..=0.15,
                false,
                "Border zone that always triggers navigation (fraction of viewport, 0 = disabled)",
            );
        });

    if settings.render != before {
        actions.push(Action::SettingsChanged);
    }
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

fn labeled_slider_i32(
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

fn apply_lighting_preset_studio(config: &mut crate::settings::RenderConfig) {
    config.key_light_dir = [1.0, 2.0, 3.0];
    config.key_diffuse = 0.85;
    config.key_spec_power = 32.0;
    config.key_spec_intensity = 0.4;
    config.fill_light_dir = [-1.0, 0.5, -1.0];
    config.fill_intensity = 0.25;
    config.ambient = 0.06;
    config.sky_horizon = [0.10, 0.10, 0.16];
    config.sky_zenith = [0.02, 0.02, 0.05];
}

fn apply_lighting_preset_outdoor(config: &mut crate::settings::RenderConfig) {
    config.key_light_dir = [0.5, 3.0, 1.0];
    config.key_diffuse = 1.0;
    config.key_spec_power = 24.0;
    config.key_spec_intensity = 0.3;
    config.fill_light_dir = [-0.5, 0.3, -1.0];
    config.fill_intensity = 0.35;
    config.ambient = 0.1;
    config.sky_horizon = [0.55, 0.65, 0.80];
    config.sky_zenith = [0.20, 0.30, 0.60];
}

fn apply_lighting_preset_dramatic(config: &mut crate::settings::RenderConfig) {
    config.key_light_dir = [2.0, 1.0, 0.5];
    config.key_diffuse = 1.2;
    config.key_spec_power = 64.0;
    config.key_spec_intensity = 0.6;
    config.fill_light_dir = [-1.0, -0.2, -0.5];
    config.fill_intensity = 0.08;
    config.ambient = 0.02;
    config.sky_horizon = [0.05, 0.04, 0.06];
    config.sky_zenith = [0.01, 0.01, 0.02];
}

fn apply_lighting_preset_flat(config: &mut crate::settings::RenderConfig) {
    config.key_light_dir = [0.0, 1.0, 0.5];
    config.key_diffuse = 0.5;
    config.key_spec_power = 8.0;
    config.key_spec_intensity = 0.1;
    config.fill_light_dir = [0.0, 0.5, -1.0];
    config.fill_intensity = 0.4;
    config.ambient = 0.2;
    config.sky_horizon = [0.3, 0.3, 0.35];
    config.sky_zenith = [0.15, 0.15, 0.20];
}
