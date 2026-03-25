use std::path::Path;

use crate::app::actions::{Action, ActionSink, LightingPreset};
use crate::desktop_dialogs::FileDialogSelection;
use crate::settings::{
    AmbientOcclusionMode, BackgroundMode, EnvironmentBackgroundMode, EnvironmentSource,
    LocalReflectionMode, Settings, ShadingMode,
};

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
        if ui
            .small_button("Fast")
            .on_hover_text("Low quality, high performance")
            .clicked()
        {
            apply_preset_fast(&mut settings.render);
        }
        if ui
            .small_button("Balanced")
            .on_hover_text("Good balance of quality and speed")
            .clicked()
        {
            apply_preset_balanced(&mut settings.render);
        }
        if ui
            .small_button("Quality")
            .on_hover_text("Maximum visual quality (slower)")
            .clicked()
        {
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
                labeled_slider_i32(
                    ui,
                    "Steps",
                    &mut config.shadow_steps,
                    8..=128,
                    "Ray steps for shadow evaluation. More = sharper but slower",
                );
                labeled_slider(
                    ui,
                    "Penumbra K",
                    &mut config.shadow_penumbra_k,
                    1.0..=32.0,
                    false,
                    "Shadow softness. Higher = harder shadows",
                );
                labeled_slider(
                    ui,
                    "Bias",
                    &mut config.shadow_bias,
                    0.001..=0.2,
                    true,
                    "Offset to prevent self-shadowing artifacts",
                );
                labeled_slider(
                    ui,
                    "Min T",
                    &mut config.shadow_mint,
                    0.01..=0.5,
                    false,
                    "Shadow ray start distance (avoids self-intersection)",
                );
                labeled_slider(
                    ui,
                    "Max T",
                    &mut config.shadow_maxt,
                    5.0..=100.0,
                    false,
                    "Maximum shadow ray distance",
                );
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
                ui.horizontal(|ui| {
                    ui.label("Mode")
                        .on_hover_text("Fast keeps AO on diffuse indirect only. Balanced adds specular AO. Quality also uses a bent normal for higher-quality indirect occlusion.");
                    for mode in [
                        AmbientOcclusionMode::Fast,
                        AmbientOcclusionMode::Balanced,
                        AmbientOcclusionMode::Quality,
                    ] {
                        ui.selectable_value(&mut config.ao_mode, mode, mode.label());
                    }
                });
                labeled_slider_i32(
                    ui,
                    "Samples",
                    &mut config.ao_samples,
                    1..=16,
                    "Number of samples across the AO radius. More = smoother but not wider",
                );
                labeled_slider(
                    ui,
                    "Radius",
                    &mut config.ao_step,
                    0.01..=0.5,
                    false,
                    "Maximum AO reach in world units. More samples smooth the result without widening it",
                );
                labeled_slider(
                    ui,
                    "Decay",
                    &mut config.ao_decay,
                    0.5..=1.0,
                    false,
                    "Weight falloff across the AO radius. Lower values emphasize near-contact occlusion",
                );
                labeled_slider(
                    ui,
                    "Intensity",
                    &mut config.ao_intensity,
                    0.5..=10.0,
                    false,
                    "Strength of the normalized AO effect. Higher = darker contact shadows without changing reach",
                );
            });
            if ui.small_button("Reset").clicked() {
                config.reset_ao();
            }
        });

    // --- Raymarching ---
    egui::CollapsingHeader::new("Raymarching")
        .default_open(false)
        .show(ui, |ui| {
            labeled_slider_i32(
                ui,
                "Max Steps",
                &mut config.march_max_steps,
                32..=512,
                "Maximum ray march iterations. More = higher quality but slower",
            );
            labeled_slider(
                ui,
                "Epsilon",
                &mut config.march_epsilon,
                0.0001..=0.01,
                true,
                "Surface hit threshold. Smaller = more precise but needs more steps",
            );
            labeled_slider(
                ui,
                "Step Multiplier",
                &mut config.march_step_multiplier,
                0.01..=1.0,
                false,
                "Global base step factor. Sculpt regions may march more conservatively automatically to reduce overskip artifacts.",
            );
            labeled_slider(
                ui,
                "Max Distance",
                &mut config.march_max_distance,
                10.0..=200.0,
                false,
                "Far plane distance. Rays beyond this are considered misses",
            );
            if ui.small_button("Reset").clicked() {
                config.reset_raymarching();
            }
        });

    // --- Lighting ---
    egui::CollapsingHeader::new("Lighting")
        .default_open(false)
        .show(ui, |ui| {
            ui.weak("Key/Fill lights are scene Directional nodes.");
            ui.horizontal(|ui| {
                ui.label("Presets:");
                if ui
                    .small_button("Studio")
                    .on_hover_text("Classic warm key + cool fill studio setup")
                    .clicked()
                {
                    actions.push(Action::ApplyLightingPreset(LightingPreset::Studio));
                }
                if ui
                    .small_button("Outdoor")
                    .on_hover_text("Bright sunlight with blue sky fill")
                    .clicked()
                {
                    actions.push(Action::ApplyLightingPreset(LightingPreset::Outdoor));
                }
                if ui
                    .small_button("Dramatic")
                    .on_hover_text("High-contrast cinematic lighting")
                    .clicked()
                {
                    actions.push(Action::ApplyLightingPreset(LightingPreset::Dramatic));
                }
                if ui
                    .small_button("Flat")
                    .on_hover_text("Even, shadowless illumination")
                    .clicked()
                {
                    actions.push(Action::ApplyLightingPreset(LightingPreset::Flat));
                }
            });
            ui.separator();
            labeled_slider(
                ui,
                "Diffuse IBL",
                &mut config.ambient,
                0.0..=0.5,
                false,
                "Intensity of diffuse indirect environment lighting",
            );
            if ui.small_button("Reset").clicked() {
                config.reset_lighting();
            }
        });

    // --- Environment ---
    egui::CollapsingHeader::new("Environment")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut config.environment_source,
                    EnvironmentSource::ProceduralSky,
                    "Procedural Sky",
                );
                ui.selectable_value(
                    &mut config.environment_source,
                    EnvironmentSource::Hdri,
                    "HDR / EXR",
                );
            });

            ui.checkbox(&mut config.env_reflection_enabled, "Enable Specular IBL")
                .on_hover_text(
                    "Enable roughness-aware indirect specular from the active environment",
                );
            ui.checkbox(&mut config.specular_aa_enabled, "Enable Specular AA")
                .on_hover_text(
                    "Broadens unstable sub-pixel glossy highlights from scene lights and environment reflections without changing the stored material roughness.",
                );
            ui.horizontal(|ui| {
                ui.label("Local Reflections");
                for mode in [LocalReflectionMode::Off, LocalReflectionMode::Single] {
                    ui.selectable_value(&mut config.local_reflection_mode, mode, mode.label());
                }
            })
            .response
            .on_hover_text(
                "Off disables nearby scene reflections on glossy materials. Single enables one bounded local reflection bounce on smooth surfaces.",
            );
            ui.add_enabled_ui(config.env_reflection_enabled, |ui| {
                labeled_slider(
                    ui,
                    "Specular IBL",
                    &mut config.env_reflection_intensity,
                    0.0..=2.0,
                    false,
                    "Strength of indirect environment specular; direct scene lights are unaffected",
                );
            });

            labeled_slider(
                ui,
                "Rotation",
                &mut config.environment_rotation_degrees,
                -180.0..=180.0,
                false,
                "Rotate the environment around the world Y axis",
            );
            labeled_slider(
                ui,
                "Exposure",
                &mut config.environment_exposure,
                -8.0..=8.0,
                false,
                "Global environment exposure in stops before diffuse/specular scaling",
            );
            ui.horizontal(|ui| {
                ui.label("Lighting Bake");

                let mut use_auto_bake_resolution = config.environment_bake_resolution == 0;
                if ui
                    .checkbox(&mut use_auto_bake_resolution, "Auto")
                    .on_hover_text(
                        "Auto uses the imported HDR/EXR cubemap face-equivalent resolution. Procedural sky falls back to the default bake size.",
                    )
                    .changed()
                {
                    if use_auto_bake_resolution {
                        config.environment_bake_resolution = 0;
                    } else if config.environment_bake_resolution == 0 {
                        config.environment_bake_resolution = 512;
                    }
                }

                let mut bake_resolution = config.environment_bake_resolution.max(16) as i32;
                let response = ui.add_enabled(
                    !use_auto_bake_resolution,
                    egui::DragValue::new(&mut bake_resolution)
                        .speed(16)
                        .range(16..=4096)
                        .suffix(" px"),
                );
                if response.changed() {
                    config.environment_bake_resolution = bake_resolution as u32;
                }
            });
            ui.small(
                "This controls lighting cubemap quality. The visible HDR/EXR background stays sharp at blur 0 and can be rebaked lower separately for performance.",
            );

            if config.environment_source == EnvironmentSource::Hdri {
                ui.horizontal(|ui| {
                    if ui.button("Import HDR / EXR").clicked() {
                        match crate::desktop_dialogs::environment_hdri_dialog() {
                            FileDialogSelection::Selected(path) => {
                                config.hdri_path = Some(path.to_string_lossy().into_owned());
                            }
                            FileDialogSelection::Cancelled | FileDialogSelection::Unsupported => {}
                        }
                    }

                    if ui
                        .add_enabled(config.hdri_path.is_some(), egui::Button::new("Clear"))
                        .clicked()
                    {
                        config.hdri_path = None;
                    }
                });

                if let Some(path) = config.hdri_path.as_deref() {
                    let label = Path::new(path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(path);
                    ui.label(format!("Loaded: {label}"));
                    ui.small(path);
                } else {
                    ui.weak("No HDR or EXR environment selected. Lighting falls back to the procedural sky until one is imported.");
                }
            }

            ui.separator();
            ui.label("Visible Background");
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut config.environment_background_mode,
                    EnvironmentBackgroundMode::Environment,
                    "Environment",
                );
                ui.selectable_value(
                    &mut config.environment_background_mode,
                    EnvironmentBackgroundMode::Procedural,
                    "Procedural",
                );
            });

            if config.environment_background_mode == EnvironmentBackgroundMode::Environment {
                labeled_slider(
                    ui,
                    "Background Blur",
                    &mut config.environment_background_blur,
                    0.0..=1.0,
                    false,
                    "0 keeps the environment sharp, 1 uses the fully blurred prefiltered background",
                );
            }

            if config.environment_source == EnvironmentSource::ProceduralSky
                || config.environment_background_mode == EnvironmentBackgroundMode::Procedural
            {
                ui.separator();
                ui.label("Procedural Background");
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut config.background_mode,
                        BackgroundMode::SkyGradient,
                        "Sky Gradient",
                    );
                    ui.selectable_value(
                        &mut config.background_mode,
                        BackgroundMode::SolidColor,
                        "Solid Color",
                    );
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
            }

            if ui.small_button("Reset").clicked() {
                config.reset_environment();
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
                labeled_slider(
                    ui,
                    "Density",
                    &mut config.fog_density,
                    0.001..=0.2,
                    true,
                    "Fog thickness. Higher = objects fade sooner",
                );
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(&mut config.fog_color);
                });
            });
            if ui.small_button("Reset").clicked() {
                config.reset_fog();
            }
        });

    // --- Volumetric Scattering ---
    egui::CollapsingHeader::new("Volumetric Scattering")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Ray march steps for volumetric light scattering (god rays).");
            ui.horizontal(|ui| {
                ui.label("Steps:");
                let mut steps = config.volumetric_steps as f32;
                if ui
                    .add(
                        egui::Slider::new(&mut steps, 8.0..=48.0)
                            .step_by(1.0)
                            .suffix(" steps"),
                    )
                    .on_hover_text("Fewer steps = faster but more banding. 24 is a good default.")
                    .changed()
                {
                    config.volumetric_steps = steps as u32;
                }
            });
        });

    // --- Bloom ---
    egui::CollapsingHeader::new("Bloom")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut config.bloom_enabled, "Enable Bloom")
                .on_hover_text("Glow effect on bright areas (post-process star pattern)");
            ui.add_enabled_ui(config.bloom_enabled, |ui| {
                labeled_slider(
                    ui,
                    "Threshold",
                    &mut config.bloom_threshold,
                    0.1..=2.0,
                    false,
                    "Brightness cutoff for bloom. Lower = more glow",
                );
                labeled_slider(
                    ui,
                    "Intensity",
                    &mut config.bloom_intensity,
                    0.05..=2.0,
                    false,
                    "Strength of the bloom glow",
                );
                labeled_slider(
                    ui,
                    "Radius",
                    &mut config.bloom_radius,
                    1.0..=8.0,
                    false,
                    "Spread of the bloom in pixels per step",
                );
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
            labeled_slider(
                ui,
                "Thickness",
                &mut config.outline_thickness,
                1.0..=8.0,
                false,
                "Outline width in screen pixels",
            );
            if ui.small_button("Reset").clicked() {
                config.reset_outline();
            }
        });

    // --- Cross-Section ---
    egui::CollapsingHeader::new("Visualization")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for mode in [
                    ShadingMode::Full,
                    ShadingMode::Solid,
                    ShadingMode::Clay,
                    ShadingMode::Normals,
                    ShadingMode::Matcap,
                    ShadingMode::StepHeatmap,
                    ShadingMode::FieldQuality,
                    ShadingMode::CrossSection,
                ] {
                    ui.selectable_value(&mut config.shading_mode, mode, mode.label());
                }
            });
            ui.label(
                "Field Quality visualizes signed-distance drift: green is close to |grad| = 1, red indicates a degraded field.",
            );
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
                labeled_slider(
                    ui,
                    "Position",
                    &mut config.cross_section_position,
                    -5.0..=5.0,
                    false,
                    "Slice plane position along the selected axis",
                );
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
            ui.checkbox(
                &mut config.debug_force_manual_sculpt_sampling,
                "Force Manual Sculpt Sampling",
            )
            .on_hover_text(
                "Render sculpt nodes through the manual storage-buffer sampler instead of the texture path.\nUseful for isolating texture interpolation artifacts.",
            );
        });

    if settings.render != before {
        if settings.render.needs_shader_rebuild(&before) {
            // Shader-affecting settings changed Ã¢â‚¬â€ full rebuild
            actions.push(Action::SettingsChanged);
        } else {
            // Light-only changes Ã¢â‚¬â€ save settings, no shader rebuild needed
            // (light values are in the uniform buffer, uploaded every frame)
            settings.save();
        }
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

fn apply_preset_fast(config: &mut crate::settings::RenderConfig) {
    apply_default_environment_preset(config);
    config.env_reflection_enabled = false;
    config.shadows_enabled = false;
    config.ao_enabled = false;
    config.ao_mode = AmbientOcclusionMode::Fast;
    config.march_max_steps = 64;
    config.march_epsilon = 0.005;
    config.fog_enabled = false;
    config.tonemapping_aces = false;
    config.specular_aa_enabled = false;
    config.local_reflection_mode = LocalReflectionMode::Off;
    config.sculpt_fast_mode = true;
    config.auto_reduce_steps = true;
    config.interaction_render_scale = 0.35;
    config.rest_render_scale = 0.75;
}

fn apply_preset_balanced(config: &mut crate::settings::RenderConfig) {
    apply_default_environment_preset(config);
    config.env_reflection_enabled = true;
    config.shadows_enabled = false;
    config.ao_enabled = true;
    config.ao_samples = 5;
    config.ao_step = 0.4;
    config.ao_decay = 0.95;
    config.ao_intensity = 1.0;
    config.ao_mode = AmbientOcclusionMode::Balanced;
    config.march_max_steps = 128;
    config.march_epsilon = 0.002;
    config.march_step_multiplier = 0.9;
    config.specular_aa_enabled = true;
    config.local_reflection_mode = LocalReflectionMode::Single;
    config.sculpt_fast_mode = false;
    config.auto_reduce_steps = true;
    config.interaction_render_scale = 0.5;
    config.rest_render_scale = 1.0;
}

fn apply_preset_quality(config: &mut crate::settings::RenderConfig) {
    apply_default_environment_preset(config);
    config.env_reflection_enabled = true;
    config.shadows_enabled = true;
    config.shadow_steps = 64;
    config.shadow_penumbra_k = 8.0;
    config.ao_enabled = true;
    config.ao_samples = 8;
    config.ao_step = 0.45;
    config.ao_decay = 0.98;
    config.ao_intensity = 1.25;
    config.ao_mode = AmbientOcclusionMode::Quality;
    config.march_max_steps = 256;
    config.march_epsilon = 0.001;
    config.march_step_multiplier = 0.9;
    config.tonemapping_aces = true;
    config.specular_aa_enabled = true;
    config.local_reflection_mode = LocalReflectionMode::Single;
    config.sculpt_fast_mode = false;
    config.auto_reduce_steps = false;
    config.interaction_render_scale = 0.5;
    config.rest_render_scale = 1.0;
}

fn apply_default_environment_preset(config: &mut crate::settings::RenderConfig) {
    let defaults = crate::settings::RenderConfig::default();
    config.env_reflection_enabled = defaults.env_reflection_enabled;
    config.env_reflection_intensity = defaults.env_reflection_intensity;
    config.environment_source = defaults.environment_source;
    config.hdri_path = defaults.hdri_path;
    config.environment_rotation_degrees = defaults.environment_rotation_degrees;
    config.environment_exposure = defaults.environment_exposure;
    config.environment_bake_resolution = defaults.environment_bake_resolution;
    config.environment_background_mode = defaults.environment_background_mode;
    config.environment_background_blur = defaults.environment_background_blur;
}

#[cfg(test)]
mod tests {
    use super::{apply_preset_balanced, apply_preset_fast, apply_preset_quality};
    use crate::settings::{EnvironmentBackgroundMode, EnvironmentSource, RenderConfig};

    #[test]
    fn presets_restore_default_environment_selection_and_shared_environment_defaults() {
        let defaults = RenderConfig::default();

        for apply_preset in [
            apply_preset_fast,
            apply_preset_balanced,
            apply_preset_quality,
        ] {
            let mut config = RenderConfig::default();
            config.environment_source = EnvironmentSource::ProceduralSky;
            config.hdri_path = Some("custom/custom.exr".into());
            config.env_reflection_enabled = !defaults.env_reflection_enabled;
            config.env_reflection_intensity = 1.1;
            config.environment_rotation_degrees = 32.0;
            config.environment_exposure = 1.5;
            config.environment_bake_resolution = 1024;
            config.environment_background_mode = EnvironmentBackgroundMode::Environment;
            config.environment_background_blur = 0.65;

            apply_preset(&mut config);

            assert_eq!(
                config.env_reflection_intensity,
                defaults.env_reflection_intensity
            );
            assert_eq!(config.environment_source, defaults.environment_source);
            assert_eq!(config.hdri_path, defaults.hdri_path);
            assert_eq!(
                config.environment_rotation_degrees,
                defaults.environment_rotation_degrees
            );
            assert_eq!(config.environment_exposure, defaults.environment_exposure);
            assert_eq!(
                config.environment_bake_resolution,
                defaults.environment_bake_resolution
            );
            assert_eq!(
                config.environment_background_mode,
                defaults.environment_background_mode
            );
            assert_eq!(
                config.environment_background_blur,
                defaults.environment_background_blur
            );
        }
    }

    #[test]
    fn fast_preset_disables_specular_ibl_and_restores_default_intensity() {
        let defaults = RenderConfig::default();
        let mut config = RenderConfig::default();
        config.env_reflection_enabled = true;
        config.env_reflection_intensity = 1.4;

        apply_preset_fast(&mut config);

        assert!(!config.env_reflection_enabled);
        assert_eq!(
            config.env_reflection_intensity,
            defaults.env_reflection_intensity
        );
    }

    #[test]
    fn balanced_and_quality_presets_enable_specular_ibl_and_restore_default_intensity() {
        let defaults = RenderConfig::default();

        for apply_preset in [apply_preset_balanced, apply_preset_quality] {
            let mut config = RenderConfig::default();
            config.env_reflection_enabled = false;
            config.env_reflection_intensity = 1.4;

            apply_preset(&mut config);

            assert!(config.env_reflection_enabled);
            assert_eq!(
                config.env_reflection_intensity,
                defaults.env_reflection_intensity
            );
        }
    }
}
