use super::super::super::host_state::SlintHostState;
use super::super::context::CallbackContext;
use super::super::mutation::mutate_host_and_tick;
use crate::app::slint_frontend::{RenderSettingsAction, SlintHostWindow};
use crate::settings::{
    AmbientOcclusionMode, BackgroundMode, EnvironmentBackgroundMode, EnvironmentSource,
    LocalReflectionMode, ShadingMode,
};

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_render_settings_action(move |action, axis, value| {
        mutate_host_and_tick(&context, move |host_state| {
            handle_render_settings_action(host_state, action, axis, value);
        });
    });
}

fn handle_render_settings_action(
    host_state: &mut SlintHostState,
    action: RenderSettingsAction,
    axis: i32,
    value: f32,
) {
    match action {
        RenderSettingsAction::SetShowGrid => {
            host_state.app.set_render_show_grid(value >= 0.5);
        }
        RenderSettingsAction::SetShowNodeLabels => {
            host_state.app.set_render_show_node_labels(value >= 0.5);
        }
        RenderSettingsAction::SetShowBoundingBox => {
            host_state.app.set_render_show_bounding_box(value >= 0.5);
        }
        RenderSettingsAction::SetShowLightGizmos => {
            host_state.app.set_render_show_light_gizmos(value >= 0.5);
        }
        RenderSettingsAction::SetShadingMode => {
            if let Some(mode) = shading_mode_from_value(value) {
                host_state
                    .app
                    .edit_render_config(|config| config.shading_mode = mode);
            }
        }
        RenderSettingsAction::SetCrossSectionAxis => {
            host_state.app.edit_render_config(|config| {
                config.cross_section_axis = value.round().clamp(0.0, 2.0) as u8;
            });
        }
        RenderSettingsAction::SetCrossSectionPosition => {
            host_state.app.edit_render_config(|config| {
                config.cross_section_position = value.clamp(-4.0, 4.0);
            });
        }
        RenderSettingsAction::SetSculptSafetyBorder => {
            host_state.app.edit_render_config(|config| {
                config.sculpt_safety_border = value.clamp(0.0, 0.25);
            });
        }
        RenderSettingsAction::SetShadowsEnabled => {
            host_state.app.set_render_shadows_enabled(value >= 0.5);
        }
        RenderSettingsAction::SetShadowSteps => {
            host_state.app.edit_render_config(|config| {
                config.shadow_steps = value.round().clamp(1.0, 256.0) as i32;
            });
        }
        RenderSettingsAction::SetShadowPenumbraK => {
            host_state.app.edit_render_config(|config| {
                config.shadow_penumbra_k = value.clamp(0.0, 64.0);
            });
        }
        RenderSettingsAction::SetShadowBias => {
            host_state.app.edit_render_config(|config| {
                config.shadow_bias = value.clamp(0.0, 0.1);
            });
        }
        RenderSettingsAction::SetShadowMint => {
            host_state.app.edit_render_config(|config| {
                config.shadow_mint = value.clamp(0.0, 2.0);
            });
        }
        RenderSettingsAction::SetShadowMaxt => {
            host_state.app.edit_render_config(|config| {
                config.shadow_maxt = value.clamp(0.1, 512.0);
            });
        }
        RenderSettingsAction::SetAoEnabled => {
            host_state.app.set_render_ao_enabled(value >= 0.5);
        }
        RenderSettingsAction::SetAoSamples => {
            host_state.app.edit_render_config(|config| {
                config.ao_samples = value.round().clamp(1.0, 128.0) as i32;
            });
        }
        RenderSettingsAction::SetAoStep => {
            host_state.app.edit_render_config(|config| {
                config.ao_step = value.clamp(0.001, 2.0);
            });
        }
        RenderSettingsAction::SetAoDecay => {
            host_state.app.edit_render_config(|config| {
                config.ao_decay = value.clamp(0.0, 2.0);
            });
        }
        RenderSettingsAction::SetAoIntensity => {
            host_state.app.edit_render_config(|config| {
                config.ao_intensity = value.clamp(0.0, 4.0);
            });
        }
        RenderSettingsAction::SetAoMode => {
            if let Some(mode) = ao_mode_from_value(value) {
                host_state.app.edit_render_config(|config| {
                    config.ao_mode = mode;
                });
            }
        }
        RenderSettingsAction::SetMarchMaxSteps => {
            host_state.app.edit_render_config(|config| {
                config.march_max_steps = value.round().clamp(8.0, 1024.0) as i32;
            });
        }
        RenderSettingsAction::SetMarchEpsilon => {
            host_state.app.edit_render_config(|config| {
                config.march_epsilon = value.clamp(0.0001, 0.1);
            });
        }
        RenderSettingsAction::SetMarchStepMultiplier => {
            host_state.app.edit_render_config(|config| {
                config.march_step_multiplier = value.clamp(0.1, 4.0);
            });
        }
        RenderSettingsAction::SetMarchMaxDistance => {
            host_state.app.edit_render_config(|config| {
                config.march_max_distance = value.clamp(1.0, 4096.0);
            });
        }
        RenderSettingsAction::SetKeyLightDir => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.key_light_dir, axis, value.clamp(-1.0, 1.0));
            });
        }
        RenderSettingsAction::SetKeyDiffuse => {
            host_state.app.edit_render_config(|config| {
                config.key_diffuse = value.clamp(0.0, 8.0);
            });
        }
        RenderSettingsAction::SetKeySpecPower => {
            host_state.app.edit_render_config(|config| {
                config.key_spec_power = value.clamp(1.0, 256.0);
            });
        }
        RenderSettingsAction::SetKeySpecIntensity => {
            host_state.app.edit_render_config(|config| {
                config.key_spec_intensity = value.clamp(0.0, 8.0);
            });
        }
        RenderSettingsAction::SetKeyLightColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.key_light_color, axis, value.clamp(0.0, 4.0));
            });
        }
        RenderSettingsAction::SetFillLightDir => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.fill_light_dir, axis, value.clamp(-1.0, 1.0));
            });
        }
        RenderSettingsAction::SetFillIntensity => {
            host_state.app.edit_render_config(|config| {
                config.fill_intensity = value.clamp(0.0, 8.0);
            });
        }
        RenderSettingsAction::SetFillLightColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.fill_light_color, axis, value.clamp(0.0, 4.0));
            });
        }
        RenderSettingsAction::SetAmbient => {
            host_state.app.edit_render_config(|config| {
                config.ambient = value.clamp(0.0, 4.0);
            });
        }
        RenderSettingsAction::SetEnvReflectionEnabled => {
            host_state.app.edit_render_config(|config| {
                config.env_reflection_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetEnvReflectionIntensity => {
            host_state.app.edit_render_config(|config| {
                config.env_reflection_intensity = value.clamp(0.0, 4.0);
            });
        }
        RenderSettingsAction::SetSpecularAaEnabled => {
            host_state.app.edit_render_config(|config| {
                config.specular_aa_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetLocalReflectionMode => {
            if let Some(mode) = local_reflection_mode_from_value(value) {
                host_state.app.edit_render_config(|config| {
                    config.local_reflection_mode = mode;
                });
            }
        }
        RenderSettingsAction::SetEnvironmentSource => {
            if let Some(source) = environment_source_from_value(value) {
                host_state.app.set_environment_source(source);
            }
        }
        RenderSettingsAction::PickEnvironmentHdri => {
            host_state.app.pick_environment_hdri();
        }
        RenderSettingsAction::SetEnvironmentRotation => {
            host_state.app.set_environment_rotation_degrees(value);
        }
        RenderSettingsAction::SetEnvironmentExposure => {
            host_state.app.set_environment_exposure(value);
        }
        RenderSettingsAction::SetEnvironmentBakeResolution => {
            host_state
                .app
                .set_environment_bake_resolution(value.max(0.0) as u32);
        }
        RenderSettingsAction::SetEnvironmentBackgroundMode => {
            if let Some(mode) = environment_background_mode_from_value(value) {
                host_state.app.set_environment_background_mode(mode);
            }
        }
        RenderSettingsAction::SetEnvironmentBackgroundBlur => {
            host_state.app.set_environment_background_blur(value);
        }
        RenderSettingsAction::SetSkyHorizon => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.sky_horizon, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetSkyZenith => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.sky_zenith, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetBackgroundMode => {
            if let Some(mode) = background_mode_from_value(value) {
                host_state.app.edit_render_config(|config| {
                    config.background_mode = mode;
                });
            }
        }
        RenderSettingsAction::SetBgSolidColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.bg_solid_color, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetSssEnabled => {
            host_state.app.edit_render_config(|config| {
                config.sss_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetSssStrength => {
            host_state.app.edit_render_config(|config| {
                config.sss_strength = value.clamp(0.0, 2.0);
            });
        }
        RenderSettingsAction::SetSssColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.sss_color, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetVolumetricSteps => {
            host_state.app.edit_render_config(|config| {
                config.volumetric_steps = value.round().clamp(1.0, 256.0) as u32;
            });
        }
        RenderSettingsAction::SetFogEnabled => {
            host_state.app.edit_render_config(|config| {
                config.fog_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetFogDensity => {
            host_state.app.edit_render_config(|config| {
                config.fog_density = value.clamp(0.0, 2.0);
            });
        }
        RenderSettingsAction::SetFogColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.fog_color, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetBloomEnabled => {
            host_state.app.edit_render_config(|config| {
                config.bloom_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetBloomThreshold => {
            host_state.app.edit_render_config(|config| {
                config.bloom_threshold = value.clamp(0.0, 8.0);
            });
        }
        RenderSettingsAction::SetBloomIntensity => {
            host_state.app.edit_render_config(|config| {
                config.bloom_intensity = value.clamp(0.0, 8.0);
            });
        }
        RenderSettingsAction::SetBloomRadius => {
            host_state.app.edit_render_config(|config| {
                config.bloom_radius = value.clamp(0.0, 4.0);
            });
        }
        RenderSettingsAction::SetGamma => {
            host_state.app.edit_render_config(|config| {
                config.gamma = value.clamp(0.5, 4.0);
            });
        }
        RenderSettingsAction::SetTonemappingAces => {
            host_state.app.edit_render_config(|config| {
                config.tonemapping_aces = value >= 0.5;
            });
        }
        RenderSettingsAction::SetOutlineColor => {
            host_state.app.edit_render_config(|config| {
                set_array_component(&mut config.outline_color, axis, value.clamp(0.0, 1.0));
            });
        }
        RenderSettingsAction::SetOutlineThickness => {
            host_state.app.edit_render_config(|config| {
                config.outline_thickness = value.clamp(0.0, 16.0);
            });
        }
        RenderSettingsAction::SetSculptFastMode => {
            host_state.app.edit_render_config(|config| {
                config.sculpt_fast_mode = value >= 0.5;
            });
        }
        RenderSettingsAction::SetAutoReduceSteps => {
            host_state.app.edit_render_config(|config| {
                config.auto_reduce_steps = value >= 0.5;
            });
        }
        RenderSettingsAction::SetInteractionRenderScale => {
            host_state.app.edit_render_config(|config| {
                config.interaction_render_scale = value.clamp(0.25, 1.0);
            });
        }
        RenderSettingsAction::SetRestRenderScale => {
            host_state.app.edit_render_config(|config| {
                config.rest_render_scale = value.clamp(0.25, 1.0);
            });
        }
        RenderSettingsAction::SetCompositeVolumeEnabled => {
            host_state.app.edit_render_config(|config| {
                config.composite_volume_enabled = value >= 0.5;
            });
        }
        RenderSettingsAction::SetCompositeVolumeResolution => {
            host_state.app.edit_render_config(|config| {
                config.composite_volume_resolution = value.round().clamp(64.0, 256.0) as u32;
            });
        }
        RenderSettingsAction::SetDebugForceManualSculptSampling => {
            host_state.app.edit_render_config(|config| {
                config.debug_force_manual_sculpt_sampling = value >= 0.5;
            });
        }
        RenderSettingsAction::SetExportResolution => {
            host_state.app.set_export_resolution(value.max(16.0) as u32);
        }
        RenderSettingsAction::SetAdaptiveExport => {
            host_state.app.set_adaptive_export(value >= 0.5);
        }
    }
}

fn ao_mode_from_value(value: f32) -> Option<AmbientOcclusionMode> {
    match value.round() as i32 {
        0 => Some(AmbientOcclusionMode::Fast),
        1 => Some(AmbientOcclusionMode::Balanced),
        2 => Some(AmbientOcclusionMode::Quality),
        _ => None,
    }
}

fn background_mode_from_value(value: f32) -> Option<BackgroundMode> {
    match value.round() as i32 {
        0 => Some(BackgroundMode::SkyGradient),
        1 => Some(BackgroundMode::SolidColor),
        _ => None,
    }
}

fn environment_background_mode_from_value(value: f32) -> Option<EnvironmentBackgroundMode> {
    match value.round() as i32 {
        0 => Some(EnvironmentBackgroundMode::Environment),
        1 => Some(EnvironmentBackgroundMode::Procedural),
        _ => None,
    }
}

fn environment_source_from_value(value: f32) -> Option<EnvironmentSource> {
    match value.round() as i32 {
        0 => Some(EnvironmentSource::ProceduralSky),
        1 => Some(EnvironmentSource::Hdri),
        _ => None,
    }
}

fn local_reflection_mode_from_value(value: f32) -> Option<LocalReflectionMode> {
    match value.round() as i32 {
        0 => Some(LocalReflectionMode::Off),
        1 => Some(LocalReflectionMode::Single),
        _ => None,
    }
}

fn shading_mode_from_value(value: f32) -> Option<ShadingMode> {
    match value.round() as i32 {
        0 => Some(ShadingMode::Full),
        1 => Some(ShadingMode::Solid),
        2 => Some(ShadingMode::Clay),
        3 => Some(ShadingMode::Normals),
        4 => Some(ShadingMode::Matcap),
        5 => Some(ShadingMode::StepHeatmap),
        6 => Some(ShadingMode::FieldQuality),
        7 => Some(ShadingMode::CrossSection),
        _ => None,
    }
}

fn set_array_component(values: &mut [f32; 3], axis: i32, next: f32) {
    match axis {
        0 => values[0] = next,
        1 => values[1] = next,
        2 => values[2] = next,
        _ => {}
    }
}
