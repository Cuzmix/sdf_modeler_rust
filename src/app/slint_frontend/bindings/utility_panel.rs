use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{HistoryEntry, ReferenceImageRow, ShellSnapshot};
use crate::app::slint_frontend::{
    HistoryRowView, ReferenceRowView, RenderAoModeView, RenderBackgroundModeView,
    RenderEnvironmentBackgroundModeView, RenderEnvironmentSourceView,
    RenderLocalReflectionModeView, RenderSettingsState, RenderShadingModeView, SlintHostWindow,
    UtilityPanelState,
};
use crate::settings::{
    AmbientOcclusionMode, BackgroundMode, EnvironmentBackgroundMode, EnvironmentSource,
    LocalReflectionMode, ShadingMode,
};

use super::property_sections::{bool_field, scalar_field};

pub(super) fn build_utility_panel_snapshot(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> UtilityPanelState {
    UtilityPanelState {
        history_summary: snapshot.utility.history_summary.clone().into(),
        history_rows: Rc::new(VecModel::from(
            snapshot
                .utility
                .history_rows
                .iter()
                .map(history_row_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
        reference_summary: snapshot.utility.reference_summary.clone().into(),
        reference_rows: Rc::new(VecModel::from(
            snapshot
                .utility
                .reference_rows
                .iter()
                .map(reference_row_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
        render_settings: render_settings_state(snapshot),
        import_dialog: window.get_utility_panel_state().import_dialog,
    }
}

fn render_settings_state(snapshot: &ShellSnapshot) -> RenderSettingsState {
    let render = &snapshot.utility.render_settings;
    RenderSettingsState {
        show_grid: bool_field(&render.show_grid),
        show_node_labels: bool_field(&render.show_node_labels),
        show_bounding_box: bool_field(&render.show_bounding_box),
        show_light_gizmos: bool_field(&render.show_light_gizmos),
        shading_mode: shading_mode_view(render.shading_mode),
        cross_section_axis: scalar_field(&render.cross_section_axis),
        cross_section_position: scalar_field(&render.cross_section_position),
        sculpt_safety_border: scalar_field(&render.sculpt_safety_border),
        shadows_enabled: bool_field(&render.shadows_enabled),
        shadow_steps: scalar_field(&render.shadow_steps),
        shadow_penumbra_k: scalar_field(&render.shadow_penumbra_k),
        shadow_bias: scalar_field(&render.shadow_bias),
        shadow_mint: scalar_field(&render.shadow_mint),
        shadow_maxt: scalar_field(&render.shadow_maxt),
        ao_enabled: bool_field(&render.ao_enabled),
        ao_samples: scalar_field(&render.ao_samples),
        ao_step: scalar_field(&render.ao_step),
        ao_decay: scalar_field(&render.ao_decay),
        ao_intensity: scalar_field(&render.ao_intensity),
        ao_mode: ao_mode_view(render.ao_mode),
        march_max_steps: scalar_field(&render.march_max_steps),
        march_epsilon: scalar_field(&render.march_epsilon),
        march_step_multiplier: scalar_field(&render.march_step_multiplier),
        march_max_distance: scalar_field(&render.march_max_distance),
        key_light_dir_x: scalar_field(&render.key_light_dir[0]),
        key_light_dir_y: scalar_field(&render.key_light_dir[1]),
        key_light_dir_z: scalar_field(&render.key_light_dir[2]),
        key_diffuse: scalar_field(&render.key_diffuse),
        key_spec_power: scalar_field(&render.key_spec_power),
        key_spec_intensity: scalar_field(&render.key_spec_intensity),
        key_light_color_r: scalar_field(&render.key_light_color[0]),
        key_light_color_g: scalar_field(&render.key_light_color[1]),
        key_light_color_b: scalar_field(&render.key_light_color[2]),
        fill_light_dir_x: scalar_field(&render.fill_light_dir[0]),
        fill_light_dir_y: scalar_field(&render.fill_light_dir[1]),
        fill_light_dir_z: scalar_field(&render.fill_light_dir[2]),
        fill_intensity: scalar_field(&render.fill_intensity),
        fill_light_color_r: scalar_field(&render.fill_light_color[0]),
        fill_light_color_g: scalar_field(&render.fill_light_color[1]),
        fill_light_color_b: scalar_field(&render.fill_light_color[2]),
        ambient: scalar_field(&render.ambient),
        env_reflection_enabled: bool_field(&render.env_reflection_enabled),
        env_reflection_intensity: scalar_field(&render.env_reflection_intensity),
        specular_aa_enabled: bool_field(&render.specular_aa_enabled),
        local_reflection_mode: local_reflection_mode_view(render.local_reflection_mode),
        environment_source: environment_source_view(render.environment_source),
        hdri_path_display: render.hdri_path_display.clone().into(),
        environment_rotation_degrees: scalar_field(&render.environment_rotation_degrees),
        environment_exposure: scalar_field(&render.environment_exposure),
        environment_bake_resolution: scalar_field(&render.environment_bake_resolution),
        environment_background_mode: environment_background_mode_view(
            render.environment_background_mode,
        ),
        environment_background_blur: scalar_field(&render.environment_background_blur),
        sky_horizon_r: scalar_field(&render.sky_horizon[0]),
        sky_horizon_g: scalar_field(&render.sky_horizon[1]),
        sky_horizon_b: scalar_field(&render.sky_horizon[2]),
        sky_zenith_r: scalar_field(&render.sky_zenith[0]),
        sky_zenith_g: scalar_field(&render.sky_zenith[1]),
        sky_zenith_b: scalar_field(&render.sky_zenith[2]),
        background_mode: background_mode_view(render.background_mode),
        bg_solid_color_r: scalar_field(&render.bg_solid_color[0]),
        bg_solid_color_g: scalar_field(&render.bg_solid_color[1]),
        bg_solid_color_b: scalar_field(&render.bg_solid_color[2]),
        sss_enabled: bool_field(&render.sss_enabled),
        sss_strength: scalar_field(&render.sss_strength),
        sss_color_r: scalar_field(&render.sss_color[0]),
        sss_color_g: scalar_field(&render.sss_color[1]),
        sss_color_b: scalar_field(&render.sss_color[2]),
        volumetric_steps: scalar_field(&render.volumetric_steps),
        fog_enabled: bool_field(&render.fog_enabled),
        fog_density: scalar_field(&render.fog_density),
        fog_color_r: scalar_field(&render.fog_color[0]),
        fog_color_g: scalar_field(&render.fog_color[1]),
        fog_color_b: scalar_field(&render.fog_color[2]),
        bloom_enabled: bool_field(&render.bloom_enabled),
        bloom_threshold: scalar_field(&render.bloom_threshold),
        bloom_intensity: scalar_field(&render.bloom_intensity),
        bloom_radius: scalar_field(&render.bloom_radius),
        gamma: scalar_field(&render.gamma),
        tonemapping_aces: bool_field(&render.tonemapping_aces),
        outline_color_r: scalar_field(&render.outline_color[0]),
        outline_color_g: scalar_field(&render.outline_color[1]),
        outline_color_b: scalar_field(&render.outline_color[2]),
        outline_thickness: scalar_field(&render.outline_thickness),
        sculpt_fast_mode: bool_field(&render.sculpt_fast_mode),
        auto_reduce_steps: bool_field(&render.auto_reduce_steps),
        interaction_render_scale: scalar_field(&render.interaction_render_scale),
        rest_render_scale: scalar_field(&render.rest_render_scale),
        composite_volume_enabled: bool_field(&render.composite_volume_enabled),
        composite_volume_resolution: scalar_field(&render.composite_volume_resolution),
        debug_force_manual_sculpt_sampling: bool_field(&render.debug_force_manual_sculpt_sampling),
        export_resolution: scalar_field(&render.export_resolution),
        adaptive_export: bool_field(&render.adaptive_export),
    }
}

fn ao_mode_view(mode: AmbientOcclusionMode) -> RenderAoModeView {
    match mode {
        AmbientOcclusionMode::Fast => RenderAoModeView::Fast,
        AmbientOcclusionMode::Balanced => RenderAoModeView::Balanced,
        AmbientOcclusionMode::Quality => RenderAoModeView::Quality,
    }
}

fn background_mode_view(mode: BackgroundMode) -> RenderBackgroundModeView {
    match mode {
        BackgroundMode::SkyGradient => RenderBackgroundModeView::SkyGradient,
        BackgroundMode::SolidColor => RenderBackgroundModeView::SolidColor,
    }
}

fn environment_background_mode_view(
    mode: EnvironmentBackgroundMode,
) -> RenderEnvironmentBackgroundModeView {
    match mode {
        EnvironmentBackgroundMode::Environment => RenderEnvironmentBackgroundModeView::Environment,
        EnvironmentBackgroundMode::Procedural => RenderEnvironmentBackgroundModeView::Procedural,
    }
}

fn environment_source_view(source: EnvironmentSource) -> RenderEnvironmentSourceView {
    match source {
        EnvironmentSource::ProceduralSky => RenderEnvironmentSourceView::ProceduralSky,
        EnvironmentSource::Hdri => RenderEnvironmentSourceView::Hdri,
    }
}

fn local_reflection_mode_view(mode: LocalReflectionMode) -> RenderLocalReflectionModeView {
    match mode {
        LocalReflectionMode::Off => RenderLocalReflectionModeView::Off,
        LocalReflectionMode::Single => RenderLocalReflectionModeView::Single,
    }
}

fn shading_mode_view(mode: ShadingMode) -> RenderShadingModeView {
    match mode {
        ShadingMode::Full => RenderShadingModeView::Full,
        ShadingMode::Solid => RenderShadingModeView::Solid,
        ShadingMode::Clay => RenderShadingModeView::Clay,
        ShadingMode::Normals => RenderShadingModeView::Normals,
        ShadingMode::Matcap => RenderShadingModeView::Matcap,
        ShadingMode::StepHeatmap => RenderShadingModeView::StepHeatmap,
        ShadingMode::FieldQuality => RenderShadingModeView::FieldQuality,
        ShadingMode::CrossSection => RenderShadingModeView::CrossSection,
    }
}

fn history_row_view(entry: &HistoryEntry) -> HistoryRowView {
    HistoryRowView {
        label: entry.label.clone().into(),
        direction_label: entry.direction_label.clone().into(),
        is_current: entry.is_current,
        jump_enabled: entry.jump_enabled,
    }
}

fn reference_row_view(entry: &ReferenceImageRow) -> ReferenceRowView {
    ReferenceRowView {
        label: entry.label.clone().into(),
        plane_label: entry.plane_label.clone().into(),
        status_label: entry.status_label.clone().into(),
        visible: entry.visible,
        locked: entry.locked,
        opacity: entry.opacity,
        scale: entry.scale,
    }
}
