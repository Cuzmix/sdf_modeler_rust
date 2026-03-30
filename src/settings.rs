#![allow(dead_code)]

use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::keymap::KeymapConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::native_paths;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default, Debug)]
pub enum BackgroundMode {
    #[default]
    SkyGradient,
    SolidColor,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default, Debug)]
pub enum EnvironmentSource {
    #[default]
    ProceduralSky,
    Hdri,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default, Debug)]
pub enum EnvironmentBackgroundMode {
    #[default]
    Environment,
    Procedural,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default, Debug)]
pub enum AmbientOcclusionMode {
    Fast,
    #[default]
    Balanced,
    Quality,
}

impl AmbientOcclusionMode {
    pub fn gpu_value(&self) -> f32 {
        match self {
            Self::Fast => 0.0,
            Self::Balanced => 1.0,
            Self::Quality => 2.0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::Quality => "Quality",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Default, Debug)]
pub enum LocalReflectionMode {
    Off,
    #[default]
    Single,
}

impl LocalReflectionMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Single => "Single",
        }
    }

    pub fn flag_bit(&self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Single => 1 << 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Shading modes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
pub enum ShadingMode {
    #[default]
    Full,
    Solid,
    Clay,
    Normals,
    Matcap,
    StepHeatmap,
    FieldQuality,
    CrossSection,
}

impl ShadingMode {
    pub fn gpu_value(&self) -> f32 {
        match self {
            Self::Full => 0.0,
            Self::Solid => 1.0,
            Self::Clay => 2.0,
            Self::Normals => 3.0,
            Self::Matcap => 4.0,
            Self::StepHeatmap => 5.0,
            Self::FieldQuality => 6.0,
            Self::CrossSection => 7.0,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Solid => "Solid",
            Self::Clay => "Clay",
            Self::Normals => "Normals",
            Self::Matcap => "Matcap",
            Self::StepHeatmap => "Step Heatmap",
            Self::FieldQuality => "Field Quality",
            Self::CrossSection => "Cross-Section",
        }
    }
    pub fn cycle(&self) -> Self {
        match self {
            Self::Full => Self::Solid,
            Self::Solid => Self::Clay,
            Self::Clay => Self::Normals,
            Self::Normals => Self::Matcap,
            Self::Matcap => Self::StepHeatmap,
            Self::StepHeatmap => Self::FieldQuality,
            Self::FieldQuality => Self::CrossSection,
            Self::CrossSection => Self::Full,
        }
    }
}

// ---------------------------------------------------------------------------
// Snap configuration
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct SnapConfig {
    pub translate_snap: f32,
    pub rotate_snap: f32,
    pub scale_snap: f32,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            translate_snap: 0.25,
            rotate_snap: 15.0,
            scale_snap: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-selection transform behavior
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MultiAxisOrientation {
    #[default]
    WorldZero,
    ActiveObject,
}

impl MultiAxisOrientation {
    pub fn label(&self) -> &'static str {
        match self {
            Self::WorldZero => "World (Zero)",
            Self::ActiveObject => "Active Object",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GroupRotateDirection {
    #[default]
    Standard,
    Inverted,
}

impl GroupRotateDirection {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Inverted => "Inverted",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MultiPivotMode {
    #[default]
    SelectionCenter,
    ActiveObject,
}

impl MultiPivotMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::SelectionCenter => "Selection Center",
            Self::ActiveObject => "Active Object",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(default)]
pub struct SelectionBehaviorSettings {
    pub multi_axis_orientation: MultiAxisOrientation,
    pub group_rotate_direction: GroupRotateDirection,
    pub multi_pivot_mode: MultiPivotMode,
}

impl Default for SelectionBehaviorSettings {
    fn default() -> Self {
        Self {
            multi_axis_orientation: MultiAxisOrientation::WorldZero,
            group_rotate_direction: GroupRotateDirection::Standard,
            multi_pivot_mode: MultiPivotMode::SelectionCenter,
        }
    }
}

// ---------------------------------------------------------------------------
// Shell chrome persistence
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum MenuLauncherPreference {
    File,
    Edit,
    View,
    Settings,
    Help,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum PanelKindPreference {
    Tool,
    ObjectProperties,
    RenderSettings,
    Scene,
    NodeGraph,
    History,
    ReferenceImages,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
#[serde(default)]
pub struct ShellFloatingRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct PinnedPanelPreference {
    pub kind: PanelKindPreference,
    pub collapsed: bool,
    pub rect: Option<ShellFloatingRect>,
}

impl Default for PinnedPanelPreference {
    fn default() -> Self {
        Self {
            kind: PanelKindPreference::Tool,
            collapsed: false,
            rect: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct ShellChromeSettings {
    pub menu_strip_visible: bool,
    pub menu_focused_launcher: Option<MenuLauncherPreference>,
    pub primary_transient_rect: Option<ShellFloatingRect>,
    pub pinned_panels: Vec<PinnedPanelPreference>,
}

impl Default for ShellChromeSettings {
    fn default() -> Self {
        Self {
            menu_strip_visible: true,
            menu_focused_launcher: None,
            primary_transient_rect: None,
            pinned_panels: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Export presets
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct ExportPreset {
    pub name: String,
    pub resolution: u32,
}

fn default_export_presets() -> Vec<ExportPreset> {
    vec![
        ExportPreset {
            name: "Low".into(),
            resolution: 64,
        },
        ExportPreset {
            name: "Medium".into(),
            resolution: 128,
        },
        ExportPreset {
            name: "High".into(),
            resolution: 256,
        },
        ExportPreset {
            name: "Ultra".into(),
            resolution: 512,
        },
    ]
}

// ---------------------------------------------------------------------------
// Camera bookmarks
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct CameraBookmark {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub distance: f32,
    pub target: [f32; 3],
    pub orthographic: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    pub vsync_enabled: bool,
    #[serde(default = "default_true")]
    pub show_fps_overlay: bool,
    #[serde(default)]
    pub continuous_repaint: bool,
    #[serde(default)]
    pub render: RenderConfig,
    #[serde(default = "default_true")]
    pub auto_save_enabled: bool,
    #[serde(default = "default_auto_save_interval")]
    pub auto_save_interval_secs: u32,
    #[serde(default)]
    pub recent_files: Vec<String>,
    #[serde(default = "default_export_resolution")]
    pub export_resolution: u32,
    #[serde(default = "default_max_export_resolution")]
    pub max_export_resolution: u32,
    #[serde(default = "default_max_sculpt_resolution")]
    pub max_sculpt_resolution: u32,
    #[serde(default)]
    pub adaptive_export: bool,
    #[serde(default)]
    pub snap: SnapConfig,
    #[serde(default = "default_bookmarks")]
    pub bookmarks: Vec<Option<CameraBookmark>>,
    #[serde(default = "default_export_presets")]
    pub export_presets: Vec<ExportPreset>,
    #[serde(default)]
    pub keymap: KeymapConfig,
    #[serde(default)]
    pub selection_behavior: SelectionBehaviorSettings,
    #[serde(default)]
    pub auto_switch_sculpt_target_during_brush: bool,
    #[serde(default)]
    pub shell_chrome: ShellChromeSettings,
    #[serde(default = "default_true")]
    pub last_clean_exit: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            vsync_enabled: true,
            show_fps_overlay: true,
            continuous_repaint: false,
            render: RenderConfig::default(),
            auto_save_enabled: true,
            auto_save_interval_secs: 120,
            recent_files: Vec::new(),
            export_resolution: 128,
            max_export_resolution: 2048,
            max_sculpt_resolution: 320,
            adaptive_export: false,
            snap: SnapConfig::default(),
            bookmarks: default_bookmarks(),
            export_presets: default_export_presets(),
            keymap: KeymapConfig::default(),
            selection_behavior: SelectionBehaviorSettings::default(),
            auto_switch_sculpt_target_during_brush: false,
            shell_chrome: ShellChromeSettings::default(),
            last_clean_exit: true,
        }
    }
}

impl Settings {
    pub fn add_recent_file(&mut self, path: &str) {
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_string());
        self.recent_files.truncate(10);
        self.save();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Settings {
    fn path() -> std::path::PathBuf {
        native_paths::app_storage_file("settings.json")
    }

    pub fn load() -> Self {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) {
        let path = Self::path();
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = native_paths::ensure_parent_dir(&path) {
                log::error!("Failed to create settings directory: {}", e);
                return;
            }
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save settings: {}", e);
            }
        }
    }

    pub fn export_dialog(&self) {
        let crate::desktop_dialogs::FileDialogSelection::Selected(path) =
            crate::desktop_dialogs::settings_export_dialog("settings.json")
        else {
            return;
        };

        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = native_paths::ensure_parent_dir(&path) {
                log::error!("Failed to create export directory: {}", e);
                return;
            }
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to export settings: {}", e);
            }
        }
    }

    pub fn import_dialog(&mut self) -> bool {
        let crate::desktop_dialogs::FileDialogSelection::Selected(path) =
            crate::desktop_dialogs::settings_import_dialog()
        else {
            return false;
        };

        match std::fs::read_to_string(&path) {
            Ok(json) => match serde_json::from_str::<Settings>(&json) {
                Ok(imported) => {
                    let recent = std::mem::take(&mut self.recent_files);
                    *self = imported;
                    self.recent_files = recent;
                    self.save();
                    return true;
                }
                Err(e) => log::error!("Failed to parse settings file: {}", e),
            },
            Err(e) => log::error!("Failed to read settings file: {}", e),
        }
        false
    }
}

#[cfg(target_arch = "wasm32")]
impl Settings {
    const STORAGE_KEY: &'static str = "sdf_modeler_settings";

    pub fn load() -> Self {
        let storage = web_sys::window().and_then(|w| w.local_storage().ok().flatten());
        let Some(storage) = storage else {
            return Self::default();
        };
        match storage.get_item(Self::STORAGE_KEY) {
            Ok(Some(json)) => serde_json::from_str(&json).unwrap_or_default(),
            _ => Self::default(),
        }
    }

    pub fn save(&self) {
        let storage = web_sys::window().and_then(|w| w.local_storage().ok().flatten());
        let Some(storage) = storage else { return };
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = storage.set_item(Self::STORAGE_KEY, &json) {
                log::error!("Failed to save settings: {:?}", e);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct RenderConfig {
    // Shadows
    pub shadows_enabled: bool,
    pub shadow_steps: i32,
    pub shadow_penumbra_k: f32,
    pub shadow_bias: f32,
    pub shadow_mint: f32,
    pub shadow_maxt: f32,

    // Ambient Occlusion
    pub ao_enabled: bool,
    pub ao_samples: i32,
    pub ao_step: f32,
    pub ao_decay: f32,
    pub ao_intensity: f32,
    #[serde(default)]
    pub ao_mode: AmbientOcclusionMode,

    // Raymarching
    pub march_max_steps: i32,
    pub march_epsilon: f32,
    pub march_step_multiplier: f32,
    pub march_max_distance: f32,

    // Lighting
    pub key_light_dir: [f32; 3],
    pub key_diffuse: f32,
    pub key_spec_power: f32,
    pub key_spec_intensity: f32,
    #[serde(default = "default_white")]
    pub key_light_color: [f32; 3],
    pub fill_light_dir: [f32; 3],
    pub fill_intensity: f32,
    #[serde(default = "default_white")]
    pub fill_light_color: [f32; 3],
    pub ambient: f32,

    // Environment Reflection
    #[serde(default)]
    pub env_reflection_enabled: bool,
    #[serde(default = "default_env_reflection_intensity")]
    pub env_reflection_intensity: f32,
    #[serde(default = "default_true")]
    pub specular_aa_enabled: bool,
    #[serde(default)]
    pub local_reflection_mode: LocalReflectionMode,

    // Environment Source
    #[serde(default)]
    pub environment_source: EnvironmentSource,
    #[serde(default)]
    pub hdri_path: Option<String>,
    #[serde(default)]
    pub environment_rotation_degrees: f32,
    #[serde(default)]
    pub environment_exposure: f32,
    /// Lighting cubemap bake resolution. 0 = auto from imported HDR/EXR face-equivalent size.
    #[serde(default)]
    pub environment_bake_resolution: u32,

    // Visible Background
    #[serde(default)]
    pub environment_background_mode: EnvironmentBackgroundMode,
    #[serde(default)]
    pub environment_background_blur: f32,

    // Procedural Sky / Background
    pub sky_horizon: [f32; 3],
    pub sky_zenith: [f32; 3],
    #[serde(default)]
    pub background_mode: BackgroundMode,
    #[serde(default = "default_bg_solid_color")]
    pub bg_solid_color: [f32; 3],

    // Subsurface Scattering
    #[serde(default)]
    pub sss_enabled: bool,
    #[serde(default = "default_sss_strength")]
    pub sss_strength: f32,
    #[serde(default = "default_sss_color")]
    pub sss_color: [f32; 3],

    // Volumetric scattering
    #[serde(default = "default_volumetric_steps")]
    pub volumetric_steps: u32,

    // Fog
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_color: [f32; 3],

    // Bloom
    #[serde(default)]
    pub bloom_enabled: bool,
    #[serde(default = "default_bloom_threshold")]
    pub bloom_threshold: f32,
    #[serde(default = "default_bloom_intensity")]
    pub bloom_intensity: f32,
    #[serde(default = "default_bloom_radius")]
    pub bloom_radius: f32,

    // Gamma / Tonemapping
    pub gamma: f32,
    #[serde(default)]
    pub tonemapping_aces: bool,

    // Selection Outline
    #[serde(default = "default_outline_color")]
    pub outline_color: [f32; 3],
    #[serde(default = "default_outline_thickness")]
    pub outline_thickness: f32,

    // Viewport
    #[serde(default = "default_true")]
    pub show_grid: bool,

    // Performance
    /// Use fast quality mode during sculpt brush strokes (half steps, skip AO/shadows).
    #[serde(default)]
    pub sculpt_fast_mode: bool,
    /// Automatically reduce march steps when multiple sculpt nodes exist.
    #[serde(default = "default_true")]
    pub auto_reduce_steps: bool,
    /// Render resolution scale during interaction (0.25 to 1.0).
    #[serde(default = "default_interaction_scale")]
    pub interaction_render_scale: f32,
    /// Render resolution scale at rest (0.5 to 1.0).
    #[serde(default = "default_rest_scale")]
    pub rest_render_scale: f32,
    /// Enable GPU compute composite volume cache.
    #[serde(default)]
    pub composite_volume_enabled: bool,
    /// Resolution of the composite scene volume (64-256).
    #[serde(default = "default_composite_resolution")]
    pub composite_volume_resolution: u32,
    /// Force render/composite shaders to use the manual storage-buffer voxel path.
    /// Useful for diagnosing texture-vs-storage sculpt sampling mismatches.
    #[serde(default)]
    pub debug_force_manual_sculpt_sampling: bool,

    // Touch input
    #[serde(default = "default_touch_zoom_sensitivity")]
    pub touch_zoom_sensitivity: f32,
    #[serde(default)]
    pub invert_touch_pan: bool,

    // Roll
    #[serde(default = "default_roll_sensitivity")]
    pub roll_sensitivity: f32,
    #[serde(default)]
    pub invert_roll: bool,

    // Tablet / Pressure
    #[serde(default)]
    pub pressure_sensitivity: bool,

    // Navigation
    #[serde(default)]
    pub clamp_orbit_pitch: bool,

    // Shading mode
    #[serde(default)]
    pub shading_mode: ShadingMode,

    // Cross-section visualization
    /// Slice axis: 0=X, 1=Y, 2=Z
    #[serde(default = "default_cross_section_axis")]
    pub cross_section_axis: u8,
    /// Slice plane position along the selected axis
    #[serde(default)]
    pub cross_section_position: f32,

    // Viewport overlays
    #[serde(default)]
    pub show_node_labels: bool,
    #[serde(default = "default_true")]
    pub show_bounding_box: bool,
    #[serde(default = "default_true")]
    pub show_light_gizmos: bool,

    // Sculpt safety border (fraction of viewport, 0.0 = disabled, default 5%)
    #[serde(default = "default_safety_border")]
    pub sculpt_safety_border: f32,
}

fn default_true() -> bool {
    true
}
fn default_auto_save_interval() -> u32 {
    120
}
fn default_export_resolution() -> u32 {
    128
}
fn default_max_export_resolution() -> u32 {
    2048
}
fn default_max_sculpt_resolution() -> u32 {
    320
}
fn default_bg_solid_color() -> [f32; 3] {
    [0.12, 0.12, 0.15]
}
fn default_interaction_scale() -> f32 {
    0.5
}
fn default_rest_scale() -> f32 {
    1.0
}
fn default_composite_resolution() -> u32 {
    128
}
fn default_outline_color() -> [f32; 3] {
    [1.0, 0.8, 0.2]
}
fn default_outline_thickness() -> f32 {
    2.5
}
fn default_touch_zoom_sensitivity() -> f32 {
    500.0
}
fn default_roll_sensitivity() -> f32 {
    0.005
}
fn default_white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_sss_strength() -> f32 {
    5.0
}
fn default_sss_color() -> [f32; 3] {
    [1.0, 0.4, 0.2]
}
fn default_env_reflection_intensity() -> f32 {
    0.3
}
fn default_bloom_threshold() -> f32 {
    0.8
}
fn default_bloom_intensity() -> f32 {
    0.3
}
fn default_bloom_radius() -> f32 {
    3.0
}
fn default_volumetric_steps() -> u32 {
    24
}
fn default_safety_border() -> f32 {
    0.05
}
fn default_cross_section_axis() -> u8 {
    1
}
fn default_bookmarks() -> Vec<Option<CameraBookmark>> {
    vec![None; 9]
}
fn default_env_reflection_enabled_config() -> bool {
    !cfg!(any(target_arch = "wasm32", target_os = "android"))
}
fn default_environment_source_config() -> EnvironmentSource {
    if cfg!(any(target_arch = "wasm32", target_os = "android")) {
        EnvironmentSource::ProceduralSky
    } else {
        EnvironmentSource::Hdri
    }
}
fn default_hdri_path_config() -> Option<String> {
    if cfg!(any(target_arch = "wasm32", target_os = "android")) {
        None
    } else {
        Some(crate::native_paths::DEFAULT_BUNDLED_ENVIRONMENT_HDRI_PATH.into())
    }
}
fn default_environment_background_mode_config() -> EnvironmentBackgroundMode {
    if cfg!(any(target_arch = "wasm32", target_os = "android")) {
        EnvironmentBackgroundMode::Environment
    } else {
        EnvironmentBackgroundMode::Procedural
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            shadows_enabled: false,
            shadow_steps: 32,
            shadow_penumbra_k: 8.0,
            shadow_bias: 0.04,
            shadow_mint: 0.06,
            shadow_maxt: 20.0,

            ao_enabled: true,
            ao_samples: 5,
            ao_step: 0.4,
            ao_decay: 0.95,
            ao_intensity: 1.0,
            ao_mode: AmbientOcclusionMode::Balanced,

            march_max_steps: 128,
            march_epsilon: 0.002,
            march_step_multiplier: 0.9,
            march_max_distance: 50.0,

            key_light_dir: [1.0, 2.0, 3.0],
            key_diffuse: 0.85,
            key_spec_power: 32.0,
            key_spec_intensity: 0.4,
            key_light_color: [1.0, 1.0, 1.0],
            fill_light_dir: [-1.0, 0.5, -1.0],
            fill_intensity: 0.25,
            fill_light_color: [1.0, 1.0, 1.0],
            ambient: 0.06,

            env_reflection_enabled: default_env_reflection_enabled_config(),
            env_reflection_intensity: 0.3,
            specular_aa_enabled: true,
            local_reflection_mode: LocalReflectionMode::Single,
            environment_source: default_environment_source_config(),
            hdri_path: default_hdri_path_config(),
            environment_rotation_degrees: 0.0,
            environment_exposure: 0.0,
            environment_bake_resolution: 0,
            environment_background_mode: default_environment_background_mode_config(),
            environment_background_blur: 0.0,

            sky_horizon: [0.10, 0.10, 0.16],
            sky_zenith: [0.02, 0.02, 0.05],
            background_mode: BackgroundMode::SkyGradient,
            bg_solid_color: [0.12, 0.12, 0.15],

            sss_enabled: false,
            sss_strength: 5.0,
            sss_color: [1.0, 0.4, 0.2],

            volumetric_steps: 24,

            fog_enabled: false,
            fog_density: 0.04,
            fog_color: [0.5, 0.55, 0.65],

            bloom_enabled: false,
            bloom_threshold: 0.8,
            bloom_intensity: 0.3,
            bloom_radius: 3.0,

            gamma: 2.2,
            tonemapping_aces: false,

            outline_color: [1.0, 0.8, 0.2],
            outline_thickness: 2.5,

            show_grid: true,

            sculpt_fast_mode: false,
            auto_reduce_steps: true,
            interaction_render_scale: 0.5,
            rest_render_scale: 1.0,
            composite_volume_enabled: false,
            composite_volume_resolution: 128,
            debug_force_manual_sculpt_sampling: false,

            touch_zoom_sensitivity: 500.0,
            invert_touch_pan: false,
            roll_sensitivity: 0.005,
            invert_roll: false,
            pressure_sensitivity: false,
            clamp_orbit_pitch: false,
            shading_mode: ShadingMode::default(),
            cross_section_axis: 1,
            cross_section_position: 0.0,
            show_node_labels: false,
            show_bounding_box: true,
            show_light_gizmos: true,
            sculpt_safety_border: 0.05,
        }
    }
}

impl RenderConfig {
    pub fn environment_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match self.environment_source {
            EnvironmentSource::ProceduralSky => 0_u8.hash(&mut hasher),
            EnvironmentSource::Hdri => 1_u8.hash(&mut hasher),
        }
        self.environment_rotation_degrees
            .to_bits()
            .hash(&mut hasher);
        self.environment_exposure.to_bits().hash(&mut hasher);
        self.environment_bake_resolution.hash(&mut hasher);

        match self.environment_source {
            EnvironmentSource::ProceduralSky => {
                match self.background_mode {
                    BackgroundMode::SkyGradient => 0_u8.hash(&mut hasher),
                    BackgroundMode::SolidColor => 1_u8.hash(&mut hasher),
                }
                for value in self
                    .sky_horizon
                    .iter()
                    .chain(self.sky_zenith.iter())
                    .chain(self.bg_solid_color.iter())
                {
                    value.to_bits().hash(&mut hasher);
                }
            }
            EnvironmentSource::Hdri => {
                self.hdri_path
                    .as_deref()
                    .unwrap_or_default()
                    .hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub fn environment_specular_intensity(&self) -> f32 {
        if self.env_reflection_enabled {
            self.env_reflection_intensity
        } else {
            0.0
        }
    }

    pub fn reset_shadows(&mut self) {
        let d = Self::default();
        self.shadows_enabled = d.shadows_enabled;
        self.shadow_steps = d.shadow_steps;
        self.shadow_penumbra_k = d.shadow_penumbra_k;
        self.shadow_bias = d.shadow_bias;
        self.shadow_mint = d.shadow_mint;
        self.shadow_maxt = d.shadow_maxt;
    }

    pub fn reset_ao(&mut self) {
        let d = Self::default();
        self.ao_enabled = d.ao_enabled;
        self.ao_samples = d.ao_samples;
        self.ao_step = d.ao_step;
        self.ao_decay = d.ao_decay;
        self.ao_intensity = d.ao_intensity;
        self.ao_mode = d.ao_mode;
    }

    pub fn reset_raymarching(&mut self) {
        let d = Self::default();
        self.march_max_steps = d.march_max_steps;
        self.march_epsilon = d.march_epsilon;
        self.march_step_multiplier = d.march_step_multiplier;
        self.march_max_distance = d.march_max_distance;
    }

    pub fn reset_lighting(&mut self) {
        let d = Self::default();
        self.ambient = d.ambient;
    }

    pub fn reset_environment(&mut self) {
        let d = Self::default();
        self.env_reflection_enabled = d.env_reflection_enabled;
        self.env_reflection_intensity = d.env_reflection_intensity;
        self.specular_aa_enabled = d.specular_aa_enabled;
        self.local_reflection_mode = d.local_reflection_mode;
        self.environment_source = d.environment_source;
        self.hdri_path = d.hdri_path;
        self.environment_rotation_degrees = d.environment_rotation_degrees;
        self.environment_exposure = d.environment_exposure;
        self.environment_bake_resolution = d.environment_bake_resolution;
        self.environment_background_mode = d.environment_background_mode;
        self.environment_background_blur = d.environment_background_blur;
        self.sky_horizon = d.sky_horizon;
        self.sky_zenith = d.sky_zenith;
        self.background_mode = d.background_mode;
        self.bg_solid_color = d.bg_solid_color;
    }

    pub fn reset_sss(&mut self) {
        let d = Self::default();
        self.sss_enabled = d.sss_enabled;
        self.sss_strength = d.sss_strength;
        self.sss_color = d.sss_color;
    }

    pub fn reset_fog(&mut self) {
        let d = Self::default();
        self.fog_enabled = d.fog_enabled;
        self.fog_density = d.fog_density;
        self.fog_color = d.fog_color;
    }

    pub fn reset_gamma(&mut self) {
        let d = Self::default();
        self.gamma = d.gamma;
        self.tonemapping_aces = d.tonemapping_aces;
    }

    pub fn reset_outline(&mut self) {
        let d = Self::default();
        self.outline_color = d.outline_color;
        self.outline_thickness = d.outline_thickness;
    }

    pub fn reset_all(&mut self) {
        *self = Self::default();
    }

    /// Returns true if the change between self and other requires a shader
    /// recompilation (i.e., a placeholder-affecting field changed). Lighting
    /// intensities and environment content now update through uniforms/resources.
    pub fn needs_shader_rebuild(&self, other: &Self) -> bool {
        // Compare all fields EXCEPT lighting (those are in uniform now)
        self.shadows_enabled != other.shadows_enabled
            || self.shadow_steps != other.shadow_steps
            || self.shadow_penumbra_k != other.shadow_penumbra_k
            || self.shadow_bias != other.shadow_bias
            || self.shadow_mint != other.shadow_mint
            || self.shadow_maxt != other.shadow_maxt
            || self.ao_enabled != other.ao_enabled
            || self.ao_samples != other.ao_samples
            || self.ao_step != other.ao_step
            || self.ao_decay != other.ao_decay
            || self.ao_intensity != other.ao_intensity
            || self.march_max_steps != other.march_max_steps
            || self.march_epsilon != other.march_epsilon
            || self.march_step_multiplier != other.march_step_multiplier
            || self.march_max_distance != other.march_max_distance
            || self.sss_enabled != other.sss_enabled
            || self.sss_strength != other.sss_strength
            || self.sss_color != other.sss_color
            || self.fog_enabled != other.fog_enabled
            || self.fog_density != other.fog_density
            || self.fog_color != other.fog_color
            || self.gamma != other.gamma
            || self.tonemapping_aces != other.tonemapping_aces
            || self.outline_color != other.outline_color
            || self.outline_thickness != other.outline_thickness
            || self.debug_force_manual_sculpt_sampling != other.debug_force_manual_sculpt_sampling
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AmbientOcclusionMode, BackgroundMode, EnvironmentBackgroundMode, EnvironmentSource,
        GroupRotateDirection, LocalReflectionMode, MenuLauncherPreference, MultiAxisOrientation,
        MultiPivotMode, PanelKindPreference, PinnedPanelPreference, RenderConfig,
        SelectionBehaviorSettings, Settings, ShadingMode, ShellChromeSettings, ShellFloatingRect,
    };

    #[test]
    fn shading_mode_cycle_includes_field_quality() {
        assert_eq!(ShadingMode::StepHeatmap.cycle(), ShadingMode::FieldQuality);
        assert_eq!(ShadingMode::FieldQuality.cycle(), ShadingMode::CrossSection);
    }

    #[test]
    fn field_quality_mode_has_expected_metadata() {
        assert_eq!(ShadingMode::FieldQuality.gpu_value(), 6.0);
        assert_eq!(ShadingMode::FieldQuality.label(), "Field Quality");
    }

    #[test]
    fn manual_sculpt_sampling_requires_shader_rebuild() {
        let base = RenderConfig::default();
        let mut changed = base.clone();
        changed.debug_force_manual_sculpt_sampling = true;
        assert!(changed.needs_shader_rebuild(&base));
    }

    #[test]
    fn environment_settings_do_not_require_shader_rebuild() {
        let base = RenderConfig::default();
        let mut changed = base.clone();
        changed.env_reflection_enabled = !changed.env_reflection_enabled;
        changed.env_reflection_intensity = 0.9;
        changed.specular_aa_enabled = !changed.specular_aa_enabled;
        changed.local_reflection_mode = LocalReflectionMode::Off;
        changed.environment_source = EnvironmentSource::Hdri;
        changed.hdri_path = Some("studio.hdr".into());
        changed.environment_rotation_degrees = 45.0;
        changed.environment_exposure = 1.0;
        changed.environment_bake_resolution = 512;
        changed.environment_background_mode = EnvironmentBackgroundMode::Procedural;
        changed.environment_background_blur = 0.35;
        changed.background_mode = BackgroundMode::SolidColor;
        changed.sky_horizon = [0.3, 0.2, 0.1];
        changed.sky_zenith = [0.1, 0.2, 0.3];
        changed.bg_solid_color = [0.5, 0.4, 0.3];
        assert!(!changed.needs_shader_rebuild(&base));
    }

    #[test]
    fn ao_mode_toggle_does_not_require_shader_rebuild() {
        let base = RenderConfig::default();
        let mut changed = base.clone();
        changed.ao_mode = AmbientOcclusionMode::Quality;
        assert!(!changed.needs_shader_rebuild(&base));
    }

    #[test]
    fn local_reflection_mode_toggle_does_not_require_shader_rebuild() {
        let base = RenderConfig::default();
        let mut changed = base.clone();
        changed.local_reflection_mode = LocalReflectionMode::Off;
        assert!(!changed.needs_shader_rebuild(&base));
    }

    #[test]
    fn environment_fingerprint_tracks_background_bake_inputs() {
        let mut base = RenderConfig::default();
        base.environment_source = EnvironmentSource::ProceduralSky;
        base.hdri_path = None;
        let mut changed = base.clone();
        changed.sky_zenith = [0.3, 0.4, 0.5];
        assert_ne!(
            changed.environment_fingerprint(),
            base.environment_fingerprint()
        );
    }

    #[test]
    fn environment_fingerprint_tracks_hdri_inputs() {
        let base = RenderConfig::default();
        let mut changed = base.clone();
        changed.environment_source = EnvironmentSource::Hdri;
        changed.hdri_path = Some("studio.hdr".into());
        changed.environment_rotation_degrees = 30.0;
        changed.environment_exposure = 1.5;
        changed.environment_bake_resolution = 512;
        assert_ne!(
            changed.environment_fingerprint(),
            base.environment_fingerprint()
        );
    }

    #[test]
    fn render_defaults_use_platform_expected_environment_defaults() {
        let defaults = RenderConfig::default();
        if cfg!(any(target_arch = "wasm32", target_os = "android")) {
            assert!(!defaults.env_reflection_enabled);
            assert_eq!(
                defaults.environment_source,
                EnvironmentSource::ProceduralSky
            );
            assert_eq!(defaults.hdri_path, None);
            assert_eq!(
                defaults.environment_background_mode,
                EnvironmentBackgroundMode::Environment
            );
        } else {
            assert!(defaults.env_reflection_enabled);
            assert_eq!(defaults.environment_source, EnvironmentSource::Hdri);
            assert_eq!(
                defaults.hdri_path.as_deref(),
                Some(crate::native_paths::DEFAULT_BUNDLED_ENVIRONMENT_HDRI_PATH)
            );
            assert_eq!(
                defaults.environment_background_mode,
                EnvironmentBackgroundMode::Procedural
            );
        }
    }

    #[test]
    fn reset_environment_restores_platform_expected_environment_defaults() {
        let mut config = RenderConfig::default();
        config.env_reflection_enabled = !config.env_reflection_enabled;
        config.env_reflection_intensity = 0.9;
        config.environment_source = EnvironmentSource::ProceduralSky;
        config.hdri_path = Some("custom.hdr".into());
        config.environment_background_mode = EnvironmentBackgroundMode::Environment;
        config.environment_background_blur = 0.5;
        config.environment_rotation_degrees = 22.0;
        config.environment_exposure = 1.0;

        config.reset_environment();

        let defaults = RenderConfig::default();
        assert_eq!(
            config.env_reflection_enabled,
            defaults.env_reflection_enabled
        );
        assert_eq!(
            config.env_reflection_intensity,
            defaults.env_reflection_intensity
        );
        assert_eq!(config.environment_source, defaults.environment_source);
        assert_eq!(config.hdri_path, defaults.hdri_path);
        assert_eq!(
            config.environment_background_mode,
            defaults.environment_background_mode
        );
        assert_eq!(
            config.environment_rotation_degrees,
            defaults.environment_rotation_degrees
        );
        assert_eq!(config.environment_exposure, defaults.environment_exposure);
        assert_eq!(
            config.environment_background_blur,
            defaults.environment_background_blur
        );
    }

    #[test]
    fn reset_all_restores_platform_expected_environment_defaults() {
        let mut config = RenderConfig::default();
        config.env_reflection_enabled = !config.env_reflection_enabled;
        config.env_reflection_intensity = 1.2;
        config.environment_source = EnvironmentSource::ProceduralSky;
        config.hdri_path = Some("custom.hdr".into());
        config.environment_background_mode = EnvironmentBackgroundMode::Environment;
        config.environment_background_blur = 0.65;

        config.reset_all();

        let defaults = RenderConfig::default();
        assert_eq!(
            config.env_reflection_enabled,
            defaults.env_reflection_enabled
        );
        assert_eq!(
            config.env_reflection_intensity,
            defaults.env_reflection_intensity
        );
        assert_eq!(config.environment_source, defaults.environment_source);
        assert_eq!(config.hdri_path, defaults.hdri_path);
        assert_eq!(
            config.environment_background_mode,
            defaults.environment_background_mode
        );
        assert_eq!(
            config.environment_background_blur,
            defaults.environment_background_blur
        );
    }

    #[test]
    fn selection_behavior_defaults_are_expected() {
        let defaults = SelectionBehaviorSettings::default();
        assert_eq!(
            defaults.multi_axis_orientation,
            MultiAxisOrientation::WorldZero
        );
        assert_eq!(
            defaults.group_rotate_direction,
            GroupRotateDirection::Standard
        );
        assert_eq!(defaults.multi_pivot_mode, MultiPivotMode::SelectionCenter);
    }

    #[test]
    fn settings_legacy_deserialize_defaults_selection_behavior() {
        let mut legacy = serde_json::to_value(Settings::default()).expect("serialize settings");
        legacy
            .as_object_mut()
            .expect("settings object")
            .remove("selection_behavior");

        let parsed: Settings = serde_json::from_value(legacy).expect("deserialize settings");
        assert_eq!(
            parsed.selection_behavior,
            SelectionBehaviorSettings::default()
        );
    }

    #[test]
    fn settings_legacy_deserialize_defaults_auto_switch_sculpt_target_during_brush() {
        let mut legacy = serde_json::to_value(Settings::default()).expect("serialize settings");
        legacy
            .as_object_mut()
            .expect("settings object")
            .remove("auto_switch_sculpt_target_during_brush");

        let parsed: Settings = serde_json::from_value(legacy).expect("deserialize settings");
        assert!(!parsed.auto_switch_sculpt_target_during_brush);
    }

    #[test]
    fn settings_legacy_deserialize_defaults_shell_chrome() {
        let mut legacy = serde_json::to_value(Settings::default()).expect("serialize settings");
        legacy
            .as_object_mut()
            .expect("settings object")
            .remove("shell_chrome");

        let parsed: Settings = serde_json::from_value(legacy).expect("deserialize settings");
        assert_eq!(parsed.shell_chrome, ShellChromeSettings::default());
    }

    #[test]
    fn shell_chrome_serialization_roundtrip() {
        let mut settings = Settings::default();
        settings.shell_chrome = ShellChromeSettings {
            menu_strip_visible: false,
            menu_focused_launcher: Some(MenuLauncherPreference::Help),
            primary_transient_rect: Some(ShellFloatingRect {
                x: 12.0,
                y: 34.0,
                width: 320.0,
                height: 260.0,
            }),
            pinned_panels: vec![PinnedPanelPreference {
                kind: PanelKindPreference::RenderSettings,
                collapsed: true,
                rect: Some(ShellFloatingRect {
                    x: 56.0,
                    y: 78.0,
                    width: 360.0,
                    height: 300.0,
                }),
            },
            PinnedPanelPreference {
                kind: PanelKindPreference::NodeGraph,
                collapsed: false,
                rect: None,
            }],
        };

        let json = serde_json::to_string(&settings).expect("serialize settings");
        let parsed: Settings = serde_json::from_str(&json).expect("deserialize settings");

        assert_eq!(parsed.shell_chrome, settings.shell_chrome);
    }
}
