use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum BackgroundMode {
    SkyGradient,
    SolidColor,
}

impl Default for BackgroundMode {
    fn default() -> Self {
        Self::SkyGradient
    }
}

// ---------------------------------------------------------------------------
// Shading modes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum ShadingMode {
    Full,
    Solid,
    Clay,
    Normals,
    Matcap,
    StepHeatmap,
}

impl Default for ShadingMode {
    fn default() -> Self { Self::Full }
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
        }
    }
    pub fn cycle(&self) -> Self {
        match self {
            Self::Full => Self::Solid,
            Self::Solid => Self::Clay,
            Self::Clay => Self::Normals,
            Self::Normals => Self::Matcap,
            Self::Matcap => Self::StepHeatmap,
            Self::StepHeatmap => Self::Full,
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
// Export presets
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct ExportPreset {
    pub name: String,
    pub resolution: u32,
}

fn default_export_presets() -> Vec<ExportPreset> {
    vec![
        ExportPreset { name: "Draft".into(), resolution: 64 },
        ExportPreset { name: "Medium".into(), resolution: 128 },
        ExportPreset { name: "High".into(), resolution: 256 },
        ExportPreset { name: "Ultra".into(), resolution: 512 },
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
    #[serde(default)]
    pub adaptive_export: bool,
    #[serde(default)]
    pub snap: SnapConfig,
    #[serde(default = "default_bookmarks")]
    pub bookmarks: Vec<Option<CameraBookmark>>,
    #[serde(default = "default_export_presets")]
    pub export_presets: Vec<ExportPreset>,
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
            adaptive_export: false,
            snap: SnapConfig::default(),
            bookmarks: default_bookmarks(),
            export_presets: default_export_presets(),
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
        let mut path = std::env::current_exe().unwrap_or_default();
        path.pop();
        path.push("settings.json");
        path
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
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save settings: {}", e);
            }
        }
    }

    pub fn export_dialog(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Export Settings")
            .add_filter("JSON", &["json"])
            .set_file_name("settings.json")
            .save_file()
        {
            if let Ok(json) = serde_json::to_string_pretty(self) {
                if let Err(e) = std::fs::write(&path, json) {
                    log::error!("Failed to export settings: {}", e);
                }
            }
        }
    }

    pub fn import_dialog(&mut self) -> bool {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Import Settings")
            .add_filter("JSON", &["json"])
            .pick_file()
        {
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
        }
        false
    }
}

#[cfg(target_arch = "wasm32")]
impl Settings {
    const STORAGE_KEY: &'static str = "sdf_modeler_settings";

    pub fn load() -> Self {
        let storage = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten());
        let Some(storage) = storage else { return Self::default() };
        match storage.get_item(Self::STORAGE_KEY) {
            Ok(Some(json)) => serde_json::from_str(&json).unwrap_or_default(),
            _ => Self::default(),
        }
    }

    pub fn save(&self) {
        let storage = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten());
        let Some(storage) = storage else { return };
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = storage.set_item(Self::STORAGE_KEY, &json) {
                log::error!("Failed to save settings: {:?}", e);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
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
    pub fill_light_dir: [f32; 3],
    pub fill_intensity: f32,
    pub ambient: f32,

    // Environment Reflection
    #[serde(default)]
    pub env_reflection_enabled: bool,
    #[serde(default = "default_env_reflection_intensity")]
    pub env_reflection_intensity: f32,

    // Sky / Background
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

    // Viewport overlays
    #[serde(default)]
    pub show_node_labels: bool,
    #[serde(default = "default_true")]
    pub show_bounding_box: bool,

    // Sculpt safety border (fraction of viewport, 0.0 = disabled, default 5%)
    #[serde(default = "default_safety_border")]
    pub sculpt_safety_border: f32,
}

fn default_true() -> bool { true }
fn default_auto_save_interval() -> u32 { 120 }
fn default_export_resolution() -> u32 { 128 }
fn default_bg_solid_color() -> [f32; 3] { [0.12, 0.12, 0.15] }
fn default_interaction_scale() -> f32 { 0.5 }
fn default_rest_scale() -> f32 { 1.0 }
fn default_composite_resolution() -> u32 { 128 }
fn default_outline_color() -> [f32; 3] { [1.0, 0.8, 0.2] }
fn default_outline_thickness() -> f32 { 2.5 }
fn default_touch_zoom_sensitivity() -> f32 { 500.0 }
fn default_roll_sensitivity() -> f32 { 0.005 }
fn default_sss_strength() -> f32 { 5.0 }
fn default_sss_color() -> [f32; 3] { [1.0, 0.4, 0.2] }
fn default_env_reflection_intensity() -> f32 { 0.3 }
fn default_bloom_threshold() -> f32 { 0.8 }
fn default_bloom_intensity() -> f32 { 0.3 }
fn default_bloom_radius() -> f32 { 3.0 }
fn default_safety_border() -> f32 { 0.05 }
fn default_bookmarks() -> Vec<Option<CameraBookmark>> { vec![None; 9] }

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
            ao_step: 0.08,
            ao_decay: 0.95,
            ao_intensity: 3.0,

            march_max_steps: 128,
            march_epsilon: 0.002,
            march_step_multiplier: 0.9,
            march_max_distance: 50.0,

            key_light_dir: [1.0, 2.0, 3.0],
            key_diffuse: 0.85,
            key_spec_power: 32.0,
            key_spec_intensity: 0.4,
            fill_light_dir: [-1.0, 0.5, -1.0],
            fill_intensity: 0.25,
            ambient: 0.06,

            env_reflection_enabled: false,
            env_reflection_intensity: 0.3,

            sky_horizon: [0.10, 0.10, 0.16],
            sky_zenith: [0.02, 0.02, 0.05],
            background_mode: BackgroundMode::SkyGradient,
            bg_solid_color: [0.12, 0.12, 0.15],

            sss_enabled: false,
            sss_strength: 5.0,
            sss_color: [1.0, 0.4, 0.2],

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

            touch_zoom_sensitivity: 500.0,
            invert_touch_pan: false,
            roll_sensitivity: 0.005,
            invert_roll: false,
            pressure_sensitivity: false,
            clamp_orbit_pitch: false,
            shading_mode: ShadingMode::default(),
            show_node_labels: false,
            show_bounding_box: true,
            sculpt_safety_border: 0.05,
        }
    }
}

impl RenderConfig {
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
        self.key_light_dir = d.key_light_dir;
        self.key_diffuse = d.key_diffuse;
        self.key_spec_power = d.key_spec_power;
        self.key_spec_intensity = d.key_spec_intensity;
        self.fill_light_dir = d.fill_light_dir;
        self.fill_intensity = d.fill_intensity;
        self.ambient = d.ambient;
    }

    pub fn reset_sky(&mut self) {
        let d = Self::default();
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
}
