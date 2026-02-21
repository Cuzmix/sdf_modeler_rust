use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    pub vsync_enabled: bool,
    #[serde(default)]
    pub render: RenderConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            vsync_enabled: true,
            render: RenderConfig::default(),
        }
    }
}

impl Settings {
    fn path() -> PathBuf {
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

    // Sky
    pub sky_horizon: [f32; 3],
    pub sky_zenith: [f32; 3],

    // Gamma
    pub gamma: f32,
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
            ao_step: 0.08,
            ao_decay: 0.95,
            ao_intensity: 3.0,

            march_max_steps: 128,
            march_epsilon: 0.002,
            march_step_multiplier: 0.8,
            march_max_distance: 50.0,

            key_light_dir: [1.0, 2.0, 3.0],
            key_diffuse: 0.85,
            key_spec_power: 32.0,
            key_spec_intensity: 0.4,
            fill_light_dir: [-1.0, 0.5, -1.0],
            fill_intensity: 0.25,
            ambient: 0.06,

            sky_horizon: [0.10, 0.10, 0.16],
            sky_zenith: [0.02, 0.02, 0.05],

            gamma: 2.2,
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
    }

    pub fn reset_gamma(&mut self) {
        self.gamma = Self::default().gamma;
    }

    pub fn reset_all(&mut self) {
        *self = Self::default();
    }
}
