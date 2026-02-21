use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    pub vsync_enabled: bool,
    #[serde(default)]
    pub shadows_enabled: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            vsync_enabled: true,
            shadows_enabled: false,
        }
    }
}

impl Settings {
    fn path() -> PathBuf {
        let mut path = std::env::current_exe().unwrap_or_default();
        path.pop(); // remove executable name
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
