use serde::{Deserialize, Serialize};

/// A material preset with physically-based properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPreset {
    pub name: String,
    pub category: String,
    /// RGB color [0..1]. `None` means "keep current color".
    pub color: Option<[f32; 3]>,
    pub roughness: f32,
    pub metallic: f32,
    pub fresnel: f32,
    pub emissive_intensity: f32,
}

impl MaterialPreset {
    pub fn from_node_material(
        name: &str,
        color: [f32; 3],
        roughness: f32,
        metallic: f32,
        fresnel: f32,
        emissive_intensity: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            category: "User".to_string(),
            color: Some(color),
            roughness,
            metallic,
            fresnel,
            emissive_intensity,
        }
    }
}

/// Categories for organizing built-in presets.
pub const CATEGORY_METALS: &str = "Metals";
pub const CATEGORY_PLASTICS: &str = "Plastics";
pub const CATEGORY_STONE: &str = "Stone";
pub const CATEGORY_ORGANIC: &str = "Organic";
pub const CATEGORY_SPECIAL: &str = "Special";
pub const CATEGORY_USER: &str = "User";

/// All category names in display order.
pub const CATEGORIES: &[&str] = &[
    CATEGORY_METALS,
    CATEGORY_PLASTICS,
    CATEGORY_STONE,
    CATEGORY_ORGANIC,
    CATEGORY_SPECIAL,
    CATEGORY_USER,
];

/// Returns ~20 physically-accurate built-in material presets.
/// Values based on real-world IOR/albedo reference data.
pub fn built_in_presets() -> Vec<MaterialPreset> {
    vec![
        // ── Metals ──
        MaterialPreset {
            name: "Gold".into(),
            category: CATEGORY_METALS.into(),
            color: Some([1.0, 0.766, 0.336]),
            metallic: 1.0,
            roughness: 0.3,
            fresnel: 0.95,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Silver".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.972, 0.960, 0.915]),
            metallic: 1.0,
            roughness: 0.2,
            fresnel: 0.97,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Copper".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.955, 0.638, 0.538]),
            metallic: 1.0,
            roughness: 0.35,
            fresnel: 0.93,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Brushed Steel".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.77, 0.78, 0.78]),
            metallic: 1.0,
            roughness: 0.45,
            fresnel: 0.85,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Aluminum".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.913, 0.922, 0.924]),
            metallic: 1.0,
            roughness: 0.25,
            fresnel: 0.91,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Bronze".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.714, 0.494, 0.267]),
            metallic: 1.0,
            roughness: 0.4,
            fresnel: 0.88,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Iron".into(),
            category: CATEGORY_METALS.into(),
            color: Some([0.560, 0.570, 0.580]),
            metallic: 1.0,
            roughness: 0.55,
            fresnel: 0.78,
            emissive_intensity: 0.0,
        },
        // ── Plastics ──
        MaterialPreset {
            name: "Red Plastic".into(),
            category: CATEGORY_PLASTICS.into(),
            color: Some([0.85, 0.12, 0.12]),
            metallic: 0.0,
            roughness: 0.35,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "White Plastic".into(),
            category: CATEGORY_PLASTICS.into(),
            color: Some([0.95, 0.95, 0.95]),
            metallic: 0.0,
            roughness: 0.35,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Black Rubber".into(),
            category: CATEGORY_PLASTICS.into(),
            color: Some([0.05, 0.05, 0.05]),
            metallic: 0.0,
            roughness: 0.9,
            fresnel: 0.02,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Glossy Plastic".into(),
            category: CATEGORY_PLASTICS.into(),
            color: None,
            metallic: 0.0,
            roughness: 0.1,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        // ── Stone ──
        MaterialPreset {
            name: "Marble".into(),
            category: CATEGORY_STONE.into(),
            color: Some([0.92, 0.91, 0.88]),
            metallic: 0.0,
            roughness: 0.2,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Concrete".into(),
            category: CATEGORY_STONE.into(),
            color: Some([0.55, 0.55, 0.52]),
            metallic: 0.0,
            roughness: 0.85,
            fresnel: 0.03,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Granite".into(),
            category: CATEGORY_STONE.into(),
            color: Some([0.45, 0.42, 0.40]),
            metallic: 0.0,
            roughness: 0.6,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Sandstone".into(),
            category: CATEGORY_STONE.into(),
            color: Some([0.76, 0.70, 0.55]),
            metallic: 0.0,
            roughness: 0.75,
            fresnel: 0.03,
            emissive_intensity: 0.0,
        },
        // ── Organic ──
        MaterialPreset {
            name: "Skin".into(),
            category: CATEGORY_ORGANIC.into(),
            color: Some([0.80, 0.60, 0.48]),
            metallic: 0.0,
            roughness: 0.5,
            fresnel: 0.028,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Wood".into(),
            category: CATEGORY_ORGANIC.into(),
            color: Some([0.55, 0.35, 0.18]),
            metallic: 0.0,
            roughness: 0.65,
            fresnel: 0.04,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Leather".into(),
            category: CATEGORY_ORGANIC.into(),
            color: Some([0.40, 0.25, 0.13]),
            metallic: 0.0,
            roughness: 0.7,
            fresnel: 0.035,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Clay".into(),
            category: CATEGORY_ORGANIC.into(),
            color: Some([0.72, 0.50, 0.35]),
            metallic: 0.0,
            roughness: 0.8,
            fresnel: 0.03,
            emissive_intensity: 0.0,
        },
        // ── Special ──
        MaterialPreset {
            name: "Glass".into(),
            category: CATEGORY_SPECIAL.into(),
            color: Some([0.95, 0.95, 0.95]),
            metallic: 0.0,
            roughness: 0.05,
            fresnel: 0.5,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Mirror".into(),
            category: CATEGORY_SPECIAL.into(),
            color: Some([0.95, 0.95, 0.95]),
            metallic: 1.0,
            roughness: 0.01,
            fresnel: 1.0,
            emissive_intensity: 0.0,
        },
        MaterialPreset {
            name: "Emissive White".into(),
            category: CATEGORY_SPECIAL.into(),
            color: Some([1.0, 1.0, 1.0]),
            metallic: 0.0,
            roughness: 0.5,
            fresnel: 0.04,
            emissive_intensity: 2.0,
        },
    ]
}

/// Runtime material library: built-in + user-saved presets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterialLibrary {
    /// User-created presets (persisted to materials.json).
    pub user_presets: Vec<MaterialPreset>,
}

impl MaterialLibrary {
    /// Save a new user preset. Returns the index in user_presets.
    pub fn save_preset(&mut self, preset: MaterialPreset) -> usize {
        // Replace existing preset with the same name if present
        if let Some(pos) = self.user_presets.iter().position(|p| p.name == preset.name) {
            self.user_presets[pos] = preset;
            pos
        } else {
            self.user_presets.push(preset);
            self.user_presets.len() - 1
        }
    }

    /// Remove a user preset by index.
    pub fn remove_preset(&mut self, index: usize) {
        if index < self.user_presets.len() {
            self.user_presets.remove(index);
        }
    }

    /// Returns the file path for persisting user presets.
    #[cfg(not(target_arch = "wasm32"))]
    fn library_path() -> std::path::PathBuf {
        let mut path = std::env::current_exe().unwrap_or_default();
        path.pop();
        path.push("materials.json");
        path
    }

    /// Load user presets from disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load() -> Self {
        let path = Self::library_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save user presets to disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self) {
        let path = Self::library_path();
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save material library: {}", e);
            }
        }
    }

    /// Stub for WASM — no filesystem persistence.
    #[cfg(target_arch = "wasm32")]
    pub fn load() -> Self {
        Self::default()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn save(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_presets_has_expected_count() {
        let presets = built_in_presets();
        assert!(
            presets.len() >= 20,
            "Expected at least 20 built-in presets, got {}",
            presets.len()
        );
    }

    #[test]
    fn built_in_presets_cover_all_categories() {
        let presets = built_in_presets();
        for category in &[
            CATEGORY_METALS,
            CATEGORY_PLASTICS,
            CATEGORY_STONE,
            CATEGORY_ORGANIC,
            CATEGORY_SPECIAL,
        ] {
            let count = presets.iter().filter(|p| p.category == *category).count();
            assert!(count > 0, "Category '{}' has no presets", category);
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let preset = MaterialPreset::from_node_material(
            "Test Material",
            [0.5, 0.3, 0.1],
            0.6,
            0.8,
            0.04,
            0.0,
        );
        let json = serde_json::to_string(&preset).unwrap();
        let loaded: MaterialPreset = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "Test Material");
        assert_eq!(loaded.category, "User");
        assert!((loaded.roughness - 0.6).abs() < 0.001);
        assert!((loaded.metallic - 0.8).abs() < 0.001);
    }

    #[test]
    fn library_save_preset_replaces_duplicate_name() {
        let mut library = MaterialLibrary::default();
        let preset_a =
            MaterialPreset::from_node_material("Shiny", [1.0, 0.0, 0.0], 0.1, 1.0, 0.9, 0.0);
        let preset_b =
            MaterialPreset::from_node_material("Shiny", [0.0, 1.0, 0.0], 0.2, 0.5, 0.5, 0.0);

        library.save_preset(preset_a);
        assert_eq!(library.user_presets.len(), 1);

        library.save_preset(preset_b);
        assert_eq!(
            library.user_presets.len(),
            1,
            "Duplicate name should replace, not append"
        );
        assert!((library.user_presets[0].roughness - 0.2).abs() < 0.001);
    }

    #[test]
    fn library_remove_preset() {
        let mut library = MaterialLibrary::default();
        library.save_preset(MaterialPreset::from_node_material(
            "A", [1.0; 3], 0.5, 0.0, 0.04, 0.0,
        ));
        library.save_preset(MaterialPreset::from_node_material(
            "B", [0.0; 3], 0.5, 0.0, 0.04, 0.0,
        ));
        assert_eq!(library.user_presets.len(), 2);

        library.remove_preset(0);
        assert_eq!(library.user_presets.len(), 1);
        assert_eq!(library.user_presets[0].name, "B");
    }

    #[test]
    fn library_serialization_roundtrip() {
        let mut library = MaterialLibrary::default();
        library.save_preset(MaterialPreset::from_node_material(
            "Custom Gold",
            [1.0, 0.8, 0.3],
            0.3,
            1.0,
            0.95,
            0.0,
        ));
        library.save_preset(MaterialPreset::from_node_material(
            "Neon",
            [0.0, 1.0, 0.5],
            0.5,
            0.0,
            0.04,
            3.0,
        ));

        let json = serde_json::to_string_pretty(&library).unwrap();
        let loaded: MaterialLibrary = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.user_presets.len(), 2);
        assert_eq!(loaded.user_presets[0].name, "Custom Gold");
        assert_eq!(loaded.user_presets[1].name, "Neon");
    }
}
