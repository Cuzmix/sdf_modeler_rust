use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::graph::scene::MaterialParams;
#[cfg(not(target_arch = "wasm32"))]
use crate::native_paths;

/// A material preset with physically-based properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPreset {
    pub name: String,
    pub category: String,
    /// When true, applying this preset preserves the current base color.
    pub preserve_existing_base_color: bool,
    pub material: MaterialParams,
}

impl MaterialPreset {
    pub fn from_node_material(name: &str, material: MaterialParams) -> Self {
        Self {
            name: name.to_string(),
            category: "User".to_string(),
            preserve_existing_base_color: false,
            material,
        }
    }
}

fn rgb(color: [f32; 3]) -> Vec3 {
    Vec3::new(color[0], color[1], color[2])
}

#[allow(clippy::too_many_arguments)]
fn make_preset(
    name: &str,
    category: &str,
    base_color: Option<[f32; 3]>,
    metallic: f32,
    roughness: f32,
    reflectance_f0: f32,
    emissive_color: [f32; 3],
    emissive_intensity: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen_color: [f32; 3],
    sheen_roughness: f32,
) -> MaterialPreset {
    let mut material = MaterialParams::default();
    if let Some(color) = base_color {
        material.base_color = rgb(color);
    }
    material.metallic = metallic;
    material.roughness = roughness;
    material.reflectance_f0 = reflectance_f0;
    material.emissive_color = rgb(emissive_color);
    material.emissive_intensity = emissive_intensity;
    material.clearcoat = clearcoat;
    material.clearcoat_roughness = clearcoat_roughness;
    material.sheen_color = rgb(sheen_color);
    material.sheen_roughness = sheen_roughness;

    MaterialPreset {
        name: name.to_string(),
        category: category.to_string(),
        preserve_existing_base_color: base_color.is_none(),
        material,
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

/// Returns built-in material presets.
pub fn built_in_presets() -> Vec<MaterialPreset> {
    vec![
        make_preset(
            "Gold",
            CATEGORY_METALS,
            Some([1.0, 0.766, 0.336]),
            1.0,
            0.3,
            0.95,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Silver",
            CATEGORY_METALS,
            Some([0.972, 0.960, 0.915]),
            1.0,
            0.2,
            0.97,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Copper",
            CATEGORY_METALS,
            Some([0.955, 0.638, 0.538]),
            1.0,
            0.35,
            0.93,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        {
            let mut preset = make_preset(
                "Brushed Steel",
                CATEGORY_METALS,
                Some([0.77, 0.78, 0.78]),
                1.0,
                0.45,
                0.85,
                [0.0; 3],
                0.0,
                0.0,
                0.2,
                [0.0; 3],
                0.5,
            );
            preset.material.anisotropy_strength = 0.65;
            preset.material.anisotropy_direction_local = Vec3::X;
            preset
        },
        make_preset(
            "Aluminum",
            CATEGORY_METALS,
            Some([0.913, 0.922, 0.924]),
            1.0,
            0.25,
            0.91,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Bronze",
            CATEGORY_METALS,
            Some([0.714, 0.494, 0.267]),
            1.0,
            0.4,
            0.88,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Iron",
            CATEGORY_METALS,
            Some([0.560, 0.570, 0.580]),
            1.0,
            0.55,
            0.78,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Red Plastic",
            CATEGORY_PLASTICS,
            Some([0.85, 0.12, 0.12]),
            0.0,
            0.35,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "White Plastic",
            CATEGORY_PLASTICS,
            Some([0.95, 0.95, 0.95]),
            0.0,
            0.35,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Black Rubber",
            CATEGORY_PLASTICS,
            Some([0.05, 0.05, 0.05]),
            0.0,
            0.9,
            0.02,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Glossy Plastic",
            CATEGORY_PLASTICS,
            None,
            0.0,
            0.1,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Marble",
            CATEGORY_STONE,
            Some([0.92, 0.91, 0.88]),
            0.0,
            0.2,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Concrete",
            CATEGORY_STONE,
            Some([0.55, 0.55, 0.52]),
            0.0,
            0.85,
            0.03,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Granite",
            CATEGORY_STONE,
            Some([0.45, 0.42, 0.40]),
            0.0,
            0.6,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Sandstone",
            CATEGORY_STONE,
            Some([0.76, 0.70, 0.55]),
            0.0,
            0.75,
            0.03,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Skin",
            CATEGORY_ORGANIC,
            Some([0.80, 0.60, 0.48]),
            0.0,
            0.5,
            0.028,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.18, 0.08, 0.05],
            0.55,
        ),
        make_preset(
            "Wood",
            CATEGORY_ORGANIC,
            Some([0.55, 0.35, 0.18]),
            0.0,
            0.65,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Leather",
            CATEGORY_ORGANIC,
            Some([0.40, 0.25, 0.13]),
            0.0,
            0.7,
            0.035,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.08, 0.04, 0.02],
            0.6,
        ),
        make_preset(
            "Clay",
            CATEGORY_ORGANIC,
            Some([0.72, 0.50, 0.35]),
            0.0,
            0.8,
            0.03,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        {
            let mut preset = make_preset(
                "Glass",
                CATEGORY_SPECIAL,
                Some([0.95, 0.98, 1.0]),
                0.0,
                0.05,
                0.04,
                [0.0; 3],
                0.0,
                0.0,
                0.2,
                [0.0; 3],
                0.5,
            );
            preset.material.transmission = 1.0;
            preset.material.thickness = 0.8;
            preset.material.ior = 1.5;
            preset
        },
        {
            let mut preset = make_preset(
                "Frosted Glass",
                CATEGORY_SPECIAL,
                Some([0.90, 0.96, 1.0]),
                0.0,
                0.35,
                0.04,
                [0.0; 3],
                0.0,
                0.0,
                0.2,
                [0.0; 3],
                0.5,
            );
            preset.material.transmission = 1.0;
            preset.material.thickness = 1.2;
            preset.material.ior = 1.45;
            preset
        },
        make_preset(
            "Mirror",
            CATEGORY_SPECIAL,
            Some([0.95, 0.95, 0.95]),
            1.0,
            0.01,
            1.0,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Emissive White",
            CATEGORY_SPECIAL,
            Some([1.0, 1.0, 1.0]),
            0.0,
            0.5,
            0.04,
            [1.0, 1.0, 1.0],
            2.0,
            0.0,
            0.2,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Clearcoat Paint",
            CATEGORY_SPECIAL,
            Some([0.72, 0.08, 0.06]),
            0.0,
            0.35,
            0.04,
            [0.0; 3],
            0.0,
            1.0,
            0.08,
            [0.0; 3],
            0.5,
        ),
        make_preset(
            "Satin Fabric",
            CATEGORY_SPECIAL,
            Some([0.28, 0.26, 0.46]),
            0.0,
            0.8,
            0.04,
            [0.0; 3],
            0.0,
            0.0,
            0.2,
            [0.55, 0.48, 0.78],
            0.75,
        ),
        {
            let mut preset = make_preset(
                "Anodized Aluminum",
                CATEGORY_METALS,
                Some([0.62, 0.68, 0.78]),
                1.0,
                0.28,
                0.90,
                [0.0; 3],
                0.0,
                0.0,
                0.2,
                [0.0; 3],
                0.5,
            );
            preset.material.anisotropy_strength = 0.7;
            preset.material.anisotropy_direction_local = Vec3::X;
            preset
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

    #[cfg(not(target_arch = "wasm32"))]
    fn library_path() -> std::path::PathBuf {
        native_paths::app_storage_file("materials.json")
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load() -> Self {
        let path = Self::library_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self) {
        let path = Self::library_path();
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = native_paths::ensure_parent_dir(&path) {
                log::error!("Failed to create material library directory: {}", e);
                return;
            }
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save material library: {}", e);
            }
        }
    }

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
            MaterialParams {
                base_color: Vec3::new(0.5, 0.3, 0.1),
                roughness: 0.6,
                metallic: 0.8,
                reflectance_f0: 0.04,
                ..MaterialParams::default()
            },
        );
        let json = serde_json::to_string(&preset).unwrap();
        let loaded: MaterialPreset = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "Test Material");
        assert_eq!(loaded.category, "User");
        assert!((loaded.material.roughness - 0.6).abs() < 0.001);
        assert!((loaded.material.metallic - 0.8).abs() < 0.001);
    }

    #[test]
    fn library_save_preset_replaces_duplicate_name() {
        let mut library = MaterialLibrary::default();
        let preset_a = MaterialPreset::from_node_material(
            "Shiny",
            MaterialParams {
                base_color: Vec3::new(1.0, 0.0, 0.0),
                roughness: 0.1,
                metallic: 1.0,
                reflectance_f0: 0.9,
                ..MaterialParams::default()
            },
        );
        let preset_b = MaterialPreset::from_node_material(
            "Shiny",
            MaterialParams {
                base_color: Vec3::new(0.0, 1.0, 0.0),
                roughness: 0.2,
                metallic: 0.5,
                reflectance_f0: 0.5,
                ..MaterialParams::default()
            },
        );

        library.save_preset(preset_a);
        assert_eq!(library.user_presets.len(), 1);

        library.save_preset(preset_b);
        assert_eq!(
            library.user_presets.len(),
            1,
            "Duplicate name should replace, not append"
        );
        assert!((library.user_presets[0].material.roughness - 0.2).abs() < 0.001);
    }

    #[test]
    fn library_remove_preset() {
        let mut library = MaterialLibrary::default();
        library.save_preset(MaterialPreset::from_node_material(
            "A",
            MaterialParams {
                base_color: Vec3::ONE,
                ..MaterialParams::default()
            },
        ));
        library.save_preset(MaterialPreset::from_node_material(
            "B",
            MaterialParams {
                base_color: Vec3::ZERO,
                ..MaterialParams::default()
            },
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
            MaterialParams {
                base_color: Vec3::new(1.0, 0.8, 0.3),
                roughness: 0.3,
                metallic: 1.0,
                reflectance_f0: 0.95,
                ..MaterialParams::default()
            },
        ));
        library.save_preset(MaterialPreset::from_node_material(
            "Neon",
            MaterialParams {
                base_color: Vec3::new(0.0, 1.0, 0.5),
                emissive_color: Vec3::new(0.0, 1.0, 0.5),
                emissive_intensity: 3.0,
                ..MaterialParams::default()
            },
        ));

        let json = serde_json::to_string_pretty(&library).unwrap();
        let loaded: MaterialLibrary = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.user_presets.len(), 2);
        assert_eq!(loaded.user_presets[0].name, "Custom Gold");
        assert_eq!(loaded.user_presets[1].name, "Neon");
        assert!((loaded.user_presets[1].material.emissive_intensity - 3.0).abs() < 0.001);
    }
}
