use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use crate::desktop_dialogs::FileDialogSelection;
use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode};
#[cfg(not(target_arch = "wasm32"))]
use crate::native_paths;
use crate::sculpt::{PersistedSculptState, SculptState};
use crate::settings::RenderConfig;

const CURRENT_VERSION: u32 = 6;

#[derive(Serialize, Deserialize)]
pub struct ProjectFile {
    pub version: u32,
    pub scene: Scene,
    pub camera: Camera,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_config: Option<RenderConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sculpt_state: Option<PersistedSculptState>,
}

const PRESET_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SubtreePresetFile {
    version: u32,
    root_id: u64,
    nodes: Vec<SubtreePresetNode>,
}

#[derive(Serialize, Deserialize)]
struct SubtreePresetNode {
    id: u64,
    name: String,
    locked: bool,
    hidden: bool,
    light_mask: u8,
    data: NodeData,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RecoveryMeta {
    pub autosave_unix_secs: u64,
    #[serde(default)]
    pub project_path: Option<String>,
    #[serde(default)]
    pub last_project_save_unix_secs: Option<u64>,
}

// ── Pure serialization (shared by native + WASM) ────────────────────────────

#[cfg_attr(not(test), allow(dead_code))]
pub fn project_to_json(
    scene: &Scene,
    camera: &Camera,
    render_config: Option<&RenderConfig>,
) -> Result<String, String> {
    project_to_json_with_sculpt(scene, camera, render_config, None)
}

pub fn project_to_json_with_sculpt(
    scene: &Scene,
    camera: &Camera,
    render_config: Option<&RenderConfig>,
    sculpt_state: Option<&SculptState>,
) -> Result<String, String> {
    let project = ProjectFile {
        version: CURRENT_VERSION,
        scene: scene.clone(),
        camera: Camera {
            yaw: camera.yaw,
            pitch: camera.pitch,
            roll: camera.roll,
            distance: camera.distance,
            target: camera.target,
            fov: camera.fov,
            orthographic: camera.orthographic,
            transition: None,
        },
        render_config: render_config.cloned(),
        sculpt_state: sculpt_state.map(SculptState::to_persisted),
    };
    serde_json::to_string_pretty(&project).map_err(|e| e.to_string())
}

pub fn json_to_project(json: &str) -> Result<ProjectFile, String> {
    // Parse as raw JSON first to check version and do pre-deserialization migrations
    let mut raw: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

    let version = raw.get("version").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

    if version > CURRENT_VERSION {
        return Err(format!(
            "Project file version {} is newer than supported version {}. Please update SDF Modeler.",
            version, CURRENT_VERSION
        ));
    }

    // v4→v5: Transform nodes changed from {kind, value} to {translation, rotation, scale}
    if version < 5 {
        migrate_v4_to_v5_json(&mut raw);
        raw["version"] = serde_json::Value::from(CURRENT_VERSION);
    }

    let mut project: ProjectFile = serde_json::from_value(raw).map_err(|e| e.to_string())?;

    if project.version < 3 {
        migrate_v2_to_v3(&mut project);
        project.version = CURRENT_VERSION;
    }

    Ok(project)
}

// ── Native file I/O ─────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub fn auto_save_path() -> std::path::PathBuf {
    native_paths::app_storage_file("autosave.sdf")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn auto_save_meta_path() -> std::path::PathBuf {
    native_paths::app_storage_file("autosave.meta")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_recovery_meta() -> Option<RecoveryMeta> {
    let meta_path = auto_save_meta_path();
    let data = std::fs::read_to_string(meta_path).ok()?;
    serde_json::from_str(&data).ok()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_recovery_meta(project_path: Option<&std::path::Path>) -> Result<(), String> {
    let autosave_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let project_path_str = project_path.map(|path| path.to_string_lossy().to_string());
    let last_project_save_unix_secs = project_path
        .and_then(|path| std::fs::metadata(path).ok())
        .and_then(|meta| meta.modified().ok())
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    let meta = RecoveryMeta {
        autosave_unix_secs,
        project_path: project_path_str,
        last_project_save_unix_secs,
    };
    let json = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    let path = auto_save_meta_path();
    native_paths::ensure_parent_dir(&path)?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn has_recovery_file() -> bool {
    let autosave_path = auto_save_path();
    if !autosave_path.exists() {
        return false;
    }
    let autosave_mtime = std::fs::metadata(&autosave_path)
        .ok()
        .and_then(|meta| meta.modified().ok())
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    let Some(meta) = read_recovery_meta() else {
        // If metadata is missing but autosave exists, still offer recovery.
        return true;
    };

    if let Some(project_path) = meta.project_path.as_deref() {
        let project_mtime = std::fs::metadata(project_path)
            .ok()
            .and_then(|meta| meta.modified().ok())
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        if let (Some(autosave_secs), Some(project_secs)) = (autosave_mtime, project_mtime) {
            return autosave_secs > project_secs;
        }
    }

    match meta.last_project_save_unix_secs {
        Some(last_save) => autosave_mtime.unwrap_or(meta.autosave_unix_secs) > last_save,
        None => true,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn remove_recovery_files() -> Result<(), String> {
    let autosave = auto_save_path();
    if autosave.exists() {
        std::fs::remove_file(&autosave).map_err(|e| e.to_string())?;
    }
    let meta = auto_save_meta_path();
    if meta.exists() {
        std::fs::remove_file(&meta).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn remap_subtree_node_references(data: &mut NodeData, id_map: &HashMap<NodeId, NodeId>) {
    let remap_opt = |value: &mut Option<NodeId>| {
        *value = value.and_then(|old_id| id_map.get(&old_id).copied());
    };

    match data {
        NodeData::Primitive { .. } => {}
        NodeData::Operation { left, right, .. } => {
            remap_opt(left);
            remap_opt(right);
        }
        NodeData::Sculpt { input, .. } => remap_opt(input),
        NodeData::Transform { input, .. } => remap_opt(input),
        NodeData::Modifier { input, .. } => remap_opt(input),
        NodeData::Light { cookie_node, .. } => remap_opt(cookie_node),
    }
}

/// Save a subtree rooted at `root` as a reusable `.sdfpreset` file.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_subtree_preset(
    scene: &Scene,
    root: NodeId,
    path: &std::path::Path,
) -> Result<(), String> {
    if !scene.nodes.contains_key(&root) {
        return Err(format!("Node {root} does not exist"));
    }

    let subtree_nodes = scene.collect_subtree_nodes(root);
    if subtree_nodes.is_empty() {
        return Err("Subtree is empty".to_string());
    }

    let mut relative_id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for (index, old_id) in subtree_nodes.iter().enumerate() {
        relative_id_map.insert(*old_id, index as NodeId);
    }

    let mut preset_nodes = Vec::with_capacity(subtree_nodes.len());
    for old_id in &subtree_nodes {
        let Some(node) = scene.nodes.get(old_id) else {
            return Err(format!("Missing subtree node {old_id}"));
        };
        let mut data = node.data.clone();
        remap_subtree_node_references(&mut data, &relative_id_map);
        preset_nodes.push(SubtreePresetNode {
            id: *relative_id_map
                .get(old_id)
                .ok_or_else(|| format!("Missing relative mapping for node {old_id}"))?,
            name: node.name.clone(),
            locked: node.locked,
            hidden: scene.hidden_nodes.contains(old_id),
            light_mask: scene.get_light_mask(*old_id),
            data,
        });
    }

    let preset = SubtreePresetFile {
        version: PRESET_VERSION,
        root_id: *relative_id_map
            .get(&root)
            .ok_or_else(|| "Missing relative root id".to_string())?,
        nodes: preset_nodes,
    };
    let json = serde_json::to_string_pretty(&preset).map_err(|e| e.to_string())?;
    native_paths::ensure_parent_dir(path)?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

/// Load a subtree preset and insert all nodes with fresh NodeIds.
/// Returns the newly inserted root node id.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_subtree_preset(scene: &mut Scene, path: &std::path::Path) -> Result<NodeId, String> {
    let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let preset: SubtreePresetFile = serde_json::from_str(&json).map_err(|e| e.to_string())?;

    if preset.version > PRESET_VERSION {
        return Err(format!(
            "Preset version {} is newer than supported version {}",
            preset.version, PRESET_VERSION
        ));
    }
    if preset.nodes.is_empty() {
        return Err("Preset has no nodes".to_string());
    }

    let mut fresh_id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in &preset.nodes {
        if fresh_id_map.contains_key(&node.id) {
            return Err(format!("Duplicate preset node id {}", node.id));
        }
        let new_id = scene.next_id;
        scene.next_id += 1;
        fresh_id_map.insert(node.id, new_id);
    }

    let root_new_id = *fresh_id_map
        .get(&preset.root_id)
        .ok_or_else(|| format!("Missing preset root id {}", preset.root_id))?;

    for preset_node in &preset.nodes {
        let new_id = *fresh_id_map
            .get(&preset_node.id)
            .ok_or_else(|| format!("Missing new id for preset node {}", preset_node.id))?;

        let mut data = preset_node.data.clone();
        remap_subtree_node_references(&mut data, &fresh_id_map);

        scene.nodes.insert(
            new_id,
            SceneNode {
                id: new_id,
                name: preset_node.name.clone(),
                locked: preset_node.locked,
                data,
            },
        );

        if preset_node.hidden {
            scene.hidden_nodes.insert(new_id);
        }
        if preset_node.light_mask != 0xFF {
            scene.light_masks.insert(new_id, preset_node.light_mask);
        }
    }

    Ok(root_new_id)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn save_project(
    scene: &Scene,
    camera: &Camera,
    render_config: &RenderConfig,
    path: &std::path::PathBuf,
) -> Result<(), String> {
    save_project_with_sculpt(scene, camera, render_config, None, path)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_project_with_sculpt(
    scene: &Scene,
    camera: &Camera,
    render_config: &RenderConfig,
    sculpt_state: Option<&SculptState>,
    path: &std::path::PathBuf,
) -> Result<(), String> {
    let json = project_to_json_with_sculpt(scene, camera, Some(render_config), sculpt_state)?;
    native_paths::ensure_parent_dir(path)?;
    std::fs::write(path, json).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_project(path: &std::path::PathBuf) -> Result<ProjectFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    json_to_project(&data)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_dialog() -> FileDialogSelection {
    crate::desktop_dialogs::save_project_dialog()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn open_dialog() -> FileDialogSelection {
    crate::desktop_dialogs::open_project_dialog()
}

// ── WASM browser I/O ────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub fn web_download(filename: &str, data: &[u8], mime: &str) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
    array.copy_from(data);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());
    let mut opts = web_sys::BlobPropertyBag::new();
    opts.set_type(mime);
    let Ok(blob) = web_sys::Blob::new_with_buffer_source_sequence_and_options(&parts, &opts) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };

    if let Ok(a) = document.create_element("a") {
        let a: web_sys::HtmlAnchorElement = a.unchecked_into();
        a.set_href(&url);
        a.set_download(filename);
        a.click();
        let _ = web_sys::Url::revoke_object_url(&url);
    }
}

#[cfg(target_arch = "wasm32")]
pub fn web_save_project(scene: &Scene, camera: &Camera, render_config: &RenderConfig) {
    web_save_project_with_sculpt(scene, camera, render_config, None)
}

#[cfg(target_arch = "wasm32")]
pub fn web_save_project_with_sculpt(
    scene: &Scene,
    camera: &Camera,
    render_config: &RenderConfig,
    sculpt_state: Option<&SculptState>,
) {
    match project_to_json_with_sculpt(scene, camera, Some(render_config), sculpt_state) {
        Ok(json) => web_download("project.sdf", json.as_bytes(), "application/json"),
        Err(e) => log::error!("Failed to serialize project: {}", e),
    }
}

/// Migrate v4→v5 at the raw JSON level (before deserialization).
/// Transform nodes changed from `{kind, value}` to `{translation, rotation, scale}`.
fn migrate_v4_to_v5_json(root: &mut serde_json::Value) {
    let Some(nodes) = root.get_mut("scene").and_then(|s| s.get_mut("nodes")) else {
        return;
    };

    // nodes is a JSON map of id → node object
    let Some(nodes_map) = nodes.as_object_mut() else {
        return;
    };

    for (_id, node) in nodes_map.iter_mut() {
        let Some(data) = node.get_mut("data") else {
            continue;
        };
        let Some(transform) = data.get_mut("Transform") else {
            continue;
        };

        // Read old fields
        let kind = transform
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("")
            .to_string();
        let value = transform
            .get("value")
            .cloned()
            .unwrap_or(serde_json::json!([0.0, 0.0, 0.0]));
        let input = transform
            .get("input")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        // Convert based on old kind
        let zero = serde_json::json!([0.0, 0.0, 0.0]);
        let one = serde_json::json!([1.0, 1.0, 1.0]);

        let (translation, rotation, scale) = match kind.as_str() {
            "Translate" => (value, zero, one),
            "Rotate" => (zero, value, one),
            "Scale" => (zero.clone(), zero, value),
            _ => (zero.clone(), zero, one),
        };

        // Rewrite the Transform object with new fields
        *transform = serde_json::json!({
            "input": input,
            "translation": translation,
            "rotation": rotation,
            "scale": scale,
        });
    }
}

/// Migrate v2 projects: convert Primitive nodes with embedded voxel_grid
/// into proper Sculpt modifier nodes.
fn migrate_v2_to_v3(project: &mut ProjectFile) {
    // Collect primitives that have a legacy voxel_grid
    let to_migrate: Vec<(u64, Vec3, Vec3)> = project
        .scene
        .nodes
        .values()
        .filter_map(|n| match &n.data {
            NodeData::Primitive {
                voxel_grid: Some(_),
                position,
                material,
                ..
            } => Some((n.id, *position, material.base_color)),
            _ => None,
        })
        .collect();

    let count = to_migrate.len();
    for (prim_id, position, color) in to_migrate {
        // Take the voxel grid out of the Primitive
        let grid = {
            let node = project.scene.nodes.get_mut(&prim_id).unwrap();
            if let NodeData::Primitive {
                ref mut voxel_grid, ..
            } = node.data
            {
                voxel_grid.take()
            } else {
                None
            }
        };

        if let Some(grid) = grid {
            project
                .scene
                .insert_sculpt_above(prim_id, position, Vec3::ZERO, color, grid);
        }
    }

    log::info!("Migrated {} sculpt nodes from v2 to v3 format", count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{LightType, NodeData, SdfPrimitive};
    use crate::sculpt::{BrushMode, SculptState};
    use crate::settings::{AmbientOcclusionMode, EnvironmentSource, RenderConfig};
    use std::collections::{HashMap, HashSet};

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
            structure_version: 0,
            data_version: 0,
        }
    }

    #[test]
    fn project_json_roundtrip_preserves_render_config() {
        let scene = empty_scene();
        let camera = Camera::default();
        let mut render = RenderConfig::default();
        render.env_reflection_enabled = true;
        render.env_reflection_intensity = 1.6;
        render.environment_source = EnvironmentSource::ProceduralSky;
        render.ao_mode = AmbientOcclusionMode::Quality;
        render.show_grid = false;

        let json =
            project_to_json(&scene, &camera, Some(&render)).expect("serialize project with render");
        let project = json_to_project(&json).expect("deserialize project with render");

        assert_eq!(project.version, CURRENT_VERSION);
        assert_eq!(project.render_config, Some(render));
    }

    #[test]
    fn project_json_roundtrip_preserves_sculpt_state() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt = scene.create_sculpt(
            sphere,
            glam::Vec3::ZERO,
            glam::Vec3::ZERO,
            glam::Vec3::new(0.8, 0.4, 0.2),
            crate::graph::voxel::VoxelGrid::new_displacement(
                16,
                glam::Vec3::splat(-1.0),
                glam::Vec3::splat(1.0),
            ),
        );
        let camera = Camera::default();
        let mut sculpt_state = SculptState::new_active(sculpt);
        sculpt_state.set_selected_brush(BrushMode::Grab);
        sculpt_state.profile_mut(BrushMode::Grab).strength = 1.7;
        sculpt_state.detail_state_mut().last_pre_expand_detail_size = Some(0.2);
        sculpt_state.detail_state_mut().detail_limited_after_growth = true;

        let json = project_to_json_with_sculpt(&scene, &camera, None, Some(&sculpt_state))
            .expect("serialize project with sculpt state");
        let project = json_to_project(&json).expect("deserialize project with sculpt state");
        let restored = project
            .sculpt_state
            .map(|persisted| SculptState::from_persisted(persisted, &project.scene))
            .expect("sculpt state should deserialize");

        assert_eq!(restored.active_node(), Some(sculpt));
        assert_eq!(restored.selected_brush(), BrushMode::Grab);
        assert!((restored.profile(BrushMode::Grab).strength - 1.7).abs() < 1e-5);
        assert_eq!(
            restored.detail_state().last_pre_expand_detail_size,
            Some(0.2)
        );
        assert!(restored.detail_state().detail_limited_after_growth);
    }

    #[test]
    fn material_showcase_scene_loads_and_covers_material_features() {
        let project = json_to_project(include_str!("../docs/scenes/material_showcase.sdf"))
            .expect("load material showcase project");
        let render = project
            .render_config
            .as_ref()
            .expect("material showcase should include render config");
        assert!(render.env_reflection_enabled);
        assert!(render.specular_aa_enabled);
        assert!(
            render.environment_source == EnvironmentSource::ProceduralSky
                || render.hdri_path.is_some(),
            "fixture should use either procedural IBL or a concrete HDRI path"
        );

        let mut found_clearcoat = false;
        let mut found_sheen = false;
        let mut found_anisotropy = false;
        let mut found_transmission = false;
        let mut found_emissive = false;

        for node in project.scene.nodes.values() {
            match &node.data {
                NodeData::Primitive { material, .. } | NodeData::Sculpt { material, .. } => {
                    found_clearcoat |= material.clearcoat > 0.01;
                    found_sheen |= material.sheen_color.length_squared() > 0.001;
                    found_anisotropy |= material.anisotropy_strength > 0.01;
                    found_transmission |= material.transmission > 0.01;
                    found_emissive |= material.emissive_intensity > 0.01;
                }
                _ => {}
            }
        }

        assert!(found_clearcoat, "fixture should cover clearcoat");
        assert!(found_sheen, "fixture should cover sheen");
        assert!(found_anisotropy, "fixture should cover anisotropy");
        assert!(found_transmission, "fixture should cover transmission");
        assert!(found_emissive, "fixture should cover emissive");
    }

    #[test]
    fn glass_contact_lab_loads_and_covers_glass_stress_cases() {
        let project = json_to_project(include_str!("../docs/scenes/glass_contact_lab.sdf"))
            .expect("load glass contact lab project");
        let render = project
            .render_config
            .as_ref()
            .expect("glass contact lab should include render config");
        assert!(render.shadows_enabled);
        assert!(render.env_reflection_enabled);
        assert!(render.specular_aa_enabled);

        let mut transmissive_nodes = 0;
        let mut metallic_nodes = 0;
        let mut emissive_nodes = 0;
        let mut box_primitives = 0;
        let mut point_lights = 0;

        for node in project.scene.nodes.values() {
            match &node.data {
                NodeData::Primitive { kind, material, .. } => {
                    if matches!(kind, SdfPrimitive::Box) {
                        box_primitives += 1;
                    }
                    if material.transmission > 0.01 {
                        transmissive_nodes += 1;
                    }
                    if material.metallic > 0.01 {
                        metallic_nodes += 1;
                    }
                    if material.emissive_intensity > 0.01 {
                        emissive_nodes += 1;
                    }
                }
                NodeData::Light { light_type, .. } => {
                    if matches!(light_type, LightType::Point) {
                        point_lights += 1;
                    }
                }
                _ => {}
            }
        }

        assert!(
            transmissive_nodes >= 2,
            "fixture should include at least two transmissive objects"
        );
        assert!(
            metallic_nodes >= 1,
            "fixture should include at least one metallic/glossy neighbor"
        );
        assert!(
            emissive_nodes >= 1,
            "fixture should include an emissive reference object"
        );
        assert!(
            box_primitives >= 4,
            "fixture should include enclosing box geometry for contact context"
        );
        assert!(
            point_lights >= 1,
            "fixture should include a point light for close highlight response"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn subtree_preset_roundtrip_preserves_structure_and_properties() {
        let mut source = empty_scene();
        let sphere = source.create_primitive(SdfPrimitive::Sphere);
        let root = source.create_transform(Some(sphere));

        if let Some(node) = source.nodes.get_mut(&root) {
            if let NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } = &mut node.data
            {
                *translation = Vec3::new(1.5, 2.0, -0.25);
                *rotation = Vec3::new(0.1, 0.2, 0.3);
                *scale = Vec3::new(1.1, 0.9, 1.2);
            }
        }
        if let Some(node) = source.nodes.get_mut(&sphere) {
            if let NodeData::Primitive { material, .. } = &mut node.data {
                material.base_color = Vec3::new(0.2, 0.8, 0.4);
                material.roughness = 0.32;
                material.metallic = 0.65;
            }
        }
        source.hidden_nodes.insert(sphere);
        source.set_light_mask(sphere, 0b0000_0011);

        let unique_name = format!(
            "sdf_preset_test_{}_{}.sdfpreset",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        );
        let preset_path = std::env::temp_dir().join(unique_name);

        save_subtree_preset(&source, root, &preset_path).expect("save subtree preset");

        let mut target = empty_scene();
        let loaded_root =
            load_subtree_preset(&mut target, &preset_path).expect("load subtree preset");
        let _ = std::fs::remove_file(&preset_path);

        assert!(
            target.nodes.contains_key(&loaded_root),
            "loaded root should exist"
        );
        let loaded_child = match &target.nodes[&loaded_root].data {
            NodeData::Transform {
                input,
                translation,
                rotation,
                scale,
            } => {
                assert_eq!(*translation, Vec3::new(1.5, 2.0, -0.25));
                assert_eq!(*rotation, Vec3::new(0.1, 0.2, 0.3));
                assert_eq!(*scale, Vec3::new(1.1, 0.9, 1.2));
                input.expect("transform should point to child")
            }
            _ => panic!("loaded root should be Transform"),
        };

        match &target.nodes[&loaded_child].data {
            NodeData::Primitive { material, .. } => {
                assert_eq!(material.base_color, Vec3::new(0.2, 0.8, 0.4));
                assert!((material.roughness - 0.32).abs() < 1e-6);
                assert!((material.metallic - 0.65).abs() < 1e-6);
            }
            _ => panic!("loaded child should be Primitive"),
        }
        assert!(
            target.hidden_nodes.contains(&loaded_child),
            "hidden state should roundtrip"
        );
        assert_eq!(
            target.get_light_mask(loaded_child),
            0b0000_0011,
            "light mask should roundtrip"
        );
    }
}
