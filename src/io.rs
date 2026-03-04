use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, Scene};

const CURRENT_VERSION: u32 = 5;

#[derive(Serialize, Deserialize)]
pub struct ProjectFile {
    pub version: u32,
    pub scene: Scene,
    pub camera: Camera,
}

// ── Pure serialization (shared by native + WASM) ────────────────────────────

pub fn project_to_json(scene: &Scene, camera: &Camera) -> Result<String, String> {
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
    };
    serde_json::to_string_pretty(&project).map_err(|e| e.to_string())
}

pub fn json_to_project(json: &str) -> Result<ProjectFile, String> {
    // Parse as raw JSON first to check version and do pre-deserialization migrations
    let mut raw: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

    let version = raw.get("version")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;

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
    let mut path = std::env::current_exe().unwrap_or_default();
    path.pop();
    path.push("autosave.sdf");
    path
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_project(scene: &Scene, camera: &Camera, path: &std::path::PathBuf) -> Result<(), String> {
    let json = project_to_json(scene, camera)?;
    std::fs::write(path, json).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_project(path: &std::path::PathBuf) -> Result<ProjectFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    json_to_project(&data)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn save_dialog() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .set_title("Save Project")
        .add_filter("SDF Project", &["sdf", "json"])
        .save_file()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn open_dialog() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .set_title("Open Project")
        .add_filter("SDF Project", &["sdf", "json"])
        .pick_file()
}

// ── WASM browser I/O ────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub fn web_download(filename: &str, data: &[u8], mime: &str) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else { return };
    let Some(document) = window.document() else { return };

    let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
    array.copy_from(data);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());
    let mut opts = web_sys::BlobPropertyBag::new();
    opts.set_type(mime);
    let Ok(blob) = web_sys::Blob::new_with_buffer_source_sequence_and_options(&parts, &opts) else { return };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else { return };

    if let Ok(a) = document.create_element("a") {
        let a: web_sys::HtmlAnchorElement = a.unchecked_into();
        a.set_href(&url);
        a.set_download(filename);
        a.click();
        let _ = web_sys::Url::revoke_object_url(&url);
    }
}

#[cfg(target_arch = "wasm32")]
pub fn web_save_project(scene: &Scene, camera: &Camera) {
    match project_to_json(scene, camera) {
        Ok(json) => web_download("project.sdf", json.as_bytes(), "application/json"),
        Err(e) => log::error!("Failed to serialize project: {}", e),
    }
}

/// Migrate v4→v5 at the raw JSON level (before deserialization).
/// Transform nodes changed from `{kind, value}` to `{translation, rotation, scale}`.
fn migrate_v4_to_v5_json(root: &mut serde_json::Value) {
    let Some(nodes) = root
        .get_mut("scene")
        .and_then(|s| s.get_mut("nodes"))
    else {
        return;
    };

    // nodes is a JSON map of id → node object
    let Some(nodes_map) = nodes.as_object_mut() else { return };

    for (_id, node) in nodes_map.iter_mut() {
        let Some(data) = node.get_mut("data") else { continue };
        let Some(transform) = data.get_mut("Transform") else { continue };

        // Read old fields
        let kind = transform.get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("")
            .to_string();
        let value = transform.get("value").cloned()
            .unwrap_or(serde_json::json!([0.0, 0.0, 0.0]));
        let input = transform.get("input").cloned()
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
                color,
                ..
            } => Some((n.id, *position, *color)),
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
