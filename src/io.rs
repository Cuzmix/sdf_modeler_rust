use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::gpu::camera::Camera;
use crate::graph::scene::Scene;

const CURRENT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct ProjectFile {
    pub version: u32,
    pub scene: Scene,
    pub camera: Camera,
}

pub fn save_project(scene: &Scene, camera: &Camera, path: &PathBuf) -> Result<(), String> {
    let project = ProjectFile {
        version: CURRENT_VERSION,
        scene: scene.clone(),
        camera: Camera {
            yaw: camera.yaw,
            pitch: camera.pitch,
            distance: camera.distance,
            target: camera.target,
            fov: camera.fov,
        },
    };
    let json = serde_json::to_string_pretty(&project).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn load_project(path: &PathBuf) -> Result<ProjectFile, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let project: ProjectFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    if project.version > CURRENT_VERSION {
        return Err(format!(
            "Project file version {} is newer than supported version {}. Please update SDF Modeler.",
            project.version, CURRENT_VERSION
        ));
    }

    // Future: apply migrations for project.version < CURRENT_VERSION here

    Ok(project)
}

pub fn save_dialog() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Save Project")
        .add_filter("SDF Project", &["sdf", "json"])
        .save_file()
}

pub fn open_dialog() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Open Project")
        .add_filter("SDF Project", &["sdf", "json"])
        .pick_file()
}
