use std::path::PathBuf;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, Scene};

const CURRENT_VERSION: u32 = 4;

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
    let mut project: ProjectFile = serde_json::from_str(&data).map_err(|e| e.to_string())?;

    if project.version > CURRENT_VERSION {
        return Err(format!(
            "Project file version {} is newer than supported version {}. Please update SDF Modeler.",
            project.version, CURRENT_VERSION
        ));
    }

    // Apply migrations
    if project.version < 3 {
        migrate_v2_to_v3(&mut project);
        project.version = CURRENT_VERSION;
    }

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
