use std::sync::Arc;

use glam::Vec3;

use crate::graph::scene::NodeId;
use crate::mesh_import::TriMesh;

pub const MIN_VOXEL_RESOLUTION: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptConvertMode {
    BakeActiveNode,
    BakeWholeScene,
    BakeWholeSceneFlatten,
}

impl SculptConvertMode {
    pub fn id(self) -> &'static str {
        match self {
            Self::BakeActiveNode => "active_node",
            Self::BakeWholeScene => "whole_scene",
            Self::BakeWholeSceneFlatten => "whole_scene_flatten",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BakeActiveNode => "Bake active node",
            Self::BakeWholeScene => "Bake whole scene",
            Self::BakeWholeSceneFlatten => "Bake whole scene + flatten",
        }
    }

    pub fn from_id(mode_id: &str) -> Option<Self> {
        Some(match mode_id {
            "active_node" => Self::BakeActiveNode,
            "whole_scene" => Self::BakeWholeScene,
            "whole_scene_flatten" => Self::BakeWholeSceneFlatten,
            _ => return None,
        })
    }
}

pub struct WorkflowMessage {
    pub text: String,
    pub is_error: bool,
}

pub struct SculptConvertDialogState {
    pub target: NodeId,
    pub target_name: String,
    pub mode: SculptConvertMode,
    pub resolution: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
}

impl SculptConvertDialogState {
    pub fn new(target: NodeId, target_name: String, max_resolution: u32) -> Self {
        Self {
            target,
            target_name,
            mode: SculptConvertMode::BakeActiveNode,
            resolution: 64u32.clamp(MIN_VOXEL_RESOLUTION, max_resolution.max(MIN_VOXEL_RESOLUTION)),
            min_resolution: MIN_VOXEL_RESOLUTION,
            max_resolution: max_resolution.max(MIN_VOXEL_RESOLUTION),
        }
    }

    pub fn set_resolution(&mut self, resolution: u32) {
        self.resolution = resolution.clamp(self.min_resolution, self.max_resolution);
    }

    pub fn set_max_resolution(&mut self, max_resolution: u32) {
        self.max_resolution = max_resolution.max(MIN_VOXEL_RESOLUTION);
        self.set_resolution(self.resolution);
    }
}

pub struct ImportDialogState {
    pub mesh: Arc<TriMesh>,
    pub filename: String,
    pub resolution: u32,
    pub auto_resolution: u32,
    pub use_auto: bool,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounds_size: Vec3,
    pub min_resolution: u32,
    pub max_resolution: u32,
}

impl ImportDialogState {
    pub fn new(mesh: TriMesh, filename: String, max_resolution: u32) -> Self {
        let vertex_count = mesh.vertices.len();
        let triangle_count = mesh.triangles.len();

        let mut mesh_min = Vec3::splat(f32::MAX);
        let mut mesh_max = Vec3::splat(f32::MIN);
        for vertex in &mesh.vertices {
            mesh_min = mesh_min.min(*vertex);
            mesh_max = mesh_max.max(*vertex);
        }
        let bounds_size = mesh_max - mesh_min;

        let clamped_max_resolution = max_resolution.max(MIN_VOXEL_RESOLUTION);
        let auto_resolution = ((triangle_count as f32).cbrt() * 2.5)
            .round()
            .clamp(32.0, clamped_max_resolution as f32) as u32;

        Self {
            mesh: Arc::new(mesh),
            filename,
            resolution: auto_resolution,
            auto_resolution,
            use_auto: true,
            vertex_count,
            triangle_count,
            bounds_size,
            min_resolution: MIN_VOXEL_RESOLUTION,
            max_resolution: clamped_max_resolution,
        }
    }

    pub fn set_use_auto(&mut self, use_auto: bool) {
        self.use_auto = use_auto;
        if use_auto {
            self.resolution = self.auto_resolution;
        }
    }

    pub fn set_resolution(&mut self, resolution: u32) {
        self.resolution = resolution.clamp(self.min_resolution, self.max_resolution);
    }

    pub fn set_max_resolution(&mut self, max_resolution: u32) {
        self.max_resolution = max_resolution.max(MIN_VOXEL_RESOLUTION);
        self.auto_resolution = self
            .auto_resolution
            .clamp(self.min_resolution, self.max_resolution);
        if self.use_auto {
            self.resolution = self.auto_resolution;
        } else {
            self.set_resolution(self.resolution);
        }
    }
}
