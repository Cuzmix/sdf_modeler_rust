use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl AppVec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppCameraSnapshot {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub distance: f32,
    pub fov_degrees: f32,
    pub orthographic: bool,
    pub target: AppVec3,
    pub eye: AppVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppNodeSnapshot {
    pub id: u64,
    pub name: String,
    pub kind_label: String,
    pub visible: bool,
    pub locked: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSceneTreeNodeSnapshot {
    pub id: u64,
    pub name: String,
    pub kind_label: String,
    pub visible: bool,
    pub locked: bool,
    pub children: Vec<AppSceneTreeNodeSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSceneStatsSnapshot {
    pub total_nodes: u32,
    pub visible_nodes: u32,
    pub top_level_nodes: u32,
    pub primitive_nodes: u32,
    pub operation_nodes: u32,
    pub transform_nodes: u32,
    pub modifier_nodes: u32,
    pub sculpt_nodes: u32,
    pub light_nodes: u32,
    pub voxel_memory_bytes: u64,
    pub sdf_eval_complexity: u32,
    pub structure_key: u64,
    pub data_fingerprint: u64,
    pub bounds_min: AppVec3,
    pub bounds_max: AppVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppToolSnapshot {
    pub active_tool_label: String,
    pub shading_mode_label: String,
    pub grid_enabled: bool,
    pub manipulator_mode_id: String,
    pub manipulator_mode_label: String,
    pub manipulator_space_id: String,
    pub manipulator_space_label: String,
    pub manipulator_visible: bool,
    pub can_reset_pivot: bool,
    pub pivot_offset: AppVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppHistorySnapshot {
    pub can_undo: bool,
    pub can_redo: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppDocumentSnapshot {
    pub current_file_path: Option<String>,
    pub current_file_name: Option<String>,
    pub has_unsaved_changes: bool,
    pub recent_files: Vec<String>,
    pub recovery_available: bool,
    pub recovery_summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppExportPresetSnapshot {
    pub name: String,
    pub resolution: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppExportStatusSnapshot {
    pub state: String,
    pub progress: u32,
    pub total: u32,
    pub resolution: u32,
    pub phase_label: Option<String>,
    pub target_file_name: Option<String>,
    pub target_file_path: Option<String>,
    pub format_label: Option<String>,
    pub message: Option<String>,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppExportSnapshot {
    pub resolution: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
    pub adaptive: bool,
    pub presets: Vec<AppExportPresetSnapshot>,
    pub status: AppExportStatusSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppImportDialogSnapshot {
    pub filename: String,
    pub resolution: u32,
    pub auto_resolution: u32,
    pub use_auto: bool,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounds_size: AppVec3,
    pub min_resolution: u32,
    pub max_resolution: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppImportStatusSnapshot {
    pub state: String,
    pub progress: u32,
    pub total: u32,
    pub filename: Option<String>,
    pub phase_label: Option<String>,
    pub message: Option<String>,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppImportSnapshot {
    pub dialog: Option<AppImportDialogSnapshot>,
    pub status: AppImportStatusSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSculptConvertDialogSnapshot {
    pub target_node_id: u64,
    pub target_name: String,
    pub mode_id: String,
    pub mode_label: String,
    pub resolution: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSculptConvertStatusSnapshot {
    pub state: String,
    pub progress: u32,
    pub total: u32,
    pub target_name: Option<String>,
    pub phase_label: Option<String>,
    pub message: Option<String>,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSculptConvertSnapshot {
    pub dialog: Option<AppSculptConvertDialogSnapshot>,
    pub status: AppSculptConvertStatusSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppScalarPropertySnapshot {
    pub key: String,
    pub label: String,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppTransformPropertiesSnapshot {
    pub position_label: String,
    pub position: AppVec3,
    pub rotation_degrees: AppVec3,
    pub scale: Option<AppVec3>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppPrimitivePropertiesSnapshot {
    pub primitive_kind: String,
    pub parameters: Vec<AppScalarPropertySnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppMaterialPropertiesSnapshot {
    pub color: AppVec3,
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: AppVec3,
    pub emissive_intensity: f32,
    pub fresnel: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSelectedNodePropertiesSnapshot {
    pub node_id: u64,
    pub name: String,
    pub kind_label: String,
    pub visible: bool,
    pub locked: bool,
    pub transform: Option<AppTransformPropertiesSnapshot>,
    pub primitive: Option<AppPrimitivePropertiesSnapshot>,
    pub material: Option<AppMaterialPropertiesSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSceneSnapshot {
    pub selected_node: Option<AppNodeSnapshot>,
    pub selected_node_properties: Option<AppSelectedNodePropertiesSnapshot>,
    pub top_level_nodes: Vec<AppNodeSnapshot>,
    pub scene_tree_roots: Vec<AppSceneTreeNodeSnapshot>,
    pub history: AppHistorySnapshot,
    pub document: AppDocumentSnapshot,
    pub export: AppExportSnapshot,
    pub import: AppImportSnapshot,
    pub sculpt_convert: AppSculptConvertSnapshot,
    pub camera: AppCameraSnapshot,
    pub stats: AppSceneStatsSnapshot,
    pub tool: AppToolSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppViewportFeedbackSnapshot {
    pub camera: AppCameraSnapshot,
    pub selected_node: Option<AppNodeSnapshot>,
    pub hovered_node: Option<AppNodeSnapshot>,
}
