use std::collections::HashMap;

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
    #[serde(default)]
    pub workflow_status_id: String,
    #[serde(default)]
    pub workflow_status_label: String,
    pub children: Vec<AppSceneTreeNodeSnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppWorkspaceSnapshot {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppQuickActionSnapshot {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub prominent: bool,
    #[serde(default)]
    pub shortcut_label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppCommandSnapshot {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub workspace_ids: Vec<String>,
    #[serde(default)]
    pub shortcut_label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppSelectionContextSnapshot {
    #[serde(default)]
    pub headline: String,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub selection_count: u32,
    #[serde(default)]
    pub selection_kind_id: String,
    #[serde(default)]
    pub selection_kind_label: String,
    #[serde(default)]
    pub workflow_status_id: String,
    #[serde(default)]
    pub workflow_status_label: String,
    #[serde(default)]
    pub quick_actions: Vec<AppQuickActionSnapshot>,
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
pub struct AppRenderOptionSnapshot {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppRenderSettingsSnapshot {
    pub shading_modes: Vec<AppRenderOptionSnapshot>,
    pub shading_mode_id: String,
    pub shading_mode_label: String,
    pub show_grid: bool,
    pub shadows_enabled: bool,
    pub shadow_steps: u32,
    pub ao_enabled: bool,
    pub ao_samples: u32,
    pub ao_intensity: f32,
    pub march_max_steps: u32,
    pub sculpt_fast_mode: bool,
    pub auto_reduce_steps: bool,
    pub interaction_render_scale: f32,
    pub rest_render_scale: f32,
    pub fog_enabled: bool,
    pub fog_density: f32,
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub gamma: f32,
    pub tonemapping_aces: bool,
    pub cross_section_axis: u8,
    pub cross_section_position: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppKeyOptionSnapshot {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppKeyComboSnapshot {
    pub key_id: String,
    pub key_label: String,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub shortcut_label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppKeybindingSnapshot {
    pub action_id: String,
    pub action_label: String,
    pub category: String,
    pub binding: Option<AppKeyComboSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppCameraBookmarkSnapshot {
    pub slot_index: u8,
    pub saved: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppShellPreferencesSnapshot {
    #[serde(default)]
    pub leading_edge_side: String,
    pub desktop_scene_pinned: bool,
    pub desktop_properties_pinned: bool,
    #[serde(default)]
    pub favorite_command_ids_by_workspace: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub preferred_drawer_tab: String,
    pub quick_wheel_hint_dismissed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppShellPreferencesUpdate {
    #[serde(default)]
    pub leading_edge_side: Option<String>,
    #[serde(default)]
    pub desktop_scene_pinned: Option<bool>,
    #[serde(default)]
    pub desktop_properties_pinned: Option<bool>,
    #[serde(default)]
    pub favorite_command_ids_by_workspace: Option<HashMap<String, Vec<String>>>,
    #[serde(default)]
    pub preferred_drawer_tab: Option<String>,
    #[serde(default)]
    pub quick_wheel_hint_dismissed: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSettingsSnapshot {
    pub show_fps_overlay: bool,
    pub show_node_labels: bool,
    pub show_bounding_box: bool,
    pub show_light_gizmos: bool,
    pub auto_save_enabled: bool,
    pub auto_save_interval_secs: u32,
    pub max_export_resolution: u32,
    pub max_sculpt_resolution: u32,
    pub camera_bookmarks: Vec<AppCameraBookmarkSnapshot>,
    pub shell_preferences: AppShellPreferencesSnapshot,
    pub key_options: Vec<AppKeyOptionSnapshot>,
    pub keybindings: Vec<AppKeybindingSnapshot>,
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
pub struct AppWorkflowStatusSnapshot {
    pub export_status: AppExportStatusSnapshot,
    pub import_status: AppImportStatusSnapshot,
    pub sculpt_convert_status: AppSculptConvertStatusSnapshot,
    pub scene_changed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSelectedSculptSnapshot {
    pub node_id: u64,
    pub node_name: String,
    pub current_resolution: u32,
    pub desired_resolution: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSculptSessionSnapshot {
    pub node_id: u64,
    pub node_name: String,
    pub brush_mode_id: String,
    pub brush_mode_label: String,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub symmetry_axis_id: String,
    pub symmetry_axis_label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSculptSnapshot {
    pub selected: Option<AppSelectedSculptSnapshot>,
    pub session: Option<AppSculptSessionSnapshot>,
    pub can_resume_selected: bool,
    pub can_stop: bool,
    pub max_resolution: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppLightCookieCandidateSnapshot {
    pub node_id: u64,
    pub name: String,
    pub kind_label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppLightPropertiesSnapshot {
    pub node_id: u64,
    pub transform_node_id: Option<u64>,
    pub light_type_id: String,
    pub light_type_label: String,
    pub color: AppVec3,
    pub intensity: f32,
    pub range: f32,
    pub spot_angle: f32,
    pub cast_shadows: bool,
    pub shadow_softness: f32,
    pub shadow_color: AppVec3,
    pub volumetric: bool,
    pub volumetric_density: f32,
    pub cookie_node_id: Option<u64>,
    pub cookie_node_name: Option<String>,
    pub cookie_candidates: Vec<AppLightCookieCandidateSnapshot>,
    pub proximity_mode_id: String,
    pub proximity_mode_label: String,
    pub proximity_range: f32,
    pub array_pattern_id: Option<String>,
    pub array_pattern_label: Option<String>,
    pub array_count: Option<u32>,
    pub array_radius: Option<f32>,
    pub array_color_variation: Option<f32>,
    pub intensity_expression: Option<String>,
    pub intensity_expression_error: Option<String>,
    pub color_hue_expression: Option<String>,
    pub color_hue_expression_error: Option<String>,
    pub supports_range: bool,
    pub supports_spot_angle: bool,
    pub supports_shadows: bool,
    pub supports_volumetric: bool,
    pub supports_cookie: bool,
    pub supports_proximity: bool,
    pub supports_expressions: bool,
    pub supports_array: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppViewportLightSnapshot {
    pub light_node_id: u64,
    pub transform_node_id: u64,
    pub light_type_id: String,
    pub light_type_label: String,
    pub world_position: AppVec3,
    pub direction: AppVec3,
    pub color: AppVec3,
    pub intensity: f32,
    pub range: f32,
    pub spot_angle: f32,
    pub active: bool,
    pub array_positions: Vec<AppVec3>,
    pub array_colors: Vec<AppVec3>,
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
    pub light: Option<AppLightPropertiesSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppLightLinkTargetSnapshot {
    pub light_node_id: u64,
    pub light_name: String,
    pub light_type_label: String,
    pub active: bool,
    pub mask_bit: u8,
    pub color: AppVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppLightLinkNodeSnapshot {
    pub node_id: u64,
    pub node_name: String,
    pub kind_label: String,
    pub light_mask: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppLightLinkingSnapshot {
    pub lights: Vec<AppLightLinkTargetSnapshot>,
    pub geometry_nodes: Vec<AppLightLinkNodeSnapshot>,
    pub total_visible_light_count: u32,
    pub max_light_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSceneSnapshot {
    pub selected_node: Option<AppNodeSnapshot>,
    pub selected_node_properties: Option<AppSelectedNodePropertiesSnapshot>,
    #[serde(default)]
    pub selected_node_ids: Vec<u64>,
    pub top_level_nodes: Vec<AppNodeSnapshot>,
    pub scene_tree_roots: Vec<AppSceneTreeNodeSnapshot>,
    #[serde(default)]
    pub viewport_lights: Vec<AppViewportLightSnapshot>,
    #[serde(default)]
    pub workspace: AppWorkspaceSnapshot,
    #[serde(default)]
    pub selection_context: AppSelectionContextSnapshot,
    #[serde(default)]
    pub commands: Vec<AppCommandSnapshot>,
    pub history: AppHistorySnapshot,
    pub document: AppDocumentSnapshot,
    pub render: AppRenderSettingsSnapshot,
    pub settings: AppSettingsSnapshot,
    pub export: AppExportSnapshot,
    pub import: AppImportSnapshot,
    pub sculpt_convert: AppSculptConvertSnapshot,
    pub sculpt: AppSculptSnapshot,
    pub light_linking: AppLightLinkingSnapshot,
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
