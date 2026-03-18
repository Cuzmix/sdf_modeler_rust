use sdf_modeler::app_bridge as bridge;

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppVec3))]
pub struct AppVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppCameraSnapshot))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppNodeSnapshot))]
pub struct AppNodeSnapshot {
    pub id: u64,
    pub name: String,
    pub kind_label: String,
    pub visible: bool,
    pub locked: bool,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSceneTreeNodeSnapshot
))]
pub struct AppSceneTreeNodeSnapshot {
    pub id: u64,
    pub name: String,
    pub kind_label: String,
    pub visible: bool,
    pub locked: bool,
    pub workflow_status_id: String,
    pub workflow_status_label: String,
    pub children: Vec<AppSceneTreeNodeSnapshot>,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppWorkspaceSnapshot))]
pub struct AppWorkspaceSnapshot {
    pub id: String,
    pub label: String,
    pub description: String,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppQuickActionSnapshot))]
pub struct AppQuickActionSnapshot {
    pub id: String,
    pub label: String,
    pub category: String,
    pub enabled: bool,
    pub prominent: bool,
    pub shortcut_label: Option<String>,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppCommandSnapshot))]
pub struct AppCommandSnapshot {
    pub id: String,
    pub label: String,
    pub category: String,
    pub enabled: bool,
    pub workspace_ids: Vec<String>,
    pub shortcut_label: Option<String>,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSelectionContextSnapshot
))]
pub struct AppSelectionContextSnapshot {
    pub headline: String,
    pub detail: String,
    pub selection_count: u32,
    pub selection_kind_id: String,
    pub selection_kind_label: String,
    pub workflow_status_id: String,
    pub workflow_status_label: String,
    pub quick_actions: Vec<AppQuickActionSnapshot>,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppSceneStatsSnapshot))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppToolSnapshot))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppRenderOptionSnapshot))]
pub struct AppRenderOptionSnapshot {
    pub id: String,
    pub label: String,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppRenderSettingsSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppKeyOptionSnapshot))]
pub struct AppKeyOptionSnapshot {
    pub id: String,
    pub label: String,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppKeyComboSnapshot))]
pub struct AppKeyComboSnapshot {
    pub key_id: String,
    pub key_label: String,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub shortcut_label: String,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppKeybindingSnapshot))]
pub struct AppKeybindingSnapshot {
    pub action_id: String,
    pub action_label: String,
    pub category: String,
    pub binding: Option<AppKeyComboSnapshot>,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppCameraBookmarkSnapshot
))]
pub struct AppCameraBookmarkSnapshot {
    pub slot_index: u8,
    pub saved: bool,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppShellPreferencesSnapshot
))]
pub struct AppShellPreferencesSnapshot {
    pub leading_edge_side: String,
    pub desktop_scene_pinned: bool,
    pub desktop_properties_pinned: bool,
    pub favorite_command_ids_by_workspace: std::collections::HashMap<String, Vec<String>>,
    pub preferred_drawer_tab: String,
    pub quick_wheel_hint_dismissed: bool,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppShellPreferencesUpdate
))]
pub struct AppShellPreferencesUpdate {
    pub leading_edge_side: Option<String>,
    pub desktop_scene_pinned: Option<bool>,
    pub desktop_properties_pinned: Option<bool>,
    pub favorite_command_ids_by_workspace: Option<std::collections::HashMap<String, Vec<String>>>,
    pub preferred_drawer_tab: Option<String>,
    pub quick_wheel_hint_dismissed: Option<bool>,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppSettingsSnapshot))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppHistorySnapshot))]
pub struct AppHistorySnapshot {
    pub can_undo: bool,
    pub can_redo: bool,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppDocumentSnapshot))]
pub struct AppDocumentSnapshot {
    pub current_file_path: Option<String>,
    pub current_file_name: Option<String>,
    pub has_unsaved_changes: bool,
    pub recent_files: Vec<String>,
    pub recovery_available: bool,
    pub recovery_summary: Option<String>,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppExportPresetSnapshot
))]
pub struct AppExportPresetSnapshot {
    pub name: String,
    pub resolution: u32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppExportStatusSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppExportSnapshot))]
pub struct AppExportSnapshot {
    pub resolution: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
    pub adaptive: bool,
    pub presets: Vec<AppExportPresetSnapshot>,
    pub status: AppExportStatusSnapshot,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppImportDialogSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppImportStatusSnapshot
))]
pub struct AppImportStatusSnapshot {
    pub state: String,
    pub progress: u32,
    pub total: u32,
    pub filename: Option<String>,
    pub phase_label: Option<String>,
    pub message: Option<String>,
    pub is_error: bool,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppImportSnapshot))]
pub struct AppImportSnapshot {
    pub dialog: Option<AppImportDialogSnapshot>,
    pub status: AppImportStatusSnapshot,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSculptConvertDialogSnapshot
))]
pub struct AppSculptConvertDialogSnapshot {
    pub target_node_id: u64,
    pub target_name: String,
    pub mode_id: String,
    pub mode_label: String,
    pub resolution: u32,
    pub min_resolution: u32,
    pub max_resolution: u32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSculptConvertStatusSnapshot
))]
pub struct AppSculptConvertStatusSnapshot {
    pub state: String,
    pub progress: u32,
    pub total: u32,
    pub target_name: Option<String>,
    pub phase_label: Option<String>,
    pub message: Option<String>,
    pub is_error: bool,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSculptConvertSnapshot
))]
pub struct AppSculptConvertSnapshot {
    pub dialog: Option<AppSculptConvertDialogSnapshot>,
    pub status: AppSculptConvertStatusSnapshot,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppWorkflowStatusSnapshot
))]
pub struct AppWorkflowStatusSnapshot {
    pub export_status: AppExportStatusSnapshot,
    pub import_status: AppImportStatusSnapshot,
    pub sculpt_convert_status: AppSculptConvertStatusSnapshot,
    pub scene_changed: bool,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSelectedSculptSnapshot
))]
pub struct AppSelectedSculptSnapshot {
    pub node_id: u64,
    pub node_name: String,
    pub current_resolution: u32,
    pub desired_resolution: u32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSculptSessionSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppSculptSnapshot))]
pub struct AppSculptSnapshot {
    pub selected: Option<AppSelectedSculptSnapshot>,
    pub session: Option<AppSculptSessionSnapshot>,
    pub can_resume_selected: bool,
    pub can_stop: bool,
    pub max_resolution: u32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppLightCookieCandidateSnapshot
))]
pub struct AppLightCookieCandidateSnapshot {
    pub node_id: u64,
    pub name: String,
    pub kind_label: String,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppLightPropertiesSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppViewportLightSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppScalarPropertySnapshot
))]
pub struct AppScalarPropertySnapshot {
    pub key: String,
    pub label: String,
    pub value: f32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppTransformPropertiesSnapshot
))]
pub struct AppTransformPropertiesSnapshot {
    pub position_label: String,
    pub position: AppVec3,
    pub rotation_degrees: AppVec3,
    pub scale: Option<AppVec3>,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppPrimitivePropertiesSnapshot
))]
pub struct AppPrimitivePropertiesSnapshot {
    pub primitive_kind: String,
    pub parameters: Vec<AppScalarPropertySnapshot>,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppMaterialPropertiesSnapshot
))]
pub struct AppMaterialPropertiesSnapshot {
    pub color: AppVec3,
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: AppVec3,
    pub emissive_intensity: f32,
    pub fresnel: f32,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppSelectedNodePropertiesSnapshot
))]
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

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppLightLinkTargetSnapshot
))]
pub struct AppLightLinkTargetSnapshot {
    pub light_node_id: u64,
    pub light_name: String,
    pub light_type_label: String,
    pub active: bool,
    pub mask_bit: u8,
    pub color: AppVec3,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppLightLinkNodeSnapshot
))]
pub struct AppLightLinkNodeSnapshot {
    pub node_id: u64,
    pub node_name: String,
    pub kind_label: String,
    pub light_mask: u8,
}

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppLightLinkingSnapshot
))]
pub struct AppLightLinkingSnapshot {
    pub lights: Vec<AppLightLinkTargetSnapshot>,
    pub geometry_nodes: Vec<AppLightLinkNodeSnapshot>,
    pub total_visible_light_count: u32,
    pub max_light_count: u32,
}

#[flutter_rust_bridge::frb(mirror(sdf_modeler::app_bridge::AppSceneSnapshot))]
pub struct AppSceneSnapshot {
    pub selected_node: Option<AppNodeSnapshot>,
    pub selected_node_properties: Option<AppSelectedNodePropertiesSnapshot>,
    pub selected_node_ids: Vec<u64>,
    pub top_level_nodes: Vec<AppNodeSnapshot>,
    pub scene_tree_roots: Vec<AppSceneTreeNodeSnapshot>,
    pub viewport_lights: Vec<AppViewportLightSnapshot>,
    pub workspace: AppWorkspaceSnapshot,
    pub selection_context: AppSelectionContextSnapshot,
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

#[flutter_rust_bridge::frb(mirror(
    sdf_modeler::app_bridge::AppViewportFeedbackSnapshot
))]
pub struct AppViewportFeedbackSnapshot {
    pub camera: AppCameraSnapshot,
    pub selected_node: Option<AppNodeSnapshot>,
    pub hovered_node: Option<AppNodeSnapshot>,
}

fn convert_vec<T, U: From<T>>(values: Vec<T>) -> Vec<U> {
    values.into_iter().map(Into::into).collect()
}

fn convert_opt<T, U: From<T>>(value: Option<T>) -> Option<U> {
    value.map(Into::into)
}

impl From<bridge::AppVec3> for AppVec3 {
    fn from(value: bridge::AppVec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<bridge::AppCameraSnapshot> for AppCameraSnapshot {
    fn from(value: bridge::AppCameraSnapshot) -> Self {
        Self {
            yaw: value.yaw,
            pitch: value.pitch,
            roll: value.roll,
            distance: value.distance,
            fov_degrees: value.fov_degrees,
            orthographic: value.orthographic,
            target: value.target.into(),
            eye: value.eye.into(),
        }
    }
}

impl From<bridge::AppNodeSnapshot> for AppNodeSnapshot {
    fn from(value: bridge::AppNodeSnapshot) -> Self {
        Self {
            id: value.id,
            name: value.name,
            kind_label: value.kind_label,
            visible: value.visible,
            locked: value.locked,
        }
    }
}

impl From<bridge::AppSceneTreeNodeSnapshot> for AppSceneTreeNodeSnapshot {
    fn from(value: bridge::AppSceneTreeNodeSnapshot) -> Self {
        Self {
            id: value.id,
            name: value.name,
            kind_label: value.kind_label,
            visible: value.visible,
            locked: value.locked,
            workflow_status_id: value.workflow_status_id,
            workflow_status_label: value.workflow_status_label,
            children: convert_vec(value.children),
        }
    }
}

impl From<bridge::AppWorkspaceSnapshot> for AppWorkspaceSnapshot {
    fn from(value: bridge::AppWorkspaceSnapshot) -> Self {
        Self {
            id: value.id,
            label: value.label,
            description: value.description,
        }
    }
}

impl From<bridge::AppQuickActionSnapshot> for AppQuickActionSnapshot {
    fn from(value: bridge::AppQuickActionSnapshot) -> Self {
        Self {
            id: value.id,
            label: value.label,
            category: value.category,
            enabled: value.enabled,
            prominent: value.prominent,
            shortcut_label: value.shortcut_label,
        }
    }
}

impl From<bridge::AppCommandSnapshot> for AppCommandSnapshot {
    fn from(value: bridge::AppCommandSnapshot) -> Self {
        Self {
            id: value.id,
            label: value.label,
            category: value.category,
            enabled: value.enabled,
            workspace_ids: value.workspace_ids,
            shortcut_label: value.shortcut_label,
        }
    }
}

impl From<bridge::AppSelectionContextSnapshot> for AppSelectionContextSnapshot {
    fn from(value: bridge::AppSelectionContextSnapshot) -> Self {
        Self {
            headline: value.headline,
            detail: value.detail,
            selection_count: value.selection_count,
            selection_kind_id: value.selection_kind_id,
            selection_kind_label: value.selection_kind_label,
            workflow_status_id: value.workflow_status_id,
            workflow_status_label: value.workflow_status_label,
            quick_actions: convert_vec(value.quick_actions),
        }
    }
}

impl From<bridge::AppSceneStatsSnapshot> for AppSceneStatsSnapshot {
    fn from(value: bridge::AppSceneStatsSnapshot) -> Self {
        Self {
            total_nodes: value.total_nodes,
            visible_nodes: value.visible_nodes,
            top_level_nodes: value.top_level_nodes,
            primitive_nodes: value.primitive_nodes,
            operation_nodes: value.operation_nodes,
            transform_nodes: value.transform_nodes,
            modifier_nodes: value.modifier_nodes,
            sculpt_nodes: value.sculpt_nodes,
            light_nodes: value.light_nodes,
            voxel_memory_bytes: value.voxel_memory_bytes,
            sdf_eval_complexity: value.sdf_eval_complexity,
            structure_key: value.structure_key,
            data_fingerprint: value.data_fingerprint,
            bounds_min: value.bounds_min.into(),
            bounds_max: value.bounds_max.into(),
        }
    }
}

impl From<bridge::AppToolSnapshot> for AppToolSnapshot {
    fn from(value: bridge::AppToolSnapshot) -> Self {
        Self {
            active_tool_label: value.active_tool_label,
            shading_mode_label: value.shading_mode_label,
            grid_enabled: value.grid_enabled,
            manipulator_mode_id: value.manipulator_mode_id,
            manipulator_mode_label: value.manipulator_mode_label,
            manipulator_space_id: value.manipulator_space_id,
            manipulator_space_label: value.manipulator_space_label,
            manipulator_visible: value.manipulator_visible,
            can_reset_pivot: value.can_reset_pivot,
            pivot_offset: value.pivot_offset.into(),
        }
    }
}

impl From<bridge::AppRenderOptionSnapshot> for AppRenderOptionSnapshot {
    fn from(value: bridge::AppRenderOptionSnapshot) -> Self {
        Self {
            id: value.id,
            label: value.label,
        }
    }
}

impl From<bridge::AppRenderSettingsSnapshot> for AppRenderSettingsSnapshot {
    fn from(value: bridge::AppRenderSettingsSnapshot) -> Self {
        Self {
            shading_modes: convert_vec(value.shading_modes),
            shading_mode_id: value.shading_mode_id,
            shading_mode_label: value.shading_mode_label,
            show_grid: value.show_grid,
            shadows_enabled: value.shadows_enabled,
            shadow_steps: value.shadow_steps,
            ao_enabled: value.ao_enabled,
            ao_samples: value.ao_samples,
            ao_intensity: value.ao_intensity,
            march_max_steps: value.march_max_steps,
            sculpt_fast_mode: value.sculpt_fast_mode,
            auto_reduce_steps: value.auto_reduce_steps,
            interaction_render_scale: value.interaction_render_scale,
            rest_render_scale: value.rest_render_scale,
            fog_enabled: value.fog_enabled,
            fog_density: value.fog_density,
            bloom_enabled: value.bloom_enabled,
            bloom_intensity: value.bloom_intensity,
            gamma: value.gamma,
            tonemapping_aces: value.tonemapping_aces,
            cross_section_axis: value.cross_section_axis,
            cross_section_position: value.cross_section_position,
        }
    }
}

impl From<bridge::AppKeyOptionSnapshot> for AppKeyOptionSnapshot {
    fn from(value: bridge::AppKeyOptionSnapshot) -> Self {
        Self {
            id: value.id,
            label: value.label,
        }
    }
}

impl From<bridge::AppKeyComboSnapshot> for AppKeyComboSnapshot {
    fn from(value: bridge::AppKeyComboSnapshot) -> Self {
        Self {
            key_id: value.key_id,
            key_label: value.key_label,
            ctrl: value.ctrl,
            shift: value.shift,
            alt: value.alt,
            shortcut_label: value.shortcut_label,
        }
    }
}

impl From<bridge::AppKeybindingSnapshot> for AppKeybindingSnapshot {
    fn from(value: bridge::AppKeybindingSnapshot) -> Self {
        Self {
            action_id: value.action_id,
            action_label: value.action_label,
            category: value.category,
            binding: convert_opt(value.binding),
        }
    }
}

impl From<bridge::AppCameraBookmarkSnapshot> for AppCameraBookmarkSnapshot {
    fn from(value: bridge::AppCameraBookmarkSnapshot) -> Self {
        Self {
            slot_index: value.slot_index,
            saved: value.saved,
        }
    }
}

impl From<bridge::AppShellPreferencesSnapshot> for AppShellPreferencesSnapshot {
    fn from(value: bridge::AppShellPreferencesSnapshot) -> Self {
        Self {
            leading_edge_side: value.leading_edge_side,
            desktop_scene_pinned: value.desktop_scene_pinned,
            desktop_properties_pinned: value.desktop_properties_pinned,
            favorite_command_ids_by_workspace: value.favorite_command_ids_by_workspace,
            preferred_drawer_tab: value.preferred_drawer_tab,
            quick_wheel_hint_dismissed: value.quick_wheel_hint_dismissed,
        }
    }
}

impl From<AppShellPreferencesUpdate> for bridge::AppShellPreferencesUpdate {
    fn from(value: AppShellPreferencesUpdate) -> Self {
        Self {
            leading_edge_side: value.leading_edge_side,
            desktop_scene_pinned: value.desktop_scene_pinned,
            desktop_properties_pinned: value.desktop_properties_pinned,
            favorite_command_ids_by_workspace: value.favorite_command_ids_by_workspace,
            preferred_drawer_tab: value.preferred_drawer_tab,
            quick_wheel_hint_dismissed: value.quick_wheel_hint_dismissed,
        }
    }
}

impl From<bridge::AppSettingsSnapshot> for AppSettingsSnapshot {
    fn from(value: bridge::AppSettingsSnapshot) -> Self {
        Self {
            show_fps_overlay: value.show_fps_overlay,
            show_node_labels: value.show_node_labels,
            show_bounding_box: value.show_bounding_box,
            show_light_gizmos: value.show_light_gizmos,
            auto_save_enabled: value.auto_save_enabled,
            auto_save_interval_secs: value.auto_save_interval_secs,
            max_export_resolution: value.max_export_resolution,
            max_sculpt_resolution: value.max_sculpt_resolution,
            camera_bookmarks: convert_vec(value.camera_bookmarks),
            shell_preferences: value.shell_preferences.into(),
            key_options: convert_vec(value.key_options),
            keybindings: convert_vec(value.keybindings),
        }
    }
}

impl From<bridge::AppHistorySnapshot> for AppHistorySnapshot {
    fn from(value: bridge::AppHistorySnapshot) -> Self {
        Self {
            can_undo: value.can_undo,
            can_redo: value.can_redo,
        }
    }
}

impl From<bridge::AppDocumentSnapshot> for AppDocumentSnapshot {
    fn from(value: bridge::AppDocumentSnapshot) -> Self {
        Self {
            current_file_path: value.current_file_path,
            current_file_name: value.current_file_name,
            has_unsaved_changes: value.has_unsaved_changes,
            recent_files: value.recent_files,
            recovery_available: value.recovery_available,
            recovery_summary: value.recovery_summary,
        }
    }
}

impl From<bridge::AppExportPresetSnapshot> for AppExportPresetSnapshot {
    fn from(value: bridge::AppExportPresetSnapshot) -> Self {
        Self {
            name: value.name,
            resolution: value.resolution,
        }
    }
}

impl From<bridge::AppExportStatusSnapshot> for AppExportStatusSnapshot {
    fn from(value: bridge::AppExportStatusSnapshot) -> Self {
        Self {
            state: value.state,
            progress: value.progress,
            total: value.total,
            resolution: value.resolution,
            phase_label: value.phase_label,
            target_file_name: value.target_file_name,
            target_file_path: value.target_file_path,
            format_label: value.format_label,
            message: value.message,
            is_error: value.is_error,
        }
    }
}

impl From<bridge::AppExportSnapshot> for AppExportSnapshot {
    fn from(value: bridge::AppExportSnapshot) -> Self {
        Self {
            resolution: value.resolution,
            min_resolution: value.min_resolution,
            max_resolution: value.max_resolution,
            adaptive: value.adaptive,
            presets: convert_vec(value.presets),
            status: value.status.into(),
        }
    }
}

impl From<bridge::AppImportDialogSnapshot> for AppImportDialogSnapshot {
    fn from(value: bridge::AppImportDialogSnapshot) -> Self {
        Self {
            filename: value.filename,
            resolution: value.resolution,
            auto_resolution: value.auto_resolution,
            use_auto: value.use_auto,
            vertex_count: value.vertex_count,
            triangle_count: value.triangle_count,
            bounds_size: value.bounds_size.into(),
            min_resolution: value.min_resolution,
            max_resolution: value.max_resolution,
        }
    }
}

impl From<bridge::AppImportStatusSnapshot> for AppImportStatusSnapshot {
    fn from(value: bridge::AppImportStatusSnapshot) -> Self {
        Self {
            state: value.state,
            progress: value.progress,
            total: value.total,
            filename: value.filename,
            phase_label: value.phase_label,
            message: value.message,
            is_error: value.is_error,
        }
    }
}

impl From<bridge::AppImportSnapshot> for AppImportSnapshot {
    fn from(value: bridge::AppImportSnapshot) -> Self {
        Self {
            dialog: convert_opt(value.dialog),
            status: value.status.into(),
        }
    }
}

impl From<bridge::AppSculptConvertDialogSnapshot> for AppSculptConvertDialogSnapshot {
    fn from(value: bridge::AppSculptConvertDialogSnapshot) -> Self {
        Self {
            target_node_id: value.target_node_id,
            target_name: value.target_name,
            mode_id: value.mode_id,
            mode_label: value.mode_label,
            resolution: value.resolution,
            min_resolution: value.min_resolution,
            max_resolution: value.max_resolution,
        }
    }
}

impl From<bridge::AppSculptConvertStatusSnapshot> for AppSculptConvertStatusSnapshot {
    fn from(value: bridge::AppSculptConvertStatusSnapshot) -> Self {
        Self {
            state: value.state,
            progress: value.progress,
            total: value.total,
            target_name: value.target_name,
            phase_label: value.phase_label,
            message: value.message,
            is_error: value.is_error,
        }
    }
}

impl From<bridge::AppSculptConvertSnapshot> for AppSculptConvertSnapshot {
    fn from(value: bridge::AppSculptConvertSnapshot) -> Self {
        Self {
            dialog: convert_opt(value.dialog),
            status: value.status.into(),
        }
    }
}

impl From<bridge::AppWorkflowStatusSnapshot> for AppWorkflowStatusSnapshot {
    fn from(value: bridge::AppWorkflowStatusSnapshot) -> Self {
        Self {
            export_status: value.export_status.into(),
            import_status: value.import_status.into(),
            sculpt_convert_status: value.sculpt_convert_status.into(),
            scene_changed: value.scene_changed,
        }
    }
}

impl From<bridge::AppSelectedSculptSnapshot> for AppSelectedSculptSnapshot {
    fn from(value: bridge::AppSelectedSculptSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            node_name: value.node_name,
            current_resolution: value.current_resolution,
            desired_resolution: value.desired_resolution,
        }
    }
}

impl From<bridge::AppSculptSessionSnapshot> for AppSculptSessionSnapshot {
    fn from(value: bridge::AppSculptSessionSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            node_name: value.node_name,
            brush_mode_id: value.brush_mode_id,
            brush_mode_label: value.brush_mode_label,
            brush_radius: value.brush_radius,
            brush_strength: value.brush_strength,
            symmetry_axis_id: value.symmetry_axis_id,
            symmetry_axis_label: value.symmetry_axis_label,
        }
    }
}

impl From<bridge::AppSculptSnapshot> for AppSculptSnapshot {
    fn from(value: bridge::AppSculptSnapshot) -> Self {
        Self {
            selected: convert_opt(value.selected),
            session: convert_opt(value.session),
            can_resume_selected: value.can_resume_selected,
            can_stop: value.can_stop,
            max_resolution: value.max_resolution,
        }
    }
}

impl From<bridge::AppLightCookieCandidateSnapshot> for AppLightCookieCandidateSnapshot {
    fn from(value: bridge::AppLightCookieCandidateSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            name: value.name,
            kind_label: value.kind_label,
        }
    }
}

impl From<bridge::AppLightPropertiesSnapshot> for AppLightPropertiesSnapshot {
    fn from(value: bridge::AppLightPropertiesSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            transform_node_id: value.transform_node_id,
            light_type_id: value.light_type_id,
            light_type_label: value.light_type_label,
            color: value.color.into(),
            intensity: value.intensity,
            range: value.range,
            spot_angle: value.spot_angle,
            cast_shadows: value.cast_shadows,
            shadow_softness: value.shadow_softness,
            shadow_color: value.shadow_color.into(),
            volumetric: value.volumetric,
            volumetric_density: value.volumetric_density,
            cookie_node_id: value.cookie_node_id,
            cookie_node_name: value.cookie_node_name,
            cookie_candidates: convert_vec(value.cookie_candidates),
            proximity_mode_id: value.proximity_mode_id,
            proximity_mode_label: value.proximity_mode_label,
            proximity_range: value.proximity_range,
            array_pattern_id: value.array_pattern_id,
            array_pattern_label: value.array_pattern_label,
            array_count: value.array_count,
            array_radius: value.array_radius,
            array_color_variation: value.array_color_variation,
            intensity_expression: value.intensity_expression,
            intensity_expression_error: value.intensity_expression_error,
            color_hue_expression: value.color_hue_expression,
            color_hue_expression_error: value.color_hue_expression_error,
            supports_range: value.supports_range,
            supports_spot_angle: value.supports_spot_angle,
            supports_shadows: value.supports_shadows,
            supports_volumetric: value.supports_volumetric,
            supports_cookie: value.supports_cookie,
            supports_proximity: value.supports_proximity,
            supports_expressions: value.supports_expressions,
            supports_array: value.supports_array,
        }
    }
}

impl From<bridge::AppScalarPropertySnapshot> for AppScalarPropertySnapshot {
    fn from(value: bridge::AppScalarPropertySnapshot) -> Self {
        Self {
            key: value.key,
            label: value.label,
            value: value.value,
        }
    }
}

impl From<bridge::AppTransformPropertiesSnapshot> for AppTransformPropertiesSnapshot {
    fn from(value: bridge::AppTransformPropertiesSnapshot) -> Self {
        Self {
            position_label: value.position_label,
            position: value.position.into(),
            rotation_degrees: value.rotation_degrees.into(),
            scale: convert_opt(value.scale),
        }
    }
}

impl From<bridge::AppPrimitivePropertiesSnapshot> for AppPrimitivePropertiesSnapshot {
    fn from(value: bridge::AppPrimitivePropertiesSnapshot) -> Self {
        Self {
            primitive_kind: value.primitive_kind,
            parameters: convert_vec(value.parameters),
        }
    }
}

impl From<bridge::AppMaterialPropertiesSnapshot> for AppMaterialPropertiesSnapshot {
    fn from(value: bridge::AppMaterialPropertiesSnapshot) -> Self {
        Self {
            color: value.color.into(),
            roughness: value.roughness,
            metallic: value.metallic,
            emissive: value.emissive.into(),
            emissive_intensity: value.emissive_intensity,
            fresnel: value.fresnel,
        }
    }
}

impl From<bridge::AppViewportLightSnapshot> for AppViewportLightSnapshot {
    fn from(value: bridge::AppViewportLightSnapshot) -> Self {
        Self {
            light_node_id: value.light_node_id,
            transform_node_id: value.transform_node_id,
            light_type_id: value.light_type_id,
            light_type_label: value.light_type_label,
            world_position: value.world_position.into(),
            direction: value.direction.into(),
            color: value.color.into(),
            intensity: value.intensity,
            range: value.range,
            spot_angle: value.spot_angle,
            active: value.active,
            array_positions: convert_vec(value.array_positions),
            array_colors: convert_vec(value.array_colors),
        }
    }
}

impl From<bridge::AppSelectedNodePropertiesSnapshot> for AppSelectedNodePropertiesSnapshot {
    fn from(value: bridge::AppSelectedNodePropertiesSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            name: value.name,
            kind_label: value.kind_label,
            visible: value.visible,
            locked: value.locked,
            transform: convert_opt(value.transform),
            primitive: convert_opt(value.primitive),
            material: convert_opt(value.material),
            light: convert_opt(value.light),
        }
    }
}

impl From<bridge::AppLightLinkTargetSnapshot> for AppLightLinkTargetSnapshot {
    fn from(value: bridge::AppLightLinkTargetSnapshot) -> Self {
        Self {
            light_node_id: value.light_node_id,
            light_name: value.light_name,
            light_type_label: value.light_type_label,
            active: value.active,
            mask_bit: value.mask_bit,
            color: value.color.into(),
        }
    }
}

impl From<bridge::AppLightLinkNodeSnapshot> for AppLightLinkNodeSnapshot {
    fn from(value: bridge::AppLightLinkNodeSnapshot) -> Self {
        Self {
            node_id: value.node_id,
            node_name: value.node_name,
            kind_label: value.kind_label,
            light_mask: value.light_mask,
        }
    }
}

impl From<bridge::AppLightLinkingSnapshot> for AppLightLinkingSnapshot {
    fn from(value: bridge::AppLightLinkingSnapshot) -> Self {
        Self {
            lights: convert_vec(value.lights),
            geometry_nodes: convert_vec(value.geometry_nodes),
            total_visible_light_count: value.total_visible_light_count,
            max_light_count: value.max_light_count,
        }
    }
}

impl From<bridge::AppSceneSnapshot> for AppSceneSnapshot {
    fn from(value: bridge::AppSceneSnapshot) -> Self {
        Self {
            selected_node: convert_opt(value.selected_node),
            selected_node_properties: convert_opt(value.selected_node_properties),
            selected_node_ids: value.selected_node_ids,
            top_level_nodes: convert_vec(value.top_level_nodes),
            scene_tree_roots: convert_vec(value.scene_tree_roots),
            viewport_lights: convert_vec(value.viewport_lights),
            workspace: value.workspace.into(),
            selection_context: value.selection_context.into(),
            commands: convert_vec(value.commands),
            history: value.history.into(),
            document: value.document.into(),
            render: value.render.into(),
            settings: value.settings.into(),
            export: value.export.into(),
            import: value.import.into(),
            sculpt_convert: value.sculpt_convert.into(),
            sculpt: value.sculpt.into(),
            light_linking: value.light_linking.into(),
            camera: value.camera.into(),
            stats: value.stats.into(),
            tool: value.tool.into(),
        }
    }
}

impl From<bridge::AppViewportFeedbackSnapshot> for AppViewportFeedbackSnapshot {
    fn from(value: bridge::AppViewportFeedbackSnapshot) -> Self {
        Self {
            camera: value.camera.into(),
            selected_node: convert_opt(value.selected_node),
            hovered_node: convert_opt(value.hovered_node),
        }
    }
}
