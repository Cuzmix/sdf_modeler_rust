use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use glam::Vec3;

use crate::compat::{Duration, Instant};
use crate::gpu::camera::Camera;
use crate::graph::history::History;
use crate::gpu::buffers::identify_active_lights;
use crate::graph::scene::{
    ArrayPattern, CsgOp, LightType, ModifierKind, NodeData, NodeId, ProximityMode, Scene,
    SceneNode, SdfPrimitive, MAX_SCENE_LIGHTS,
};
use crate::graph::voxel::create_displacement_grid_for_subtree;
use crate::sculpt::{BrushMode, SculptState, DEFAULT_BRUSH_STRENGTH};
use crate::settings::{RenderConfig, Settings};

use super::dto::{
    AppCameraSnapshot, AppDocumentSnapshot, AppExportPresetSnapshot, AppExportSnapshot,
    AppExportStatusSnapshot, AppHistorySnapshot, AppImportDialogSnapshot, AppImportSnapshot,
    AppImportStatusSnapshot, AppLightCookieCandidateSnapshot, AppLightLinkingSnapshot,
    AppLightLinkNodeSnapshot, AppLightLinkTargetSnapshot, AppLightPropertiesSnapshot,
    AppMaterialPropertiesSnapshot, AppNodeSnapshot, AppPrimitivePropertiesSnapshot,
    AppScalarPropertySnapshot, AppSceneSnapshot, AppSceneStatsSnapshot,
    AppSceneTreeNodeSnapshot, AppSculptConvertDialogSnapshot, AppSculptConvertSnapshot,
    AppSculptConvertStatusSnapshot, AppSculptSessionSnapshot, AppSculptSnapshot,
    AppSelectedNodePropertiesSnapshot, AppSelectedSculptSnapshot, AppToolSnapshot,
    AppTransformPropertiesSnapshot, AppVec3, AppViewportFeedbackSnapshot,
};
use super::renderer::{HeadlessPickRequest, HeadlessRenderRequest, HeadlessViewportRenderer};
use super::workflows::{
    ImportDialogState, SculptConvertDialogState, SculptConvertMode, WorkflowMessage,
};

pub struct RenderedViewportFrame {
    pub pixels: Vec<u8>,
    pub camera_animating: bool,
}

pub struct AppBridge {
    scene: Scene,
    camera: Camera,
    render_config: RenderConfig,
    settings: Settings,
    history: History,
    persistence: DocumentPersistence,
    selected_node: Option<NodeId>,
    hovered_node: Option<NodeId>,
    export_state: ExportState,
    import_state: ImportState,
    sculpt_convert_state: SculptConvertState,
    sculpt_state: SculptState,
    active_tool_label: String,
    manipulator_mode: ManipulatorMode,
    manipulator_space: ManipulatorSpace,
    pivot_offset: Vec3,
    renderer: HeadlessViewportRenderer,
    last_viewport_time_seconds: Option<f32>,
}

struct DocumentPersistence {
    current_file_path: Option<PathBuf>,
    saved_fingerprint: u64,
    scene_dirty: bool,
    last_auto_save: Instant,
    recovery_prompt_visible: bool,
    recovery_summary: Option<String>,
}

struct ExportState {
    task: ExportTask,
    last_message: Option<WorkflowMessage>,
}

enum ExportTask {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        total: u32,
        resolution: u32,
        receiver: std::sync::mpsc::Receiver<Option<crate::export::ExportMesh>>,
        path: PathBuf,
        cancelled: Arc<AtomicBool>,
    },
}

struct ImportState {
    dialog: Option<ImportDialogState>,
    task: ImportTask,
    last_message: Option<WorkflowMessage>,
}

enum ImportTask {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        total: u32,
        filename: String,
        receiver: std::sync::mpsc::Receiver<(crate::graph::voxel::VoxelGrid, Vec3)>,
        cancelled: Arc<AtomicBool>,
    },
}

struct SculptConvertState {
    dialog: Option<SculptConvertDialogState>,
    task: SculptConvertTask,
    last_message: Option<WorkflowMessage>,
}

enum SculptConvertTask {
    Idle,
    InProgress {
        progress: Arc<AtomicU32>,
        total: u32,
        target_name: String,
        receiver: std::sync::mpsc::Receiver<(crate::graph::voxel::VoxelGrid, Vec3)>,
        subtree_root: NodeId,
        color: Vec3,
        flatten: bool,
    },
}

struct LightLinkTarget {
    light_node_id: NodeId,
    light_name: String,
    light_type_label: String,
    active: bool,
    mask_bit: u8,
    color: Vec3,
}

const DEFAULT_SCULPT_ENTRY_RESOLUTION: u32 = 64;
const PRIMITIVE_PARAMETER_MIN: f32 = 0.01;
const PRIMITIVE_PARAMETER_MAX: f32 = 100.0;
const MATERIAL_FACTOR_MIN: f32 = 0.0;
const MATERIAL_FACTOR_MAX: f32 = 1.0;
const EMISSIVE_INTENSITY_MAX: f32 = 5.0;
const LIGHT_INTENSITY_MIN: f32 = -10.0;
const LIGHT_INTENSITY_MAX: f32 = 10.0;
const LIGHT_RANGE_MIN: f32 = 0.1;
const LIGHT_RANGE_MAX: f32 = 50.0;
const LIGHT_SPOT_ANGLE_MIN: f32 = 1.0;
const LIGHT_SPOT_ANGLE_MAX: f32 = 179.0;
const LIGHT_SHADOW_SOFTNESS_MIN: f32 = 1.0;
const LIGHT_SHADOW_SOFTNESS_MAX: f32 = 64.0;
const LIGHT_VOLUMETRIC_DENSITY_MIN: f32 = 0.01;
const LIGHT_VOLUMETRIC_DENSITY_MAX: f32 = 1.0;
const LIGHT_PROXIMITY_RANGE_MIN: f32 = 0.1;
const LIGHT_PROXIMITY_RANGE_MAX: f32 = 10.0;
const LIGHT_ARRAY_COUNT_MIN: u32 = 2;
const LIGHT_ARRAY_COUNT_MAX: u32 = 32;
const LIGHT_ARRAY_RADIUS_MIN: f32 = 0.1;
const LIGHT_ARRAY_RADIUS_MAX: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManipulatorMode {
    Translate,
    Rotate,
    Scale,
}

impl ManipulatorMode {
    fn id(self) -> &'static str {
        match self {
            Self::Translate => "translate",
            Self::Rotate => "rotate",
            Self::Scale => "scale",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Translate => "Move",
            Self::Rotate => "Rotate",
            Self::Scale => "Scale",
        }
    }

    fn from_id(mode_id: &str) -> Option<Self> {
        Some(match mode_id {
            "translate" => Self::Translate,
            "rotate" => Self::Rotate,
            "scale" => Self::Scale,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManipulatorSpace {
    Local,
    World,
}

impl ManipulatorSpace {
    fn id(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::World => "world",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Local => "Local",
            Self::World => "World",
        }
    }

    fn toggled(self) -> Self {
        match self {
            Self::Local => Self::World,
            Self::World => Self::Local,
        }
    }
}

impl Default for AppBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBridge {
    pub fn new() -> Self {
        let scene = Scene::new();
        let camera = Camera::default();
        let settings = Settings::load();
        let render_config = RenderConfig::default();
        let renderer = HeadlessViewportRenderer::new(&scene, &render_config);
        let initial_saved_fingerprint = scene.data_fingerprint();
        let recovery_summary = if crate::io::has_recovery_file() {
            Some(Self::recovery_summary_from_meta(
                crate::io::read_recovery_meta().as_ref(),
            ))
        } else {
            None
        };

        Self {
            scene,
            camera,
            render_config,
            settings,
            history: History::new(),
            persistence: DocumentPersistence {
                current_file_path: None,
                saved_fingerprint: initial_saved_fingerprint,
                scene_dirty: false,
                last_auto_save: Instant::now(),
                recovery_prompt_visible: recovery_summary.is_some(),
                recovery_summary,
            },
            selected_node: None,
            hovered_node: None,
            export_state: ExportState {
                task: ExportTask::Idle,
                last_message: None,
            },
            import_state: ImportState {
                dialog: None,
                task: ImportTask::Idle,
                last_message: None,
            },
            sculpt_convert_state: SculptConvertState {
                dialog: None,
                task: SculptConvertTask::Idle,
                last_message: None,
            },
            sculpt_state: SculptState::Inactive,
            active_tool_label: "Select".to_string(),
            manipulator_mode: ManipulatorMode::Translate,
            manipulator_space: ManipulatorSpace::Local,
            pivot_offset: Vec3::ZERO,
            renderer,
            last_viewport_time_seconds: None,
        }
    }

    pub fn scene_snapshot(&self) -> AppSceneSnapshot {
        let top_level_nodes = self
            .scene
            .top_level_nodes()
            .into_iter()
            .filter_map(|node_id| self.scene.nodes.get(&node_id))
            .map(|node| self.node_snapshot(node))
            .collect();

        let node_counts = self.scene.node_type_counts();
        let bounds = self.scene.compute_bounds();

        AppSceneSnapshot {
            selected_node: self.node_snapshot_by_id(self.selected_node),
            selected_node_properties: self.selected_node_properties_snapshot(),
            top_level_nodes,
            scene_tree_roots: self.scene_tree_roots(),
            history: AppHistorySnapshot {
                can_undo: self.history.undo_count() > 0,
                can_redo: self.history.redo_count() > 0,
            },
            document: self.document_snapshot(),
            export: self.export_snapshot(),
            import: self.import_snapshot(),
            sculpt_convert: self.sculpt_convert_snapshot(),
            sculpt: self.sculpt_snapshot(),
            light_linking: self.light_linking_snapshot(),
            camera: self.camera_snapshot(),
            stats: AppSceneStatsSnapshot {
                total_nodes: node_counts.total as u32,
                visible_nodes: node_counts.visible as u32,
                top_level_nodes: self.scene.top_level_nodes().len() as u32,
                primitive_nodes: node_counts.primitives as u32,
                operation_nodes: node_counts.operations as u32,
                transform_nodes: node_counts.transforms as u32,
                modifier_nodes: node_counts.modifiers as u32,
                sculpt_nodes: node_counts.sculpts as u32,
                light_nodes: node_counts.lights as u32,
                voxel_memory_bytes: self.scene.voxel_memory_bytes() as u64,
                sdf_eval_complexity: self.scene.sdf_eval_complexity() as u32,
                structure_key: self.scene.structure_key(),
                data_fingerprint: self.scene.data_fingerprint(),
                bounds_min: AppVec3::new(bounds.0[0], bounds.0[1], bounds.0[2]),
                bounds_max: AppVec3::new(bounds.1[0], bounds.1[1], bounds.1[2]),
            },
            tool: AppToolSnapshot {
                active_tool_label: self.active_tool_label.clone(),
                shading_mode_label: self.render_config.shading_mode.label().to_string(),
                grid_enabled: self.render_config.show_grid,
                manipulator_mode_id: self.manipulator_mode.id().to_string(),
                manipulator_mode_label: self.manipulator_mode.label().to_string(),
                manipulator_space_id: self.manipulator_space.id().to_string(),
                manipulator_space_label: self.manipulator_space.label().to_string(),
                manipulator_visible: self.selected_node.is_some(),
                can_reset_pivot: self.pivot_offset.length_squared() > 0.0,
                pivot_offset: AppVec3::new(
                    self.pivot_offset.x,
                    self.pivot_offset.y,
                    self.pivot_offset.z,
                ),
            },
        }
    }

    pub fn viewport_feedback(&self) -> AppViewportFeedbackSnapshot {
        AppViewportFeedbackSnapshot {
            camera: self.camera_snapshot(),
            selected_node: self.node_snapshot_by_id(self.selected_node),
            hovered_node: self.node_snapshot_by_id(self.hovered_node),
        }
    }

    pub fn refresh_background_state(&mut self) {
        self.poll_export();
        self.poll_import();
        self.poll_sculpt_convert();
    }

    pub fn render_viewport_frame(
        &mut self,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> RenderedViewportFrame {
        let camera_animating = self.tick_camera_animation(time_seconds);
        let pixels = self.renderer.render_scene(HeadlessRenderRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            selected_node: self.selected_node,
            time_seconds,
            width,
            height,
        });

        RenderedViewportFrame {
            pixels,
            camera_animating,
        }
    }

    pub fn orbit_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.cancel_camera_transition();
        self.camera.orbit(delta_x, delta_y);
        self.camera.clamp_pitch();
    }

    pub fn pan_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.cancel_camera_transition();
        self.camera.pan(delta_x, delta_y);
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.cancel_camera_transition();
        self.camera.zoom(delta);
    }

    pub fn hover_node_at_viewport(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> Option<u64> {
        let next_hovered_node = self.renderer.pick_node(HeadlessPickRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            time_seconds,
            width,
            height,
            mouse_x,
            mouse_y,
        });
        self.hovered_node = next_hovered_node;
        next_hovered_node
    }

    pub fn clear_hovered_node(&mut self) {
        self.hovered_node = None;
    }

    pub fn select_node_at_viewport(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        width: u32,
        height: u32,
        time_seconds: f32,
    ) -> Option<u64> {
        let next_selected_node = self.renderer.pick_node(HeadlessPickRequest {
            scene: &self.scene,
            camera: &self.camera,
            render_config: &self.render_config,
            time_seconds,
            width,
            height,
            mouse_x,
            mouse_y,
        });
        self.selected_node = next_selected_node;
        self.hovered_node = next_selected_node;
        next_selected_node
    }

    pub fn focus_selected(&mut self) {
        let Some(selected_node) = self.selected_node else {
            return;
        };

        self.cancel_camera_transition();
        let parent_map = self.scene.build_parent_map();
        let (center, radius) = self
            .scene
            .compute_subtree_sphere(selected_node, &parent_map);
        self.camera
            .focus_on(Vec3::new(center[0], center[1], center[2]), radius.max(0.5));
    }

    pub fn frame_all(&mut self) {
        self.cancel_camera_transition();
        let bounds = self.scene.compute_bounds();
        let bounds_min = Vec3::new(bounds.0[0], bounds.0[1], bounds.0[2]);
        let bounds_max = Vec3::new(bounds.1[0], bounds.1[1], bounds.1[2]);
        let center = (bounds_min + bounds_max) * 0.5;
        let radius = ((bounds_max - bounds_min) * 0.5).length().max(0.5);
        self.camera.focus_on(center, radius);
    }

    pub fn camera_front(&mut self) {
        self.start_camera_view_transition(0.0, 0.0, 0.0);
    }

    pub fn camera_top(&mut self) {
        self.start_camera_view_transition(0.0, std::f32::consts::FRAC_PI_2, 0.0);
    }

    pub fn camera_right(&mut self) {
        self.start_camera_view_transition(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    }

    pub fn camera_back(&mut self) {
        self.start_camera_view_transition(std::f32::consts::PI, 0.0, 0.0);
    }

    pub fn camera_left(&mut self) {
        self.start_camera_view_transition(-std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    }

    pub fn camera_bottom(&mut self) {
        self.start_camera_view_transition(0.0, -std::f32::consts::FRAC_PI_2, 0.0);
    }

    pub fn toggle_orthographic(&mut self) {
        self.cancel_camera_transition();
        self.camera.toggle_ortho();
    }

    pub fn set_manipulator_mode(&mut self, mode_id: &str) -> bool {
        let Some(mode) = ManipulatorMode::from_id(mode_id) else {
            return false;
        };

        self.manipulator_mode = mode;
        true
    }

    pub fn toggle_manipulator_space(&mut self) {
        self.manipulator_space = self.manipulator_space.toggled();
    }

    pub fn nudge_manipulator_pivot_offset(&mut self, x: f32, y: f32, z: f32) {
        self.pivot_offset += Vec3::new(x, y, z);
    }

    pub fn reset_manipulator_pivot(&mut self) {
        self.pivot_offset = Vec3::ZERO;
    }

    pub fn new_scene(&mut self) {
        self.scene = Scene::new();
        self.camera = Camera::default();
        self.history = History::new();
        self.selected_node = None;
        self.hovered_node = None;
        self.last_viewport_time_seconds = None;
        self.clear_workflow_state();
        self.persistence.current_file_path = None;
        self.persistence.saved_fingerprint = self.scene.data_fingerprint();
        self.persistence.scene_dirty = false;
        self.clear_recovery_prompt();
    }

    pub fn open_scene(&mut self) -> bool {
        let Some(path) = crate::io::open_dialog() else {
            return false;
        };

        self.open_scene_from_path(&path)
    }

    pub fn open_recent_scene(&mut self, recent_path: &str) -> bool {
        let path = PathBuf::from(recent_path);
        if self.open_scene_from_path(&path) {
            true
        } else {
            self.settings
                .recent_files
                .retain(|path_entry| path_entry != recent_path);
            self.settings.save();
            false
        }
    }

    pub fn save_scene(&mut self) -> bool {
        let Some(path) = self
            .persistence
            .current_file_path
            .clone()
            .or_else(crate::io::save_dialog)
        else {
            return false;
        };

        self.save_scene_to_path(&path)
    }

    pub fn save_scene_as(&mut self) -> bool {
        let Some(path) = crate::io::save_dialog() else {
            return false;
        };

        self.save_scene_to_path(&path)
    }

    pub fn recover_autosave(&mut self) -> bool {
        let autosave_path = crate::io::auto_save_path();
        let Ok(project) = crate::io::load_project(&autosave_path) else {
            return false;
        };

        self.scene = project.scene;
        self.camera = project.camera;
        self.history = History::new();
        self.selected_node = None;
        self.hovered_node = None;
        self.last_viewport_time_seconds = None;
        self.clear_workflow_state();
        self.persistence.current_file_path = None;
        self.persistence.saved_fingerprint = 0;
        self.persistence.scene_dirty = true;
        self.clear_recovery_prompt();
        true
    }

    pub fn discard_recovery(&mut self) -> bool {
        if crate::io::remove_recovery_files().is_err() {
            return false;
        }

        self.clear_recovery_prompt();
        true
    }

    pub fn open_import_dialog(&mut self) -> bool {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Import Mesh")
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("STL Binary", &["stl"])
            .add_filter("All Mesh Files", &["obj", "stl"])
            .pick_file()
        else {
            return false;
        };

        let mesh = match crate::mesh_import::load_mesh(&path) {
            Ok(mesh) => mesh,
            Err(error) => {
                self.import_state.last_message = Some(WorkflowMessage {
                    text: format!("Import failed: {}", error),
                    is_error: true,
                });
                return false;
            }
        };

        let filename = path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("mesh")
            .to_string();
        self.import_state.dialog = Some(ImportDialogState::new(
            mesh,
            filename,
            self.settings.max_sculpt_resolution,
        ));
        self.import_state.last_message = None;
        true
    }

    pub fn cancel_import_dialog(&mut self) -> bool {
        if self.import_state.dialog.is_none() {
            return false;
        }

        self.import_state.dialog = None;
        true
    }

    pub fn set_import_use_auto(&mut self, use_auto: bool) -> bool {
        let Some(dialog) = self.import_state.dialog.as_mut() else {
            return false;
        };

        dialog.set_use_auto(use_auto);
        true
    }

    pub fn set_import_resolution(&mut self, resolution: u32) -> bool {
        let Some(dialog) = self.import_state.dialog.as_mut() else {
            return false;
        };

        dialog.set_use_auto(false);
        dialog.set_resolution(resolution);
        true
    }

    pub fn start_import(&mut self) -> bool {
        if matches!(self.import_state.task, ImportTask::InProgress { .. }) {
            return false;
        }

        let Some(dialog) = self.import_state.dialog.take() else {
            return false;
        };

        let mesh = dialog.mesh;
        let filename = dialog.filename;
        let resolution = dialog.resolution;
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);
        let (sender, receiver) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let result = crate::mesh_import::mesh_to_sdf(&mesh, resolution, &progress_clone);
            if cancelled_clone.load(Ordering::Relaxed) {
                return;
            }
            let _ = sender.send(result);
        });

        self.import_state.last_message = None;
        self.import_state.task = ImportTask::InProgress {
            progress,
            total: resolution,
            filename,
            receiver,
            cancelled,
        };
        true
    }

    pub fn cancel_import(&mut self) -> bool {
        let ImportTask::InProgress { cancelled, .. } = &self.import_state.task else {
            return false;
        };

        cancelled.store(true, Ordering::Relaxed);
        true
    }

    pub fn open_sculpt_convert_dialog_for_selected(&mut self) -> bool {
        let Some(selected_node) = self.selected_node else {
            self.sculpt_convert_state.last_message = Some(WorkflowMessage {
                text: "Select a node to sculpt".to_string(),
                is_error: true,
            });
            return false;
        };

        let Some(node) = self.scene.nodes.get(&selected_node) else {
            return false;
        };

        self.sculpt_convert_state.dialog = Some(SculptConvertDialogState::new(
            selected_node,
            node.name.clone(),
            self.settings.max_sculpt_resolution,
        ));
        self.sculpt_convert_state.last_message = None;
        true
    }

    pub fn cancel_sculpt_convert_dialog(&mut self) -> bool {
        if self.sculpt_convert_state.dialog.is_none() {
            return false;
        }

        self.sculpt_convert_state.dialog = None;
        true
    }

    pub fn set_sculpt_convert_mode(&mut self, mode_id: &str) -> bool {
        let Some(dialog) = self.sculpt_convert_state.dialog.as_mut() else {
            return false;
        };
        let Some(mode) = SculptConvertMode::from_id(mode_id) else {
            return false;
        };

        dialog.mode = mode;
        true
    }

    pub fn set_sculpt_convert_resolution(&mut self, resolution: u32) -> bool {
        let Some(dialog) = self.sculpt_convert_state.dialog.as_mut() else {
            return false;
        };

        dialog.set_resolution(resolution);
        true
    }

    pub fn start_sculpt_convert(&mut self) -> bool {
        if matches!(self.sculpt_convert_state.task, SculptConvertTask::InProgress { .. }) {
            return false;
        }

        let Some(dialog) = self.sculpt_convert_state.dialog.take() else {
            return false;
        };

        let Some(target_node) = self.scene.nodes.get(&dialog.target) else {
            return false;
        };

        let (subtree_root, flatten) = match dialog.mode {
            SculptConvertMode::BakeWholeScene => (self.topmost_ancestor(dialog.target), false),
            SculptConvertMode::BakeWholeSceneFlatten => {
                (self.topmost_ancestor(dialog.target), true)
            }
            SculptConvertMode::BakeActiveNode => (dialog.target, false),
        };
        let color = match &target_node.data {
            NodeData::Primitive { color, .. } => *color,
            _ => Vec3::new(0.8, 0.8, 0.8),
        };

        if flatten {
            let scene_clone = self.scene.clone();
            let progress = Arc::new(AtomicU32::new(0));
            let progress_clone = Arc::clone(&progress);
            let (sender, receiver) = std::sync::mpsc::channel();
            let resolution = dialog.resolution;
            let target_name = dialog.target_name.clone();
            std::thread::spawn(move || {
                let result = crate::graph::voxel::bake_subtree_with_progress(
                    &scene_clone,
                    subtree_root,
                    resolution,
                    progress_clone,
                );
                let _ = sender.send(result);
            });
            self.sculpt_convert_state.last_message = None;
            self.sculpt_convert_state.task = SculptConvertTask::InProgress {
                progress,
                total: dialog.resolution,
                target_name,
                receiver,
                subtree_root,
                color,
                flatten,
            };
            return true;
        }

        let (grid, center) =
            create_displacement_grid_for_subtree(&self.scene, subtree_root, dialog.resolution);
        self.apply_sculpt_convert_result(grid, center, subtree_root, color, false);
        self.sculpt_convert_state.last_message = Some(WorkflowMessage {
            text: format!("Converted {} to sculpt", dialog.target_name),
            is_error: false,
        });
        true
    }

    pub fn resume_sculpting_selected(&mut self) -> bool {
        let Some(selected_node) = self.selected_node else {
            return false;
        };

        self.activate_sculpt_session(selected_node)
    }

    pub fn stop_sculpting(&mut self) -> bool {
        if !self.sculpt_state.is_active() {
            return false;
        }

        self.deactivate_sculpt_session();
        true
    }

    pub fn set_sculpt_brush_mode(&mut self, mode_id: &str) -> bool {
        let Some(brush_mode) = brush_mode_from_id(mode_id) else {
            return false;
        };

        let SculptState::Active {
            brush_mode: current_mode,
            brush_strength,
            ..
        } = &mut self.sculpt_state
        else {
            return false;
        };

        let previous_mode = current_mode.clone();
        *current_mode = brush_mode.clone();
        if brush_mode == BrushMode::Grab && *brush_strength < 0.5 {
            *brush_strength = 1.0;
        } else if previous_mode == BrushMode::Grab && *brush_strength > 0.5 {
            *brush_strength = DEFAULT_BRUSH_STRENGTH;
        }
        *brush_strength = clamp_sculpt_strength_for_mode(current_mode, *brush_strength);
        true
    }

    pub fn set_sculpt_brush_radius(&mut self, radius: f32) -> bool {
        let SculptState::Active { brush_radius, .. } = &mut self.sculpt_state else {
            return false;
        };

        *brush_radius = radius.clamp(0.05, 2.0);
        true
    }

    pub fn set_sculpt_brush_strength(&mut self, strength: f32) -> bool {
        let SculptState::Active {
            brush_mode,
            brush_strength,
            ..
        } = &mut self.sculpt_state
        else {
            return false;
        };

        *brush_strength = clamp_sculpt_strength_for_mode(brush_mode, strength);
        true
    }

    pub fn set_sculpt_symmetry_axis(&mut self, axis_id: &str) -> bool {
        let Some(parsed_symmetry_axis) = sculpt_symmetry_axis_from_id(axis_id) else {
            return false;
        };

        let SculptState::Active { symmetry_axis, .. } = &mut self.sculpt_state else {
            return false;
        };

        *symmetry_axis = parsed_symmetry_axis;
        true
    }

    pub fn set_selected_sculpt_resolution(&mut self, resolution: u32) -> bool {
        let Some(selected_node) = self.selected_node else {
            return false;
        };
        let clamped_resolution = resolution.clamp(16, self.settings.max_sculpt_resolution.max(16));

        self.run_document_command(|bridge| {
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };
            let NodeData::Sculpt {
                desired_resolution, ..
            } = &mut node.data
            else {
                return false;
            };

            *desired_resolution = clamped_resolution;
            true
        })
    }

    pub fn set_export_resolution(&mut self, resolution: u32) -> u32 {
        let max_resolution = self.settings.max_export_resolution.max(16);
        let clamped_resolution = resolution.clamp(16, max_resolution);
        self.settings.export_resolution = clamped_resolution;
        self.settings.save();
        clamped_resolution
    }

    pub fn set_adaptive_export(&mut self, enabled: bool) {
        self.settings.adaptive_export = enabled;
        self.settings.save();
    }

    pub fn start_export(&mut self) -> bool {
        if matches!(self.export_state.task, ExportTask::InProgress { .. }) {
            return false;
        }

        let Some(path) = rfd::FileDialog::new()
            .set_title("Export Mesh")
            .add_filter("Wavefront OBJ", &["obj"])
            .add_filter("STL Binary", &["stl"])
            .add_filter("Stanford PLY", &["ply"])
            .add_filter("glTF Binary", &["glb"])
            .add_filter("USD ASCII", &["usda"])
            .save_file()
        else {
            return false;
        };

        self.start_export_to_path(path)
    }

    pub fn cancel_export(&mut self) -> bool {
        let ExportTask::InProgress { cancelled, .. } = &self.export_state.task else {
            return false;
        };

        cancelled.store(true, Ordering::Relaxed);
        true
    }

    pub fn add_sphere(&mut self) -> u64 {
        self.add_primitive(SdfPrimitive::Sphere)
    }

    pub fn add_box(&mut self) -> u64 {
        self.add_primitive(SdfPrimitive::Box)
    }

    pub fn add_cylinder(&mut self) -> u64 {
        self.add_primitive(SdfPrimitive::Cylinder)
    }

    pub fn add_torus(&mut self) -> u64 {
        self.add_primitive(SdfPrimitive::Torus)
    }

    pub fn delete_selected(&mut self) {
        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return;
            };

            if bridge.node_is_locked(selected_node) {
                return;
            }

            bridge.scene.remove_node(selected_node);
            if bridge.selected_node == Some(selected_node) {
                bridge.selected_node = None;
            }
            if bridge.hovered_node == Some(selected_node) {
                bridge.hovered_node = None;
            }
        });
    }

    pub fn duplicate_selected(&mut self) -> Option<u64> {
        self.run_document_command(|bridge| {
            let selected_node = bridge.selected_node?;
            let duplicated_node = bridge.scene.duplicate_subtree(selected_node)?;
            bridge.offset_duplicated_root(duplicated_node);
            bridge.selected_node = Some(duplicated_node);
            bridge.hovered_node = Some(duplicated_node);
            Some(duplicated_node)
        })
    }

    pub fn rename_node(&mut self, node_id: u64, name: &str) -> bool {
        let trimmed_name = name.trim();
        if trimmed_name.is_empty() {
            return false;
        }

        self.run_document_command(|bridge| {
            let Some(node) = bridge.scene.nodes.get_mut(&node_id) else {
                return false;
            };

            node.name = trimmed_name.to_string();
            true
        })
    }

    pub fn create_operation(&mut self, op: CsgOp) -> u64 {
        self.run_document_command(|bridge| {
            let geometry_roots = bridge.top_level_geometry_roots();
            let left = geometry_roots
                .get(geometry_roots.len().saturating_sub(2))
                .copied();
            let right = geometry_roots.last().copied();
            let operation_id = bridge.scene.create_operation(op, left, right);
            bridge.selected_node = Some(operation_id);
            bridge.hovered_node = Some(operation_id);
            operation_id
        })
    }

    pub fn create_transform(&mut self) -> u64 {
        self.run_document_command(|bridge| {
            let transform_id = if let Some(selected_node) = bridge.selected_node {
                bridge.scene.insert_transform_above(selected_node)
            } else {
                bridge.scene.create_transform(None)
            };
            bridge.selected_node = Some(transform_id);
            bridge.hovered_node = Some(transform_id);
            transform_id
        })
    }

    pub fn create_modifier(&mut self, kind: ModifierKind) -> u64 {
        self.run_document_command(|bridge| {
            let modifier_id = if let Some(selected_node) = bridge.selected_node {
                bridge.scene.insert_modifier_above(selected_node, kind)
            } else {
                bridge.scene.create_modifier(kind, None)
            };
            bridge.selected_node = Some(modifier_id);
            bridge.hovered_node = Some(modifier_id);
            modifier_id
        })
    }

    pub fn create_light(&mut self, light_type: LightType) -> u64 {
        self.run_document_command(|bridge| {
            let (_light_id, transform_id) = bridge.scene.create_light(light_type);
            bridge.selected_node = Some(transform_id);
            bridge.hovered_node = Some(transform_id);
            transform_id
        })
    }

    pub fn create_sculpt(&mut self) -> Option<u64> {
        let selected_node = self.selected_node?;

        if self
            .scene
            .nodes
            .get(&selected_node)
            .is_some_and(|node| matches!(node.data, NodeData::Sculpt { .. }))
        {
            self.hovered_node = Some(selected_node);
            self.activate_sculpt_session(selected_node);
            return Some(selected_node);
        }

        let parent_map = self.scene.build_parent_map();
        if let Some(sculpt_id) = self.scene.find_sculpt_parent(selected_node, &parent_map) {
            self.selected_node = Some(sculpt_id);
            self.hovered_node = Some(sculpt_id);
            self.activate_sculpt_session(sculpt_id);
            return Some(sculpt_id);
        }

        self.run_document_command(|bridge| {
            let selected_node = bridge.selected_node?;
            let (grid, center) = create_displacement_grid_for_subtree(
                &bridge.scene,
                selected_node,
                DEFAULT_SCULPT_ENTRY_RESOLUTION,
            );
            let color = bridge
                .scene
                .nodes
                .get(&selected_node)
                .map(|node| match &node.data {
                    NodeData::Primitive { color, .. } => *color,
                    _ => Vec3::new(0.8, 0.8, 0.8),
                })
                .unwrap_or(Vec3::new(0.8, 0.8, 0.8));
            let sculpt_id =
                bridge
                    .scene
                    .insert_sculpt_above(selected_node, center, Vec3::ZERO, color, grid);
            bridge.selected_node = Some(sculpt_id);
            bridge.hovered_node = Some(sculpt_id);
            bridge.activate_sculpt_session(sculpt_id);
            Some(sculpt_id)
        })
    }

    pub fn set_selected_primitive_parameter(&mut self, parameter_key: &str, value: f32) -> bool {
        let clamped_value = value.clamp(PRIMITIVE_PARAMETER_MIN, PRIMITIVE_PARAMETER_MAX);

        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Primitive { kind, scale, .. } => {
                    let axis = kind.scale_params().iter().find_map(|(label, axis)| {
                        (property_key(label) == parameter_key).then_some(*axis)
                    });
                    let Some(axis) = axis else {
                        return false;
                    };

                    set_scale_component(scale, axis, clamped_value);
                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_material_float(&mut self, field_id: &str, value: f32) -> bool {
        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Primitive {
                    roughness,
                    metallic,
                    emissive_intensity,
                    fresnel,
                    ..
                }
                | NodeData::Sculpt {
                    roughness,
                    metallic,
                    emissive_intensity,
                    fresnel,
                    ..
                } => {
                    match field_id {
                        "roughness" => {
                            *roughness = value.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX)
                        }
                        "metallic" => {
                            *metallic = value.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX)
                        }
                        "fresnel" => {
                            *fresnel = value.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX)
                        }
                        "emissive_intensity" => {
                            *emissive_intensity =
                                value.clamp(MATERIAL_FACTOR_MIN, EMISSIVE_INTENSITY_MAX)
                        }
                        _ => return false,
                    }

                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_material_color(
        &mut self,
        field_id: &str,
        red: f32,
        green: f32,
        blue: f32,
    ) -> bool {
        let clamped_color = clamp_color(Vec3::new(red, green, blue));

        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Primitive {
                    color, emissive, ..
                }
                | NodeData::Sculpt {
                    color, emissive, ..
                } => {
                    match field_id {
                        "color" => *color = clamped_color,
                        "emissive" => *emissive = clamped_color,
                        _ => return false,
                    }

                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_transform_position(&mut self, x: f32, y: f32, z: f32) -> bool {
        let next_position = Vec3::new(x, y, z);

        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Primitive { position, .. } | NodeData::Sculpt { position, .. } => {
                    *position = next_position;
                    true
                }
                NodeData::Transform { translation, .. } => {
                    *translation = next_position;
                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_transform_rotation_degrees(
        &mut self,
        x_degrees: f32,
        y_degrees: f32,
        z_degrees: f32,
    ) -> bool {
        let next_rotation = Vec3::new(
            x_degrees.to_radians(),
            y_degrees.to_radians(),
            z_degrees.to_radians(),
        );

        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Primitive { rotation, .. }
                | NodeData::Sculpt { rotation, .. }
                | NodeData::Transform { rotation, .. } => {
                    *rotation = next_rotation;
                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_transform_scale(&mut self, x: f32, y: f32, z: f32) -> bool {
        let next_scale = Vec3::new(
            x.clamp(PRIMITIVE_PARAMETER_MIN, PRIMITIVE_PARAMETER_MAX),
            y.clamp(PRIMITIVE_PARAMETER_MIN, PRIMITIVE_PARAMETER_MAX),
            z.clamp(PRIMITIVE_PARAMETER_MIN, PRIMITIVE_PARAMETER_MAX),
        );

        self.run_document_command(|bridge| {
            let Some(selected_node) = bridge.selected_node else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&selected_node) else {
                return false;
            };

            match &mut node.data {
                NodeData::Transform { scale, .. } => {
                    *scale = next_scale;
                    true
                }
                _ => false,
            }
        })
    }

    pub fn set_selected_light_type(&mut self, light_type_id: &str) -> bool {
        let Some(next_light_type) = parse_light_type_id(light_type_id) else {
            return false;
        };

        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                array_config,
                ..
            } => {
                *light_type = next_light_type.clone();
                if matches!(next_light_type, LightType::Array) && array_config.is_none() {
                    *array_config = Some(crate::graph::scene::LightArrayConfig::default());
                }
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_color(&mut self, red: f32, green: f32, blue: f32) -> bool {
        let clamped_color = clamp_color(Vec3::new(red, green, blue));

        self.run_selected_light_command(|data| match data {
            NodeData::Light { color, .. } => {
                *color = clamped_color;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_intensity(&mut self, intensity: f32) -> bool {
        if !intensity.is_finite() {
            return false;
        }

        let clamped_intensity = intensity.clamp(LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light { intensity, .. } => {
                *intensity = clamped_intensity;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_range(&mut self, range: f32) -> bool {
        if !range.is_finite() {
            return false;
        }

        let clamped_range = range.clamp(LIGHT_RANGE_MIN, LIGHT_RANGE_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                range,
                ..
            } if light_type_supports_range(light_type) => {
                *range = clamped_range;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_spot_angle(&mut self, angle_degrees: f32) -> bool {
        if !angle_degrees.is_finite() {
            return false;
        }

        let clamped_angle = angle_degrees.clamp(LIGHT_SPOT_ANGLE_MIN, LIGHT_SPOT_ANGLE_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type: LightType::Spot,
                spot_angle,
                ..
            } => {
                *spot_angle = clamped_angle;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_cast_shadows(&mut self, enabled: bool) -> bool {
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                cast_shadows,
                ..
            } if light_type_supports_shadows(light_type) => {
                *cast_shadows = enabled;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_shadow_softness(&mut self, softness: f32) -> bool {
        if !softness.is_finite() {
            return false;
        }

        let clamped_softness =
            softness.clamp(LIGHT_SHADOW_SOFTNESS_MIN, LIGHT_SHADOW_SOFTNESS_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                shadow_softness,
                ..
            } if light_type_supports_shadows(light_type) => {
                *shadow_softness = clamped_softness;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_shadow_color(&mut self, red: f32, green: f32, blue: f32) -> bool {
        let clamped_color = clamp_color(Vec3::new(red, green, blue));

        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                shadow_color,
                ..
            } if light_type_supports_shadows(light_type) => {
                *shadow_color = clamped_color;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_volumetric(&mut self, enabled: bool) -> bool {
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                volumetric,
                ..
            } if light_type_supports_volumetric(light_type) => {
                *volumetric = enabled;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_volumetric_density(&mut self, density: f32) -> bool {
        if !density.is_finite() {
            return false;
        }

        let clamped_density =
            density.clamp(LIGHT_VOLUMETRIC_DENSITY_MIN, LIGHT_VOLUMETRIC_DENSITY_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                volumetric_density,
                ..
            } if light_type_supports_volumetric(light_type) => {
                *volumetric_density = clamped_density;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_cookie(&mut self, cookie_node_id: u64) -> bool {
        let cookie_target = self.is_valid_light_cookie_target(cookie_node_id);
        if !cookie_target {
            return false;
        }

        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                cookie_node,
                ..
            } if light_type_supports_cookie(light_type) => {
                *cookie_node = Some(cookie_node_id);
                true
            }
            _ => false,
        })
    }

    pub fn clear_selected_light_cookie(&mut self) -> bool {
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                cookie_node,
                ..
            } if light_type_supports_cookie(light_type) => {
                *cookie_node = None;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_proximity_mode(&mut self, mode_id: &str) -> bool {
        let Some(next_mode) = parse_proximity_mode_id(mode_id) else {
            return false;
        };

        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                proximity_mode,
                ..
            } if light_type_supports_proximity(light_type) => {
                *proximity_mode = next_mode.clone();
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_proximity_range(&mut self, range: f32) -> bool {
        if !range.is_finite() {
            return false;
        }

        let clamped_range = range.clamp(LIGHT_PROXIMITY_RANGE_MIN, LIGHT_PROXIMITY_RANGE_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                proximity_range,
                ..
            } if light_type_supports_proximity(light_type) => {
                *proximity_range = clamped_range;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_array_pattern(&mut self, pattern_id: &str) -> bool {
        let Some(next_pattern) = parse_array_pattern_id(pattern_id) else {
            return false;
        };

        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type: LightType::Array,
                array_config,
                ..
            } => {
                let config =
                    array_config.get_or_insert_with(crate::graph::scene::LightArrayConfig::default);
                config.pattern = next_pattern.clone();
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_array_count(&mut self, count: u32) -> bool {
        let clamped_count = count.clamp(LIGHT_ARRAY_COUNT_MIN, LIGHT_ARRAY_COUNT_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type: LightType::Array,
                array_config,
                ..
            } => {
                let config =
                    array_config.get_or_insert_with(crate::graph::scene::LightArrayConfig::default);
                config.count = clamped_count;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_array_radius(&mut self, radius: f32) -> bool {
        if !radius.is_finite() {
            return false;
        }

        let clamped_radius = radius.clamp(LIGHT_ARRAY_RADIUS_MIN, LIGHT_ARRAY_RADIUS_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type: LightType::Array,
                array_config,
                ..
            } => {
                let config =
                    array_config.get_or_insert_with(crate::graph::scene::LightArrayConfig::default);
                config.radius = clamped_radius;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_array_color_variation(&mut self, value: f32) -> bool {
        if !value.is_finite() {
            return false;
        }

        let clamped_value = value.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type: LightType::Array,
                array_config,
                ..
            } => {
                let config =
                    array_config.get_or_insert_with(crate::graph::scene::LightArrayConfig::default);
                config.color_variation = clamped_value;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_intensity_expression(&mut self, expression: &str) -> bool {
        let next_expression = normalize_optional_text(expression);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                intensity_expr,
                ..
            } if light_type_supports_expressions(light_type) => {
                *intensity_expr = next_expression;
                true
            }
            _ => false,
        })
    }

    pub fn set_selected_light_color_hue_expression(&mut self, expression: &str) -> bool {
        let next_expression = normalize_optional_text(expression);
        self.run_selected_light_command(|data| match data {
            NodeData::Light {
                light_type,
                color_hue_expr,
                ..
            } if light_type_supports_expressions(light_type) => {
                *color_hue_expr = next_expression;
                true
            }
            _ => false,
        })
    }

    pub fn set_node_light_mask(&mut self, node_id: u64, mask: u8) -> bool {
        self.run_document_command(|bridge| {
            let Some(node) = bridge.scene.nodes.get(&node_id) else {
                return false;
            };
            if !node_supports_light_linking(&node.data) {
                return false;
            }

            bridge.scene.set_light_mask(node_id, mask);
            true
        })
    }

    pub fn set_node_light_link_enabled(
        &mut self,
        node_id: u64,
        light_id: u64,
        enabled: bool,
    ) -> bool {
        let Some(mask_bit) = self.light_mask_bit_for_light(light_id) else {
            return false;
        };

        self.run_document_command(|bridge| {
            let Some(node) = bridge.scene.nodes.get(&node_id) else {
                return false;
            };
            if !node_supports_light_linking(&node.data) {
                return false;
            }

            let mut next_mask = bridge.scene.get_light_mask(node_id);
            if enabled {
                next_mask |= mask_bit;
            } else {
                next_mask &= !mask_bit;
            }
            bridge.scene.set_light_mask(node_id, next_mask);
            true
        })
    }

    pub fn nudge_selected_translation(&mut self, delta_x: f32, delta_y: f32, delta_z: f32) -> bool {
        let Some(position) = self.selected_transform_position() else {
            return false;
        };

        self.set_selected_transform_position(
            position.x + delta_x,
            position.y + delta_y,
            position.z + delta_z,
        )
    }

    pub fn nudge_selected_rotation_degrees(
        &mut self,
        delta_x_degrees: f32,
        delta_y_degrees: f32,
        delta_z_degrees: f32,
    ) -> bool {
        let Some(rotation_degrees) = self.selected_transform_rotation_degrees() else {
            return false;
        };

        self.set_selected_transform_rotation_degrees(
            rotation_degrees.x + delta_x_degrees,
            rotation_degrees.y + delta_y_degrees,
            rotation_degrees.z + delta_z_degrees,
        )
    }

    pub fn nudge_selected_scale(&mut self, delta_x: f32, delta_y: f32, delta_z: f32) -> bool {
        let Some(scale) = self.selected_transform_scale() else {
            return false;
        };

        self.set_selected_transform_scale(scale.x + delta_x, scale.y + delta_y, scale.z + delta_z)
    }

    pub fn toggle_node_visibility(&mut self, node_id: u64) {
        self.run_document_command(|bridge| {
            if bridge.scene.nodes.contains_key(&node_id) {
                bridge.scene.toggle_visibility(node_id);
            }
        });
    }

    pub fn toggle_node_lock(&mut self, node_id: u64) {
        self.run_document_command(|bridge| {
            if let Some(node) = bridge.scene.nodes.get_mut(&node_id) {
                node.locked = !node.locked;
            }
        });
    }

    pub fn add_primitive(&mut self, kind: SdfPrimitive) -> u64 {
        self.run_document_command(|bridge| {
            let new_node_id = bridge.scene.create_primitive(kind);
            bridge.selected_node = Some(new_node_id);
            bridge.hovered_node = Some(new_node_id);
            new_node_id
        })
    }

    pub fn select_node(&mut self, node_id: Option<u64>) {
        self.selected_node = node_id.filter(|id| self.scene.nodes.contains_key(id));
        self.hovered_node = self.selected_node;
    }

    pub fn reset_scene(&mut self) {
        self.scene = Scene::new();
        self.camera = Camera::default();
        self.history = History::new();
        self.selected_node = None;
        self.hovered_node = None;
        self.last_viewport_time_seconds = None;
        self.clear_workflow_state();
        self.sync_document_persistence();
    }

    pub fn undo(&mut self) {
        if let Some((restored_scene, restored_selected)) =
            self.history.undo(&self.scene, self.selected_node)
        {
            self.restore_history_state(restored_scene, restored_selected);
        }
    }

    pub fn redo(&mut self) {
        if let Some((restored_scene, restored_selected)) =
            self.history.redo(&self.scene, self.selected_node)
        {
            self.restore_history_state(restored_scene, restored_selected);
        }
    }

    fn cancel_camera_transition(&mut self) {
        self.camera.transition = None;
    }

    fn start_camera_view_transition(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.camera.start_transition(yaw, pitch, roll);
    }

    fn tick_camera_animation(&mut self, time_seconds: f32) -> bool {
        let delta_seconds = self
            .last_viewport_time_seconds
            .map(|last_time_seconds| (time_seconds - last_time_seconds).clamp(0.0, 0.1))
            .unwrap_or(0.0);
        self.last_viewport_time_seconds = Some(time_seconds);
        self.camera.tick_transition(delta_seconds as f64)
    }

    fn run_document_command<T>(&mut self, command: impl FnOnce(&mut Self) -> T) -> T {
        self.history.begin_frame(&self.scene, self.selected_node);
        let result = command(self);
        self.history
            .end_frame(&self.scene, self.selected_node, false);
        self.sync_document_persistence();
        self.sync_sculpt_session();
        result
    }

    fn restore_history_state(&mut self, restored_scene: Scene, restored_selected: Option<NodeId>) {
        self.scene = restored_scene;
        self.selected_node =
            restored_selected.filter(|node_id| self.scene.nodes.contains_key(node_id));
        self.hovered_node = self.selected_node;
        self.sync_document_persistence();
        self.sync_sculpt_session();
    }

    fn run_selected_light_command(
        &mut self,
        command: impl FnOnce(&mut NodeData) -> bool,
    ) -> bool {
        self.run_document_command(|bridge| {
            let Some(light_id) = bridge.selected_light_node_id() else {
                return false;
            };
            let Some(node) = bridge.scene.nodes.get_mut(&light_id) else {
                return false;
            };
            command(&mut node.data)
        })
    }

    fn selected_light_node_id(&self) -> Option<NodeId> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;

        match &node.data {
            NodeData::Light { .. } => Some(selected_node),
            NodeData::Transform {
                input: Some(light_id),
                ..
            } => self
                .scene
                .nodes
                .get(light_id)
                .and_then(|child| matches!(child.data, NodeData::Light { .. }).then_some(*light_id)),
            _ => None,
        }
    }

    fn selected_light_transform_id(&self) -> Option<NodeId> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;

        match &node.data {
            NodeData::Transform {
                input: Some(light_id),
                ..
            } if self
                .scene
                .nodes
                .get(light_id)
                .is_some_and(|child| matches!(child.data, NodeData::Light { .. })) =>
            {
                Some(selected_node)
            }
            NodeData::Light { .. } => {
                let parent_map = self.scene.build_parent_map();
                parent_map.get(&selected_node).copied().filter(|parent_id| {
                    self.scene.nodes.get(parent_id).is_some_and(
                        |parent| matches!(parent.data, NodeData::Transform { .. }),
                    )
                })
            }
            _ => None,
        }
    }

    fn selected_light_properties_snapshot(&self) -> Option<AppLightPropertiesSnapshot> {
        let light_node_id = self.selected_light_node_id()?;
        let transform_node_id = self.selected_light_transform_id();
        let light_node = self.scene.nodes.get(&light_node_id)?;
        let NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            spot_angle,
            cast_shadows,
            shadow_softness,
            shadow_color,
            volumetric,
            volumetric_density,
            cookie_node,
            proximity_mode,
            proximity_range,
            array_config,
            intensity_expr,
            color_hue_expr,
        } = &light_node.data
        else {
            return None;
        };

        let cookie_candidates = self.light_cookie_candidates();
        let cookie_node_name = cookie_node
            .and_then(|node_id| self.scene.nodes.get(&node_id))
            .map(|node| node.name.clone());
        let intensity_expression = intensity_expr.clone();
        let color_hue_expression = color_hue_expr.clone();

        Some(AppLightPropertiesSnapshot {
            node_id: light_node_id,
            transform_node_id,
            light_type_id: light_type_id(light_type).to_string(),
            light_type_label: light_type.label().to_string(),
            color: app_vec3(*color),
            intensity: *intensity,
            range: *range,
            spot_angle: *spot_angle,
            cast_shadows: *cast_shadows,
            shadow_softness: *shadow_softness,
            shadow_color: app_vec3(*shadow_color),
            volumetric: *volumetric,
            volumetric_density: *volumetric_density,
            cookie_node_id: *cookie_node,
            cookie_node_name,
            cookie_candidates,
            proximity_mode_id: proximity_mode_id(proximity_mode).to_string(),
            proximity_mode_label: proximity_mode.label().to_string(),
            proximity_range: *proximity_range,
            array_pattern_id: array_config
                .as_ref()
                .map(|config| array_pattern_id(&config.pattern).to_string()),
            array_pattern_label: array_config
                .as_ref()
                .map(|config| config.pattern.label().to_string()),
            array_count: array_config.as_ref().map(|config| config.count),
            array_radius: array_config.as_ref().map(|config| config.radius),
            array_color_variation: array_config
                .as_ref()
                .map(|config| config.color_variation),
            intensity_expression: intensity_expression.clone(),
            intensity_expression_error: expression_error_message(intensity_expression.as_deref()),
            color_hue_expression: color_hue_expression.clone(),
            color_hue_expression_error: expression_error_message(color_hue_expression.as_deref()),
            supports_range: light_type_supports_range(light_type),
            supports_spot_angle: matches!(light_type, LightType::Spot),
            supports_shadows: light_type_supports_shadows(light_type),
            supports_volumetric: light_type_supports_volumetric(light_type),
            supports_cookie: light_type_supports_cookie(light_type),
            supports_proximity: light_type_supports_proximity(light_type),
            supports_expressions: light_type_supports_expressions(light_type),
            supports_array: matches!(light_type, LightType::Array),
        })
    }

    fn light_cookie_candidates(&self) -> Vec<AppLightCookieCandidateSnapshot> {
        let mut candidates: Vec<_> = self
            .scene
            .nodes
            .iter()
            .filter(|(_, node)| {
                matches!(
                    node.data,
                    NodeData::Primitive { .. } | NodeData::Operation { .. }
                )
            })
            .map(|(&node_id, node)| AppLightCookieCandidateSnapshot {
                    node_id,
                    name: node.name.clone(),
                    kind_label: node_kind_label(node),
                })
            .collect();
        candidates.sort_by(|left, right| left.name.cmp(&right.name).then(left.node_id.cmp(&right.node_id)));
        candidates
    }

    fn light_link_targets(&self) -> (Vec<LightLinkTarget>, usize) {
        let (active_light_ids, _) = identify_active_lights(&self.scene, self.camera.eye());
        let parent_map = self.scene.build_parent_map();
        let mut light_ids: Vec<NodeId> = self
            .scene
            .nodes
            .iter()
            .filter_map(|(&node_id, node)| {
                matches!(node.data, NodeData::Light { .. })
                    .then_some(node_id)
                    .filter(|id| !self.scene.is_hidden(*id))
                    .filter(|id| parent_map.contains_key(id))
            })
            .collect();
        light_ids.sort_unstable();

        let total_visible_light_count = light_ids.len();
        let targets = light_ids
            .into_iter()
            .take(MAX_SCENE_LIGHTS)
            .enumerate()
            .filter_map(|(slot, light_node_id)| {
                let node = self.scene.nodes.get(&light_node_id)?;
                let NodeData::Light {
                    light_type, color, ..
                } = &node.data
                else {
                    return None;
                };

                Some(LightLinkTarget {
                    light_node_id,
                    light_name: node.name.clone(),
                    light_type_label: light_type.label().to_string(),
                    active: active_light_ids.contains(&light_node_id),
                    mask_bit: 1u8 << slot,
                    color: *color,
                })
            })
            .collect();

        (targets, total_visible_light_count)
    }

    fn light_mask_bit_for_light(&self, light_id: NodeId) -> Option<u8> {
        self.light_link_targets()
            .0
            .into_iter()
            .find(|target| target.light_node_id == light_id)
            .map(|target| target.mask_bit)
    }

    fn light_linking_snapshot(&self) -> AppLightLinkingSnapshot {
        let (targets, total_visible_light_count) = self.light_link_targets();
        let mut geometry_nodes: Vec<_> = self
            .scene
            .nodes
            .iter()
            .filter(|(_, node)| node_supports_light_linking(&node.data))
            .map(|(&node_id, node)| AppLightLinkNodeSnapshot {
                    node_id,
                    node_name: node.name.clone(),
                    kind_label: node_kind_label(node),
                    light_mask: self.scene.get_light_mask(node_id),
                })
            .collect();
        geometry_nodes.sort_by_key(|node| node.node_id);

        AppLightLinkingSnapshot {
            lights: targets
                .into_iter()
                .map(|target| AppLightLinkTargetSnapshot {
                    light_node_id: target.light_node_id,
                    light_name: target.light_name,
                    light_type_label: target.light_type_label,
                    active: target.active,
                    mask_bit: target.mask_bit,
                    color: app_vec3(target.color),
                })
                .collect(),
            geometry_nodes,
            total_visible_light_count: total_visible_light_count as u32,
            max_light_count: MAX_SCENE_LIGHTS as u32,
        }
    }

    fn is_valid_light_cookie_target(&self, node_id: NodeId) -> bool {
        self.scene.nodes.get(&node_id).is_some_and(|node| {
            matches!(
                node.data,
                NodeData::Primitive { .. } | NodeData::Operation { .. }
            )
        })
    }

    fn recovery_summary_from_meta(meta: Option<&crate::io::RecoveryMeta>) -> String {
        let timestamp = meta
            .map(|recovery_meta| recovery_meta.autosave_unix_secs)
            .unwrap_or(0);
        let project_hint = meta
            .and_then(|recovery_meta| recovery_meta.project_path.as_deref())
            .map(|path| format!("\nSource project: {path}"))
            .unwrap_or_default();
        format!("Recovered unsaved work found from UNIX timestamp {timestamp}.{project_hint}")
    }

    fn open_scene_from_path(&mut self, path: &Path) -> bool {
        let Ok(project) = crate::io::load_project(&path.to_path_buf()) else {
            return false;
        };

        self.scene = project.scene;
        self.camera = project.camera;
        self.history = History::new();
        self.selected_node = None;
        self.hovered_node = None;
        self.last_viewport_time_seconds = None;
        self.clear_workflow_state();
        self.persistence.current_file_path = Some(path.to_path_buf());
        self.persistence.saved_fingerprint = self.scene.data_fingerprint();
        self.persistence.scene_dirty = false;
        self.settings.add_recent_file(&path.to_string_lossy());
        self.clear_recovery_prompt();
        true
    }

    fn save_scene_to_path(&mut self, path: &Path) -> bool {
        let save_path = path.to_path_buf();
        if crate::io::save_project(&self.scene, &self.camera, &save_path).is_err() {
            return false;
        }

        self.persistence.current_file_path = Some(save_path.clone());
        self.persistence.saved_fingerprint = self.scene.data_fingerprint();
        self.persistence.scene_dirty = false;
        self.settings.add_recent_file(&save_path.to_string_lossy());
        let _ = crate::io::remove_recovery_files();
        self.clear_recovery_prompt();
        true
    }

    fn sync_document_persistence(&mut self) {
        let current_fingerprint = self.scene.data_fingerprint();
        self.persistence.scene_dirty = current_fingerprint != self.persistence.saved_fingerprint;

        if !self.settings.auto_save_enabled
            || !self.persistence.scene_dirty
            || self.persistence.last_auto_save.elapsed()
                < Duration::from_secs(self.settings.auto_save_interval_secs as u64)
        {
            return;
        }

        self.persistence.last_auto_save = Instant::now();
        let autosave_path = crate::io::auto_save_path();
        if crate::io::save_project(&self.scene, &self.camera, &autosave_path).is_err() {
            return;
        }

        let _ = crate::io::write_recovery_meta(self.persistence.current_file_path.as_deref());
    }

    fn clear_recovery_prompt(&mut self) {
        self.persistence.recovery_prompt_visible = false;
        self.persistence.recovery_summary = None;
    }

    fn clear_workflow_state(&mut self) {
        self.import_state.dialog = None;
        self.import_state.task = ImportTask::Idle;
        self.import_state.last_message = None;
        self.sculpt_convert_state.dialog = None;
        self.sculpt_convert_state.task = SculptConvertTask::Idle;
        self.sculpt_convert_state.last_message = None;
        self.sculpt_state = SculptState::Inactive;
        self.active_tool_label = "Select".to_string();
    }

    fn start_export_to_path(&mut self, path: PathBuf) -> bool {
        if matches!(self.export_state.task, ExportTask::InProgress { .. }) {
            return false;
        }

        let scene_clone = self.scene.clone();
        let bounds = self.scene.compute_bounds();
        let padding = 0.5;
        let bounds_min = Vec3::from(bounds.0) - Vec3::splat(padding);
        let bounds_max = Vec3::from(bounds.1) + Vec3::splat(padding);
        let max_resolution = self.settings.max_export_resolution.max(16);
        let resolution = self.settings.export_resolution.clamp(16, max_resolution);
        let adaptive = self.settings.adaptive_export;
        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = Arc::clone(&progress);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);
        let (sender, receiver) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let mesh = crate::export::marching_cubes(
                &scene_clone,
                resolution,
                bounds_min,
                bounds_max,
                &progress_clone,
                adaptive,
                &cancelled_clone,
            );
            let _ = sender.send(mesh);
        });

        self.settings.save();
        self.export_state.last_message = None;
        self.export_state.task = ExportTask::InProgress {
            progress,
            total: (resolution + 1) + resolution,
            resolution,
            receiver,
            path,
            cancelled,
        };
        true
    }

    fn poll_export(&mut self) {
        let completed = if let ExportTask::InProgress { receiver, .. } = &self.export_state.task {
            receiver.try_recv().ok()
        } else {
            None
        };

        let Some(maybe_mesh) = completed else {
            return;
        };

        let path = match &self.export_state.task {
            ExportTask::InProgress { path, .. } => path.clone(),
            ExportTask::Idle => return,
        };

        self.export_state.last_message = Some(match maybe_mesh {
            Some(mesh) => match crate::export::write_mesh(&mesh, &path) {
                Ok(()) => WorkflowMessage {
                    text: format!(
                        "Exported {} ({} verts, {} tris)",
                        export_format_label(&path),
                        mesh.vertices.len(),
                        mesh.triangles.len(),
                    ),
                    is_error: false,
                },
                Err(error) => WorkflowMessage {
                    text: format!("Export failed: {}", error),
                    is_error: true,
                },
            },
            None => WorkflowMessage {
                text: "Export cancelled".to_string(),
                is_error: false,
            },
        });
        self.export_state.task = ExportTask::Idle;
    }

    fn poll_import(&mut self) {
        if let ImportTask::InProgress {
            cancelled,
            receiver,
            ..
        } = &self.import_state.task
        {
            if cancelled.load(Ordering::Relaxed) {
                let _ = receiver.try_recv();
                self.import_state.last_message = Some(WorkflowMessage {
                    text: "Import cancelled".to_string(),
                    is_error: false,
                });
                self.import_state.task = ImportTask::Idle;
                return;
            }
        }

        let completed = if let ImportTask::InProgress { receiver, .. } = &self.import_state.task {
            receiver.try_recv().ok()
        } else {
            None
        };

        let Some((grid, center)) = completed else {
            return;
        };

        let filename = match &self.import_state.task {
            ImportTask::InProgress { filename, .. } => filename.clone(),
            ImportTask::Idle => return,
        };

        self.run_document_command(|bridge| {
            let desired_resolution = grid.resolution;
            let node_name = bridge.scene.next_name("Import");
            let sculpt_id = bridge.scene.add_node(
                node_name,
                NodeData::Sculpt {
                    input: None,
                    position: center,
                    rotation: Vec3::ZERO,
                    color: Vec3::new(0.7, 0.7, 0.7),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    fresnel: 0.04,
                    layer_intensity: 1.0,
                    voxel_grid: grid,
                    desired_resolution,
                },
            );
            bridge.selected_node = Some(sculpt_id);
            bridge.hovered_node = Some(sculpt_id);
            bridge.activate_sculpt_session(sculpt_id);
        });

        self.import_state.last_message = Some(WorkflowMessage {
            text: format!("Imported {} as sculpt geometry", filename),
            is_error: false,
        });
        self.import_state.task = ImportTask::Idle;
    }

    fn poll_sculpt_convert(&mut self) {
        let completed = if let SculptConvertTask::InProgress { receiver, .. } =
            &self.sculpt_convert_state.task
        {
            receiver.try_recv().ok()
        } else {
            None
        };

        let Some((grid, center)) = completed else {
            return;
        };

        let (subtree_root, color, flatten, target_name) = match &self.sculpt_convert_state.task {
            SculptConvertTask::InProgress {
                subtree_root,
                color,
                flatten,
                target_name,
                ..
            } => (*subtree_root, *color, *flatten, target_name.clone()),
            SculptConvertTask::Idle => return,
        };

        self.apply_sculpt_convert_result(grid, center, subtree_root, color, flatten);
        self.sculpt_convert_state.last_message = Some(WorkflowMessage {
            text: format!("Converted {} to sculpt", target_name),
            is_error: false,
        });
        self.sculpt_convert_state.task = SculptConvertTask::Idle;
    }

    fn apply_sculpt_convert_result(
        &mut self,
        grid: crate::graph::voxel::VoxelGrid,
        center: Vec3,
        subtree_root: NodeId,
        color: Vec3,
        flatten: bool,
    ) {
        self.run_document_command(|bridge| {
            if flatten {
                let sculpt_id = bridge.scene.flatten_subtree(subtree_root, grid, center, color);
                bridge.selected_node = Some(sculpt_id);
                bridge.hovered_node = Some(sculpt_id);
                bridge.activate_sculpt_session(sculpt_id);
            } else {
                let sculpt_id =
                    bridge
                        .scene
                        .insert_sculpt_above(subtree_root, center, Vec3::ZERO, color, grid);
                bridge.selected_node = Some(sculpt_id);
                bridge.hovered_node = Some(sculpt_id);
                bridge.activate_sculpt_session(sculpt_id);
            }
        });
    }

    fn offset_duplicated_root(&mut self, node_id: NodeId) {
        let Some(node) = self.scene.nodes.get_mut(&node_id) else {
            return;
        };

        match &mut node.data {
            NodeData::Primitive { position, .. } | NodeData::Sculpt { position, .. } => {
                position.x += 1.0;
            }
            _ => {}
        }
    }

    fn topmost_ancestor(&self, start_node: NodeId) -> NodeId {
        let parent_map = self.scene.build_parent_map();
        let mut current_node = start_node;
        while let Some(&parent_id) = parent_map.get(&current_node) {
            current_node = parent_id;
        }
        current_node
    }

    fn top_level_geometry_roots(&self) -> Vec<NodeId> {
        self.scene
            .top_level_nodes()
            .into_iter()
            .filter(|node_id| self.subtree_has_visible_geometry(*node_id, &mut HashSet::new()))
            .collect()
    }

    fn subtree_has_visible_geometry(&self, node_id: NodeId, visited: &mut HashSet<NodeId>) -> bool {
        if !visited.insert(node_id) || self.scene.is_hidden(node_id) {
            return false;
        }

        let Some(node) = self.scene.nodes.get(&node_id) else {
            return false;
        };

        if node.data.geometry_local_sphere().is_some() {
            return true;
        }

        node.data
            .children()
            .any(|child_id| self.subtree_has_visible_geometry(child_id, visited))
    }

    fn scene_tree_roots(&self) -> Vec<AppSceneTreeNodeSnapshot> {
        let mut visited = HashSet::new();
        self.scene
            .top_level_nodes()
            .into_iter()
            .filter_map(|node_id| self.scene_tree_node_snapshot(node_id, &mut visited))
            .collect()
    }

    fn scene_tree_node_snapshot(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
    ) -> Option<AppSceneTreeNodeSnapshot> {
        if !visited.insert(node_id) {
            return None;
        }

        let node = self.scene.nodes.get(&node_id)?;
        let children = node
            .data
            .children()
            .filter_map(|child_id| self.scene_tree_node_snapshot(child_id, visited))
            .collect();

        Some(AppSceneTreeNodeSnapshot {
            id: node.id,
            name: node.name.clone(),
            kind_label: node_kind_label(node),
            visible: !self.scene.is_hidden(node.id),
            locked: node.locked,
            children,
        })
    }

    fn document_snapshot(&self) -> AppDocumentSnapshot {
        let current_file_path = self
            .persistence
            .current_file_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string());
        let current_file_name = self
            .persistence
            .current_file_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|file_name| file_name.to_string_lossy().to_string());

        AppDocumentSnapshot {
            current_file_path,
            current_file_name,
            has_unsaved_changes: self.persistence.scene_dirty,
            recent_files: self.settings.recent_files.clone(),
            recovery_available: self.persistence.recovery_prompt_visible,
            recovery_summary: self.persistence.recovery_summary.clone(),
        }
    }

    fn export_snapshot(&self) -> AppExportSnapshot {
        let max_resolution = self.settings.max_export_resolution.max(16);
        let status = match &self.export_state.task {
            ExportTask::Idle => AppExportStatusSnapshot {
                state: "idle".to_string(),
                progress: 0,
                total: 0,
                resolution: self.settings.export_resolution.clamp(16, max_resolution),
                phase_label: None,
                target_file_name: None,
                target_file_path: None,
                format_label: None,
                message: self
                    .export_state
                    .last_message
                    .as_ref()
                    .map(|message| message.text.clone()),
                is_error: self
                    .export_state
                    .last_message
                    .as_ref()
                    .is_some_and(|message| message.is_error),
            },
            ExportTask::InProgress {
                progress,
                total,
                resolution,
                path,
                ..
            } => {
                let completed_steps = progress.load(Ordering::Relaxed).min(*total);
                AppExportStatusSnapshot {
                    state: "in_progress".to_string(),
                    progress: completed_steps,
                    total: *total,
                    resolution: *resolution,
                    phase_label: Some(export_phase_label(completed_steps, *total, *resolution)),
                    target_file_name: path
                        .file_name()
                        .map(|file_name| file_name.to_string_lossy().to_string()),
                    target_file_path: Some(path.to_string_lossy().to_string()),
                    format_label: Some(export_format_label(path)),
                    message: None,
                    is_error: false,
                }
            }
        };

        AppExportSnapshot {
            resolution: self.settings.export_resolution.clamp(16, max_resolution),
            min_resolution: 16,
            max_resolution,
            adaptive: self.settings.adaptive_export,
            presets: self
                .settings
                .export_presets
                .iter()
                .map(|preset| AppExportPresetSnapshot {
                    name: preset.name.clone(),
                    resolution: preset.resolution,
                })
                .collect(),
            status,
        }
    }

    fn import_snapshot(&self) -> AppImportSnapshot {
        let dialog = self.import_state.dialog.as_ref().map(|dialog| AppImportDialogSnapshot {
            filename: dialog.filename.clone(),
            resolution: dialog.resolution,
            auto_resolution: dialog.auto_resolution,
            use_auto: dialog.use_auto,
            vertex_count: dialog.vertex_count,
            triangle_count: dialog.triangle_count,
            bounds_size: app_vec3(dialog.bounds_size),
            min_resolution: dialog.min_resolution,
            max_resolution: dialog.max_resolution,
        });
        let status = match &self.import_state.task {
            ImportTask::Idle => AppImportStatusSnapshot {
                state: "idle".to_string(),
                progress: 0,
                total: 0,
                filename: None,
                phase_label: None,
                message: self
                    .import_state
                    .last_message
                    .as_ref()
                    .map(|message| message.text.clone()),
                is_error: self
                    .import_state
                    .last_message
                    .as_ref()
                    .is_some_and(|message| message.is_error),
            },
            ImportTask::InProgress {
                progress,
                total,
                filename,
                ..
            } => {
                let completed_steps = progress.load(Ordering::Relaxed).min(*total);
                AppImportStatusSnapshot {
                    state: "in_progress".to_string(),
                    progress: completed_steps,
                    total: *total,
                    filename: Some(filename.clone()),
                    phase_label: Some(format!(
                        "Voxelizing slice {}/{}",
                        completed_steps, total
                    )),
                    message: None,
                    is_error: false,
                }
            }
        };

        AppImportSnapshot { dialog, status }
    }

    fn sculpt_convert_snapshot(&self) -> AppSculptConvertSnapshot {
        let dialog = self
            .sculpt_convert_state
            .dialog
            .as_ref()
            .map(|dialog| AppSculptConvertDialogSnapshot {
                target_node_id: dialog.target,
                target_name: dialog.target_name.clone(),
                mode_id: dialog.mode.id().to_string(),
                mode_label: dialog.mode.label().to_string(),
                resolution: dialog.resolution,
                min_resolution: dialog.min_resolution,
                max_resolution: dialog.max_resolution,
            });
        let status = match &self.sculpt_convert_state.task {
            SculptConvertTask::Idle => AppSculptConvertStatusSnapshot {
                state: "idle".to_string(),
                progress: 0,
                total: 0,
                target_name: None,
                phase_label: None,
                message: self
                    .sculpt_convert_state
                    .last_message
                    .as_ref()
                    .map(|message| message.text.clone()),
                is_error: self
                    .sculpt_convert_state
                    .last_message
                    .as_ref()
                    .is_some_and(|message| message.is_error),
            },
            SculptConvertTask::InProgress {
                progress,
                total,
                target_name,
                ..
            } => {
                let completed_steps = progress.load(Ordering::Relaxed).min(*total);
                AppSculptConvertStatusSnapshot {
                    state: "in_progress".to_string(),
                    progress: completed_steps,
                    total: *total,
                    target_name: Some(target_name.clone()),
                    phase_label: Some(format!(
                        "Baking sculpt volume {}/{}",
                        completed_steps, total
                    )),
                    message: None,
                    is_error: false,
                }
            }
        };

        AppSculptConvertSnapshot { dialog, status }
    }

    fn sculpt_snapshot(&self) -> AppSculptSnapshot {
        let selected = self.selected_node.and_then(|selected_node| {
            let node = self.scene.nodes.get(&selected_node)?;
            let NodeData::Sculpt {
                voxel_grid,
                desired_resolution,
                ..
            } = &node.data
            else {
                return None;
            };

            Some(AppSelectedSculptSnapshot {
                node_id: selected_node,
                node_name: node.name.clone(),
                current_resolution: voxel_grid.resolution,
                desired_resolution: *desired_resolution,
            })
        });
        let active_node_id = self.sculpt_state.active_node();
        let session = match &self.sculpt_state {
            SculptState::Active {
                node_id,
                brush_mode,
                brush_radius,
                brush_strength,
                symmetry_axis,
                ..
            } => self
                .scene
                .nodes
                .get(node_id)
                .map(|node| AppSculptSessionSnapshot {
                    node_id: *node_id,
                    node_name: node.name.clone(),
                    brush_mode_id: brush_mode_id(brush_mode).to_string(),
                    brush_mode_label: brush_mode_label(brush_mode).to_string(),
                    brush_radius: *brush_radius,
                    brush_strength: *brush_strength,
                    symmetry_axis_id: sculpt_symmetry_axis_id(*symmetry_axis).to_string(),
                    symmetry_axis_label: sculpt_symmetry_axis_label(*symmetry_axis).to_string(),
                }),
            SculptState::Inactive => None,
        };

        AppSculptSnapshot {
            can_resume_selected: selected
                .as_ref()
                .is_some_and(|selected_sculpt| active_node_id != Some(selected_sculpt.node_id)),
            can_stop: self.sculpt_state.is_active(),
            max_resolution: self.settings.max_sculpt_resolution.max(16),
            selected,
            session,
        }
    }

    fn node_is_locked(&self, node_id: NodeId) -> bool {
        self.scene
            .nodes
            .get(&node_id)
            .is_some_and(|node| node.locked)
    }

    fn activate_sculpt_session(&mut self, node_id: NodeId) -> bool {
        let Some(node) = self.scene.nodes.get(&node_id) else {
            return false;
        };
        let NodeData::Sculpt { .. } = &node.data else {
            return false;
        };

        let extent = node
            .data
            .geometry_local_sphere()
            .map(|(_, radius)| radius.max(0.5))
            .unwrap_or(0.5);
        self.sculpt_state = SculptState::new_active_with_radius(node_id, extent);
        self.active_tool_label = "Sculpt".to_string();
        true
    }

    fn deactivate_sculpt_session(&mut self) {
        self.sculpt_state = SculptState::Inactive;
        self.active_tool_label = "Select".to_string();
    }

    fn sync_sculpt_session(&mut self) {
        let Some(active_node) = self.sculpt_state.active_node() else {
            if self.active_tool_label == "Sculpt" {
                self.active_tool_label = "Select".to_string();
            }
            return;
        };

        if self
            .scene
            .nodes
            .get(&active_node)
            .is_some_and(|node| matches!(node.data, NodeData::Sculpt { .. }))
        {
            self.active_tool_label = "Sculpt".to_string();
            return;
        }

        self.deactivate_sculpt_session();
    }

    fn selected_transform_position(&self) -> Option<Vec3> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;
        match &node.data {
            NodeData::Primitive { position, .. } | NodeData::Sculpt { position, .. } => {
                Some(*position)
            }
            NodeData::Transform { translation, .. } => Some(*translation),
            _ => None,
        }
    }

    fn selected_transform_rotation_degrees(&self) -> Option<Vec3> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;
        match &node.data {
            NodeData::Primitive { rotation, .. }
            | NodeData::Sculpt { rotation, .. }
            | NodeData::Transform { rotation, .. } => Some(Vec3::new(
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            )),
            _ => None,
        }
    }

    fn selected_transform_scale(&self) -> Option<Vec3> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;
        match &node.data {
            NodeData::Transform { scale, .. } => Some(*scale),
            _ => None,
        }
    }

    fn camera_snapshot(&self) -> AppCameraSnapshot {
        let eye = self.camera.eye();
        AppCameraSnapshot {
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            roll: self.camera.roll,
            distance: self.camera.distance,
            fov_degrees: self.camera.fov.to_degrees(),
            orthographic: self.camera.orthographic,
            target: app_vec3(self.camera.target),
            eye: AppVec3::new(eye.x, eye.y, eye.z),
        }
    }

    fn node_snapshot_by_id(&self, node_id: Option<NodeId>) -> Option<AppNodeSnapshot> {
        node_id
            .and_then(|resolved_node_id| self.scene.nodes.get(&resolved_node_id))
            .map(|node| self.node_snapshot(node))
    }

    fn node_snapshot(&self, node: &SceneNode) -> AppNodeSnapshot {
        AppNodeSnapshot {
            id: node.id,
            name: node.name.clone(),
            kind_label: node_kind_label(node),
            visible: !self.scene.is_hidden(node.id),
            locked: node.locked,
        }
    }

    fn selected_node_properties_snapshot(&self) -> Option<AppSelectedNodePropertiesSnapshot> {
        let selected_node = self.selected_node?;
        let node = self.scene.nodes.get(&selected_node)?;

        Some(AppSelectedNodePropertiesSnapshot {
            node_id: node.id,
            name: node.name.clone(),
            kind_label: node_kind_label(node),
            visible: !self.scene.is_hidden(node.id),
            locked: node.locked,
            transform: self.node_transform_properties(&node.data),
            primitive: self.node_primitive_properties(&node.data),
            material: self.node_material_properties(&node.data),
            light: self.selected_light_properties_snapshot(),
        })
    }

    fn node_transform_properties(&self, data: &NodeData) -> Option<AppTransformPropertiesSnapshot> {
        match data {
            NodeData::Primitive {
                position, rotation, ..
            }
            | NodeData::Sculpt {
                position, rotation, ..
            } => Some(AppTransformPropertiesSnapshot {
                position_label: "Position".to_string(),
                position: app_vec3(*position),
                rotation_degrees: app_degrees_vec3(*rotation),
                scale: None,
            }),
            NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } => Some(AppTransformPropertiesSnapshot {
                position_label: "Translation".to_string(),
                position: app_vec3(*translation),
                rotation_degrees: app_degrees_vec3(*rotation),
                scale: Some(app_vec3(*scale)),
            }),
            _ => None,
        }
    }

    fn node_primitive_properties(&self, data: &NodeData) -> Option<AppPrimitivePropertiesSnapshot> {
        let NodeData::Primitive { kind, scale, .. } = data else {
            return None;
        };

        let parameters = kind
            .scale_params()
            .iter()
            .map(|(label, axis)| AppScalarPropertySnapshot {
                key: property_key(label),
                label: (*label).to_string(),
                value: scale_component(*scale, *axis),
            })
            .collect();

        Some(AppPrimitivePropertiesSnapshot {
            primitive_kind: kind.base_name().to_string(),
            parameters,
        })
    }

    fn node_material_properties(&self, data: &NodeData) -> Option<AppMaterialPropertiesSnapshot> {
        match data {
            NodeData::Primitive {
                color,
                roughness,
                metallic,
                emissive,
                emissive_intensity,
                fresnel,
                ..
            }
            | NodeData::Sculpt {
                color,
                roughness,
                metallic,
                emissive,
                emissive_intensity,
                fresnel,
                ..
            } => Some(AppMaterialPropertiesSnapshot {
                color: app_vec3(*color),
                roughness: *roughness,
                metallic: *metallic,
                emissive: app_vec3(*emissive),
                emissive_intensity: *emissive_intensity,
                fresnel: *fresnel,
            }),
            _ => None,
        }
    }
}

fn node_kind_label(node: &SceneNode) -> String {
    match &node.data {
        NodeData::Primitive { kind, .. } => kind.base_name().to_string(),
        NodeData::Operation { op, .. } => op.base_name().to_string(),
        NodeData::Sculpt { .. } => "Sculpt".to_string(),
        NodeData::Transform { .. } => "Transform".to_string(),
        NodeData::Modifier { kind, .. } => kind.base_name().to_string(),
        NodeData::Light { light_type, .. } => light_type.label().to_string(),
    }
}

fn app_vec3(value: Vec3) -> AppVec3 {
    AppVec3::new(value.x, value.y, value.z)
}

fn app_degrees_vec3(value: Vec3) -> AppVec3 {
    AppVec3::new(
        value.x.to_degrees(),
        value.y.to_degrees(),
        value.z.to_degrees(),
    )
}

fn scale_component(scale: Vec3, axis: usize) -> f32 {
    match axis {
        0 => scale.x,
        1 => scale.y,
        _ => scale.z,
    }
}

fn set_scale_component(scale: &mut Vec3, axis: usize, value: f32) {
    match axis {
        0 => scale.x = value,
        1 => scale.y = value,
        _ => scale.z = value,
    }
}

fn clamp_color(color: Vec3) -> Vec3 {
    Vec3::new(
        color.x.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX),
        color.y.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX),
        color.z.clamp(MATERIAL_FACTOR_MIN, MATERIAL_FACTOR_MAX),
    )
}

fn light_type_id(light_type: &LightType) -> &'static str {
    match light_type {
        LightType::Point => "point",
        LightType::Spot => "spot",
        LightType::Directional => "directional",
        LightType::Ambient => "ambient",
        LightType::Array => "array",
    }
}

fn parse_light_type_id(light_type_id: &str) -> Option<LightType> {
    Some(match light_type_id {
        "point" => LightType::Point,
        "spot" => LightType::Spot,
        "directional" => LightType::Directional,
        "ambient" => LightType::Ambient,
        "array" => LightType::Array,
        _ => return None,
    })
}

fn proximity_mode_id(proximity_mode: &ProximityMode) -> &'static str {
    match proximity_mode {
        ProximityMode::Off => "off",
        ProximityMode::Brighten => "brighten",
        ProximityMode::Dim => "dim",
    }
}

fn parse_proximity_mode_id(mode_id: &str) -> Option<ProximityMode> {
    Some(match mode_id {
        "off" => ProximityMode::Off,
        "brighten" => ProximityMode::Brighten,
        "dim" => ProximityMode::Dim,
        _ => return None,
    })
}

fn array_pattern_id(pattern: &ArrayPattern) -> &'static str {
    match pattern {
        ArrayPattern::Ring => "ring",
        ArrayPattern::Line => "line",
        ArrayPattern::Grid => "grid",
        ArrayPattern::Spiral => "spiral",
    }
}

fn parse_array_pattern_id(pattern_id: &str) -> Option<ArrayPattern> {
    Some(match pattern_id {
        "ring" => ArrayPattern::Ring,
        "line" => ArrayPattern::Line,
        "grid" => ArrayPattern::Grid,
        "spiral" => ArrayPattern::Spiral,
        _ => return None,
    })
}

fn light_type_supports_range(light_type: &LightType) -> bool {
    matches!(light_type, LightType::Point | LightType::Spot | LightType::Array)
}

fn light_type_supports_shadows(light_type: &LightType) -> bool {
    !matches!(light_type, LightType::Ambient | LightType::Array)
}

fn light_type_supports_volumetric(light_type: &LightType) -> bool {
    matches!(light_type, LightType::Point | LightType::Spot)
}

fn light_type_supports_cookie(light_type: &LightType) -> bool {
    matches!(light_type, LightType::Point | LightType::Spot)
}

fn light_type_supports_proximity(light_type: &LightType) -> bool {
    matches!(light_type, LightType::Point | LightType::Spot)
}

fn light_type_supports_expressions(light_type: &LightType) -> bool {
    !matches!(light_type, LightType::Ambient | LightType::Array)
}

fn normalize_optional_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn expression_error_message(expression: Option<&str>) -> Option<String> {
    let expression = expression?.trim();
    if expression.is_empty() {
        return None;
    }

    crate::expression::parse_expression(expression)
        .err()
        .map(|error| error.to_string())
}

fn node_supports_light_linking(data: &NodeData) -> bool {
    matches!(data, NodeData::Primitive { .. } | NodeData::Sculpt { .. })
}

fn property_key(label: &str) -> String {
    let mut key = String::with_capacity(label.len());
    let mut previous_was_separator = false;

    for character in label.chars() {
        if character.is_ascii_alphanumeric() {
            key.push(character.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !previous_was_separator && !key.is_empty() {
            key.push('_');
            previous_was_separator = true;
        }
    }

    key.trim_end_matches('_').to_string()
}

fn export_phase_label(progress: u32, total: u32, resolution: u32) -> String {
    let sample_slices = resolution + 1;
    if progress < sample_slices {
        format!(
            "Phase 1/3: Sampling SDF field ({}/{})",
            progress, sample_slices
        )
    } else if progress < total {
        let extracted_slices = progress - sample_slices;
        let extracted_total = total - sample_slices;
        format!(
            "Phase 2/3: Extracting triangles ({}/{})",
            extracted_slices, extracted_total
        )
    } else {
        "Phase 3/3: Merging vertices and sampling colors...".to_string()
    }
}

fn export_format_label(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("obj")
        .to_uppercase()
}

fn brush_mode_from_id(mode_id: &str) -> Option<BrushMode> {
    Some(match mode_id {
        "add" => BrushMode::Add,
        "carve" => BrushMode::Carve,
        "smooth" => BrushMode::Smooth,
        "flatten" => BrushMode::Flatten,
        "inflate" => BrushMode::Inflate,
        "grab" => BrushMode::Grab,
        _ => return None,
    })
}

fn brush_mode_id(mode: &BrushMode) -> &'static str {
    match mode {
        BrushMode::Add => "add",
        BrushMode::Carve => "carve",
        BrushMode::Smooth => "smooth",
        BrushMode::Flatten => "flatten",
        BrushMode::Inflate => "inflate",
        BrushMode::Grab => "grab",
    }
}

fn brush_mode_label(mode: &BrushMode) -> &'static str {
    match mode {
        BrushMode::Add => "Add",
        BrushMode::Carve => "Carve",
        BrushMode::Smooth => "Smooth",
        BrushMode::Flatten => "Flatten",
        BrushMode::Inflate => "Inflate",
        BrushMode::Grab => "Grab",
    }
}

fn sculpt_strength_range_for_mode(mode: &BrushMode) -> (f32, f32) {
    match mode {
        BrushMode::Grab => (0.1, 3.0),
        BrushMode::Add
        | BrushMode::Carve
        | BrushMode::Smooth
        | BrushMode::Flatten
        | BrushMode::Inflate => (0.01, 0.5),
    }
}

fn clamp_sculpt_strength_for_mode(mode: &BrushMode, value: f32) -> f32 {
    let (min_strength, max_strength) = sculpt_strength_range_for_mode(mode);
    value.clamp(min_strength, max_strength)
}

fn sculpt_symmetry_axis_from_id(axis_id: &str) -> Option<Option<u8>> {
    Some(match axis_id {
        "off" => None,
        "x" => Some(0),
        "y" => Some(1),
        "z" => Some(2),
        _ => return None,
    })
}

fn sculpt_symmetry_axis_id(axis: Option<u8>) -> &'static str {
    match axis {
        None => "off",
        Some(0) => "x",
        Some(1) => "y",
        Some(2) => "z",
        Some(_) => "off",
    }
}

fn sculpt_symmetry_axis_label(axis: Option<u8>) -> &'static str {
    match axis {
        None => "Off",
        Some(0) => "X",
        Some(1) => "Y",
        Some(2) => "Z",
        Some(_) => "Off",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AppBridge, DEFAULT_SCULPT_ENTRY_RESOLUTION, EMISSIVE_INTENSITY_MAX,
        LIGHT_INTENSITY_MAX, LIGHT_PROXIMITY_RANGE_MAX, LIGHT_RANGE_MIN,
        LIGHT_VOLUMETRIC_DENSITY_MAX, MATERIAL_FACTOR_MIN, PRIMITIVE_PARAMETER_MAX,
        PRIMITIVE_PARAMETER_MIN,
    };
    use crate::app_bridge::{AppScalarPropertySnapshot, AppVec3};
    use crate::app_bridge::workflows::{ImportDialogState, SculptConvertMode};
    use crate::graph::scene::{
        CsgOp, LightType, ModifierKind, NodeData, SdfPrimitive, MAX_SCENE_LIGHTS,
    };
    use crate::mesh_import::TriMesh;
    use glam::Vec3;

    #[test]
    fn scene_snapshot_includes_recursive_scene_tree() {
        let bridge = AppBridge::new();
        let snapshot = bridge.scene_snapshot();

        assert_eq!(
            snapshot.scene_tree_roots.len(),
            snapshot.top_level_nodes.len()
        );
        assert!(snapshot
            .scene_tree_roots
            .iter()
            .any(|node| !node.children.is_empty()));
    }

    #[test]
    fn scene_snapshot_omits_selected_node_properties_without_selection() {
        let bridge = AppBridge::new();

        assert!(bridge.scene_snapshot().selected_node_properties.is_none());
    }

    #[test]
    fn scene_snapshot_includes_export_settings() {
        let bridge = AppBridge::new();

        let export = bridge.scene_snapshot().export;
        assert_eq!(export.min_resolution, 16);
        assert_eq!(export.max_resolution, bridge.settings.max_export_resolution.max(16));
        assert_eq!(
            export.resolution,
            bridge
                .settings
                .export_resolution
                .clamp(16, bridge.settings.max_export_resolution.max(16))
        );
        assert_eq!(export.adaptive, bridge.settings.adaptive_export);
        assert_eq!(export.status.state, "idle");
        assert!(export.status.message.is_none());
        assert_eq!(export.presets.len(), 4);
        assert_eq!(export.presets[0].name, "Low");
        assert_eq!(export.presets[0].resolution, 64);
    }

    #[test]
    fn export_setting_commands_update_snapshot_and_clamp() {
        let mut bridge = AppBridge::new();

        let resolution = bridge.set_export_resolution(9_999);
        bridge.set_adaptive_export(true);

        let export = bridge.scene_snapshot().export;
        assert_eq!(resolution, 2048);
        assert_eq!(export.resolution, 2048);
        assert!(export.adaptive);
    }

    #[test]
    fn export_background_task_reports_completion_in_snapshot() {
        let mut bridge = AppBridge::new();
        let temp_path = std::env::temp_dir().join(format!(
            "sdf_modeler_bridge_export_{}.obj",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&temp_path);

        bridge.set_export_resolution(16);
        assert!(bridge.start_export_to_path(temp_path.clone()));
        assert_eq!(bridge.scene_snapshot().export.status.state, "in_progress");

        for _ in 0..200 {
            bridge.poll_export();
            if bridge.scene_snapshot().export.status.state == "idle" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let export = bridge.scene_snapshot().export;
        assert_eq!(export.status.state, "idle");
        assert_eq!(export.status.is_error, false);
        assert!(export
            .status
            .message
            .as_deref()
            .is_some_and(|message| message.contains("Exported OBJ")));
        assert!(temp_path.exists());

        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn import_background_task_reports_completion_in_snapshot() {
        let mut bridge = AppBridge::new();
        bridge.import_state.dialog = Some(ImportDialogState::new(
            TriMesh {
                vertices: vec![
                    Vec3::new(-0.5, 0.0, 0.0),
                    Vec3::new(0.5, 0.0, 0.0),
                    Vec3::new(0.0, 0.75, 0.0),
                ],
                triangles: vec![[0, 1, 2]],
            },
            "hero_mesh.obj".to_string(),
            128,
        ));
        bridge.import_state.dialog.as_mut().unwrap().set_use_auto(false);
        bridge.import_state.dialog.as_mut().unwrap().set_resolution(32);

        assert!(bridge.start_import());
        assert_eq!(bridge.scene_snapshot().import.status.state, "in_progress");

        for _ in 0..200 {
            bridge.poll_import();
            if bridge.scene_snapshot().import.status.state == "idle" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let import = bridge.scene_snapshot().import;
        assert_eq!(import.status.state, "idle");
        assert!(!import.status.is_error);
        assert!(import
            .status
            .message
            .as_deref()
            .is_some_and(|message| message.contains("Imported hero_mesh.obj as sculpt geometry")));
        assert_eq!(bridge.scene_snapshot().tool.active_tool_label, "Sculpt");
        assert_eq!(
            bridge
                .scene_snapshot()
                .selected_node
                .as_ref()
                .map(|node| node.kind_label.as_str()),
            Some("Sculpt")
        );
        let sculpt = bridge.scene_snapshot().sculpt;
        assert_eq!(sculpt.selected.as_ref().map(|selected| selected.node_name.as_str()), Some("Import"));
        assert_eq!(sculpt.session.as_ref().map(|session| session.brush_mode_id.as_str()), Some("add"));
    }

    #[test]
    fn sculpt_convert_entry_flow_updates_snapshot_and_selection() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.open_sculpt_convert_dialog_for_selected());
        let dialog = bridge
            .scene_snapshot()
            .sculpt_convert
            .dialog
            .expect("sculpt convert dialog");
        assert_eq!(dialog.target_node_id, sphere_id);
        assert_eq!(dialog.target_name, "Sphere");
        assert_eq!(dialog.mode_id, SculptConvertMode::BakeActiveNode.id());

        assert!(bridge.set_sculpt_convert_mode(SculptConvertMode::BakeActiveNode.id()));
        assert!(bridge.set_sculpt_convert_resolution(48));
        assert_eq!(
            bridge
                .scene_snapshot()
                .sculpt_convert
                .dialog
                .as_ref()
                .map(|dialog| dialog.resolution),
            Some(48)
        );
        assert!(bridge.start_sculpt_convert());

        let snapshot = bridge.scene_snapshot();
        assert_eq!(snapshot.tool.active_tool_label, "Sculpt");
        assert_eq!(
            snapshot.selected_node.as_ref().map(|node| node.kind_label.as_str()),
            Some("Sculpt")
        );
        assert_eq!(snapshot.sculpt_convert.status.state, "idle");
        assert!(snapshot
            .sculpt_convert
            .status
            .message
            .as_deref()
            .is_some_and(|message| message.contains("Converted Sphere to sculpt")));
    }

    #[test]
    fn selected_primitive_property_snapshot_includes_transform_material_and_parameters() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        let selected_properties = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected node properties");

        assert_eq!(selected_properties.node_id, sphere_id);
        assert_eq!(selected_properties.name, "Sphere");
        assert_eq!(selected_properties.kind_label, "Sphere");
        assert!(selected_properties.visible);
        assert!(!selected_properties.locked);

        let transform = selected_properties.transform.expect("transform snapshot");
        assert_eq!(transform.position_label, "Position");
        assert_eq!(transform.position, AppVec3::new(0.0, 0.0, 0.0));
        assert_eq!(transform.rotation_degrees, AppVec3::new(0.0, 0.0, 0.0));
        assert_eq!(transform.scale, None);

        let primitive = selected_properties.primitive.expect("primitive snapshot");
        assert_eq!(primitive.primitive_kind, "Sphere");
        assert_eq!(
            primitive.parameters,
            vec![AppScalarPropertySnapshot {
                key: "radius".to_string(),
                label: "Radius".to_string(),
                value: 1.0,
            }]
        );

        let material = selected_properties.material.expect("material snapshot");
        assert_eq!(material.color, AppVec3::new(0.8, 0.3, 0.2));
        assert_eq!(material.roughness, 0.5);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.emissive, AppVec3::new(0.0, 0.0, 0.0));
        assert_eq!(material.emissive_intensity, 0.0);
        assert_eq!(material.fresnel, 0.04);
    }

    #[test]
    fn selected_transform_property_snapshot_includes_translation_rotation_and_scale() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        let transform_id = bridge.create_transform();

        let selected_properties = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected node properties");

        assert_eq!(selected_properties.node_id, transform_id);
        assert_eq!(selected_properties.kind_label, "Transform");
        assert!(selected_properties.primitive.is_none());
        assert!(selected_properties.material.is_none());

        let transform = selected_properties.transform.expect("transform snapshot");
        assert_eq!(transform.position_label, "Translation");
        assert_eq!(transform.position, AppVec3::new(0.0, 0.0, 0.0));
        assert_eq!(transform.rotation_degrees, AppVec3::new(0.0, 0.0, 0.0));
        assert_eq!(transform.scale, Some(AppVec3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn selected_light_property_snapshot_includes_transform_wrapped_light() {
        let mut bridge = AppBridge::new();
        let transform_id = bridge.create_light(LightType::Spot);

        let selected_properties = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected node properties");

        assert_eq!(selected_properties.node_id, transform_id);
        assert_eq!(selected_properties.kind_label, "Transform");
        assert!(selected_properties.primitive.is_none());
        assert!(selected_properties.material.is_none());

        let light = selected_properties.light.expect("light snapshot");
        assert_eq!(light.transform_node_id, Some(transform_id));
        assert_eq!(light.light_type_id, "spot");
        assert_eq!(light.light_type_label, "Spot");
        assert!(light.supports_range);
        assert!(light.supports_spot_angle);
        assert!(light.supports_cookie);
        assert!(light.cookie_candidates.iter().any(|candidate| candidate.name == "Sphere"));
    }

    #[test]
    fn selected_light_commands_update_snapshot_and_clamp() {
        let mut bridge = AppBridge::new();
        bridge.create_light(LightType::Point);
        let cookie_node_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("cookie candidate");

        assert!(bridge.set_selected_light_intensity(99.0));
        assert!(bridge.set_selected_light_range(0.01));
        assert!(bridge.set_selected_light_volumetric(true));
        assert!(bridge.set_selected_light_volumetric_density(9.0));
        assert!(bridge.set_selected_light_proximity_mode("brighten"));
        assert!(bridge.set_selected_light_proximity_range(12.0));
        assert!(bridge.set_selected_light_cookie(cookie_node_id));
        assert!(bridge.set_selected_light_intensity_expression("sin(t * 2.0)"));
        assert!(bridge.set_selected_light_color_hue_expression("bad("));

        let light = bridge
            .scene_snapshot()
            .selected_node_properties
            .and_then(|properties| properties.light)
            .expect("light snapshot");
        assert_eq!(light.intensity, LIGHT_INTENSITY_MAX);
        assert_eq!(light.range, LIGHT_RANGE_MIN);
        assert!(light.volumetric);
        assert_eq!(light.volumetric_density, LIGHT_VOLUMETRIC_DENSITY_MAX);
        assert_eq!(light.proximity_mode_id, "brighten");
        assert_eq!(light.proximity_range, LIGHT_PROXIMITY_RANGE_MAX);
        assert_eq!(light.cookie_node_id, Some(cookie_node_id));
        assert_eq!(light.intensity_expression.as_deref(), Some("sin(t * 2.0)"));
        assert!(light.intensity_expression_error.is_none());
        assert_eq!(light.color_hue_expression.as_deref(), Some("bad("));
        assert!(light.color_hue_expression_error.is_some());
    }

    #[test]
    fn light_linking_snapshot_and_commands_round_trip_masks() {
        let mut bridge = AppBridge::new();
        bridge.create_light(LightType::Point);
        bridge.create_light(LightType::Spot);
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");

        let light_linking_before = bridge.scene_snapshot().light_linking;
        assert_eq!(light_linking_before.geometry_nodes.len(), 1);
        assert_eq!(light_linking_before.geometry_nodes[0].light_mask, 0xFF);
        assert_eq!(light_linking_before.max_light_count, MAX_SCENE_LIGHTS as u32);
        let first_light = light_linking_before
            .lights
            .first()
            .expect("first light target");

        assert!(bridge.set_node_light_link_enabled(
            sphere_id,
            first_light.light_node_id,
            false,
        ));
        let light_linking_after_toggle = bridge.scene_snapshot().light_linking;
        assert_eq!(
            light_linking_after_toggle.geometry_nodes[0].light_mask,
            0xFF & !first_light.mask_bit
        );

        assert!(bridge.set_node_light_mask(sphere_id, 0x03));
        assert_eq!(
            bridge.scene_snapshot().light_linking.geometry_nodes[0].light_mask,
            0x03
        );
    }

    #[test]
    fn tool_snapshot_includes_default_manipulator_state() {
        let bridge = AppBridge::new();

        let tool = bridge.scene_snapshot().tool;
        assert_eq!(tool.manipulator_mode_id, "translate");
        assert_eq!(tool.manipulator_mode_label, "Move");
        assert_eq!(tool.manipulator_space_id, "local");
        assert_eq!(tool.manipulator_space_label, "Local");
        assert!(!tool.manipulator_visible);
        assert!(!tool.can_reset_pivot);
        assert_eq!(tool.pivot_offset, AppVec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn manipulator_mode_and_space_commands_update_tool_snapshot() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_manipulator_mode("rotate"));
        bridge.toggle_manipulator_space();

        let tool = bridge.scene_snapshot().tool;
        assert_eq!(tool.manipulator_mode_id, "rotate");
        assert_eq!(tool.manipulator_space_id, "world");
        assert!(tool.manipulator_visible);
    }

    #[test]
    fn selected_sculpt_property_snapshot_includes_material_without_primitive_parameters() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        bridge.create_sculpt().expect("sculpt");

        let selected_properties = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected node properties");

        assert_eq!(selected_properties.kind_label, "Sculpt");
        assert!(selected_properties.primitive.is_none());

        let transform = selected_properties.transform.expect("transform snapshot");
        assert_eq!(transform.position_label, "Position");
        assert_eq!(transform.scale, None);

        let material = selected_properties.material.expect("material snapshot");
        assert_eq!(material.roughness, 0.5);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.fresnel, 0.04);
    }

    #[test]
    fn set_selected_primitive_parameter_updates_snapshot_and_clamps() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_selected_primitive_parameter("radius", 500.0));

        match &bridge.scene.nodes[&sphere_id].data {
            NodeData::Primitive { scale, .. } => assert_eq!(scale.x, PRIMITIVE_PARAMETER_MAX),
            _ => panic!("expected primitive"),
        }
        assert_eq!(
            bridge
                .scene_snapshot()
                .selected_node_properties
                .expect("selected properties")
                .primitive
                .expect("primitive properties")
                .parameters,
            vec![AppScalarPropertySnapshot {
                key: "radius".to_string(),
                label: "Radius".to_string(),
                value: PRIMITIVE_PARAMETER_MAX,
            }]
        );
    }

    #[test]
    fn set_selected_material_float_updates_snapshot_and_clamps() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_selected_material_float("roughness", -2.0));
        assert!(bridge.set_selected_material_float("emissive_intensity", 99.0));

        let material = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .material
            .expect("material properties");
        assert_eq!(material.roughness, MATERIAL_FACTOR_MIN);
        assert_eq!(material.emissive_intensity, EMISSIVE_INTENSITY_MAX);
    }

    #[test]
    fn set_selected_material_color_updates_snapshot_and_clamps() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_selected_material_color("color", -0.5, 0.4, 2.0));
        assert!(bridge.set_selected_material_color("emissive", 0.2, 1.7, 0.3));

        let material = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .material
            .expect("material properties");
        assert_eq!(material.color, AppVec3::new(0.0, 0.4, 1.0));
        assert_eq!(material.emissive, AppVec3::new(0.2, 1.0, 0.3));
    }

    #[test]
    fn set_selected_transform_position_updates_snapshot() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_selected_transform_position(1.5, -2.0, 3.25));

        let transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert_eq!(transform.position, AppVec3::new(1.5, -2.0, 3.25));
    }

    #[test]
    fn set_selected_transform_rotation_degrees_updates_snapshot() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        assert!(bridge.set_selected_transform_rotation_degrees(15.0, 45.0, 90.0));

        let transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert!((transform.rotation_degrees.x - 15.0).abs() < 1e-3);
        assert!((transform.rotation_degrees.y - 45.0).abs() < 1e-3);
        assert!((transform.rotation_degrees.z - 90.0).abs() < 1e-3);
    }

    #[test]
    fn set_selected_transform_scale_updates_transform_snapshot_and_clamps() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        bridge.create_transform();

        assert!(bridge.set_selected_transform_scale(-1.0, 2.0, 500.0));

        let transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert_eq!(
            transform.scale,
            Some(AppVec3::new(
                PRIMITIVE_PARAMETER_MIN,
                2.0,
                PRIMITIVE_PARAMETER_MAX
            ))
        );
    }

    #[test]
    fn pivot_and_manipulation_nudges_update_snapshots() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        bridge.create_transform();

        bridge.nudge_manipulator_pivot_offset(0.25, 0.0, 0.0);
        let tool = bridge.scene_snapshot().tool;
        assert!(tool.can_reset_pivot);
        assert_eq!(tool.pivot_offset, AppVec3::new(0.25, 0.0, 0.0));

        bridge.reset_manipulator_pivot();
        assert_eq!(
            bridge.scene_snapshot().tool.pivot_offset,
            AppVec3::new(0.0, 0.0, 0.0)
        );

        assert!(bridge.nudge_selected_translation(0.5, -0.25, 1.0));
        let moved_transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert_eq!(moved_transform.position, AppVec3::new(0.5, -0.25, 1.0));

        assert!(bridge.nudge_selected_rotation_degrees(15.0, 0.0, -30.0));
        let rotated_transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert!((rotated_transform.rotation_degrees.x - 15.0).abs() < 1e-3);
        assert!((rotated_transform.rotation_degrees.z + 30.0).abs() < 1e-3);

        assert!(bridge.nudge_selected_scale(-5.0, 1.0, 500.0));
        let scaled_transform = bridge
            .scene_snapshot()
            .selected_node_properties
            .expect("selected properties")
            .transform
            .expect("transform properties");
        assert_eq!(
            scaled_transform.scale,
            Some(AppVec3::new(
                PRIMITIVE_PARAMETER_MIN,
                2.0,
                PRIMITIVE_PARAMETER_MAX
            ))
        );
    }

    #[test]
    fn delete_selected_removes_unlocked_node() {
        let mut bridge = AppBridge::new();
        let node_id = bridge.add_box();

        bridge.delete_selected();

        assert!(!bridge.scene.nodes.contains_key(&node_id));
        assert_eq!(bridge.selected_node, None);
        assert_eq!(bridge.hovered_node, None);
    }

    #[test]
    fn delete_selected_keeps_locked_node() {
        let mut bridge = AppBridge::new();
        let node_id = bridge.add_box();
        bridge.scene.nodes.get_mut(&node_id).unwrap().locked = true;

        bridge.delete_selected();

        assert!(bridge.scene.nodes.contains_key(&node_id));
        assert_eq!(bridge.selected_node, Some(node_id));
    }

    #[test]
    fn undo_restores_previous_scene_and_selection() {
        let mut bridge = AppBridge::new();
        let initial_selected = bridge.selected_node;
        let initial_node_count = bridge.scene.nodes.len();

        let added_node = bridge.add_box();
        assert_eq!(bridge.selected_node, Some(added_node));
        assert!(bridge.scene_snapshot().history.can_undo);

        bridge.undo();

        assert_eq!(bridge.scene.nodes.len(), initial_node_count);
        assert_eq!(bridge.selected_node, initial_selected);
        assert_eq!(bridge.hovered_node, initial_selected);
        assert!(!bridge.scene.nodes.contains_key(&added_node));
        assert!(!bridge.scene_snapshot().history.can_undo);
        assert!(bridge.scene_snapshot().history.can_redo);
    }

    #[test]
    fn redo_restores_undone_scene_and_selection() {
        let mut bridge = AppBridge::new();
        let added_node = bridge.add_box();

        bridge.undo();
        assert!(bridge.scene_snapshot().history.can_redo);

        bridge.redo();

        assert!(bridge.scene.nodes.contains_key(&added_node));
        assert_eq!(bridge.selected_node, Some(added_node));
        assert_eq!(bridge.hovered_node, Some(added_node));
        assert!(bridge.scene_snapshot().history.can_undo);
        assert!(!bridge.scene_snapshot().history.can_redo);
    }

    #[test]
    fn duplicate_selected_creates_offset_copy_and_selects_it() {
        let mut bridge = AppBridge::new();
        let original_node = bridge.add_box();
        let original_position = match &bridge.scene.nodes[&original_node].data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        };

        let duplicated_node = bridge.duplicate_selected().expect("duplicate node");

        assert_ne!(duplicated_node, original_node);
        assert_eq!(bridge.selected_node, Some(duplicated_node));
        assert_eq!(bridge.hovered_node, Some(duplicated_node));
        assert!(bridge.scene_snapshot().history.can_undo);
        assert!(bridge.scene.nodes[&duplicated_node].name.ends_with(" Copy"));

        match &bridge.scene.nodes[&duplicated_node].data {
            NodeData::Primitive { position, .. } => {
                assert_eq!(position.x, original_position.x + 1.0);
                assert_eq!(position.y, original_position.y);
                assert_eq!(position.z, original_position.z);
            }
            _ => panic!("expected duplicated primitive"),
        }
    }

    #[test]
    fn undo_removes_duplicated_selection() {
        let mut bridge = AppBridge::new();
        let original_node = bridge.add_box();
        let duplicated_node = bridge.duplicate_selected().expect("duplicate node");

        bridge.undo();

        assert!(bridge.scene.nodes.contains_key(&original_node));
        assert!(!bridge.scene.nodes.contains_key(&duplicated_node));
        assert_eq!(bridge.selected_node, Some(original_node));
        assert_eq!(bridge.hovered_node, Some(original_node));
        assert!(bridge.scene_snapshot().history.can_redo);
    }

    #[test]
    fn rename_node_updates_snapshot_and_history() {
        let mut bridge = AppBridge::new();
        let node_id = bridge.add_box();

        assert!(bridge.rename_node(node_id, "  Hero Box  "));

        let snapshot = bridge.scene_snapshot();
        assert_eq!(bridge.scene.nodes[&node_id].name, "Hero Box");
        assert_eq!(
            snapshot
                .selected_node
                .as_ref()
                .map(|node| node.name.as_str()),
            Some("Hero Box")
        );
        assert_eq!(
            snapshot
                .scene_tree_roots
                .iter()
                .find(|node| node.id == node_id)
                .map(|node| node.name.as_str()),
            Some("Hero Box")
        );
        assert!(snapshot.history.can_undo);
    }

    #[test]
    fn rename_node_rejects_blank_names() {
        let mut bridge = AppBridge::new();
        let node_id = bridge.add_box();
        let original_name = bridge.scene.nodes[&node_id].name.clone();
        let history_before_rename = bridge.scene_snapshot().history.can_undo;

        assert!(!bridge.rename_node(node_id, "   "));

        let snapshot = bridge.scene_snapshot();
        assert_eq!(bridge.scene.nodes[&node_id].name, original_name);
        assert_eq!(snapshot.history.can_undo, history_before_rename);
    }

    #[test]
    fn create_transform_wraps_selected_node_and_selects_new_transform() {
        let mut bridge = AppBridge::new();
        let selected_node = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(selected_node));

        let transform_id = bridge.create_transform();

        assert_eq!(bridge.selected_node, Some(transform_id));
        assert_eq!(bridge.hovered_node, Some(transform_id));
        assert!(bridge.scene_snapshot().history.can_undo);
        match &bridge.scene.nodes[&transform_id].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(selected_node)),
            _ => panic!("expected transform"),
        }
        assert!(bridge.scene.top_level_nodes().contains(&transform_id));
        assert!(!bridge.scene.top_level_nodes().contains(&selected_node));
    }

    #[test]
    fn create_modifier_wraps_selected_node_and_selects_new_modifier() {
        let mut bridge = AppBridge::new();
        let selected_node = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(selected_node));

        let modifier_id = bridge.create_modifier(ModifierKind::Twist);

        assert_eq!(bridge.selected_node, Some(modifier_id));
        assert_eq!(bridge.hovered_node, Some(modifier_id));
        assert!(bridge.scene_snapshot().history.can_undo);
        match &bridge.scene.nodes[&modifier_id].data {
            NodeData::Modifier { kind, input, .. } => {
                assert_eq!(*kind, ModifierKind::Twist);
                assert_eq!(*input, Some(selected_node));
            }
            _ => panic!("expected modifier"),
        }
        assert!(bridge.scene.top_level_nodes().contains(&modifier_id));
        assert!(!bridge.scene.top_level_nodes().contains(&selected_node));
    }

    #[test]
    fn create_operation_uses_last_two_visible_geometry_roots() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        let hidden_box_id = bridge.add_box();
        bridge.toggle_node_visibility(hidden_box_id);
        let torus_id = bridge.add_torus();

        let operation_id = bridge.create_operation(CsgOp::Union);

        assert_eq!(bridge.selected_node, Some(operation_id));
        assert_eq!(bridge.hovered_node, Some(operation_id));
        assert!(bridge.scene_snapshot().history.can_undo);
        match &bridge.scene.nodes[&operation_id].data {
            NodeData::Operation {
                op, left, right, ..
            } => {
                assert_eq!(*op, CsgOp::Union);
                assert_eq!(*left, Some(sphere_id));
                assert_eq!(*right, Some(torus_id));
            }
            _ => panic!("expected operation"),
        }
    }

    #[test]
    fn create_light_selects_transform_wrapper() {
        let mut bridge = AppBridge::new();

        let transform_id = bridge.create_light(LightType::Point);

        assert_eq!(bridge.selected_node, Some(transform_id));
        assert_eq!(bridge.hovered_node, Some(transform_id));
        assert!(bridge.scene_snapshot().history.can_undo);
        match &bridge.scene.nodes[&transform_id].data {
            NodeData::Transform { input, .. } => {
                let light_id = input.expect("light child");
                match &bridge.scene.nodes[&light_id].data {
                    NodeData::Light { light_type, .. } => {
                        assert_eq!(*light_type, LightType::Point);
                    }
                    _ => panic!("expected light child"),
                }
            }
            _ => panic!("expected transform"),
        }
    }

    #[test]
    fn create_sculpt_wraps_selected_node_with_default_resolution() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));

        let sculpt_id = bridge.create_sculpt().expect("sculpt id");

        assert_eq!(bridge.selected_node, Some(sculpt_id));
        assert_eq!(bridge.hovered_node, Some(sculpt_id));
        assert!(bridge.scene_snapshot().history.can_undo);
        let sculpt = bridge.scene_snapshot().sculpt;
        assert_eq!(
            sculpt.selected.as_ref().map(|selected| selected.node_id),
            Some(sculpt_id)
        );
        assert_eq!(
            sculpt.session.as_ref().map(|session| session.node_id),
            Some(sculpt_id)
        );
        assert!(sculpt.can_stop);
        match &bridge.scene.nodes[&sculpt_id].data {
            NodeData::Sculpt {
                input,
                voxel_grid,
                desired_resolution,
                ..
            } => {
                assert_eq!(*input, Some(sphere_id));
                assert_eq!(voxel_grid.resolution, DEFAULT_SCULPT_ENTRY_RESOLUTION);
                assert_eq!(*desired_resolution, DEFAULT_SCULPT_ENTRY_RESOLUTION);
            }
            _ => panic!("expected sculpt"),
        }
    }

    #[test]
    fn create_sculpt_reuses_existing_sculpt_parent() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        let sculpt_id = bridge.create_sculpt().expect("initial sculpt");

        bridge.select_node(Some(sphere_id));
        let scene_node_count = bridge.scene.nodes.len();
        let can_undo_before_reuse = bridge.scene_snapshot().history.can_undo;

        let reused_id = bridge.create_sculpt().expect("reused sculpt");

        assert_eq!(reused_id, sculpt_id);
        assert_eq!(bridge.selected_node, Some(sculpt_id));
        assert_eq!(bridge.hovered_node, Some(sculpt_id));
        assert_eq!(bridge.scene.nodes.len(), scene_node_count);
        assert_eq!(
            bridge.scene_snapshot().history.can_undo,
            can_undo_before_reuse
        );
        assert_eq!(
            bridge.scene_snapshot().sculpt.session.as_ref().map(|session| session.node_id),
            Some(sculpt_id)
        );
    }

    #[test]
    fn sculpt_snapshot_commands_update_session_and_selected_resolution() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        let sculpt_id = bridge.create_sculpt().expect("sculpt id");

        assert!(bridge.set_sculpt_brush_mode("grab"));
        assert!(bridge.set_sculpt_brush_strength(9.0));
        assert!(bridge.set_sculpt_symmetry_axis("z"));
        assert!(bridge.set_selected_sculpt_resolution(999));

        let sculpt = bridge.scene_snapshot().sculpt;
        let session = sculpt.session.expect("active sculpt session");
        let selected = sculpt.selected.expect("selected sculpt");
        assert_eq!(session.node_id, sculpt_id);
        assert_eq!(session.brush_mode_id, "grab");
        assert_eq!(session.brush_mode_label, "Grab");
        assert_eq!(session.brush_strength, 3.0);
        assert_eq!(session.symmetry_axis_id, "z");
        assert_eq!(session.symmetry_axis_label, "Z");
        assert_eq!(
            selected.desired_resolution,
            bridge.settings.max_sculpt_resolution.max(16)
        );
    }

    #[test]
    fn sculpt_session_can_stop_and_resume_selected() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        let sculpt_id = bridge.create_sculpt().expect("sculpt id");

        assert!(bridge.stop_sculpting());
        let stopped = bridge.scene_snapshot().sculpt;
        assert!(!stopped.can_stop);
        assert!(stopped.session.is_none());
        assert!(stopped.can_resume_selected);
        assert_eq!(bridge.scene_snapshot().tool.active_tool_label, "Select");

        assert!(bridge.resume_sculpting_selected());
        let resumed = bridge.scene_snapshot().sculpt;
        assert_eq!(
            resumed.session.as_ref().map(|session| session.node_id),
            Some(sculpt_id)
        );
        assert!(resumed.can_stop);
        assert!(!resumed.can_resume_selected);
        assert_eq!(bridge.scene_snapshot().tool.active_tool_label, "Sculpt");
    }

    #[test]
    fn sculpt_session_syncs_when_active_sculpt_is_deleted() {
        let mut bridge = AppBridge::new();
        let sphere_id = bridge
            .scene
            .top_level_nodes()
            .into_iter()
            .find(|node_id| {
                matches!(
                    bridge.scene.nodes[node_id].data,
                    NodeData::Primitive {
                        kind: SdfPrimitive::Sphere,
                        ..
                    }
                )
            })
            .expect("default sphere");
        bridge.select_node(Some(sphere_id));
        let sculpt_id = bridge.create_sculpt().expect("sculpt id");

        bridge.delete_selected();

        let sculpt = bridge.scene_snapshot().sculpt;
        assert!(sculpt.session.is_none());
        assert!(sculpt.selected.is_none());
        assert!(!bridge.scene.nodes.contains_key(&sculpt_id));
        assert_eq!(bridge.scene_snapshot().tool.active_tool_label, "Select");
    }

    #[test]
    fn reset_scene_clears_undo_and_redo_history() {
        let mut bridge = AppBridge::new();
        bridge.add_box();
        bridge.undo();

        assert!(bridge.scene_snapshot().history.can_redo);

        bridge.reset_scene();

        assert!(!bridge.scene_snapshot().history.can_undo);
        assert!(!bridge.scene_snapshot().history.can_redo);
    }
}
