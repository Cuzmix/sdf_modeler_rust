use crate::gpu::camera::Camera;
use crate::graph::history::History;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::sculpt::{ActiveTool, SculptState};

use super::commands::{CoreCommand, CoreCommandResult, NodeTransformPatch};
use super::types::{CoreAsyncState, CoreSelection, CoreSnapshot};

pub struct AppCoreInit {
    pub scene: Scene,
    pub history: History,
    pub camera: Camera,
    pub selection: CoreSelection,
    pub active_tool: ActiveTool,
    pub sculpt_state: SculptState,
    pub async_state: CoreAsyncState,
    pub soloed_light: Option<NodeId>,
    pub show_debug: bool,
    pub show_settings: bool,
}

/// UI-agnostic application core state.
///
/// This mirrors mutable document state from the egui app layer while exposing
/// framework-neutral command/snapshot interfaces.
pub struct AppCore {
    pub scene: Scene,
    pub history: History,
    pub camera: Camera,
    pub selection: CoreSelection,
    pub active_tool: ActiveTool,
    pub sculpt_state: SculptState,
    pub async_state: CoreAsyncState,
    pub soloed_light: Option<NodeId>,
    pub show_debug: bool,
    pub show_settings: bool,
}

impl AppCore {
    pub fn from_init(init: AppCoreInit) -> Self {
        Self {
            scene: init.scene,
            history: init.history,
            camera: init.camera,
            selection: init.selection,
            active_tool: init.active_tool,
            sculpt_state: init.sculpt_state,
            async_state: init.async_state,
            soloed_light: init.soloed_light,
            show_debug: init.show_debug,
            show_settings: init.show_settings,
        }
    }

    pub fn snapshot(&self) -> CoreSnapshot {
        CoreSnapshot {
            node_count: self.scene.nodes.len(),
            hidden_node_count: self.scene.hidden_nodes.len(),
            selected_count: self.selection.set.len(),
            selected_primary: self.selection.primary,
            active_tool: match self.active_tool {
                ActiveTool::Select => "Select".to_string(),
                ActiveTool::Sculpt => "Sculpt".to_string(),
            },
            undo_count: self.history.undo_count(),
            redo_count: self.history.redo_count(),
            camera_target: self.camera.target,
            camera_distance: self.camera.distance,
            bake_in_progress: self.async_state.bake_in_progress,
            export_in_progress: self.async_state.export_in_progress,
            import_in_progress: self.async_state.import_in_progress,
            show_debug: self.show_debug,
            show_settings: self.show_settings,
        }
    }

    pub fn apply_command(&mut self, command: CoreCommand) -> CoreCommandResult {
        match command {
            CoreCommand::Undo => self.handle_undo(),
            CoreCommand::Redo => self.handle_redo(),
            non_history_command => self.handle_with_history(non_history_command),
        }
    }

    fn handle_with_history(&mut self, command: CoreCommand) -> CoreCommandResult {
        self.history
            .begin_frame(&self.scene, self.selection.primary);
        let result = self.handle_non_history_command(command);
        self.history
            .end_frame(&self.scene, self.selection.primary, false);
        result
    }

    fn handle_non_history_command(&mut self, command: CoreCommand) -> CoreCommandResult {
        let mut result = CoreCommandResult {
            handled: true,
            ..CoreCommandResult::default()
        };

        match command {
            CoreCommand::NewScene => {
                self.scene = Scene::new();
                self.history = History::new();
                self.selection.clear();
                self.sculpt_state = SculptState::Inactive;
                self.soloed_light = None;
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.clear_isolation = true;
                result.toast_message = Some("New scene".to_string());
            }
            CoreCommand::Select(id) => {
                if let Some(node_id) = id {
                    self.selection.select_single(node_id);
                } else {
                    self.selection.clear();
                }
                result.buffer_dirty = true;
            }
            CoreCommand::ToggleSelect(node_id) => {
                if self.selection.set.contains(&node_id) {
                    self.selection.remove(node_id);
                } else {
                    self.selection.set.insert(node_id);
                    self.selection.primary = Some(node_id);
                }
                result.buffer_dirty = true;
            }
            CoreCommand::SetActiveTool(tool) => {
                self.active_tool = tool;
                if tool == ActiveTool::Select {
                    self.sculpt_state = SculptState::Inactive;
                }
                result.toast_message = Some(format!(
                    "Tool: {}",
                    match tool {
                        ActiveTool::Select => "Select",
                        ActiveTool::Sculpt => "Sculpt",
                    }
                ));
            }
            CoreCommand::ToggleDebug => {
                self.show_debug = !self.show_debug;
                result.toast_message = Some(format!(
                    "Debug {}",
                    if self.show_debug {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));
            }
            CoreCommand::ToggleSettings => {
                self.show_settings = !self.show_settings;
                result.toast_message = Some(format!(
                    "Settings {}",
                    if self.show_settings {
                        "opened"
                    } else {
                        "closed"
                    }
                ));
            }
            CoreCommand::CreatePrimitive(prim) => {
                let id = self.scene.create_primitive(prim);
                self.selection.select_single(id);
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.pending_center_node = Some(id);
            }
            CoreCommand::CreateOperation { op, left, right } => {
                let id = self.scene.create_operation(op, left, right);
                self.selection.select_single(id);
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.pending_center_node = Some(id);
            }
            CoreCommand::CreateTransform { input } => {
                let id = self.scene.create_transform(input);
                self.selection.select_single(id);
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.pending_center_node = Some(id);
            }
            CoreCommand::CreateModifier { kind, input } => {
                let id = self.scene.create_modifier(kind, input);
                self.selection.select_single(id);
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.pending_center_node = Some(id);
            }
            CoreCommand::CreateLight(light_type) => {
                let (_light_id, transform_id) = self.scene.create_light(light_type);
                self.selection.select_single(transform_id);
                result.buffer_dirty = true;
                result.needs_graph_rebuild = true;
                result.pending_center_node = Some(transform_id);
            }
            CoreCommand::SoloSelectedLight => {
                let selected_light_id = self.selection.primary.and_then(|selected_id| {
                    let selected_node = self.scene.nodes.get(&selected_id)?;
                    match &selected_node.data {
                        NodeData::Light { .. } => Some(selected_id),
                        NodeData::Transform {
                            input: Some(input_id),
                            ..
                        } => self.scene.nodes.get(input_id).and_then(|input_node| {
                            if matches!(&input_node.data, NodeData::Light { .. }) {
                                Some(*input_id)
                            } else {
                                None
                            }
                        }),
                        _ => None,
                    }
                });

                if let Some(light_id) = selected_light_id {
                    if self.soloed_light == Some(light_id) {
                        self.soloed_light = None;
                        result.toast_message = Some("Light solo cleared".to_string());
                    } else {
                        self.soloed_light = Some(light_id);
                        let light_name = self
                            .scene
                            .nodes
                            .get(&light_id)
                            .map(|node| node.name.clone())
                            .unwrap_or_else(|| format!("#{}", light_id));
                        result.toast_message = Some(format!("Solo light: {light_name}"));
                    }
                    result.buffer_dirty = true;
                } else {
                    result.toast_message =
                        Some("Select a light or its transform to toggle solo".to_string());
                }
            }
            CoreCommand::ClearSoloLight => {
                if self.soloed_light.is_some() {
                    self.soloed_light = None;
                    result.buffer_dirty = true;
                    result.toast_message = Some("Light solo cleared".to_string());
                }
            }
            CoreCommand::DeleteSelected => {
                if let Some(sel) = self.selection.primary {
                    let locked = self.scene.nodes.get(&sel).is_some_and(|node| node.locked);
                    if !locked {
                        self.scene.remove_node(sel);
                        self.selection.clear();
                        if self
                            .soloed_light
                            .is_some_and(|light_id| !self.scene.nodes.contains_key(&light_id))
                        {
                            self.soloed_light = None;
                        }
                        result.buffer_dirty = true;
                        result.needs_graph_rebuild = true;
                        result.deactivate_sculpt = true;
                    }
                }
            }
            CoreCommand::DeleteNode(id) => {
                let locked = self.scene.nodes.get(&id).is_some_and(|node| node.locked);
                if !locked {
                    self.scene.remove_node(id);
                    self.selection.remove(id);
                    if self
                        .soloed_light
                        .is_some_and(|light_id| !self.scene.nodes.contains_key(&light_id))
                    {
                        self.soloed_light = None;
                    }
                    result.buffer_dirty = true;
                    result.needs_graph_rebuild = true;
                    result.deactivate_sculpt = true;
                }
            }
            CoreCommand::ToggleSelectedVisibility => {
                if let Some(selected_id) = self.selection.primary {
                    self.scene.toggle_visibility(selected_id);
                    result.buffer_dirty = true;
                    result.needs_graph_rebuild = true;
                }
            }
            CoreCommand::RenameNode { id, name } => {
                let trimmed_name = name.trim();
                if !trimmed_name.is_empty() {
                    if let Some(node) = self.scene.nodes.get_mut(&id) {
                        node.name = trimmed_name.to_string();
                        result.needs_graph_rebuild = true;
                    }
                }
            }
            CoreCommand::RenameSelected(name) => {
                if let Some(selected_id) = self.selection.primary {
                    let trimmed_name = name.trim();
                    if !trimmed_name.is_empty() {
                        if let Some(node) = self.scene.nodes.get_mut(&selected_id) {
                            node.name = trimmed_name.to_string();
                            result.needs_graph_rebuild = true;
                        }
                    }
                }
            }
            CoreCommand::FocusSelected => {
                if let Some(selected_id) = self.selection.primary {
                    let parent_map = self.scene.build_parent_map();
                    let (center, radius) =
                        self.scene.compute_subtree_sphere(selected_id, &parent_map);
                    self.camera.focus_on(
                        glam::Vec3::new(center[0], center[1], center[2]),
                        radius.max(0.5),
                    );
                }
            }
            CoreCommand::ApplyNodeTransformPatch(patch) => {
                if self.apply_transform_patch(patch) {
                    result.buffer_dirty = true;
                }
            }
            CoreCommand::Undo | CoreCommand::Redo => {
                result.handled = false;
            }
        }

        result
    }

    fn handle_undo(&mut self) -> CoreCommandResult {
        let mut result = CoreCommandResult {
            handled: true,
            ..CoreCommandResult::default()
        };
        if let Some((restored_scene, restored_sel)) =
            self.history.undo(&self.scene, self.selection.primary)
        {
            self.scene = restored_scene;
            if let Some(id) = restored_sel {
                self.selection.select_single(id);
            } else {
                self.selection.clear();
            }
            result.buffer_dirty = true;
            result.needs_graph_rebuild = true;
            result.clear_isolation = true;
            result.toast_message = Some("Undo".to_string());
        }
        result
    }

    fn handle_redo(&mut self) -> CoreCommandResult {
        let mut result = CoreCommandResult {
            handled: true,
            ..CoreCommandResult::default()
        };
        if let Some((restored_scene, restored_sel)) =
            self.history.redo(&self.scene, self.selection.primary)
        {
            self.scene = restored_scene;
            if let Some(id) = restored_sel {
                self.selection.select_single(id);
            } else {
                self.selection.clear();
            }
            result.buffer_dirty = true;
            result.needs_graph_rebuild = true;
            result.clear_isolation = true;
            result.toast_message = Some("Redo".to_string());
        }
        result
    }

    fn apply_transform_patch(&mut self, patch: NodeTransformPatch) -> bool {
        let Some(node) = self.scene.nodes.get_mut(&patch.node_id) else {
            return false;
        };

        let mut changed = false;

        match &mut node.data {
            NodeData::Primitive {
                position,
                rotation,
                scale,
                ..
            } => {
                if let Some(set_position) = patch.set_position {
                    *position = set_position;
                    changed = true;
                }
                if let Some(set_rotation) = patch.set_rotation {
                    *rotation = set_rotation;
                    changed = true;
                }
                if let Some(set_scale) = patch.set_scale {
                    *scale = set_scale;
                    changed = true;
                }
                if patch.add_position != glam::Vec3::ZERO {
                    *position += patch.add_position;
                    changed = true;
                }
                if patch.add_rotation != glam::Vec3::ZERO {
                    *rotation += patch.add_rotation;
                    changed = true;
                }
                if patch.add_scale != glam::Vec3::ZERO {
                    *scale += patch.add_scale;
                    changed = true;
                }
            }
            NodeData::Sculpt {
                position, rotation, ..
            } => {
                if let Some(set_position) = patch.set_position {
                    *position = set_position;
                    changed = true;
                }
                if let Some(set_rotation) = patch.set_rotation {
                    *rotation = set_rotation;
                    changed = true;
                }
                if patch.add_position != glam::Vec3::ZERO {
                    *position += patch.add_position;
                    changed = true;
                }
                if patch.add_rotation != glam::Vec3::ZERO {
                    *rotation += patch.add_rotation;
                    changed = true;
                }
            }
            NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } => {
                if let Some(set_position) = patch.set_position {
                    *translation = set_position;
                    changed = true;
                }
                if let Some(set_rotation) = patch.set_rotation {
                    *rotation = set_rotation;
                    changed = true;
                }
                if let Some(set_scale) = patch.set_scale {
                    *scale = set_scale;
                    changed = true;
                }
                if patch.add_position != glam::Vec3::ZERO {
                    *translation += patch.add_position;
                    changed = true;
                }
                if patch.add_rotation != glam::Vec3::ZERO {
                    *rotation += patch.add_rotation;
                    changed = true;
                }
                if patch.add_scale != glam::Vec3::ZERO {
                    *scale += patch.add_scale;
                    changed = true;
                }
            }
            _ => {}
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::{CoreCommand, CoreSelection, NodeTransformPatch};
    use crate::graph::scene::SdfPrimitive;

    fn new_core() -> AppCore {
        AppCore::from_init(AppCoreInit {
            scene: Scene::new(),
            history: History::new(),
            camera: Camera::default(),
            selection: CoreSelection::default(),
            active_tool: ActiveTool::Select,
            sculpt_state: SculptState::Inactive,
            async_state: CoreAsyncState::default(),
            soloed_light: None,
            show_debug: false,
            show_settings: false,
        })
    }

    fn primitive_node_id(core: &AppCore) -> NodeId {
        *core
            .scene
            .nodes
            .iter()
            .find_map(|(id, node)| {
                if matches!(node.data, NodeData::Primitive { .. }) {
                    Some(id)
                } else {
                    None
                }
            })
            .expect("expected primitive node")
    }

    fn key_light_transform_node_id(core: &AppCore) -> NodeId {
        *core
            .scene
            .nodes
            .iter()
            .find_map(|(id, node)| {
                if node.name == "Key Light Transform" {
                    Some(id)
                } else {
                    None
                }
            })
            .expect("expected key light transform")
    }

    #[test]
    fn select_updates_primary_and_set() {
        let mut core = new_core();
        let id = primitive_node_id(&core);
        let result = core.apply_command(CoreCommand::Select(Some(id)));
        assert!(result.handled);
        assert!(result.buffer_dirty);
        assert_eq!(core.selection.primary, Some(id));
        assert!(core.selection.set.contains(&id));
    }

    #[test]
    fn toggle_select_adds_and_removes_node() {
        let mut core = new_core();
        let id = primitive_node_id(&core);

        let add_result = core.apply_command(CoreCommand::ToggleSelect(id));
        assert!(add_result.handled);
        assert!(add_result.buffer_dirty);
        assert_eq!(core.selection.primary, Some(id));
        assert!(core.selection.set.contains(&id));

        let remove_result = core.apply_command(CoreCommand::ToggleSelect(id));
        assert!(remove_result.handled);
        assert!(remove_result.buffer_dirty);
        assert_eq!(core.selection.primary, None);
        assert!(!core.selection.set.contains(&id));
    }

    #[test]
    fn toggle_selected_visibility_hides_selected_node() {
        let mut core = new_core();
        let id = primitive_node_id(&core);
        core.apply_command(CoreCommand::Select(Some(id)));

        let result = core.apply_command(CoreCommand::ToggleSelectedVisibility);
        assert!(result.handled);
        assert!(result.buffer_dirty);
        assert!(result.needs_graph_rebuild);
        assert!(core.scene.is_hidden(id));
    }

    #[test]
    fn rename_selected_updates_node_name() {
        let mut core = new_core();
        let id = primitive_node_id(&core);
        core.apply_command(CoreCommand::Select(Some(id)));

        let result = core.apply_command(CoreCommand::RenameSelected("Primary Sphere".to_string()));
        assert!(result.handled);
        assert!(result.needs_graph_rebuild);

        let node = core.scene.nodes.get(&id).expect("selected node exists");
        assert_eq!(node.name, "Primary Sphere");
    }

    #[test]
    fn focus_selected_updates_camera_target() {
        let mut core = new_core();
        let id = core.scene.create_primitive(SdfPrimitive::Box);
        if let Some(node) = core.scene.nodes.get_mut(&id) {
            if let NodeData::Primitive { position, .. } = &mut node.data {
                *position = glam::Vec3::new(6.0, 2.0, -3.0);
            }
        }
        core.apply_command(CoreCommand::Select(Some(id)));

        let result = core.apply_command(CoreCommand::FocusSelected);
        assert!(result.handled);

        let target = core.camera.target;
        assert!((target.x - 6.0).abs() < 0.01);
        assert!((target.y - 2.0).abs() < 0.01);
        assert!((target.z + 3.0).abs() < 0.01);
    }
    #[test]
    fn create_delete_and_undo_redo_roundtrip() {
        let mut core = new_core();
        let initial_count = core.scene.nodes.len();

        let create_result = core.apply_command(CoreCommand::CreatePrimitive(SdfPrimitive::Box));
        assert!(create_result.handled);
        assert!(create_result.needs_graph_rebuild);
        let created = core.selection.primary.expect("new node should be selected");
        assert_eq!(core.scene.nodes.len(), initial_count + 1);

        let delete_result = core.apply_command(CoreCommand::DeleteSelected);
        assert!(delete_result.handled);
        assert_eq!(core.scene.nodes.len(), initial_count);
        assert_eq!(core.selection.primary, None);

        let undo_result = core.apply_command(CoreCommand::Undo);
        assert!(undo_result.handled);
        assert_eq!(core.scene.nodes.len(), initial_count + 1);
        assert_eq!(core.selection.primary, Some(created));

        let redo_result = core.apply_command(CoreCommand::Redo);
        assert!(redo_result.handled);
        assert_eq!(core.scene.nodes.len(), initial_count);
        assert_eq!(core.selection.primary, None);
    }

    #[test]
    fn solo_selected_light_toggles_from_light_transform() {
        let mut core = new_core();
        let transform_id = key_light_transform_node_id(&core);
        core.apply_command(CoreCommand::Select(Some(transform_id)));

        let result = core.apply_command(CoreCommand::SoloSelectedLight);
        assert!(result.handled);
        assert!(result.buffer_dirty);

        let light_id = match &core
            .scene
            .nodes
            .get(&transform_id)
            .expect("transform exists")
            .data
        {
            NodeData::Transform {
                input: Some(light_id),
                ..
            } => *light_id,
            _ => panic!("expected transform with light input"),
        };
        assert_eq!(core.soloed_light, Some(light_id));

        let toggle_result = core.apply_command(CoreCommand::SoloSelectedLight);
        assert!(toggle_result.handled);
        assert!(toggle_result.buffer_dirty);
        assert_eq!(core.soloed_light, None);
    }

    #[test]
    fn clear_solo_light_resets_state() {
        let mut core = new_core();
        let transform_id = key_light_transform_node_id(&core);
        core.apply_command(CoreCommand::Select(Some(transform_id)));
        core.apply_command(CoreCommand::SoloSelectedLight);
        assert!(core.soloed_light.is_some());

        let clear_result = core.apply_command(CoreCommand::ClearSoloLight);
        assert!(clear_result.handled);
        assert!(clear_result.buffer_dirty);
        assert_eq!(core.soloed_light, None);
    }

    #[test]
    fn transform_patch_updates_transform_node_translation() {
        let mut core = new_core();
        let transform_id = core.scene.create_transform(None);
        core.apply_command(CoreCommand::Select(Some(transform_id)));

        let mut patch = NodeTransformPatch::new(transform_id);
        patch.set_position = Some(glam::Vec3::new(1.0, 2.0, 3.0));
        patch.add_position = glam::Vec3::new(0.5, -0.5, 1.0);

        let result = core.apply_command(CoreCommand::ApplyNodeTransformPatch(patch));
        assert!(result.handled);
        assert!(result.buffer_dirty);

        let node = core
            .scene
            .nodes
            .get(&transform_id)
            .expect("transform exists");
        match &node.data {
            NodeData::Transform { translation, .. } => {
                assert_eq!(*translation, glam::Vec3::new(1.5, 1.5, 4.0));
            }
            _ => panic!("expected transform node"),
        }
    }
}
