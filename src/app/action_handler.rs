use eframe::egui;
use glam::Vec3;

use crate::desktop_dialogs::FileDialogSelection;
use crate::graph::history::{History, RestoreSnapshot, RestoreState};
use crate::graph::scene::{NodeData, Scene};
use crate::sculpt::SculptState;

use super::actions::{Action, LightingPreset, SculptConvertMode};
use super::state::SculptConvertDialog;
use super::SdfApp;

/// Maximum storage buffer binding size configured for wgpu (128MB).
const MAX_STORAGE_BUFFER_BYTES: u64 = 1 << 27;

/// Maximum safe sculpt resolution per node (cube root of max voxels at 4 bytes each).
/// 320^3 * 4 = 131,072,000 bytes < 128MB limit. 322 is the true max but 320 is a clean value.
pub const MAX_SCULPT_RESOLUTION: u32 = 320;

fn push_platform_unsupported_toast(app: &mut SdfApp, feature: &str) {
    app.ui.toasts.push(super::Toast {
        message: format!("{feature} is not supported on this platform yet"),
        is_error: true,
        created: crate::compat::Instant::now(),
        duration: crate::compat::Duration::from_secs(4),
    });
}

impl SdfApp {
    /// Process all collected actions. This is the single mutation point — the
    /// equivalent of a Redux reducer. All structural state changes flow through
    /// here, making the data flow explicit and easy to trace.
    pub(super) fn process_actions(&mut self, actions: Vec<Action>, ctx: &egui::Context) {
        for action in actions {
            match action {
                // ── Scene ────────────────────────────────────────────
                Action::NewScene => {
                    self.doc.scene = Scene::new();
                    self.doc.history = History::new();
                    self.ui.node_graph_state.clear_selection();
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.doc.sculpt_state = SculptState::new_inactive();
                    self.ui.isolation_state = None;
                    self.doc.soloed_light = None;
                    self.gpu.current_structure_key = 0;
                    self.gpu.buffer_dirty = true;
                    self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                    self.persistence.scene_dirty = false;
                    self.persistence.current_file_path = None;
                }
                Action::OpenProject =>
                {
                    #[cfg(not(target_arch = "wasm32"))]
                    match crate::io::open_dialog() {
                        FileDialogSelection::Selected(path) => {
                            self.load_project_from_path(&path);
                        }
                        FileDialogSelection::Unsupported => {
                            push_platform_unsupported_toast(self, "Opening project files");
                        }
                        FileDialogSelection::Cancelled => {}
                    }
                }
                Action::OpenRecentProject(ref recent_path) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let path = std::path::PathBuf::from(recent_path);
                        if !self.load_project_from_path(&path) {
                            self.settings.recent_files.retain(|p| p != recent_path);
                            self.settings.save();
                        }
                    }
                }
                Action::SaveProject => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let path = if let Some(path) = self.persistence.current_file_path.clone() {
                            Some(path)
                        } else {
                            match crate::io::save_dialog() {
                                FileDialogSelection::Selected(path) => Some(path),
                                FileDialogSelection::Unsupported => {
                                    push_platform_unsupported_toast(self, "Saving project files");
                                    None
                                }
                                FileDialogSelection::Cancelled => None,
                            }
                        };
                        if let Some(path) = path {
                            if let Err(e) = crate::io::save_project_with_sculpt(
                                &self.doc.scene,
                                &self.doc.camera,
                                &self.settings.render,
                                Some(&self.doc.sculpt_state),
                                &path,
                            ) {
                                log::error!("Failed to save project: {}", e);
                            } else {
                                self.persistence.current_file_path = Some(path.clone());
                                self.persistence.saved_fingerprint =
                                    self.doc.scene.data_fingerprint();
                                self.persistence.scene_dirty = false;
                                self.settings.add_recent_file(&path.to_string_lossy());
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        crate::io::web_save_project_with_sculpt(
                            &self.doc.scene,
                            &self.doc.camera,
                            &self.settings.render,
                            Some(&self.doc.sculpt_state),
                        );
                        self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                        self.persistence.scene_dirty = false;
                    }
                }

                // ── Selection ────────────────────────────────────────
                Action::SaveNodePreset(node_id) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let default_name = self
                            .doc
                            .scene
                            .nodes
                            .get(&node_id)
                            .map(|n| format!("{}.sdfpreset", n.name.replace(' ', "_")))
                            .unwrap_or_else(|| "node_preset.sdfpreset".to_string());
                        match crate::desktop_dialogs::save_node_preset_dialog(&default_name) {
                            FileDialogSelection::Selected(path) => {
                                match crate::io::save_subtree_preset(
                                    &self.doc.scene,
                                    node_id,
                                    &path,
                                ) {
                                    Ok(()) => {
                                        self.ui.toasts.push(super::Toast {
                                            message: format!("Saved preset: {}", path.display()),
                                            is_error: false,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(4),
                                        });
                                    }
                                    Err(e) => {
                                        log::error!("Failed to save node preset: {}", e);
                                        self.ui.toasts.push(super::Toast {
                                            message: format!("Failed to save preset: {}", e),
                                            is_error: true,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(5),
                                        });
                                    }
                                }
                            }
                            FileDialogSelection::Unsupported => {
                                push_platform_unsupported_toast(self, "Saving node presets");
                            }
                            FileDialogSelection::Cancelled => {}
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.ui.toasts.push(super::Toast {
                            message: "Node presets are not supported in web builds".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
                    }
                }
                Action::LoadNodePreset => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match crate::desktop_dialogs::load_node_preset_dialog() {
                            FileDialogSelection::Selected(path) => {
                                match crate::io::load_subtree_preset(&mut self.doc.scene, &path) {
                                    Ok(root_id) => {
                                        self.ui.node_graph_state.select_single(root_id);
                                        self.ui.node_graph_state.needs_initial_rebuild = true;
                                        self.gpu.buffer_dirty = true;
                                        self.ui.toasts.push(super::Toast {
                                            message: format!("Loaded preset: {}", path.display()),
                                            is_error: false,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(4),
                                        });
                                    }
                                    Err(e) => {
                                        log::error!("Failed to load node preset: {}", e);
                                        self.ui.toasts.push(super::Toast {
                                            message: format!("Failed to load preset: {}", e),
                                            is_error: true,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(5),
                                        });
                                    }
                                }
                            }
                            FileDialogSelection::Unsupported => {
                                push_platform_unsupported_toast(self, "Loading node presets");
                            }
                            FileDialogSelection::Cancelled => {}
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.ui.toasts.push(super::Toast {
                            message: "Node presets are not supported in web builds".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
                    }
                }
                Action::AddReferenceImage => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match crate::desktop_dialogs::reference_image_dialog() {
                            FileDialogSelection::Selected(path) => {
                                match self.ui.reference_images.add_from_path(ctx, &path) {
                                    Ok(()) => {
                                        self.ui.toasts.push(super::Toast {
                                            message: format!(
                                                "Loaded reference image: {}",
                                                path.display()
                                            ),
                                            is_error: false,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(4),
                                        });
                                    }
                                    Err(e) => {
                                        log::error!("Failed to load reference image: {}", e);
                                        self.ui.toasts.push(super::Toast {
                                            message: format!("Failed to load image: {}", e),
                                            is_error: true,
                                            created: crate::compat::Instant::now(),
                                            duration: crate::compat::Duration::from_secs(5),
                                        });
                                    }
                                }
                            }
                            FileDialogSelection::Unsupported => {
                                push_platform_unsupported_toast(self, "Picking reference images");
                            }
                            FileDialogSelection::Cancelled => {}
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.ui.toasts.push(super::Toast {
                            message: "Reference images are not supported in web builds".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
                    }
                }
                Action::RemoveReferenceImage(index) => {
                    self.ui.reference_images.remove(index);
                }
                Action::ToggleReferenceImageVisibility(index) => {
                    self.ui.reference_images.toggle_visibility(index);
                }
                Action::ToggleAllReferenceImages => {
                    self.ui.reference_images.toggle_all_visibility();
                }
                Action::Select(id) => {
                    if let Some(node_id) = id {
                        self.ui.node_graph_state.select_single(node_id);
                    } else {
                        self.ui.node_graph_state.clear_selection();
                    }
                    self.gpu.buffer_dirty = true;
                }
                Action::DeleteSelected => {
                    let locked = self
                        .ui
                        .node_graph_state
                        .selected
                        .and_then(|id| self.doc.scene.nodes.get(&id))
                        .is_some_and(|n| n.locked);
                    if !locked {
                        self.delete_selected();
                    }
                }
                Action::DeleteNode(id) => {
                    let locked = self.doc.scene.nodes.get(&id).is_some_and(|n| n.locked);
                    if !locked {
                        self.doc.scene.remove_node(id);
                        if self.ui.node_graph_state.selected == Some(id) {
                            self.ui.node_graph_state.clear_selection();
                        } else {
                            self.ui.node_graph_state.selected_set.remove(&id);
                        }
                        if self.doc.soloed_light == Some(id) {
                            self.doc.soloed_light = None;
                        }
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.doc.sculpt_state = SculptState::new_inactive();
                        self.gpu.buffer_dirty = true;
                    }
                }

                // ── Clipboard ────────────────────────────────────────
                Action::Copy => {
                    self.doc.clipboard_node = self.ui.node_graph_state.selected;
                }
                Action::Paste => {
                    if let Some(src) = self.doc.clipboard_node {
                        self.duplicate_and_offset(src);
                    }
                }
                Action::Duplicate => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        self.duplicate_and_offset(sel);
                    }
                }

                // ── History ──────────────────────────────────────────
                Action::Undo => {
                    if let Some(restored) = self
                        .doc
                        .history
                        .undo(&self.doc.scene, self.ui.node_graph_state.selected)
                    {
                        self.apply_history_restore(restored);
                    }
                }
                Action::Redo => {
                    if let Some(restored) = self
                        .doc
                        .history
                        .redo(&self.doc.scene, self.ui.node_graph_state.selected)
                    {
                        self.apply_history_restore(restored);
                    }
                }

                // ── Camera ───────────────────────────────────────────
                Action::FocusSelected => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        let parent_map = self.doc.scene.build_parent_map();
                        let (center, radius) =
                            self.doc.scene.compute_subtree_sphere(sel, &parent_map);
                        self.doc
                            .camera
                            .focus_on(Vec3::new(center[0], center[1], center[2]), radius.max(0.5));
                    }
                }
                Action::FrameAll => {
                    let bounds = self.doc.scene.compute_bounds();
                    let center = Vec3::new(
                        (bounds.0[0] + bounds.1[0]) * 0.5,
                        (bounds.0[1] + bounds.1[1]) * 0.5,
                        (bounds.0[2] + bounds.1[2]) * 0.5,
                    );
                    let half = Vec3::new(
                        (bounds.1[0] - bounds.0[0]) * 0.5,
                        (bounds.1[1] - bounds.0[1]) * 0.5,
                        (bounds.1[2] - bounds.0[2]) * 0.5,
                    );
                    self.doc.camera.focus_on(center, half.length().max(0.5));
                }
                Action::CameraFront => self.doc.camera.start_transition(0.0, 0.0, 0.0),
                Action::CameraTop => {
                    self.doc
                        .camera
                        .start_transition(0.0, std::f32::consts::FRAC_PI_2, 0.0)
                }
                Action::CameraRight => {
                    self.doc
                        .camera
                        .start_transition(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
                }
                Action::CameraBack => {
                    self.doc
                        .camera
                        .start_transition(std::f32::consts::PI, 0.0, 0.0)
                }
                Action::CameraLeft => {
                    self.doc
                        .camera
                        .start_transition(-std::f32::consts::FRAC_PI_2, 0.0, 0.0)
                }
                Action::CameraBottom => {
                    self.doc
                        .camera
                        .start_transition(0.0, -std::f32::consts::FRAC_PI_2, 0.0)
                }
                Action::ToggleOrtho => self.doc.camera.toggle_ortho(),

                // ── Tools ────────────────────────────────────────────
                Action::SetTool(tool) => {
                    self.doc.active_tool = tool;
                    match tool {
                        crate::sculpt::ActiveTool::Select => {
                            self.doc.sculpt_state = SculptState::new_inactive();
                            self.async_state.last_sculpt_hit = None;
                            self.async_state.lazy_brush_pos = None;
                        }
                        crate::sculpt::ActiveTool::Sculpt => {
                            if let Some(sel) = self.ui.node_graph_state.selected {
                                if self
                                    .doc
                                    .scene
                                    .nodes
                                    .get(&sel)
                                    .is_some_and(|n| matches!(n.data, NodeData::Sculpt { .. }))
                                {
                                    let extent = self.scene_avg_extent();
                                    self.doc
                                        .sculpt_state
                                        .activate_preserving_session(sel, Some(extent));
                                    self.ensure_brush_settings_tab();
                                }
                            }
                        }
                    }
                }
                Action::SetGizmoMode(mode) => {
                    if self.gizmo.mode == mode && self.gizmo.gizmo_visible {
                        self.gizmo.gizmo_visible = false;
                    } else {
                        self.gizmo.mode = mode;
                        self.gizmo.gizmo_visible = true;
                    }
                }
                Action::ToggleGizmoSpace => {
                    self.gizmo.space = match self.gizmo.space {
                        crate::ui::gizmo::GizmoSpace::Local => crate::ui::gizmo::GizmoSpace::World,
                        crate::ui::gizmo::GizmoSpace::World => crate::ui::gizmo::GizmoSpace::Local,
                    };
                }
                Action::ResetPivot => {
                    self.gizmo.pivot_offset = Vec3::ZERO;
                }

                // ── Sculpt entry ──────────────────────────────────────
                Action::EnterSculptMode => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        if let Some(node) = self.doc.scene.nodes.get(&sel) {
                            if matches!(node.data, NodeData::Sculpt { .. }) {
                                // Case 1: already a sculpt node — activate immediately
                                let extent = self.scene_avg_extent();
                                self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
                                self.doc
                                    .sculpt_state
                                    .activate_preserving_session(sel, Some(extent));
                                self.ensure_brush_settings_tab();
                            } else {
                                // Case 2: check for sculpt parent
                                let parent_map = self.doc.scene.build_parent_map();
                                if let Some(sculpt_id) =
                                    self.doc.scene.find_sculpt_parent(sel, &parent_map)
                                {
                                    let extent = self.scene_avg_extent();
                                    self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
                                    self.doc
                                        .sculpt_state
                                        .activate_preserving_session(sculpt_id, Some(extent));
                                    self.ui.node_graph_state.select_single(sculpt_id);
                                    self.ensure_brush_settings_tab();
                                } else {
                                    // Case 3: non-sculpt node — open convert dialog
                                    self.ui.sculpt_convert_dialog =
                                        Some(SculptConvertDialog::new(sel));
                                }
                            }
                        }
                    } else {
                        // Nothing selected
                        self.ui.toasts.push(super::Toast {
                            message: "Select a node to sculpt".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
                    }
                }
                Action::ShowSculptConvertDialog { target } => {
                    self.ui.sculpt_convert_dialog = Some(SculptConvertDialog::new(target));
                }
                Action::CommitSculptConvert {
                    target,
                    mode,
                    resolution,
                } => {
                    self.ui.sculpt_convert_dialog = None;
                    let baking = !matches!(self.async_state.bake_status, super::BakeStatus::Idle);
                    if baking {
                        self.ui.toasts.push(super::Toast {
                            message: "A bake is already in progress".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
                    } else if !self.validate_sculpt_resolution(resolution) {
                        // Resolution too high — toast already shown by validate fn
                    } else {
                        // Determine the subtree root and flatten flag based on convert mode
                        let (subtree_root, flatten) = match mode {
                            SculptConvertMode::BakeWholeScene => {
                                // Walk up from target to the topmost ancestor
                                let parent_map = self.doc.scene.build_parent_map();
                                let mut root = target;
                                while let Some(&pid) = parent_map.get(&root) {
                                    root = pid;
                                }
                                (root, false)
                            }
                            SculptConvertMode::BakeWholeSceneFlatten => {
                                let parent_map = self.doc.scene.build_parent_map();
                                let mut root = target;
                                while let Some(&pid) = parent_map.get(&root) {
                                    root = pid;
                                }
                                (root, true)
                            }
                            SculptConvertMode::BakeActiveNode => (target, false),
                        };
                        // Get color from the target node (or default white)
                        let color = self
                            .doc
                            .scene
                            .nodes
                            .get(&subtree_root)
                            .and_then(|n| n.data.material().map(|material| material.base_color))
                            .unwrap_or(Vec3::new(0.8, 0.8, 0.8));

                        let req = super::BakeRequest {
                            subtree_root,
                            resolution,
                            color,
                            existing_sculpt: None,
                            flatten,
                        };

                        if flatten {
                            self.start_async_bake(req, ctx);
                        } else {
                            self.apply_instant_displacement_bake(req);
                            // Activate sculpt on the newly created sculpt node
                            // The bake created a sculpt node above subtree_root — find it
                            let parent_map = self.doc.scene.build_parent_map();
                            if let Some(&sculpt_id) = parent_map.get(&subtree_root) {
                                if self
                                    .doc
                                    .scene
                                    .nodes
                                    .get(&sculpt_id)
                                    .is_some_and(|n| matches!(n.data, NodeData::Sculpt { .. }))
                                {
                                    let extent = self.scene_avg_extent();
                                    self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
                                    self.doc
                                        .sculpt_state
                                        .activate_preserving_session(sculpt_id, Some(extent));
                                    self.ui.node_graph_state.select_single(sculpt_id);
                                    self.ensure_brush_settings_tab();
                                }
                            }
                        }
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.gpu.buffer_dirty = true;
                    }
                }

                // ── Scene mutations (structural) ─────────────────────
                Action::IncreaseSculptDetail(node_id) => {
                    self.increase_sculpt_detail(node_id);
                }
                Action::DecreaseSculptDetail(node_id) => {
                    self.decrease_sculpt_detail(node_id);
                }
                Action::RemeshSculptAtCurrentDetail(node_id) => {
                    self.remesh_sculpt_at_current_detail(node_id);
                }
                Action::ExpandSculptVolume(node_id) => {
                    self.expand_sculpt_volume(node_id);
                }
                Action::FitSculptVolume(node_id) => {
                    self.fit_sculpt_volume(node_id);
                }
                Action::CreatePrimitive(prim) => {
                    let id = self.doc.scene.create_primitive(prim);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateOperation { op, left, right } => {
                    let id = self.doc.scene.create_operation(op.clone(), left, right);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateTransform { input } => {
                    let id = self.doc.scene.create_transform(input);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateReroute { input } => {
                    let id = self.doc.scene.create_reroute(input);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateModifier { kind, input } => {
                    let id = self.doc.scene.create_modifier(kind, input);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateLight(light_type) => {
                    let (_light_id, transform_id) = self.doc.scene.create_light(light_type);
                    self.ui.node_graph_state.select_single(transform_id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(transform_id);
                    self.gpu.buffer_dirty = true;
                }
                Action::InsertModifierAbove { target, kind } => {
                    let new_id = self.doc.scene.insert_modifier_above(target, kind);
                    self.ui.node_graph_state.select_single(new_id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(new_id);
                    self.gpu.buffer_dirty = true;
                }
                Action::InsertTransformAbove { target } => {
                    let new_id = self.doc.scene.insert_transform_above(target);
                    self.ui.node_graph_state.select_single(new_id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(new_id);
                    self.gpu.buffer_dirty = true;
                }
                Action::ToggleVisibility(id) => {
                    self.doc.scene.toggle_visibility(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::ToggleLock(id) => {
                    if let Some(node) = self.doc.scene.nodes.get_mut(&id) {
                        node.locked = !node.locked;
                    }
                }
                Action::SwapChildren(id) => {
                    self.doc.scene.swap_children(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::ReparentNode {
                    dragged,
                    new_parent,
                } => {
                    self.doc.scene.reparent(dragged, new_parent);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.gpu.buffer_dirty = true;
                }
                Action::RenameNode { id, name } => {
                    if let Some(node) = self.doc.scene.nodes.get_mut(&id) {
                        node.name = name;
                    }
                }

                // ── Graph connections ─────────────────────────────────
                Action::SetLeftChild { parent, child } => {
                    self.doc.scene.set_left_child(parent, child);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.gpu.buffer_dirty = true;
                }
                Action::SetRightChild { parent, child } => {
                    self.doc.scene.set_right_child(parent, child);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.gpu.buffer_dirty = true;
                }
                Action::SetSculptInput { parent, child } => {
                    self.doc.scene.set_sculpt_input(parent, child);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.gpu.buffer_dirty = true;
                }

                // ── Bake / Export ────────────────────────────────────
                Action::RequestBake(req) => {
                    let baking = !matches!(self.async_state.bake_status, super::BakeStatus::Idle);
                    if baking {
                        // Already baking — skip
                    } else if !self.validate_sculpt_resolution(req.resolution) {
                        // Resolution too high — toast already shown by validate fn
                    } else if req.flatten {
                        self.start_async_bake(req, ctx);
                    } else {
                        self.apply_instant_displacement_bake(req);
                    }
                }
                Action::ShowExportDialog => {
                    self.ui.show_export_dialog = true;
                }
                Action::ImportMesh => {
                    let importing =
                        !matches!(self.async_state.import_status, super::ImportStatus::Idle);
                    if !importing && self.ui.import_dialog.is_none() {
                        self.open_import_dialog();
                    }
                }
                Action::CommitImport { resolution } => {
                    let importing =
                        !matches!(self.async_state.import_status, super::ImportStatus::Idle);
                    if !importing {
                        if !self.validate_sculpt_resolution(resolution) {
                            // Resolution too high — toast already shown, clear dialog
                            self.ui.import_dialog = None;
                        } else {
                            // start_import_voxelize takes dialog via .take()
                            self.start_import_voxelize(resolution, ctx);
                        }
                    }
                }
                Action::TakeScreenshot => {
                    self.take_screenshot();
                }

                // ── Viewport ─────────────────────────────────────────
                Action::ToggleIsolation => {
                    if self.ui.isolation_state.is_some() {
                        // Exit isolation: restore previous hidden set
                        if let Some(iso) = self.ui.isolation_state.take() {
                            self.doc.scene.hidden_nodes = iso.pre_hidden;
                        }
                    } else if let Some(sel) = self.ui.node_graph_state.selected {
                        // Enter isolation: hide everything not in subtree + ancestors
                        let pre_hidden = self.doc.scene.hidden_nodes.clone();
                        let subtree = self.doc.scene.collect_subtree(sel);
                        let parent_map = self.doc.scene.build_parent_map();
                        // Walk ancestors
                        let mut ancestors = std::collections::HashSet::new();
                        let mut cur = sel;
                        while let Some(&parent) = parent_map.get(&cur) {
                            ancestors.insert(parent);
                            cur = parent;
                        }
                        // Hide all nodes not in subtree or ancestor chain
                        let all_ids: Vec<_> = self.doc.scene.nodes.keys().copied().collect();
                        self.doc.scene.hidden_nodes.clear();
                        for id in all_ids {
                            if !subtree.contains(&id) && !ancestors.contains(&id) && id != sel {
                                self.doc.scene.hidden_nodes.insert(id);
                            }
                        }
                        self.ui.isolation_state = Some(super::state::IsolationState {
                            pre_hidden,
                            isolated_node: sel,
                        });
                    }
                    self.gpu.buffer_dirty = true;
                }
                Action::CycleShadingMode => {
                    self.settings.render.shading_mode = self.settings.render.shading_mode.cycle();
                    self.settings.save();
                    self.gpu.buffer_dirty = true;
                }
                Action::ToggleTurntable => {
                    self.ui.turntable_active = !self.ui.turntable_active;
                }

                // ── Properties ─────────────────────────────────────
                Action::ToggleDistanceReadout => {
                    self.ui.show_distance_readout = !self.ui.show_distance_readout;
                }
                Action::ToggleMeasurementTool => {
                    self.ui.measurement_mode = !self.ui.measurement_mode;
                    self.ui.measurement_points.clear();
                }
                Action::CopyProperties => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        if let Some(node) = self.doc.scene.nodes.get(&sel) {
                            let clip = node
                                .data
                                .material()
                                .cloned()
                                .map(|material| super::state::PropertyClipboard { material });
                            if let Some(c) = clip {
                                self.ui.property_clipboard = Some(c);
                                self.ui.toasts.push(super::Toast {
                                    message: "Properties copied".into(),
                                    is_error: false,
                                    created: crate::compat::Instant::now(),
                                    duration: crate::compat::Duration::from_secs(2),
                                });
                            }
                        }
                    }
                }
                Action::PasteProperties => {
                    if let Some(ref clip) = self.ui.property_clipboard.clone() {
                        if let Some(sel) = self.ui.node_graph_state.selected {
                            if let Some(node) = self.doc.scene.nodes.get_mut(&sel) {
                                let applied = if let Some(material) = node.data.material_mut() {
                                    *material = clip.material.clone();
                                    true
                                } else {
                                    false
                                };
                                if applied {
                                    self.gpu.buffer_dirty = true;
                                    self.ui.toasts.push(super::Toast {
                                        message: "Properties pasted".into(),
                                        is_error: false,
                                        created: crate::compat::Instant::now(),
                                        duration: crate::compat::Duration::from_secs(2),
                                    });
                                }
                            }
                        }
                    }
                }

                // ── Camera bookmarks ───────────────────────────────
                Action::SaveBookmark(slot) => {
                    if slot < 9 {
                        while self.settings.bookmarks.len() < 9 {
                            self.settings.bookmarks.push(None);
                        }
                        self.settings.bookmarks[slot] = Some(crate::settings::CameraBookmark {
                            yaw: self.doc.camera.yaw,
                            pitch: self.doc.camera.pitch,
                            roll: self.doc.camera.roll,
                            distance: self.doc.camera.distance,
                            target: [
                                self.doc.camera.target.x,
                                self.doc.camera.target.y,
                                self.doc.camera.target.z,
                            ],
                            orthographic: self.doc.camera.orthographic,
                        });
                        self.settings.save();
                        self.ui.toasts.push(super::Toast {
                            message: format!("Bookmark {} saved", slot + 1),
                            is_error: false,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(2),
                        });
                    }
                }
                Action::RestoreBookmark(slot) => {
                    if slot < self.settings.bookmarks.len() {
                        if let Some(ref bm) = self.settings.bookmarks[slot].clone() {
                            self.doc.camera.yaw = bm.yaw;
                            self.doc.camera.pitch = bm.pitch;
                            self.doc.camera.roll = bm.roll;
                            self.doc.camera.distance = bm.distance;
                            self.doc.camera.target =
                                Vec3::new(bm.target[0], bm.target[1], bm.target[2]);
                            self.doc.camera.orthographic = bm.orthographic;
                            self.ui.toasts.push(super::Toast {
                                message: format!("Bookmark {} restored", slot + 1),
                                is_error: false,
                                created: crate::compat::Instant::now(),
                                duration: crate::compat::Duration::from_secs(2),
                            });
                        }
                    }
                }

                // ── UI toggles ───────────────────────────────────────
                Action::ToggleDebug => {
                    self.ui.show_debug = !self.ui.show_debug;
                }
                Action::ToggleHelp => {
                    self.ui.show_help = !self.ui.show_help;
                }
                Action::ToggleSettings => {
                    self.ui.show_settings = !self.ui.show_settings;
                }
                Action::ToggleCommandPalette => {
                    self.ui.command_palette_open = !self.ui.command_palette_open;
                    if self.ui.command_palette_open {
                        self.ui.command_palette_query.clear();
                        self.ui.command_palette_selected = 0;
                    }
                }
                Action::ShowToast { message, is_error } => {
                    self.ui.toasts.push(super::Toast {
                        message,
                        is_error,
                        created: crate::compat::Instant::now(),
                        duration: if is_error {
                            crate::compat::Duration::from_secs(6)
                        } else {
                            crate::compat::Duration::from_secs(4)
                        },
                    });
                }

                // ── Workspace ────────────────────────────────────────
                Action::SetWorkspace(preset) => {
                    use crate::app::actions::WorkspacePreset;
                    use crate::ui::dock;
                    self.ui.dock_state = match preset {
                        WorkspacePreset::Modeling => dock::create_dock_state(),
                        WorkspacePreset::Sculpting => dock::create_dock_sculpting(),
                        WorkspacePreset::Rendering => dock::create_dock_rendering(),
                    };
                }

                // ── Light linking ────────────────────────────────────
                Action::SetLightMask { node_id, mask } => {
                    self.doc.scene.set_light_mask(node_id, mask);
                    self.gpu.buffer_dirty = true;
                }
                Action::ToggleLightMaskBit {
                    node_id,
                    light_slot,
                    enabled,
                } => {
                    if light_slot < 8 {
                        let current = self.doc.scene.get_light_mask(node_id);
                        let new_mask = if enabled {
                            current | (1 << light_slot)
                        } else {
                            current & !(1 << light_slot)
                        };
                        self.doc.scene.set_light_mask(node_id, new_mask);
                        self.gpu.buffer_dirty = true;
                    }
                }

                // ── Light Solo ───────────────────────────────────────
                Action::SoloLight(target) => {
                    match target {
                        Some(id) => {
                            if self.doc.soloed_light == Some(id) {
                                // Already soloed — toggle off
                                self.doc.soloed_light = None;
                            } else {
                                self.doc.soloed_light = Some(id);
                            }
                        }
                        None => {
                            self.doc.soloed_light = None;
                        }
                    }
                    self.gpu.buffer_dirty = true;
                }

                // ── Light Cookie ─────────────────────────────────────
                Action::SetLightCookie { light_id, cookie } => {
                    // Validate: cookie must be a Primitive or Operation node (not Light/Transform/etc.)
                    let valid = cookie.is_none_or(|cookie_id| {
                        self.doc.scene.nodes.get(&cookie_id).is_some_and(|n| {
                            matches!(
                                n.data,
                                NodeData::Primitive { .. } | NodeData::Operation { .. }
                            )
                        })
                    });
                    if valid {
                        if let Some(node) = self.doc.scene.nodes.get_mut(&light_id) {
                            if let NodeData::Light {
                                ref mut cookie_node,
                                ..
                            } = node.data
                            {
                                *cookie_node = cookie;
                            }
                        }
                        // Cookie changes affect shader codegen (new cookie_sdf function)
                        self.gpu.current_structure_key = 0;
                        self.gpu.buffer_dirty = true;
                    }
                }

                // ── Lighting presets ─────────────────────────────────
                Action::ApplyLightingPreset(preset) => {
                    apply_lighting_preset_to_scene(
                        &mut self.doc.scene,
                        &mut self.settings.render,
                        preset,
                    );
                    self.settings.save();
                    self.gpu.buffer_dirty = true;
                }

                // ── Settings / GPU ───────────────────────────────────
                Action::SettingsChanged => {
                    self.settings.save();
                    self.gpu.current_structure_key = 0;
                    self.gpu.buffer_dirty = true;
                }
                Action::MarkBufferDirty => {
                    self.gpu.buffer_dirty = true;
                }
            }
        }
    }

    /// Ensure the Brush Settings tab is visible in the dock when sculpt mode activates.
    fn apply_history_restore(&mut self, restore: RestoreSnapshot) {
        match restore.state {
            RestoreState::Scene(scene) => {
                self.doc.scene = scene;
            }
            RestoreState::Sculpt {
                node_id,
                voxel_data,
            } => {
                let Some(node) = self.doc.scene.nodes.get_mut(&node_id) else {
                    return;
                };
                let NodeData::Sculpt { voxel_grid, .. } = &mut node.data else {
                    return;
                };
                voxel_grid.data = voxel_data;
            }
        }

        if let Some(id) = restore.selected {
            self.ui.node_graph_state.select_single(id);
        } else {
            self.ui.node_graph_state.clear_selection();
        }
        self.ui.node_graph_state.needs_initial_rebuild = true;
        self.ui.isolation_state = None;
        self.async_state.last_sculpt_hit = None;
        self.async_state.lazy_brush_pos = None;
        self.async_state.hover_world_pos = None;
        self.async_state.cursor_over_geometry = false;
        self.async_state.sculpt_dragging = false;
        self.clear_sculpt_runtime_cache();
        self.gpu.buffer_dirty = true;
    }

    /// Ensure the Brush Settings tab is visible in the dock when sculpt mode activates.
    /// Check if the requested sculpt resolution fits within the GPU storage buffer limit.
    /// Returns true if valid, false (with toast error) if too large.
    fn validate_sculpt_resolution(&mut self, resolution: u32) -> bool {
        let voxel_bytes = (resolution as u64).pow(3) * 4;
        if voxel_bytes > MAX_STORAGE_BUFFER_BYTES {
            let mem_mb = voxel_bytes as f64 / (1024.0 * 1024.0);
            self.ui.toasts.push(super::Toast {
                message: format!(
                    "Resolution {}^3 requires {:.0} MB — exceeds GPU buffer limit of {} MB. Max safe resolution is {}.",
                    resolution, mem_mb, MAX_STORAGE_BUFFER_BYTES / (1024 * 1024), MAX_SCULPT_RESOLUTION
                ),
                is_error: true,
                created: crate::compat::Instant::now(),
                duration: crate::compat::Duration::from_secs(6),
            });
            return false;
        }
        true
    }

    /// Compute average half-extent of the scene bounding box for adaptive brush sizing.
    pub(super) fn scene_avg_extent(&self) -> f32 {
        let (min, max) = self.doc.scene.compute_bounds();
        ((max[0] - min[0]) + (max[1] - min[1]) + (max[2] - min[2])) / 6.0
    }

    pub(super) fn ensure_brush_settings_tab(&mut self) {
        use crate::ui::dock::Tab;
        if self.ui.dock_state.find_tab(&Tab::BrushSettings).is_none() {
            // Add BrushSettings as a sibling tab next to Properties (not into the viewport).
            if let Some((surface_idx, node_idx, _tab_idx)) =
                self.ui.dock_state.find_tab(&Tab::Properties)
            {
                self.ui.dock_state[surface_idx][node_idx].append_tab(Tab::BrushSettings);
            } else {
                // Fallback: no Properties tab found, add next to SceneTree
                if let Some((surface_idx, node_idx, _tab_idx)) =
                    self.ui.dock_state.find_tab(&Tab::SceneTree)
                {
                    self.ui.dock_state[surface_idx][node_idx].append_tab(Tab::BrushSettings);
                } else {
                    self.ui.dock_state.push_to_focused_leaf(Tab::BrushSettings);
                }
            }
        }
    }

    /// Load a project file and update all relevant state.
    pub(super) fn load_project_from_path(&mut self, path: &std::path::Path) -> bool {
        match crate::io::load_project(&path.to_path_buf()) {
            Ok(project) => {
                self.doc.scene = project.scene;
                self.doc.camera = project.camera;
                self.doc.sculpt_state = project
                    .sculpt_state
                    .map(|persisted| SculptState::from_persisted(persisted, &self.doc.scene))
                    .unwrap_or_else(SculptState::new_inactive);
                self.doc.active_tool = if self.doc.sculpt_state.is_active() {
                    crate::sculpt::ActiveTool::Sculpt
                } else {
                    crate::sculpt::ActiveTool::Select
                };
                if let Some(render_config) = project.render_config {
                    self.settings.render = render_config;
                    self.gpu.last_environment_fingerprint = 0;
                    self.settings.save();
                }
                self.doc.history = History::new();
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.gpu.current_structure_key = 0;
                self.gpu.buffer_dirty = true;
                self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                self.persistence.scene_dirty = false;
                self.persistence.current_file_path = Some(path.to_path_buf());
                self.settings.add_recent_file(&path.to_string_lossy());
                true
            }
            Err(e) => {
                log::error!("Failed to load project: {}", e);
                false
            }
        }
    }

    /// Duplicate a subtree and offset the clone's position.
    fn duplicate_and_offset(&mut self, source_id: crate::graph::scene::NodeId) {
        if !self.doc.scene.nodes.contains_key(&source_id) {
            return;
        }
        if let Some(new_id) = self.doc.scene.duplicate_subtree(source_id) {
            if let Some(node) = self.doc.scene.nodes.get_mut(&new_id) {
                match &mut node.data {
                    NodeData::Primitive {
                        ref mut position, ..
                    }
                    | NodeData::Sculpt {
                        ref mut position, ..
                    } => {
                        position.x += 1.0;
                    }
                    _ => {}
                }
            }
            self.ui.node_graph_state.select_single(new_id);
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.gpu.buffer_dirty = true;
        }
    }
}

/// Apply a lighting preset by finding Key/Fill Light nodes in the scene
/// and updating their properties. Also sets ambient and sky colors.
fn apply_lighting_preset_to_scene(
    scene: &mut Scene,
    render_config: &mut crate::settings::RenderConfig,
    preset: LightingPreset,
) {
    // Preset definitions: (key_color, key_intensity, key_dir, fill_color, fill_intensity, fill_dir, ambient, sky_horizon, sky_zenith)
    let (
        key_color,
        key_intensity,
        key_dir,
        fill_color,
        fill_intensity,
        fill_dir,
        ambient,
        sky_horizon,
        sky_zenith,
    ) = match preset {
        LightingPreset::Studio => (
            Vec3::new(1.0, 0.98, 0.95), // warm white key
            1.5,
            Vec3::new(0.9593, 0.5990, 0.8957),
            Vec3::new(0.85, 0.9, 1.0), // cool blue-white fill
            0.4,
            Vec3::new(-0.5400, 0.2128, -0.7362),
            0.05,
            [0.7, 0.8, 0.95],
            [0.2, 0.3, 0.6],
        ),
        LightingPreset::Outdoor => (
            Vec3::new(1.0, 0.95, 0.85), // warm sunlight
            2.0,
            Vec3::new(0.8221, 0.7508, 0.7608),
            Vec3::new(0.6, 0.75, 1.0), // sky blue fill
            0.6,
            Vec3::new(-0.3228, 0.1596, -0.5564),
            0.08,
            [0.85, 0.9, 1.0],
            [0.35, 0.55, 0.9],
        ),
        LightingPreset::Dramatic => (
            Vec3::new(1.0, 0.85, 0.7), // warm amber key
            2.5,
            Vec3::new(0.8623, 0.4871, 0.6877),
            Vec3::new(0.4, 0.5, 0.7), // dim cool fill
            0.15,
            Vec3::new(-0.5669, -0.1107, -0.7449),
            0.02,
            [0.15, 0.1, 0.1],
            [0.05, 0.05, 0.15],
        ),
        LightingPreset::Flat => (
            Vec3::new(1.0, 1.0, 1.0), // neutral white key
            1.0,
            Vec3::new(0.7163, 0.8171, 0.6860),
            Vec3::new(1.0, 1.0, 1.0), // neutral white fill
            0.8,
            Vec3::new(-0.5190, 0.2633, -0.6474),
            0.15,
            [0.9, 0.9, 0.9],
            [0.7, 0.7, 0.8],
        ),
    };

    // Update sky colors (ambient is now controlled by scene Ambient Light node)
    render_config.sky_horizon = sky_horizon;
    render_config.sky_zenith = sky_zenith;

    // Find and update Key Light, Fill Light, and Ambient Light nodes
    let node_ids: Vec<_> = scene.nodes.keys().copied().collect();
    for id in node_ids {
        let name = scene.nodes.get(&id).map(|n| n.name.as_str()).unwrap_or("");
        if name == "Key Light" {
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Light {
                    ref mut color,
                    ref mut intensity,
                    ..
                } = &mut node.data
                {
                    *color = key_color;
                    *intensity = key_intensity;
                }
            }
            // Update the Key Light Transform's rotation (direction)
            if let Some(transform_id) = find_parent_transform(scene, id) {
                if let Some(t_node) = scene.nodes.get_mut(&transform_id) {
                    if let NodeData::Transform {
                        ref mut rotation, ..
                    } = &mut t_node.data
                    {
                        *rotation = key_dir;
                    }
                }
            }
        } else if name == "Fill Light" {
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Light {
                    ref mut color,
                    ref mut intensity,
                    ..
                } = &mut node.data
                {
                    *color = fill_color;
                    *intensity = fill_intensity;
                }
            }
            if let Some(transform_id) = find_parent_transform(scene, id) {
                if let Some(t_node) = scene.nodes.get_mut(&transform_id) {
                    if let NodeData::Transform {
                        ref mut rotation, ..
                    } = &mut t_node.data
                    {
                        *rotation = fill_dir;
                    }
                }
            }
        } else if name == "Ambient Light" {
            if let Some(node) = scene.nodes.get_mut(&id) {
                if let NodeData::Light {
                    ref mut intensity, ..
                } = &mut node.data
                {
                    *intensity = ambient;
                }
            }
        }
    }
}

/// Find the Transform node that parents a given node (has `input = Some(child_id)`).
fn find_parent_transform(
    scene: &Scene,
    child_id: crate::graph::scene::NodeId,
) -> Option<crate::graph::scene::NodeId> {
    scene.nodes.iter().find_map(|(&id, node)| {
        if let NodeData::Transform {
            input: Some(inp), ..
        } = &node.data
        {
            if *inp == child_id {
                return Some(id);
            }
        }
        None
    })
}
