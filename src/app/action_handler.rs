use eframe::egui;
use glam::Vec3;

use crate::graph::history::History;
use crate::graph::scene::{NodeData, Scene};
use crate::sculpt::SculptState;

use super::actions::{Action, SculptConvertMode};
use super::state::SculptConvertDialog;
use super::SdfApp;

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
                    self.doc.sculpt_state = SculptState::Inactive;
                    self.ui.isolation_state = None;
                    self.gpu.current_structure_key = 0;
                    self.gpu.buffer_dirty = true;
                    self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                    self.persistence.scene_dirty = false;
                    self.persistence.current_file_path = None;
                }
                Action::OpenProject => {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = crate::io::open_dialog() {
                        self.load_project_from_path(&path);
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
                        let path = if let Some(ref p) = self.persistence.current_file_path {
                            Some(p.clone())
                        } else {
                            crate::io::save_dialog()
                        };
                        if let Some(path) = path {
                            if let Err(e) = crate::io::save_project(&self.doc.scene, &self.doc.camera, &path) {
                                log::error!("Failed to save project: {}", e);
                            } else {
                                self.persistence.current_file_path = Some(path.clone());
                                self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                                self.persistence.scene_dirty = false;
                                self.settings.add_recent_file(&path.to_string_lossy());
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        crate::io::web_save_project(&self.doc.scene, &self.doc.camera);
                        self.persistence.saved_fingerprint = self.doc.scene.data_fingerprint();
                        self.persistence.scene_dirty = false;
                    }
                }

                // ── Selection ────────────────────────────────────────
                Action::Select(id) => {
                    if let Some(node_id) = id {
                        self.ui.node_graph_state.select_single(node_id);
                    } else {
                        self.ui.node_graph_state.clear_selection();
                    }
                    self.gpu.buffer_dirty = true;
                }
                Action::DeleteSelected => {
                    let locked = self.ui.node_graph_state.selected
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
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.doc.sculpt_state = SculptState::Inactive;
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
                    if let Some((restored_scene, restored_sel)) =
                        self.doc.history.undo(&self.doc.scene, self.ui.node_graph_state.selected)
                    {
                        self.doc.scene = restored_scene;
                        if let Some(id) = restored_sel {
                            self.ui.node_graph_state.select_single(id);
                        } else {
                            self.ui.node_graph_state.clear_selection();
                        }
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.ui.isolation_state = None;
                        self.gpu.buffer_dirty = true;
                    }
                }
                Action::Redo => {
                    if let Some((restored_scene, restored_sel)) =
                        self.doc.history.redo(&self.doc.scene, self.ui.node_graph_state.selected)
                    {
                        self.doc.scene = restored_scene;
                        if let Some(id) = restored_sel {
                            self.ui.node_graph_state.select_single(id);
                        } else {
                            self.ui.node_graph_state.clear_selection();
                        }
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.ui.isolation_state = None;
                        self.gpu.buffer_dirty = true;
                    }
                }
                Action::SculptUndo => {
                    if let SculptState::Active { node_id, .. } = self.doc.sculpt_state {
                        let current_data = self.doc.scene.nodes.get(&node_id)
                            .and_then(|n| match &n.data {
                                NodeData::Sculpt { voxel_grid, .. } => Some(voxel_grid.data.clone()),
                                _ => None,
                            });
                        if let Some(current) = current_data {
                            if let Some(restored) = self.doc.sculpt_history.undo(&current) {
                                if let Some(node) = self.doc.scene.nodes.get_mut(&node_id) {
                                    if let NodeData::Sculpt { ref mut voxel_grid, .. } = node.data {
                                        voxel_grid.data = restored;
                                    }
                                }
                                self.gpu.buffer_dirty = true;
                            }
                        }
                    }
                }
                Action::SculptRedo => {
                    if let SculptState::Active { node_id, .. } = self.doc.sculpt_state {
                        let current_data = self.doc.scene.nodes.get(&node_id)
                            .and_then(|n| match &n.data {
                                NodeData::Sculpt { voxel_grid, .. } => Some(voxel_grid.data.clone()),
                                _ => None,
                            });
                        if let Some(current) = current_data {
                            if let Some(restored) = self.doc.sculpt_history.redo(&current) {
                                if let Some(node) = self.doc.scene.nodes.get_mut(&node_id) {
                                    if let NodeData::Sculpt { ref mut voxel_grid, .. } = node.data {
                                        voxel_grid.data = restored;
                                    }
                                }
                                self.gpu.buffer_dirty = true;
                            }
                        }
                    }
                }

                // ── Camera ───────────────────────────────────────────
                Action::FocusSelected => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        let parent_map = self.doc.scene.build_parent_map();
                        let (center, radius) = self.doc.scene.compute_subtree_sphere(sel, &parent_map);
                        self.doc.camera.focus_on(
                            Vec3::new(center[0], center[1], center[2]),
                            radius.max(0.5),
                        );
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
                Action::CameraTop => self.doc.camera.start_transition(0.0, std::f32::consts::FRAC_PI_2, 0.0),
                Action::CameraRight => self.doc.camera.start_transition(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
                Action::CameraBack => self.doc.camera.start_transition(std::f32::consts::PI, 0.0, 0.0),
                Action::CameraLeft => self.doc.camera.start_transition(-std::f32::consts::FRAC_PI_2, 0.0, 0.0),
                Action::CameraBottom => self.doc.camera.start_transition(0.0, -std::f32::consts::FRAC_PI_2, 0.0),
                Action::ToggleOrtho => self.doc.camera.toggle_ortho(),

                // ── Tools ────────────────────────────────────────────
                Action::SetTool(tool) => {
                    self.doc.active_tool = tool;
                    match tool {
                        crate::sculpt::ActiveTool::Select => {
                            self.doc.sculpt_state = SculptState::Inactive;
                            self.async_state.last_sculpt_hit = None;
                            self.async_state.lazy_brush_pos = None;
                        }
                        crate::sculpt::ActiveTool::Sculpt => {
                            if let Some(sel) = self.ui.node_graph_state.selected {
                                if self.doc.scene.nodes.get(&sel).is_some_and(|n| {
                                    matches!(n.data, NodeData::Sculpt { .. })
                                }) {
                                    let extent = self.scene_avg_extent();
                                    self.doc.sculpt_state = SculptState::new_active_with_radius(sel, extent);
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
                                self.doc.sculpt_state = SculptState::new_active_with_radius(sel, extent);
                                self.ensure_brush_settings_tab();
                            } else {
                                // Case 2: check for sculpt parent
                                let parent_map = self.doc.scene.build_parent_map();
                                if let Some(sculpt_id) = self.doc.scene.find_sculpt_parent(sel, &parent_map) {
                                    let extent = self.scene_avg_extent();
                                    self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
                                    self.doc.sculpt_state = SculptState::new_active_with_radius(sculpt_id, extent);
                                    self.ui.node_graph_state.select_single(sculpt_id);
                                    self.ensure_brush_settings_tab();
                                } else {
                                    // Case 3: non-sculpt node — open convert dialog
                                    self.ui.sculpt_convert_dialog = Some(SculptConvertDialog::new(sel));
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
                Action::CommitSculptConvert { target, mode, resolution } => {
                    self.ui.sculpt_convert_dialog = None;
                    let baking = !matches!(self.async_state.bake_status, super::BakeStatus::Idle);
                    if baking {
                        self.ui.toasts.push(super::Toast {
                            message: "A bake is already in progress".into(),
                            is_error: true,
                            created: crate::compat::Instant::now(),
                            duration: crate::compat::Duration::from_secs(4),
                        });
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
                            SculptConvertMode::BakeActiveNode => {
                                (target, false)
                            }
                        };
                        // Get color from the target node (or default white)
                        let color = self.doc.scene.nodes.get(&subtree_root)
                            .map(|n| match &n.data {
                                NodeData::Primitive { color, .. } => *color,
                                _ => Vec3::new(0.8, 0.8, 0.8),
                            })
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
                                if self.doc.scene.nodes.get(&sculpt_id).is_some_and(|n| {
                                    matches!(n.data, NodeData::Sculpt { .. })
                                }) {
                                    let extent = self.scene_avg_extent();
                                    self.doc.active_tool = crate::sculpt::ActiveTool::Sculpt;
                                    self.doc.sculpt_state = SculptState::new_active_with_radius(sculpt_id, extent);
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
                Action::CreateModifier { kind, input } => {
                    let id = self.doc.scene.create_modifier(kind, input);
                    self.ui.node_graph_state.select_single(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
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
                Action::ReparentNode { dragged, new_parent } => {
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
                    if !baking {
                        if req.flatten {
                            self.start_async_bake(req, ctx);
                        } else {
                            self.apply_instant_displacement_bake(req);
                        }
                    }
                }
                Action::ShowExportDialog => {
                    self.ui.show_export_dialog = true;
                }
                Action::ImportMesh => {
                    let importing = !matches!(self.async_state.import_status, super::ImportStatus::Idle);
                    if !importing {
                        self.start_import(ctx);
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
                Action::CopyProperties => {
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        if let Some(node) = self.doc.scene.nodes.get(&sel) {
                            let clip = match &node.data {
                                crate::graph::scene::NodeData::Primitive {
                                    color, roughness, metallic, emissive, emissive_intensity, fresnel, ..
                                } => Some(super::state::PropertyClipboard {
                                    color: [color.x, color.y, color.z],
                                    roughness: *roughness,
                                    metallic: *metallic,
                                    emissive: [emissive.x, emissive.y, emissive.z],
                                    emissive_intensity: *emissive_intensity,
                                    fresnel: *fresnel,
                                }),
                                crate::graph::scene::NodeData::Sculpt {
                                    color, roughness, metallic, emissive, emissive_intensity, fresnel, ..
                                } => Some(super::state::PropertyClipboard {
                                    color: [color.x, color.y, color.z],
                                    roughness: *roughness,
                                    metallic: *metallic,
                                    emissive: [emissive.x, emissive.y, emissive.z],
                                    emissive_intensity: *emissive_intensity,
                                    fresnel: *fresnel,
                                }),
                                _ => None,
                            };
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
                                let applied = match &mut node.data {
                                    crate::graph::scene::NodeData::Primitive {
                                        color, roughness, metallic, emissive, emissive_intensity, fresnel, ..
                                    } => {
                                        *color = Vec3::new(clip.color[0], clip.color[1], clip.color[2]);
                                        *roughness = clip.roughness;
                                        *metallic = clip.metallic;
                                        *emissive = Vec3::new(clip.emissive[0], clip.emissive[1], clip.emissive[2]);
                                        *emissive_intensity = clip.emissive_intensity;
                                        *fresnel = clip.fresnel;
                                        true
                                    }
                                    crate::graph::scene::NodeData::Sculpt {
                                        color, roughness, metallic, emissive, emissive_intensity, fresnel, ..
                                    } => {
                                        *color = Vec3::new(clip.color[0], clip.color[1], clip.color[2]);
                                        *roughness = clip.roughness;
                                        *metallic = clip.metallic;
                                        *emissive = Vec3::new(clip.emissive[0], clip.emissive[1], clip.emissive[2]);
                                        *emissive_intensity = clip.emissive_intensity;
                                        *fresnel = clip.fresnel;
                                        true
                                    }
                                    _ => false,
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
                            self.doc.camera.target = Vec3::new(bm.target[0], bm.target[1], bm.target[2]);
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
    /// Compute average half-extent of the scene bounding box for adaptive brush sizing.
    fn scene_avg_extent(&self) -> f32 {
        let (min, max) = self.doc.scene.compute_bounds();
        ((max[0] - min[0]) + (max[1] - min[1]) + (max[2] - min[2])) / 6.0
    }

    fn ensure_brush_settings_tab(&mut self) {
        use crate::ui::dock::Tab;
        if self.ui.dock_state.find_tab(&Tab::BrushSettings).is_none() {
            // Add BrushSettings as a sibling tab next to Properties (not into the viewport).
            if let Some((surface_idx, node_idx, _tab_idx)) = self.ui.dock_state.find_tab(&Tab::Properties) {
                self.ui.dock_state[surface_idx][node_idx]
                    .append_tab(Tab::BrushSettings);
            } else {
                // Fallback: no Properties tab found, add next to SceneTree
                if let Some((surface_idx, node_idx, _tab_idx)) = self.ui.dock_state.find_tab(&Tab::SceneTree) {
                    self.ui.dock_state[surface_idx][node_idx]
                        .append_tab(Tab::BrushSettings);
                } else {
                    self.ui.dock_state.push_to_focused_leaf(Tab::BrushSettings);
                }
            }
        }
    }

    /// Load a project file and update all relevant state.
    fn load_project_from_path(&mut self, path: &std::path::Path) -> bool {
        match crate::io::load_project(&path.to_path_buf()) {
            Ok(project) => {
                self.doc.scene = project.scene;
                self.doc.camera = project.camera;
                self.doc.history = History::new();
                self.ui.node_graph_state.clear_selection();
                self.ui.node_graph_state.needs_initial_rebuild = true;
                self.doc.sculpt_state = SculptState::Inactive;
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
                    NodeData::Primitive { ref mut position, .. }
                    | NodeData::Sculpt { ref mut position, .. } => {
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
