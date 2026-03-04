use eframe::egui;
use glam::Vec3;

use crate::graph::history::History;
use crate::graph::scene::{NodeData, Scene};
use crate::sculpt::SculptState;

use super::actions::Action;
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
                    self.ui.node_graph_state.selected = None;
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
                    self.ui.node_graph_state.selected = id;
                    self.gpu.buffer_dirty = true;
                }
                Action::DeleteSelected => {
                    let locked = self.ui.node_graph_state.selected
                        .and_then(|id| self.doc.scene.nodes.get(&id))
                        .map_or(false, |n| n.locked);
                    if !locked {
                        self.delete_selected();
                    }
                }
                Action::DeleteNode(id) => {
                    let locked = self.doc.scene.nodes.get(&id).map_or(false, |n| n.locked);
                    if !locked {
                        self.doc.scene.remove_node(id);
                        if self.ui.node_graph_state.selected == Some(id) {
                            self.ui.node_graph_state.selected = None;
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
                        self.ui.node_graph_state.selected = restored_sel;
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
                        self.ui.node_graph_state.selected = restored_sel;
                        self.ui.node_graph_state.needs_initial_rebuild = true;
                        self.ui.isolation_state = None;
                        self.gpu.buffer_dirty = true;
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
                                if self.doc.scene.nodes.get(&sel).map_or(false, |n| {
                                    matches!(n.data, NodeData::Sculpt { .. })
                                }) {
                                    self.doc.sculpt_state = SculptState::new_active(sel);
                                }
                            }
                        }
                    }
                }
                Action::SetGizmoMode(mode) => {
                    self.gizmo.mode = mode;
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

                // ── Scene mutations (structural) ─────────────────────
                Action::CreatePrimitive(prim) => {
                    let id = self.doc.scene.create_primitive(prim);
                    self.ui.node_graph_state.selected = Some(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateOperation { op, left, right } => {
                    let id = self.doc.scene.create_operation(op.clone(), left, right);
                    self.ui.node_graph_state.selected = Some(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateTransform { input } => {
                    let id = self.doc.scene.create_transform(input);
                    self.ui.node_graph_state.selected = Some(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::CreateModifier { kind, input } => {
                    let id = self.doc.scene.create_modifier(kind, input);
                    self.ui.node_graph_state.selected = Some(id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(id);
                    self.gpu.buffer_dirty = true;
                }
                Action::InsertModifierAbove { target, kind } => {
                    let new_id = self.doc.scene.insert_modifier_above(target, kind);
                    self.ui.node_graph_state.selected = Some(new_id);
                    self.ui.node_graph_state.needs_initial_rebuild = true;
                    self.ui.node_graph_state.pending_center_node = Some(new_id);
                    self.gpu.buffer_dirty = true;
                }
                Action::InsertTransformAbove { target } => {
                    let new_id = self.doc.scene.insert_transform_above(target);
                    self.ui.node_graph_state.selected = Some(new_id);
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

    /// Load a project file and update all relevant state.
    fn load_project_from_path(&mut self, path: &std::path::Path) -> bool {
        match crate::io::load_project(&path.to_path_buf()) {
            Ok(project) => {
                self.doc.scene = project.scene;
                self.doc.camera = project.camera;
                self.doc.history = History::new();
                self.ui.node_graph_state.selected = None;
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
            self.ui.node_graph_state.selected = Some(new_id);
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.gpu.buffer_dirty = true;
        }
    }
}
