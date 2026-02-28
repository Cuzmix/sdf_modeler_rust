use std::sync::atomic::Ordering;

use eframe::egui;
use glam::Vec3;

use crate::graph::history::History;
use crate::graph::scene::Scene;
use crate::sculpt::{BrushMode, SculptState};
use crate::ui::viewport::ViewportResources;

use super::{BakeStatus, ExportStatus, SdfApp, TIMING_HISTORY_LEN};

impl SdfApp {
    pub(super) fn show_menu_bar(&mut self, ctx: &egui::Context) {
        let mut action_new = false;
        let mut action_open = false;
        let mut action_save = false;
        let mut action_screenshot = false;
        let mut action_export = false;
        let mut action_undo = false;
        let mut action_redo = false;
        let mut action_delete = false;
        let mut action_focus = false;
        let mut action_copy = false;
        let mut action_paste = false;
        let mut action_duplicate = false;
        let mut action_open_recent: Option<String> = None;

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // --- File ---
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene").clicked() {
                        action_new = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                        action_open = true;
                        ui.close_menu();
                    }
                    let save_label = if self.current_file_path.is_some() { "Save" } else { "Save As..." };
                    if ui.add(egui::Button::new(save_label).shortcut_text("Ctrl+S")).clicked() {
                        action_save = true;
                        ui.close_menu();
                    }
                    ui.separator();

                    // Recent files
                    if !self.settings.recent_files.is_empty() {
                        ui.menu_button("Recent Files", |ui| {
                            let files = self.settings.recent_files.clone();
                            for file_path in &files {
                                let name = std::path::Path::new(file_path)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| file_path.clone());
                                if ui.button(&name).on_hover_text(file_path).clicked() {
                                    action_open_recent = Some(file_path.clone());
                                    ui.close_menu();
                                }
                            }
                        });
                        ui.separator();
                    }

                    if ui.add(egui::Button::new("Screenshot...").shortcut_text("Ctrl+P")).clicked() {
                        action_screenshot = true;
                        ui.close_menu();
                    }
                    let export_idle = matches!(self.export_status, ExportStatus::Idle);
                    if ui.add_enabled(export_idle, egui::Button::new("Export Mesh...").shortcut_text("Ctrl+E")).clicked() {
                        action_export = true;
                        ui.close_menu();
                    }
                    ui.separator();

                    // Auto-save settings
                    ui.horizontal(|ui| {
                        let mut auto_save = self.settings.auto_save_enabled;
                        if ui.checkbox(&mut auto_save, "Auto-save").changed() {
                            self.settings.auto_save_enabled = auto_save;
                            self.settings.save();
                        }
                    });
                    if self.settings.auto_save_enabled {
                        ui.horizontal(|ui| {
                            ui.label("Interval:");
                            let mut secs = self.settings.auto_save_interval_secs;
                            if ui.add(egui::DragValue::new(&mut secs).range(30..=600).suffix("s").speed(5)).changed() {
                                self.settings.auto_save_interval_secs = secs;
                                self.settings.save();
                            }
                        });
                    }
                });

                // --- Edit ---
                ui.menu_button("Edit", |ui| {
                    if ui.add(egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                        action_undo = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                        action_redo = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    let has_sel = self.node_graph_state.selected.is_some();
                    if ui.add_enabled(has_sel, egui::Button::new("Copy").shortcut_text("Ctrl+C")).clicked() {
                        action_copy = true;
                        ui.close_menu();
                    }
                    let has_clip = self.clipboard_node.map_or(false, |id| self.scene.nodes.contains_key(&id));
                    if ui.add_enabled(has_clip, egui::Button::new("Paste").shortcut_text("Ctrl+V")).clicked() {
                        action_paste = true;
                        ui.close_menu();
                    }
                    if ui.add_enabled(has_sel, egui::Button::new("Duplicate").shortcut_text("Ctrl+D")).clicked() {
                        action_duplicate = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add_enabled(has_sel, egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                        action_delete = true;
                        ui.close_menu();
                    }
                });

                // --- View ---
                ui.menu_button("View", |ui| {
                    let has_sel = self.node_graph_state.selected.is_some();
                    if ui.add_enabled(has_sel, egui::Button::new("Focus Selected").shortcut_text("F")).clicked() {
                        action_focus = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    let profiler_label = if self.show_debug { "Hide Profiler" } else { "Show Profiler" };
                    if ui.add(egui::Button::new(profiler_label).shortcut_text("F4")).clicked() {
                        self.show_debug = !self.show_debug;
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label("Camera Presets");
                    if ui.add(egui::Button::new("Front").shortcut_text("F5")).clicked() {
                        self.camera.set_front();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Top").shortcut_text("F6")).clicked() {
                        self.camera.set_top();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Right").shortcut_text("F7")).clicked() {
                        self.camera.set_right();
                        ui.close_menu();
                    }
                });

                // --- Help ---
                ui.menu_button("Help", |ui| {
                    if ui.add(egui::Button::new("Keyboard Shortcuts").shortcut_text("F1")).clicked() {
                        self.show_help = !self.show_help;
                        ui.close_menu();
                    }
                });


                // Progress indicators (right-aligned)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let ExportStatus::InProgress { ref progress, total, .. } = self.export_status {
                        let done = progress.load(Ordering::Relaxed);
                        let frac = done as f32 / total.max(1) as f32;
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("Exporting... {:.0}%", frac * 100.0))
                                .desired_width(200.0),
                        );
                    }
                    if let BakeStatus::InProgress { ref progress, total, .. } = self.bake_status {
                        let done = progress.load(Ordering::Relaxed);
                        let frac = done as f32 / total.max(1) as f32;
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("Baking... {:.0}%", frac * 100.0))
                                .desired_width(200.0),
                        );
                    }
                });
            });
        });

        // Process deferred menu actions
        if action_new {
            self.scene = Scene::new();
            self.history = History::new();
            self.node_graph_state.selected = None;
            self.node_graph_state.needs_initial_rebuild = true;
            self.sculpt_state = SculptState::Inactive;
            self.current_structure_key = 0;
            self.buffer_dirty = true;
            self.saved_fingerprint = self.scene.data_fingerprint();
            self.scene_dirty = false;
            self.current_file_path = None;
        }
        // Helper closure for loading a project from a path
        let load_from_path = |app: &mut Self, path: &std::path::Path| {
            match crate::io::load_project(&path.to_path_buf()) {
                Ok(project) => {
                    app.scene = project.scene;
                    app.camera = project.camera;
                    app.history = History::new();
                    app.node_graph_state.selected = None;
                    app.node_graph_state.needs_initial_rebuild = true;
                    app.sculpt_state = SculptState::Inactive;
                    app.current_structure_key = 0;
                    app.buffer_dirty = true;
                    app.saved_fingerprint = app.scene.data_fingerprint();
                    app.scene_dirty = false;
                    app.current_file_path = Some(path.to_path_buf());
                    app.settings.add_recent_file(&path.to_string_lossy());
                    true
                }
                Err(e) => {
                    log::error!("Failed to load project: {}", e);
                    false
                }
            }
        };
        if action_open {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(path) = crate::io::open_dialog() {
                load_from_path(self, &path);
            }
        }
        if let Some(ref recent_path) = action_open_recent {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = std::path::PathBuf::from(recent_path);
                if !load_from_path(self, &path) {
                    // Remove stale entry
                    self.settings.recent_files.retain(|p| p != recent_path);
                    self.settings.save();
                }
            }
        }
        if action_save {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = if let Some(ref p) = self.current_file_path {
                    Some(p.clone())
                } else {
                    crate::io::save_dialog()
                };
                if let Some(path) = path {
                    if let Err(e) = crate::io::save_project(&self.scene, &self.camera, &path) {
                        log::error!("Failed to save project: {}", e);
                    } else {
                        self.current_file_path = Some(path.clone());
                        self.saved_fingerprint = self.scene.data_fingerprint();
                        self.scene_dirty = false;
                        self.settings.add_recent_file(&path.to_string_lossy());
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                crate::io::web_save_project(&self.scene, &self.camera);
                self.saved_fingerprint = self.scene.data_fingerprint();
                self.scene_dirty = false;
            }
        }
        if action_screenshot {
            self.take_screenshot();
        }
        if action_export {
            self.start_export(ctx);
        }
        if action_undo {
            if let Some((restored_scene, restored_sel)) =
                self.history.undo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.needs_initial_rebuild = true;
                self.buffer_dirty = true;
            }
        }
        if action_redo {
            if let Some((restored_scene, restored_sel)) =
                self.history.redo(&self.scene, self.node_graph_state.selected)
            {
                self.scene = restored_scene;
                self.node_graph_state.selected = restored_sel;
                self.node_graph_state.needs_initial_rebuild = true;
                self.buffer_dirty = true;
            }
        }
        if action_delete {
            self.delete_selected();
        }
        if action_focus {
            if let Some(sel) = self.node_graph_state.selected {
                let parent_map = self.scene.build_parent_map();
                let (center, radius) = self.scene.compute_subtree_sphere(sel, &parent_map);
                self.camera.focus_on(
                    Vec3::new(center[0], center[1], center[2]),
                    radius.max(0.5),
                );
            }
        }
        if action_copy {
            self.clipboard_node = self.node_graph_state.selected;
        }
        if action_paste {
            if let Some(src) = self.clipboard_node {
                if self.scene.nodes.contains_key(&src) {
                    if let Some(new_id) = self.scene.duplicate_subtree(src) {
                        if let Some(node) = self.scene.nodes.get_mut(&new_id) {
                            match &mut node.data {
                                crate::graph::scene::NodeData::Primitive { ref mut position, .. }
                                | crate::graph::scene::NodeData::Sculpt { ref mut position, .. } => {
                                    position.x += 1.0;
                                }
                                _ => {}
                            }
                        }
                        self.node_graph_state.selected = Some(new_id);
                        self.node_graph_state.needs_initial_rebuild = true;
                        self.buffer_dirty = true;
                    }
                }
            }
        }
        if action_duplicate {
            if let Some(sel) = self.node_graph_state.selected {
                if let Some(new_id) = self.scene.duplicate_subtree(sel) {
                    if let Some(node) = self.scene.nodes.get_mut(&new_id) {
                        match &mut node.data {
                            crate::graph::scene::NodeData::Primitive { ref mut position, .. }
                            | crate::graph::scene::NodeData::Sculpt { ref mut position, .. } => {
                                position.x += 1.0;
                            }
                            _ => {}
                        }
                    }
                    self.node_graph_state.selected = Some(new_id);
                    self.node_graph_state.needs_initial_rebuild = true;
                    self.buffer_dirty = true;
                }
            }
        }
    }

    pub(super) fn show_debug_window(&self, ctx: &egui::Context) {
        if !self.show_debug {
            return;
        }
        let t = &self.timings;
        egui::Window::new("Profiler")
            .default_pos([10.0, 10.0])
            .default_size([280.0, 320.0])
            .show(ctx, |ui| {
                // --- FPS / Frame time ---
                let color = if t.avg_fps >= 55.0 {
                    egui::Color32::from_rgb(100, 255, 100)
                } else if t.avg_fps >= 30.0 {
                    egui::Color32::from_rgb(255, 255, 100)
                } else {
                    egui::Color32::from_rgb(255, 100, 100)
                };
                ui.colored_label(color, format!(
                    "FPS: {:.0}  ({:.2} ms)", t.avg_fps, t.avg_frame_ms
                ));

                // --- Frame time sparkline ---
                let history_ordered: Vec<f32> = {
                    let idx = t.history_idx;
                    let mut v = Vec::with_capacity(TIMING_HISTORY_LEN);
                    v.extend_from_slice(&t.history[idx..]);
                    v.extend_from_slice(&t.history[..idx]);
                    v
                };
                let max_ms = history_ordered.iter().cloned().fold(1.0_f32, f32::max);
                let target_ms = 16.67_f32; // 60 FPS target line

                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 50.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));

                // Draw 60fps target line
                let target_y = rect.bottom() - (target_ms / max_ms) * rect.height();
                if target_y > rect.top() {
                    painter.line_segment(
                        [egui::pos2(rect.left(), target_y), egui::pos2(rect.right(), target_y)],
                        egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(100, 100, 255, 80)),
                    );
                }

                // Draw bars
                let bar_w = rect.width() / TIMING_HISTORY_LEN as f32;
                for (i, &ms) in history_ordered.iter().enumerate() {
                    let h = (ms / max_ms) * rect.height();
                    let x = rect.left() + i as f32 * bar_w;
                    let bar_color = if ms <= 16.67 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else if ms <= 33.33 {
                        egui::Color32::from_rgb(200, 200, 80)
                    } else {
                        egui::Color32::from_rgb(200, 80, 80)
                    };
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, rect.bottom() - h),
                            egui::vec2(bar_w.max(1.0), h),
                        ),
                        0.0,
                        bar_color,
                    );
                }

                ui.add_space(2.0);

                // --- CPU phase breakdown ---
                egui::CollapsingHeader::new("CPU Phases")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.monospace(format!("Pipeline sync:  {:6.2} ms", t.pipeline_sync_s * 1000.0));
                        ui.monospace(format!("Buffer upload:  {:6.2} ms", t.buffer_upload_s * 1000.0));
                        ui.monospace(format!("Comp dispatch:  {:6.2} ms", t.composite_dispatch_s * 1000.0));
                        ui.monospace(format!("UI draw:        {:6.2} ms", t.ui_draw_s * 1000.0));
                        ui.monospace(format!("Total CPU:      {:6.2} ms", t.total_cpu_s * 1000.0));
                    });

                ui.separator();

                // --- Scene stats ---
                egui::CollapsingHeader::new("Scene")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label(format!("Nodes: {}", self.scene.nodes.len()));
                        ui.label(format!("Top-level: {}", self.scene.top_level_nodes().len()));
                        ui.label(format!("Sculpt textures: {}", self.sculpt_tex_indices.len()));
                        ui.label(format!(
                            "Composite: {}",
                            if self.settings.render.composite_volume_enabled { "ON" } else { "OFF" }
                        ));
                    });

                // --- Render state ---
                egui::CollapsingHeader::new("Render State")
                    .default_open(true)
                    .show(ui, |ui| {
                        let renderer = self.render_state.renderer.read();
                        if let Some(res) = renderer.callback_resources.get::<ViewportResources>() {
                            ui.label(format!(
                                "Render size: {}x{}", res.render_width, res.render_height
                            ));
                            ui.label(format!("Composite active: {}", res.use_composite));
                        }
                    });

                // --- Camera ---
                egui::CollapsingHeader::new("Camera")
                    .default_open(false)
                    .show(ui, |ui| {
                        let eye = self.camera.eye();
                        ui.label(format!("Eye: ({:.2}, {:.2}, {:.2})", eye.x, eye.y, eye.z));
                        ui.label(format!("Distance: {:.2}", self.camera.distance));
                        ui.label(format!(
                            "Yaw: {:.1} Pitch: {:.1}",
                            self.camera.yaw.to_degrees(),
                            self.camera.pitch.to_degrees(),
                        ));
                    });
            });
    }

    pub(super) fn show_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help { return; }
        egui::Window::new("Keyboard Shortcuts")
            .open(&mut self.show_help)
            .resizable(false)
            .default_width(380.0)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let section = |ui: &mut egui::Ui, title: &str| {
                            ui.colored_label(egui::Color32::from_rgb(180, 200, 255), title);
                            ui.end_row();
                        };
                        let row = |ui: &mut egui::Ui, key: &str, desc: &str| {
                            ui.monospace(key);
                            ui.label(desc);
                            ui.end_row();
                        };

                        section(ui, "General");
                        row(ui, "Ctrl+O", "Open project");
                        row(ui, "Ctrl+S", "Save project");
                        row(ui, "Ctrl+Z", "Undo");
                        row(ui, "Ctrl+Y", "Redo");
                        row(ui, "Ctrl+C", "Copy node");
                        row(ui, "Ctrl+V", "Paste node");
                        row(ui, "Ctrl+D", "Duplicate node");
                        row(ui, "Delete", "Delete selected node");
                        row(ui, "Ctrl+P", "Screenshot");
                        row(ui, "Ctrl+E", "Export OBJ");
                        row(ui, "F1", "Toggle this help");
                        row(ui, "F4", "Toggle profiler");

                        ui.separator(); ui.end_row();
                        section(ui, "Camera");
                        row(ui, "LMB drag", "Orbit");
                        row(ui, "RMB drag", "Pan");
                        row(ui, "Scroll", "Zoom");
                        row(ui, "F", "Focus selected");
                        row(ui, "F5", "Front view");
                        row(ui, "F6", "Top view");
                        row(ui, "F7", "Right view");

                        ui.separator(); ui.end_row();
                        section(ui, "Gizmo");
                        row(ui, "W", "Move tool");
                        row(ui, "E", "Rotate tool");
                        row(ui, "R", "Scale tool");
                        row(ui, "G", "Toggle Local / World");
                        row(ui, "Alt+Drag", "Move pivot");
                        row(ui, "Alt+C", "Reset pivot");

                        ui.separator(); ui.end_row();
                        section(ui, "Sculpt Mode");
                        row(ui, "LMB drag", "Paint brush");
                        row(ui, "RMB drag", "Pan camera");
                        row(ui, "MMB drag", "Orbit camera");
                        row(ui, "1", "Add brush");
                        row(ui, "2", "Carve brush");
                        row(ui, "3", "Smooth brush");
                        row(ui, "4", "Flatten brush");
                        row(ui, "5", "Inflate brush");
                        row(ui, "6", "Grab brush");
                        row(ui, "X / Y / Z", "Toggle symmetry axis");

                        ui.separator(); ui.end_row();
                        section(ui, "Scene Tree");
                        row(ui, "Double-click", "Rename node");
                        row(ui, "Right-click", "Context menu");
                    });
            });
    }

    pub(super) fn show_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Mode indicator
                    match &self.sculpt_state {
                        SculptState::Active { brush_mode, symmetry_axis, .. } => {
                            let mode_name = match brush_mode {
                                BrushMode::Add => "Add",
                                BrushMode::Carve => "Carve",
                                BrushMode::Smooth => "Smooth",
                                BrushMode::Flatten => "Flatten",
                                BrushMode::Inflate => "Inflate",
                                BrushMode::Grab => "Grab",
                            };
                            ui.colored_label(
                                egui::Color32::from_rgb(180, 130, 255),
                                format!("Sculpt: {}", mode_name),
                            );
                            if let Some(axis) = symmetry_axis {
                                let axis_name = match axis {
                                    0 => "X",
                                    1 => "Y",
                                    _ => "Z",
                                };
                                ui.separator();
                                ui.label(format!("Sym: {}", axis_name));
                            }
                        }
                        SculptState::Inactive => {
                            ui.colored_label(
                                egui::Color32::from_rgb(130, 200, 255),
                                self.gizmo_mode.label(),
                            );
                            ui.separator();
                            ui.weak(self.gizmo_space.label());
                        }
                    }

                    ui.separator();

                    // Selection info
                    if let Some(sel) = self.node_graph_state.selected {
                        if let Some(node) = self.scene.nodes.get(&sel) {
                            ui.weak(&node.name);
                        }
                    }

                    // Right-aligned control hints
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.sculpt_state.is_active() {
                            ui.weak("LMB: Paint | RMB: Pan | MMB: Orbit | 1-6: Brush | X/Y/Z: Sym");
                        } else {
                            ui.weak("LMB: Orbit | RMB: Pan | Scroll: Zoom | Del: Delete");
                        }
                    });
                });
            });
    }

    pub(super) fn show_toasts(&mut self, ctx: &egui::Context) {
        let now = crate::compat::Instant::now();
        self.toasts.retain(|t| now.duration_since(t.created) < t.duration);

        if self.toasts.is_empty() {
            return;
        }

        egui::Area::new(egui::Id::new("toasts"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -36.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                for toast in &self.toasts {
                    let elapsed = now.duration_since(toast.created).as_secs_f32();
                    let remaining = toast.duration.as_secs_f32() - elapsed;
                    // Fade out over last 0.5s
                    let alpha = (remaining / 0.5).clamp(0.0, 1.0);

                    let (bg, text_color) = if toast.is_error {
                        (
                            egui::Color32::from_rgba_unmultiplied(120, 30, 30, (220.0 * alpha) as u8),
                            egui::Color32::from_rgba_unmultiplied(255, 180, 180, (255.0 * alpha) as u8),
                        )
                    } else {
                        (
                            egui::Color32::from_rgba_unmultiplied(30, 90, 50, (220.0 * alpha) as u8),
                            egui::Color32::from_rgba_unmultiplied(180, 255, 200, (255.0 * alpha) as u8),
                        )
                    };

                    egui::Frame::none()
                        .fill(bg)
                        .rounding(6.0)
                        .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&toast.message).color(text_color));
                        });
                    ui.add_space(4.0);
                }
            });

        // Keep repainting while toasts are visible
        ctx.request_repaint();
    }
}
