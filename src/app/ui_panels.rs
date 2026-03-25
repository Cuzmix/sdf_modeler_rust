use std::sync::atomic::Ordering;

use crate::sculpt::{BrushMode, SculptState};

use super::actions::{Action, ActionSink};
use super::{BakeStatus, ExportStatus, ImportStatus, SdfApp};

impl SdfApp {
    /// Draw the menu bar and push any triggered actions into the action sink.
    /// The menu bar reads state to show enabled/disabled items, and emits
    /// actions for anything the user clicks Ã¢â‚¬â€ no direct state mutation.
    pub(super) fn show_menu_bar(&mut self, ctx: &egui::Context, actions: &mut ActionSink) {
        let mut action_open_recent: Option<String> = None;

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                // --- File ---
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene").clicked() {
                        actions.push(Action::NewScene);
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add(egui::Button::new("Open...").shortcut_text("Ctrl+O"))
                        .clicked()
                    {
                        actions.push(Action::OpenProject);
                        ui.close();
                    }
                    let save_label = if self.persistence.current_file_path.is_some() {
                        "Save"
                    } else {
                        "Save As..."
                    };
                    if ui
                        .add(egui::Button::new(save_label).shortcut_text("Ctrl+S"))
                        .clicked()
                    {
                        actions.push(Action::SaveProject);
                        ui.close();
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
                                    ui.close();
                                }
                            }
                        });
                        ui.separator();
                    }

                    if ui
                        .add(egui::Button::new("Screenshot...").shortcut_text("Ctrl+P"))
                        .clicked()
                    {
                        actions.push(Action::TakeScreenshot);
                        ui.close();
                    }
                    let import_idle = matches!(self.async_state.import_status, ImportStatus::Idle);
                    if ui
                        .add_enabled(import_idle, egui::Button::new("Import Mesh..."))
                        .clicked()
                    {
                        actions.push(Action::ImportMesh);
                        ui.close();
                    }
                    let export_idle = matches!(self.async_state.export_status, ExportStatus::Idle);
                    if ui
                        .add_enabled(
                            export_idle,
                            egui::Button::new("Export Mesh...").shortcut_text("Ctrl+E"),
                        )
                        .clicked()
                    {
                        actions.push(Action::ShowExportDialog);
                        ui.close();
                    }
                });

                // --- Edit ---
                ui.menu_button("Edit", |ui| {
                    if ui
                        .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                        .clicked()
                    {
                        actions.push(Action::Undo);
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::new("Redo").shortcut_text("Ctrl+Y"))
                        .clicked()
                    {
                        actions.push(Action::Redo);
                        ui.close();
                    }
                    ui.separator();
                    let has_sel = self.ui.selection.selected.is_some();
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy").shortcut_text("Ctrl+C"))
                        .clicked()
                    {
                        actions.push(Action::Copy);
                        ui.close();
                    }
                    let has_clip = self
                        .doc
                        .clipboard_node
                        .is_some_and(|id| self.doc.scene.nodes.contains_key(&id));
                    if ui
                        .add_enabled(has_clip, egui::Button::new("Paste").shortcut_text("Ctrl+V"))
                        .clicked()
                    {
                        actions.push(Action::Paste);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            has_sel,
                            egui::Button::new("Duplicate").shortcut_text("Ctrl+D"),
                        )
                        .clicked()
                    {
                        actions.push(Action::Duplicate);
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Delete").shortcut_text("Del"))
                        .clicked()
                    {
                        actions.push(Action::DeleteSelected);
                        ui.close();
                    }
                });

                // --- View ---
                ui.menu_button("View", |ui| {
                    let has_sel = self.ui.selection.selected.is_some();
                    if ui
                        .add_enabled(
                            has_sel,
                            egui::Button::new("Focus Selected").shortcut_text("F"),
                        )
                        .clicked()
                    {
                        actions.push(Action::FocusSelected);
                        ui.close();
                    }
                    ui.separator();
                    let profiler_label = if self.ui.show_debug {
                        "Hide Profiler"
                    } else {
                        "Show Profiler"
                    };
                    if ui
                        .add(egui::Button::new(profiler_label).shortcut_text("F4"))
                        .clicked()
                    {
                        actions.push(Action::ToggleDebug);
                        ui.close();
                    }
                    ui.separator();
                    ui.label("Camera Presets");
                    if ui
                        .add(egui::Button::new("Front").shortcut_text("F5"))
                        .clicked()
                    {
                        actions.push(Action::CameraFront);
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::new("Top").shortcut_text("F6"))
                        .clicked()
                    {
                        actions.push(Action::CameraTop);
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::new("Right").shortcut_text("F7"))
                        .clicked()
                    {
                        actions.push(Action::CameraRight);
                        ui.close();
                    }
                    ui.separator();
                    ui.menu_button("Bookmarks", |ui| {
                        for i in 0..9 {
                            let label = if i < self.settings.bookmarks.len() {
                                if self.settings.bookmarks[i].is_some() {
                                    format!("Slot {} (saved)", i + 1)
                                } else {
                                    format!("Slot {} (empty)", i + 1)
                                }
                            } else {
                                format!("Slot {} (empty)", i + 1)
                            };
                            let has_bookmark = i < self.settings.bookmarks.len()
                                && self.settings.bookmarks[i].is_some();
                            if ui
                                .add_enabled(has_bookmark, egui::Button::new(&label))
                                .clicked()
                            {
                                actions.push(Action::RestoreBookmark(i));
                                ui.close();
                            }
                        }
                        ui.separator();
                        ui.weak("Ctrl+1-9 to save current view");
                    });
                    ui.separator();
                    ui.menu_button("Panels", |ui| {
                        for panel in crate::app::state::ExpertPanelKind::ALL {
                            let is_open = self.ui.expert_panels.is_open(panel);
                            if ui.selectable_label(is_open, panel.label()).clicked() {
                                actions.push(Action::ToggleExpertPanel(panel));
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui.button("Reset Layout").clicked() {
                            actions.push(Action::ResetPrimaryShellLayout);
                            ui.close();
                        }
                    });
                    ui.menu_button("Workspace", |ui| {
                        use crate::app::actions::WorkspacePreset;
                        if ui
                            .button("Modeling")
                            .on_hover_text("Balanced layout with all panels")
                            .clicked()
                        {
                            actions.push(Action::SetWorkspace(WorkspacePreset::Modeling));
                            ui.close();
                        }
                        if ui
                            .button("Sculpting")
                            .on_hover_text("Large viewport with properties sidebar")
                            .clicked()
                        {
                            actions.push(Action::SetWorkspace(WorkspacePreset::Sculpting));
                            ui.close();
                        }
                        if ui
                            .button("Rendering")
                            .on_hover_text("Viewport with render settings sidebar")
                            .clicked()
                        {
                            actions.push(Action::SetWorkspace(WorkspacePreset::Rendering));
                            ui.close();
                        }
                    });
                });

                // --- Settings ---
                if ui.button("Settings").clicked() {
                    actions.push(Action::ToggleSettings);
                }

                // --- Help ---
                ui.menu_button("Help", |ui| {
                    if ui
                        .add(egui::Button::new("Keyboard Shortcuts").shortcut_text("F1"))
                        .clicked()
                    {
                        actions.push(Action::ToggleHelp);
                        ui.close();
                    }
                });

                // Progress indicators (right-aligned)
                // Export and Import progress are shown in dedicated modals (export_progress.rs)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let BakeStatus::InProgress {
                        ref progress,
                        total,
                        ..
                    } = self.async_state.bake_status
                    {
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

        // Recent file action must be pushed after the panel closure
        if let Some(recent_path) = action_open_recent {
            actions.push(Action::OpenRecentProject(recent_path));
        }
    }

    pub(super) fn show_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Mode indicator
                    match &self.doc.sculpt_state {
                        SculptState::Active { session, .. } => {
                            let mode_name = match session.selected_brush {
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
                            if let Some(axis) = session.symmetry_axis {
                                let axis_name = match axis {
                                    0 => "X",
                                    1 => "Y",
                                    _ => "Z",
                                };
                                ui.separator();
                                ui.label(format!("Sym: {}", axis_name));
                            }
                        }
                        SculptState::Inactive { .. } => {
                            ui.colored_label(
                                egui::Color32::from_rgb(130, 200, 255),
                                self.gizmo.mode.label(),
                            );
                            ui.separator();
                            ui.weak(self.gizmo.space.label());
                        }
                    }

                    ui.separator();

                    // Selection info
                    if let Some(sel) = self.ui.selection.selected {
                        if let Some(node) = self.doc.scene.nodes.get(&sel) {
                            ui.weak(&node.name);
                        }
                    }

                    // Right-aligned control hints
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.doc.sculpt_state.is_active() {
                            ui.weak(
                                "LMB: Sculpt | Shift+LMB: Smooth | F: Radius | Shift+F: Strength",
                            );
                        } else {
                            ui.weak("LMB: Orbit | RMB: Pan | Scroll: Zoom | Del: Delete");
                        }
                    });
                });
            });
    }
}
