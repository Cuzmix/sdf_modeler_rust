use std::sync::atomic::Ordering;

use eframe::egui::{self, RichText};

use crate::app::actions::{Action, ActionSink, WorkspacePreset};
use crate::sculpt::{ActiveTool, BrushMode, SculptState};
use crate::ui::chrome::{self, BadgeTone};

use super::{BakeStatus, ExportStatus, ImportStatus, SdfApp};

impl SdfApp {
    /// Draw the desktop app header and push any triggered actions into the action sink.
    pub(super) fn show_menu_bar(&mut self, ctx: &egui::Context, actions: &mut ActionSink) {
        let mut action_open_recent: Option<String> = None;
        let current_file_label = self
            .persistence
            .current_file_path
            .as_deref()
            .and_then(|path| std::path::Path::new(path).file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("Scratch Scene")
            .to_string();
        let selected_name = self
            .ui
            .node_graph_state
            .selected
            .and_then(|id| self.doc.scene.nodes.get(&id).map(|node| node.name.clone()));
        let tool_is_sculpt = matches!(self.doc.active_tool, ActiveTool::Sculpt);

        egui::TopBottomPanel::top("app_header")
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                chrome::app_header_frame(ui).show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(10.0, 10.0);

                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.label(RichText::new("SDF Modeler").size(20.0).strong());
                            ui.label(
                                RichText::new(
                                    "Dark-first desktop workspace for signed-distance modeling",
                                )
                                .small()
                                .color(chrome::tokens(ui).muted_text),
                            );
                        });

                        ui.add_space(12.0);

                        chrome::header_group(ui, |ui| {
                            chrome::badge(ui, BadgeTone::Accent, "Workspace");
                            ui.menu_button("Presets", |ui| {
                                if ui
                                    .button("Modeling")
                                    .on_hover_text("Balanced layout with all panels")
                                    .clicked()
                                {
                                    actions.push(Action::SetWorkspace(WorkspacePreset::Modeling));
                                    ui.close_menu();
                                }
                                if ui
                                    .button("Sculpting")
                                    .on_hover_text("Large viewport with sculpt controls")
                                    .clicked()
                                {
                                    actions.push(Action::SetWorkspace(WorkspacePreset::Sculpting));
                                    ui.close_menu();
                                }
                                if ui
                                    .button("Rendering")
                                    .on_hover_text("Viewport-first lighting and render setup")
                                    .clicked()
                                {
                                    actions.push(Action::SetWorkspace(WorkspacePreset::Rendering));
                                    ui.close_menu();
                                }
                            });
                        });

                        chrome::badge(
                            ui,
                            if self.persistence.scene_dirty {
                                BadgeTone::Warning
                            } else {
                                BadgeTone::Muted
                            },
                            current_file_label,
                        );
                        if self.persistence.scene_dirty {
                            chrome::badge(ui, BadgeTone::Warning, "Unsaved");
                        }
                        if let Some(name) = selected_name.as_deref() {
                            chrome::badge(ui, BadgeTone::Muted, format!("Selected: {name}"));
                        }

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
                                        .text(format!("Bake {:.0}%", frac * 100.0))
                                        .desired_width(160.0),
                                );
                            }

                            if ui
                                .button(if self.ui.show_debug {
                                    "Profiler On"
                                } else {
                                    "Profiler"
                                })
                                .clicked()
                            {
                                actions.push(Action::ToggleDebug);
                            }
                            if ui.button("Settings").clicked() {
                                actions.push(Action::ToggleSettings);
                            }
                            if ui.button("Help").clicked() {
                                actions.push(Action::ToggleHelp);
                            }
                            if ui.button("Command").clicked() {
                                actions.push(Action::ToggleCommandPalette);
                            }
                        });
                    });

                    ui.add_space(10.0);

                    ui.horizontal_wrapped(|ui| {
                        chrome::header_group(ui, |ui| {
                            ui.menu_button("File", |ui| {
                                if ui.button("New Scene").clicked() {
                                    actions.push(Action::NewScene);
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui
                                    .add(egui::Button::new("Open...").shortcut_text("Ctrl+O"))
                                    .clicked()
                                {
                                    actions.push(Action::OpenProject);
                                    ui.close_menu();
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
                                    ui.close_menu();
                                }
                                ui.separator();

                                if !self.settings.recent_files.is_empty() {
                                    ui.menu_button("Recent Files", |ui| {
                                        let files = self.settings.recent_files.clone();
                                        for file_path in &files {
                                            let name = std::path::Path::new(file_path)
                                                .file_name()
                                                .map(|name| name.to_string_lossy().to_string())
                                                .unwrap_or_else(|| file_path.clone());
                                            if ui.button(&name).on_hover_text(file_path).clicked() {
                                                action_open_recent = Some(file_path.clone());
                                                ui.close_menu();
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
                                    ui.close_menu();
                                }
                                let import_idle =
                                    matches!(self.async_state.import_status, ImportStatus::Idle);
                                if ui
                                    .add_enabled(import_idle, egui::Button::new("Import Mesh..."))
                                    .clicked()
                                {
                                    actions.push(Action::ImportMesh);
                                    ui.close_menu();
                                }
                                let export_idle =
                                    matches!(self.async_state.export_status, ExportStatus::Idle);
                                if ui
                                    .add_enabled(
                                        export_idle,
                                        egui::Button::new("Export Mesh...").shortcut_text("Ctrl+E"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::ShowExportDialog);
                                    ui.close_menu();
                                }
                            });

                            ui.menu_button("Edit", |ui| {
                                if ui
                                    .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                                    .clicked()
                                {
                                    actions.push(Action::Undo);
                                    ui.close_menu();
                                }
                                if ui
                                    .add(egui::Button::new("Redo").shortcut_text("Ctrl+Y"))
                                    .clicked()
                                {
                                    actions.push(Action::Redo);
                                    ui.close_menu();
                                }
                                ui.separator();
                                let has_sel = self.ui.node_graph_state.selected.is_some();
                                if ui
                                    .add_enabled(
                                        has_sel,
                                        egui::Button::new("Copy").shortcut_text("Ctrl+C"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::Copy);
                                    ui.close_menu();
                                }
                                let has_clip = self
                                    .doc
                                    .clipboard_node
                                    .is_some_and(|id| self.doc.scene.nodes.contains_key(&id));
                                if ui
                                    .add_enabled(
                                        has_clip,
                                        egui::Button::new("Paste").shortcut_text("Ctrl+V"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::Paste);
                                    ui.close_menu();
                                }
                                if ui
                                    .add_enabled(
                                        has_sel,
                                        egui::Button::new("Duplicate").shortcut_text("Ctrl+D"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::Duplicate);
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui
                                    .add_enabled(
                                        has_sel,
                                        egui::Button::new("Delete").shortcut_text("Del"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::DeleteSelected);
                                    ui.close_menu();
                                }
                            });

                            ui.menu_button("View", |ui| {
                                let has_sel = self.ui.node_graph_state.selected.is_some();
                                if ui
                                    .add_enabled(
                                        has_sel,
                                        egui::Button::new("Focus Selected").shortcut_text("F"),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::FocusSelected);
                                    ui.close_menu();
                                }
                                if ui.button("Frame All").clicked() {
                                    actions.push(Action::FrameAll);
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui.button("Toggle Profiler").clicked() {
                                    actions.push(Action::ToggleDebug);
                                    ui.close_menu();
                                }
                                ui.separator();
                                ui.label("Camera Presets");
                                if ui.button("Front").clicked() {
                                    actions.push(Action::CameraFront);
                                    ui.close_menu();
                                }
                                if ui.button("Top").clicked() {
                                    actions.push(Action::CameraTop);
                                    ui.close_menu();
                                }
                                if ui.button("Right").clicked() {
                                    actions.push(Action::CameraRight);
                                    ui.close_menu();
                                }
                                if ui.button("Back").clicked() {
                                    actions.push(Action::CameraBack);
                                    ui.close_menu();
                                }
                                if ui.button("Left").clicked() {
                                    actions.push(Action::CameraLeft);
                                    ui.close_menu();
                                }
                                if ui.button("Bottom").clicked() {
                                    actions.push(Action::CameraBottom);
                                    ui.close_menu();
                                }
                                if ui.button("Toggle Ortho").clicked() {
                                    actions.push(Action::ToggleOrtho);
                                    ui.close_menu();
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
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.weak("Ctrl+1-9 saves the current view");
                                });
                                ui.separator();
                                ui.menu_button("Panels", |ui| {
                                    use crate::ui::dock::Tab;
                                    for tab in Tab::ALL {
                                        let is_open = self.ui.dock_state.find_tab(tab).is_some();
                                        if ui.selectable_label(is_open, tab.label()).clicked() {
                                            if let Some(location) = self.ui.dock_state.find_tab(tab)
                                            {
                                                self.ui.dock_state.remove_tab(location);
                                            } else {
                                                self.ui
                                                    .dock_state
                                                    .push_to_focused_leaf(tab.clone());
                                            }
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    if ui.button("Reset Layout").clicked() {
                                        self.ui.dock_state = crate::ui::dock::create_dock_state();
                                        ui.close_menu();
                                    }
                                });
                            });
                        });

                        chrome::header_group(ui, |ui| {
                            if ui.button("New").clicked() {
                                actions.push(Action::NewScene);
                            }
                            if ui.button("Open").clicked() {
                                actions.push(Action::OpenProject);
                            }
                            if ui.button("Save").clicked() {
                                actions.push(Action::SaveProject);
                            }

                            let import_idle =
                                matches!(self.async_state.import_status, ImportStatus::Idle);
                            if ui
                                .add_enabled(import_idle, egui::Button::new("Import"))
                                .clicked()
                            {
                                actions.push(Action::ImportMesh);
                            }

                            let export_idle =
                                matches!(self.async_state.export_status, ExportStatus::Idle);
                            if ui
                                .add_enabled(export_idle, egui::Button::new("Export"))
                                .clicked()
                            {
                                actions.push(Action::ShowExportDialog);
                            }

                            if ui.button("Shot").clicked() {
                                actions.push(Action::TakeScreenshot);
                            }
                        });

                        chrome::header_group(ui, |ui| {
                            chrome::badge(ui, BadgeTone::Muted, "Mode");
                            if ui
                                .selectable_label(!tool_is_sculpt, "Select")
                                .on_hover_text("Selection and transform workflow")
                                .clicked()
                            {
                                actions.push(Action::SetTool(ActiveTool::Select));
                            }
                            if ui
                                .selectable_label(tool_is_sculpt, "Sculpt")
                                .on_hover_text("Enter sculpt mode or open sculpt conversion")
                                .clicked()
                            {
                                actions.push(Action::EnterSculptMode);
                            }

                            ui.separator();

                            let gizmo_mode = self.gizmo.mode.clone();
                            if ui
                                .selectable_label(
                                    gizmo_mode == crate::ui::gizmo::GizmoMode::Translate,
                                    "Move",
                                )
                                .clicked()
                            {
                                actions.push(Action::SetGizmoMode(
                                    crate::ui::gizmo::GizmoMode::Translate,
                                ));
                            }
                            if ui
                                .selectable_label(
                                    gizmo_mode == crate::ui::gizmo::GizmoMode::Rotate,
                                    "Rotate",
                                )
                                .clicked()
                            {
                                actions.push(Action::SetGizmoMode(
                                    crate::ui::gizmo::GizmoMode::Rotate,
                                ));
                            }
                            if ui
                                .selectable_label(
                                    gizmo_mode == crate::ui::gizmo::GizmoMode::Scale,
                                    "Scale",
                                )
                                .clicked()
                            {
                                actions
                                    .push(Action::SetGizmoMode(crate::ui::gizmo::GizmoMode::Scale));
                            }
                        });

                        chrome::header_group(ui, |ui| {
                            if ui.button("Focus").clicked() {
                                actions.push(Action::FocusSelected);
                            }
                            if ui.button("Frame").clicked() {
                                actions.push(Action::FrameAll);
                            }
                            if ui.button("Isolate").clicked() {
                                actions.push(Action::ToggleIsolation);
                            }
                            if ui.button("Turntable").clicked() {
                                actions.push(Action::ToggleTurntable);
                            }
                        });
                    });
                });
            });

        if let Some(recent_path) = action_open_recent {
            actions.push(Action::OpenRecentProject(recent_path));
        }
    }

    pub(super) fn show_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_ribbon")
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                chrome::ribbon_frame(ui).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        match &self.doc.sculpt_state {
                            SculptState::Active { session, .. } => {
                                let mode_name = match session.selected_brush {
                                    BrushMode::Add => "Sculpt / Add",
                                    BrushMode::Carve => "Sculpt / Carve",
                                    BrushMode::Smooth => "Sculpt / Smooth",
                                    BrushMode::Flatten => "Sculpt / Flatten",
                                    BrushMode::Inflate => "Sculpt / Inflate",
                                    BrushMode::Grab => "Sculpt / Grab",
                                };
                                chrome::badge(ui, BadgeTone::Accent, mode_name);
                                if let Some(axis) = session.symmetry_axis {
                                    let axis_name = match axis {
                                        0 => "Sym X",
                                        1 => "Sym Y",
                                        _ => "Sym Z",
                                    };
                                    chrome::badge(ui, BadgeTone::Muted, axis_name);
                                }
                            }
                            SculptState::Inactive { .. } => {
                                chrome::badge(ui, BadgeTone::Accent, self.gizmo.mode.label());
                                chrome::badge(ui, BadgeTone::Muted, self.gizmo.space.label());
                            }
                        }

                        if let Some(sel) = self.ui.node_graph_state.selected {
                            if let Some(node) = self.doc.scene.nodes.get(&sel) {
                                chrome::badge(ui, BadgeTone::Muted, format!("Node {}", node.name));
                            }
                        }

                        if self.persistence.scene_dirty {
                            chrome::badge(ui, BadgeTone::Warning, "Unsaved changes");
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if self.doc.sculpt_state.is_active() {
                                chrome::kbd_chip(ui, "F Strength");
                                chrome::kbd_chip(ui, "Shift+F Radius");
                                chrome::kbd_chip(ui, "Shift+Drag Smooth");
                            } else {
                                chrome::kbd_chip(ui, "Del Delete");
                                chrome::kbd_chip(ui, "Scroll Zoom");
                                chrome::kbd_chip(ui, "RMB Pan");
                                chrome::kbd_chip(ui, "LMB Orbit");
                            }
                        });
                    });
                });
            });
    }
}
