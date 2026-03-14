use std::sync::atomic::Ordering;

use eframe::egui;

use crate::sculpt::{BrushMode, SculptState};

use super::actions::{Action, ActionSink};
use super::{BakeStatus, ExportStatus, ImportStatus, SdfApp};

impl SdfApp {
    /// Draw the menu bar and push any triggered actions into the action sink.
    /// The menu bar reads state to show enabled/disabled items, and emits
    /// actions for anything the user clicks — no direct state mutation.
    pub(super) fn show_menu_bar(&mut self, ctx: &egui::Context, actions: &mut ActionSink) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // --- File ---
                ui.menu_button("File", |ui| {
                    if ui
                        .add(egui::Button::new("Screenshot...").shortcut_text("Ctrl+P"))
                        .clicked()
                    {
                        actions.push(Action::TakeScreenshot);
                        ui.close_menu();
                    }
                    let import_idle = matches!(self.async_state.import_status, ImportStatus::Idle);
                    if ui
                        .add_enabled(import_idle, egui::Button::new("Import Mesh..."))
                        .clicked()
                    {
                        actions.push(Action::ImportMesh);
                        ui.close_menu();
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
                        ui.close_menu();
                    }
                });

                // --- Edit ---
                ui.menu_button("Edit", |ui| {
                    let has_sel = self.ui.node_graph_state.selected.is_some();
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy").shortcut_text("Ctrl+C"))
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
                        .add_enabled(has_clip, egui::Button::new("Paste").shortcut_text("Ctrl+V"))
                        .clicked()
                    {
                        actions.push(Action::Paste);
                        ui.close_menu();
                    }
                });

                // --- View ---
                ui.menu_button("View", |ui| {
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
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Panels", |ui| {
                        use crate::ui::dock::Tab;
                        for tab in Tab::ALL {
                            let is_open = self.ui.dock_state.find_tab(tab).is_some();
                            if ui.selectable_label(is_open, tab.label()).clicked() {
                                if let Some(location) = self.ui.dock_state.find_tab(tab) {
                                    self.ui.dock_state.remove_tab(location);
                                } else {
                                    self.ui.dock_state.push_to_focused_leaf(tab.clone());
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
                    ui.menu_button("Workspace", |ui| {
                        use crate::app::actions::WorkspacePreset;
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
                            .on_hover_text("Large viewport with properties sidebar")
                            .clicked()
                        {
                            actions.push(Action::SetWorkspace(WorkspacePreset::Sculpting));
                            ui.close_menu();
                        }
                        if ui
                            .button("Rendering")
                            .on_hover_text("Viewport with render settings sidebar")
                            .clicked()
                        {
                            actions.push(Action::SetWorkspace(WorkspacePreset::Rendering));
                            ui.close_menu();
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
                        ui.close_menu();
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
    }

    pub(super) fn show_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Mode indicator
                    match &self.doc.sculpt_state {
                        SculptState::Active {
                            brush_mode,
                            symmetry_axis,
                            ..
                        } => {
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
                                self.gizmo.mode.label(),
                            );
                            ui.separator();
                            ui.weak(self.gizmo.space.label());
                        }
                    }

                    ui.separator();

                    // Selection info
                    if let Some(sel) = self.ui.node_graph_state.selected {
                        if let Some(node) = self.doc.scene.nodes.get(&sel) {
                            ui.weak(&node.name);
                        }
                    }

                    // Right-aligned control hints
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.doc.sculpt_state.is_active() {
                            ui.weak("LMB: Paint | RMB: Pan | MMB: Orbit | 1-6: Brush | X/Y/Z: Sym");
                        } else {
                            ui.weak("LMB: Orbit | RMB: Pan | Scroll: Zoom | Del: Delete");
                        }
                    });
                });
            });
    }
}
