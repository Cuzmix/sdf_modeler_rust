use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{NodeId, Scene};
use crate::keymap::{ActionBinding, KeymapConfig};
use crate::ui::gizmo::GizmoMode;

/// A single entry in the command palette.
struct CommandEntry {
    label: String,
    /// Optional keymap binding for dynamic shortcut display.
    binding: Option<ActionBinding>,
    action: Action,
}

/// Build the list of all available commands.
fn build_entries() -> Vec<CommandEntry> {
    vec![
        // Scene
        CommandEntry {
            label: "New Scene".into(),
            binding: Some(ActionBinding::NewScene),
            action: Action::NewScene,
        },
        CommandEntry {
            label: "Open Project".into(),
            binding: Some(ActionBinding::OpenProject),
            action: Action::OpenProject,
        },
        CommandEntry {
            label: "Save Project".into(),
            binding: Some(ActionBinding::SaveProject),
            action: Action::SaveProject,
        },
        // History
        CommandEntry {
            label: "Undo".into(),
            binding: Some(ActionBinding::Undo),
            action: Action::Undo,
        },
        CommandEntry {
            label: "Redo".into(),
            binding: Some(ActionBinding::Redo),
            action: Action::Redo,
        },
        // Clipboard
        CommandEntry {
            label: "Copy".into(),
            binding: Some(ActionBinding::Copy),
            action: Action::Copy,
        },
        CommandEntry {
            label: "Paste".into(),
            binding: Some(ActionBinding::Paste),
            action: Action::Paste,
        },
        CommandEntry {
            label: "Duplicate".into(),
            binding: Some(ActionBinding::Duplicate),
            action: Action::Duplicate,
        },
        CommandEntry {
            label: "Delete Selected".into(),
            binding: Some(ActionBinding::DeleteSelected),
            action: Action::DeleteSelected,
        },
        // Camera
        CommandEntry {
            label: "Focus Selected".into(),
            binding: Some(ActionBinding::FocusSelected),
            action: Action::FocusSelected,
        },
        CommandEntry {
            label: "Frame All".into(),
            binding: Some(ActionBinding::FrameAll),
            action: Action::FrameAll,
        },
        CommandEntry {
            label: "Camera: Front".into(),
            binding: Some(ActionBinding::CameraFront),
            action: Action::CameraFront,
        },
        CommandEntry {
            label: "Camera: Top".into(),
            binding: Some(ActionBinding::CameraTop),
            action: Action::CameraTop,
        },
        CommandEntry {
            label: "Camera: Right".into(),
            binding: Some(ActionBinding::CameraRight),
            action: Action::CameraRight,
        },
        CommandEntry {
            label: "Camera: Back".into(),
            binding: Some(ActionBinding::CameraBack),
            action: Action::CameraBack,
        },
        CommandEntry {
            label: "Camera: Left".into(),
            binding: Some(ActionBinding::CameraLeft),
            action: Action::CameraLeft,
        },
        CommandEntry {
            label: "Camera: Bottom".into(),
            binding: Some(ActionBinding::CameraBottom),
            action: Action::CameraBottom,
        },
        CommandEntry {
            label: "Toggle Orthographic".into(),
            binding: Some(ActionBinding::ToggleOrtho),
            action: Action::ToggleOrtho,
        },
        // Tools
        CommandEntry {
            label: "Tool: Select".into(),
            binding: None,
            action: Action::SetInteractionMode(crate::app::state::InteractionMode::Select),
        },
        CommandEntry {
            label: "Tool: Sculpt".into(),
            binding: None,
            action: Action::EnterSculptMode,
        },
        CommandEntry {
            label: "Gizmo: Translate".into(),
            binding: Some(ActionBinding::GizmoTranslate),
            action: Action::SetGizmoMode(GizmoMode::Translate),
        },
        CommandEntry {
            label: "Gizmo: Rotate".into(),
            binding: Some(ActionBinding::GizmoRotate),
            action: Action::SetGizmoMode(GizmoMode::Rotate),
        },
        CommandEntry {
            label: "Gizmo: Scale".into(),
            binding: Some(ActionBinding::GizmoScale),
            action: Action::SetGizmoMode(GizmoMode::Scale),
        },
        CommandEntry {
            label: "Toggle Gizmo Space".into(),
            binding: Some(ActionBinding::ToggleGizmoSpace),
            action: Action::ToggleGizmoSpace,
        },
        CommandEntry {
            label: "Reset Pivot".into(),
            binding: Some(ActionBinding::ResetPivot),
            action: Action::ResetPivot,
        },
        // Viewport
        CommandEntry {
            label: "Toggle Isolation Mode".into(),
            binding: Some(ActionBinding::ToggleIsolation),
            action: Action::ToggleIsolation,
        },
        CommandEntry {
            label: "Cycle Shading Mode".into(),
            binding: Some(ActionBinding::CycleShadingMode),
            action: Action::CycleShadingMode,
        },
        CommandEntry {
            label: "Toggle Turntable".into(),
            binding: Some(ActionBinding::ToggleTurntable),
            action: Action::ToggleTurntable,
        },
        // Properties
        CommandEntry {
            label: "Copy Properties".into(),
            binding: Some(ActionBinding::CopyProperties),
            action: Action::CopyProperties,
        },
        CommandEntry {
            label: "Paste Properties".into(),
            binding: Some(ActionBinding::PasteProperties),
            action: Action::PasteProperties,
        },
        // Export
        CommandEntry {
            label: "Export Mesh".into(),
            binding: Some(ActionBinding::ShowExportDialog),
            action: Action::ShowExportDialog,
        },
        CommandEntry {
            label: "Take Screenshot".into(),
            binding: Some(ActionBinding::TakeScreenshot),
            action: Action::TakeScreenshot,
        },
        // UI
        CommandEntry {
            label: "Toggle Help".into(),
            binding: Some(ActionBinding::ToggleHelp),
            action: Action::ToggleHelp,
        },
        CommandEntry {
            label: "Toggle Profiler".into(),
            binding: Some(ActionBinding::ToggleDebug),
            action: Action::ToggleDebug,
        },
        CommandEntry {
            label: "Toggle Settings".into(),
            binding: None,
            action: Action::ToggleSettings,
        },
    ]
}

/// Build node search entries from the scene (prefixed with @).
fn build_node_entries(scene: &Scene) -> Vec<(String, NodeId)> {
    let mut entries: Vec<(String, NodeId)> = scene
        .nodes
        .iter()
        .map(|(&id, node)| (node.name.clone(), id))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Draw the command palette modal. Returns actions to execute.
pub fn draw(
    ctx: &egui::Context,
    open: &mut bool,
    query: &mut String,
    selected_index: &mut usize,
    scene: &Scene,
    keymap: &KeymapConfig,
    actions: &mut ActionSink,
) {
    if !*open {
        return;
    }

    // Dim background
    let screen_rect = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("command_palette_bg"),
    ));
    painter.rect_filled(
        screen_rect,
        0.0,
        egui::Color32::from_rgba_premultiplied(0, 0, 0, 120),
    );

    // Palette window
    let palette_width = 400.0_f32.min(screen_rect.width() - 40.0);
    let palette_pos = egui::pos2(
        screen_rect.center().x - palette_width * 0.5,
        screen_rect.top() + 80.0,
    );

    let area_resp = egui::Area::new(egui::Id::new("command_palette"))
        .order(egui::Order::Foreground)
        .fixed_pos(palette_pos)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style())
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.set_min_width(palette_width);

                    // Search input
                    let input_resp = ui.add(
                        egui::TextEdit::singleline(query)
                            .desired_width(palette_width - 16.0)
                            .hint_text("Type a command or @node name..."),
                    );
                    // Auto-focus on open
                    input_resp.request_focus();

                    ui.separator();

                    let is_node_search = query.starts_with('@');
                    let mut executed_action: Option<Action> = None;
                    let mut executed_select: Option<NodeId> = None;

                    if is_node_search {
                        // Node search mode
                        let search_term = query[1..].to_lowercase();
                        let nodes = build_node_entries(scene);
                        let filtered: Vec<_> = nodes
                            .iter()
                            .filter(|(name, _)| {
                                search_term.is_empty() || name.to_lowercase().contains(&search_term)
                            })
                            .collect();

                        if *selected_index >= filtered.len() && !filtered.is_empty() {
                            *selected_index = filtered.len() - 1;
                        }

                        let max_show = 10.min(filtered.len());
                        egui::ScrollArea::vertical()
                            .max_height(300.0)
                            .show(ui, |ui| {
                                for (i, (name, id)) in filtered.iter().enumerate().take(max_show) {
                                    let is_selected = i == *selected_index;
                                    let resp =
                                        ui.selectable_label(is_selected, format!("  {}", name));
                                    if resp.clicked() {
                                        executed_select = Some(*id);
                                    }
                                }
                                if filtered.is_empty() {
                                    ui.weak("No matching nodes");
                                }
                            });

                        // Keyboard nav
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown))
                            && *selected_index + 1 < filtered.len()
                        {
                            *selected_index += 1;
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) && *selected_index > 0 {
                            *selected_index -= 1;
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::Enter)) && !filtered.is_empty() {
                            if let Some((_, id)) = filtered.get(*selected_index) {
                                executed_select = Some(*id);
                            }
                        }
                    } else {
                        // Command search mode
                        let entries = build_entries();
                        let lower_query = query.to_lowercase();
                        let filtered: Vec<_> = entries
                            .iter()
                            .enumerate()
                            .filter(|(_, e)| {
                                lower_query.is_empty()
                                    || e.label.to_lowercase().contains(&lower_query)
                            })
                            .collect();

                        if *selected_index >= filtered.len() && !filtered.is_empty() {
                            *selected_index = filtered.len() - 1;
                        }

                        let max_show = 10.min(filtered.len());
                        egui::ScrollArea::vertical()
                            .max_height(300.0)
                            .show(ui, |ui| {
                                for (display_idx, (_, entry)) in
                                    filtered.iter().enumerate().take(max_show)
                                {
                                    let is_selected = display_idx == *selected_index;
                                    let mut text = egui::RichText::new(&entry.label);
                                    if is_selected {
                                        text = text.strong();
                                    }
                                    let resp = ui.horizontal(|ui| {
                                        let r = ui.selectable_label(is_selected, text);
                                        // Look up shortcut from keymap (single source of truth)
                                        if let Some(binding) = entry.binding {
                                            if let Some(shortcut) = keymap.format_shortcut(binding)
                                            {
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.weak(&shortcut);
                                                    },
                                                );
                                            }
                                        }
                                        r
                                    });
                                    if resp.inner.clicked() {
                                        executed_action = Some(entry.action.clone());
                                    }
                                }
                                if filtered.is_empty() {
                                    ui.weak("No matching commands");
                                }
                            });

                        // Keyboard nav
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown))
                            && *selected_index + 1 < filtered.len()
                        {
                            *selected_index += 1;
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) && *selected_index > 0 {
                            *selected_index -= 1;
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::Enter)) && !filtered.is_empty() {
                            if let Some((_, entry)) = filtered.get(*selected_index) {
                                executed_action = Some(entry.action.clone());
                            }
                        }
                    }

                    // Execute
                    if let Some(action) = executed_action {
                        actions.push(action);
                        *open = false;
                    }
                    if let Some(id) = executed_select {
                        actions.push(Action::Select(Some(id)));
                        *open = false;
                    }
                });
        });

    // Close on Escape or click outside
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        *open = false;
    }
    // Close if clicked outside the palette area
    let palette_rect = area_resp.response.rect;
    if ctx.input(|i| i.pointer.any_pressed()) {
        if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
            if !palette_rect.contains(pos) {
                *open = false;
            }
        }
    }
}
