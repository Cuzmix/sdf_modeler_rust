use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::keymap::{ActionBinding, KeyCombo, KeymapConfig, SerializableKey};
use crate::settings::{FrontendKind, Settings};

/// Draw the System Settings window. Pushes `Action::SettingsChanged` if a
/// shader-affecting setting changed.
pub fn draw(
    ctx: &egui::Context,
    open: &mut bool,
    settings: &mut Settings,
    show_debug: &mut bool,
    initial_vsync: bool,
    actions: &mut ActionSink,
    rebinding_action: &mut Option<ActionBinding>,
) {
    let before_render = settings.render.clone();
    let before_frontend = settings.preferred_frontend;
    let mut imported = false;

    egui::Window::new("Settings")
        .open(open)
        .default_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            // --- Top toolbar: Reset / Export / Import ---
            ui.horizontal(|ui| {
                if ui.button("Reset All").on_hover_text("Reset all settings to defaults").clicked() {
                    let recent = std::mem::take(&mut settings.recent_files);
                    *settings = Settings::default();
                    settings.recent_files = recent;
                    settings.save();
                }
                ui.separator();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if ui.button("Export...").on_hover_text("Save settings to a file").clicked() {
                        settings.export_dialog();
                    }
                    if ui.button("Import...").on_hover_text("Load settings from a file").clicked() {
                        imported = settings.import_dialog();
                    }
                }
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // --- Display ---
                egui::CollapsingHeader::new("Display")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.show_fps_overlay, "Show FPS Overlay")
                            .on_hover_text("Display FPS counter in the top-left of the viewport");
                        ui.checkbox(show_debug, "Show Profiler")
                            .on_hover_text("Toggle the profiler window (F4)");
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Frontend");
                            egui::ComboBox::from_id_salt("preferred_frontend")
                                .selected_text(settings.preferred_frontend.label())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut settings.preferred_frontend,
                                        FrontendKind::Egui,
                                        FrontendKind::Egui.label(),
                                    );
                                    ui.selectable_value(
                                        &mut settings.preferred_frontend,
                                        FrontendKind::Slint,
                                        FrontendKind::Slint.label(),
                                    );
                                });
                        });
                        ui.small("(applies on next launch)");
                        ui.separator();
                        ui.checkbox(&mut settings.vsync_enabled, "VSync");
                        if settings.vsync_enabled != initial_vsync {
                            ui.weak("(restart required)");
                        }
                        ui.checkbox(&mut settings.continuous_repaint, "Continuous Repaint")
                            .on_hover_text("Force repaint every frame (useful for benchmarking)");
                    });

                // --- Viewport ---
                egui::CollapsingHeader::new("Viewport")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.render.show_grid, "Show Grid")
                            .on_hover_text("Display ground plane grid at Y=0");
                        ui.checkbox(&mut settings.render.show_node_labels, "Show Node Labels")
                            .on_hover_text("Display node names at their 3D positions in the viewport");
                        ui.checkbox(&mut settings.render.show_bounding_box, "Show Bounding Box")
                            .on_hover_text("Display wireframe bounding box around the selected node");
                        ui.checkbox(&mut settings.render.show_light_gizmos, "Show Light Gizmos")
                            .on_hover_text("Display billboard icons and wireframe gizmos for light nodes");
                        ui.checkbox(&mut settings.render.clamp_orbit_pitch, "Clamp Orbit Pitch")
                            .on_hover_text("Limit vertical orbit to +/-89 deg. When off, allows full 360 deg gimbal rotation.");
                        ui.separator();
                        labeled_slider(ui, "Roll Sensitivity", &mut settings.render.roll_sensitivity, 0.001..=0.02, false,
                            "How fast Ctrl+Alt+drag and touch twist roll the camera");
                        ui.checkbox(&mut settings.render.invert_roll, "Invert Roll")
                            .on_hover_text("Reverse the roll direction for both touch twist and Ctrl+Alt+drag");
                    });

                // --- Snapping ---
                egui::CollapsingHeader::new("Snapping")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label("Hold Ctrl while dragging a gizmo to snap.");
                        labeled_slider(ui, "Translate", &mut settings.snap.translate_snap, 0.05..=2.0, false,
                            "Snap increment for position (world units)");
                        labeled_slider(ui, "Rotate (deg)", &mut settings.snap.rotate_snap, 1.0..=90.0, false,
                            "Snap increment for rotation (degrees)");
                        labeled_slider(ui, "Scale", &mut settings.snap.scale_snap, 0.01..=1.0, false,
                            "Snap increment for scale");
                    });

                // --- Touch Input ---
                egui::CollapsingHeader::new("Touch / Tablet Input")
                    .default_open(true)
                    .show(ui, |ui| {
                        labeled_slider(ui, "Zoom Sensitivity", &mut settings.render.touch_zoom_sensitivity, 100.0..=2000.0, false,
                            "How fast pinch-to-zoom responds");
                        ui.checkbox(&mut settings.render.invert_touch_pan, "Invert Touch Pan")
                            .on_hover_text("Reverse two-finger pan direction");
                        ui.separator();
                        ui.checkbox(&mut settings.render.pressure_sensitivity, "Pen Pressure Sensitivity")
                            .on_hover_text("Modulate sculpt brush strength by tablet pen pressure");
                    });

                // --- Auto-save ---
                egui::CollapsingHeader::new("Auto-save")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.auto_save_enabled, "Enable Auto-save");
                        ui.add_enabled_ui(settings.auto_save_enabled, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Interval:");
                                ui.add(egui::DragValue::new(&mut settings.auto_save_interval_secs)
                                    .range(30..=600).suffix("s").speed(5));
                            });
                        });
                    });

                // --- Resolution Limits ---
                egui::CollapsingHeader::new("Resolution Limits")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Max Export Resolution:");
                            ui.add(
                                egui::DragValue::new(&mut settings.max_export_resolution)
                                    .speed(1)
                                    .range(64..=4096_u32)
                                    .suffix("^3"),
                            );
                        });
                        ui.weak("Maximum resolution allowed in the export dialog.");

                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            ui.label("Max Sculpt Resolution:");
                            ui.add(
                                egui::DragValue::new(&mut settings.max_sculpt_resolution)
                                    .speed(1)
                                    .range(16..=512_u32)
                                    .suffix("^3"),
                            );
                        });
                        let voxel_bytes = (settings.max_sculpt_resolution as u64).pow(3) * 4;
                        let gpu_limit_bytes: u64 = 1 << 27; // 128MB
                        if voxel_bytes > gpu_limit_bytes {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 100, 100),
                                format!(
                                    "Exceeds GPU buffer limit ({} MB > {} MB) -- will be rejected at bake time.",
                                    voxel_bytes / (1024 * 1024),
                                    gpu_limit_bytes / (1024 * 1024),
                                ),
                            );
                        } else {
                            ui.weak(format!(
                                "GPU buffer usage: {} / {} MB",
                                voxel_bytes / (1024 * 1024),
                                gpu_limit_bytes / (1024 * 1024),
                            ));
                        }
                    });

                // --- Performance ---
                let config = &mut settings.render;
                egui::CollapsingHeader::new("Performance")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.checkbox(&mut config.sculpt_fast_mode, "Fast mode while sculpting")
                            .on_hover_text("Half steps + skip AO/shadows during brush strokes");
                        ui.checkbox(&mut config.auto_reduce_steps, "Auto-reduce steps (multi-sculpt)")
                            .on_hover_text("Halve march steps when 2+ sculpt nodes exist");
                        ui.separator();
                        // Anti-aliasing mode (convenience for rest scale)
                        ui.horizontal(|ui| {
                            ui.label("Anti-Aliasing:");
                            let current = if config.rest_render_scale <= 1.01 {
                                "Off"
                            } else if config.rest_render_scale <= 1.45 {
                                "SSAA 1.4x"
                            } else {
                                "SSAA 2x"
                            };
                            egui::ComboBox::from_id_salt("aa_mode")
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current == "Off", "Off (1.0x)").clicked() {
                                        config.rest_render_scale = 1.0;
                                    }
                                    if ui.selectable_label(current == "SSAA 1.4x", "SSAA 1.4x").clicked() {
                                        config.rest_render_scale = 1.414;
                                    }
                                    if ui.selectable_label(current == "SSAA 2x", "SSAA 2x").clicked() {
                                        config.rest_render_scale = 2.0;
                                    }
                                });
                        });
                        labeled_slider(ui, "Interaction Scale", &mut config.interaction_render_scale, 0.25..=1.0, false,
                            "Render resolution during orbit/sculpt (0.5 = half res)");
                        labeled_slider(ui, "Rest Scale", &mut config.rest_render_scale, 0.25..=2.0, false,
                            "Render resolution when idle (1.0 = native, 2.0 = 2x SSAA)");
                        ui.separator();
                        ui.checkbox(&mut config.composite_volume_enabled, "Composite Volume Cache")
                            .on_hover_text("Pre-composite all sculpts into a single 3D texture.\nDecouples render cost from sculpt count.");
                        ui.add_enabled_ui(config.composite_volume_enabled, |ui| {
                            let mut res_i32 = config.composite_volume_resolution as i32;
                            ui.horizontal(|ui| {
                                ui.label("Volume Resolution");
                                ui.add(egui::Slider::new(&mut res_i32, 64..=256));
                            });
                            config.composite_volume_resolution = res_i32 as u32;
                            ui.indent("comp_hint", |ui| {
                                let mem_mb = (res_i32 as f32).powi(3) * 12.0 / (1024.0 * 1024.0);
                                ui.weak(format!("~{:.0} MB VRAM ({}^3 x 3 textures)", mem_mb, res_i32));
                            });
                        });
                    });

                // --- Keybindings ---
                draw_keybindings_section(ui, &mut settings.keymap, rebinding_action);
            });
        });

    if imported || settings.render != before_render || settings.preferred_frontend != before_frontend {
        actions.push(Action::SettingsChanged);
    }
}

// ---------------------------------------------------------------------------
// Keybindings section
// ---------------------------------------------------------------------------

fn draw_keybindings_section(
    ui: &mut egui::Ui,
    keymap: &mut KeymapConfig,
    rebinding_action: &mut Option<ActionBinding>,
) {
    egui::CollapsingHeader::new("Keybindings")
        .default_open(false)
        .show(ui, |ui| {
            // Toolbar: Reset / Export / Import
            ui.horizontal(|ui| {
                if ui.button("Reset to Defaults").on_hover_text("Restore all keybindings to defaults").clicked() {
                    keymap.reset_to_defaults();
                    *rebinding_action = None;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.separator();
                    if ui.button("Export...").on_hover_text("Save keybindings to a JSON file").clicked() {
                        export_keymap_dialog(keymap);
                    }
                    if ui.button("Import...").on_hover_text("Load keybindings from a JSON file").clicked() {
                        import_keymap_dialog(keymap);
                        *rebinding_action = None;
                    }
                }
            });
            ui.separator();

            // If currently rebinding, capture key presses
            if let Some(action) = *rebinding_action {
                capture_rebind_key(ui, keymap, action, rebinding_action);
            }

            // Group actions by category and render as a table
            let categories = ["General", "Camera", "Tools", "Viewport", "Sculpt"];
            for category in categories {
                egui::CollapsingHeader::new(category)
                    .default_open(category == "General")
                    .show(ui, |ui| {
                        egui::Grid::new(format!("keymap_grid_{}", category))
                            .num_columns(3)
                            .spacing([8.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for &action in ActionBinding::ALL {
                                    if action.category() != category {
                                        continue;
                                    }
                                    draw_binding_row(ui, keymap, action, rebinding_action);
                                }
                            });
                    });
            }
        });
}

fn draw_binding_row(
    ui: &mut egui::Ui,
    keymap: &mut KeymapConfig,
    action: ActionBinding,
    rebinding_action: &mut Option<ActionBinding>,
) {
    // Column 1: Action label
    ui.label(action.label());

    // Column 2: Current shortcut
    let is_rebinding_this = *rebinding_action == Some(action);
    if is_rebinding_this {
        ui.colored_label(egui::Color32::YELLOW, "Press key...");
    } else {
        let shortcut_text = keymap.format_shortcut(action)
            .unwrap_or_else(|| "--".to_string());
        ui.monospace(&shortcut_text);
    }

    // Column 3: Rebind / Cancel button
    if is_rebinding_this {
        if ui.small_button("Cancel").clicked() {
            *rebinding_action = None;
        }
    } else if ui.small_button("Rebind").clicked() {
        *rebinding_action = Some(action);
    }

    ui.end_row();
}

fn capture_rebind_key(
    ui: &mut egui::Ui,
    keymap: &mut KeymapConfig,
    action: ActionBinding,
    rebinding_action: &mut Option<ActionBinding>,
) {
    // Check for any key press events this frame
    let events = ui.input(|input| input.events.clone());
    for event in &events {
        if let egui::Event::Key { key, pressed: true, modifiers, .. } = event {
            // Skip bare modifier keys (Ctrl/Shift/Alt alone)
            if matches!(key,
                egui::Key::Backspace | egui::Key::Insert | egui::Key::PageUp | egui::Key::PageDown
            ) {
                continue;
            }

            // Escape cancels the rebind
            if *key == egui::Key::Escape && !modifiers.ctrl && !modifiers.shift && !modifiers.alt {
                *rebinding_action = None;
                return;
            }

            // Try to convert to our SerializableKey
            if let Some(ser_key) = SerializableKey::from_egui(*key) {
                let new_combo = KeyCombo {
                    key: ser_key,
                    ctrl: modifiers.ctrl,
                    shift: modifiers.shift,
                    alt: modifiers.alt,
                };

                // Check for conflicts
                if let Some(conflicting) = keymap.find_conflict(&new_combo, action) {
                    // Remove the conflicting binding to resolve the conflict
                    log::warn!(
                        "Keybinding conflict: {} was bound to {}. Unbound {}.",
                        new_combo, conflicting.label(), conflicting.label()
                    );
                    keymap.remove_binding(conflicting);
                }

                keymap.set_binding(action, new_combo);
                *rebinding_action = None;
                return;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn export_keymap_dialog(keymap: &KeymapConfig) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Export Keybindings")
        .add_filter("JSON", &["json"])
        .set_file_name("keybindings.json")
        .save_file()
    {
        match serde_json::to_string_pretty(keymap) {
            Ok(json) => {
                if let Err(err) = std::fs::write(&path, json) {
                    log::error!("Failed to export keybindings: {}", err);
                }
            }
            Err(err) => log::error!("Failed to serialize keybindings: {}", err),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn import_keymap_dialog(keymap: &mut KeymapConfig) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Import Keybindings")
        .add_filter("JSON", &["json"])
        .pick_file()
    {
        match std::fs::read_to_string(&path) {
            Ok(json) => match serde_json::from_str::<KeymapConfig>(&json) {
                Ok(imported) => {
                    *keymap = imported;
                }
                Err(err) => log::error!("Failed to parse keybindings file: {}", err),
            },
            Err(err) => log::error!("Failed to read keybindings file: {}", err),
        }
    }
}

fn labeled_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    tooltip: &str,
) {
    ui.horizontal(|ui| {
        let lbl = ui.label(label);
        if !tooltip.is_empty() {
            lbl.on_hover_text(tooltip);
        }
        let mut slider = egui::Slider::new(value, range);
        if logarithmic {
            slider = slider.logarithmic(true);
        }
        ui.add(slider);
    });
}

