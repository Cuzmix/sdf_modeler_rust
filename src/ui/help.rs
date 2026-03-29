use eframe::egui;

use crate::keymap::{ActionBinding, KeymapConfig};
use crate::ui::chrome::{self, BadgeTone};

/// Draw the keyboard shortcuts help window.
/// Shortcut strings are read from the configurable keymap (single source of truth).
pub fn draw(ctx: &egui::Context, open: &mut bool, keymap: &KeymapConfig) {
    let motion = crate::ui::motion::settings(ctx);
    let window_id = egui::Id::new("keyboard_shortcuts_window");
    let t = crate::ui::motion::surface_open_t(ctx, window_id, *open, motion);
    if !crate::ui::motion::should_draw_surface(*open, t) {
        crate::ui::motion::clear_surface_layers(ctx, window_id);
        return;
    }
    let alpha = crate::ui::motion::fade_alpha(t, motion.reduced_motion);

    // Helper: look up a shortcut string from the keymap, falling back to "—" if unbound.
    let sk = |binding: ActionBinding| -> String {
        keymap
            .format_shortcut(binding)
            .unwrap_or_else(|| "\u{2014}".to_string())
    };

    let mut still_open = true;
    let mut window = egui::Window::new("Keyboard Shortcuts")
        .id(window_id)
        .fade_in(false)
        .resizable(false)
        .default_width(380.0)
        .frame(crate::ui::motion::frame_with_alpha(
            egui::Frame::window(&ctx.style()),
            t,
            motion,
        ));
    if *open {
        window = window.open(&mut still_open);
    }

    if let Some(window_response) = window.show(ctx, |ui| {
        ui.multiply_opacity(alpha);
        chrome::panel_header(
            ui,
            "Keyboard Shortcuts",
            "Core navigation, sculpting, and workflow shortcuts for the current keymap.",
        );
        ui.add_space(10.0);
        egui::Grid::new("shortcuts_grid")
            .num_columns(2)
            .spacing([24.0, 8.0])
            .show(ui, |ui| {
                let section = |ui: &mut egui::Ui, title: &str| {
                    chrome::badge(ui, BadgeTone::Accent, title);
                    ui.label("");
                    ui.end_row();
                };
                let row = |ui: &mut egui::Ui, key: &str, desc: &str| {
                    chrome::kbd_chip(ui, key);
                    ui.label(desc);
                    ui.end_row();
                };

                section(ui, "General");
                row(ui, &sk(ActionBinding::OpenProject), "Open project");
                row(ui, &sk(ActionBinding::SaveProject), "Save project");
                row(ui, &sk(ActionBinding::Undo), "Undo");
                row(ui, &sk(ActionBinding::Redo), "Redo");
                row(ui, &sk(ActionBinding::Copy), "Copy node");
                row(ui, &sk(ActionBinding::Paste), "Paste node");
                row(ui, &sk(ActionBinding::Duplicate), "Duplicate node");
                row(
                    ui,
                    &sk(ActionBinding::DeleteSelected),
                    "Delete selected node",
                );
                row(ui, &sk(ActionBinding::CopyProperties), "Copy properties");
                row(ui, &sk(ActionBinding::PasteProperties), "Paste properties");
                row(ui, &sk(ActionBinding::TakeScreenshot), "Screenshot");
                row(ui, &sk(ActionBinding::ShowExportDialog), "Export OBJ");
                row(
                    ui,
                    &sk(ActionBinding::ToggleCommandPalette),
                    "Command palette",
                );
                row(ui, &sk(ActionBinding::ToggleHelp), "Toggle this help");
                row(ui, &sk(ActionBinding::ToggleDebug), "Toggle profiler");

                ui.separator();
                ui.end_row();
                section(ui, "Camera");
                row(ui, "LMB drag", "Orbit");
                row(ui, "RMB drag", "Pan");
                row(ui, "Scroll", "Zoom");
                row(ui, "Ctrl+Alt+Drag", "Roll camera");
                row(ui, &sk(ActionBinding::FocusSelected), "Focus selected");
                row(ui, &sk(ActionBinding::FrameAll), "Frame all");
                row(
                    ui,
                    &sk(ActionBinding::ToggleOrtho),
                    "Toggle Ortho / Perspective",
                );
                row(ui, &sk(ActionBinding::CameraFront), "Front view");
                row(ui, &sk(ActionBinding::CameraTop), "Top view");
                row(ui, &sk(ActionBinding::CameraRight), "Right view");
                row(ui, &sk(ActionBinding::CameraBack), "Back view");
                row(ui, &sk(ActionBinding::CameraLeft), "Left view");
                row(ui, &sk(ActionBinding::CameraBottom), "Bottom view");
                row(ui, "Gizmo click", "Snap to axis view");
                row(
                    ui,
                    &sk(ActionBinding::ToggleIsolation),
                    "Toggle isolation mode",
                );
                row(
                    ui,
                    &sk(ActionBinding::CycleShadingMode),
                    "Cycle shading mode",
                );
                row(ui, &sk(ActionBinding::ToggleTurntable), "Toggle turntable");
                row(ui, "Ctrl+1-9", "Save camera bookmark");

                ui.separator();
                ui.end_row();
                section(ui, "Gizmo");
                row(ui, &sk(ActionBinding::GizmoTranslate), "Move tool");
                row(ui, &sk(ActionBinding::GizmoRotate), "Rotate tool");
                row(ui, &sk(ActionBinding::GizmoScale), "Scale tool");
                row(
                    ui,
                    &sk(ActionBinding::ToggleGizmoSpace),
                    "Toggle Local / World",
                );
                row(ui, "Ctrl+Drag", "Snap to grid");
                row(ui, "Alt+Drag", "Move pivot");
                row(ui, &sk(ActionBinding::ResetPivot), "Reset pivot");

                ui.separator();
                ui.end_row();
                section(ui, "Sculpt Mode");
                row(ui, "LMB drag", "Paint brush");
                row(ui, "Ctrl+drag", "Invert brush (Add\u{2194}Carve)");
                row(ui, "Shift+drag", "Smooth override");
                row(ui, "RMB drag", "Pan camera");
                row(ui, "MMB drag", "Orbit camera");
                row(
                    ui,
                    &format!(
                        "{} / {}",
                        sk(ActionBinding::SculptBrushShrink),
                        sk(ActionBinding::SculptBrushGrow)
                    ),
                    "Decrease / increase brush size",
                );
                row(ui, &sk(ActionBinding::SculptBrushAdd), "Add brush");
                row(ui, &sk(ActionBinding::SculptBrushCarve), "Carve brush");
                row(ui, &sk(ActionBinding::SculptBrushSmooth), "Smooth brush");
                row(ui, &sk(ActionBinding::SculptBrushFlatten), "Flatten brush");
                row(ui, &sk(ActionBinding::SculptBrushInflate), "Inflate brush");
                row(ui, &sk(ActionBinding::SculptBrushGrab), "Grab brush");
                row(
                    ui,
                    &format!(
                        "{} / {} / {}",
                        sk(ActionBinding::SculptSymmetryX),
                        sk(ActionBinding::SculptSymmetryY),
                        sk(ActionBinding::SculptSymmetryZ)
                    ),
                    "Toggle symmetry axis",
                );

                ui.separator();
                ui.end_row();
                section(ui, "Scene Tree");
                row(ui, "Double-click", "Rename node");
                row(ui, "Right-click", "Context menu");
            });
    }) {
        crate::ui::motion::apply_surface_transform(ctx, &window_response.response, t, motion);
    }

    if *open && !still_open {
        *open = false;
    }
}
