use eframe::egui;

/// Draw the keyboard shortcuts help window.
pub fn draw(ctx: &egui::Context, open: &mut bool) {
    if !*open { return; }
    egui::Window::new("Keyboard Shortcuts")
        .open(open)
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
