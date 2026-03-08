use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::graph::history::History;

pub fn draw(ui: &mut egui::Ui, history: &History, actions: &mut ActionSink) {
    ui.heading("History");
    ui.separator();

    ui.horizontal(|ui| {
        if ui
            .add_enabled(history.undo_count() > 0, egui::Button::new("Undo"))
            .clicked()
        {
            actions.push(Action::Undo);
        }
        if ui
            .add_enabled(history.redo_count() > 0, egui::Button::new("Redo"))
            .clicked()
        {
            actions.push(Action::Redo);
        }
    });

    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            // Redo stack (future states, shown above current)
            let redo_labels = history.redo_labels();
            if !redo_labels.is_empty() {
                for label in &redo_labels {
                    ui.weak(format!("  \u{21B7} {}", label));
                }
                ui.separator();
            }

            // Current state marker
            ui.colored_label(
                egui::Color32::from_rgb(100, 200, 100),
                "\u{25B6} Current State",
            );

            // Undo stack (past states, most recent first)
            let undo_labels = history.undo_labels();
            if !undo_labels.is_empty() {
                ui.separator();
                for label in undo_labels.iter().rev() {
                    ui.label(format!("  \u{25CB} {}", label));
                }
            }

            if undo_labels.is_empty() && redo_labels.is_empty() {
                ui.add_space(8.0);
                ui.weak("No history yet");
            }
        });
}
