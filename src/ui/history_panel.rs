use eframe::egui;

use crate::app::actions::{Action, ActionSink};
use crate::graph::history::History;
use crate::ui::chrome::{self, BadgeTone};

#[derive(Clone, Copy)]
enum TimelinePhase {
    Future,
    Current,
    Past,
}

pub fn draw(ui: &mut egui::Ui, history: &History, actions: &mut ActionSink) {
    chrome::panel_header(
        ui,
        "History",
        "Track undo and redo checkpoints with a cleaner timeline-oriented view.",
    );
    ui.add_space(10.0);

    chrome::section_card(
        ui,
        "Controls",
        "Step backward or forward through recent scene changes.",
        |ui| {
            ui.horizontal(|ui| {
                chrome::badge(
                    ui,
                    BadgeTone::Muted,
                    format!("Undo {}", history.undo_count()),
                );
                chrome::badge(
                    ui,
                    BadgeTone::Muted,
                    format!("Redo {}", history.redo_count()),
                );
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
        },
    );

    ui.add_space(10.0);

    let redo_labels = history.redo_labels();
    let undo_labels = history.undo_labels();
    if undo_labels.is_empty() && redo_labels.is_empty() {
        chrome::empty_state(
            ui,
            "No history yet",
            "Scene edits, transforms, and sculpt sessions will appear here as soon as you make changes.",
        );
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let original_spacing = ui.spacing().item_spacing.y;
            ui.spacing_mut().item_spacing.y = 6.0;

            for label in &redo_labels {
                draw_timeline_row(ui, label, TimelinePhase::Future);
            }

            draw_timeline_row(ui, "Now", TimelinePhase::Current);

            for label in undo_labels.iter().rev() {
                draw_timeline_row(ui, label, TimelinePhase::Past);
            }

            ui.spacing_mut().item_spacing.y = original_spacing;
        });
}

fn draw_timeline_row(ui: &mut egui::Ui, label: &str, phase: TimelinePhase) {
    let visuals = ui.visuals();
    let subtle_stroke = visuals.widgets.noninteractive.bg_stroke.color;
    let (icon, fill, stroke, text_color, strong) = match phase {
        TimelinePhase::Future => (
            "\u{25A1}",
            chrome::tokens(ui).muted,
            egui::Stroke::new(1.0, subtle_stroke),
            visuals.weak_text_color(),
            false,
        ),
        TimelinePhase::Current => (
            "\u{25CF}",
            chrome::tokens(ui).badge,
            egui::Stroke::new(1.0, chrome::tokens(ui).ring),
            chrome::tokens(ui).text,
            true,
        ),
        TimelinePhase::Past => (
            "\u{25CB}",
            chrome::tokens(ui).card,
            egui::Stroke::new(1.0, subtle_stroke),
            visuals.text_color(),
            false,
        ),
    };

    chrome::item_frame(ui, matches!(phase, TimelinePhase::Current))
        .fill(fill)
        .stroke(stroke)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.colored_label(stroke.color, icon);
                ui.add_space(6.0);

                let text = egui::RichText::new(label).color(text_color);
                if strong {
                    ui.label(text.strong());
                } else {
                    ui.label(text);
                }
            });
        });
}
