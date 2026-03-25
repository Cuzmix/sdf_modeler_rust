use crate::app::actions::{Action, ActionSink};
use crate::graph::history::History;
use crate::ui::egui_compat::{corner_radius, margin_symmetric};

#[derive(Clone, Copy)]
enum TimelinePhase {
    Future,
    Current,
    Past,
}

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

    let redo_labels = history.redo_labels();
    let undo_labels = history.undo_labels();
    if undo_labels.is_empty() && redo_labels.is_empty() {
        ui.add_space(8.0);
        ui.weak("No history yet");
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let original_spacing = ui.spacing().item_spacing.y;
            ui.spacing_mut().item_spacing.y = 3.0;

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
    let subtle_stroke = visuals
        .widgets
        .noninteractive
        .bg_stroke
        .color
        .gamma_multiply(0.35);
    let (icon, fill, stroke, text_color, strong) = match phase {
        TimelinePhase::Future => (
            "\u{25A1}",
            egui::Color32::TRANSPARENT,
            egui::Stroke::new(1.0, subtle_stroke),
            visuals.weak_text_color(),
            false,
        ),
        TimelinePhase::Current => (
            "\u{25CF}",
            egui::Color32::from_rgb(54, 78, 64),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(101, 188, 133)),
            egui::Color32::from_rgb(229, 244, 235),
            true,
        ),
        TimelinePhase::Past => (
            "\u{25CB}",
            egui::Color32::TRANSPARENT,
            egui::Stroke::new(1.0, subtle_stroke),
            visuals.text_color(),
            false,
        ),
    };

    egui::Frame::new()
        .inner_margin(margin_symmetric(8.0, 6.0))
        .fill(fill)
        .stroke(stroke)
        .corner_radius(corner_radius(4.0))
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
