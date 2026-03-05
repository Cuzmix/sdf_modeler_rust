use eframe::egui;

use crate::app::actions::{Action, ActionSink, SculptConvertMode};
use crate::app::state::SculptConvertDialog;

/// Draw the "Convert to Sculpt" dialog. Returns true if the dialog is still open.
pub fn draw(
    ctx: &egui::Context,
    dialog: &mut Option<SculptConvertDialog>,
    actions: &mut ActionSink,
) {
    let Some(state) = dialog.as_mut() else {
        return;
    };

    let target = state.target;
    let mut do_convert = false;
    let mut do_cancel = false;

    egui::Window::new("Convert to Sculpt")
        .resizable(false)
        .collapsible(false)
        .default_width(320.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.add_space(4.0);

            ui.label("Choose how to create the sculpt layer:");
            ui.add_space(8.0);

            // Bake mode selection
            ui.radio_value(
                &mut state.mode,
                SculptConvertMode::BakeActiveNode,
                "Bake active node",
            );
            ui.radio_value(
                &mut state.mode,
                SculptConvertMode::BakeWholeScene,
                "Bake whole scene",
            );
            ui.radio_value(
                &mut state.mode,
                SculptConvertMode::BakeWholeSceneFlatten,
                "Bake whole scene + flatten",
            );

            ui.add_space(12.0);

            // Resolution picker — discrete steps
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                let resolutions: &[u32] = &[32, 48, 64, 96, 128];
                for &res in resolutions {
                    let label = format!("{}", res);
                    if ui.selectable_label(state.resolution == res, &label).clicked() {
                        state.resolution = res;
                    }
                }
            });

            let voxels = (state.resolution as u64).pow(3);
            ui.weak(format!(
                "{} voxels (~{:.1} MB)",
                voxels,
                (voxels as f64 * 4.0) / (1024.0 * 1024.0)
            ));

            if state.mode == SculptConvertMode::BakeWholeSceneFlatten {
                ui.add_space(4.0);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 180, 80),
                    "Flatten is destructive — the original nodes will be replaced.",
                );
            }

            ui.add_space(12.0);

            // Buttons
            ui.horizontal(|ui| {
                if ui.button("Convert").clicked() {
                    do_convert = true;
                }
                if ui.button("Cancel").clicked() {
                    do_cancel = true;
                }
            });
        });

    if do_convert {
        let mode = state.mode;
        let resolution = state.resolution;
        actions.push(Action::CommitSculptConvert {
            target,
            mode,
            resolution,
        });
        *dialog = None;
    } else if do_cancel {
        *dialog = None;
    }
}
