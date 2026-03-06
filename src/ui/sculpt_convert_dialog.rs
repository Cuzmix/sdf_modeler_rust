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

            // Resolution presets
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                for &(label, res) in &[("Low", 32u32), ("Medium", 64), ("High", 96), ("Ultra", 128), ("Extreme", 256)] {
                    if ui.selectable_label(state.resolution == res, label).clicked() {
                        state.resolution = res;
                    }
                }
            });

            // Custom resolution input
            ui.horizontal(|ui| {
                ui.label("Custom:");
                let mut res_i32 = state.resolution as i32;
                let response = ui.add(
                    egui::DragValue::new(&mut res_i32)
                        .speed(1)
                        .range(16..=320)
                        .suffix("^3"),
                );
                if response.changed() {
                    state.resolution = (res_i32 as u32).clamp(16, 320);
                }
            });

            // Stats and warnings
            let voxels = (state.resolution as u64).pow(3);
            let mem_mb = (voxels as f64 * 4.0) / (1024.0 * 1024.0);
            ui.weak(format!(
                "{} voxels ({:.1} MB)",
                format_voxel_count(voxels), mem_mb
            ));

            if state.resolution > 256 {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 100, 100),
                    format!("Warning: {:.0} MB RAM — may cause slowdowns or crashes", mem_mb),
                );
            } else if state.resolution > 128 {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "High resolution — sculpting may be slower",
                );
            }

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

/// Format large voxel counts with K/M suffixes for readability.
fn format_voxel_count(voxels: u64) -> String {
    if voxels >= 1_000_000 {
        format!("{:.1}M", voxels as f64 / 1_000_000.0)
    } else if voxels >= 1_000 {
        format!("{:.0}K", voxels as f64 / 1_000.0)
    } else {
        format!("{}", voxels)
    }
}
