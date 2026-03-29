use eframe::egui;

use crate::sculpt::{BrushMode, BrushShape, FalloffMode, SculptState};
use crate::ui::chrome::{self, BadgeTone};

/// Draw the brush settings panel contents.
///
/// This is the dedicated dockable panel for sculpt brush controls.
/// When sculpting is active, shows all brush parameters (mode, falloff,
/// shape, radius, strength, symmetry, etc.) for direct mutation.
/// When not sculpting, shows an informational message.
pub fn draw(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    if sculpt_state.is_active() {
        draw_active_brush_controls(ui, sculpt_state);
    } else {
        chrome::panel_header(
            ui,
            "Brush Settings",
            "Brush mode, falloff, and stroke tuning appear here while sculpt mode is active.",
        );
        chrome::empty_state(
            ui,
            "No active sculpt session",
            "Select a sculpt node and enter sculpt mode to unlock brush presets, falloff, and stroke controls.",
        );
    }
}

/// Draw all brush controls when sculpting is active.
fn draw_active_brush_controls(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    chrome::panel_header(
        ui,
        "Brush Settings",
        "Tune brush behavior, stroke feel, and symmetry without leaving the sculpt workflow.",
    );
    ui.add_space(10.0);

    // Brush mode
    let prev_mode = sculpt_state.selected_brush();
    let mut brush_mode = prev_mode;
    chrome::section_card(
        ui,
        "Brush Mode",
        "Switch between additive, smoothing, and deformation tools.",
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut brush_mode, BrushMode::Add, "Add");
                ui.selectable_value(&mut brush_mode, BrushMode::Carve, "Carve");
                ui.selectable_value(&mut brush_mode, BrushMode::Smooth, "Smooth");
                ui.selectable_value(&mut brush_mode, BrushMode::Flatten, "Flatten");
                ui.selectable_value(&mut brush_mode, BrushMode::Inflate, "Inflate");
                ui.selectable_value(&mut brush_mode, BrushMode::Grab, "Grab");
            });
        },
    );
    if brush_mode != prev_mode {
        sculpt_state.set_selected_brush(brush_mode);
    }

    let brush_mode = sculpt_state.selected_brush();
    let profile = sculpt_state.selected_profile_mut();
    let brush_mode_label = match brush_mode {
        BrushMode::Add => "Add",
        BrushMode::Carve => "Carve",
        BrushMode::Smooth => "Smooth",
        BrushMode::Flatten => "Flatten",
        BrushMode::Inflate => "Inflate",
        BrushMode::Grab => "Grab",
    };
    ui.add_space(10.0);

    chrome::section_card(
        ui,
        "Profile",
        "Shape the brush falloff, footprint, and main influence range.",
        |ui| {
            ui.horizontal(|ui| {
                chrome::badge(ui, BadgeTone::Muted, format!("Profile: {brush_mode_label}"));
            });
            ui.add_space(6.0);
            ui.label("Falloff");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Smooth, "Smooth");
                ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Linear, "Linear");
                ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Sharp, "Sharp");
                ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Flat, "Flat");
            });

            ui.add_space(8.0);
            ui.label("Brush Shape");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut profile.brush_shape, BrushShape::Sphere, "Sphere");
                ui.selectable_value(&mut profile.brush_shape, BrushShape::Cube, "Cube");
                ui.selectable_value(&mut profile.brush_shape, BrushShape::Diamond, "Diamond");
                ui.selectable_value(&mut profile.brush_shape, BrushShape::Ring, "Ring");
                ui.selectable_value(&mut profile.brush_shape, BrushShape::Cylinder, "Cylinder");
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Radius");
                ui.add(egui::Slider::new(&mut profile.radius, 0.05..=2.0));
            });
            ui.horizontal(|ui| {
                ui.label("Strength");
                let (min_strength, max_strength) =
                    crate::sculpt::SculptBrushProfile::strength_limits(brush_mode);
                ui.add(egui::Slider::new(
                    &mut profile.strength,
                    min_strength..=max_strength,
                ));
            });

            if brush_mode == BrushMode::Smooth {
                ui.horizontal(|ui| {
                    ui.label("Iterations");
                    let mut iters = profile.smooth_iterations as i32;
                    ui.add(egui::Slider::new(&mut iters, 1..=10));
                    profile.smooth_iterations = iters as u32;
                });
            }
        },
    );

    ui.add_space(10.0);

    chrome::section_card(
        ui,
        "Stroke Feel",
        "Retune stroke smoothing, spacing, and surface adherence.",
        |ui| {
            ui.horizontal(|ui| {
                ui.label("Stabilize");
                ui.add(egui::Slider::new(&mut profile.lazy_radius, 0.0..=0.5));
            });
            ui.horizontal(|ui| {
                ui.label("Spacing");
                ui.add(egui::Slider::new(&mut profile.stroke_spacing, 0.05..=0.6));
            });
            ui.horizontal(|ui| {
                ui.label("Surface");
                ui.add(egui::Slider::new(
                    &mut profile.surface_constraint,
                    0.0..=1.0,
                ));
            });
            ui.checkbox(&mut profile.front_faces_only, "Front Faces")
                .on_hover_text("Attenuate brush influence on back-facing voxels");
        },
    );

    ui.add_space(10.0);

    let mut symmetry_axis = sculpt_state.symmetry_axis();
    chrome::section_card(
        ui,
        "Symmetry",
        "Mirror strokes across a world axis for cleaner repeated edits.",
        |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut symmetry_axis, None, "Off");
                ui.selectable_value(&mut symmetry_axis, Some(0), "X");
                ui.selectable_value(&mut symmetry_axis, Some(1), "Y");
                ui.selectable_value(&mut symmetry_axis, Some(2), "Z");
            });
        },
    );
    sculpt_state.set_symmetry_axis(symmetry_axis);
}
