use eframe::egui;

use crate::sculpt::{BrushMode, BrushShape, FalloffMode, SculptState};

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
        ui.add_space(20.0);
        ui.vertical_centered(|ui| {
            ui.weak("No active sculpt session.");
            ui.add_space(8.0);
            ui.weak("Press Ctrl+R to start sculpting.");
        });
    }
}

/// Draw all brush controls when sculpting is active.
fn draw_active_brush_controls(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    let brush_mode = sculpt_state.selected_brush();
    ui.label(format!("Active Brush: {}", brush_mode.label()));
    ui.add_space(6.0);
    ui.separator();

    let profile = sculpt_state.selected_profile_mut();

    // Falloff
    ui.label("Falloff");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Smooth, "Smooth");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Linear, "Linear");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Sharp, "Sharp");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Flat, "Flat");
    });

    ui.add_space(6.0);

    // Shape
    ui.label("Brush Shape");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Sphere, "Sphere");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Cube, "Cube");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Diamond, "Diamond");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Ring, "Ring");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Cylinder, "Cylinder");
    });

    ui.add_space(6.0);
    ui.separator();

    // Radius
    ui.horizontal(|ui| {
        ui.label("Radius:");
        ui.add(egui::Slider::new(&mut profile.radius, 0.05..=2.0));
    });

    // Strength
    ui.horizontal(|ui| {
        ui.label("Strength:");
        let (min_strength, max_strength) =
            crate::sculpt::SculptBrushProfile::strength_limits(brush_mode);
        ui.add(egui::Slider::new(
            &mut profile.strength,
            min_strength..=max_strength,
        ));
    });

    // Smooth iterations (only visible for Smooth brush)
    if brush_mode == BrushMode::Smooth {
        ui.horizontal(|ui| {
            ui.label("Iterations:");
            let mut iters = profile.smooth_iterations as i32;
            ui.add(egui::Slider::new(&mut iters, 1..=10));
            profile.smooth_iterations = iters as u32;
        });
    }

    ui.add_space(6.0);
    ui.separator();

    // Stabilize (lazy radius)
    ui.horizontal(|ui| {
        ui.label("Stabilize:");
        ui.add(egui::Slider::new(&mut profile.lazy_radius, 0.0..=0.5));
    });

    // Stroke spacing (fraction of radius)
    ui.horizontal(|ui| {
        ui.label("Spacing:");
        ui.add(egui::Slider::new(&mut profile.stroke_spacing, 0.05..=0.6));
    });

    // Surface constraint
    ui.horizontal(|ui| {
        ui.label("Surface:");
        ui.add(egui::Slider::new(
            &mut profile.surface_constraint,
            0.0..=1.0,
        ));
    });

    ui.checkbox(&mut profile.front_faces_only, "Front Faces")
        .on_hover_text("Attenuate brush influence on back-facing voxels");

    ui.add_space(6.0);
    ui.separator();

    // Symmetry
    let mut symmetry_axis = sculpt_state.symmetry_axis();
    ui.horizontal(|ui| {
        ui.label("Symmetry:");
        ui.selectable_value(&mut symmetry_axis, None, "Off");
        ui.selectable_value(&mut symmetry_axis, Some(0), "X");
        ui.selectable_value(&mut symmetry_axis, Some(1), "Y");
        ui.selectable_value(&mut symmetry_axis, Some(2), "Z");
    });
    sculpt_state.set_symmetry_axis(symmetry_axis);
}
