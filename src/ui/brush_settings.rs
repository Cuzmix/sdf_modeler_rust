use eframe::egui;

use crate::sculpt::{BrushMode, BrushShape, FalloffMode, SculptState};

/// Draw the expert dockable brush settings panel.
pub fn draw(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    if !sculpt_state.is_active() {
        draw_inactive_hint(ui);
        return;
    }

    ui.label(format!("Active Brush: {}", sculpt_state.selected_brush().label()));
    ui.add_space(6.0);
    ui.separator();
    draw_brush_hot_controls(ui, sculpt_state);
    ui.add_space(6.0);
    ui.separator();
    draw_brush_advanced_controls(ui, sculpt_state);
}

pub fn draw_brush_hot_controls(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    if !sculpt_state.is_active() {
        draw_inactive_hint(ui);
        return;
    }

    let brush_mode = sculpt_state.selected_brush();
    let profile = sculpt_state.selected_profile_mut();

    ui.horizontal(|ui| {
        ui.small("Radius");
        ui.add(
            egui::Slider::new(&mut profile.radius, 0.05..=2.0)
                .show_value(false),
        );
        ui.monospace(format!("{:.2}", profile.radius));
    });

    let (min_strength, max_strength) =
        crate::sculpt::SculptBrushProfile::strength_limits(brush_mode);
    ui.horizontal(|ui| {
        ui.small("Strength");
        ui.add(
            egui::Slider::new(&mut profile.strength, min_strength..=max_strength)
                .show_value(false),
        );
        ui.monospace(format!("{:.2}", profile.strength));
    });

    ui.label(egui::RichText::new("Falloff").small());
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Smooth, "Smooth");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Linear, "Linear");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Sharp, "Sharp");
        ui.selectable_value(&mut profile.falloff_mode, FalloffMode::Flat, "Flat");
    });
}

pub fn draw_brush_advanced_controls(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    if !sculpt_state.is_active() {
        draw_inactive_hint(ui);
        return;
    }

    let brush_mode = sculpt_state.selected_brush();
    let profile = sculpt_state.selected_profile_mut();

    ui.label(egui::RichText::new("Shape").small().strong());
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Sphere, "Sphere");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Cube, "Cube");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Diamond, "Diamond");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Ring, "Ring");
        ui.selectable_value(&mut profile.brush_shape, BrushShape::Cylinder, "Cylinder");
    });

    ui.add_space(6.0);
    ui.separator();

    if brush_mode == BrushMode::Smooth {
        ui.horizontal(|ui| {
            ui.small("Smooth Iterations");
            let mut iters = profile.smooth_iterations as i32;
            ui.add(egui::Slider::new(&mut iters, 1..=10));
            profile.smooth_iterations = iters as u32;
        });
    }

    ui.horizontal(|ui| {
        ui.small("Stabilize");
        ui.add(egui::Slider::new(&mut profile.lazy_radius, 0.0..=0.5));
    });

    ui.horizontal(|ui| {
        ui.small("Spacing");
        ui.add(egui::Slider::new(&mut profile.stroke_spacing, 0.05..=0.6));
    });

    ui.horizontal(|ui| {
        ui.small("Surface");
        ui.add(egui::Slider::new(
            &mut profile.surface_constraint,
            0.0..=1.0,
        ));
    });

    ui.checkbox(&mut profile.front_faces_only, "Front Faces")
        .on_hover_text("Attenuate brush influence on back-facing voxels");

    ui.add_space(6.0);
    ui.separator();

    let mut symmetry_axis = sculpt_state.symmetry_axis();
    ui.label(egui::RichText::new("Symmetry").small().strong());
    ui.horizontal(|ui| {
        ui.selectable_value(&mut symmetry_axis, None, "Off");
        ui.selectable_value(&mut symmetry_axis, Some(0), "X");
        ui.selectable_value(&mut symmetry_axis, Some(1), "Y");
        ui.selectable_value(&mut symmetry_axis, Some(2), "Z");
    });
    sculpt_state.set_symmetry_axis(symmetry_axis);
}

fn draw_inactive_hint(ui: &mut egui::Ui) {
    ui.add_space(20.0);
    ui.vertical_centered(|ui| {
        ui.weak("No active sculpt session.");
        ui.add_space(8.0);
        ui.weak("Press Ctrl+R to start sculpting.");
    });
}
