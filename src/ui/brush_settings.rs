use eframe::egui;

use crate::sculpt::{BrushMode, BrushShape, FalloffMode, SculptState, DEFAULT_BRUSH_STRENGTH};

/// Draw the brush settings panel contents.
///
/// This is the dedicated dockable panel for sculpt brush controls.
/// When sculpting is active, shows all brush parameters (mode, falloff,
/// shape, radius, strength, symmetry, etc.) for direct mutation.
/// When not sculpting, shows an informational message.
pub fn draw(ui: &mut egui::Ui, sculpt_state: &mut SculptState) {
    if let SculptState::Active {
        ref mut brush_mode,
        ref mut brush_radius,
        ref mut brush_strength,
        ref mut falloff_mode,
        ref mut brush_shape,
        ref mut smooth_iterations,
        ref mut lazy_radius,
        ref mut stroke_spacing,
        ref mut surface_constraint,
        ref mut front_faces_only,
        ref mut symmetry_axis,
        ..
    } = sculpt_state
    {
        draw_active_brush_controls(
            ui,
            brush_mode,
            brush_radius,
            brush_strength,
            falloff_mode,
            brush_shape,
            smooth_iterations,
            lazy_radius,
            stroke_spacing,
            surface_constraint,
            front_faces_only,
            symmetry_axis,
        );
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
#[allow(clippy::too_many_arguments)]
fn draw_active_brush_controls(
    ui: &mut egui::Ui,
    brush_mode: &mut BrushMode,
    brush_radius: &mut f32,
    brush_strength: &mut f32,
    falloff_mode: &mut FalloffMode,
    brush_shape: &mut BrushShape,
    smooth_iterations: &mut u32,
    lazy_radius: &mut f32,
    stroke_spacing: &mut f32,
    surface_constraint: &mut f32,
    front_faces_only: &mut bool,
    symmetry_axis: &mut Option<u8>,
) {
    // Brush mode
    let prev_mode = brush_mode.clone();
    ui.label("Brush Mode");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(brush_mode, BrushMode::Add, "Add");
        ui.selectable_value(brush_mode, BrushMode::Carve, "Carve");
        ui.selectable_value(brush_mode, BrushMode::Smooth, "Smooth");
        ui.selectable_value(brush_mode, BrushMode::Flatten, "Flatten");
        ui.selectable_value(brush_mode, BrushMode::Inflate, "Inflate");
        ui.selectable_value(brush_mode, BrushMode::Grab, "Grab");
    });
    // Auto-adjust strength when switching to/from Grab
    if *brush_mode != prev_mode {
        if *brush_mode == BrushMode::Grab && *brush_strength < 0.5 {
            *brush_strength = 1.0;
        } else if prev_mode == BrushMode::Grab && *brush_strength > 0.5 {
            *brush_strength = DEFAULT_BRUSH_STRENGTH;
        }
    }

    ui.add_space(6.0);
    ui.separator();

    // Falloff
    ui.label("Falloff");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(falloff_mode, FalloffMode::Smooth, "Smooth");
        ui.selectable_value(falloff_mode, FalloffMode::Linear, "Linear");
        ui.selectable_value(falloff_mode, FalloffMode::Sharp, "Sharp");
        ui.selectable_value(falloff_mode, FalloffMode::Flat, "Flat");
    });

    ui.add_space(6.0);

    // Shape
    ui.label("Brush Shape");
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(brush_shape, BrushShape::Sphere, "Sphere");
        ui.selectable_value(brush_shape, BrushShape::Cube, "Cube");
        ui.selectable_value(brush_shape, BrushShape::Diamond, "Diamond");
        ui.selectable_value(brush_shape, BrushShape::Ring, "Ring");
        ui.selectable_value(brush_shape, BrushShape::Cylinder, "Cylinder");
    });

    ui.add_space(6.0);
    ui.separator();

    // Radius
    ui.horizontal(|ui| {
        ui.label("Radius:");
        ui.add(egui::Slider::new(brush_radius, 0.05..=2.0));
    });

    // Strength
    ui.horizontal(|ui| {
        ui.label("Strength:");
        let range = if *brush_mode == BrushMode::Grab {
            0.1..=3.0
        } else {
            0.01..=0.5
        };
        ui.add(egui::Slider::new(brush_strength, range));
    });

    // Smooth iterations (only visible for Smooth brush)
    if *brush_mode == BrushMode::Smooth {
        ui.horizontal(|ui| {
            ui.label("Iterations:");
            let mut iters = *smooth_iterations as i32;
            ui.add(egui::Slider::new(&mut iters, 1..=10));
            *smooth_iterations = iters as u32;
        });
    }

    ui.add_space(6.0);
    ui.separator();

    // Stabilize (lazy radius)
    ui.horizontal(|ui| {
        ui.label("Stabilize:");
        ui.add(egui::Slider::new(lazy_radius, 0.0..=0.5));
    });

    // Stroke spacing (fraction of radius)
    ui.horizontal(|ui| {
        ui.label("Spacing:");
        ui.add(egui::Slider::new(stroke_spacing, 0.05..=0.6));
    });

    // Surface constraint
    ui.horizontal(|ui| {
        ui.label("Surface:");
        ui.add(egui::Slider::new(surface_constraint, 0.0..=1.0));
    });

    ui.checkbox(front_faces_only, "Front Faces")
        .on_hover_text("Attenuate brush influence on back-facing voxels");

    ui.add_space(6.0);
    ui.separator();

    // Symmetry
    ui.horizontal(|ui| {
        ui.label("Symmetry:");
        ui.selectable_value(symmetry_axis, None, "Off");
        ui.selectable_value(symmetry_axis, Some(0), "X");
        ui.selectable_value(symmetry_axis, Some(1), "Y");
        ui.selectable_value(symmetry_axis, Some(2), "Z");
    });
}
