use crate::app::frontend_models::ShellSnapshot;
use crate::app::slint_frontend::InspectorPanelState;

pub(super) fn build_inspector_panel_state(snapshot: &ShellSnapshot) -> InspectorPanelState {
    let transform = snapshot.inspector.transform.as_ref();
    let material = snapshot.inspector.material.as_ref();
    let operation = snapshot.inspector.operation.as_ref();
    let sculpt = snapshot.inspector.sculpt.as_ref();
    let light = snapshot.inspector.light.as_ref();

    InspectorPanelState {
        title: snapshot.inspector.title.clone().into(),
        chips: snapshot.inspector.chips.join(" | ").into(),
        display: join_lines(&snapshot.inspector.display_lines).into(),
        property_summary: join_lines(&snapshot.inspector.property_lines).into(),
        has_transform: transform.is_some(),
        transform_pos_x: transform.map_or(0.0, |model| model.position[0]),
        transform_pos_y: transform.map_or(0.0, |model| model.position[1]),
        transform_pos_z: transform.map_or(0.0, |model| model.position[2]),
        transform_rot_x: transform.map_or(0.0, |model| model.rotation_deg[0]),
        transform_rot_y: transform.map_or(0.0, |model| model.rotation_deg[1]),
        transform_rot_z: transform.map_or(0.0, |model| model.rotation_deg[2]),
        selected_scale_x: transform.map_or(1.0, |model| model.scale[0]),
        selected_scale_y: transform.map_or(1.0, |model| model.scale[1]),
        selected_scale_z: transform.map_or(1.0, |model| model.scale[2]),
        can_scale: transform.is_some_and(|model| model.can_scale),
        has_material: material.is_some(),
        material_color_r: material.map_or(0.0, |model| model.base_color[0]),
        material_color_g: material.map_or(0.0, |model| model.base_color[1]),
        material_color_b: material.map_or(0.0, |model| model.base_color[2]),
        material_roughness: material.map_or(0.0, |model| model.roughness),
        material_metallic: material.map_or(0.0, |model| model.metallic),
        has_operation: operation.is_some(),
        operation_label: operation
            .map(|model| model.op_label.clone())
            .unwrap_or_default()
            .into(),
        operation_smooth_k: operation.map_or(0.0, |model| model.smooth_k),
        operation_steps: operation.map_or(0.0, |model| model.steps),
        operation_color_blend: operation.map_or(0.0, |model| model.color_blend),
        has_sculpt: sculpt.is_some(),
        sculpt_resolution: sculpt.map_or(0, |model| model.desired_resolution as i32),
        sculpt_layer_intensity: sculpt.map_or(0.0, |model| model.layer_intensity),
        sculpt_brush_radius: sculpt.map_or(0.0, |model| model.brush_radius),
        sculpt_brush_strength: sculpt.map_or(0.0, |model| model.brush_strength),
        has_light: light.is_some(),
        light_label: light
            .map(|model| model.light_type_label.clone())
            .unwrap_or_default()
            .into(),
        light_color_r: light.map_or(0.0, |model| model.color[0]),
        light_color_g: light.map_or(0.0, |model| model.color[1]),
        light_color_b: light.map_or(0.0, |model| model.color[2]),
        light_intensity: light.map_or(0.0, |model| model.intensity),
        light_range: light.map_or(0.0, |model| model.range),
        light_cast_shadows: light.is_some_and(|model| model.cast_shadows),
        light_volumetric: light.is_some_and(|model| model.volumetric),
        light_volumetric_density: light.map_or(0.0, |model| model.volumetric_density),
    }
}

fn join_lines(lines: &[String]) -> String {
    lines.join("\n")
}
