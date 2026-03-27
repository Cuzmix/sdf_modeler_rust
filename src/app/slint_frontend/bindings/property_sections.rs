use crate::app::frontend_models::{
    InspectorBoolFieldModel, InspectorLightModel, InspectorMaterialModel, InspectorOperationModel,
    InspectorScalarFieldModel, InspectorSculptModel, InspectorTransformModel,
};
use crate::app::slint_frontend::{
    LightSectionState, MaterialSectionState, OperationSectionState, PropertyBoolFieldView,
    PropertyScalarFieldView, SculptSectionState, TransformSectionState,
};

pub(super) fn transform_section(
    section: Option<&InspectorTransformModel>,
) -> TransformSectionState {
    section.map_or_else(blank_transform_section, |section| TransformSectionState {
        visible: true,
        can_scale: section.can_scale,
        multi_editing: section.multi_editing,
        pos_x: scalar_field(&section.position[0]),
        pos_y: scalar_field(&section.position[1]),
        pos_z: scalar_field(&section.position[2]),
        rot_x: scalar_field(&section.rotation_deg[0]),
        rot_y: scalar_field(&section.rotation_deg[1]),
        rot_z: scalar_field(&section.rotation_deg[2]),
        scale_x: scalar_field(&section.scale[0]),
        scale_y: scalar_field(&section.scale[1]),
        scale_z: scalar_field(&section.scale[2]),
    })
}

pub(super) fn material_section(section: Option<&InspectorMaterialModel>) -> MaterialSectionState {
    section.map_or_else(blank_material_section, |section| MaterialSectionState {
        visible: true,
        color_r: scalar_field(&section.base_color[0]),
        color_g: scalar_field(&section.base_color[1]),
        color_b: scalar_field(&section.base_color[2]),
        roughness: scalar_field(&section.roughness),
        metallic: scalar_field(&section.metallic),
    })
}

pub(super) fn operation_section(
    section: Option<&InspectorOperationModel>,
) -> OperationSectionState {
    section.map_or_else(blank_operation_section, |section| OperationSectionState {
        visible: true,
        label: section.op_label.clone().into(),
        smooth_k: scalar_field(&section.smooth_k),
        steps: scalar_field(&section.steps),
        color_blend: scalar_field(&section.color_blend),
    })
}

pub(super) fn sculpt_section(section: Option<&InspectorSculptModel>) -> SculptSectionState {
    section.map_or_else(blank_sculpt_section, |section| SculptSectionState {
        visible: true,
        resolution: scalar_field(&section.desired_resolution),
        layer_intensity: scalar_field(&section.layer_intensity),
        brush_radius: scalar_field(&section.brush_radius),
        brush_strength: scalar_field(&section.brush_strength),
    })
}

pub(super) fn light_section(section: Option<&InspectorLightModel>) -> LightSectionState {
    section.map_or_else(blank_light_section, |section| LightSectionState {
        visible: true,
        label: section.light_type_label.clone().into(),
        color_r: scalar_field(&section.color[0]),
        color_g: scalar_field(&section.color[1]),
        color_b: scalar_field(&section.color[2]),
        intensity: scalar_field(&section.intensity),
        range: scalar_field(&section.range),
        cast_shadows: bool_field(&section.cast_shadows),
        volumetric: bool_field(&section.volumetric),
        volumetric_density: scalar_field(&section.volumetric_density),
    })
}

fn blank_scalar() -> PropertyScalarFieldView {
    PropertyScalarFieldView {
        value: 0.0,
        display_text: "".into(),
        enabled: false,
        mixed: false,
        minimum: 0.0,
        maximum: 1.0,
        step: 0.01,
    }
}

fn blank_bool() -> PropertyBoolFieldView {
    PropertyBoolFieldView {
        value: false,
        display_text: "".into(),
        enabled: false,
        mixed: false,
    }
}

fn blank_transform_section() -> TransformSectionState {
    TransformSectionState {
        visible: false,
        can_scale: false,
        multi_editing: false,
        pos_x: blank_scalar(),
        pos_y: blank_scalar(),
        pos_z: blank_scalar(),
        rot_x: blank_scalar(),
        rot_y: blank_scalar(),
        rot_z: blank_scalar(),
        scale_x: blank_scalar(),
        scale_y: blank_scalar(),
        scale_z: blank_scalar(),
    }
}

fn blank_material_section() -> MaterialSectionState {
    MaterialSectionState {
        visible: false,
        color_r: blank_scalar(),
        color_g: blank_scalar(),
        color_b: blank_scalar(),
        roughness: blank_scalar(),
        metallic: blank_scalar(),
    }
}

fn blank_operation_section() -> OperationSectionState {
    OperationSectionState {
        visible: false,
        label: "".into(),
        smooth_k: blank_scalar(),
        steps: blank_scalar(),
        color_blend: blank_scalar(),
    }
}

fn blank_sculpt_section() -> SculptSectionState {
    SculptSectionState {
        visible: false,
        resolution: blank_scalar(),
        layer_intensity: blank_scalar(),
        brush_radius: blank_scalar(),
        brush_strength: blank_scalar(),
    }
}

fn blank_light_section() -> LightSectionState {
    LightSectionState {
        visible: false,
        label: "".into(),
        color_r: blank_scalar(),
        color_g: blank_scalar(),
        color_b: blank_scalar(),
        intensity: blank_scalar(),
        range: blank_scalar(),
        cast_shadows: blank_bool(),
        volumetric: blank_bool(),
        volumetric_density: blank_scalar(),
    }
}

fn scalar_field(field: &InspectorScalarFieldModel) -> PropertyScalarFieldView {
    PropertyScalarFieldView {
        value: field.value,
        display_text: field.display_text.clone().into(),
        enabled: field.enabled,
        mixed: field.mixed,
        minimum: field.minimum,
        maximum: field.maximum,
        step: field.step,
    }
}

fn bool_field(field: &InspectorBoolFieldModel) -> PropertyBoolFieldView {
    PropertyBoolFieldView {
        value: field.value,
        display_text: field.display_text.clone().into(),
        enabled: field.enabled,
        mixed: field.mixed,
    }
}
