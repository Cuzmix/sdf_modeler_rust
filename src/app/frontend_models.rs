use std::path::Path;

use crate::app::reference_images::ReferenceImageStore;
use crate::app::state::{
    ExpertPanelKind, ExpertPanelRegistry, InteractionMode, SceneSelectionState,
};
use crate::gizmo::{GizmoMode, GizmoSpace};
use crate::graph::history::History;
use crate::graph::presented_object::{
    collect_presented_selection, current_transform_owner, presented_children,
    presented_top_level_objects, resolve_presented_object, PresentedObjectKind, PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::sculpt::SculptState;
use crate::settings::Settings;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelRow {
    pub host_id: NodeId,
    pub object_root_id: NodeId,
    pub label: String,
    pub depth: usize,
    pub selected: bool,
    pub hidden: bool,
    pub locked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelModel {
    pub rows: Vec<ScenePanelRow>,
    pub selection_count: usize,
    pub selected_host: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorTransformModel {
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
    pub can_scale: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorMaterialModel {
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorOperationModel {
    pub op_label: String,
    pub smooth_k: f32,
    pub steps: f32,
    pub color_blend: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorSculptModel {
    pub desired_resolution: u32,
    pub layer_intensity: f32,
    pub brush_radius: f32,
    pub brush_strength: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorLightModel {
    pub light_type_label: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub cast_shadows: bool,
    pub volumetric: bool,
    pub volumetric_density: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorModel {
    pub title: String,
    pub name: String,
    pub kind_label: String,
    pub chips: Vec<String>,
    pub property_lines: Vec<String>,
    pub display_lines: Vec<String>,
    pub transform: Option<InspectorTransformModel>,
    pub material: Option<InspectorMaterialModel>,
    pub operation: Option<InspectorOperationModel>,
    pub sculpt: Option<InspectorSculptModel>,
    pub light: Option<InspectorLightModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub label: String,
    pub is_undo: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceImageRow {
    pub label: String,
    pub plane_label: String,
    pub visible: bool,
    pub locked: bool,
    pub opacity: f32,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPanelEntry {
    pub kind: ExpertPanelKind,
    pub label: String,
    pub open: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderSettingsModel {
    pub show_grid: bool,
    pub show_node_labels: bool,
    pub show_bounding_box: bool,
    pub show_light_gizmos: bool,
    pub shadows_enabled: bool,
    pub ao_enabled: bool,
    pub export_resolution: u32,
    pub adaptive_export: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UtilityModel {
    pub history_lines: Vec<String>,
    pub reference_lines: Vec<String>,
    pub history_rows: Vec<HistoryEntry>,
    pub reference_rows: Vec<ReferenceImageRow>,
    pub expert_panels: Vec<ExpertPanelEntry>,
    pub render_settings: RenderSettingsModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportStatusModel {
    pub interaction_label: String,
    pub transform_label: String,
    pub space_label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportFrameImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShellSnapshot {
    pub scene_panel: ScenePanelModel,
    pub inspector: InspectorModel,
    pub utility: UtilityModel,
    pub viewport_status: ViewportStatusModel,
}

pub struct ShellSnapshotInputs<'a> {
    pub scene: &'a Scene,
    pub selection: &'a SceneSelectionState,
    pub scene_filter_query: &'a str,
    pub history: &'a History,
    pub reference_images: &'a ReferenceImageStore,
    pub expert_panels: &'a ExpertPanelRegistry,
    pub settings: &'a Settings,
    pub sculpt_state: &'a SculptState,
    pub interaction_mode: InteractionMode,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
}

pub fn build_shell_snapshot(inputs: ShellSnapshotInputs<'_>) -> ShellSnapshot {
    ShellSnapshot {
        scene_panel: build_scene_panel_model(
            inputs.scene,
            inputs.selection,
            inputs.scene_filter_query,
        ),
        inspector: build_inspector_model(
            inputs.scene,
            inputs.selection,
            inputs.settings,
            inputs.sculpt_state,
            inputs.interaction_mode,
            &inputs.gizmo_mode,
            &inputs.gizmo_space,
        ),
        utility: build_utility_model(
            inputs.history,
            inputs.reference_images,
            inputs.expert_panels,
            inputs.settings,
        ),
        viewport_status: ViewportStatusModel {
            interaction_label: interaction_mode_label(inputs.interaction_mode).to_string(),
            transform_label: inputs.gizmo_mode.label().to_string(),
            space_label: inputs.gizmo_space.label().to_string(),
        },
    }
}

pub fn build_scene_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    filter_query: &str,
) -> ScenePanelModel {
    let mut rows = Vec::new();
    let roots = presented_top_level_objects(scene);
    let selected_host = selection
        .selected
        .and_then(|selected_id| resolve_presented_object(scene, selected_id))
        .map(|object| object.host_id);
    let normalized_filter = filter_query.trim().to_ascii_lowercase();
    for root in roots {
        collect_scene_rows(scene, root, selection, 0, &normalized_filter, &mut rows);
    }
    ScenePanelModel {
        rows,
        selection_count: selection.selected_set.len(),
        selected_host,
    }
}

pub fn build_inspector_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    let presented_selection =
        collect_presented_selection(scene, selection.selected, &selection.selected_set);
    match (
        presented_selection.primary,
        presented_selection.ordered.len(),
    ) {
        (None, _) => InspectorModel {
            title: "No selection".to_string(),
            name: String::new(),
            kind_label: "NONE".to_string(),
            chips: Vec::new(),
            property_lines: vec!["Click an object in the scene list or viewport.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        },
        (_, count) if count > 1 => InspectorModel {
            title: format!("{count} objects selected"),
            name: String::new(),
            kind_label: "MULTI".to_string(),
            chips: vec!["MULTI".to_string()],
            property_lines: presented_selection
                .ordered
                .into_iter()
                .filter_map(|object| {
                    scene
                        .nodes
                        .get(&object.host_id)
                        .map(|node| format!("• {}", node.name))
                })
                .collect(),
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        },
        (Some(object), _) => build_single_object_inspector(
            scene,
            object,
            settings,
            sculpt_state,
            interaction_mode,
            gizmo_mode,
            gizmo_space,
        ),
    }
}

pub fn build_utility_model(
    history: &History,
    reference_images: &ReferenceImageStore,
    expert_panels: &ExpertPanelRegistry,
    settings: &Settings,
) -> UtilityModel {
    let undo_labels = history.undo_labels();
    let redo_labels = history.redo_labels();

    let mut history_lines = vec![
        format!("Undo depth: {}", history.undo_count()),
        format!("Redo depth: {}", history.redo_count()),
    ];
    if let Some(last_undo) = undo_labels.last() {
        history_lines.insert(1, format!("Latest undo: {last_undo}"));
    }
    if let Some(next_redo) = redo_labels.first() {
        history_lines.push(format!("Next redo: {next_redo}"));
    }

    let mut history_rows = Vec::new();
    history_rows.extend(undo_labels.iter().rev().map(|label| HistoryEntry {
        label: label.clone(),
        is_undo: true,
    }));
    history_rows.extend(redo_labels.iter().map(|label| HistoryEntry {
        label: label.clone(),
        is_undo: false,
    }));

    let visible_count = reference_images
        .images
        .iter()
        .filter(|image| image.visible)
        .count();
    let mut reference_lines = vec![
        format!("Images: {}", reference_images.images.len()),
        format!("Visible: {visible_count}"),
    ];
    reference_lines.extend(reference_images.images.iter().take(3).map(|image| {
        let name = Path::new(&image.path)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| image.path.clone());
        format!(
            "{} • {} • {}",
            name,
            image.plane.label(),
            if image.visible { "Visible" } else { "Hidden" }
        )
    }));

    let reference_rows = reference_images
        .images
        .iter()
        .map(|image| ReferenceImageRow {
            label: Path::new(&image.path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| image.path.clone()),
            plane_label: image.plane.label().to_string(),
            visible: image.visible,
            locked: image.locked,
            opacity: image.opacity,
            scale: image.scale,
        })
        .collect();

    let expert_panels = ExpertPanelKind::ALL
        .into_iter()
        .map(|kind| ExpertPanelEntry {
            kind,
            label: kind.label().to_string(),
            open: expert_panels.is_open(kind),
        })
        .collect();

    UtilityModel {
        history_lines,
        reference_lines,
        history_rows,
        reference_rows,
        expert_panels,
        render_settings: RenderSettingsModel {
            show_grid: settings.render.show_grid,
            show_node_labels: settings.render.show_node_labels,
            show_bounding_box: settings.render.show_bounding_box,
            show_light_gizmos: settings.render.show_light_gizmos,
            shadows_enabled: settings.render.shadows_enabled,
            ao_enabled: settings.render.ao_enabled,
            export_resolution: settings.export_resolution,
            adaptive_export: settings.adaptive_export,
        },
    }
}

fn collect_scene_rows(
    scene: &Scene,
    object: PresentedObjectRef,
    selection: &SceneSelectionState,
    depth: usize,
    filter_query: &str,
    rows: &mut Vec<ScenePanelRow>,
) -> bool {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return false;
    };
    let row = ScenePanelRow {
        host_id: object.host_id,
        object_root_id: object.object_root_id,
        label: format!("{} {}", object_prefix(object), node.name),
        depth,
        selected: selection.selected_set.contains(&object.host_id)
            || selection.selected == Some(object.host_id),
        hidden: scene.is_hidden(object.object_root_id),
        locked: node.locked,
    };

    let matches_self = filter_query.is_empty()
        || row.label.to_ascii_lowercase().contains(filter_query)
        || node.name.to_ascii_lowercase().contains(filter_query);
    let insert_index = rows.len();
    let mut descendant_matched = false;
    for child in presented_children(scene, object) {
        descendant_matched |=
            collect_scene_rows(scene, child, selection, depth + 1, filter_query, rows);
    }
    if matches_self || descendant_matched {
        rows.insert(insert_index, row);
        true
    } else {
        false
    }
}

fn build_single_object_inspector(
    scene: &Scene,
    object: PresentedObjectRef,
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return InspectorModel {
            title: "Missing object".to_string(),
            name: String::new(),
            kind_label: "MISSING".to_string(),
            chips: Vec::new(),
            property_lines: vec!["The selected object is no longer present.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        };
    };

    let mut chips = vec![object_kind_chip(object).to_string()];
    if object.attached_sculpt_id.is_some() {
        chips.push("SCULPT".to_string());
    }
    if matches!(object.kind, PresentedObjectKind::Voxel) {
        chips.push("VOXEL".to_string());
    }
    if node.locked {
        chips.push("LOCKED".to_string());
    }

    let transform = build_transform_model(scene, object);
    let material = node.data.material().map(|material| InspectorMaterialModel {
        base_color: [
            material.base_color.x,
            material.base_color.y,
            material.base_color.z,
        ],
        roughness: material.roughness,
        metallic: material.metallic,
    });

    let mut property_lines = match &node.data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            material,
            ..
        } => vec![
            format!("Primitive: {}", kind.base_name()),
            format_vec3("Position", *position),
            format_vec3_degrees("Rotation", *rotation),
            format_vec3("Scale", *scale),
            format_material(material),
        ],
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            left,
            right,
        } => vec![
            format!("Operation: {}", op.base_name()),
            format!("Smooth K: {:.3}", smooth_k),
            format!("Steps: {:.0}", steps),
            format!("Color blend: {:.3}", color_blend),
            format_child(scene, "Left", *left),
            format_child(scene, "Right", *right),
        ],
        NodeData::Sculpt {
            input,
            position,
            rotation,
            material,
            layer_intensity,
            voxel_grid,
            desired_resolution,
        } => vec![
            if input.is_some() {
                "Sculpt layer: Attached".to_string()
            } else {
                "Sculpt object: Standalone".to_string()
            },
            format_vec3("Position", *position),
            format_vec3_degrees("Rotation", *rotation),
            format!("Resolution: {}^3", desired_resolution),
            format!("Detail size: {:.4}", voxel_grid.voxel_pitch()),
            format!("Layer intensity: {:.2}", layer_intensity),
            format_material(material),
        ],
        NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
            ..
        } => vec![
            format!("Light: {}", light_type.label()),
            format!("Color: {:.2}, {:.2}, {:.2}", color.x, color.y, color.z),
            format!("Intensity: {:.2}", intensity),
            format!("Range: {:.2}", range),
            format!("Shadows: {}", if *cast_shadows { "On" } else { "Off" }),
            format!("Volumetric: {}", if *volumetric { "On" } else { "Off" }),
            format!("Volumetric density: {:.2}", volumetric_density),
        ],
        NodeData::Transform { .. } | NodeData::Modifier { .. } => {
            vec!["This object is represented through its host wrapper stack.".to_string()]
        }
    };

    if object.attached_sculpt_id.is_some()
        && !matches!(node.data, NodeData::Sculpt { input: None, .. })
    {
        property_lines.push("Attached sculpt layer is active on this object.".to_string());
    }

    let sculpt = match &node.data {
        NodeData::Sculpt {
            desired_resolution,
            layer_intensity,
            ..
        } => Some(InspectorSculptModel {
            desired_resolution: *desired_resolution,
            layer_intensity: *layer_intensity,
            brush_radius: sculpt_state.selected_profile().radius,
            brush_strength: sculpt_state.selected_profile().strength,
        }),
        _ => None,
    };

    let operation = match &node.data {
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            ..
        } => Some(InspectorOperationModel {
            op_label: op.base_name().to_string(),
            smooth_k: *smooth_k,
            steps: *steps,
            color_blend: *color_blend,
        }),
        _ => None,
    };

    let light = match &node.data {
        NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
            ..
        } => Some(InspectorLightModel {
            light_type_label: light_type.label().to_string(),
            color: [color.x, color.y, color.z],
            intensity: *intensity,
            range: *range,
            cast_shadows: *cast_shadows,
            volumetric: *volumetric,
            volumetric_density: *volumetric_density,
        }),
        _ => None,
    };

    InspectorModel {
        title: node.name.clone(),
        name: node.name.clone(),
        kind_label: object_kind_chip(object).to_string(),
        chips,
        property_lines,
        display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
        transform,
        material,
        operation,
        sculpt,
        light,
    }
}

fn build_transform_model(
    scene: &Scene,
    object: PresentedObjectRef,
) -> Option<InspectorTransformModel> {
    let target_id = editable_transform_target(scene, object)?;
    let node = scene.nodes.get(&target_id)?;
    match &node.data {
        NodeData::Primitive {
            position,
            rotation,
            scale,
            ..
        } => Some(InspectorTransformModel {
            position: [position.x, position.y, position.z],
            rotation_deg: [
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            ],
            scale: [scale.x, scale.y, scale.z],
            can_scale: true,
        }),
        NodeData::Sculpt {
            position, rotation, ..
        } => Some(InspectorTransformModel {
            position: [position.x, position.y, position.z],
            rotation_deg: [
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            ],
            scale: [1.0, 1.0, 1.0],
            can_scale: false,
        }),
        NodeData::Transform {
            translation,
            rotation,
            scale,
            ..
        } => Some(InspectorTransformModel {
            position: [translation.x, translation.y, translation.z],
            rotation_deg: [
                rotation.x.to_degrees(),
                rotation.y.to_degrees(),
                rotation.z.to_degrees(),
            ],
            scale: [scale.x, scale.y, scale.z],
            can_scale: true,
        }),
        _ => None,
    }
}

fn editable_transform_target(scene: &Scene, object: PresentedObjectRef) -> Option<NodeId> {
    match object.kind {
        PresentedObjectKind::Parametric => current_transform_owner(scene, object.host_id),
        PresentedObjectKind::Voxel => Some(object.host_id),
        PresentedObjectKind::Light => match scene
            .nodes
            .get(&object.object_root_id)
            .map(|node| &node.data)
        {
            Some(NodeData::Transform { .. }) => Some(object.object_root_id),
            _ => Some(object.host_id),
        },
    }
}

fn display_lines(
    settings: &Settings,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> Vec<String> {
    vec![
        format!("Grid: {}", on_off(settings.render.show_grid)),
        format!("Node labels: {}", on_off(settings.render.show_node_labels)),
        format!(
            "Bounding box: {}",
            on_off(settings.render.show_bounding_box)
        ),
        format!(
            "Light gizmos: {}",
            on_off(settings.render.show_light_gizmos)
        ),
        format!("Shadows: {}", on_off(settings.render.shadows_enabled)),
        format!("Ambient occlusion: {}", on_off(settings.render.ao_enabled)),
        format!("Interaction: {}", interaction_mode_label(interaction_mode)),
        format!("Transform: {}", gizmo_mode.label()),
        format!("Space: {}", gizmo_space.label()),
    ]
}

fn object_prefix(object: PresentedObjectRef) -> &'static str {
    match object.kind {
        PresentedObjectKind::Parametric => "[Obj]",
        PresentedObjectKind::Voxel => "[Vox]",
        PresentedObjectKind::Light => "[Lgt]",
    }
}

fn object_kind_chip(object: PresentedObjectRef) -> &'static str {
    match object.kind {
        PresentedObjectKind::Parametric => "OBJECT",
        PresentedObjectKind::Voxel => "VOXEL",
        PresentedObjectKind::Light => "LIGHT",
    }
}

fn interaction_mode_label(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Select => "Select",
        InteractionMode::Measure => "Measure",
        InteractionMode::Sculpt(brush) => brush.label(),
    }
}

fn on_off(value: bool) -> &'static str {
    if value {
        "On"
    } else {
        "Off"
    }
}

fn format_vec3(label: &str, value: glam::Vec3) -> String {
    format!("{label}: {:.3}, {:.3}, {:.3}", value.x, value.y, value.z)
}

fn format_vec3_degrees(label: &str, value: glam::Vec3) -> String {
    format!(
        "{label}: {:.1}, {:.1}, {:.1} deg",
        value.x.to_degrees(),
        value.y.to_degrees(),
        value.z.to_degrees()
    )
}

fn format_material(material: &crate::graph::scene::MaterialParams) -> String {
    format!(
        "Material: color {:.2}/{:.2}/{:.2}, roughness {:.2}, metallic {:.2}",
        material.base_color.x,
        material.base_color.y,
        material.base_color.z,
        material.roughness,
        material.metallic
    )
}

fn format_child(scene: &Scene, label: &str, child: Option<NodeId>) -> String {
    let child_label = child
        .and_then(|child_id| {
            resolve_presented_object(scene, child_id)
                .and_then(|object| scene.nodes.get(&object.host_id))
                .map(|node| node.name.clone())
        })
        .unwrap_or_else(|| "(empty)".to_string());
    format!("{label}: {child_label}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::SdfPrimitive;

    #[test]
    fn scene_panel_marks_selected_row() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);

        let model = build_scene_panel_model(&scene, &selection, "");

        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.selected));
    }

    #[test]
    fn scene_panel_filter_keeps_matching_rows() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let box_id = scene.create_primitive(SdfPrimitive::Box);

        let selection = SceneSelectionState::default();
        let model = build_scene_panel_model(&scene, &selection, "sphere");

        assert!(model.rows.iter().any(|row| row.host_id == sphere));
        assert!(!model.rows.iter().any(|row| row.host_id == box_id));
    }

    #[test]
    fn inspector_reports_missing_selection_prompt() {
        let scene = Scene::new();
        let selection = SceneSelectionState::default();
        let settings = Settings::default();
        let sculpt_state = SculptState::new_inactive();

        let model = build_inspector_model(
            &scene,
            &selection,
            &settings,
            &sculpt_state,
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        assert_eq!(model.title, "No selection");
        assert!(model.property_lines[0].contains("Click an object"));
    }
}
