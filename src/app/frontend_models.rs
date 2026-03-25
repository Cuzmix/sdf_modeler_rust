use crate::app::reference_images::ReferenceImageStore;
use crate::app::state::{
    ExpertPanelKind, ExpertPanelRegistry, InteractionMode, SceneSelectionState,
};
use crate::graph::history::History;
use crate::graph::presented_object::{
    collect_presented_selection, presented_children, presented_top_level_objects,
    PresentedObjectKind, PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::Settings;
use crate::ui::gizmo::{GizmoMode, GizmoSpace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelRow {
    pub host_id: NodeId,
    pub label: String,
    pub depth: usize,
    pub selected: bool,
    pub hidden: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelModel {
    pub rows: Vec<ScenePanelRow>,
    pub selection_count: usize,
    pub selected_host: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InspectorModel {
    pub title: String,
    pub chips: Vec<String>,
    pub property_lines: Vec<String>,
    pub display_lines: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPanelEntry {
    pub kind: ExpertPanelKind,
    pub label: String,
    pub open: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UtilityModel {
    pub history_lines: Vec<String>,
    pub reference_lines: Vec<String>,
    pub expert_panels: Vec<ExpertPanelEntry>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShellSnapshot {
    pub scene_panel: ScenePanelModel,
    pub inspector: InspectorModel,
    pub utility: UtilityModel,
    pub viewport_status: ViewportStatusModel,
}

pub struct ShellSnapshotInputs<'a> {
    pub scene: &'a Scene,
    pub selection: &'a SceneSelectionState,
    pub history: &'a History,
    pub reference_images: &'a ReferenceImageStore,
    pub expert_panels: &'a ExpertPanelRegistry,
    pub settings: &'a Settings,
    pub interaction_mode: InteractionMode,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
}

pub fn build_shell_snapshot(inputs: ShellSnapshotInputs<'_>) -> ShellSnapshot {
    ShellSnapshot {
        scene_panel: build_scene_panel_model(inputs.scene, inputs.selection),
        inspector: build_inspector_model(
            inputs.scene,
            inputs.selection,
            inputs.settings,
            inputs.interaction_mode,
            &inputs.gizmo_mode,
            &inputs.gizmo_space,
        ),
        utility: build_utility_model(
            inputs.history,
            inputs.reference_images,
            inputs.expert_panels,
        ),
        viewport_status: ViewportStatusModel {
            interaction_label: interaction_mode_label(inputs.interaction_mode).to_string(),
            transform_label: inputs.gizmo_mode.label().to_string(),
            space_label: inputs.gizmo_space.label().to_string(),
        },
    }
}

pub fn build_scene_panel_model(scene: &Scene, selection: &SceneSelectionState) -> ScenePanelModel {
    let mut rows = Vec::new();
    let roots = presented_top_level_objects(scene);
    let selected_host = selection
        .selected
        .and_then(|selected_id| {
            crate::graph::presented_object::resolve_presented_object(scene, selected_id)
        })
        .map(|object| object.host_id);
    for root in roots {
        collect_scene_rows(scene, root, selection, 0, &mut rows);
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
            chips: Vec::new(),
            property_lines: vec!["Click an object in the scene list or viewport.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
        },
        (_, count) if count > 1 => InspectorModel {
            title: format!("{count} objects selected"),
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
        },
        (Some(object), _) => build_single_object_inspector(
            scene,
            object,
            settings,
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
) -> UtilityModel {
    let mut history_lines = Vec::new();
    let undo_labels = history.undo_labels();
    let redo_labels = history.redo_labels();
    history_lines.push(format!("Undo depth: {}", history.undo_count()));
    if let Some(last_undo) = undo_labels.last() {
        history_lines.push(format!("Latest undo: {last_undo}"));
    }
    history_lines.push(format!("Redo depth: {}", history.redo_count()));
    if let Some(next_redo) = redo_labels.first() {
        history_lines.push(format!("Next redo: {next_redo}"));
    }

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
        let name = std::path::Path::new(&image.path)
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
        expert_panels,
    }
}

fn collect_scene_rows(
    scene: &Scene,
    object: PresentedObjectRef,
    selection: &SceneSelectionState,
    depth: usize,
    rows: &mut Vec<ScenePanelRow>,
) {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return;
    };
    rows.push(ScenePanelRow {
        host_id: object.host_id,
        label: format!("{} {}", object_prefix(object), node.name),
        depth,
        selected: selection.selected_set.contains(&object.host_id)
            || selection.selected == Some(object.host_id),
        hidden: scene.is_hidden(object.object_root_id),
    });
    for child in presented_children(scene, object) {
        collect_scene_rows(scene, child, selection, depth + 1, rows);
    }
}

fn build_single_object_inspector(
    scene: &Scene,
    object: PresentedObjectRef,
    settings: &Settings,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return InspectorModel {
            title: "Missing object".to_string(),
            chips: Vec::new(),
            property_lines: vec!["The selected object is no longer present.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
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

    InspectorModel {
        title: node.name.clone(),
        chips,
        property_lines,
        display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
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
            crate::graph::presented_object::resolve_presented_object(scene, child_id)
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

        let model = build_scene_panel_model(&scene, &selection);

        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.selected));
    }

    #[test]
    fn inspector_reports_missing_selection_prompt() {
        let scene = Scene::new();
        let selection = SceneSelectionState::default();
        let settings = Settings::default();

        let model = build_inspector_model(
            &scene,
            &selection,
            &settings,
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        assert_eq!(model.title, "No selection");
        assert!(model.property_lines[0].contains("Click an object"));
    }
}
