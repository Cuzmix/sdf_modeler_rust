use std::path::Path;

use crate::app::reference_images::ReferenceImageStore;
use crate::app::state::{
    ExpertPanelKind, ExpertPanelRegistry, InteractionMode, PanelBarEdge, PanelBarOrientation,
    PanelFrameworkState, PanelKind, PanelSheetAnchor, PrimaryShellState, ScenePanelUiState,
    SceneSelectionState, WorkspaceRoute, WorkspaceUiState,
};
use crate::gizmo::{GizmoMode, GizmoSpace};
use crate::graph::history::History;
use crate::graph::presented_object::{
    collect_presented_base_wrapper_chain, collect_presented_selection, current_transform_owner,
    presented_children, presented_top_level_objects, resolve_presented_object, PresentedObjectKind,
    PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::sculpt::BrushMode;
use crate::sculpt::SculptState;
use crate::settings::{EnvironmentBackgroundMode, EnvironmentSource, Settings};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelRow {
    pub host_id: NodeId,
    pub object_root_id: NodeId,
    pub label: String,
    pub kind_label: String,
    pub depth: usize,
    pub has_children: bool,
    pub expanded: bool,
    pub selected: bool,
    pub hidden: bool,
    pub locked: bool,
    pub renaming: bool,
    pub rename_value: String,
    pub dragging: bool,
    pub drop_allowed: bool,
    pub drop_target: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelModel {
    pub rows: Vec<ScenePanelRow>,
    pub selection_count: usize,
    pub selected_host: Option<NodeId>,
    pub filter_query: String,
    pub drag_active: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorScalarFieldModel {
    pub value: f32,
    pub display_text: String,
    pub enabled: bool,
    pub mixed: bool,
    pub minimum: f32,
    pub maximum: f32,
    pub step: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorBoolFieldModel {
    pub value: bool,
    pub display_text: String,
    pub enabled: bool,
    pub mixed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorTransformModel {
    pub position: [InspectorScalarFieldModel; 3],
    pub rotation_deg: [InspectorScalarFieldModel; 3],
    pub scale: [InspectorScalarFieldModel; 3],
    pub can_scale: bool,
    pub multi_editing: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorMaterialModel {
    pub base_color: [InspectorScalarFieldModel; 3],
    pub roughness: InspectorScalarFieldModel,
    pub metallic: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorOperationModel {
    pub op_label: String,
    pub smooth_k: InspectorScalarFieldModel,
    pub steps: InspectorScalarFieldModel,
    pub color_blend: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorSculptModel {
    pub desired_resolution: InspectorScalarFieldModel,
    pub layer_intensity: InspectorScalarFieldModel,
    pub brush_radius: InspectorScalarFieldModel,
    pub brush_strength: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorLightModel {
    pub light_type_label: String,
    pub color: [InspectorScalarFieldModel; 3],
    pub intensity: InspectorScalarFieldModel,
    pub range: InspectorScalarFieldModel,
    pub cast_shadows: InspectorBoolFieldModel,
    pub volumetric: InspectorBoolFieldModel,
    pub volumetric_density: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorModel {
    pub title: String,
    pub name: String,
    pub kind_label: String,
    pub chips: Vec<String>,
    pub property_lines: Vec<String>,
    pub display_lines: Vec<String>,
    pub multi_selection_summary: Option<String>,
    pub transform: Option<InspectorTransformModel>,
    pub material: Option<InspectorMaterialModel>,
    pub operation: Option<InspectorOperationModel>,
    pub sculpt: Option<InspectorSculptModel>,
    pub light: Option<InspectorLightModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub label: String,
    pub direction_label: String,
    pub is_current: bool,
    pub jump_steps: usize,
    pub jump_enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceImageRow {
    pub label: String,
    pub plane_label: String,
    pub status_label: String,
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

#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettingsModel {
    pub show_grid: bool,
    pub show_node_labels: bool,
    pub show_bounding_box: bool,
    pub show_light_gizmos: bool,
    pub shadows_enabled: bool,
    pub ao_enabled: bool,
    pub export_resolution: u32,
    pub adaptive_export: bool,
    pub environment_is_hdri: bool,
    pub hdri_path_display: String,
    pub environment_rotation_degrees: f32,
    pub environment_exposure: f32,
    pub environment_bake_resolution: u32,
    pub background_is_procedural: bool,
    pub environment_background_blur: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UtilityModel {
    pub history_summary: String,
    pub reference_summary: String,
    pub history_rows: Vec<HistoryEntry>,
    pub reference_rows: Vec<ReferenceImageRow>,
    pub expert_panels: Vec<ExpertPanelEntry>,
    pub render_settings: RenderSettingsModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPaletteEntry {
    pub kind: ToolPaletteKind,
    pub label: String,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolPaletteKind {
    Select,
    Brush(BrushMode),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPaletteModel {
    pub visible: bool,
    pub select_tool: ToolPaletteEntry,
    pub brush_tools: Vec<ToolPaletteEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelLauncherItemModel {
    pub kind: PanelKind,
    pub label: String,
    pub active: bool,
    pub pinned: bool,
    pub show_drag_indicator: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelBarModel {
    pub visible: bool,
    pub edge: PanelBarEdge,
    pub orientation: PanelBarOrientation,
    pub items: Vec<PanelLauncherItemModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivePanelContentModel {
    pub kind: PanelKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelFrameModel {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelSheetModel {
    pub kind: PanelKind,
    pub title: String,
    pub pinned: bool,
    pub collapsed: bool,
    pub anchor: PanelSheetAnchor,
    pub frame: PanelFrameModel,
    pub movable: bool,
    pub content: ActivePanelContentModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelFrameworkModel {
    pub bar: PanelBarModel,
    pub transient_panel: Option<PanelSheetModel>,
    pub pinned_panels: Vec<PanelSheetModel>,
    pub panel_drag_active: bool,
    pub drag_panel_kind: PanelKind,
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
pub struct WorkspaceSummaryEntry {
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspacePanelModel {
    pub visible: bool,
    pub route: WorkspaceRoute,
    pub route_label: String,
    pub selection_summary: String,
    pub detail_text: String,
    pub context_rows: Vec<WorkspaceSummaryEntry>,
    pub input_rows: Vec<WorkspaceSummaryEntry>,
    pub output_rows: Vec<WorkspaceSummaryEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShellSnapshot {
    pub tool_palette: ToolPaletteModel,
    pub panel_framework: PanelFrameworkModel,
    pub scene_panel: ScenePanelModel,
    pub inspector: InspectorModel,
    pub utility: UtilityModel,
    pub viewport_status: ViewportStatusModel,
    pub workspace: WorkspacePanelModel,
}

pub struct ShellSnapshotInputs<'a> {
    pub scene: &'a Scene,
    pub selection: &'a SceneSelectionState,
    pub scene_panel_ui: &'a ScenePanelUiState,
    pub primary_shell: &'a PrimaryShellState,
    pub panel_framework: &'a PanelFrameworkState,
    pub viewport_size_logical: [f32; 2],
    pub workspace: &'a WorkspaceUiState,
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
        tool_palette: build_tool_palette_model(inputs.primary_shell),
        panel_framework: build_panel_framework_model(
            inputs.panel_framework,
            inputs.viewport_size_logical,
        ),
        scene_panel: build_scene_panel_model(inputs.scene, inputs.selection, inputs.scene_panel_ui),
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
        workspace: build_workspace_panel_model(
            inputs.scene,
            inputs.selection,
            inputs.workspace,
            inputs.primary_shell,
        ),
    }
}

pub fn build_tool_palette_model(primary_shell: &PrimaryShellState) -> ToolPaletteModel {
    let active_brush = match primary_shell.interaction_mode {
        InteractionMode::Sculpt(brush) => Some(brush),
        _ => None,
    };

    ToolPaletteModel {
        visible: primary_shell.tool_rail_visible,
        select_tool: ToolPaletteEntry {
            kind: ToolPaletteKind::Select,
            label: "Select".to_string(),
            active: matches!(primary_shell.interaction_mode, InteractionMode::Select),
        },
        brush_tools: BrushMode::ALL
            .into_iter()
            .map(|brush| ToolPaletteEntry {
                kind: ToolPaletteKind::Brush(brush),
                label: brush.label().to_string(),
                active: active_brush == Some(brush),
            })
            .collect(),
    }
}

pub fn build_panel_framework_model(
    panel_framework: &PanelFrameworkState,
    viewport_size_logical: [f32; 2],
) -> PanelFrameworkModel {
    let primary_bar = panel_framework
        .bar(crate::app::state::PanelBarId::PrimaryRight)
        .or_else(|| panel_framework.bars.first());
    let default_bar = crate::app::state::PanelBarState {
        id: crate::app::state::PanelBarId::PrimaryRight,
        edge: PanelBarEdge::Right,
        orientation: PanelBarOrientation::Vertical,
        items: PanelKind::ALL.to_vec(),
        active_transient: None,
        transient_rect: None,
    };
    let bar = primary_bar.unwrap_or(&default_bar);
    let transient_kind = bar.active_transient;
    let transient_frame = transient_kind.map(|_| {
        panel_frame_model(panel_framework.resolved_transient_rect(
            bar.id,
            viewport_size_logical[0],
            viewport_size_logical[1],
        ))
    });

    let items = bar
        .items
        .iter()
        .copied()
        .map(|kind| PanelLauncherItemModel {
            kind,
            label: kind.label().to_string(),
            active: transient_kind == Some(kind) || panel_framework.pinned_instance(kind).is_some(),
            pinned: panel_framework.pinned_instance(kind).is_some(),
            show_drag_indicator: transient_kind == Some(kind)
                || panel_framework.pinned_instance(kind).is_some(),
        })
        .collect::<Vec<_>>();

    let transient_panel =
        transient_kind
            .zip(transient_frame)
            .map(|(kind, frame)| PanelSheetModel {
                kind,
                title: kind.label().to_string(),
                pinned: false,
                collapsed: false,
                anchor: panel_framework.sheet_anchor_for_bar(bar.id),
                frame,
                movable: true,
                content: ActivePanelContentModel { kind },
            });

    let mut pinned_instances = panel_framework
        .pinned_instances
        .iter()
        .filter(|instance| instance.visible)
        .collect::<Vec<_>>();
    pinned_instances.sort_by_key(|instance| {
        panel_framework
            .focus_order
            .iter()
            .position(|focused_id| *focused_id == instance.id)
            .unwrap_or(usize::MAX)
    });

    let pinned_panels = pinned_instances
        .into_iter()
        .map(|instance| PanelSheetModel {
            kind: instance.kind,
            title: instance.kind.label().to_string(),
            pinned: true,
            collapsed: instance.collapsed,
            anchor: panel_framework.sheet_anchor_for_bar(instance.anchor_bar),
            frame: panel_frame_model(
                panel_framework
                    .resolved_panel_rect(
                        instance.kind,
                        instance.anchor_bar,
                        viewport_size_logical[0],
                        viewport_size_logical[1],
                    )
                    .unwrap_or_else(|| {
                        panel_framework.resolved_transient_rect(
                            instance.anchor_bar,
                            viewport_size_logical[0],
                            viewport_size_logical[1],
                        )
                    }),
            ),
            movable: true,
            content: ActivePanelContentModel {
                kind: instance.kind,
            },
        })
        .collect();

    PanelFrameworkModel {
        bar: PanelBarModel {
            visible: true,
            edge: bar.edge,
            orientation: bar.orientation,
            items,
        },
        transient_panel,
        pinned_panels,
        panel_drag_active: panel_framework.panel_drag.is_some(),
        drag_panel_kind: panel_framework
            .panel_drag
            .map(|drag| drag.kind)
            .unwrap_or(PanelKind::ObjectProperties),
    }
}

fn panel_frame_model(bounds: crate::app::ui_geometry::FloatingPanelBounds) -> PanelFrameModel {
    PanelFrameModel {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
    }
}

pub fn build_scene_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    scene_panel_ui: &ScenePanelUiState,
) -> ScenePanelModel {
    let mut rows = Vec::new();
    let selected_host = selection
        .selected
        .and_then(|selected_id| resolve_presented_object(scene, selected_id))
        .map(|object| object.host_id);
    let normalized_filter = scene_panel_ui.filter_query.trim().to_ascii_lowercase();
    for root in presented_top_level_objects(scene) {
        collect_scene_rows(
            scene,
            root,
            selection,
            scene_panel_ui,
            0,
            &normalized_filter,
            &mut rows,
        );
    }
    ScenePanelModel {
        rows,
        selection_count: selection.selected_set.len(),
        selected_host,
        filter_query: scene_panel_ui.filter_query.clone(),
        drag_active: scene_panel_ui.drag_source.is_some(),
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
            multi_selection_summary: None,
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        },
        (_, count) if count > 1 => build_multi_selection_inspector(
            scene,
            &presented_selection.ordered,
            settings,
            sculpt_state,
            interaction_mode,
            gizmo_mode,
            gizmo_space,
        ),
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

    let mut history_rows = Vec::new();
    history_rows.extend(
        undo_labels
            .iter()
            .rev()
            .enumerate()
            .map(|(index, label)| HistoryEntry {
                label: label.clone(),
                direction_label: "Undo".to_string(),
                is_current: false,
                jump_steps: index + 1,
                jump_enabled: true,
            }),
    );
    history_rows.push(HistoryEntry {
        label: "Current state".to_string(),
        direction_label: String::new(),
        is_current: true,
        jump_steps: 0,
        jump_enabled: false,
    });
    history_rows.extend(
        redo_labels
            .iter()
            .enumerate()
            .map(|(index, label)| HistoryEntry {
                label: label.clone(),
                direction_label: "Redo".to_string(),
                is_current: false,
                jump_steps: index + 1,
                jump_enabled: true,
            }),
    );

    let visible_count = reference_images
        .images
        .iter()
        .filter(|image| image.visible)
        .count();

    UtilityModel {
        history_summary: format!(
            "{} undo | {} redo",
            history.undo_count(),
            history.redo_count()
        ),
        reference_summary: format!(
            "{} images | {} visible",
            reference_images.images.len(),
            visible_count
        ),
        history_rows,
        reference_rows: reference_images
            .images
            .iter()
            .map(|image| ReferenceImageRow {
                label: Path::new(&image.path)
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| image.path.clone()),
                plane_label: image.plane.label().to_string(),
                status_label: format!(
                    "{} | {}",
                    if image.visible { "Visible" } else { "Hidden" },
                    if image.locked { "Locked" } else { "Unlocked" }
                ),
                visible: image.visible,
                locked: image.locked,
                opacity: image.opacity,
                scale: image.scale,
            })
            .collect(),
        expert_panels: ExpertPanelKind::ALL
            .into_iter()
            .map(|kind| ExpertPanelEntry {
                kind,
                label: kind.label().to_string(),
                open: expert_panels.is_open(kind),
            })
            .collect(),
        render_settings: RenderSettingsModel {
            show_grid: settings.render.show_grid,
            show_node_labels: settings.render.show_node_labels,
            show_bounding_box: settings.render.show_bounding_box,
            show_light_gizmos: settings.render.show_light_gizmos,
            shadows_enabled: settings.render.shadows_enabled,
            ao_enabled: settings.render.ao_enabled,
            export_resolution: settings.export_resolution,
            adaptive_export: settings.adaptive_export,
            environment_is_hdri: matches!(
                settings.render.environment_source,
                EnvironmentSource::Hdri
            ),
            hdri_path_display: settings
                .render
                .hdri_path
                .clone()
                .unwrap_or_else(|| "No HDRI selected".to_string()),
            environment_rotation_degrees: settings.render.environment_rotation_degrees,
            environment_exposure: settings.render.environment_exposure,
            environment_bake_resolution: settings.render.environment_bake_resolution,
            background_is_procedural: matches!(
                settings.render.environment_background_mode,
                EnvironmentBackgroundMode::Procedural
            ),
            environment_background_blur: settings.render.environment_background_blur,
        },
    }
}

pub fn build_workspace_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    workspace: &WorkspaceUiState,
    primary_shell: &PrimaryShellState,
) -> WorkspacePanelModel {
    let mut context_rows = vec![WorkspaceSummaryEntry {
        label: "Route".to_string(),
        value: workspace.route.label().to_string(),
    }];
    let mut input_rows = Vec::new();
    let mut output_rows = Vec::new();
    let detail_text = match workspace.route {
        WorkspaceRoute::NodeGraph => {
            build_node_workspace_rows(
                scene,
                selection.selected,
                &mut context_rows,
                &mut input_rows,
                &mut output_rows,
            );
            "Graph scaffold for selected object structure. Routing and context are live; node editing comes next.".to_string()
        }
        WorkspaceRoute::LightGraph => {
            build_light_workspace_rows(
                scene,
                selection.selected,
                &mut context_rows,
                &mut input_rows,
                &mut output_rows,
            );
            "Lighting workspace scaffold. Scene light context is live; graph canvas and wiring are a follow-on slice.".to_string()
        }
    };

    WorkspacePanelModel {
        visible: !primary_shell.drawer_panel.is_hidden(),
        route: workspace.route,
        route_label: workspace.route.label().to_string(),
        selection_summary: selection
            .selected
            .and_then(|selected_id| resolve_presented_object(scene, selected_id))
            .and_then(|object| scene.nodes.get(&object.host_id))
            .map(|node| format!("Selected: {}", node.name))
            .unwrap_or_else(|| "No selection".to_string()),
        detail_text,
        context_rows,
        input_rows,
        output_rows,
    }
}

struct ScenePanelBuildContext<'a> {
    selection: &'a SceneSelectionState,
    scene_panel_ui: &'a ScenePanelUiState,
    filter_query: &'a str,
}

fn collect_scene_rows(
    scene: &Scene,
    object: PresentedObjectRef,
    selection: &SceneSelectionState,
    scene_panel_ui: &ScenePanelUiState,
    depth: usize,
    filter_query: &str,
    rows: &mut Vec<ScenePanelRow>,
) -> bool {
    let wrapper_chain = collect_presented_base_wrapper_chain(scene, object);
    let context = ScenePanelBuildContext {
        selection,
        scene_panel_ui,
        filter_query,
    };
    collect_scene_rows_for_object(scene, object, &wrapper_chain, &context, depth, rows)
}

fn collect_scene_rows_for_object(
    scene: &Scene,
    object: PresentedObjectRef,
    wrapper_chain: &[NodeId],
    context: &ScenePanelBuildContext<'_>,
    depth: usize,
    rows: &mut Vec<ScenePanelRow>,
) -> bool {
    if let Some((&wrapper_id, remaining_wrappers)) = wrapper_chain.split_first() {
        let Some(node) = scene.nodes.get(&wrapper_id) else {
            return false;
        };

        let mut child_rows = Vec::new();
        let descendant_matched = collect_scene_rows_for_object(
            scene,
            object,
            remaining_wrappers,
            context,
            depth + 1,
            &mut child_rows,
        );
        let matches_self = node_matches_filter(node, node_kind_label(node), context.filter_query);
        if !context.filter_query.is_empty() && !matches_self && !descendant_matched {
            return false;
        }

        let expanded = wrapper_row_expanded(
            context.scene_panel_ui,
            wrapper_id,
            depth,
            context.filter_query,
            descendant_matched,
        );
        rows.push(ScenePanelRow {
            host_id: wrapper_id,
            object_root_id: wrapper_id,
            label: node.name.clone(),
            kind_label: node_kind_label(node),
            depth,
            has_children: true,
            expanded,
            selected: context.selection.selected == Some(wrapper_id)
                || context.selection.selected_set.contains(&wrapper_id),
            hidden: scene.is_hidden(wrapper_id),
            locked: node.locked,
            renaming: context.scene_panel_ui.renaming_node == Some(wrapper_id),
            rename_value: if context.scene_panel_ui.renaming_node == Some(wrapper_id) {
                context.scene_panel_ui.rename_buffer.clone()
            } else {
                node.name.clone()
            },
            dragging: context.scene_panel_ui.drag_source == Some(wrapper_id),
            drop_allowed: context
                .scene_panel_ui
                .drag_source
                .is_some_and(|dragged| scene.is_valid_drop_target(wrapper_id, dragged)),
            drop_target: context.scene_panel_ui.drop_target == Some(wrapper_id),
        });
        if expanded {
            rows.extend(child_rows);
        }
        return true;
    }

    let Some(node) = scene.nodes.get(&object.host_id) else {
        return false;
    };

    let children = presented_children(scene, object);
    let has_children = !children.is_empty();
    let matches_self = node_matches_filter(
        node,
        object_kind_chip(object).to_string(),
        context.filter_query,
    );
    let manual_expanded = context.scene_panel_ui.is_expanded(object.host_id, depth);

    let mut child_rows = Vec::new();
    let mut descendant_matched = false;
    for child in children {
        descendant_matched |=
            collect_scene_rows_for_object(scene, child, &[], context, depth + 1, &mut child_rows);
    }

    if !context.filter_query.is_empty() && !matches_self && !descendant_matched {
        return false;
    }

    let expanded = has_children
        && if context.filter_query.is_empty() {
            manual_expanded
        } else {
            descendant_matched || manual_expanded
        };

    rows.push(ScenePanelRow {
        host_id: object.host_id,
        object_root_id: object.object_root_id,
        label: node.name.clone(),
        kind_label: object_kind_chip(object).to_string(),
        depth,
        has_children,
        expanded,
        selected: context.selection.selected_set.contains(&object.host_id)
            || context.selection.selected == Some(object.host_id),
        hidden: scene.is_hidden(object.object_root_id),
        locked: node.locked,
        renaming: context.scene_panel_ui.renaming_node == Some(object.host_id),
        rename_value: if context.scene_panel_ui.renaming_node == Some(object.host_id) {
            context.scene_panel_ui.rename_buffer.clone()
        } else {
            node.name.clone()
        },
        dragging: context.scene_panel_ui.drag_source == Some(object.object_root_id),
        drop_allowed: context
            .scene_panel_ui
            .drag_source
            .is_some_and(|dragged| scene.is_valid_drop_target(object.object_root_id, dragged)),
        drop_target: context.scene_panel_ui.drop_target == Some(object.object_root_id),
    });

    if expanded {
        rows.extend(child_rows);
    }

    true
}

fn wrapper_row_expanded(
    scene_panel_ui: &ScenePanelUiState,
    wrapper_id: NodeId,
    depth: usize,
    filter_query: &str,
    descendant_matched: bool,
) -> bool {
    let manual_expanded = scene_panel_ui.is_expanded(wrapper_id, depth);
    if filter_query.is_empty() {
        manual_expanded
    } else {
        descendant_matched || manual_expanded
    }
}

fn node_matches_filter(
    node: &crate::graph::scene::SceneNode,
    kind_label: String,
    filter_query: &str,
) -> bool {
    filter_query.is_empty()
        || node.name.to_ascii_lowercase().contains(filter_query)
        || kind_label.to_ascii_lowercase().contains(filter_query)
}

fn node_kind_label(node: &crate::graph::scene::SceneNode) -> String {
    match &node.data {
        NodeData::Primitive { kind, .. } => kind.base_name().to_ascii_uppercase(),
        NodeData::Transform { .. } => "TRANSFORM".to_string(),
        NodeData::Modifier { kind, .. } => kind.base_name().to_ascii_uppercase(),
        NodeData::Operation { op, .. } => op.base_name().to_ascii_uppercase(),
        NodeData::Sculpt { .. } => "SCULPT".to_string(),
        NodeData::Light { light_type, .. } => light_type.label().to_ascii_uppercase(),
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
            multi_selection_summary: None,
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
            format!("Shadows: {}", on_off(*cast_shadows)),
            format!("Volumetric: {}", on_off(*volumetric)),
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

    let operation = match &node.data {
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            ..
        } => Some(InspectorOperationModel {
            op_label: op.base_name().to_string(),
            smooth_k: float_field(*smooth_k, 0.0, 4.0, 0.01, 3),
            steps: int_field(*steps, 0.0, 32.0, 1.0),
            color_blend: float_field(*color_blend, -1.0, 1.0, 0.01, 2),
        }),
        _ => None,
    };

    let sculpt = match &node.data {
        NodeData::Sculpt {
            desired_resolution,
            layer_intensity,
            ..
        } => Some(InspectorSculptModel {
            desired_resolution: int_field(*desired_resolution as f32, 8.0, 512.0, 8.0),
            layer_intensity: float_field(*layer_intensity, 0.0, 4.0, 0.01, 2),
            brush_radius: float_field(sculpt_state.selected_profile().radius, 0.05, 2.0, 0.01, 2),
            brush_strength: float_field(
                sculpt_state.selected_profile().strength,
                -2.0,
                2.0,
                0.01,
                2,
            ),
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
            color: [
                float_field(color.x, 0.0, 10.0, 0.01, 2),
                float_field(color.y, 0.0, 10.0, 0.01, 2),
                float_field(color.z, 0.0, 10.0, 0.01, 2),
            ],
            intensity: float_field(*intensity, 0.0, 20.0, 0.05, 2),
            range: float_field(*range, 0.01, 50.0, 0.05, 2),
            cast_shadows: bool_field(*cast_shadows),
            volumetric: bool_field(*volumetric),
            volumetric_density: float_field(*volumetric_density, 0.0, 1.0, 0.01, 2),
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
        multi_selection_summary: None,
        transform: build_transform_model(scene, &[object]),
        material: build_material_model(scene, &[object]),
        operation,
        sculpt,
        light,
    }
}

fn build_multi_selection_inspector(
    scene: &Scene,
    objects: &[PresentedObjectRef],
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    InspectorModel {
        title: format!("{} objects selected", objects.len()),
        name: String::new(),
        kind_label: "MULTI".to_string(),
        chips: vec!["MULTI".to_string()],
        property_lines: objects
            .iter()
            .filter_map(|object| scene.nodes.get(&object.host_id))
            .map(|node| format!("• {}", node.name))
            .collect(),
        display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
        multi_selection_summary: Some(format!("{} selected", objects.len())),
        transform: build_transform_model(scene, objects).map(|mut model| {
            model.multi_editing = true;
            model
        }),
        material: build_material_model(scene, objects),
        operation: None,
        sculpt: Some(InspectorSculptModel {
            desired_resolution: disabled_field("Single selection".to_string(), 8.0, 512.0, 8.0),
            layer_intensity: disabled_field("Single selection".to_string(), 0.0, 4.0, 0.01),
            brush_radius: float_field(sculpt_state.selected_profile().radius, 0.05, 2.0, 0.01, 2),
            brush_strength: float_field(
                sculpt_state.selected_profile().strength,
                -2.0,
                2.0,
                0.01,
                2,
            ),
        }),
        light: None,
    }
}

fn build_transform_model(
    scene: &Scene,
    objects: &[PresentedObjectRef],
) -> Option<InspectorTransformModel> {
    let targets = objects
        .iter()
        .filter_map(|object| editable_transform_target(scene, *object))
        .collect::<Vec<_>>();
    if targets.len() != objects.len() || targets.is_empty() {
        return None;
    }

    let mut positions = [Vec::new(), Vec::new(), Vec::new()];
    let mut rotations = [Vec::new(), Vec::new(), Vec::new()];
    let mut scales = [Vec::new(), Vec::new(), Vec::new()];
    let mut can_scale = true;

    for target_id in targets {
        let node = scene.nodes.get(&target_id)?;
        match &node.data {
            NodeData::Primitive {
                position,
                rotation,
                scale,
                ..
            } => {
                positions[0].push(position.x);
                positions[1].push(position.y);
                positions[2].push(position.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                scales[0].push(scale.x);
                scales[1].push(scale.y);
                scales[2].push(scale.z);
            }
            NodeData::Sculpt {
                position, rotation, ..
            } => {
                positions[0].push(position.x);
                positions[1].push(position.y);
                positions[2].push(position.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                can_scale = false;
            }
            NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } => {
                positions[0].push(translation.x);
                positions[1].push(translation.y);
                positions[2].push(translation.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                scales[0].push(scale.x);
                scales[1].push(scale.y);
                scales[2].push(scale.z);
            }
            _ => return None,
        }
    }

    Some(InspectorTransformModel {
        position: [
            float_field_from_values(&positions[0], -20.0, 20.0, 0.01, 3),
            float_field_from_values(&positions[1], -20.0, 20.0, 0.01, 3),
            float_field_from_values(&positions[2], -20.0, 20.0, 0.01, 3),
        ],
        rotation_deg: [
            float_field_from_values(&rotations[0], -180.0, 180.0, 1.0, 1),
            float_field_from_values(&rotations[1], -180.0, 180.0, 1.0, 1),
            float_field_from_values(&rotations[2], -180.0, 180.0, 1.0, 1),
        ],
        scale: [
            float_field_from_values(&scales[0], 0.01, 10.0, 0.01, 3),
            float_field_from_values(&scales[1], 0.01, 10.0, 0.01, 3),
            float_field_from_values(&scales[2], 0.01, 10.0, 0.01, 3),
        ],
        can_scale,
        multi_editing: objects.len() > 1,
    })
}

fn build_material_model(
    scene: &Scene,
    objects: &[PresentedObjectRef],
) -> Option<InspectorMaterialModel> {
    let mut red = Vec::new();
    let mut green = Vec::new();
    let mut blue = Vec::new();
    let mut roughness = Vec::new();
    let mut metallic = Vec::new();

    for object in objects {
        let node = scene.nodes.get(&object.host_id)?;
        let material = node.data.material()?;
        red.push(material.base_color.x);
        green.push(material.base_color.y);
        blue.push(material.base_color.z);
        roughness.push(material.roughness);
        metallic.push(material.metallic);
    }

    if red.is_empty() {
        return None;
    }

    Some(InspectorMaterialModel {
        base_color: [
            float_field_from_values(&red, 0.0, 1.0, 0.01, 2),
            float_field_from_values(&green, 0.0, 1.0, 0.01, 2),
            float_field_from_values(&blue, 0.0, 1.0, 0.01, 2),
        ],
        roughness: float_field_from_values(&roughness, 0.0, 1.0, 0.01, 2),
        metallic: float_field_from_values(&metallic, 0.0, 1.0, 0.01, 2),
    })
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

fn build_node_workspace_rows(
    scene: &Scene,
    selected: Option<NodeId>,
    context_rows: &mut Vec<WorkspaceSummaryEntry>,
    input_rows: &mut Vec<WorkspaceSummaryEntry>,
    output_rows: &mut Vec<WorkspaceSummaryEntry>,
) {
    let Some(selected_id) = selected else {
        context_rows.push(WorkspaceSummaryEntry {
            label: "Selection".to_string(),
            value: "Choose an object to inspect graph context".to_string(),
        });
        return;
    };
    let Some(object) = resolve_presented_object(scene, selected_id) else {
        return;
    };
    let Some(node) = scene.nodes.get(&object.object_root_id) else {
        return;
    };

    context_rows.push(WorkspaceSummaryEntry {
        label: "Selected host".to_string(),
        value: scene
            .nodes
            .get(&object.host_id)
            .map(|node| node.name.clone())
            .unwrap_or_else(|| "Missing".to_string()),
    });
    context_rows.push(WorkspaceSummaryEntry {
        label: "Object root".to_string(),
        value: node.name.clone(),
    });
    context_rows.push(WorkspaceSummaryEntry {
        label: "Kind".to_string(),
        value: object_kind_chip(object).to_string(),
    });

    match &node.data {
        NodeData::Primitive { .. } | NodeData::Light { .. } => {
            input_rows.push(WorkspaceSummaryEntry {
                label: "Inputs".to_string(),
                value: "Leaf object".to_string(),
            })
        }
        NodeData::Operation { left, right, .. } => {
            input_rows.push(WorkspaceSummaryEntry {
                label: "Left".to_string(),
                value: child_name(scene, *left),
            });
            input_rows.push(WorkspaceSummaryEntry {
                label: "Right".to_string(),
                value: child_name(scene, *right),
            });
        }
        NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. }
        | NodeData::Sculpt { input, .. } => input_rows.push(WorkspaceSummaryEntry {
            label: "Input".to_string(),
            value: child_name(scene, *input),
        }),
    }

    let parents = parent_names(scene, object.object_root_id);
    if parents.is_empty() {
        output_rows.push(WorkspaceSummaryEntry {
            label: "Outputs".to_string(),
            value: "Top-level object".to_string(),
        });
    } else {
        for (index, parent) in parents.into_iter().enumerate() {
            output_rows.push(WorkspaceSummaryEntry {
                label: format!("Parent {}", index + 1),
                value: parent,
            });
        }
    }
}

fn build_light_workspace_rows(
    scene: &Scene,
    selected: Option<NodeId>,
    context_rows: &mut Vec<WorkspaceSummaryEntry>,
    input_rows: &mut Vec<WorkspaceSummaryEntry>,
    output_rows: &mut Vec<WorkspaceSummaryEntry>,
) {
    context_rows.push(WorkspaceSummaryEntry {
        label: "Scene lights".to_string(),
        value: scene
            .nodes
            .values()
            .filter(|node| matches!(node.data, NodeData::Light { .. }))
            .count()
            .to_string(),
    });

    match selected.and_then(|selected_id| resolve_presented_object(scene, selected_id)) {
        Some(object) if matches!(object.kind, PresentedObjectKind::Light) => {
            context_rows.push(WorkspaceSummaryEntry {
                label: "Selected light".to_string(),
                value: scene
                    .nodes
                    .get(&object.host_id)
                    .map(|node| node.name.clone())
                    .unwrap_or_else(|| "Missing".to_string()),
            });
        }
        _ => {
            context_rows.push(WorkspaceSummaryEntry {
                label: "Selection".to_string(),
                value: "Select a light to inspect routing context".to_string(),
            });
        }
    }

    input_rows.push(WorkspaceSummaryEntry {
        label: "Light source".to_string(),
        value: "Direct scene light".to_string(),
    });
    output_rows.push(WorkspaceSummaryEntry {
        label: "Affected geometry".to_string(),
        value: scene
            .nodes
            .values()
            .filter(|node| node.data.material().is_some())
            .count()
            .to_string(),
    });
}

fn parent_names(scene: &Scene, child_id: NodeId) -> Vec<String> {
    scene
        .nodes
        .values()
        .filter(|node| node.data.children().any(|candidate| candidate == child_id))
        .map(|node| node.name.clone())
        .collect()
}

fn child_name(scene: &Scene, child: Option<NodeId>) -> String {
    child
        .and_then(|child_id| scene.nodes.get(&child_id))
        .map(|node| node.name.clone())
        .unwrap_or_else(|| "(empty)".to_string())
}

fn float_field(
    value: f32,
    minimum: f32,
    maximum: f32,
    step: f32,
    precision: usize,
) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value,
        display_text: format!("{value:.precision$}"),
        enabled: true,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn int_field(value: f32, minimum: f32, maximum: f32, step: f32) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value,
        display_text: format!("{value:.0}"),
        enabled: true,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn disabled_field(
    display_text: String,
    minimum: f32,
    maximum: f32,
    step: f32,
) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value: minimum,
        display_text,
        enabled: false,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn bool_field(value: bool) -> InspectorBoolFieldModel {
    InspectorBoolFieldModel {
        value,
        display_text: if value { "On" } else { "Off" }.to_string(),
        enabled: true,
        mixed: false,
    }
}

fn float_field_from_values(
    values: &[f32],
    minimum: f32,
    maximum: f32,
    step: f32,
    precision: usize,
) -> InspectorScalarFieldModel {
    let average = if values.is_empty() {
        minimum
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    };
    let mixed = values
        .first()
        .is_some_and(|first| values.iter().any(|value| (*value - *first).abs() > 0.0001));
    InspectorScalarFieldModel {
        value: average,
        display_text: if mixed {
            "Mixed".to_string()
        } else {
            format!("{average:.precision$}")
        },
        enabled: true,
        mixed,
        minimum,
        maximum,
        step,
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
    format!("{label}: {}", child_name(scene, child))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::{PanelBarId, PanelFrameworkState, PanelKind};
    use crate::graph::scene::SdfPrimitive;

    #[test]
    fn scene_panel_marks_selected_row() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);

        let model = build_scene_panel_model(&scene, &selection, &ScenePanelUiState::default());

        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.selected));
    }

    #[test]
    fn scene_panel_filter_expands_matching_ancestor_rows() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let transform = scene.create_transform(Some(sphere));
        let mut ui = ScenePanelUiState::default();
        ui.filter_query = "sphere".to_string();

        let model = build_scene_panel_model(&scene, &SceneSelectionState::default(), &ui);

        let parent_row = model
            .rows
            .iter()
            .find(|row| row.object_root_id == transform)
            .expect("transform row");
        assert!(parent_row.expanded);
        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.depth == parent_row.depth + 1));
    }

    #[test]
    fn utility_maps_environment_settings() {
        let mut settings = Settings::default();
        settings.render.environment_source = EnvironmentSource::Hdri;
        settings.render.hdri_path = Some("studio.hdr".to_string());
        settings.render.environment_background_mode = EnvironmentBackgroundMode::Procedural;

        let model = build_utility_model(
            &History::new(),
            &ReferenceImageStore::default(),
            &ExpertPanelRegistry::default(),
            &settings,
        );

        assert!(model.render_settings.environment_is_hdri);
        assert_eq!(model.render_settings.hdri_path_display, "studio.hdr");
        assert!(model.render_settings.background_is_procedural);
    }

    #[test]
    fn tool_palette_marks_select_mode_active() {
        let model = build_tool_palette_model(&PrimaryShellState::default());

        assert!(model.select_tool.active);
        assert!(model.brush_tools.iter().all(|tool| !tool.active));
    }

    #[test]
    fn tool_palette_marks_active_brush() {
        let mut shell = PrimaryShellState::default();
        shell.interaction_mode = InteractionMode::Sculpt(BrushMode::Flatten);

        let model = build_tool_palette_model(&shell);

        assert!(!model.select_tool.active);
        let active_labels = model
            .brush_tools
            .iter()
            .filter(|tool| tool.active)
            .map(|tool| tool.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(active_labels, vec!["Flatten"]);
    }

    #[test]
    fn panel_framework_model_marks_open_transient_panel_active() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, [1200.0, 800.0]);

        assert_eq!(
            model.transient_panel.as_ref().map(|panel| panel.kind),
            Some(PanelKind::ObjectProperties)
        );
        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::ObjectProperties && item.active));
    }

    #[test]
    fn panel_framework_model_marks_pinned_panel_without_duplication() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        state.toggle_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, [1200.0, 800.0]);

        assert!(model.transient_panel.is_none());
        assert_eq!(model.pinned_panels.len(), 1);
        assert_eq!(model.pinned_panels[0].kind, PanelKind::RenderSettings);
        assert!(model.pinned_panels[0].movable);
        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::RenderSettings && item.pinned && item.active));
        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::RenderSettings && item.show_drag_indicator));
    }

    #[test]
    fn panel_framework_model_resolves_default_transient_frame() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, [1200.0, 800.0]);
        let panel = model.transient_panel.expect("transient panel");

        assert!((panel.frame.width - 390.0).abs() < 0.01);
        assert!((panel.frame.x - 620.0).abs() < 0.01);
        assert!((panel.frame.y - 20.0).abs() < 0.01);
        assert!(panel.movable);
    }

    #[test]
    fn panel_framework_model_clamps_remembered_transient_frame() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state
            .bar_mut(PanelBarId::PrimaryRight)
            .expect("primary bar")
            .transient_rect = Some(crate::app::ui_geometry::FloatingPanelBounds::from_min_size(
            900.0, 700.0, 390.0, 420.0,
        ));

        let model = build_panel_framework_model(&state, [1000.0, 600.0]);
        let panel = model.transient_panel.expect("transient panel");

        assert!((panel.frame.x - 590.0).abs() < 0.01);
        assert!((panel.frame.y - 160.0).abs() < 0.01);
        assert!((panel.frame.height - 420.0).abs() < 0.01);
    }

    #[test]
    fn panel_framework_model_marks_only_active_transient_launcher_with_drag_hint() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, [1200.0, 800.0]);

        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::RenderSettings && item.show_drag_indicator));
        assert!(model
            .bar
            .items
            .iter()
            .all(|item| item.kind == PanelKind::RenderSettings || !item.show_drag_indicator));
    }

    #[test]
    fn inspector_model_keeps_no_selection_state_available() {
        let model = build_inspector_model(
            &Scene::new(),
            &SceneSelectionState::default(),
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        assert_eq!(model.title, "No selection");
        assert!(model
            .property_lines
            .iter()
            .any(|line| line.contains("Click an object")));
    }
}
