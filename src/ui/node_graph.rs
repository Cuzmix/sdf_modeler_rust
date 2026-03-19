use std::collections::{BTreeMap, HashMap, HashSet};

use eframe::egui::{self, Color32};
use egui_node_graph2::*;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{
    CsgOp, LightType, ModifierKind, NodeData, NodeId as SceneNodeId, Scene, SdfPrimitive,
};

// ---------------------------------------------------------------------------
// Badge / port colors (matching old theme)
// ---------------------------------------------------------------------------

const COLOR_PRIM_BADGE: Color32 = Color32::from_rgb(70, 130, 180);
const COLOR_OP_BADGE: Color32 = Color32::from_rgb(200, 120, 50);
const COLOR_SCULPT_BADGE: Color32 = Color32::from_rgb(150, 100, 200);
const COLOR_TRANSFORM_BADGE: Color32 = Color32::from_rgb(100, 180, 170);
const COLOR_MODIFIER_BADGE: Color32 = Color32::from_rgb(180, 160, 80);
const COLOR_LIGHT_BADGE: Color32 = Color32::from_rgb(220, 200, 80);
const COLOR_PORT: Color32 = Color32::from_rgb(100, 200, 100);

const COL_SPACING: f32 = 260.0;
const ROW_SPACING: f32 = 190.0;
const NODE_WIDTH: f32 = 140.0;
const DEFAULT_NODE_HEIGHT: f32 = 180.0;
const GRID_STEP: f32 = 20.0;
const NODE_PADDING: f32 = 16.0;
const SPAWN_X_OFFSET: f32 = 180.0;

// ---------------------------------------------------------------------------
// Custom types for the graph library
// ---------------------------------------------------------------------------

/// Single data type - all connections carry SDF signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfDataType {
    Sdf,
}

/// We don't use per-port inline value widgets; all editing is in bottom_ui.
#[derive(Clone, Debug, Default)]
pub enum SdfValueType {
    #[default]
    None,
}

/// Custom response type for user events from bottom_ui.
#[derive(Clone, Debug)]
pub enum SdfResponse {
    /// Placeholder - inline editing is tracked via dirty_nodes in user_state.
    _Unused,
}
impl UserResponseTrait for SdfResponse {}

/// Per-node user data stored inside the library's Graph.
#[derive(Clone, Debug)]
pub struct SdfNodeData {
    /// Maps back to the Scene's NodeId.
    pub scene_node_id: SceneNodeId,
}

/// Node categories for the node finder.
#[derive(Clone, Debug)]
pub enum SdfCategory {
    Primitive,
    Operation,
    Transform,
    Modifier,
    Utility,
    Light,
}
impl CategoryTrait for SdfCategory {
    fn name(&self) -> String {
        match self {
            Self::Primitive => "Primitive".into(),
            Self::Operation => "Operation".into(),
            Self::Transform => "Transform".into(),
            Self::Modifier => "Modifier".into(),
            Self::Utility => "Utility".into(),
            Self::Light => "Light".into(),
        }
    }
}

/// Node templates - one per creatable node variant.
#[derive(Clone, Debug)]
pub enum SdfNodeTemplate {
    Primitive(SdfPrimitive),
    Operation(CsgOp),
    Transform,
    Modifier(ModifierKind),
    Reroute,
    Light(LightType),
}

/// User state passed through to all trait methods.
pub struct SdfGraphUserState {
    /// Mutable access to scene data for inline editing.
    /// We snapshot node data before drawing so bottom_ui can mutate it,
    /// then write back changes after drawing.
    pub node_data_snapshot: HashMap<SceneNodeId, NodeData>,
    /// Tracks which snapshot entries were mutated.
    pub dirty_nodes: HashSet<SceneNodeId>,
    /// Nodes that were created via the node finder during this frame.
    pub created_via_finder: Vec<(NodeId, SdfNodeTemplate)>,
    /// Current graph zoom level, used to scale hardcoded widget widths.
    pub zoom: f32,
}

impl SdfGraphUserState {
    pub fn new() -> Self {
        Self {
            node_data_snapshot: HashMap::new(),
            dirty_nodes: HashSet::new(),
            created_via_finder: Vec::new(),
            zoom: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl DataTypeTrait<SdfGraphUserState> for SdfDataType {
    fn data_type_color(&self, _user_state: &mut SdfGraphUserState) -> Color32 {
        COLOR_PORT
    }

    fn name(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("SDF")
    }
}

impl WidgetValueTrait for SdfValueType {
    type Response = SdfResponse;
    type UserState = SdfGraphUserState;
    type NodeData = SdfNodeData;

    fn value_widget(
        &mut self,
        param_name: &str,
        _node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut Self::UserState,
        _node_data: &Self::NodeData,
    ) -> Vec<Self::Response> {
        ui.label(param_name);
        vec![]
    }

    fn value_widget_connected(
        &mut self,
        param_name: &str,
        _node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut Self::UserState,
        _node_data: &Self::NodeData,
    ) -> Vec<Self::Response> {
        ui.label(param_name);
        vec![]
    }
}

impl NodeDataTrait for SdfNodeData {
    type Response = SdfResponse;
    type UserState = SdfGraphUserState;
    type DataType = SdfDataType;
    type ValueType = SdfValueType;

    fn titlebar_color(
        &self,
        _ui: &egui::Ui,
        _node_id: NodeId,
        _graph: &Graph<Self, Self::DataType, Self::ValueType>,
        user_state: &mut Self::UserState,
    ) -> Option<Color32> {
        user_state
            .node_data_snapshot
            .get(&self.scene_node_id)
            .map(|data| match data {
                NodeData::Primitive { .. } => COLOR_PRIM_BADGE,
                NodeData::Operation { .. } => COLOR_OP_BADGE,
                NodeData::Sculpt { .. } => COLOR_SCULPT_BADGE,
                NodeData::Transform { .. } => COLOR_TRANSFORM_BADGE,
                NodeData::Modifier { .. } => COLOR_MODIFIER_BADGE,
                NodeData::Light { .. } => COLOR_LIGHT_BADGE,
            })
    }

    fn bottom_ui(
        &self,
        ui: &mut egui::Ui,
        _node_id: NodeId,
        _graph: &Graph<Self, Self::DataType, Self::ValueType>,
        user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<Self::Response, Self>>
    where
        Self::Response: UserResponseTrait,
    {
        let sid = self.scene_node_id;
        let Some(data) = user_state.node_data_snapshot.get_mut(&sid) else {
            return vec![];
        };

        let zoom = user_state.zoom;
        ui.spacing_mut().item_spacing.y = 2.0 * zoom;

        let mut changed = false;

        match data {
            NodeData::Primitive {
                ref mut kind,
                ref mut position,
                ref mut rotation,
                ref mut scale,
                ref mut material,
                ..
            } => {
                let mut new_kind = kind.clone();
                egui::ComboBox::from_id_salt(egui::Id::new(("prim_type", sid)))
                    .selected_text(new_kind.base_name())
                    .width((NODE_WIDTH - 12.0) * zoom)
                    .show_ui(ui, |ui| {
                        for v in SdfPrimitive::ALL {
                            ui.selectable_value(&mut new_kind, v.clone(), v.base_name());
                        }
                    });
                if new_kind != *kind {
                    *scale = new_kind.default_scale();
                    *kind = new_kind;
                    changed = true;
                }

                changed |= compact_vec3(ui, "Pos", position, 0.05, None);
                changed |= compact_rotation(ui, "Rot", rotation);

                let params = kind.scale_params();
                if !params.is_empty() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        for &(label, axis) in params {
                            ui.label(egui::RichText::new(label).small());
                            let val = match axis {
                                0 => &mut scale.x,
                                1 => &mut scale.y,
                                _ => &mut scale.z,
                            };
                            if ui
                                .add(
                                    egui::DragValue::new(val)
                                        .speed(0.05)
                                        .range(0.01..=100.0)
                                        .max_decimals(2),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });
                }

                let mut c = [
                    material.base_color.x,
                    material.base_color.y,
                    material.base_color.z,
                ];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Col").small());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        changed = true;
                    }
                });
                material.base_color = glam::Vec3::new(c[0], c[1], c[2]);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Met").small());
                    if ui
                        .add(
                            egui::DragValue::new(&mut material.metallic)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .max_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label(egui::RichText::new("Rgh").small());
                    if ui
                        .add(
                            egui::DragValue::new(&mut material.roughness)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .max_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            NodeData::Operation {
                ref mut op,
                ref mut smooth_k,
                ref mut steps,
                ref mut color_blend,
                ..
            } => {
                let mut new_op = op.clone();
                egui::ComboBox::from_id_salt(egui::Id::new(("op_type", sid)))
                    .selected_text(new_op.base_name())
                    .width((NODE_WIDTH - 12.0) * zoom)
                    .show_ui(ui, |ui| {
                        for v in CsgOp::ALL {
                            ui.selectable_value(&mut new_op, v.clone(), v.base_name());
                        }
                    });
                if new_op != *op {
                    *smooth_k = new_op.default_smooth_k();
                    *steps = new_op.default_steps();
                    *color_blend = -1.0;
                    *op = new_op;
                    changed = true;
                }

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Smooth K").small());
                    if ui
                        .add(
                            egui::DragValue::new(smooth_k)
                                .speed(0.01)
                                .range(0.0..=2.0)
                                .max_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });

                if op.has_steps_param() {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Count").small());
                        if ui
                            .add(
                                egui::DragValue::new(steps)
                                    .speed(0.1)
                                    .range(2.0..=16.0)
                                    .max_decimals(0),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                }

                if op.has_color_blend_param() {
                    let mut independent = *color_blend >= 0.0;
                    if ui
                        .checkbox(&mut independent, egui::RichText::new("Color Blend").small())
                        .changed()
                    {
                        if independent {
                            *color_blend = *smooth_k;
                        } else {
                            *color_blend = -1.0;
                        }
                        changed = true;
                    }
                    if independent {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Color K").small());
                            if ui
                                .add(
                                    egui::DragValue::new(color_blend)
                                        .speed(0.01)
                                        .range(0.0..=2.0)
                                        .max_decimals(2),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                    }
                }
            }
            NodeData::Transform {
                ref mut translation,
                ref mut rotation,
                ref mut scale,
                ..
            } => {
                changed |= compact_vec3(ui, "Pos", translation, 0.05, None);
                changed |= compact_rotation(ui, "Rot", rotation);
                changed |= compact_vec3(ui, "Scl", scale, 0.05, Some(0.01..=100.0));
            }
            NodeData::Modifier {
                ref mut kind,
                ref mut value,
                ref mut extra,
                ..
            } => {
                let mut new_kind = kind.clone();
                egui::ComboBox::from_id_salt(egui::Id::new(("mod_type", sid)))
                    .selected_text(new_kind.base_name())
                    .width((NODE_WIDTH - 12.0) * zoom)
                    .show_ui(ui, |ui| {
                        for v in ModifierKind::ALL {
                            ui.selectable_value(&mut new_kind, v.clone(), v.base_name());
                        }
                    });
                if new_kind != *kind {
                    *value = new_kind.default_value();
                    *extra = new_kind.default_extra();
                    *kind = new_kind;
                    changed = true;
                }

                changed |= match kind {
                    ModifierKind::Twist => scalar_drag(ui, "Rate", &mut value.x, 0.05, None),
                    ModifierKind::Bend => scalar_drag(ui, "Amount", &mut value.x, 0.05, None),
                    ModifierKind::Taper => scalar_drag(ui, "Factor", &mut value.x, 0.05, None),
                    ModifierKind::Round => {
                        scalar_drag(ui, "Radius", &mut value.x, 0.01, Some(0.0..=5.0))
                    }
                    ModifierKind::Onion => {
                        scalar_drag(ui, "Thick", &mut value.x, 0.01, Some(0.001..=5.0))
                    }
                    ModifierKind::Elongate => {
                        compact_vec3(ui, "Elong", value, 0.05, Some(0.0..=10.0))
                    }
                    ModifierKind::Mirror => {
                        let mut c = false;
                        ui.horizontal(|ui| {
                            let mut mx = value.x > 0.5;
                            let mut my = value.y > 0.5;
                            let mut mz = value.z > 0.5;
                            c |= ui.checkbox(&mut mx, "X").changed();
                            c |= ui.checkbox(&mut my, "Y").changed();
                            c |= ui.checkbox(&mut mz, "Z").changed();
                            value.x = if mx { 1.0 } else { 0.0 };
                            value.y = if my { 1.0 } else { 0.0 };
                            value.z = if mz { 1.0 } else { 0.0 };
                        });
                        c
                    }
                    ModifierKind::Repeat => compact_vec3(ui, "Space", value, 0.1, Some(0.0..=20.0)),
                    ModifierKind::FiniteRepeat => {
                        let c1 = compact_vec3(ui, "Space", value, 0.1, Some(0.0..=20.0));
                        let c2 = compact_vec3(ui, "Count", extra, 1.0, Some(0.0..=50.0));
                        c1 || c2
                    }
                    ModifierKind::RadialRepeat => {
                        scalar_drag(ui, "Count", &mut value.x, 1.0, Some(1.0..=64.0))
                    }
                    ModifierKind::Offset => {
                        scalar_drag(ui, "Offset", &mut value.x, 0.01, Some(-1.0..=1.0))
                    }
                    ModifierKind::Noise => {
                        let c1 = scalar_drag(ui, "Freq", &mut value.x, 0.1, Some(0.1..=20.0));
                        let c2 = scalar_drag(ui, "Amp", &mut value.y, 0.01, Some(0.0..=2.0));
                        let c3 = scalar_drag(ui, "Oct", &mut value.z, 1.0, Some(1.0..=8.0));
                        c1 || c2 || c3
                    }
                };
            }
            NodeData::Sculpt {
                ref mut position,
                ref mut rotation,
                ref mut material,
                ..
            } => {
                changed |= compact_vec3(ui, "Pos", position, 0.05, None);
                changed |= compact_rotation(ui, "Rot", rotation);

                let mut c = [
                    material.base_color.x,
                    material.base_color.y,
                    material.base_color.z,
                ];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Col").small());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        changed = true;
                    }
                });
                material.base_color = glam::Vec3::new(c[0], c[1], c[2]);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Met").small());
                    if ui
                        .add(
                            egui::DragValue::new(&mut material.metallic)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .max_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.label(egui::RichText::new("Rgh").small());
                    if ui
                        .add(
                            egui::DragValue::new(&mut material.roughness)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .max_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            NodeData::Light {
                ref mut light_type,
                ref mut color,
                ref mut intensity,
                ref mut range,
                ref mut spot_angle,
                ..
            } => {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Type").small());
                    egui::ComboBox::from_id_salt(format!("lt_{sid}"))
                        .width(80.0 * zoom)
                        .selected_text(light_type.label())
                        .show_ui(ui, |ui| {
                            for lt in crate::graph::scene::LightType::ALL {
                                if ui
                                    .selectable_label(
                                        std::mem::discriminant(light_type)
                                            == std::mem::discriminant(lt),
                                        lt.label(),
                                    )
                                    .clicked()
                                {
                                    *light_type = lt.clone();
                                    changed = true;
                                }
                            }
                        });
                });
                let mut c = [color.x, color.y, color.z];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Col").small());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        changed = true;
                    }
                });
                *color = glam::Vec3::new(c[0], c[1], c[2]);
                changed |= scalar_drag(ui, "Int", intensity, 0.05, Some(0.0..=10.0));
                changed |= scalar_drag(ui, "Rng", range, 0.1, Some(0.1..=50.0));
                if matches!(light_type, crate::graph::scene::LightType::Spot) {
                    changed |= scalar_drag(ui, "Ang", spot_angle, 1.0, Some(1.0..=179.0));
                }
            }
        }

        if changed {
            user_state.dirty_nodes.insert(sid);
        }

        vec![]
    }
}

impl NodeTemplateTrait for SdfNodeTemplate {
    type NodeData = SdfNodeData;
    type DataType = SdfDataType;
    type ValueType = SdfValueType;
    type UserState = SdfGraphUserState;
    type CategoryType = SdfCategory;

    fn node_finder_label(&self, _user_state: &mut Self::UserState) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Primitive(p) => std::borrow::Cow::Borrowed(p.base_name()),
            Self::Operation(o) => std::borrow::Cow::Borrowed(o.base_name()),
            Self::Transform => std::borrow::Cow::Borrowed("Transform"),
            Self::Modifier(m) => std::borrow::Cow::Borrowed(m.base_name()),
            Self::Reroute => std::borrow::Cow::Borrowed("Reroute"),
            Self::Light(l) => std::borrow::Cow::Owned(format!("{} Light", l.label())),
        }
    }

    fn node_finder_categories(&self, _user_state: &mut Self::UserState) -> Vec<Self::CategoryType> {
        match self {
            Self::Primitive(_) => vec![SdfCategory::Primitive],
            Self::Operation(_) => vec![SdfCategory::Operation],
            Self::Transform => vec![SdfCategory::Transform],
            Self::Modifier(_) => vec![SdfCategory::Modifier],
            Self::Reroute => vec![SdfCategory::Utility],
            Self::Light(_) => vec![SdfCategory::Light],
        }
    }

    fn node_graph_label(&self, _user_state: &mut Self::UserState) -> String {
        match self {
            Self::Primitive(p) => p.base_name().to_string(),
            Self::Operation(o) => o.base_name().to_string(),
            Self::Transform => "Transform".to_string(),
            Self::Modifier(m) => m.base_name().to_string(),
            Self::Reroute => "Reroute".to_string(),
            Self::Light(l) => format!("{} Light", l.label()),
        }
    }

    fn user_data(&self, _user_state: &mut Self::UserState) -> Self::NodeData {
        // scene_node_id will be set by the sync layer after scene creation
        SdfNodeData { scene_node_id: 0 }
    }

    fn build_node(
        &self,
        graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        user_state: &mut Self::UserState,
        node_id: NodeId,
    ) {
        // Add ports based on node type
        match self {
            Self::Primitive(_) | Self::Light(_) => {
                // Output only
                graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
            }
            Self::Operation(_) => {
                graph.add_input_param(
                    node_id,
                    "left".to_string(),
                    SdfDataType::Sdf,
                    SdfValueType::None,
                    InputParamKind::ConnectionOnly,
                    true,
                );
                graph.add_input_param(
                    node_id,
                    "right".to_string(),
                    SdfDataType::Sdf,
                    SdfValueType::None,
                    InputParamKind::ConnectionOnly,
                    true,
                );
                graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
            }
            Self::Transform | Self::Modifier(_) | Self::Reroute => {
                graph.add_input_param(
                    node_id,
                    "input".to_string(),
                    SdfDataType::Sdf,
                    SdfValueType::None,
                    InputParamKind::ConnectionOnly,
                    true,
                );
                graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
            }
        }

        // Record so the sync layer can create the scene node
        user_state.created_via_finder.push((node_id, self.clone()));
    }
}

/// Enumerates all templates for the node finder.
pub struct AllSdfTemplates;

impl NodeTemplateIter for AllSdfTemplates {
    type Item = SdfNodeTemplate;

    fn all_kinds(&self) -> Vec<Self::Item> {
        let mut templates = Vec::new();
        for p in SdfPrimitive::ALL {
            templates.push(SdfNodeTemplate::Primitive(p.clone()));
        }
        for o in CsgOp::ALL {
            templates.push(SdfNodeTemplate::Operation(o.clone()));
        }
        templates.push(SdfNodeTemplate::Transform);
        templates.push(SdfNodeTemplate::Reroute);
        for m in ModifierKind::ALL {
            templates.push(SdfNodeTemplate::Modifier(m.clone()));
        }
        templates
    }
}

/// Enumerates templates for the dedicated Light Graph finder.
pub struct LightSdfTemplates;

impl NodeTemplateIter for LightSdfTemplates {
    type Item = SdfNodeTemplate;

    fn all_kinds(&self) -> Vec<Self::Item> {
        let mut templates = Vec::new();
        for l in LightType::ALL {
            templates.push(SdfNodeTemplate::Light(l.clone()));
        }
        templates
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

pub type SdfGraphState =
    GraphEditorState<SdfNodeData, SdfDataType, SdfValueType, SdfNodeTemplate, SdfGraphUserState>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphFilterMode {
    SdfOnly,
    LightsOnly,
}

// ---------------------------------------------------------------------------
// ID Mapping
// ---------------------------------------------------------------------------

pub struct NodeIdMap {
    pub scene_to_graph: HashMap<SceneNodeId, NodeId>,
    pub graph_to_scene: HashMap<NodeId, SceneNodeId>,
}

impl NodeIdMap {
    pub fn new() -> Self {
        Self {
            scene_to_graph: HashMap::new(),
            graph_to_scene: HashMap::new(),
        }
    }

    pub fn insert(&mut self, scene_id: SceneNodeId, graph_id: NodeId) {
        self.scene_to_graph.insert(scene_id, graph_id);
        self.graph_to_scene.insert(graph_id, scene_id);
    }

    pub fn clear(&mut self) {
        self.scene_to_graph.clear();
        self.graph_to_scene.clear();
    }
}

// ---------------------------------------------------------------------------
// Sync: Scene -> Graph (full rebuild)
// ---------------------------------------------------------------------------

fn light_related_scene_ids(scene: &Scene) -> HashSet<SceneNodeId> {
    let parent_map = scene.build_parent_map();
    let mut included: HashSet<SceneNodeId> = HashSet::new();

    for (&id, node) in &scene.nodes {
        if matches!(node.data, NodeData::Light { .. }) {
            included.insert(id);
            let mut cursor = id;
            while let Some(parent) = parent_map.get(&cursor).copied() {
                let Some(parent_node) = scene.nodes.get(&parent) else {
                    break;
                };
                // Light graph shows light chains (Light <- Transform <- Transform ...).
                // Stop if the chain reaches a non-transform parent.
                if !matches!(parent_node.data, NodeData::Transform { .. }) {
                    break;
                }
                if included.insert(parent) {
                    cursor = parent;
                } else {
                    break;
                }
            }
        }
    }

    included
}

pub fn rebuild_graph_from_scene(
    scene: &Scene,
    graph_state: &mut SdfGraphState,
    id_map: &mut NodeIdMap,
) {
    rebuild_graph_from_scene_with_filter(scene, graph_state, id_map, GraphFilterMode::SdfOnly);
}

fn rebuild_graph_from_scene_impl(
    scene: &Scene,
    graph_state: &mut SdfGraphState,
    id_map: &mut NodeIdMap,
    include_node: impl Fn(SceneNodeId) -> bool,
) {
    // Save old positions (keyed by scene_node_id) before clearing
    let mut saved_positions: HashMap<SceneNodeId, egui::Pos2> = HashMap::new();
    for (graph_id, pos) in graph_state.node_positions.iter() {
        if let Some(&scene_id) = id_map.graph_to_scene.get(&graph_id) {
            saved_positions.insert(scene_id, *pos);
        }
    }

    // Clear the graph
    id_map.clear();
    graph_state.graph = Graph::new();
    graph_state.node_order.clear();
    graph_state.node_positions = Default::default();
    graph_state.selected_nodes.clear();
    graph_state.connection_in_progress = None;

    // Create nodes (sorted by id for determinism)
    let mut scene_ids: Vec<SceneNodeId> = scene.nodes.keys().copied().collect();
    scene_ids.sort();

    for &sid in &scene_ids {
        if !include_node(sid) {
            continue;
        }
        let Some(node) = scene.nodes.get(&sid) else {
            continue;
        };
        let user_data = SdfNodeData { scene_node_id: sid };
        let label = node.name.clone();

        let graph_node_id = graph_state
            .graph
            .add_node(label, user_data, |graph, node_id| {
                // Add ports based on node type
                match &node.data {
                    NodeData::Primitive { .. } | NodeData::Light { .. } => {
                        graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
                    }
                    NodeData::Operation { .. } => {
                        graph.add_input_param(
                            node_id,
                            "left".to_string(),
                            SdfDataType::Sdf,
                            SdfValueType::None,
                            InputParamKind::ConnectionOnly,
                            true,
                        );
                        graph.add_input_param(
                            node_id,
                            "right".to_string(),
                            SdfDataType::Sdf,
                            SdfValueType::None,
                            InputParamKind::ConnectionOnly,
                            true,
                        );
                        graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
                    }
                    NodeData::Sculpt { .. }
                    | NodeData::Transform { .. }
                    | NodeData::Modifier { .. } => {
                        graph.add_input_param(
                            node_id,
                            "input".to_string(),
                            SdfDataType::Sdf,
                            SdfValueType::None,
                            InputParamKind::ConnectionOnly,
                            true,
                        );
                        graph.add_output_param(node_id, "out".to_string(), SdfDataType::Sdf);
                    }
                }
            });

        graph_state.node_order.push(graph_node_id);
        id_map.insert(sid, graph_node_id);

        // Restore saved position or leave for auto-layout
        if let Some(pos) = saved_positions.get(&sid) {
            graph_state.node_positions.insert(graph_node_id, *pos);
        }
    }

    // Create connections
    for &sid in &scene_ids {
        if !include_node(sid) {
            continue;
        }
        let Some(node) = scene.nodes.get(&sid) else {
            continue;
        };
        let Some(&graph_id) = id_map.scene_to_graph.get(&sid) else {
            continue;
        };

        match &node.data {
            NodeData::Operation { left, right, .. } => {
                if let Some(left_id) = left {
                    if let Some(&left_graph_id) = id_map.scene_to_graph.get(left_id) {
                        let input_id = graph_state.graph[graph_id]
                            .get_input("left")
                            .expect("left input");
                        let output_id = graph_state.graph[left_graph_id]
                            .get_output("out")
                            .expect("out output");
                        graph_state.graph.add_connection(output_id, input_id, 0);
                    }
                }
                if let Some(right_id) = right {
                    if let Some(&right_graph_id) = id_map.scene_to_graph.get(right_id) {
                        let input_id = graph_state.graph[graph_id]
                            .get_input("right")
                            .expect("right input");
                        let output_id = graph_state.graph[right_graph_id]
                            .get_output("out")
                            .expect("out output");
                        graph_state.graph.add_connection(output_id, input_id, 0);
                    }
                }
            }
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => {
                if let Some(input_id_scene) = input {
                    if let Some(&input_graph_id) = id_map.scene_to_graph.get(input_id_scene) {
                        let input_id = graph_state.graph[graph_id]
                            .get_input("input")
                            .expect("input param");
                        let output_id = graph_state.graph[input_graph_id]
                            .get_output("out")
                            .expect("out output");
                        graph_state.graph.add_connection(output_id, input_id, 0);
                    }
                }
            }
            _ => {}
        }
    }
}

fn rebuild_graph_from_scene_with_filter(
    scene: &Scene,
    graph_state: &mut SdfGraphState,
    id_map: &mut NodeIdMap,
    filter_mode: GraphFilterMode,
) {
    match filter_mode {
        GraphFilterMode::SdfOnly => {
            let light_related = light_related_scene_ids(scene);
            rebuild_graph_from_scene_impl(scene, graph_state, id_map, move |sid| {
                !light_related.contains(&sid)
            });
        }
        GraphFilterMode::LightsOnly => {
            let light_related = light_related_scene_ids(scene);
            rebuild_graph_from_scene_impl(scene, graph_state, id_map, move |sid| {
                light_related.contains(&sid)
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Layout / placement helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConsumerSlot {
    Left,
    Right,
    Input,
}

fn node_size_for_scene(scene: &Scene, sid: SceneNodeId) -> egui::Vec2 {
    let h = scene
        .nodes
        .get(&sid)
        .map(|n| match n.data {
            NodeData::Primitive { .. } => 190.0,
            NodeData::Operation { .. } => 150.0,
            NodeData::Sculpt { .. } => 185.0,
            NodeData::Transform { .. } => 165.0,
            NodeData::Modifier { .. } => 175.0,
            NodeData::Light { .. } => 170.0,
        })
        .unwrap_or(DEFAULT_NODE_HEIGHT);
    egui::vec2(NODE_WIDTH, h)
}

fn snap_to_grid(pos: egui::Pos2) -> egui::Pos2 {
    egui::Pos2::new(
        (pos.x / GRID_STEP).round() * GRID_STEP,
        (pos.y / GRID_STEP).round() * GRID_STEP,
    )
}

fn find_non_overlapping_position(
    candidate: egui::Pos2,
    size: egui::Vec2,
    occupied: &[egui::Rect],
) -> egui::Pos2 {
    let base = snap_to_grid(candidate);
    let is_free = |pos: egui::Pos2| {
        let rect = egui::Rect::from_min_size(pos, size).expand(NODE_PADDING);
        occupied.iter().all(|r| !r.intersects(rect))
    };

    if is_free(base) {
        return base;
    }

    for ring in 1_i32..=24_i32 {
        for dy in -ring..=ring {
            for dx in -ring..=ring {
                if dx.abs() != ring && dy.abs() != ring {
                    continue;
                }
                let pos = snap_to_grid(egui::Pos2::new(
                    base.x + dx as f32 * GRID_STEP,
                    base.y + dy as f32 * GRID_STEP,
                ));
                if is_free(pos) {
                    return pos;
                }
            }
        }
    }

    base
}

fn incoming_inputs_for_node(
    scene: &Scene,
    node_id: SceneNodeId,
    included: &HashSet<SceneNodeId>,
) -> Vec<SceneNodeId> {
    let Some(node) = scene.nodes.get(&node_id) else {
        return Vec::new();
    };

    match &node.data {
        NodeData::Operation { left, right, .. } => {
            let mut v = Vec::new();
            if let Some(id) = left.filter(|id| included.contains(id)) {
                v.push(id);
            }
            if let Some(id) = right.filter(|id| included.contains(id)) {
                v.push(id);
            }
            v
        }
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => input
            .filter(|id| included.contains(id))
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

fn node_consumers(
    scene: &Scene,
    node_id: SceneNodeId,
    included: &HashSet<SceneNodeId>,
) -> Vec<(SceneNodeId, ConsumerSlot)> {
    let mut consumers = Vec::new();

    for (&sid, node) in &scene.nodes {
        if !included.contains(&sid) {
            continue;
        }

        match &node.data {
            NodeData::Operation { left, right, .. } => {
                if *left == Some(node_id) {
                    consumers.push((sid, ConsumerSlot::Left));
                }
                if *right == Some(node_id) {
                    consumers.push((sid, ConsumerSlot::Right));
                }
            }
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => {
                if *input == Some(node_id) {
                    consumers.push((sid, ConsumerSlot::Input));
                }
            }
            _ => {}
        }
    }

    consumers
}

fn compute_longest_path_level(
    scene: &Scene,
    id: SceneNodeId,
    included: &HashSet<SceneNodeId>,
    cache: &mut HashMap<SceneNodeId, u32>,
    visiting: &mut HashSet<SceneNodeId>,
) -> u32 {
    if let Some(&level) = cache.get(&id) {
        return level;
    }
    if !visiting.insert(id) {
        return 0;
    }

    let mut max_input_level = 0;
    for input in incoming_inputs_for_node(scene, id, included) {
        let input_level = compute_longest_path_level(scene, input, included, cache, visiting);
        max_input_level = max_input_level.max(input_level + 1);
    }

    visiting.remove(&id);
    cache.insert(id, max_input_level);
    max_input_level
}

fn build_layered_layout(scene: &Scene, id_map: &NodeIdMap) -> HashMap<SceneNodeId, egui::Pos2> {
    let included: HashSet<SceneNodeId> = id_map.scene_to_graph.keys().copied().collect();
    if included.is_empty() {
        return HashMap::new();
    }

    let mut level_cache = HashMap::new();
    let mut visiting = HashSet::new();
    for sid in &included {
        let _ = compute_longest_path_level(scene, *sid, &included, &mut level_cache, &mut visiting);
    }

    let mut layers: BTreeMap<u32, Vec<SceneNodeId>> = BTreeMap::new();
    for sid in included.iter().copied() {
        let level = *level_cache.get(&sid).unwrap_or(&0);
        layers.entry(level).or_default().push(sid);
    }

    let mut rank_by_node: HashMap<SceneNodeId, f32> = HashMap::new();
    let mut ordered_layers: BTreeMap<u32, Vec<SceneNodeId>> = BTreeMap::new();

    for (level, mut nodes) in layers {
        nodes.sort_by(|a, b| {
            let avg_rank = |sid: SceneNodeId| {
                let preds = incoming_inputs_for_node(scene, sid, &included);
                let mut total = 0.0;
                let mut count = 0.0;
                for pred in preds {
                    if let Some(rank) = rank_by_node.get(&pred) {
                        total += *rank;
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    total / count
                } else {
                    sid as f32
                }
            };

            let ka = avg_rank(*a);
            let kb = avg_rank(*b);
            ka.partial_cmp(&kb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        for (idx, sid) in nodes.iter().enumerate() {
            rank_by_node.insert(*sid, idx as f32);
        }
        ordered_layers.insert(level, nodes);
    }

    let mut layout = HashMap::new();
    for (level, nodes) in ordered_layers {
        for (row, sid) in nodes.iter().enumerate() {
            layout.insert(
                *sid,
                egui::Pos2::new(level as f32 * COL_SPACING, row as f32 * ROW_SPACING),
            );
        }
    }
    layout
}

fn place_unpositioned_nodes(
    scene: &Scene,
    graph_state: &mut SdfGraphState,
    id_map: &NodeIdMap,
    spawn_anchor: egui::Pos2,
) {
    let layout = build_layered_layout(scene, id_map);
    let included: HashSet<SceneNodeId> = id_map.scene_to_graph.keys().copied().collect();

    let mut occupied_rects: Vec<egui::Rect> = id_map
        .scene_to_graph
        .iter()
        .filter_map(|(sid, graph_id)| {
            graph_state.node_positions.get(*graph_id).map(|pos| {
                egui::Rect::from_min_size(*pos, node_size_for_scene(scene, *sid))
                    .expand(NODE_PADDING)
            })
        })
        .collect();

    let mut unpositioned: Vec<(SceneNodeId, NodeId)> = id_map
        .scene_to_graph
        .iter()
        .filter_map(|(sid, graph_id)| {
            (!graph_state.node_positions.contains_key(*graph_id)).then_some((*sid, *graph_id))
        })
        .collect();
    unpositioned.sort_by_key(|(sid, _)| *sid);

    let mut isolated_count = 0usize;

    for (sid, graph_id) in unpositioned {
        let size = node_size_for_scene(scene, sid);
        let incoming = incoming_inputs_for_node(scene, sid, &included);
        let consumers = node_consumers(scene, sid, &included);

        let preferred = if !incoming.is_empty() {
            let mut max_x = f32::NEG_INFINITY;
            let mut y_sum = 0.0;
            let mut count = 0.0;

            for input_sid in incoming {
                if let Some(&input_graph_id) = id_map.scene_to_graph.get(&input_sid) {
                    if let Some(input_pos) = graph_state.node_positions.get(input_graph_id) {
                        let input_size = node_size_for_scene(scene, input_sid);
                        max_x = max_x.max(input_pos.x + input_size.x);
                        y_sum += input_pos.y;
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                egui::Pos2::new(max_x + SPAWN_X_OFFSET, y_sum / count)
            } else {
                *layout.get(&sid).unwrap_or(&spawn_anchor)
            }
        } else if !consumers.is_empty() {
            let mut min_x = f32::INFINITY;
            let mut y_sum = 0.0;
            let mut count = 0.0;

            for (consumer_sid, _) in consumers {
                if let Some(&consumer_graph_id) = id_map.scene_to_graph.get(&consumer_sid) {
                    if let Some(consumer_pos) = graph_state.node_positions.get(consumer_graph_id) {
                        min_x = min_x.min(consumer_pos.x);
                        y_sum += consumer_pos.y;
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                egui::Pos2::new(min_x - SPAWN_X_OFFSET, y_sum / count)
            } else {
                *layout.get(&sid).unwrap_or(&spawn_anchor)
            }
        } else {
            let col = (isolated_count % 3) as f32;
            let row = (isolated_count / 3) as f32;
            isolated_count += 1;
            egui::Pos2::new(
                spawn_anchor.x + SPAWN_X_OFFSET + col * (NODE_WIDTH + NODE_PADDING * 2.0),
                spawn_anchor.y + row * (DEFAULT_NODE_HEIGHT + NODE_PADDING * 2.0),
            )
        };

        let final_pos = find_non_overlapping_position(preferred, size, &occupied_rects);
        graph_state.node_positions.insert(graph_id, final_pos);
        occupied_rects.push(egui::Rect::from_min_size(final_pos, size).expand(NODE_PADDING));
    }
}

fn organize_graph(scene: &Scene, graph_state: &mut SdfGraphState, id_map: &NodeIdMap) {
    let layout = build_layered_layout(scene, id_map);
    graph_state.node_positions = Default::default();

    let mut occupied_rects: Vec<egui::Rect> = Vec::new();
    let mut ids: Vec<_> = layout.keys().copied().collect();
    ids.sort_by(|a, b| {
        let pa = layout.get(a).copied().unwrap_or_default();
        let pb = layout.get(b).copied().unwrap_or_default();
        pa.x.partial_cmp(&pb.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| pa.y.partial_cmp(&pb.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    for sid in ids {
        let Some(&graph_id) = id_map.scene_to_graph.get(&sid) else {
            continue;
        };
        let preferred = *layout.get(&sid).unwrap_or(&egui::Pos2::ZERO);
        let size = node_size_for_scene(scene, sid);
        let pos = find_non_overlapping_position(preferred, size, &occupied_rects);
        graph_state.node_positions.insert(graph_id, pos);
        occupied_rects.push(egui::Rect::from_min_size(pos, size).expand(NODE_PADDING));
    }
}

// ---------------------------------------------------------------------------
// Process graph responses (Graph -> Scene sync)
// ---------------------------------------------------------------------------

fn process_graph_responses(
    responses: &GraphResponse<SdfResponse, SdfNodeData>,
    graph_state: &mut SdfGraphState,
    id_map: &mut NodeIdMap,
    selected: &mut Option<SceneNodeId>,
    layout_dirty: &mut bool,
    actions: &mut ActionSink,
) {
    for response in &responses.node_responses {
        match response {
            NodeResponse::ConnectEventEnded { output, input, .. } => {
                // Find which scene nodes are involved
                let output_graph_node = graph_state.graph.get_output(*output).node;
                let input_graph_node = graph_state.graph.get_input(*input).node;

                let Some(&source_scene_id) = id_map.graph_to_scene.get(&output_graph_node) else {
                    continue;
                };
                let Some(&target_scene_id) = id_map.graph_to_scene.get(&input_graph_node) else {
                    continue;
                };

                // Determine which input port it is by name
                let input_name = graph_state.graph[input_graph_node]
                    .inputs
                    .iter()
                    .find(|(_, iid)| *iid == *input)
                    .map(|(name, _)| name.as_str());

                match input_name {
                    Some("left") => actions.push(Action::SetLeftChild {
                        parent: target_scene_id,
                        child: Some(source_scene_id),
                    }),
                    Some("right") => actions.push(Action::SetRightChild {
                        parent: target_scene_id,
                        child: Some(source_scene_id),
                    }),
                    Some("input") => actions.push(Action::SetSculptInput {
                        parent: target_scene_id,
                        child: Some(source_scene_id),
                    }),
                    _ => {}
                }
            }
            NodeResponse::DisconnectEvent { input, .. } => {
                // The InputId may already be removed from the graph if this
                // disconnect was caused by a node deletion (the library removes
                // the node's inputs before returning DisconnectEvent responses).
                let Some(input_param) = graph_state.graph.try_get_input(*input) else {
                    continue;
                };
                let input_graph_node = input_param.node;
                let Some(&target_scene_id) = id_map.graph_to_scene.get(&input_graph_node) else {
                    continue;
                };

                let input_name = graph_state.graph[input_graph_node]
                    .inputs
                    .iter()
                    .find(|(_, iid)| *iid == *input)
                    .map(|(name, _)| name.as_str());

                match input_name {
                    Some("left") => actions.push(Action::SetLeftChild {
                        parent: target_scene_id,
                        child: None,
                    }),
                    Some("right") => actions.push(Action::SetRightChild {
                        parent: target_scene_id,
                        child: None,
                    }),
                    Some("input") => actions.push(Action::SetSculptInput {
                        parent: target_scene_id,
                        child: None,
                    }),
                    _ => {}
                }
            }
            NodeResponse::SelectNode(graph_id) => {
                if let Some(&scene_id) = id_map.graph_to_scene.get(graph_id) {
                    *selected = Some(scene_id);
                }
            }
            NodeResponse::DeleteNodeFull { node_id, node } => {
                let scene_id = node.user_data.scene_node_id;
                actions.push(Action::DeleteNode(scene_id));
                id_map.scene_to_graph.remove(&scene_id);
                id_map.graph_to_scene.remove(node_id);
                if *selected == Some(scene_id) {
                    *selected = None;
                }
                *layout_dirty = true;
            }
            NodeResponse::CreatedNode(graph_id) => {
                // Node was created via the node finder.
                // We need to create the corresponding scene node.
                // The template was recorded in user_state.created_via_finder
                // (handled below in draw())
                let _ = graph_id;
            }
            _ => {}
        }
    }
}

fn is_reroute_transform(scene: &Scene, node_id: SceneNodeId) -> bool {
    scene.nodes.get(&node_id).is_some_and(|n| {
        matches!(n.data, NodeData::Transform { .. }) && n.name.starts_with("Reroute")
    })
}

fn is_insertable_passthrough(scene: &Scene, node_id: SceneNodeId) -> bool {
    scene.nodes.get(&node_id).is_some_and(|n| {
        matches!(n.data, NodeData::Modifier { .. }) || is_reroute_transform(scene, node_id)
    })
}

fn graph_node_has_any_connections(graph: &SdfGraphState, graph_node_id: NodeId) -> bool {
    let node = &graph.graph[graph_node_id];

    let has_input = node
        .inputs
        .iter()
        .any(|(_, iid)| !graph.graph.connections(*iid).is_empty());
    if has_input {
        return true;
    }

    node.outputs
        .iter()
        .any(|(_, oid)| graph.graph.iter_connections().any(|(_, out)| out == *oid))
}

fn try_auto_insert_candidates_on_hovered_connection(
    scene: &mut Scene,
    state: &mut NodeGraphState,
    responses: &GraphResponse<SdfResponse, SdfNodeData>,
    candidate_graph_nodes: &[NodeId],
) -> bool {
    let Some((target_input, source_output)) = responses.connection_under_cursor else {
        return false;
    };

    let source_graph_node = state.graph_state.graph.get_output(source_output).node;
    let target_graph_node = state.graph_state.graph.get_input(target_input).node;

    let Some(&source_scene_id) = state.id_map.graph_to_scene.get(&source_graph_node) else {
        return false;
    };
    let Some(&target_scene_id) = state.id_map.graph_to_scene.get(&target_graph_node) else {
        return false;
    };

    let input_name = state.graph_state.graph[target_graph_node]
        .inputs
        .iter()
        .find(|(_, iid)| *iid == target_input)
        .map(|(name, _)| name.as_str());

    for graph_node_id in candidate_graph_nodes {
        if *graph_node_id == source_graph_node || *graph_node_id == target_graph_node {
            continue;
        }

        let Some(&insert_scene_id) = state.id_map.graph_to_scene.get(graph_node_id) else {
            continue;
        };

        if !is_insertable_passthrough(scene, insert_scene_id) {
            continue;
        }

        // Safety: only auto-insert clean pass-through nodes (no existing links).
        if graph_node_has_any_connections(&state.graph_state, *graph_node_id) {
            continue;
        }

        scene.set_sculpt_input(insert_scene_id, Some(source_scene_id));
        match input_name {
            Some("left") => scene.set_left_child(target_scene_id, Some(insert_scene_id)),
            Some("right") => scene.set_right_child(target_scene_id, Some(insert_scene_id)),
            Some("input") => scene.set_sculpt_input(target_scene_id, Some(insert_scene_id)),
            _ => continue,
        }

        state.select_single(insert_scene_id);
        state.needs_initial_rebuild = true;
        state.last_structure_key = scene.structure_key();
        return true;
    }

    false
}

fn viewport_center_graph_space(graph_rect: egui::Rect, graph_state: &SdfGraphState) -> egui::Pos2 {
    let pan = graph_state.pan_zoom.pan;
    let zoom = graph_state.pan_zoom.zoom;
    egui::Pos2::new(
        (graph_rect.width() / 2.0 - pan.x) / zoom,
        (graph_rect.height() / 2.0 - pan.y) / zoom,
    )
}

fn graph_pos_to_screen(
    graph_rect: egui::Rect,
    pan_zoom: &egui_node_graph2::PanZoom,
    graph_pos: egui::Pos2,
) -> egui::Pos2 {
    egui::Pos2::new(
        graph_rect.min.x + pan_zoom.pan.x + graph_pos.x * pan_zoom.zoom,
        graph_rect.min.y + pan_zoom.pan.y + graph_pos.y * pan_zoom.zoom,
    )
}

fn screen_to_graph_space(
    graph_rect: egui::Rect,
    pan_zoom: &egui_node_graph2::PanZoom,
    screen_pos: egui::Pos2,
) -> egui::Pos2 {
    egui::Pos2::new(
        (screen_pos.x - graph_rect.min.x - pan_zoom.pan.x) / pan_zoom.zoom,
        (screen_pos.y - graph_rect.min.y - pan_zoom.pan.y) / pan_zoom.zoom,
    )
}

fn center_view_on_node(
    scene: &Scene,
    state: &mut NodeGraphState,
    graph_rect: egui::Rect,
    sid: SceneNodeId,
) {
    let Some(&graph_id) = state.id_map.scene_to_graph.get(&sid) else {
        return;
    };
    let Some(&pos) = state.graph_state.node_positions.get(graph_id) else {
        return;
    };

    let zoom = state.graph_state.pan_zoom.zoom;
    let size = node_size_for_scene(scene, sid);
    state.graph_state.pan_zoom.pan = egui::Vec2::new(
        graph_rect.width() / 2.0 - (pos.x + size.x * 0.5) * zoom,
        graph_rect.height() / 2.0 - (pos.y + size.y * 0.5) * zoom,
    );
}

fn compute_spawn_anchor(
    state: &NodeGraphState,
    selected_visible: Option<SceneNodeId>,
    graph_rect: egui::Rect,
    ctx: &egui::Context,
) -> egui::Pos2 {
    if let Some(sel) = selected_visible {
        if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sel) {
            if let Some(&pos) = state.graph_state.node_positions.get(graph_id) {
                return snap_to_grid(egui::Pos2::new(pos.x + SPAWN_X_OFFSET, pos.y));
            }
        }
    }

    if let Some(pointer) = ctx.input(|i| i.pointer.hover_pos()) {
        if graph_rect.contains(pointer) {
            return snap_to_grid(screen_to_graph_space(
                graph_rect,
                &state.graph_state.pan_zoom,
                pointer,
            ));
        }
    }

    snap_to_grid(viewport_center_graph_space(graph_rect, &state.graph_state))
}

fn node_under_pointer(
    scene: &Scene,
    state: &NodeGraphState,
    graph_rect: egui::Rect,
    pointer_screen: egui::Pos2,
) -> Option<NodeId> {
    let mut ordered = state.graph_state.node_order.clone();
    ordered.reverse();

    for graph_id in ordered {
        let Some(&graph_pos) = state.graph_state.node_positions.get(graph_id) else {
            continue;
        };
        let Some(&sid) = state.id_map.graph_to_scene.get(&graph_id) else {
            continue;
        };
        let size = node_size_for_scene(scene, sid) * state.graph_state.pan_zoom.zoom;
        let top_left = graph_pos_to_screen(graph_rect, &state.graph_state.pan_zoom, graph_pos);
        let rect = egui::Rect::from_min_size(top_left, size);
        if rect.contains(pointer_screen) {
            return Some(graph_id);
        }
    }

    None
}

fn primary_input_for_bypass(scene: &Scene, node_id: SceneNodeId) -> Option<SceneNodeId> {
    scene.nodes.get(&node_id).and_then(|n| match &n.data {
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => *input,
        NodeData::Operation { left, right, .. } => match (left, right) {
            (Some(v), None) | (None, Some(v)) => Some(*v),
            _ => None,
        },
        _ => None,
    })
}

fn clear_node_inputs(scene: &mut Scene, node_id: SceneNodeId) {
    let Some(node) = scene.nodes.get(&node_id) else {
        return;
    };
    match node.data {
        NodeData::Operation { .. } => {
            scene.set_left_child(node_id, None);
            scene.set_right_child(node_id, None);
        }
        NodeData::Sculpt { .. } | NodeData::Transform { .. } | NodeData::Modifier { .. } => {
            scene.set_sculpt_input(node_id, None);
        }
        _ => {}
    }
}

fn disconnect_node_with_bypass(
    scene: &mut Scene,
    id_map: &NodeIdMap,
    node_id: SceneNodeId,
) -> bool {
    let included: HashSet<SceneNodeId> = id_map.scene_to_graph.keys().copied().collect();
    let consumers = node_consumers(scene, node_id, &included);
    let upstream = primary_input_for_bypass(scene, node_id);

    let mut changed = false;
    for (consumer_id, slot) in consumers {
        match slot {
            ConsumerSlot::Left => {
                scene.set_left_child(consumer_id, upstream);
                changed = true;
            }
            ConsumerSlot::Right => {
                scene.set_right_child(consumer_id, upstream);
                changed = true;
            }
            ConsumerSlot::Input => {
                scene.set_sculpt_input(consumer_id, upstream);
                changed = true;
            }
        }
    }

    clear_node_inputs(scene, node_id);
    changed || upstream.is_some()
}
// ---------------------------------------------------------------------------
// Public state & draw
// ---------------------------------------------------------------------------

pub struct NodeGraphState {
    pub graph_state: SdfGraphState,
    pub id_map: NodeIdMap,
    pub user_state: SdfGraphUserState,
    /// Primary selected node (receives gizmo, properties panel).
    pub selected: Option<SceneNodeId>,
    /// Full set of selected nodes (for multi-select). Always contains `selected` when non-empty.
    pub selected_set: std::collections::HashSet<SceneNodeId>,
    pub layout_dirty: bool,
    pub last_structure_key: u64,
    pub needs_initial_rebuild: bool,
    /// When set, the next frame will center the viewport on this node.
    pub pending_center_node: Option<SceneNodeId>,
    /// Optional graph-space anchor used when placing freshly created nodes.
    pub pending_spawn_anchor: Option<egui::Pos2>,
    /// Latch for ALT+LMB disconnect so it triggers once per press.
    pub alt_disconnect_latch: bool,
}

impl NodeGraphState {
    pub fn new() -> Self {
        Self {
            graph_state: SdfGraphState::new(1.0),
            id_map: NodeIdMap::new(),
            user_state: SdfGraphUserState::new(),
            selected: None,
            selected_set: std::collections::HashSet::new(),
            layout_dirty: true,
            last_structure_key: 0,
            needs_initial_rebuild: true,
            pending_center_node: None,
            pending_spawn_anchor: None,
            alt_disconnect_latch: false,
        }
    }

    /// Select a single node, clearing any previous multi-selection.
    pub fn select_single(&mut self, id: SceneNodeId) {
        self.selected = Some(id);
        self.selected_set.clear();
        self.selected_set.insert(id);
    }

    /// Toggle a node in/out of the selection set (Ctrl+click).
    pub fn toggle_select(&mut self, id: SceneNodeId) {
        if self.selected_set.remove(&id) {
            // Was selected - deselected. Update primary if needed.
            if self.selected == Some(id) {
                self.selected = self.selected_set.iter().next().copied();
            }
        } else {
            // Wasn't selected - add it and make it primary.
            self.selected_set.insert(id);
            self.selected = Some(id);
        }
    }

    /// Add a node to the selection set without removing others.
    #[allow(dead_code)]
    pub fn add_to_selection(&mut self, id: SceneNodeId) {
        self.selected_set.insert(id);
        self.selected = Some(id);
    }

    /// Clear all selection.
    pub fn clear_selection(&mut self) {
        self.selected = None;
        self.selected_set.clear();
    }

    /// Check if a specific node is in the selection set.
    #[allow(dead_code)]
    pub fn is_selected(&self, id: SceneNodeId) -> bool {
        self.selected_set.contains(&id)
    }

    /// Number of nodes currently selected.
    #[allow(dead_code)]
    pub fn selected_count(&self) -> usize {
        self.selected_set.len()
    }
}

#[cfg(test)]
mod multi_select_tests {
    use super::NodeGraphState;

    #[test]
    fn select_single_sets_primary_and_set() {
        let mut state = NodeGraphState::new();
        state.select_single(5);
        assert_eq!(state.selected, Some(5));
        assert_eq!(state.selected_count(), 1);
        assert!(state.is_selected(5));
    }

    #[test]
    fn select_single_replaces_previous() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.select_single(2);
        assert_eq!(state.selected, Some(2));
        assert_eq!(state.selected_count(), 1);
        assert!(!state.is_selected(1));
        assert!(state.is_selected(2));
    }

    #[test]
    fn toggle_select_adds_new_node() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.toggle_select(2);
        assert_eq!(state.selected, Some(2));
        assert_eq!(state.selected_count(), 2);
        assert!(state.is_selected(1));
        assert!(state.is_selected(2));
    }

    #[test]
    fn toggle_select_removes_existing_node() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.toggle_select(2);
        state.toggle_select(1); // remove node 1
        assert_eq!(state.selected_count(), 1);
        assert!(!state.is_selected(1));
        assert!(state.is_selected(2));
    }

    #[test]
    fn toggle_select_updates_primary_when_removing_primary() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.toggle_select(2); // primary = 2
        state.toggle_select(2); // remove primary
        assert_eq!(state.selected_count(), 1);
        // Primary should fall back to remaining node
        assert_eq!(state.selected, Some(1));
    }

    #[test]
    fn add_to_selection_keeps_existing() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.add_to_selection(2);
        state.add_to_selection(3);
        assert_eq!(state.selected, Some(3)); // most recently added
        assert_eq!(state.selected_count(), 3);
        assert!(state.is_selected(1));
        assert!(state.is_selected(2));
        assert!(state.is_selected(3));
    }

    #[test]
    fn clear_selection_empties_everything() {
        let mut state = NodeGraphState::new();
        state.select_single(1);
        state.toggle_select(2);
        state.clear_selection();
        assert_eq!(state.selected, None);
        assert_eq!(state.selected_count(), 0);
        assert!(!state.is_selected(1));
        assert!(!state.is_selected(2));
    }

    #[test]
    fn is_selected_returns_false_for_empty() {
        let state = NodeGraphState::new();
        assert!(!state.is_selected(0));
        assert!(!state.is_selected(42));
    }

    #[test]
    fn selected_count_zero_initially() {
        let state = NodeGraphState::new();
        assert_eq!(state.selected_count(), 0);
    }

    #[test]
    fn graph_editor_instance_ids_are_unique() {
        let sdf_state = NodeGraphState::new();
        let light_state = NodeGraphState::new();
        assert_ne!(sdf_state.graph_state.instance_id, 0);
        assert_ne!(light_state.graph_state.instance_id, 0);
        assert_ne!(
            sdf_state.graph_state.instance_id,
            light_state.graph_state.instance_id
        );
    }
}
#[cfg(test)]
mod auto_insert_tests {
    use super::*;
    use egui_node_graph2::GraphResponse;

    fn build_graph_state(scene: &Scene) -> NodeGraphState {
        let mut state = NodeGraphState::new();
        rebuild_graph_from_scene_with_filter(
            scene,
            &mut state.graph_state,
            &mut state.id_map,
            GraphFilterMode::SdfOnly,
        );
        state
    }

    #[test]
    fn auto_insert_rewires_modifier_between_connected_nodes() {
        let mut scene = Scene::new();
        let source = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let target = scene.create_operation(CsgOp::Union, Some(source), Some(right));
        let insert = scene.create_modifier(ModifierKind::Round, None);

        let mut state = build_graph_state(&scene);
        let source_graph = *state.id_map.scene_to_graph.get(&source).unwrap();
        let target_graph = *state.id_map.scene_to_graph.get(&target).unwrap();
        let insert_graph = *state.id_map.scene_to_graph.get(&insert).unwrap();

        let target_input = state.graph_state.graph[target_graph]
            .get_input("left")
            .unwrap();
        let source_output = state.graph_state.graph[source_graph]
            .get_output("out")
            .unwrap();

        let mut responses: GraphResponse<SdfResponse, SdfNodeData> = GraphResponse::default();
        responses.connection_under_cursor = Some((target_input, source_output));

        let rewired = try_auto_insert_candidates_on_hovered_connection(
            &mut scene,
            &mut state,
            &responses,
            &[insert_graph],
        );
        assert!(rewired);

        match &scene.nodes.get(&insert).unwrap().data {
            NodeData::Modifier { input, .. } => assert_eq!(*input, Some(source)),
            _ => panic!("expected inserted node to be a modifier"),
        }
        match &scene.nodes.get(&target).unwrap().data {
            NodeData::Operation { left, right: _, .. } => assert_eq!(*left, Some(insert)),
            _ => panic!("expected target node to be an operation"),
        }
        assert!(state.needs_initial_rebuild);
    }

    #[test]
    fn auto_insert_rejects_non_reroute_transform() {
        let mut scene = Scene::new();
        let source = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let target = scene.create_operation(CsgOp::Union, Some(source), Some(right));
        let plain_transform = scene.create_transform(None);

        let mut state = build_graph_state(&scene);
        let source_graph = *state.id_map.scene_to_graph.get(&source).unwrap();
        let target_graph = *state.id_map.scene_to_graph.get(&target).unwrap();
        let transform_graph = *state.id_map.scene_to_graph.get(&plain_transform).unwrap();

        let target_input = state.graph_state.graph[target_graph]
            .get_input("left")
            .unwrap();
        let source_output = state.graph_state.graph[source_graph]
            .get_output("out")
            .unwrap();

        let mut responses: GraphResponse<SdfResponse, SdfNodeData> = GraphResponse::default();
        responses.connection_under_cursor = Some((target_input, source_output));

        let rewired = try_auto_insert_candidates_on_hovered_connection(
            &mut scene,
            &mut state,
            &responses,
            &[transform_graph],
        );
        assert!(!rewired);
    }

    #[test]
    fn auto_insert_accepts_reroute_transform() {
        let mut scene = Scene::new();
        let source = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let target = scene.create_operation(CsgOp::Union, Some(source), Some(right));
        let reroute = scene.create_reroute(None);

        let mut state = build_graph_state(&scene);
        let source_graph = *state.id_map.scene_to_graph.get(&source).unwrap();
        let target_graph = *state.id_map.scene_to_graph.get(&target).unwrap();
        let reroute_graph = *state.id_map.scene_to_graph.get(&reroute).unwrap();

        let target_input = state.graph_state.graph[target_graph]
            .get_input("left")
            .unwrap();
        let source_output = state.graph_state.graph[source_graph]
            .get_output("out")
            .unwrap();

        let mut responses: GraphResponse<SdfResponse, SdfNodeData> = GraphResponse::default();
        responses.connection_under_cursor = Some((target_input, source_output));

        let rewired = try_auto_insert_candidates_on_hovered_connection(
            &mut scene,
            &mut state,
            &responses,
            &[reroute_graph],
        );
        assert!(rewired);

        match &scene.nodes.get(&reroute).unwrap().data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(source)),
            _ => panic!("expected inserted node to be a transform"),
        }
        match &scene.nodes.get(&target).unwrap().data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(reroute)),
            _ => panic!("expected target node to be an operation"),
        }
    }

    #[test]
    fn auto_insert_rejects_already_connected_candidate() {
        let mut scene = Scene::new();
        let source = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let target = scene.create_operation(CsgOp::Union, Some(source), Some(right));
        let connected_modifier = scene.create_modifier(ModifierKind::Round, Some(source));

        let mut state = build_graph_state(&scene);
        let source_graph = *state.id_map.scene_to_graph.get(&source).unwrap();
        let target_graph = *state.id_map.scene_to_graph.get(&target).unwrap();
        let modifier_graph = *state
            .id_map
            .scene_to_graph
            .get(&connected_modifier)
            .unwrap();

        let target_input = state.graph_state.graph[target_graph]
            .get_input("left")
            .unwrap();
        let source_output = state.graph_state.graph[source_graph]
            .get_output("out")
            .unwrap();

        let mut responses: GraphResponse<SdfResponse, SdfNodeData> = GraphResponse::default();
        responses.connection_under_cursor = Some((target_input, source_output));

        let rewired = try_auto_insert_candidates_on_hovered_connection(
            &mut scene,
            &mut state,
            &responses,
            &[modifier_graph],
        );
        assert!(!rewired);

        match &scene.nodes.get(&target).unwrap().data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(source)),
            _ => panic!("expected target node to be an operation"),
        }
    }
}

#[cfg(test)]
mod layout_and_disconnect_tests {
    use super::*;

    fn empty_scene() -> Scene {
        let mut scene = Scene::new();
        scene.nodes.clear();
        scene.next_id = 0;
        scene.name_counters.clear();
        scene.hidden_nodes.clear();
        scene.light_masks.clear();
        scene
    }

    fn build_filtered_state(scene: &Scene, filter: GraphFilterMode) -> NodeGraphState {
        let mut state = NodeGraphState::new();
        rebuild_graph_from_scene_with_filter(
            scene,
            &mut state.graph_state,
            &mut state.id_map,
            filter,
        );
        state
    }

    #[test]
    fn place_unpositioned_nodes_avoids_overlap() {
        let mut scene = empty_scene();
        let _a = scene.create_primitive(SdfPrimitive::Sphere);
        let _b = scene.create_primitive(SdfPrimitive::Box);
        let _c = scene.create_primitive(SdfPrimitive::Plane);

        let mut state = build_filtered_state(&scene, GraphFilterMode::SdfOnly);
        place_unpositioned_nodes(
            &scene,
            &mut state.graph_state,
            &state.id_map,
            egui::Pos2::new(0.0, 0.0),
        );

        let mut rects = Vec::new();
        for (&sid, &gid) in &state.id_map.scene_to_graph {
            let pos = *state
                .graph_state
                .node_positions
                .get(gid)
                .expect("position should be assigned");
            rects.push(egui::Rect::from_min_size(
                pos,
                node_size_for_scene(&scene, sid),
            ));
        }

        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(!rects[i]
                    .expand(NODE_PADDING)
                    .intersects(rects[j].expand(NODE_PADDING)));
            }
        }
    }

    #[test]
    fn organize_graph_flows_left_to_right_by_dependencies() {
        let mut scene = empty_scene();
        let a = scene.create_primitive(SdfPrimitive::Sphere);
        let b = scene.create_modifier(ModifierKind::Round, Some(a));
        let c = scene.create_transform(Some(b));

        let mut state = build_filtered_state(&scene, GraphFilterMode::SdfOnly);
        organize_graph(&scene, &mut state.graph_state, &state.id_map);

        let pa = state.graph_state.node_positions[*state.id_map.scene_to_graph.get(&a).unwrap()];
        let pb = state.graph_state.node_positions[*state.id_map.scene_to_graph.get(&b).unwrap()];
        let pc = state.graph_state.node_positions[*state.id_map.scene_to_graph.get(&c).unwrap()];

        assert!(pa.x < pb.x);
        assert!(pb.x < pc.x);
    }

    #[test]
    fn disconnect_node_with_bypass_reconnects_chain() {
        let mut scene = empty_scene();
        let a = scene.create_primitive(SdfPrimitive::Sphere);
        let b = scene.create_modifier(ModifierKind::Round, Some(a));
        let c = scene.create_transform(Some(b));

        let state = build_filtered_state(&scene, GraphFilterMode::SdfOnly);
        let changed = disconnect_node_with_bypass(&mut scene, &state.id_map, b);
        assert!(changed);

        match &scene.nodes.get(&c).unwrap().data {
            NodeData::Transform { input, .. } => assert_eq!(input, &Some(a)),
            _ => panic!("expected transform node"),
        }
        match &scene.nodes.get(&b).unwrap().data {
            NodeData::Modifier { input, .. } => assert_eq!(input, &None),
            _ => panic!("expected modifier node"),
        }
    }
}
pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    state: &mut NodeGraphState,
    actions: &mut ActionSink,
) {
    ui.push_id("sdf_node_graph_panel", |ui| {
        draw_with_filter(ui, scene, state, actions, GraphFilterMode::SdfOnly);
    });
}

pub fn draw_lights(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    state: &mut NodeGraphState,
    selected: &mut Option<SceneNodeId>,
    selected_set: &mut HashSet<SceneNodeId>,
    actions: &mut ActionSink,
) {
    let light_related = light_related_scene_ids(scene);
    if let Some(sel) = *selected {
        if light_related.contains(&sel) {
            state.selected = Some(sel);
            state.selected_set = selected_set
                .iter()
                .copied()
                .filter(|id| light_related.contains(id))
                .collect();
            if state.selected_set.is_empty() {
                state.selected_set.insert(sel);
            }
        }
    }

    let prev_local_selected = state.selected;
    let prev_local_set = state.selected_set.clone();
    ui.push_id("light_node_graph_panel", |ui| {
        draw_with_filter(ui, scene, state, actions, GraphFilterMode::LightsOnly);
    });
    if state.selected != prev_local_selected || state.selected_set != prev_local_set {
        *selected = state.selected;
        *selected_set = state.selected_set.clone();
    }
}

fn draw_with_filter(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    state: &mut NodeGraphState,
    actions: &mut ActionSink,
    filter_mode: GraphFilterMode,
) {
    // Peek at graph_rect for toolbar's Organize button (toolbar is drawn first,
    // but the rect doesn't change between toolbar and graph area).
    let full_rect = ui.available_rect_before_wrap();

    // Draw toolbar above the graph
    draw_toolbar(ui, scene, state, full_rect, actions, filter_mode);

    // Draw the graph editor in remaining space (below toolbar)
    let graph_rect = ui.available_rect_before_wrap();

    // Detect if scene changed externally (undo/redo, load, scene tree edit)
    let structure_key = scene.structure_key();
    if state.needs_initial_rebuild || structure_key != state.last_structure_key {
        match filter_mode {
            GraphFilterMode::SdfOnly => {
                rebuild_graph_from_scene(scene, &mut state.graph_state, &mut state.id_map);
            }
            GraphFilterMode::LightsOnly => {
                rebuild_graph_from_scene_with_filter(
                    scene,
                    &mut state.graph_state,
                    &mut state.id_map,
                    filter_mode,
                );
            }
        }
        state.last_structure_key = structure_key;
        state.needs_initial_rebuild = false;
        state.layout_dirty = false;

        // Restore selection in graph
        if let Some(sel) = state.selected {
            if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sel) {
                state.graph_state.selected_nodes = vec![graph_id];
            }
        }
        let spawn_anchor = state
            .pending_spawn_anchor
            .take()
            .unwrap_or_else(|| viewport_center_graph_space(graph_rect, &state.graph_state));
        place_unpositioned_nodes(scene, &mut state.graph_state, &state.id_map, spawn_anchor);
    }

    // Snapshot node data for inline editing
    state.user_state.node_data_snapshot.clear();
    state.user_state.dirty_nodes.clear();
    state.user_state.created_via_finder.clear();
    for (sid, node) in &scene.nodes {
        state
            .user_state
            .node_data_snapshot
            .insert(*sid, node.data.clone());
    }
    state.user_state.zoom = state.graph_state.pan_zoom.zoom;
    let responses = ui
        .allocate_ui(graph_rect.size(), |ui| match filter_mode {
            GraphFilterMode::SdfOnly => state.graph_state.draw_graph_editor(
                ui,
                AllSdfTemplates,
                &mut state.user_state,
                vec![],
            ),
            GraphFilterMode::LightsOnly => state.graph_state.draw_graph_editor(
                ui,
                LightSdfTemplates,
                &mut state.user_state,
                vec![],
            ),
        })
        .inner;

    let mut auto_insert_candidates: Vec<NodeId> = responses
        .node_responses
        .iter()
        .filter_map(|r| match r {
            NodeResponse::MoveNodeEnded(id) => Some(*id),
            _ => None,
        })
        .collect();

    // Handle nodes created via the node finder
    let created: Vec<_> = state.user_state.created_via_finder.drain(..).collect();
    for (graph_id, template) in created {
        let is_light = matches!(template, SdfNodeTemplate::Light(_));
        let is_auto_insert_candidate = matches!(
            template,
            SdfNodeTemplate::Modifier(_) | SdfNodeTemplate::Transform | SdfNodeTemplate::Reroute
        );
        let scene_id = match template {
            SdfNodeTemplate::Primitive(kind) => scene.create_primitive(kind),
            SdfNodeTemplate::Operation(op) => scene.create_operation(op, None, None),
            SdfNodeTemplate::Transform => scene.create_transform(None),
            SdfNodeTemplate::Modifier(kind) => scene.create_modifier(kind, None),
            SdfNodeTemplate::Reroute => scene.create_reroute(None),
            SdfNodeTemplate::Light(light_type) => {
                // create_light returns (light_id, transform_id) - select transform
                let (_light_id, transform_id) = scene.create_light(light_type);
                transform_id
            }
        };
        if is_light {
            // Light creation adds both Light + Transform nodes to the scene,
            // so a full graph rebuild is needed to show both nodes correctly.
            if let Some(pos) = state.graph_state.node_positions.get(graph_id) {
                state.pending_spawn_anchor = Some(*pos);
            }
            state.needs_initial_rebuild = true;
            state.pending_center_node = Some(scene_id);
        } else {
            // Update the graph node's user_data with the real scene id
            state.graph_state.graph[graph_id].user_data.scene_node_id = scene_id;
            // Update the label to match the scene node name
            if let Some(node) = scene.nodes.get(&scene_id) {
                state.graph_state.graph[graph_id].label = node.name.clone();
            }
            state.id_map.insert(scene_id, graph_id);
            if is_auto_insert_candidate {
                auto_insert_candidates.push(graph_id);
            }
        }
        state.select_single(scene_id);
        state.last_structure_key = scene.structure_key();
    }

    // Write back dirty node data from inline editing
    for sid in &state.user_state.dirty_nodes {
        if let Some(data) = state.user_state.node_data_snapshot.get(sid) {
            if let Some(node) = scene.nodes.get_mut(sid) {
                node.data = data.clone();
            }
        }
    }

    // Process graph events
    let mut layout_dirty = false;
    process_graph_responses(
        &responses,
        &mut state.graph_state,
        &mut state.id_map,
        &mut state.selected,
        &mut layout_dirty,
        actions,
    );
    if layout_dirty {
        state.layout_dirty = true;
        state.last_structure_key = scene.structure_key();
    }

    if matches!(filter_mode, GraphFilterMode::SdfOnly)
        && !auto_insert_candidates.is_empty()
        && try_auto_insert_candidates_on_hovered_connection(
            scene,
            state,
            &responses,
            &auto_insert_candidates,
        )
    {
        state.layout_dirty = true;
    }

    // Consume layout_dirty: only place nodes that are still unpositioned.
    if state.layout_dirty {
        let spawn_anchor = state
            .pending_spawn_anchor
            .take()
            .unwrap_or_else(|| viewport_center_graph_space(graph_rect, &state.graph_state));
        place_unpositioned_nodes(scene, &mut state.graph_state, &state.id_map, spawn_anchor);
        state.layout_dirty = false;

        // Restore selection in graph
        if let Some(sel) = state.selected {
            if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sel) {
                state.graph_state.selected_nodes = vec![graph_id];
            }
        }
    }

    // Sync selection from graph to our state
    if let Some(first_selected) = state.graph_state.selected_nodes.first() {
        if let Some(&scene_id) = state.id_map.graph_to_scene.get(first_selected) {
            state.select_single(scene_id);
        }
    } else if responses.cursor_in_editor && !responses.cursor_in_finder {
        // User clicked empty space in the graph.
        // Don't clear selection for nodes that are intentionally filtered out in this view.
        let selected_is_visible = state
            .selected
            .and_then(|sid| state.id_map.scene_to_graph.get(&sid).copied())
            .is_some();
        if state.graph_state.selected_nodes.is_empty()
            && (selected_is_visible || state.selected.is_none())
        {
            state.clear_selection();
        }
    }

    // ALT + LMB: disconnect node and bypass when possible (A->B->C becomes A->C).
    let (alt_down, primary_down, pointer_pos) = ui.ctx().input(|i| {
        (
            i.modifiers.alt,
            i.pointer.button_down(egui::PointerButton::Primary),
            i.pointer.interact_pos().or_else(|| i.pointer.hover_pos()),
        )
    });

    if !primary_down {
        state.alt_disconnect_latch = false;
    } else if alt_down && !state.alt_disconnect_latch {
        state.alt_disconnect_latch = true;
        if let Some(pointer) = pointer_pos {
            if graph_rect.contains(pointer) {
                if let Some(graph_node_id) = node_under_pointer(scene, state, graph_rect, pointer) {
                    if let Some(&scene_id) = state.id_map.graph_to_scene.get(&graph_node_id) {
                        if disconnect_node_with_bypass(scene, &state.id_map, scene_id) {
                            state.select_single(scene_id);
                            state.needs_initial_rebuild = true;
                            state.last_structure_key = scene.structure_key();
                        }
                    }
                }
            }
        }
    }
    // Focus request (used for create actions and external selection sync).
    if let Some(sid) = state.pending_center_node.take() {
        center_view_on_node(scene, state, graph_rect, sid);
    }

    // Draw minimap overlay in the bottom-right corner
    draw_minimap(ui, graph_rect, scene, state, filter_mode);
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

fn draw_toolbar(
    ui: &mut egui::Ui,
    scene: &Scene,
    state: &mut NodeGraphState,
    graph_rect: egui::Rect,
    actions: &mut ActionSink,
    filter_mode: GraphFilterMode,
) {
    egui::Frame::none()
        .fill(Color32::from_rgb(30, 30, 35))
        .inner_margin(egui::Margin::symmetric(6.0, 4.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let selected_visible = state
                    .selected
                    .filter(|sid| state.id_map.scene_to_graph.contains_key(sid));

                match filter_mode {
                    GraphFilterMode::SdfOnly => {
                        ui.menu_button("+ Primitive", |ui| {
                            for kind in SdfPrimitive::ALL {
                                if ui.button(kind.base_name()).clicked() {
                                    state.pending_spawn_anchor = Some(compute_spawn_anchor(
                                        state,
                                        selected_visible,
                                        graph_rect,
                                        ui.ctx(),
                                    ));
                                    actions.push(Action::CreatePrimitive(kind.clone()));
                                    ui.close_menu();
                                }
                            }
                        });

                        ui.menu_button("+ Operation", |ui| {
                            for op in CsgOp::ALL {
                                if ui.button(op.base_name()).clicked() {
                                    state.pending_spawn_anchor = Some(compute_spawn_anchor(
                                        state,
                                        selected_visible,
                                        graph_rect,
                                        ui.ctx(),
                                    ));
                                    create_op_from_selection_filtered(
                                        scene,
                                        op.clone(),
                                        actions,
                                        |id| state.id_map.scene_to_graph.contains_key(&id),
                                    );
                                    ui.close_menu();
                                }
                            }
                        });

                        ui.menu_button("+ Transform", |ui| {
                            if ui.button("Transform").clicked() {
                                if let Some(sel) = selected_visible {
                                    actions.push(Action::InsertTransformAbove { target: sel });
                                } else {
                                    state.pending_spawn_anchor = Some(compute_spawn_anchor(
                                        state,
                                        selected_visible,
                                        graph_rect,
                                        ui.ctx(),
                                    ));
                                    actions.push(Action::CreateTransform { input: None });
                                }
                                ui.close_menu();
                            }
                        });

                        ui.menu_button("+ Utility", |ui| {
                            if ui.button("Reroute").clicked() {
                                state.pending_spawn_anchor = Some(compute_spawn_anchor(
                                    state,
                                    selected_visible,
                                    graph_rect,
                                    ui.ctx(),
                                ));
                                actions.push(Action::CreateReroute { input: None });
                                ui.close_menu();
                            }
                        });

                        ui.menu_button("+ Modifier", |ui| {
                            ui.label("Deform");
                            for kind in [
                                ModifierKind::Twist,
                                ModifierKind::Bend,
                                ModifierKind::Taper,
                                ModifierKind::Noise,
                            ] {
                                if ui.button(kind.base_name()).clicked() {
                                    toolbar_add_modifier(
                                        state,
                                        selected_visible,
                                        kind,
                                        actions,
                                        graph_rect,
                                        ui.ctx(),
                                    );
                                    ui.close_menu();
                                }
                            }
                            ui.separator();
                            ui.label("Shape");
                            for kind in [
                                ModifierKind::Round,
                                ModifierKind::Onion,
                                ModifierKind::Elongate,
                            ] {
                                if ui.button(kind.base_name()).clicked() {
                                    toolbar_add_modifier(
                                        state,
                                        selected_visible,
                                        kind,
                                        actions,
                                        graph_rect,
                                        ui.ctx(),
                                    );
                                    ui.close_menu();
                                }
                            }
                            ui.separator();
                            ui.label("Repeat");
                            for kind in [
                                ModifierKind::Mirror,
                                ModifierKind::Repeat,
                                ModifierKind::FiniteRepeat,
                            ] {
                                if ui.button(kind.base_name()).clicked() {
                                    toolbar_add_modifier(
                                        state,
                                        selected_visible,
                                        kind,
                                        actions,
                                        graph_rect,
                                        ui.ctx(),
                                    );
                                    ui.close_menu();
                                }
                            }
                        });

                        if ui.button("Load Preset").clicked() {
                            actions.push(Action::LoadNodePreset);
                        }
                    }
                    GraphFilterMode::LightsOnly => {
                        ui.menu_button("+ Light", |ui| {
                            for light_type in LightType::ALL {
                                if ui.button(light_type.label()).clicked() {
                                    state.pending_spawn_anchor = Some(compute_spawn_anchor(
                                        state,
                                        selected_visible,
                                        graph_rect,
                                        ui.ctx(),
                                    ));
                                    actions.push(Action::CreateLight(light_type.clone()));
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                }

                if matches!(filter_mode, GraphFilterMode::SdfOnly) {
                    if ui
                        .add_enabled(selected_visible.is_some(), egui::Button::new("+ Sculpt"))
                        .on_hover_text("Add sculpt modifier to selected node (Ctrl+R)")
                        .clicked()
                    {
                        actions.push(Action::EnterSculptMode);
                    }

                    ui.separator();
                }

                let has_selection = selected_visible.is_some();
                if ui
                    .add_enabled(has_selection, egui::Button::new("Delete"))
                    .clicked()
                {
                    if let Some(sel) = selected_visible {
                        actions.push(Action::DeleteNode(sel));
                    }
                }

                ui.separator();

                if ui
                    .add_enabled(has_selection, egui::Button::new("Focus Selected"))
                    .clicked()
                {
                    if let Some(sel) = selected_visible {
                        center_view_on_node(scene, state, graph_rect, sel);
                    }
                }

                if ui.button("Organize").clicked() {
                    organize_graph(scene, &mut state.graph_state, &state.id_map);
                    center_view_on_nodes(state, graph_rect);
                }
            });
        });
}
// ---------------------------------------------------------------------------
// Minimap
// ---------------------------------------------------------------------------

const MINIMAP_W: f32 = 180.0;
const MINIMAP_H: f32 = 130.0;
const MINIMAP_MARGIN: f32 = 8.0;
const MINIMAP_NODE_W: f32 = 18.0;
const MINIMAP_NODE_H: f32 = 10.0;
const MINIMAP_BG: Color32 = Color32::from_rgba_premultiplied(20, 20, 25, 200);
const MINIMAP_VIEWPORT_COLOR: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 18);
const MINIMAP_VIEWPORT_STROKE: Color32 = Color32::from_rgba_premultiplied(180, 180, 190, 100);
const MINIMAP_VIEWPORT_ROUNDING: f32 = 4.0;

fn draw_minimap(
    ui: &mut egui::Ui,
    graph_rect: egui::Rect,
    scene: &Scene,
    state: &mut NodeGraphState,
    filter_mode: GraphFilterMode,
) {
    let node_positions = &state.graph_state.node_positions;
    if node_positions.is_empty() {
        return;
    }

    let pan = state.graph_state.pan_zoom.pan;
    let zoom = state.graph_state.pan_zoom.zoom;

    // Collect node positions and their badge colors
    let mut nodes_info: Vec<(egui::Pos2, Color32, bool)> = Vec::new();
    let selected_set: HashSet<_> = state.graph_state.selected_nodes.iter().copied().collect();

    for node_id in state.graph_state.node_order.iter().copied() {
        if let Some(&pos) = node_positions.get(node_id) {
            let color = state
                .graph_state
                .graph
                .nodes
                .get(node_id)
                .and_then(|n| {
                    scene
                        .nodes
                        .get(&n.user_data.scene_node_id)
                        .map(|sn| match &sn.data {
                            NodeData::Primitive { .. } => COLOR_PRIM_BADGE,
                            NodeData::Operation { .. } => COLOR_OP_BADGE,
                            NodeData::Sculpt { .. } => COLOR_SCULPT_BADGE,
                            NodeData::Transform { .. } => COLOR_TRANSFORM_BADGE,
                            NodeData::Modifier { .. } => COLOR_MODIFIER_BADGE,
                            NodeData::Light { .. } => COLOR_LIGHT_BADGE,
                        })
                })
                .unwrap_or(Color32::GRAY);
            let is_selected = selected_set.contains(&node_id);
            nodes_info.push((pos, color, is_selected));
        }
    }

    if nodes_info.is_empty() {
        return;
    }

    // Compute bounding box of all nodes in graph space
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(pos, _, _) in &nodes_info {
        min_x = min_x.min(pos.x);
        min_y = min_y.min(pos.y);
        max_x = max_x.max(pos.x + NODE_WIDTH);
        max_y = max_y.max(pos.y + 60.0); // approximate node height
    }

    // Include viewport bounds in the world bounding box
    let vp_min_x = -pan.x / zoom;
    let vp_min_y = -pan.y / zoom;
    let vp_max_x = (graph_rect.width() - pan.x) / zoom;
    let vp_max_y = (graph_rect.height() - pan.y) / zoom;
    min_x = min_x.min(vp_min_x);
    min_y = min_y.min(vp_min_y);
    max_x = max_x.max(vp_max_x);
    max_y = max_y.max(vp_max_y);

    // Add padding
    let pad = 40.0;
    min_x -= pad;
    min_y -= pad;
    max_x += pad;
    max_y += pad;

    let world_w = (max_x - min_x).max(1.0);
    let world_h = (max_y - min_y).max(1.0);

    // Compute scale to fit world into minimap
    let inner_w = MINIMAP_W - 8.0;
    let inner_h = MINIMAP_H - 8.0;
    let scale = (inner_w / world_w).min(inner_h / world_h);

    // Minimap position: bottom-right of graph area
    let minimap_pos = egui::Pos2::new(
        graph_rect.right() - MINIMAP_W - MINIMAP_MARGIN,
        graph_rect.bottom() - MINIMAP_H - MINIMAP_MARGIN,
    );

    // Use a foreground Area so the minimap floats above the graph and captures input
    let minimap_id = match filter_mode {
        GraphFilterMode::SdfOnly => "node_graph_minimap_sdf",
        GraphFilterMode::LightsOnly => "node_graph_minimap_lights",
    };
    let area_resp = egui::Area::new(egui::Id::new(minimap_id))
        .order(egui::Order::Foreground)
        .fixed_pos(minimap_pos)
        .interactable(true)
        .show(ui.ctx(), |ui| {
            let (minimap_rect, minimap_response) = ui.allocate_exact_size(
                egui::Vec2::new(MINIMAP_W, MINIMAP_H),
                egui::Sense::click_and_drag(),
            );

            let painter = ui.painter();

            // Background
            painter.rect_filled(minimap_rect, 4.0, MINIMAP_BG);
            painter.rect_stroke(
                minimap_rect,
                4.0,
                egui::Stroke::new(1.0, Color32::from_rgb(60, 60, 65)),
            );

            let inner_origin = minimap_rect.min + egui::Vec2::new(4.0, 4.0);

            // Center the content within the minimap
            let content_w = world_w * scale;
            let content_h = world_h * scale;
            let offset_x = (inner_w - content_w) * 0.5;
            let offset_y = (inner_h - content_h) * 0.5;
            let origin = inner_origin + egui::Vec2::new(offset_x, offset_y);

            // Helper: world pos to minimap pos
            let to_minimap = |wx: f32, wy: f32| -> egui::Pos2 {
                egui::Pos2::new(
                    origin.x + (wx - min_x) * scale,
                    origin.y + (wy - min_y) * scale,
                )
            };

            // Draw viewport rectangle with rounded corners
            let vp_tl = to_minimap(vp_min_x, vp_min_y);
            let vp_br = to_minimap(vp_max_x, vp_max_y);
            let vp_rect = egui::Rect::from_min_max(vp_tl, vp_br).intersect(minimap_rect);
            painter.rect_filled(vp_rect, MINIMAP_VIEWPORT_ROUNDING, MINIMAP_VIEWPORT_COLOR);
            painter.rect_stroke(
                vp_rect,
                MINIMAP_VIEWPORT_ROUNDING,
                egui::Stroke::new(1.0, MINIMAP_VIEWPORT_STROKE),
            );

            // Draw nodes
            let node_w = MINIMAP_NODE_W.min(NODE_WIDTH * scale);
            let node_h = MINIMAP_NODE_H.min(60.0 * scale);
            for &(pos, color, is_selected) in &nodes_info {
                let tl = to_minimap(pos.x, pos.y);
                let node_rect = egui::Rect::from_min_size(tl, egui::Vec2::new(node_w, node_h));

                if !minimap_rect.intersects(node_rect) {
                    continue;
                }

                painter.rect_filled(node_rect, 2.0, color);
                if is_selected {
                    painter.rect_stroke(node_rect, 2.0, egui::Stroke::new(1.5, Color32::WHITE));
                }
            }

            minimap_response
        });

    let resp = area_resp.inner;
    // Click-to-pan: clicking/dragging on the minimap centers the viewport
    if resp.clicked() || resp.dragged() {
        if let Some(pointer) = ui.ctx().input(|i| i.pointer.interact_pos()) {
            let minimap_rect =
                egui::Rect::from_min_size(minimap_pos, egui::Vec2::new(MINIMAP_W, MINIMAP_H));
            if minimap_rect.contains(pointer) {
                let inner_origin = minimap_rect.min + egui::Vec2::new(4.0, 4.0);
                let content_w = world_w * scale;
                let content_h = world_h * scale;
                let offset_x = (inner_w - content_w) * 0.5;
                let offset_y = (inner_h - content_h) * 0.5;
                let origin = inner_origin + egui::Vec2::new(offset_x, offset_y);
                let rel = pointer - origin;
                let rel_x = rel.x.clamp(0.0, content_w.max(1.0));
                let rel_y = rel.y.clamp(0.0, content_h.max(1.0));
                let world_x = min_x + rel_x / scale;
                let world_y = min_y + rel_y / scale;
                // Target: center viewport directly on clicked/dragged world position.
                let half_vp_w = graph_rect.width() / zoom * 0.5;
                let half_vp_h = graph_rect.height() / zoom * 0.5;
                state.graph_state.pan_zoom.pan =
                    egui::Vec2::new(-(world_x - half_vp_w) * zoom, -(world_y - half_vp_h) * zoom);
                if resp.dragged() {
                    ui.ctx().request_repaint();
                }
            }
        }
    }
}

fn center_view_on_nodes(state: &mut NodeGraphState, graph_rect: egui::Rect) {
    let positions = &state.graph_state.node_positions;
    if positions.is_empty() {
        return;
    }
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for (_, pos) in positions.iter() {
        min_x = min_x.min(pos.x);
        min_y = min_y.min(pos.y);
        max_x = max_x.max(pos.x + NODE_WIDTH);
        max_y = max_y.max(pos.y + 200.0);
    }
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;
    state.graph_state.pan_zoom.zoom = 1.0;
    state.graph_state.pan_zoom.pan = egui::Vec2::new(
        graph_rect.width() / 2.0 - center_x,
        graph_rect.height() / 2.0 - center_y,
    );
}

fn toolbar_add_modifier(
    state: &mut NodeGraphState,
    selected: Option<SceneNodeId>,
    kind: ModifierKind,
    actions: &mut ActionSink,
    graph_rect: egui::Rect,
    ctx: &egui::Context,
) {
    if let Some(sel) = selected {
        actions.push(Action::InsertModifierAbove { target: sel, kind });
    } else {
        state.pending_spawn_anchor = Some(compute_spawn_anchor(state, selected, graph_rect, ctx));
        actions.push(Action::CreateModifier { kind, input: None });
    }
}

fn create_op_from_selection_filtered(
    scene: &Scene,
    op: CsgOp,
    actions: &mut ActionSink,
    include_node: impl Fn(SceneNodeId) -> bool,
) {
    let tops: Vec<_> = scene
        .top_level_nodes()
        .into_iter()
        .filter(|id| include_node(*id))
        .collect();

    let left = if tops.len() >= 2 {
        Some(tops[tops.len() - 2])
    } else {
        None
    };
    let right = tops.last().copied();
    actions.push(Action::CreateOperation { op, left, right });
}

// ---------------------------------------------------------------------------
// Widget helpers
// ---------------------------------------------------------------------------

fn compact_vec3(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut glam::Vec3,
    speed: f32,
    range: Option<std::ops::RangeInclusive<f32>>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.label(egui::RichText::new(label).small());
        for (color, comp) in [
            (Color32::from_rgb(200, 80, 80), &mut value.x),
            (Color32::from_rgb(80, 200, 80), &mut value.y),
            (Color32::from_rgb(80, 80, 200), &mut value.z),
        ] {
            let mut drag = egui::DragValue::new(comp)
                .speed(speed)
                .max_decimals(2)
                .update_while_editing(false);
            if let Some(ref r) = range {
                drag = drag.range(r.clone());
            }
            let response = ui.add(drag);
            if response.changed() {
                changed = true;
            }
            let r = response.rect;
            ui.painter().line_segment(
                [
                    egui::Pos2::new(r.left(), r.bottom()),
                    egui::Pos2::new(r.right(), r.bottom()),
                ],
                egui::Stroke::new(2.0, color),
            );
        }
    });
    changed
}

fn compact_rotation(ui: &mut egui::Ui, label: &str, value: &mut glam::Vec3) -> bool {
    let mut deg = glam::Vec3::new(
        value.x.to_degrees(),
        value.y.to_degrees(),
        value.z.to_degrees(),
    );
    let changed = compact_vec3(ui, label, &mut deg, 1.0, None);
    *value = glam::Vec3::new(deg.x.to_radians(), deg.y.to_radians(), deg.z.to_radians());
    changed
}

fn scalar_drag(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    speed: f32,
    range: Option<std::ops::RangeInclusive<f32>>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).small());
        let mut drag = egui::DragValue::new(value).speed(speed).max_decimals(2);
        if let Some(r) = range {
            drag = drag.range(r);
        }
        if ui.add(drag).changed() {
            changed = true;
        }
    });
    changed
}
