use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Color32};
use egui_node_graph2::*;

use crate::app::actions::{Action, ActionSink};
use crate::graph::scene::{
    CsgOp, ModifierKind, NodeData, NodeId as SceneNodeId, Scene, SdfPrimitive,
};

// ---------------------------------------------------------------------------
// Badge / port colors (matching old theme)
// ---------------------------------------------------------------------------

const COLOR_PRIM_BADGE: Color32 = Color32::from_rgb(70, 130, 180);
const COLOR_OP_BADGE: Color32 = Color32::from_rgb(200, 120, 50);
const COLOR_SCULPT_BADGE: Color32 = Color32::from_rgb(150, 100, 200);
const COLOR_TRANSFORM_BADGE: Color32 = Color32::from_rgb(100, 180, 170);
const COLOR_MODIFIER_BADGE: Color32 = Color32::from_rgb(180, 160, 80);
const COLOR_PORT: Color32 = Color32::from_rgb(100, 200, 100);

const COL_SPACING: f32 = 260.0;
const NODE_WIDTH: f32 = 140.0;

// ---------------------------------------------------------------------------
// Custom types for the graph library
// ---------------------------------------------------------------------------

/// Single data type — all connections carry SDF signals.
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
    /// Placeholder — inline editing is tracked via dirty_nodes in user_state.
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
}
impl CategoryTrait for SdfCategory {
    fn name(&self) -> String {
        match self {
            Self::Primitive => "Primitive".into(),
            Self::Operation => "Operation".into(),
            Self::Transform => "Transform".into(),
            Self::Modifier => "Modifier".into(),
        }
    }
}

/// Node templates — one per creatable node variant.
#[derive(Clone, Debug)]
pub enum SdfNodeTemplate {
    Primitive(SdfPrimitive),
    Operation(CsgOp),
    Transform,
    Modifier(ModifierKind),
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
                ref mut color,
                ref mut metallic,
                ref mut roughness,
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

                let mut c = [color.x, color.y, color.z];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Col").small());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        changed = true;
                    }
                });
                *color = glam::Vec3::new(c[0], c[1], c[2]);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Met").small());
                    if ui
                        .add(
                            egui::DragValue::new(metallic)
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
                            egui::DragValue::new(roughness)
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
                    ModifierKind::Repeat => {
                        compact_vec3(ui, "Space", value, 0.1, Some(0.0..=20.0))
                    }
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
                ref mut color,
                ref mut metallic,
                ref mut roughness,
                ..
            } => {
                changed |= compact_vec3(ui, "Pos", position, 0.05, None);
                changed |= compact_rotation(ui, "Rot", rotation);

                let mut c = [color.x, color.y, color.z];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Col").small());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        changed = true;
                    }
                });
                *color = glam::Vec3::new(c[0], c[1], c[2]);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Met").small());
                    if ui
                        .add(
                            egui::DragValue::new(metallic)
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
                            egui::DragValue::new(roughness)
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
        }
    }

    fn node_finder_categories(
        &self,
        _user_state: &mut Self::UserState,
    ) -> Vec<Self::CategoryType> {
        match self {
            Self::Primitive(_) => vec![SdfCategory::Primitive],
            Self::Operation(_) => vec![SdfCategory::Operation],
            Self::Transform => vec![SdfCategory::Transform],
            Self::Modifier(_) => vec![SdfCategory::Modifier],
        }
    }

    fn node_graph_label(&self, _user_state: &mut Self::UserState) -> String {
        match self {
            Self::Primitive(p) => p.base_name().to_string(),
            Self::Operation(o) => o.base_name().to_string(),
            Self::Transform => "Transform".to_string(),
            Self::Modifier(m) => m.base_name().to_string(),
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
            Self::Primitive(_) => {
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
            Self::Transform | Self::Modifier(_) => {
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
        user_state
            .created_via_finder
            .push((node_id, self.clone()));
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
        for m in ModifierKind::ALL {
            templates.push(SdfNodeTemplate::Modifier(m.clone()));
        }
        templates
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

pub type SdfGraphState =
    GraphEditorState<SdfNodeData, SdfDataType, SdfValueType, SdfNodeTemplate, SdfGraphUserState>;

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

pub fn rebuild_graph_from_scene(
    scene: &Scene,
    graph_state: &mut SdfGraphState,
    id_map: &mut NodeIdMap,
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
        let Some(node) = scene.nodes.get(&sid) else {
            continue;
        };
        let user_data = SdfNodeData {
            scene_node_id: sid,
        };
        let label = node.name.clone();

        let graph_node_id = graph_state.graph.add_node(label, user_data, |graph, node_id| {
            // Add ports based on node type
            match &node.data {
                NodeData::Primitive { .. } => {
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

    // Auto-layout nodes that don't have saved positions
    auto_layout_graph(scene, graph_state, id_map);
}

// ---------------------------------------------------------------------------
// Auto-layout (depth-based columnar)
// ---------------------------------------------------------------------------

fn compute_depth(scene: &Scene, id: SceneNodeId, cache: &mut HashMap<SceneNodeId, u32>) -> u32 {
    if let Some(&d) = cache.get(&id) {
        return d;
    }
    let depth = match scene.nodes.get(&id).map(|n| &n.data) {
        Some(NodeData::Operation { left, right, .. }) => {
            let ld = left.map_or(0, |l| compute_depth(scene, l, cache));
            let rd = right.map_or(0, |r| compute_depth(scene, r, cache));
            1 + ld.max(rd)
        }
        Some(NodeData::Sculpt { input, .. })
        | Some(NodeData::Transform { input, .. })
        | Some(NodeData::Modifier { input, .. }) => {
            1 + input.map_or(0, |i| compute_depth(scene, i, cache))
        }
        _ => 0,
    };
    cache.insert(id, depth);
    depth
}

fn auto_layout_graph(scene: &Scene, graph_state: &mut SdfGraphState, id_map: &NodeIdMap) {
    let mut depth_cache: HashMap<SceneNodeId, u32> = HashMap::new();
    for &id in scene.nodes.keys() {
        compute_depth(scene, id, &mut depth_cache);
    }

    let mut columns: HashMap<u32, Vec<SceneNodeId>> = HashMap::new();
    for (&id, &col) in &depth_cache {
        columns.entry(col).or_default().push(id);
    }
    for col in columns.values_mut() {
        col.sort();
    }

    for (&col, nodes) in &columns {
        let mut y = 0.0f32;
        for &sid in nodes {
            if let Some(&graph_id) = id_map.scene_to_graph.get(&sid) {
                // Only set position if not already set (from saved positions)
                if !graph_state.node_positions.contains_key(graph_id) {
                    let x = col as f32 * COL_SPACING;
                    graph_state
                        .node_positions
                        .insert(graph_id, egui::Pos2::new(x, y));
                }
            }
            y += 200.0; // Default row spacing
        }
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
            NodeResponse::ConnectEventEnded {
                output, input, ..
            } => {
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
                    Some("left") => actions.push(Action::SetLeftChild { parent: target_scene_id, child: Some(source_scene_id) }),
                    Some("right") => actions.push(Action::SetRightChild { parent: target_scene_id, child: Some(source_scene_id) }),
                    Some("input") => actions.push(Action::SetSculptInput { parent: target_scene_id, child: Some(source_scene_id) }),
                    _ => {}
                }
                *layout_dirty = true;
            }
            NodeResponse::DisconnectEvent { input, .. } => {
                let input_graph_node = graph_state.graph.get_input(*input).node;
                let Some(&target_scene_id) = id_map.graph_to_scene.get(&input_graph_node) else {
                    continue;
                };

                let input_name = graph_state.graph[input_graph_node]
                    .inputs
                    .iter()
                    .find(|(_, iid)| *iid == *input)
                    .map(|(name, _)| name.as_str());

                match input_name {
                    Some("left") => actions.push(Action::SetLeftChild { parent: target_scene_id, child: None }),
                    Some("right") => actions.push(Action::SetRightChild { parent: target_scene_id, child: None }),
                    Some("input") => actions.push(Action::SetSculptInput { parent: target_scene_id, child: None }),
                    _ => {}
                }
                *layout_dirty = true;
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

// ---------------------------------------------------------------------------
// Public state & draw
// ---------------------------------------------------------------------------

pub struct NodeGraphState {
    pub graph_state: SdfGraphState,
    pub id_map: NodeIdMap,
    pub user_state: SdfGraphUserState,
    pub selected: Option<SceneNodeId>,
    pub layout_dirty: bool,
    pub last_structure_key: u64,
    pub needs_initial_rebuild: bool,
    /// When set, the next frame will place this node at the viewport center.
    pub pending_center_node: Option<SceneNodeId>,
}

impl NodeGraphState {
    pub fn new() -> Self {
        Self {
            graph_state: SdfGraphState::new(1.0),
            id_map: NodeIdMap::new(),
            user_state: SdfGraphUserState::new(),
            selected: None,
            layout_dirty: true,
            last_structure_key: 0,
            needs_initial_rebuild: true,
            pending_center_node: None,
        }
    }
}

pub fn draw(ui: &mut egui::Ui, scene: &mut Scene, state: &mut NodeGraphState, actions: &mut ActionSink) {
    // Peek at graph_rect for toolbar's Organize button (toolbar is drawn first,
    // but the rect doesn't change between toolbar and graph area).
    let full_rect = ui.available_rect_before_wrap();

    // Draw toolbar above the graph
    draw_toolbar(ui, scene, state, full_rect, actions);

    // Detect if scene changed externally (undo/redo, load, scene tree edit)
    let structure_key = scene.structure_key();
    if state.needs_initial_rebuild || structure_key != state.last_structure_key {
        rebuild_graph_from_scene(scene, &mut state.graph_state, &mut state.id_map);
        state.last_structure_key = structure_key;
        state.needs_initial_rebuild = false;
        state.layout_dirty = false;

        // Restore selection in graph
        if let Some(sel) = state.selected {
            if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sel) {
                state.graph_state.selected_nodes = vec![graph_id];
            }
        }
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

    // Draw the graph editor in remaining space (below toolbar)
    let graph_rect = ui.available_rect_before_wrap();

    // Place newly created nodes at the viewport center
    if let Some(sid) = state.pending_center_node.take() {
        if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sid) {
            let pan = state.graph_state.pan_zoom.pan;
            let zoom = state.graph_state.pan_zoom.zoom;
            let center = egui::Pos2::new(
                (graph_rect.width() / 2.0 - pan.x) / zoom,
                (graph_rect.height() / 2.0 - pan.y) / zoom,
            );
            state.graph_state.node_positions.insert(graph_id, center);
        }
    }

    state.user_state.zoom = state.graph_state.pan_zoom.zoom;
    let responses = ui
        .allocate_ui(graph_rect.size(), |ui| {
            state.graph_state.draw_graph_editor(
                ui,
                AllSdfTemplates,
                &mut state.user_state,
                vec![],
            )
        })
        .inner;

    // Handle nodes created via the node finder
    let created: Vec<_> = state.user_state.created_via_finder.drain(..).collect();
    for (graph_id, template) in created {
        let scene_id = match template {
            SdfNodeTemplate::Primitive(kind) => scene.create_primitive(kind),
            SdfNodeTemplate::Operation(op) => scene.create_operation(op, None, None),
            SdfNodeTemplate::Transform => scene.create_transform(None),
            SdfNodeTemplate::Modifier(kind) => scene.create_modifier(kind, None),
        };
        // Update the graph node's user_data with the real scene id
        state.graph_state.graph[graph_id].user_data.scene_node_id = scene_id;
        // Update the label to match the scene node name
        if let Some(node) = scene.nodes.get(&scene_id) {
            state.graph_state.graph[graph_id].label = node.name.clone();
        }
        state.id_map.insert(scene_id, graph_id);
        state.selected = Some(scene_id);
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
    // Consume layout_dirty: run auto-layout for any new/repositioned nodes
    if state.layout_dirty {
        auto_layout_graph(scene, &mut state.graph_state, &state.id_map);
        state.layout_dirty = false;
    }

    // Sync selection from graph to our state
    if let Some(first_selected) = state.graph_state.selected_nodes.first() {
        if let Some(&scene_id) = state.id_map.graph_to_scene.get(first_selected) {
            state.selected = Some(scene_id);
        }
    } else if responses.cursor_in_editor && !responses.cursor_in_finder {
        // User clicked empty space in the graph
        // Only deselect if no nodes are selected and cursor is in editor
        if state.graph_state.selected_nodes.is_empty() {
            state.selected = None;
        }
    }

    // Draw minimap overlay in the bottom-right corner
    draw_minimap(ui, graph_rect, scene, state);
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

fn draw_toolbar(ui: &mut egui::Ui, scene: &Scene, state: &mut NodeGraphState, graph_rect: egui::Rect, actions: &mut ActionSink) {
    egui::Frame::none()
        .fill(Color32::from_rgb(30, 30, 35))
        .inner_margin(egui::Margin::symmetric(6.0, 4.0))
        .show(ui, |ui| { ui.horizontal(|ui| {
        ui.menu_button("+ Primitive", |ui| {
            for kind in SdfPrimitive::ALL {
                if ui.button(kind.base_name()).clicked() {
                    actions.push(Action::CreatePrimitive(kind.clone()));
                    ui.close_menu();
                }
            }
        });

        ui.menu_button("+ Operation", |ui| {
            for op in CsgOp::ALL {
                if ui.button(op.base_name()).clicked() {
                    create_op_from_selection(scene, op.clone(), actions);
                    ui.close_menu();
                }
            }
        });

        ui.menu_button("+ Transform", |ui| {
            if ui.button("Transform").clicked() {
                if let Some(sel) = state.selected {
                    actions.push(Action::InsertTransformAbove { target: sel });
                } else {
                    actions.push(Action::CreateTransform { input: None });
                }
                ui.close_menu();
            }
        });

        ui.menu_button("+ Modifier", |ui| {
            ui.label("Deform");
            for kind in [ModifierKind::Twist, ModifierKind::Bend, ModifierKind::Taper, ModifierKind::Noise] {
                if ui.button(kind.base_name()).clicked() {
                    toolbar_add_modifier(state.selected, kind, actions);
                    ui.close_menu();
                }
            }
            ui.separator();
            ui.label("Shape");
            for kind in [ModifierKind::Round, ModifierKind::Onion, ModifierKind::Elongate] {
                if ui.button(kind.base_name()).clicked() {
                    toolbar_add_modifier(state.selected, kind, actions);
                    ui.close_menu();
                }
            }
            ui.separator();
            ui.label("Repeat");
            for kind in [ModifierKind::Mirror, ModifierKind::Repeat, ModifierKind::FiniteRepeat] {
                if ui.button(kind.base_name()).clicked() {
                    toolbar_add_modifier(state.selected, kind, actions);
                    ui.close_menu();
                }
            }
        });

        ui.separator();

        let has_selection = state.selected.is_some();
        if ui
            .add_enabled(has_selection, egui::Button::new("Delete"))
            .clicked()
        {
            if let Some(sel) = state.selected {
                actions.push(Action::DeleteNode(sel));
            }
        }

        ui.separator();

        if ui
            .add_enabled(has_selection, egui::Button::new("Focus Selected"))
            .clicked()
        {
            if let Some(sel) = state.selected {
                if let Some(&graph_id) = state.id_map.scene_to_graph.get(&sel) {
                    if let Some(&pos) = state.graph_state.node_positions.get(graph_id) {
                        let zoom = state.graph_state.pan_zoom.zoom;
                        state.graph_state.pan_zoom.pan = egui::Vec2::new(
                            graph_rect.width() / 2.0 - (pos.x + NODE_WIDTH / 2.0) * zoom,
                            graph_rect.height() / 2.0 - (pos.y + 30.0) * zoom,
                        );
                    }
                }
            }
        }

        if ui.button("Organize").clicked() {
            state.graph_state.node_positions = Default::default();
            auto_layout_graph(scene, &mut state.graph_state, &state.id_map);
            center_view_on_nodes(state, graph_rect);
        }
    }); });
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
const MINIMAP_PAN_LERP: f32 = 0.25; // smoothing factor for drag panning

fn draw_minimap(
    ui: &mut egui::Ui,
    graph_rect: egui::Rect,
    scene: &Scene,
    state: &mut NodeGraphState,
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
    let area_resp = egui::Area::new(egui::Id::new("node_graph_minimap"))
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
                let node_rect =
                    egui::Rect::from_min_size(tl, egui::Vec2::new(node_w, node_h));

                if !minimap_rect.intersects(node_rect) {
                    continue;
                }

                painter.rect_filled(node_rect, 2.0, color);
                if is_selected {
                    painter.rect_stroke(
                        node_rect,
                        2.0,
                        egui::Stroke::new(1.5, Color32::WHITE),
                    );
                }
            }

            minimap_response
        });

    let resp = area_resp.inner;
    // Click-to-pan: clicking/dragging on the minimap centers the viewport
    if resp.clicked() || resp.dragged() {
        if let Some(pointer) = ui.ctx().input(|i| i.pointer.hover_pos()) {
            let minimap_rect = egui::Rect::from_min_size(
                minimap_pos,
                egui::Vec2::new(MINIMAP_W, MINIMAP_H),
            );
            if minimap_rect.contains(pointer) {
                let inner_origin = minimap_rect.min + egui::Vec2::new(4.0, 4.0);
                let content_w = world_w * scale;
                let content_h = world_h * scale;
                let offset_x = (inner_w - content_w) * 0.5;
                let offset_y = (inner_h - content_h) * 0.5;
                let origin = inner_origin + egui::Vec2::new(offset_x, offset_y);
                let rel = pointer - origin;
                let world_x = min_x + rel.x / scale;
                let world_y = min_y + rel.y / scale;
                // Target: center viewport on clicked world position
                let half_vp_w = graph_rect.width() / zoom * 0.5;
                let half_vp_h = graph_rect.height() / zoom * 0.5;
                let target_pan = egui::Vec2::new(
                    -(world_x - half_vp_w) * zoom,
                    -(world_y - half_vp_h) * zoom,
                );
                // Lerp for smooth, lower-sensitivity panning
                let t = if resp.clicked() { 0.5 } else { MINIMAP_PAN_LERP };
                let current = state.graph_state.pan_zoom.pan;
                state.graph_state.pan_zoom.pan = egui::Vec2::new(
                    current.x + (target_pan.x - current.x) * t,
                    current.y + (target_pan.y - current.y) * t,
                );
                ui.ctx().request_repaint(); // keep animating while dragging
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

fn toolbar_add_modifier(selected: Option<SceneNodeId>, kind: ModifierKind, actions: &mut ActionSink) {
    if let Some(sel) = selected {
        actions.push(Action::InsertModifierAbove { target: sel, kind });
    } else {
        actions.push(Action::CreateModifier { kind, input: None });
    }
}

fn create_op_from_selection(scene: &Scene, op: CsgOp, actions: &mut ActionSink) {
    let tops = scene.top_level_nodes();
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
    *value = glam::Vec3::new(
        deg.x.to_radians(),
        deg.y.to_radians(),
        deg.z.to_radians(),
    );
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
