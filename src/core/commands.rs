use glam::Vec3;

use crate::graph::scene::{CsgOp, LightType, ModifierKind, NodeId, SdfPrimitive};
use crate::sculpt::ActiveTool;

#[derive(Debug, Clone, PartialEq)]
pub struct NodeTransformPatch {
    pub node_id: NodeId,
    pub set_position: Option<Vec3>,
    pub set_rotation: Option<Vec3>,
    pub set_scale: Option<Vec3>,
    pub add_position: Vec3,
    pub add_rotation: Vec3,
    pub add_scale: Vec3,
}

impl NodeTransformPatch {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            set_position: None,
            set_rotation: None,
            set_scale: None,
            add_position: Vec3::ZERO,
            add_rotation: Vec3::ZERO,
            add_scale: Vec3::ZERO,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoreCommand {
    NewScene,
    Select(Option<NodeId>),
    ToggleSelect(NodeId),
    SetActiveTool(ActiveTool),
    ToggleDebug,
    ToggleSettings,
    CreatePrimitive(SdfPrimitive),
    CreateOperation {
        op: CsgOp,
        left: Option<NodeId>,
        right: Option<NodeId>,
    },
    CreateTransform {
        input: Option<NodeId>,
    },
    CreateModifier {
        kind: ModifierKind,
        input: Option<NodeId>,
    },
    CreateLight(LightType),
    SoloSelectedLight,
    ClearSoloLight,
    DeleteNode(NodeId),
    DeleteSelected,
    ToggleSelectedVisibility,
    RenameNode {
        id: NodeId,
        name: String,
    },
    RenameSelected(String),
    FocusSelected,
    Undo,
    Redo,
    ApplyNodeTransformPatch(NodeTransformPatch),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoreCommandResult {
    pub handled: bool,
    pub buffer_dirty: bool,
    pub needs_graph_rebuild: bool,
    pub clear_isolation: bool,
    pub deactivate_sculpt: bool,
    pub pending_center_node: Option<NodeId>,
    pub toast_message: Option<String>,
}


