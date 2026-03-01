use crate::graph::scene::{ModifierKind, NodeId, SdfPrimitive, TransformKind, CsgOp};
use crate::sculpt::ActiveTool;
use crate::ui::gizmo::GizmoMode;

use super::BakeRequest;

/// Collects actions during a frame. Drained by the update loop.
pub type ActionSink = Vec<Action>;

/// All possible state-mutating intents that UI components can express.
/// Each variant contains exactly the data needed to execute the action.
///
/// UI components push actions instead of mutating state directly.
/// The `process_actions()` method in `action_handler.rs` is the single
/// place where these actions are applied to state — analogous to a
/// Redux reducer.
#[derive(Debug)]
#[allow(dead_code)] // Variants used progressively as UI components adopt actions
pub enum Action {
    // ── Scene ────────────────────────────────────────────────────────
    NewScene,
    OpenProject,
    OpenRecentProject(String),
    SaveProject,

    // ── Selection ────────────────────────────────────────────────────
    Select(Option<NodeId>),
    DeleteSelected,
    DeleteNode(NodeId),

    // ── Clipboard ────────────────────────────────────────────────────
    Copy,
    Paste,
    Duplicate,

    // ── History ──────────────────────────────────────────────────────
    Undo,
    Redo,

    // ── Camera ───────────────────────────────────────────────────────
    FocusSelected,
    FrameAll,
    CameraFront,
    CameraTop,
    CameraRight,
    CameraBack,
    CameraLeft,
    CameraBottom,
    ToggleOrtho,

    // ── Tools ────────────────────────────────────────────────────────
    SetTool(ActiveTool),
    SetGizmoMode(GizmoMode),
    ToggleGizmoSpace,
    ResetPivot,

    // ── Scene mutations (structural) ─────────────────────────────────
    CreatePrimitive(SdfPrimitive),
    CreateOperation { op: CsgOp, left: Option<NodeId>, right: Option<NodeId> },
    CreateTransform { kind: TransformKind, input: Option<NodeId> },
    CreateModifier { kind: ModifierKind, input: Option<NodeId> },
    InsertModifierAbove { target: NodeId, kind: ModifierKind },
    InsertTransformAbove { target: NodeId, kind: TransformKind },
    ReparentNode { dragged: NodeId, new_parent: NodeId },
    RenameNode { id: NodeId, name: String },
    ToggleVisibility(NodeId),
    SwapChildren(NodeId),

    // ── Graph connections ────────────────────────────────────────────
    SetLeftChild { parent: NodeId, child: Option<NodeId> },
    SetRightChild { parent: NodeId, child: Option<NodeId> },
    SetSculptInput { parent: NodeId, child: Option<NodeId> },

    // ── Bake / Export ────────────────────────────────────────────────
    RequestBake(BakeRequest),
    ShowExportDialog,
    TakeScreenshot,

    // ── Viewport ────────────────────────────────────────────────────
    ToggleIsolation,
    CycleShadingMode,

    // ── UI toggles ───────────────────────────────────────────────────
    ToggleDebug,
    ToggleHelp,
    ToggleSettings,
    ShowToast { message: String, is_error: bool },

    // ── Settings / GPU ───────────────────────────────────────────────
    SettingsChanged,
    MarkBufferDirty,
}
