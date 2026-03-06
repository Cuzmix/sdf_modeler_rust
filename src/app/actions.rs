use crate::graph::scene::{LightType, ModifierKind, NodeId, SdfPrimitive, CsgOp};
use crate::sculpt::ActiveTool;
use crate::ui::gizmo::GizmoMode;

use super::BakeRequest;

/// Collects actions during a frame. Drained by the update loop.
pub type ActionSink = Vec<Action>;

#[derive(Debug, Clone)]
pub enum WorkspacePreset {
    Modeling,
    Sculpting,
    Rendering,
}

/// How the user wants to bake the SDF into a sculpt voxel grid.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum SculptConvertMode {
    /// Bake the entire scene into a single sculpt object.
    BakeWholeScene,
    /// Bake the entire scene and flatten into a standalone sculpt (destructive).
    BakeWholeSceneFlatten,
    /// Bake only the selected node's subtree.
    BakeActiveNode,
}

/// All possible state-mutating intents that UI components can express.
/// Each variant contains exactly the data needed to execute the action.
///
/// UI components push actions instead of mutating state directly.
/// The `process_actions()` method in `action_handler.rs` is the single
/// place where these actions are applied to state — analogous to a
/// Redux reducer.
#[derive(Debug, Clone)]
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
    /// Undo one sculpt stroke (only while sculpt is active).
    SculptUndo,
    /// Redo one sculpt stroke (only while sculpt is active).
    SculptRedo,

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

    // ── Sculpt entry ────────────────────────────────────────────────
    /// Ctrl+R: enter sculpt mode. If target is already sculpt, activates directly.
    /// Otherwise opens the convert dialog.
    EnterSculptMode,
    /// Open the convert-to-sculpt dialog for the given node.
    ShowSculptConvertDialog { target: NodeId },
    /// User confirmed the convert dialog — bake and enter sculpt.
    CommitSculptConvert {
        target: NodeId,
        mode: SculptConvertMode,
        resolution: u32,
    },

    // ── Scene mutations (structural) ─────────────────────────────────
    CreatePrimitive(SdfPrimitive),
    CreateOperation { op: CsgOp, left: Option<NodeId>, right: Option<NodeId> },
    CreateTransform { input: Option<NodeId> },
    CreateModifier { kind: ModifierKind, input: Option<NodeId> },
    CreateLight(LightType),
    InsertModifierAbove { target: NodeId, kind: ModifierKind },
    InsertTransformAbove { target: NodeId },
    ReparentNode { dragged: NodeId, new_parent: NodeId },
    RenameNode { id: NodeId, name: String },
    ToggleVisibility(NodeId),
    ToggleLock(NodeId),
    SwapChildren(NodeId),

    // ── Graph connections ────────────────────────────────────────────
    SetLeftChild { parent: NodeId, child: Option<NodeId> },
    SetRightChild { parent: NodeId, child: Option<NodeId> },
    SetSculptInput { parent: NodeId, child: Option<NodeId> },

    // ── Bake / Export / Import ───────────────────────────────────────
    RequestBake(BakeRequest),
    ShowExportDialog,
    ImportMesh,
    /// User confirmed the import dialog — start voxelization with chosen resolution.
    CommitImport { resolution: u32 },
    TakeScreenshot,

    // ── Viewport ────────────────────────────────────────────────────
    ToggleIsolation,
    CycleShadingMode,
    ToggleTurntable,

    // ── Properties ─────────────────────────────────────────────────
    CopyProperties,
    PasteProperties,

    // ── Camera bookmarks ───────────────────────────────────────────
    SaveBookmark(usize),
    RestoreBookmark(usize),

    // ── Workspace ─────────────────────────────────────────────────────
    SetWorkspace(WorkspacePreset),

    // ── UI toggles ───────────────────────────────────────────────────
    ToggleDebug,
    ToggleHelp,
    ToggleSettings,
    ToggleCommandPalette,
    ShowToast { message: String, is_error: bool },

    // ── Settings / GPU ───────────────────────────────────────────────
    SettingsChanged,
    MarkBufferDirty,
}
