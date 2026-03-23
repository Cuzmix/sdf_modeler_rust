use eframe::egui;

use crate::graph::scene::{CsgOp, LightType, ModifierKind, NodeId, SdfPrimitive};
use crate::sculpt::ActiveTool;
use crate::settings::SelectionBehaviorSettings;
use crate::ui::dock::Tab;
use crate::ui::gizmo::GizmoMode;

use super::state::{InteractionMode, ShellPanelKind};
use super::BakeRequest;

/// Industry-standard lighting presets that configure scene Key + Fill lights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightingPreset {
    /// Classic 3-point studio setup: warm key, cool fill, moderate ambient.
    Studio,
    /// Bright daylight: strong warm key, soft sky fill, low ambient.
    Outdoor,
    /// High-contrast cinematic: strong warm key, minimal fill, very low ambient.
    Dramatic,
    /// Even, shadowless illumination: neutral key and fill, high ambient.
    Flat,
}

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
    SaveNodePreset(NodeId),
    LoadNodePreset,
    AddReferenceImage,
    RemoveReferenceImage(usize),
    ToggleReferenceImageVisibility(usize),
    ToggleAllReferenceImages,

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
    SetInteractionMode(InteractionMode),
    SetTool(ActiveTool),
    SetGizmoMode(GizmoMode),
    ToggleGizmoSpace,
    ResetPivot,

    // ── Sculpt entry ────────────────────────────────────────────────
    /// Ctrl+R: enter sculpt mode. If target is already sculpt, activates directly.
    /// Otherwise opens the convert dialog.
    EnterSculptMode,
    /// Open the convert-to-sculpt dialog for the given node.
    ShowSculptConvertDialog {
        target: NodeId,
    },
    /// User confirmed the convert dialog — bake and enter sculpt.
    CommitSculptConvert {
        target: NodeId,
        mode: SculptConvertMode,
        resolution: u32,
    },
    IncreaseSculptDetail(NodeId),
    DecreaseSculptDetail(NodeId),
    RemeshSculptAtCurrentDetail(NodeId),
    ExpandSculptVolume(NodeId),
    FitSculptVolume(NodeId),

    // ── Scene mutations (structural) ─────────────────────────────────
    CreatePrimitive(SdfPrimitive),
    ShellCreateBooleanPrimitive {
        op: CsgOp,
        primitive: SdfPrimitive,
    },
    CreateOperation {
        op: CsgOp,
        left: Option<NodeId>,
        right: Option<NodeId>,
    },
    CreateTransform {
        input: Option<NodeId>,
    },
    CreateReroute {
        input: Option<NodeId>,
    },
    CreateModifier {
        kind: ModifierKind,
        input: Option<NodeId>,
    },
    CreateLight(LightType),
    InsertModifierAbove {
        target: NodeId,
        kind: ModifierKind,
    },
    InsertTransformAbove {
        target: NodeId,
    },
    ReparentNode {
        dragged: NodeId,
        new_parent: NodeId,
    },
    RenameNode {
        id: NodeId,
        name: String,
    },
    ToggleVisibility(NodeId),
    ToggleLock(NodeId),
    SwapChildren(NodeId),

    // ── Graph connections ────────────────────────────────────────────
    SetLeftChild {
        parent: NodeId,
        child: Option<NodeId>,
    },
    SetRightChild {
        parent: NodeId,
        child: Option<NodeId>,
    },
    SetSculptInput {
        parent: NodeId,
        child: Option<NodeId>,
    },

    // ── Bake / Export / Import ───────────────────────────────────────
    RequestBake(BakeRequest),
    ShowExportDialog,
    ImportMesh,
    /// User confirmed the import dialog — start voxelization with chosen resolution.
    CommitImport {
        resolution: u32,
    },
    TakeScreenshot,

    // ── Viewport ────────────────────────────────────────────────────
    ToggleIsolation,
    CycleShadingMode,
    ToggleTurntable,
    ToggleDistanceReadout,
    ToggleMeasurementTool,

    // ── Properties ─────────────────────────────────────────────────
    CopyProperties,
    PasteProperties,

    // ── Camera bookmarks ───────────────────────────────────────────
    SaveBookmark(usize),
    RestoreBookmark(usize),

    // ── Workspace ─────────────────────────────────────────────────────
    SetWorkspace(WorkspacePreset),
    DockShellPanel {
        panel: ShellPanelKind,
        rect: egui::Rect,
    },
    UndockShellPanel(ShellPanelKind),
    HideShellPanel(ShellPanelKind),
    ResetPrimaryShellLayout,
    ToggleDockTab(Tab),

    // ── UI toggles ───────────────────────────────────────────────────
    ToggleDebug,
    ToggleHelp,
    ToggleSettings,
    ToggleCommandPalette,
    ShowToast {
        message: String,
        is_error: bool,
    },

    // ── Light linking ─────────────────────────────────────────────────
    /// Set the full light mask for a geometry node (8-bit bitmask).
    SetLightMask {
        node_id: NodeId,
        mask: u8,
    },
    /// Toggle a single light slot bit in a geometry node's light mask.
    ToggleLightMaskBit {
        node_id: NodeId,
        light_slot: u8,
        enabled: bool,
    },

    // ── Light Solo ───────────────────────────────────────────────────
    /// Toggle light solo mode. Some(id) = solo that light (or unsolo if already soloed).
    /// None = clear solo.
    SoloLight(Option<NodeId>),

    // ── Light Cookie ──────────────────────────────────────────────────
    /// Set or clear a light's SDF cookie shape. The cookie node must be a Primitive
    /// or Operation subtree whose SDF shapes the light's beam.
    SetLightCookie {
        light_id: NodeId,
        cookie: Option<NodeId>,
    },

    // ── Lighting presets ─────────────────────────────────────────────
    /// Apply a lighting preset to scene Key/Fill lights and ambient.
    ApplyLightingPreset(LightingPreset),

    // ── Settings / GPU ───────────────────────────────────────────────
    SetSelectionBehavior(SelectionBehaviorSettings),
    SettingsChanged,
    MarkBufferDirty,
}
