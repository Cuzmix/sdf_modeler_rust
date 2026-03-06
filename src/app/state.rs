use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use eframe::egui_wgpu::RenderState;
use egui_dock::DockState;
use glam::Vec3;

use crate::compat::Instant;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::{ActiveTool, SculptState};
use crate::sculpt_history::SculptHistory;
use crate::ui::dock::Tab;
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::NodeGraphState;

use super::{BakeStatus, ExportStatus, FrameTimings, ImportStatus, PickState, Toast};

// ---------------------------------------------------------------------------
// Core document state: the "model" being edited.
// ---------------------------------------------------------------------------

pub struct DocumentState {
    pub scene: Scene,
    pub camera: Camera,
    pub history: History,
    pub active_tool: ActiveTool,
    pub sculpt_state: SculptState,
    pub sculpt_history: SculptHistory,
    pub clipboard_node: Option<NodeId>,
}

// ---------------------------------------------------------------------------
// Gizmo interaction state.
// ---------------------------------------------------------------------------

pub struct GizmoContext {
    pub state: GizmoState,
    pub mode: GizmoMode,
    pub space: GizmoSpace,
    pub pivot_offset: Vec3,
    pub last_selection: Option<NodeId>,
    pub gizmo_visible: bool,
}

// ---------------------------------------------------------------------------
// GPU synchronization state — tracks what needs rebuilding / uploading.
// ---------------------------------------------------------------------------

pub struct GpuSyncState {
    pub render_state: RenderState,
    pub current_structure_key: u64,
    pub buffer_dirty: bool,
    pub last_data_fingerprint: u64,
    pub voxel_gpu_offsets: HashMap<NodeId, u32>,
    pub sculpt_tex_indices: HashMap<NodeId, usize>,
}

// ---------------------------------------------------------------------------
// Async task tracking (bake, export, sculpt pick).
// ---------------------------------------------------------------------------

pub struct AsyncState {
    pub bake_status: BakeStatus,
    pub export_status: ExportStatus,
    pub import_status: ImportStatus,
    pub pick_state: PickState,
    pub pending_pick: Option<PendingPick>,
    pub last_sculpt_hit: Option<Vec3>,
    pub lazy_brush_pos: Option<Vec3>,
    /// Modifier keys captured at the time of sculpt drag (for Ctrl-invert / Shift-smooth).
    pub sculpt_ctrl_held: bool,
    pub sculpt_shift_held: bool,
    /// Pen pressure (0.0-1.0) during sculpt drag. 0.0 = no pressure data.
    pub sculpt_pressure: f32,
    /// World position from the latest hover pick (for 3D brush preview).
    /// Independent of drag state — persists while hovering, cleared when cursor leaves.
    pub hover_world_pos: Option<Vec3>,
    /// Whether the cursor is currently over geometry (from last hover pick).
    /// Used to decide: LMB on empty space → orbit instead of sculpt.
    pub cursor_over_geometry: bool,
    /// Whether a sculpt drag is actively in progress (LMB held on geometry).
    pub sculpt_dragging: bool,
}

// ---------------------------------------------------------------------------
// UI-only state: dialog visibility, rename editing, toasts, dock layout.
// ---------------------------------------------------------------------------

/// Saved state when entering isolation mode.
pub struct IsolationState {
    pub pre_hidden: HashSet<NodeId>,
    pub isolated_node: NodeId,
}

/// Clipboard for property copy/paste.
#[derive(Clone)]
pub struct PropertyClipboard {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
    pub fresnel: f32,
}

/// State for the "Convert to Sculpt" dialog shown by Ctrl+R.
pub struct SculptConvertDialog {
    pub target: NodeId,
    pub mode: crate::app::actions::SculptConvertMode,
    pub resolution: u32,
}

impl SculptConvertDialog {
    pub fn new(target: NodeId) -> Self {
        Self {
            target,
            mode: crate::app::actions::SculptConvertMode::BakeActiveNode,
            resolution: 64,
        }
    }
}

pub struct UiState {
    pub dock_state: DockState<Tab>,
    pub node_graph_state: NodeGraphState,
    pub show_debug: bool,
    pub show_help: bool,
    pub show_export_dialog: bool,
    pub show_settings: bool,
    pub renaming_node: Option<NodeId>,
    pub rename_buf: String,
    pub scene_tree_drag: Option<NodeId>,
    pub scene_tree_search: String,
    pub isolation_state: Option<IsolationState>,
    pub toasts: Vec<Toast>,
    pub turntable_active: bool,
    pub property_clipboard: Option<PropertyClipboard>,
    pub command_palette_open: bool,
    pub command_palette_query: String,
    pub command_palette_selected: usize,
    /// Open "Convert to Sculpt" dialog state (None = hidden).
    pub sculpt_convert_dialog: Option<SculptConvertDialog>,
    /// Quick Primitives floating toolbar (Shift+A).
    pub show_quick_toolbar: bool,
    /// Keybinding editor: which action is currently waiting for a key press (None = not rebinding).
    pub rebinding_action: Option<crate::keymap::ActionBinding>,
}

// ---------------------------------------------------------------------------
// Persistence state: file path, save tracking, auto-save timer.
// ---------------------------------------------------------------------------

pub struct PersistenceState {
    pub current_file_path: Option<PathBuf>,
    pub scene_dirty: bool,
    pub saved_fingerprint: u64,
    pub last_auto_save: Instant,
}

// ---------------------------------------------------------------------------
// Performance / profiling state.
// ---------------------------------------------------------------------------

pub struct PerfState {
    pub timings: FrameTimings,
    pub resolution_upgrade_pending: bool,
    pub composite_full_update_needed: bool,
}
