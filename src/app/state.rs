use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;

use crate::app::reference_images::ReferenceImageStore;
use crate::compat::Instant;
use crate::gizmo::{GizmoMode, GizmoSelection, GizmoSpace, GizmoState};
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{MaterialParams, NodeId, Scene};
use crate::mesh_import::TriMesh;
use crate::sculpt::{ActiveTool, BrushMode, SculptState};
use crate::settings::SelectionBehaviorSettings;

use super::runtime::{AppRenderContext, ViewportResourceHandle};
use super::ui_geometry::FloatingPanelBounds;
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
    pub clipboard_node: Option<NodeId>,
    /// When set, only this Light node contributes to scene lighting (Solo mode).
    pub soloed_light: Option<NodeId>,
}

// ---------------------------------------------------------------------------
// Gizmo interaction state.
// ---------------------------------------------------------------------------

pub struct GizmoContext {
    pub state: GizmoState,
    pub mode: GizmoMode,
    pub space: GizmoSpace,
    pub pivot_offset: Vec3,
    pub last_selection_ids: Vec<NodeId>,
    pub gizmo_visible: bool,
}

// ---------------------------------------------------------------------------
// GPU synchronization state — tracks what needs rebuilding / uploading.
// ---------------------------------------------------------------------------

pub struct GpuSyncState {
    pub render_context: AppRenderContext,
    pub viewport_resources: ViewportResourceHandle,
    pub current_structure_key: u64,
    pub buffer_dirty: bool,
    pub last_data_fingerprint: u64,
    pub last_environment_fingerprint: u64,
    pub voxel_gpu_offsets: HashMap<NodeId, u32>,
    pub sculpt_tex_indices: HashMap<NodeId, usize>,
}

// ---------------------------------------------------------------------------
// Async task tracking (bake, export, sculpt pick).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SculptRuntimeCache {
    pub node_id: NodeId,
    pub structure_key: u64,
    pub material_id: i32,
    pub position: Vec3,
    pub rotation: Vec3,
    pub gpu_offset: Option<u32>,
    pub grid_resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}
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
    /// Per-stroke cached node metadata to avoid repeated topo/transform lookups.
    pub sculpt_runtime_cache: Option<SculptRuntimeCache>,
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
    pub material: MaterialParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptBrushAdjustMode {
    Radius,
    Strength,
}

#[derive(Clone, Debug)]
pub struct SculptBrushAdjustState {
    pub mode: SculptBrushAdjustMode,
    pub anchor_pos: [f32; 2],
    pub initial_value: f32,
}

#[derive(Clone, Debug)]
pub struct MultiTransformSessionState {
    pub selection_key: Vec<NodeId>,
    pub behavior_key: SelectionBehaviorSettings,
    pub baseline_selection: Option<GizmoSelection>,
    pub position_delta: Vec3,
    pub rotation_delta_deg: Vec3,
    pub scale_factor: Vec3,
}

impl Default for MultiTransformSessionState {
    fn default() -> Self {
        Self {
            selection_key: Vec::new(),
            behavior_key: SelectionBehaviorSettings::default(),
            baseline_selection: None,
            position_delta: Vec3::ZERO,
            rotation_delta_deg: Vec3::ZERO,
            scale_factor: Vec3::ONE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimaryShellInspectorTab {
    Properties,
    Display,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimaryShellUtilityTab {
    History,
    Reference,
    Advanced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkspaceRoute {
    NodeGraph,
    LightGraph,
}

impl WorkspaceRoute {
    pub const ALL: [Self; 2] = [Self::NodeGraph, Self::LightGraph];

    pub fn label(self) -> &'static str {
        match self {
            Self::NodeGraph => "Node Graph",
            Self::LightGraph => "Light Graph",
        }
    }

    pub fn expert_panel_kind(self) -> ExpertPanelKind {
        match self {
            Self::NodeGraph => ExpertPanelKind::NodeGraph,
            Self::LightGraph => ExpertPanelKind::LightGraph,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceUiState {
    pub route: WorkspaceRoute,
}

impl Default for WorkspaceUiState {
    fn default() -> Self {
        Self {
            route: WorkspaceRoute::NodeGraph,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertPanelKind {
    NodeGraph,
    LightGraph,
    Properties,
    ReferenceImages,
    SceneTree,
    RenderSettings,
    History,
    BrushSettings,
    Lights,
    LightLinking,
    SceneStats,
}

impl ExpertPanelKind {
    pub const ALL: [Self; 11] = [
        Self::NodeGraph,
        Self::LightGraph,
        Self::Properties,
        Self::ReferenceImages,
        Self::SceneTree,
        Self::RenderSettings,
        Self::History,
        Self::BrushSettings,
        Self::Lights,
        Self::LightLinking,
        Self::SceneStats,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::NodeGraph => "Node Graph",
            Self::LightGraph => "Light Graph",
            Self::Properties => "Properties",
            Self::ReferenceImages => "Reference Images",
            Self::SceneTree => "Scene Tree",
            Self::RenderSettings => "Render Settings",
            Self::History => "History",
            Self::BrushSettings => "Brush Settings",
            Self::Lights => "Lights",
            Self::LightLinking => "Light Linking",
            Self::SceneStats => "Scene Stats",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertPanelRegistry {
    open: HashSet<ExpertPanelKind>,
}

impl ExpertPanelRegistry {
    pub fn is_open(&self, panel: ExpertPanelKind) -> bool {
        self.open.contains(&panel)
    }

    pub fn set_open(&mut self, panel: ExpertPanelKind, open: bool) {
        if open {
            self.open.insert(panel);
        } else {
            self.open.remove(&panel);
        }
    }

    pub fn clear(&mut self) {
        self.open.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MenuDropdownKind {
    File,
    Edit,
    View,
    Help,
}

impl MenuDropdownKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Edit => "Edit",
            Self::View => "View",
            Self::Help => "Help",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MenuLauncherKind {
    File,
    Edit,
    View,
    Settings,
    Help,
}

impl MenuLauncherKind {
    pub fn from_dropdown(kind: MenuDropdownKind) -> Self {
        match kind {
            MenuDropdownKind::File => Self::File,
            MenuDropdownKind::Edit => Self::Edit,
            MenuDropdownKind::View => Self::View,
            MenuDropdownKind::Help => Self::Help,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MenuUiState {
    pub active_dropdown: Option<MenuDropdownKind>,
    pub settings_card_open: bool,
    pub focused_launcher: Option<MenuLauncherKind>,
    pub highlighted_command_index: Option<usize>,
}

impl MenuUiState {
    pub fn has_open_surface(&self) -> bool {
        self.active_dropdown.is_some() || self.settings_card_open
    }

    pub fn active_launcher(&self) -> Option<MenuLauncherKind> {
        if let Some(dropdown) = self.active_dropdown {
            return Some(MenuLauncherKind::from_dropdown(dropdown));
        }
        if self.settings_card_open {
            return Some(MenuLauncherKind::Settings);
        }
        None
    }

    pub fn dismiss_all(&mut self) {
        self.focused_launcher = self.active_launcher().or(self.focused_launcher);
        self.active_dropdown = None;
        self.settings_card_open = false;
        self.highlighted_command_index = None;
    }

    pub fn open_dropdown(&mut self, kind: MenuDropdownKind) {
        self.active_dropdown = Some(kind);
        self.settings_card_open = false;
        self.focused_launcher = Some(MenuLauncherKind::from_dropdown(kind));
        self.highlighted_command_index = None;
    }

    pub fn toggle_dropdown(&mut self, kind: MenuDropdownKind) {
        self.focused_launcher = Some(MenuLauncherKind::from_dropdown(kind));
        if self.active_dropdown == Some(kind) {
            self.active_dropdown = None;
        } else {
            self.active_dropdown = Some(kind);
        }
        self.settings_card_open = false;
        self.highlighted_command_index = None;
    }

    pub fn close_dropdown(&mut self) {
        self.focused_launcher = self.active_launcher().or(self.focused_launcher);
        self.active_dropdown = None;
        self.highlighted_command_index = None;
    }

    pub fn open_settings_card(&mut self) {
        self.settings_card_open = true;
        self.active_dropdown = None;
        self.focused_launcher = Some(MenuLauncherKind::Settings);
        self.highlighted_command_index = None;
    }

    pub fn toggle_settings_card(&mut self) {
        self.focused_launcher = Some(MenuLauncherKind::Settings);
        self.settings_card_open = !self.settings_card_open;
        if self.settings_card_open {
            self.active_dropdown = None;
        }
        self.highlighted_command_index = None;
    }

    pub fn close_settings_card(&mut self) {
        self.focused_launcher = self.active_launcher().or(self.focused_launcher);
        self.settings_card_open = false;
        self.highlighted_command_index = None;
    }

    pub fn set_highlighted_command_index(&mut self, index: Option<usize>) {
        self.highlighted_command_index = index;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PanelKind {
    Tool,
    ObjectProperties,
    RenderSettings,
    Scene,
    History,
    ReferenceImages,
}

impl PanelKind {
    pub const ALL: [Self; 6] = [
        Self::Tool,
        Self::ObjectProperties,
        Self::RenderSettings,
        Self::Scene,
        Self::History,
        Self::ReferenceImages,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Tool => "Tool",
            Self::ObjectProperties => "Object Properties",
            Self::RenderSettings => "Render Settings",
            Self::Scene => "Scene",
            Self::History => "History",
            Self::ReferenceImages => "Reference Images",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Tool => "Tool",
            Self::ObjectProperties => "Props",
            Self::RenderSettings => "Render",
            Self::Scene => "Scene",
            Self::History => "History",
            Self::ReferenceImages => "Refs",
        }
    }

    pub fn icon_key(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::ObjectProperties => "object-properties",
            Self::RenderSettings => "render-settings",
            Self::Scene => "scene",
            Self::History => "history",
            Self::ReferenceImages => "reference-images",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PanelBarId {
    PrimaryRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelBarOrientation {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelBarEdge {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelPresentation {
    TransientSheet,
    PinnedDocked,
    PinnedFloating,
}

pub type PanelInstanceId = u64;

#[derive(Clone, Debug, PartialEq)]
pub struct PanelBarState {
    pub id: PanelBarId,
    pub edge: PanelBarEdge,
    pub orientation: PanelBarOrientation,
    pub items: Vec<PanelKind>,
    pub active_transient: Option<PanelKind>,
    pub transient_rect: Option<FloatingPanelBounds>,
}

impl PanelBarState {
    pub fn contains_kind(&self, kind: PanelKind) -> bool {
        self.items.contains(&kind)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelInstanceState {
    pub id: PanelInstanceId,
    pub kind: PanelKind,
    pub presentation: PanelPresentation,
    pub pinned: bool,
    pub anchor_bar: PanelBarId,
    pub visible: bool,
    pub collapsed: bool,
    pub rect: Option<FloatingPanelBounds>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelResizeHandle {
    Top,
    Right,
    Bottom,
    Left,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelPointerInteractionKind {
    Move,
    Resize(PanelResizeHandle),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelPointerInteractionState {
    pub bar_id: PanelBarId,
    pub kind: PanelKind,
    pub interaction: PanelPointerInteractionKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PanelSheetAnchor {
    Left,
    Right,
    Above,
    Below,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelFrameworkState {
    pub bars: Vec<PanelBarState>,
    pub pinned_instances: Vec<PanelInstanceState>,
    pub focus_order: Vec<PanelInstanceId>,
    pub next_instance_id: PanelInstanceId,
    pub panel_interaction: Option<PanelPointerInteractionState>,
}

impl Default for PanelFrameworkState {
    fn default() -> Self {
        Self {
            bars: vec![PanelBarState {
                id: PanelBarId::PrimaryRight,
                edge: PanelBarEdge::Right,
                orientation: PanelBarOrientation::Vertical,
                items: PanelKind::ALL.to_vec(),
                active_transient: None,
                transient_rect: None,
            }],
            pinned_instances: Vec::new(),
            focus_order: Vec::new(),
            next_instance_id: 1,
            panel_interaction: None,
        }
    }
}

impl PanelFrameworkState {
    pub const TRANSIENT_PANEL_MARGIN: f32 = 20.0;
    pub const TRANSIENT_PANEL_BAR_WIDTH: f32 = 144.0;
    pub const TRANSIENT_PANEL_GAP: f32 = 14.0;
    pub const TRANSIENT_PANEL_DEFAULT_WIDTH_MAX: f32 = 390.0;
    pub const TRANSIENT_PANEL_DEFAULT_HEIGHT_MAX: f32 = 420.0;
    pub const COLLAPSED_PANEL_WIDTH: f32 = 320.0;
    pub const COLLAPSED_PANEL_HEIGHT: f32 = 76.0;
    pub const PANEL_MIN_WIDTH: f32 = 340.0;
    pub const PANEL_MIN_HEIGHT: f32 = 180.0;

    pub fn bar(&self, id: PanelBarId) -> Option<&PanelBarState> {
        self.bars.iter().find(|bar| bar.id == id)
    }

    pub fn bar_mut(&mut self, id: PanelBarId) -> Option<&mut PanelBarState> {
        self.bars.iter_mut().find(|bar| bar.id == id)
    }

    pub fn pinned_instance(&self, kind: PanelKind) -> Option<&PanelInstanceState> {
        self.pinned_instances
            .iter()
            .find(|instance| instance.kind == kind && instance.visible)
    }

    pub fn pinned_instance_mut(&mut self, kind: PanelKind) -> Option<&mut PanelInstanceState> {
        self.pinned_instances
            .iter_mut()
            .find(|instance| instance.kind == kind && instance.visible)
    }

    pub fn active_transient(&self, bar_id: PanelBarId) -> Option<PanelKind> {
        self.bar(bar_id).and_then(|bar| bar.active_transient)
    }

    pub fn remembered_transient_rect(&self, bar_id: PanelBarId) -> Option<FloatingPanelBounds> {
        self.bar(bar_id).and_then(|bar| bar.transient_rect)
    }

    pub fn dismiss_transient_panels(&mut self) {
        for bar in &mut self.bars {
            bar.active_transient = None;
        }
        self.panel_interaction = None;
    }

    pub fn close_panel(&mut self, kind: PanelKind) {
        for bar in &mut self.bars {
            if bar.active_transient == Some(kind) {
                bar.active_transient = None;
            }
        }
        self.clear_panel_interaction_for_kind(kind);

        if let Some(index) = self
            .pinned_instances
            .iter()
            .position(|instance| instance.kind == kind)
        {
            let instance = self.pinned_instances.remove(index);
            if let Some(rect) = instance.rect {
                if let Some(bar) = self.bar_mut(instance.anchor_bar) {
                    bar.transient_rect = Some(rect);
                }
            }
            let id = instance.id;
            self.focus_order.retain(|focused_id| *focused_id != id);
        }
    }

    pub fn focus_panel(&mut self, kind: PanelKind) {
        if let Some(instance_id) = self.pinned_instance(kind).map(|instance| instance.id) {
            self.focus_instance(instance_id);
        }
    }

    pub fn open_panel(&mut self, kind: PanelKind, bar_id: PanelBarId) {
        if self.pinned_instance(kind).is_some() {
            self.focus_panel(kind);
            return;
        }

        if let Some(bar) = self.bar_mut(bar_id) {
            bar.active_transient = Some(kind);
        }
    }

    pub fn toggle_panel(&mut self, kind: PanelKind, bar_id: PanelBarId) {
        if self.pinned_instance(kind).is_some() {
            self.focus_panel(kind);
            return;
        }

        let current = self.active_transient(bar_id);
        if current == Some(kind) {
            if let Some(bar) = self.bar_mut(bar_id) {
                bar.active_transient = None;
            }
        } else {
            self.open_panel(kind, bar_id);
        }
    }

    pub fn pin_panel(&mut self, kind: PanelKind) {
        if self.pinned_instance(kind).is_some() {
            self.focus_panel(kind);
            self.dismiss_kind_from_transient(kind);
            self.clear_panel_interaction_for_kind(kind);
            return;
        }

        let anchor_bar = self
            .bars
            .iter()
            .find(|bar| bar.active_transient == Some(kind))
            .map(|bar| bar.id)
            .or_else(|| {
                self.bars
                    .iter()
                    .find(|bar| bar.contains_kind(kind))
                    .map(|bar| bar.id)
            })
            .unwrap_or(PanelBarId::PrimaryRight);

        let remembered_rect = self.bar(anchor_bar).and_then(|bar| bar.transient_rect);
        self.dismiss_kind_from_transient(kind);
        self.clear_panel_interaction_for_kind(kind);

        let instance_id = self.next_instance_id;
        self.next_instance_id = self.next_instance_id.wrapping_add(1);
        self.pinned_instances.push(PanelInstanceState {
            id: instance_id,
            kind,
            presentation: PanelPresentation::PinnedFloating,
            pinned: true,
            anchor_bar,
            visible: true,
            collapsed: false,
            rect: remembered_rect,
        });
        self.focus_instance(instance_id);
    }

    pub fn unpin_panel(&mut self, kind: PanelKind) {
        let Some(index) = self
            .pinned_instances
            .iter()
            .position(|instance| instance.kind == kind)
        else {
            return;
        };

        let instance = self.pinned_instances.remove(index);
        self.focus_order
            .retain(|focused_id| *focused_id != instance.id);
        if let Some(bar) = self.bar_mut(instance.anchor_bar) {
            if let Some(rect) = instance.rect {
                bar.transient_rect = Some(rect);
            }
        }
        self.open_panel(kind, instance.anchor_bar);
    }

    pub fn toggle_pinned_collapsed(&mut self, kind: PanelKind) {
        if let Some(instance) = self.pinned_instance_mut(kind) {
            instance.collapsed = !instance.collapsed;
            let instance_id = instance.id;
            let _ = instance;
            self.focus_instance(instance_id);
        }
    }

    pub fn begin_panel_interaction(
        &mut self,
        kind: PanelKind,
        bar_id: PanelBarId,
        interaction: PanelPointerInteractionKind,
    ) {
        if let Some(instance_id) = self.pinned_instance(kind).map(|instance| instance.id) {
            self.focus_instance(instance_id);
            self.panel_interaction = Some(PanelPointerInteractionState {
                bar_id,
                kind,
                interaction,
            });
            return;
        }
        if self.active_transient(bar_id) == Some(kind) {
            self.panel_interaction = Some(PanelPointerInteractionState {
                bar_id,
                kind,
                interaction,
            });
        }
    }

    pub fn update_panel_interaction(
        &mut self,
        kind: PanelKind,
        bar_id: PanelBarId,
        delta_x: f32,
        delta_y: f32,
        usable_rect: FloatingPanelBounds,
    ) {
        let Some(interaction) = self.panel_interaction else {
            return;
        };
        if interaction.bar_id != bar_id || interaction.kind != kind {
            return;
        }

        let pinned_snapshot = self
            .pinned_instance(kind)
            .map(|instance| (instance.anchor_bar, instance.collapsed, instance.rect));
        let current = self
            .resolved_display_panel_rect(kind, bar_id, usable_rect)
            .unwrap_or_else(|| self.resolved_transient_rect(bar_id, usable_rect));
        let next = match interaction.interaction {
            PanelPointerInteractionKind::Move => clamp_panel_rect(
                FloatingPanelBounds::from_min_size(
                    current.x + delta_x,
                    current.y + delta_y,
                    current.width,
                    current.height,
                ),
                usable_rect,
            ),
            PanelPointerInteractionKind::Resize(handle) => {
                resize_panel_rect(current, handle, delta_x, delta_y, usable_rect)
            }
        };
        if let Some(instance_id) = self.pinned_instance(kind).map(|instance| instance.id) {
            let fallback_transient_rect = self.resolved_transient_rect(bar_id, usable_rect);
            let collapsed_move_stored_rect =
                pinned_snapshot.and_then(|(anchor_bar, collapsed, rect)| {
                    if collapsed
                        && matches!(interaction.interaction, PanelPointerInteractionKind::Move)
                    {
                        Some(rect.unwrap_or_else(|| {
                            self.resolved_transient_rect(anchor_bar, usable_rect)
                        }))
                    } else {
                        None
                    }
                });
            if let Some(instance) = self.pinned_instance_mut(kind) {
                instance.rect = Some(
                    if instance.collapsed
                        && matches!(interaction.interaction, PanelPointerInteractionKind::Move)
                    {
                        let stored_rect =
                            collapsed_move_stored_rect.unwrap_or(fallback_transient_rect);
                        FloatingPanelBounds::from_min_size(
                            next.x,
                            next.y,
                            stored_rect.width,
                            stored_rect.height,
                        )
                    } else {
                        next
                    },
                );
            }
            self.focus_instance(instance_id);
        } else if self.active_transient(bar_id) == Some(kind) {
            if let Some(bar) = self.bar_mut(bar_id) {
                bar.transient_rect = Some(next);
            }
        }
    }

    pub fn end_panel_interaction(&mut self, kind: PanelKind, bar_id: PanelBarId) {
        if self
            .panel_interaction
            .is_some_and(|interaction| interaction.bar_id == bar_id && interaction.kind == kind)
        {
            self.panel_interaction = None;
        }
    }

    pub fn cancel_panel_interaction(&mut self, kind: PanelKind, bar_id: PanelBarId) {
        self.end_panel_interaction(kind, bar_id);
    }

    pub fn resolved_transient_rect(
        &self,
        bar_id: PanelBarId,
        usable_rect: FloatingPanelBounds,
    ) -> FloatingPanelBounds {
        let bar = self.bar(bar_id);
        let resolved = bar
            .and_then(|current_bar| current_bar.transient_rect)
            .unwrap_or_else(|| default_transient_panel_rect(bar, usable_rect));
        clamp_panel_rect(resolved, usable_rect)
    }

    pub fn resolved_panel_rect(
        &self,
        kind: PanelKind,
        bar_id: PanelBarId,
        usable_rect: FloatingPanelBounds,
    ) -> Option<FloatingPanelBounds> {
        if let Some(instance) = self.pinned_instance(kind) {
            return Some(clamp_panel_rect(
                instance.rect.unwrap_or_else(|| {
                    self.resolved_transient_rect(instance.anchor_bar, usable_rect)
                }),
                usable_rect,
            ));
        }
        if self.active_transient(bar_id) == Some(kind) {
            return Some(self.resolved_transient_rect(bar_id, usable_rect));
        }
        None
    }

    pub fn resolved_display_panel_rect(
        &self,
        kind: PanelKind,
        bar_id: PanelBarId,
        usable_rect: FloatingPanelBounds,
    ) -> Option<FloatingPanelBounds> {
        if let Some(instance) = self.pinned_instance(kind) {
            if instance.collapsed {
                let expanded_rect = instance.rect.unwrap_or_else(|| {
                    self.resolved_transient_rect(instance.anchor_bar, usable_rect)
                });
                return Some(clamp_panel_display_rect(
                    collapsed_panel_rect(expanded_rect),
                    usable_rect,
                ));
            }
        }

        self.resolved_panel_rect(kind, bar_id, usable_rect)
    }

    pub fn sheet_anchor_for_bar(&self, bar_id: PanelBarId) -> PanelSheetAnchor {
        match self
            .bar(bar_id)
            .map(|bar| bar.edge)
            .unwrap_or(PanelBarEdge::Right)
        {
            PanelBarEdge::Left => PanelSheetAnchor::Right,
            PanelBarEdge::Right => PanelSheetAnchor::Left,
            PanelBarEdge::Top => PanelSheetAnchor::Below,
            PanelBarEdge::Bottom => PanelSheetAnchor::Above,
        }
    }

    fn dismiss_kind_from_transient(&mut self, kind: PanelKind) {
        for bar in &mut self.bars {
            if bar.active_transient == Some(kind) {
                bar.active_transient = None;
            }
        }
    }

    fn clear_panel_interaction_for_kind(&mut self, kind: PanelKind) {
        if self
            .panel_interaction
            .is_some_and(|interaction| interaction.kind == kind)
        {
            self.panel_interaction = None;
        }
    }

    fn focus_instance(&mut self, instance_id: PanelInstanceId) {
        self.focus_order
            .retain(|focused_id| *focused_id != instance_id);
        self.focus_order.push(instance_id);
    }
}

fn default_transient_panel_rect(
    bar: Option<&PanelBarState>,
    usable_rect: FloatingPanelBounds,
) -> FloatingPanelBounds {
    let margin = PanelFrameworkState::TRANSIENT_PANEL_MARGIN;
    let gap = PanelFrameworkState::TRANSIENT_PANEL_GAP;
    let bar_extent = PanelFrameworkState::TRANSIENT_PANEL_BAR_WIDTH;
    let max_width = (usable_rect.width - margin * 2.0).max(1.0);
    let min_width = PanelFrameworkState::PANEL_MIN_WIDTH.min(max_width);
    let available_width = max_width;
    let preferred_width = (usable_rect.width - bar_extent - margin * 3.0).max(min_width);
    let width = preferred_width
        .min(PanelFrameworkState::TRANSIENT_PANEL_DEFAULT_WIDTH_MAX)
        .min(available_width);
    let max_height = (usable_rect.height - margin * 2.0).max(1.0);
    let min_height = PanelFrameworkState::PANEL_MIN_HEIGHT.min(max_height);
    let height = max_height.clamp(
        min_height,
        PanelFrameworkState::TRANSIENT_PANEL_DEFAULT_HEIGHT_MAX,
    );

    let edge = bar
        .map(|current_bar| current_bar.edge)
        .unwrap_or(PanelBarEdge::Right);
    let (x, y) = match edge {
        PanelBarEdge::Left => (
            usable_rect.x + margin + bar_extent + gap,
            usable_rect.y + margin,
        ),
        PanelBarEdge::Right => (
            usable_rect.right() - margin - bar_extent - gap - width,
            usable_rect.y + margin,
        ),
        PanelBarEdge::Top => (
            usable_rect.x + margin,
            usable_rect.y + margin + bar_extent + gap,
        ),
        PanelBarEdge::Bottom => (
            usable_rect.x + margin,
            usable_rect.bottom() - margin - bar_extent - gap - height,
        ),
    };

    clamp_panel_rect(
        FloatingPanelBounds::from_min_size(x, y, width, height),
        usable_rect,
    )
}

fn collapsed_panel_rect(rect: FloatingPanelBounds) -> FloatingPanelBounds {
    FloatingPanelBounds::from_min_size(
        rect.x,
        rect.y,
        rect.width.min(PanelFrameworkState::COLLAPSED_PANEL_WIDTH),
        PanelFrameworkState::COLLAPSED_PANEL_HEIGHT,
    )
}

fn clamp_panel_rect(
    rect: FloatingPanelBounds,
    usable_rect: FloatingPanelBounds,
) -> FloatingPanelBounds {
    let margin = PanelFrameworkState::TRANSIENT_PANEL_MARGIN;
    let max_width = (usable_rect.width - margin * 2.0).max(1.0);
    let max_height = (usable_rect.height - margin * 2.0).max(1.0);
    let min_width = PanelFrameworkState::PANEL_MIN_WIDTH.min(max_width);
    let min_height = PanelFrameworkState::PANEL_MIN_HEIGHT.min(max_height);
    let width = rect.width.max(min_width).min(max_width);
    let height = rect.height.max(min_height).min(max_height);
    let min_x = usable_rect.x + margin;
    let max_x = (usable_rect.right() - margin - width).max(min_x);
    let min_y = usable_rect.y + margin;
    let max_y = (usable_rect.bottom() - margin - height).max(min_y);
    FloatingPanelBounds::from_min_size(
        rect.x.clamp(min_x, max_x),
        rect.y.clamp(min_y, max_y),
        width,
        height,
    )
}

fn clamp_panel_display_rect(
    rect: FloatingPanelBounds,
    usable_rect: FloatingPanelBounds,
) -> FloatingPanelBounds {
    let margin = PanelFrameworkState::TRANSIENT_PANEL_MARGIN;
    let max_width = (usable_rect.width - margin * 2.0).max(1.0);
    let max_height = (usable_rect.height - margin * 2.0).max(1.0);
    let width = rect.width.max(1.0).min(max_width);
    let height = rect.height.max(1.0).min(max_height);
    let min_x = usable_rect.x + margin;
    let max_x = (usable_rect.right() - margin - width).max(min_x);
    let min_y = usable_rect.y + margin;
    let max_y = (usable_rect.bottom() - margin - height).max(min_y);
    FloatingPanelBounds::from_min_size(
        rect.x.clamp(min_x, max_x),
        rect.y.clamp(min_y, max_y),
        width,
        height,
    )
}

fn resize_panel_rect(
    rect: FloatingPanelBounds,
    handle: PanelResizeHandle,
    delta_x: f32,
    delta_y: f32,
    usable_rect: FloatingPanelBounds,
) -> FloatingPanelBounds {
    let margin = PanelFrameworkState::TRANSIENT_PANEL_MARGIN;
    let min_width =
        PanelFrameworkState::PANEL_MIN_WIDTH.min((usable_rect.width - margin * 2.0).max(1.0));
    let min_height =
        PanelFrameworkState::PANEL_MIN_HEIGHT.min((usable_rect.height - margin * 2.0).max(1.0));
    let usable_right = usable_rect.right();
    let usable_bottom = usable_rect.bottom();

    let mut left = rect.x;
    let mut right = rect.right();
    let mut top = rect.y;
    let mut bottom = rect.bottom();

    if matches!(
        handle,
        PanelResizeHandle::Top | PanelResizeHandle::TopLeft | PanelResizeHandle::TopRight
    ) {
        top += delta_y;
    }
    if matches!(
        handle,
        PanelResizeHandle::Bottom | PanelResizeHandle::BottomLeft | PanelResizeHandle::BottomRight
    ) {
        bottom += delta_y;
    }

    if matches!(
        handle,
        PanelResizeHandle::Left | PanelResizeHandle::TopLeft | PanelResizeHandle::BottomLeft
    ) {
        left += delta_x;
    }
    if matches!(
        handle,
        PanelResizeHandle::Right | PanelResizeHandle::TopRight | PanelResizeHandle::BottomRight
    ) {
        right += delta_x;
    }

    if matches!(
        handle,
        PanelResizeHandle::Left | PanelResizeHandle::TopLeft | PanelResizeHandle::BottomLeft
    ) {
        left = left.clamp(usable_rect.x, right - min_width);
    }
    if matches!(
        handle,
        PanelResizeHandle::Right | PanelResizeHandle::TopRight | PanelResizeHandle::BottomRight
    ) {
        right = right.clamp(left + min_width, usable_right);
    }
    if matches!(
        handle,
        PanelResizeHandle::Top | PanelResizeHandle::TopLeft | PanelResizeHandle::TopRight
    ) {
        top = top.clamp(usable_rect.y, bottom - min_height);
    }
    if matches!(
        handle,
        PanelResizeHandle::Bottom | PanelResizeHandle::BottomLeft | PanelResizeHandle::BottomRight
    ) {
        bottom = bottom.clamp(top + min_height, usable_bottom);
    }

    let resized = FloatingPanelBounds::from_min_size(left, top, right - left, bottom - top);
    clamp_panel_rect(resized, usable_rect)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionMode {
    Select,
    Measure,
    Sculpt(BrushMode),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellPanelKind {
    Tool,
    Inspector,
    Drawer,
}

impl ShellPanelKind {
    pub const ALL: [Self; 3] = [Self::Tool, Self::Inspector, Self::Drawer];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptUtilityControl {
    Radius,
    Strength,
    Falloff,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SculptUtilityDragState {
    pub control: SculptUtilityControl,
    pub anchor_pos: [f32; 2],
    pub initial_value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellPanelPresentation {
    Hidden,
    Floating,
    Docked,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShellWindowState {
    pub presentation: ShellPanelPresentation,
    pub last_floating_rect: Option<FloatingPanelBounds>,
    pub floating_revision: u64,
}

impl ShellWindowState {
    pub const fn new(presentation: ShellPanelPresentation) -> Self {
        Self {
            presentation,
            last_floating_rect: None,
            floating_revision: 0,
        }
    }

    pub const fn is_hidden(&self) -> bool {
        matches!(self.presentation, ShellPanelPresentation::Hidden)
    }

    pub const fn is_floating(&self) -> bool {
        matches!(self.presentation, ShellPanelPresentation::Floating)
    }

    pub const fn is_docked(&self) -> bool {
        matches!(self.presentation, ShellPanelPresentation::Docked)
    }

    pub fn show_floating(&mut self, forced_rect: Option<FloatingPanelBounds>) {
        if let Some(rect) = forced_rect {
            self.last_floating_rect = Some(rect);
            self.floating_revision = self.floating_revision.wrapping_add(1);
        }
        self.presentation = ShellPanelPresentation::Floating;
    }

    pub fn hide(&mut self) {
        self.presentation = ShellPanelPresentation::Hidden;
    }

    pub fn dock(&mut self) {
        self.presentation = ShellPanelPresentation::Docked;
    }

    pub fn remember_floating_rect(&mut self, rect: FloatingPanelBounds) {
        self.last_floating_rect = Some(rect);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrimaryShellState {
    pub interaction_mode: InteractionMode,
    pub tool_panel: ShellWindowState,
    pub inspector_panel: ShellWindowState,
    pub drawer_panel: ShellWindowState,
    pub active_inspector_tab: PrimaryShellInspectorTab,
    pub active_utility_tab: PrimaryShellUtilityTab,
    pub brush_advanced_open: bool,
    pub modeling_commands_open: bool,
    pub tool_rail_visible: bool,
    pub utility_strip_visible: bool,
    pub selection_context_strip_visible: bool,
    pub selection_popup_visible: bool,
    pub sculpt_utility_strip_visible: bool,
    pub sculpt_utility_drag: Option<SculptUtilityDragState>,
    pub layout_revision: u64,
}

impl Default for PrimaryShellState {
    fn default() -> Self {
        Self {
            interaction_mode: InteractionMode::Select,
            tool_panel: ShellWindowState::new(ShellPanelPresentation::Floating),
            inspector_panel: ShellWindowState::new(ShellPanelPresentation::Floating),
            drawer_panel: ShellWindowState::new(ShellPanelPresentation::Hidden),
            active_inspector_tab: PrimaryShellInspectorTab::Properties,
            active_utility_tab: PrimaryShellUtilityTab::History,
            brush_advanced_open: false,
            modeling_commands_open: false,
            tool_rail_visible: true,
            utility_strip_visible: true,
            selection_context_strip_visible: true,
            selection_popup_visible: true,
            sculpt_utility_strip_visible: true,
            sculpt_utility_drag: None,
            layout_revision: 0,
        }
    }
}

impl PrimaryShellState {
    pub fn panel(&self, panel: ShellPanelKind) -> &ShellWindowState {
        match panel {
            ShellPanelKind::Tool => &self.tool_panel,
            ShellPanelKind::Inspector => &self.inspector_panel,
            ShellPanelKind::Drawer => &self.drawer_panel,
        }
    }

    pub fn panel_mut(&mut self, panel: ShellPanelKind) -> &mut ShellWindowState {
        match panel {
            ShellPanelKind::Tool => &mut self.tool_panel,
            ShellPanelKind::Inspector => &mut self.inspector_panel,
            ShellPanelKind::Drawer => &mut self.drawer_panel,
        }
    }

    pub fn toggle_tool_rail(&mut self) {
        self.tool_rail_visible = !self.tool_rail_visible;
    }

    pub fn reset_layout(&mut self) {
        let next_revision = self.layout_revision.wrapping_add(1);
        *self = Self::default();
        self.layout_revision = next_revision;
    }
}

impl MultiTransformSessionState {
    pub fn reset_for_selection(
        &mut self,
        selection_ids: &[NodeId],
        selection_behavior: SelectionBehaviorSettings,
    ) -> bool {
        let mut normalized_selection = selection_ids.to_vec();
        normalized_selection.sort_unstable();
        normalized_selection.dedup();

        if self.selection_key != normalized_selection || self.behavior_key != selection_behavior {
            self.selection_key = normalized_selection;
            self.behavior_key = selection_behavior;
            self.baseline_selection = None;
            self.position_delta = Vec3::ZERO;
            self.rotation_delta_deg = Vec3::ZERO;
            self.scale_factor = Vec3::ONE;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScenePanelUiState {
    pub expanded_nodes: HashSet<NodeId>,
    pub filter_query: String,
    pub renaming_node: Option<NodeId>,
    pub rename_buffer: String,
    pub drag_source: Option<NodeId>,
    pub drop_target: Option<NodeId>,
}

impl ScenePanelUiState {
    pub fn is_expanded(&self, id: NodeId, depth: usize) -> bool {
        depth == 0 || self.expanded_nodes.contains(&id)
    }

    pub fn set_expanded(&mut self, id: NodeId, expanded: bool) {
        if expanded {
            self.expanded_nodes.insert(id);
        } else {
            self.expanded_nodes.remove(&id);
        }
    }

    pub fn begin_rename(&mut self, id: NodeId, current_name: impl Into<String>) {
        self.renaming_node = Some(id);
        self.rename_buffer = current_name.into();
    }

    pub fn cancel_rename(&mut self) {
        self.renaming_node = None;
        self.rename_buffer.clear();
    }

    pub fn begin_drag(&mut self, id: NodeId) {
        self.drag_source = Some(id);
        self.drop_target = None;
    }

    pub fn clear_drag(&mut self) {
        self.drag_source = None;
        self.drop_target = None;
    }
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

/// State for the import mesh dialog shown after picking a file.
pub struct ImportDialog {
    /// The loaded triangle mesh (shared with the voxelize thread when committed).
    pub mesh: Arc<TriMesh>,
    /// Display filename (e.g. "bunny.obj").
    pub filename: String,
    /// User-chosen voxel resolution.
    pub resolution: u32,
    /// Auto-calculated resolution suggestion based on mesh stats.
    pub auto_resolution: u32,
    /// Whether the user is using the auto-calculated resolution.
    pub use_auto: bool,
    /// Mesh vertex count (cached for display).
    pub vertex_count: usize,
    /// Mesh triangle count (cached for display).
    pub triangle_count: usize,
    /// Mesh bounding box dimensions (cached for display).
    pub bounds_size: Vec3,
}

impl ImportDialog {
    pub fn new(mesh: TriMesh, filename: String, max_resolution: u32) -> Self {
        let vertex_count = mesh.vertices.len();
        let triangle_count = mesh.triangles.len();

        // Compute mesh bounds for display
        let mut mesh_min = Vec3::splat(f32::MAX);
        let mut mesh_max = Vec3::splat(f32::MIN);
        for v in &mesh.vertices {
            mesh_min = mesh_min.min(*v);
            mesh_max = mesh_max.max(*v);
        }
        let bounds_size = mesh_max - mesh_min;

        // Auto-calculate resolution: scale with cube root of triangle count.
        // ~2.5x multiplier gives reasonable results:
        //   1K tris → 25 → clamped to 32
        //   8K tris → 50
        //  64K tris → 100
        // 500K tris → 198
        let auto_resolution = ((triangle_count as f32).cbrt() * 2.5)
            .round()
            .clamp(32.0, max_resolution as f32) as u32;

        Self {
            mesh: Arc::new(mesh),
            filename,
            resolution: auto_resolution,
            auto_resolution,
            use_auto: true,
            vertex_count,
            triangle_count,
            bounds_size,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneSelectionState {
    pub selected: Option<NodeId>,
    pub selected_set: HashSet<NodeId>,
}

impl SceneSelectionState {
    pub fn select_single(&mut self, id: NodeId) {
        self.selected = Some(id);
        self.selected_set.clear();
        self.selected_set.insert(id);
    }

    pub fn toggle_select(&mut self, id: NodeId) {
        if self.selected_set.remove(&id) {
            if self.selected == Some(id) {
                self.selected = self.selected_set.iter().copied().min();
            }
        } else {
            self.selected_set.insert(id);
            self.selected = Some(id);
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
        self.selected_set.clear();
    }

    #[cfg(test)]
    pub fn is_selected(&self, id: NodeId) -> bool {
        self.selected_set.contains(&id)
    }

    #[cfg(test)]
    pub fn selected_count(&self) -> usize {
        self.selected_set.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneGraphViewState {
    pub needs_initial_rebuild: bool,
    pub pending_center_node: Option<NodeId>,
}

impl Default for SceneGraphViewState {
    fn default() -> Self {
        Self {
            needs_initial_rebuild: true,
            pending_center_node: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewportPrimaryDragMode {
    #[default]
    None,
    Orbit,
    Sculpt,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportInteractionState {
    pub last_pointer_pos_physical: Option<[f32; 2]>,
    pub primary_press_origin_physical: Option<[f32; 2]>,
    pub primary_drag_distance: f32,
    pub primary_drag_mode: ViewportPrimaryDragMode,
}

impl Default for ViewportInteractionState {
    fn default() -> Self {
        Self {
            last_pointer_pos_physical: None,
            primary_press_origin_physical: None,
            primary_drag_distance: 0.0,
            primary_drag_mode: ViewportPrimaryDragMode::None,
        }
    }
}

pub struct UiState {
    pub primary_shell: PrimaryShellState,
    pub workspace: WorkspaceUiState,
    pub expert_panels: ExpertPanelRegistry,
    pub menu: MenuUiState,
    pub panel_framework: PanelFrameworkState,
    pub scene_panel: ScenePanelUiState,
    pub selection: SceneSelectionState,
    pub scene_graph_view: SceneGraphViewState,
    pub viewport_interaction: ViewportInteractionState,
    pub show_debug: bool,
    pub show_help: bool,
    pub show_export_dialog: bool,
    pub show_settings: bool,
    pub isolation_state: Option<IsolationState>,
    pub toasts: Vec<Toast>,
    pub turntable_active: bool,
    pub property_clipboard: Option<PropertyClipboard>,
    pub command_palette_open: bool,
    pub command_palette_query: String,
    pub command_palette_selected: usize,
    /// Open "Convert to Sculpt" dialog state (None = hidden).
    pub sculpt_convert_dialog: Option<SculptConvertDialog>,
    /// Open "Import Mesh" settings dialog state (None = hidden).
    pub import_dialog: Option<ImportDialog>,
    /// Keybinding editor: which action is currently waiting for a key press (None = not rebinding).
    pub rebinding_action: Option<crate::keymap::ActionBinding>,
    /// Set of Light NodeIds currently active on GPU (nearest MAX_SCENE_LIGHTS to camera).
    pub active_light_ids: HashSet<NodeId>,
    /// Total number of visible lights in the scene (for >8 warning).
    pub total_light_count: usize,
    /// The last total light count that triggered a toast warning.
    /// Used to avoid spamming the same warning repeatedly.
    pub last_light_warning_count: Option<usize>,
    /// Startup crash recovery modal visibility.
    pub show_recovery_dialog: bool,
    /// Recovery modal context built from autosave.meta.
    pub recovery_summary: String,
    /// Reference images used as modeling guides in the viewport.
    pub reference_images: ReferenceImageStore,
    /// Active Blender-style modal brush adjustment (`F` / `Shift+F`).
    pub sculpt_brush_adjust: Option<SculptBrushAdjustState>,
    /// Show a cursor-relative SDF distance readout in the viewport.
    pub show_distance_readout: bool,
    /// Interactive two-point measurement mode.
    pub measurement_mode: bool,
    /// Measurement points in world space (0, 1, or 2 points).
    pub measurement_points: Vec<Vec3>,
    /// Batch-edit inputs for multi-selection transforms in the Properties panel.
    pub multi_transform_edit: MultiTransformSessionState,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_import::TriMesh;
    use crate::settings::{GroupRotateDirection, SelectionBehaviorSettings};

    fn usable_rect(width: f32, height: f32) -> FloatingPanelBounds {
        FloatingPanelBounds::from_min_size(0.0, 0.0, width, height)
    }

    /// Create a simple test mesh (two triangles forming a 2x2x0 quad).
    fn test_quad_mesh(num_triangles: usize) -> TriMesh {
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut triangles = vec![[0, 1, 2], [0, 2, 3]];
        // Pad with duplicate triangles to reach desired count
        while triangles.len() < num_triangles {
            let base = vertices.len() as u32;
            vertices.push(Vec3::new(0.0, 0.0, 0.0));
            vertices.push(Vec3::new(1.0, 0.0, 0.0));
            vertices.push(Vec3::new(0.0, 1.0, 0.0));
            triangles.push([base, base + 1, base + 2]);
        }
        TriMesh {
            vertices,
            triangles,
        }
    }

    #[test]
    fn import_dialog_auto_resolution_small_mesh() {
        let mesh = test_quad_mesh(100);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(100) * 2.5 = 4.64 * 2.5 = 11.6 → clamped to 32
        assert_eq!(dialog.auto_resolution, 32);
        assert_eq!(dialog.resolution, 32);
        assert!(dialog.use_auto);
    }

    #[test]
    fn import_dialog_auto_resolution_medium_mesh() {
        let mesh = test_quad_mesh(8000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(8000) = 20, * 2.5 = 50
        assert_eq!(dialog.auto_resolution, 50);
    }

    #[test]
    fn import_dialog_auto_resolution_large_mesh() {
        let mesh = test_quad_mesh(64000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(64000) = 40, * 2.5 = 100
        assert_eq!(dialog.auto_resolution, 100);
    }

    #[test]
    fn import_dialog_auto_resolution_clamped_to_max() {
        let mesh = test_quad_mesh(500_000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 128);
        // cbrt(500000) * 2.5 ≈ 198, but max is 128
        assert_eq!(dialog.auto_resolution, 128);
    }

    #[test]
    fn import_dialog_preserves_mesh_stats() {
        let mesh = test_quad_mesh(2);
        let vert_count = mesh.vertices.len();
        let tri_count = mesh.triangles.len();
        let dialog = ImportDialog::new(mesh, "monkey.obj".into(), 320);
        assert_eq!(dialog.vertex_count, vert_count);
        assert_eq!(dialog.triangle_count, tri_count);
        assert_eq!(dialog.filename, "monkey.obj");
    }

    #[test]
    fn import_dialog_bounds_size_computed() {
        let mesh = TriMesh {
            vertices: vec![
                Vec3::new(-1.0, -2.0, -3.0),
                Vec3::new(4.0, 5.0, 6.0),
                Vec3::new(0.0, 0.0, 0.0),
            ],
            triangles: vec![[0, 1, 2]],
        };
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        assert!((dialog.bounds_size.x - 5.0).abs() < f32::EPSILON);
        assert!((dialog.bounds_size.y - 7.0).abs() < f32::EPSILON);
        assert!((dialog.bounds_size.z - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn import_dialog_defaults_to_auto() {
        let mesh = test_quad_mesh(1000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        assert!(dialog.use_auto);
        assert_eq!(dialog.resolution, dialog.auto_resolution);
    }

    #[test]
    fn multi_transform_edit_state_resets_when_selection_changes() {
        let mut state = MultiTransformSessionState::default();
        state.reset_for_selection(&[1, 2], SelectionBehaviorSettings::default());
        state.position_delta = Vec3::new(1.0, 2.0, 3.0);
        state.rotation_delta_deg = Vec3::new(10.0, 20.0, 30.0);
        state.scale_factor = Vec3::splat(2.0);

        state.reset_for_selection(&[1, 2], SelectionBehaviorSettings::default());
        assert_eq!(state.position_delta, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(state.rotation_delta_deg, Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(state.scale_factor, Vec3::splat(2.0));

        state.reset_for_selection(&[2, 3], SelectionBehaviorSettings::default());
        assert_eq!(state.position_delta, Vec3::ZERO);
        assert_eq!(state.rotation_delta_deg, Vec3::ZERO);
        assert_eq!(state.scale_factor, Vec3::ONE);
        assert!(state.baseline_selection.is_none());
    }

    #[test]
    fn multi_transform_edit_state_ignores_selection_order() {
        let mut state = MultiTransformSessionState::default();
        state.reset_for_selection(&[1, 2], SelectionBehaviorSettings::default());
        state.position_delta = Vec3::new(1.0, 2.0, 3.0);

        state.reset_for_selection(&[2, 1], SelectionBehaviorSettings::default());
        assert_eq!(state.position_delta, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(state.selection_key, vec![1, 2]);
    }

    #[test]
    fn multi_transform_edit_state_resets_when_behavior_changes() {
        let mut state = MultiTransformSessionState::default();
        state.reset_for_selection(&[1, 2], SelectionBehaviorSettings::default());
        state.position_delta = Vec3::new(1.0, 2.0, 3.0);

        let mut updated_behavior = SelectionBehaviorSettings::default();
        updated_behavior.group_rotate_direction = GroupRotateDirection::Inverted;
        state.reset_for_selection(&[1, 2], updated_behavior);

        assert_eq!(state.position_delta, Vec3::ZERO);
        assert_eq!(state.rotation_delta_deg, Vec3::ZERO);
        assert_eq!(state.scale_factor, Vec3::ONE);
        assert_eq!(state.behavior_key, updated_behavior);
    }

    #[test]
    fn primary_shell_defaults_to_viewport_first_layout() {
        let state = PrimaryShellState::default();
        assert_eq!(state.interaction_mode, InteractionMode::Select);
        assert!(state.tool_panel.is_floating());
        assert!(state.inspector_panel.is_floating());
        assert!(state.drawer_panel.is_hidden());
        assert_eq!(
            state.active_inspector_tab,
            PrimaryShellInspectorTab::Properties
        );
        assert_eq!(state.active_utility_tab, PrimaryShellUtilityTab::History);
        assert!(state.tool_rail_visible);
        assert!(state.utility_strip_visible);
        assert!(state.selection_context_strip_visible);
        assert!(state.selection_popup_visible);
        assert!(state.sculpt_utility_strip_visible);
    }

    #[test]
    fn primary_shell_tool_rail_toggle_flips_visibility() {
        let mut state = PrimaryShellState::default();

        state.toggle_tool_rail();
        assert!(!state.tool_rail_visible);

        state.toggle_tool_rail();
        assert!(state.tool_rail_visible);
    }

    #[test]
    fn primary_shell_reset_layout_restores_window_defaults() {
        let mut state = PrimaryShellState::default();
        state.tool_panel.hide();
        state.inspector_panel.hide();
        state.drawer_panel.show_floating(None);
        state.active_inspector_tab = PrimaryShellInspectorTab::Display;
        state.active_utility_tab = PrimaryShellUtilityTab::Reference;
        state.brush_advanced_open = true;
        state.modeling_commands_open = true;
        state.tool_rail_visible = false;
        state.utility_strip_visible = false;
        state.selection_context_strip_visible = false;
        state.selection_popup_visible = false;
        state.sculpt_utility_strip_visible = false;
        state.sculpt_utility_drag = Some(SculptUtilityDragState {
            control: SculptUtilityControl::Radius,
            anchor_pos: [10.0, 10.0],
            initial_value: 0.5,
        });
        state.layout_revision = 9;

        state.reset_layout();

        assert!(state.tool_panel.is_floating());
        assert!(state.inspector_panel.is_floating());
        assert!(state.drawer_panel.is_hidden());
        assert_eq!(
            state.active_inspector_tab,
            PrimaryShellInspectorTab::Properties
        );
        assert_eq!(state.active_utility_tab, PrimaryShellUtilityTab::History);
        assert!(!state.brush_advanced_open);
        assert!(!state.modeling_commands_open);
        assert!(state.tool_rail_visible);
        assert!(state.utility_strip_visible);
        assert!(state.selection_context_strip_visible);
        assert!(state.selection_popup_visible);
        assert!(state.sculpt_utility_strip_visible);
        assert!(state.sculpt_utility_drag.is_none());
        assert_eq!(state.layout_revision, 10);
    }

    #[test]
    fn scene_selection_select_single_sets_primary_and_set() {
        let mut state = SceneSelectionState::default();

        state.select_single(5);

        assert_eq!(state.selected, Some(5));
        assert_eq!(state.selected_count(), 1);
        assert!(state.is_selected(5));
    }

    #[test]
    fn scene_selection_toggle_select_removes_primary_to_lowest_remaining() {
        let mut state = SceneSelectionState::default();
        state.select_single(2);
        state.toggle_select(4);
        state.toggle_select(2);

        assert_eq!(state.selected, Some(4));
        assert_eq!(state.selected_count(), 1);
        assert!(state.is_selected(4));
    }

    #[test]
    fn scene_selection_clear_selection_empties_primary_and_set() {
        let mut state = SceneSelectionState::default();
        state.select_single(3);

        state.clear_selection();

        assert_eq!(state.selected, None);
        assert_eq!(state.selected_count(), 0);
    }

    #[test]
    fn scene_graph_view_defaults_to_initial_rebuild() {
        let state = SceneGraphViewState::default();

        assert!(state.needs_initial_rebuild);
        assert!(state.pending_center_node.is_none());
    }

    #[test]
    fn menu_ui_state_toggle_dropdown_is_single_open() {
        let mut state = MenuUiState::default();

        state.toggle_dropdown(MenuDropdownKind::File);
        assert_eq!(state.active_dropdown, Some(MenuDropdownKind::File));
        assert!(!state.settings_card_open);

        state.toggle_dropdown(MenuDropdownKind::View);
        assert_eq!(state.active_dropdown, Some(MenuDropdownKind::View));
        assert!(!state.settings_card_open);
    }

    #[test]
    fn menu_ui_state_toggle_same_dropdown_closes_it() {
        let mut state = MenuUiState::default();
        state.open_dropdown(MenuDropdownKind::Edit);

        state.toggle_dropdown(MenuDropdownKind::Edit);

        assert_eq!(state.active_dropdown, None);
        assert!(!state.settings_card_open);
    }

    #[test]
    fn menu_ui_state_settings_card_is_exclusive_with_dropdowns() {
        let mut state = MenuUiState::default();
        state.open_dropdown(MenuDropdownKind::Help);

        state.open_settings_card();
        assert!(state.settings_card_open);
        assert_eq!(state.active_dropdown, None);

        state.open_dropdown(MenuDropdownKind::File);
        assert_eq!(state.active_dropdown, Some(MenuDropdownKind::File));
        assert!(!state.settings_card_open);
    }

    #[test]
    fn menu_ui_state_dismiss_all_clears_every_surface() {
        let mut state = MenuUiState::default();
        state.open_settings_card();
        state.open_dropdown(MenuDropdownKind::View);
        state.open_settings_card();

        state.dismiss_all();

        assert_eq!(state.active_dropdown, None);
        assert!(!state.settings_card_open);
    }

    #[test]
    fn panel_framework_open_and_swap_transient_panel() {
        let mut state = PanelFrameworkState::default();

        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        assert_eq!(
            state.active_transient(PanelBarId::PrimaryRight),
            Some(PanelKind::ObjectProperties)
        );

        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        assert_eq!(
            state.active_transient(PanelBarId::PrimaryRight),
            Some(PanelKind::RenderSettings)
        );
        assert!(state.pinned_instances.is_empty());
    }

    #[test]
    fn panel_framework_toggle_closes_active_transient_panel() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        state.toggle_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        assert_eq!(state.active_transient(PanelBarId::PrimaryRight), None);
    }

    #[test]
    fn panel_framework_pin_keeps_panel_persistent_and_floating() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        state.pin_panel(PanelKind::ObjectProperties);

        assert_eq!(state.active_transient(PanelBarId::PrimaryRight), None);
        let instance = state
            .pinned_instance(PanelKind::ObjectProperties)
            .expect("pinned object properties panel");
        assert!(instance.pinned);
        assert_eq!(instance.anchor_bar, PanelBarId::PrimaryRight);
        assert_eq!(instance.presentation, PanelPresentation::PinnedFloating);
    }

    #[test]
    fn panel_framework_reopening_pinned_panel_focuses_existing_instance() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        let instance_id = state
            .pinned_instance(PanelKind::RenderSettings)
            .expect("pinned render settings")
            .id;

        state.toggle_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        assert_eq!(state.pinned_instances.len(), 1);
        assert_eq!(
            state
                .pinned_instance(PanelKind::RenderSettings)
                .expect("render settings still pinned")
                .id,
            instance_id
        );
        assert_eq!(state.focus_order.last().copied(), Some(instance_id));
    }

    #[test]
    fn panel_framework_edge_maps_to_sheet_anchor() {
        let mut state = PanelFrameworkState::default();
        state
            .bar_mut(PanelBarId::PrimaryRight)
            .expect("primary bar")
            .edge = PanelBarEdge::Top;
        assert_eq!(
            state.sheet_anchor_for_bar(PanelBarId::PrimaryRight),
            PanelSheetAnchor::Below
        );

        state
            .bar_mut(PanelBarId::PrimaryRight)
            .expect("primary bar")
            .edge = PanelBarEdge::Left;
        assert_eq!(
            state.sheet_anchor_for_bar(PanelBarId::PrimaryRight),
            PanelSheetAnchor::Right
        );
    }

    #[test]
    fn panel_framework_drag_updates_and_clamps_transient_rect() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );

        state.update_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            -5000.0,
            5000.0,
            usable_rect(1200.0, 800.0),
        );

        let rect = state
            .remembered_transient_rect(PanelBarId::PrimaryRight)
            .expect("remembered rect");
        assert!(rect.x >= PanelFrameworkState::TRANSIENT_PANEL_MARGIN);
        assert!(rect.y >= PanelFrameworkState::TRANSIENT_PANEL_MARGIN);
        assert!(rect.right() <= 1200.0 - PanelFrameworkState::TRANSIENT_PANEL_MARGIN + 0.001);
        assert!(
            rect.y + rect.height <= 800.0 - PanelFrameworkState::TRANSIENT_PANEL_MARGIN + 0.001
        );
    }

    #[test]
    fn panel_framework_close_preserves_remembered_transient_rect() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );
        state.update_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            -40.0,
            22.0,
            usable_rect(1200.0, 800.0),
        );
        let remembered = state.remembered_transient_rect(PanelBarId::PrimaryRight);

        state.close_panel(PanelKind::ObjectProperties);

        assert_eq!(state.active_transient(PanelBarId::PrimaryRight), None);
        assert_eq!(
            state.remembered_transient_rect(PanelBarId::PrimaryRight),
            remembered
        );
    }

    #[test]
    fn panel_framework_switching_transient_kind_preserves_rect() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );
        state.update_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            -80.0,
            32.0,
            usable_rect(1200.0, 800.0),
        );
        let remembered = state.remembered_transient_rect(PanelBarId::PrimaryRight);

        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        assert_eq!(
            state.active_transient(PanelBarId::PrimaryRight),
            Some(PanelKind::RenderSettings)
        );
        assert_eq!(
            state.remembered_transient_rect(PanelBarId::PrimaryRight),
            remembered
        );
    }

    #[test]
    fn panel_framework_unpin_reopens_at_remembered_transient_rect() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );
        state.update_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            -60.0,
            28.0,
            usable_rect(1200.0, 800.0),
        );
        let remembered = state.remembered_transient_rect(PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);

        state.unpin_panel(PanelKind::RenderSettings);

        assert_eq!(
            state.active_transient(PanelBarId::PrimaryRight),
            Some(PanelKind::RenderSettings)
        );
        assert_eq!(
            state.remembered_transient_rect(PanelBarId::PrimaryRight),
            remembered
        );
    }

    #[test]
    fn panel_framework_drag_updates_pinned_panel_rect() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);

        let before = state
            .resolved_panel_rect(
                PanelKind::RenderSettings,
                PanelBarId::PrimaryRight,
                usable_rect(1200.0, 800.0),
            )
            .expect("pinned panel rect");

        state.begin_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );
        state.update_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            -80.0,
            24.0,
            usable_rect(1200.0, 800.0),
        );

        let after = state
            .pinned_instance(PanelKind::RenderSettings)
            .and_then(|instance| instance.rect)
            .expect("stored pinned rect");
        assert!((after.x - (before.x - 80.0)).abs() < 0.01);
        assert!((after.y - (before.y + 24.0)).abs() < 0.01);
    }

    #[test]
    fn panel_framework_resize_updates_transient_rect_and_clamps_bounds() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Resize(PanelResizeHandle::BottomRight),
        );

        state.update_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            4000.0,
            4000.0,
            usable_rect(900.0, 700.0),
        );

        let rect = state
            .remembered_transient_rect(PanelBarId::PrimaryRight)
            .expect("resized transient rect");
        assert!(rect.width <= 860.0);
        assert!(rect.height <= 660.0);
        assert!(rect.right() <= 880.0 + 0.01);
        assert!(rect.bottom() <= 680.0 + 0.01);
    }

    #[test]
    fn panel_framework_collapsed_pinned_panel_uses_display_height() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        state.toggle_pinned_collapsed(PanelKind::RenderSettings);

        let rect = state
            .resolved_display_panel_rect(
                PanelKind::RenderSettings,
                PanelBarId::PrimaryRight,
                usable_rect(1200.0, 800.0),
            )
            .expect("collapsed pinned panel rect");

        assert!((rect.width - PanelFrameworkState::COLLAPSED_PANEL_WIDTH).abs() < 0.01);
        assert!((rect.height - PanelFrameworkState::COLLAPSED_PANEL_HEIGHT).abs() < 0.01);
    }

    #[test]
    fn panel_framework_collapsed_pinned_panel_clamps_with_collapsed_footprint() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        if let Some(instance) = state.pinned_instance_mut(PanelKind::RenderSettings) {
            instance.rect = Some(FloatingPanelBounds::from_min_size(
                700.0, 40.0, 500.0, 420.0,
            ));
            instance.collapsed = true;
        }

        let rect = state
            .resolved_display_panel_rect(
                PanelKind::RenderSettings,
                PanelBarId::PrimaryRight,
                usable_rect(1000.0, 800.0),
            )
            .expect("collapsed pinned panel rect");
        let expected_x = 1000.0
            - PanelFrameworkState::TRANSIENT_PANEL_MARGIN
            - PanelFrameworkState::COLLAPSED_PANEL_WIDTH;

        assert!((rect.x - expected_x).abs() < 0.01);
        assert!((rect.width - PanelFrameworkState::COLLAPSED_PANEL_WIDTH).abs() < 0.01);
        assert!((rect.height - PanelFrameworkState::COLLAPSED_PANEL_HEIGHT).abs() < 0.01);
    }

    #[test]
    fn panel_framework_moving_collapsed_pinned_panel_preserves_expanded_rect_height() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        let before = state
            .pinned_instance(PanelKind::RenderSettings)
            .and_then(|instance| instance.rect)
            .unwrap_or_else(|| {
                state.resolved_transient_rect(PanelBarId::PrimaryRight, usable_rect(1200.0, 800.0))
            });
        state.toggle_pinned_collapsed(PanelKind::RenderSettings);

        state.begin_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Move,
        );
        state.update_panel_interaction(
            PanelKind::RenderSettings,
            PanelBarId::PrimaryRight,
            0.0,
            40.0,
            usable_rect(1200.0, 800.0),
        );

        let after = state
            .pinned_instance(PanelKind::RenderSettings)
            .and_then(|instance| instance.rect)
            .expect("stored pinned rect");
        assert!((after.height - before.height).abs() < 0.01);
    }
}
