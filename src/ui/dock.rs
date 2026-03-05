use eframe::egui;
use egui_dock::{DockState, Node, NodeIndex, Split, TabViewer};
use glam::Vec3;

use crate::app::actions::ActionSink;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::{ActiveTool, SculptState};
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::{self, NodeGraphState};
use crate::settings::Settings;
use crate::graph::history::History;
use crate::ui::{brush_settings, history_panel, properties, render_settings, scene_tree, viewport};

#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    Viewport,
    NodeGraph,
    Properties,
    SceneTree,
    RenderSettings,
    History,
    BrushSettings,
}

impl Tab {
    pub const ALL: &[Tab] = &[
        Tab::Viewport,
        Tab::NodeGraph,
        Tab::Properties,
        Tab::SceneTree,
        Tab::RenderSettings,
        Tab::History,
        Tab::BrushSettings,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Tab::Viewport => "Viewport",
            Tab::NodeGraph => "Node Graph",
            Tab::Properties => "Properties",
            Tab::SceneTree => "Scene Tree",
            Tab::RenderSettings => "Render Settings",
            Tab::History => "History",
            Tab::BrushSettings => "Brush Settings",
        }
    }
}

pub fn create_dock_state() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();

    let [center, right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.8,
        Node::leaf_with(vec![Tab::Properties, Tab::RenderSettings]),
    );

    let [_props, _tree] = surface.split(
        right,
        Split::Below,
        0.5,
        Node::leaf_with(vec![Tab::SceneTree, Tab::History]),
    );

    let [_viewport, _graph] = surface.split(
        center,
        Split::Below,
        0.7,
        Node::leaf(Tab::NodeGraph),
    );

    state
}

/// Sculpting layout: large viewport, brush settings + properties on right.
pub fn create_dock_sculpting() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();
    let [_center, right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.82,
        Node::leaf_with(vec![Tab::BrushSettings, Tab::Properties]),
    );
    let [_brush, _tree] = surface.split(
        right,
        Split::Below,
        0.6,
        Node::leaf_with(vec![Tab::SceneTree]),
    );
    state
}

/// Rendering layout: viewport + render settings side by side.
pub fn create_dock_rendering() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();
    let [_center, _right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.75,
        Node::leaf_with(vec![Tab::RenderSettings, Tab::Properties]),
    );
    state
}

// ---------------------------------------------------------------------------
// Context bundles — group related refs so SdfTabViewer stays manageable.
// ---------------------------------------------------------------------------

/// Refs needed only by the viewport tab.
pub struct ViewportContext<'a> {
    pub gizmo_state: &'a mut GizmoState,
    pub gizmo_mode: &'a GizmoMode,
    pub gizmo_space: &'a GizmoSpace,
    pub pivot_offset: &'a mut Vec3,
    pub pending_pick: &'a mut Option<PendingPick>,
    pub sculpt_count: usize,
    pub fps_info: Option<(f64, f64)>,
    /// Modifier keys captured during sculpt drag (output channel).
    pub sculpt_ctrl_held: &'a mut bool,
    pub sculpt_shift_held: &'a mut bool,
    /// Last sculpt hit world position (for drag interpolation). None = no active brush.
    pub last_sculpt_hit: Option<Vec3>,
    /// Pen pressure during sculpt drag.
    pub sculpt_pressure: &'a mut f32,
    /// Label for isolation mode indicator (None = not isolated).
    pub isolation_label: Option<String>,
    /// Whether turntable rotation is active.
    pub turntable_active: bool,
    /// Output: whether the submitted pick is hover-only (no brush application).
    pub is_hover_pick: &'a mut bool,
    /// Hover world position for 3D brush preview (from hover picks).
    pub hover_world_pos: Option<Vec3>,
    /// Whether cursor is currently over geometry (from last hover pick).
    pub cursor_over_geometry: bool,
}

/// Refs needed only by the scene tree tab.
pub struct SceneTreeContext<'a> {
    pub renaming_node: &'a mut Option<NodeId>,
    pub rename_buf: &'a mut String,
    pub drag_state: &'a mut Option<NodeId>,
    pub search_filter: &'a mut String,
}

// ---------------------------------------------------------------------------
// Tab viewer
// ---------------------------------------------------------------------------

pub struct SdfTabViewer<'a> {
    pub camera: &'a mut Camera,
    pub scene: &'a mut Scene,
    pub node_graph_state: &'a mut NodeGraphState,
    pub active_tool: &'a ActiveTool,
    pub sculpt_state: &'a mut SculptState,
    pub settings: &'a mut Settings,
    pub time: f32,
    /// (done_slices, total_slices) when a bake is in progress, None when idle.
    pub bake_progress: Option<(u32, u32)>,
    /// Viewport-specific refs (gizmo, pick, fps overlay).
    pub viewport: ViewportContext<'a>,
    /// Scene tree-specific refs (rename, drag & drop).
    pub scene_tree: SceneTreeContext<'a>,
    /// Action sink — structural mutations flow through here.
    pub actions: &'a mut ActionSink,
    /// History reference for the history panel tab.
    pub history: &'a History,
}

impl<'a> TabViewer for SdfTabViewer<'a> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "Viewport".into(),
            Tab::NodeGraph => "Node Graph".into(),
            Tab::Properties => "Properties".into(),
            Tab::SceneTree => "Scene Tree".into(),
            Tab::RenderSettings => "Render Settings".into(),
            Tab::History => "History".into(),
            Tab::BrushSettings => "Brush Settings".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewport => {
                let vp_output = viewport::draw(
                    ui,
                    self.camera,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.viewport.gizmo_state,
                    self.viewport.gizmo_mode,
                    self.viewport.gizmo_space,
                    self.viewport.pivot_offset,
                    self.sculpt_state,
                    self.active_tool,
                    self.time,
                    &self.settings.render,
                    self.viewport.sculpt_count,
                    self.viewport.fps_info,
                    self.actions,
                    &self.settings.snap,
                    self.viewport.isolation_label.as_deref(),
                    self.viewport.turntable_active,
                    self.viewport.last_sculpt_hit,
                    self.viewport.hover_world_pos,
                    self.viewport.cursor_over_geometry,
                );
                if let Some(pick) = vp_output.pending_pick {
                    *self.viewport.pending_pick = Some(pick);
                    *self.viewport.is_hover_pick = vp_output.is_hover_pick;
                    *self.viewport.sculpt_ctrl_held = vp_output.sculpt_ctrl_held;
                    *self.viewport.sculpt_shift_held = vp_output.sculpt_shift_held;
                    *self.viewport.sculpt_pressure = vp_output.sculpt_pressure;
                }
                // Apply Ctrl+right-drag brush resize/strength adjustments
                if vp_output.brush_radius_delta != 0.0 || vp_output.brush_strength_delta != 0.0 {
                    if let crate::sculpt::SculptState::Active {
                        ref mut brush_radius,
                        ref mut brush_strength,
                        ..
                    } = self.sculpt_state
                    {
                        *brush_radius = (*brush_radius + vp_output.brush_radius_delta).clamp(0.05, 2.0);
                        *brush_strength = (*brush_strength + vp_output.brush_strength_delta).clamp(0.01, 3.0);
                    }
                }
            }
            Tab::NodeGraph => {
                node_graph::draw(ui, self.scene, self.node_graph_state, self.actions);
            }
            Tab::Properties => {
                properties::draw(
                    ui,
                    self.scene,
                    self.node_graph_state.selected,
                    self.sculpt_state,
                    self.bake_progress,
                    self.actions,
                );
                // Defensive: clear selection if the node was deleted by properties panel
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.needs_initial_rebuild = true;
                    }
                }
            }
            Tab::SceneTree => {
                let prev_selected = self.node_graph_state.selected;
                scene_tree::draw(
                    ui,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.scene_tree.renaming_node,
                    self.scene_tree.rename_buf,
                    self.scene_tree.drag_state,
                    self.actions,
                    self.scene_tree.search_filter,
                );
                // If scene tree changed selection, scroll graph to it
                if self.node_graph_state.selected != prev_selected {
                    if let Some(sel) = self.node_graph_state.selected {
                        self.node_graph_state.pending_center_node = Some(sel);
                    }
                }
                // Defensive: mark layout dirty if a node was deleted via context menu
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.selected = None;
                        self.node_graph_state.needs_initial_rebuild = true;
                    }
                }
            }
            Tab::RenderSettings => {
                render_settings::draw(ui, self.settings, self.actions);
            }
            Tab::History => {
                history_panel::draw(ui, self.history, self.actions);
            }
            Tab::BrushSettings => {
                brush_settings::draw(ui, self.sculpt_state);
            }
        }
    }
}
