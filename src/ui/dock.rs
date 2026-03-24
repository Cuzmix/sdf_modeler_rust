use eframe::egui;
use egui_dock::{DockState, Node, NodeIndex, Split, TabViewer};
use glam::Vec3;

use crate::app::actions::ActionSink;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeId, Scene};
use crate::sculpt::{ActiveTool, SculptState};
use crate::settings::Settings;
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::{self, NodeGraphState};
use crate::ui::{
    brush_settings, history_panel, light_graph, properties, render_settings, scene_tree, viewport,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    Viewport,
    ToolPanel,
    InspectorPanel,
    DrawerPanel,
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

impl Tab {
    pub const EXPERT_TABS: &[Tab] = &[
        Tab::NodeGraph,
        Tab::LightGraph,
        Tab::Properties,
        Tab::ReferenceImages,
        Tab::SceneTree,
        Tab::RenderSettings,
        Tab::History,
        Tab::BrushSettings,
        Tab::Lights,
        Tab::LightLinking,
        Tab::SceneStats,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Tab::Viewport => "Viewport",
            Tab::ToolPanel => "Scene Panel",
            Tab::InspectorPanel => "Inspector",
            Tab::DrawerPanel => "Utilities",
            Tab::NodeGraph => "Node Graph",
            Tab::LightGraph => "Light Graph",
            Tab::Properties => "Properties",
            Tab::ReferenceImages => "Reference Images",
            Tab::SceneTree => "Scene Tree",
            Tab::RenderSettings => "Render Settings",
            Tab::History => "History",
            Tab::BrushSettings => "Brush Settings",
            Tab::Lights => "Lights",
            Tab::LightLinking => "Light Linking",
            Tab::SceneStats => "Scene Stats",
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
        Node::leaf_with(vec![
            Tab::Properties,
            Tab::RenderSettings,
            Tab::ReferenceImages,
        ]),
    );

    let [_props, _tree] = surface.split(
        right,
        Split::Below,
        0.5,
        Node::leaf_with(vec![
            Tab::SceneTree,
            Tab::History,
            Tab::Lights,
            Tab::LightGraph,
            Tab::SceneStats,
        ]),
    );

    let [_viewport, _graph] = surface.split(center, Split::Below, 0.7, Node::leaf(Tab::NodeGraph));

    state
}

pub fn create_primary_shell_dock() -> DockState<Tab> {
    DockState::new(vec![Tab::Viewport])
}

/// Sculpting layout: large viewport, brush settings + properties on right.
pub fn create_dock_sculpting() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();
    let [_center, right] = surface.split(
        NodeIndex::root(),
        Split::Right,
        0.82,
        Node::leaf_with(vec![
            Tab::BrushSettings,
            Tab::Properties,
            Tab::ReferenceImages,
        ]),
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
        Node::leaf_with(vec![
            Tab::RenderSettings,
            Tab::Properties,
            Tab::ReferenceImages,
        ]),
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
    pub gizmo_visible: bool,
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
    /// Output: whether the viewport gizmo is actively dragging this frame.
    pub gizmo_drag_active: &'a mut bool,
    /// Hover world position for 3D brush preview (from hover picks).
    pub hover_world_pos: Option<Vec3>,
    /// Whether cursor is currently over geometry (from last hover pick).
    pub cursor_over_geometry: bool,
    /// Modal `F` / `Shift+F` brush adjustment state.
    pub sculpt_brush_adjust: &'a mut Option<crate::app::state::SculptBrushAdjustState>,
    /// Currently soloed light node ID (None = no solo).
    pub soloed_light: Option<NodeId>,
    /// Label for solo mode indicator (None = not soloed).
    pub solo_label: Option<String>,
    /// Show cursor distance readout overlay.
    pub show_distance_readout: &'a mut bool,
    /// Multi-selection transform behavior profile.
    pub selection_behavior: crate::settings::SelectionBehaviorSettings,
    /// Two-point measurement tool mode flag.
    pub measurement_mode: &'a mut bool,
    /// Collected measurement points in world space.
    pub measurement_points: &'a mut Vec<Vec3>,
    /// Output viewport rect for primary shell placement defaults.
    pub viewport_rect: &'a mut Option<egui::Rect>,
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
    pub primary_shell: &'a mut crate::app::state::PrimaryShellState,
    pub camera: &'a mut Camera,
    pub scene: &'a mut Scene,
    pub node_graph_state: &'a mut NodeGraphState,
    pub light_graph_state: &'a mut NodeGraphState,
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
    /// Set of Light NodeIds currently active on GPU (nearest to camera).
    pub active_light_ids: &'a std::collections::HashSet<NodeId>,
    /// Material preset library (built-in + user-saved).
    pub material_library: &'a mut crate::material_preset::MaterialLibrary,
    /// Reference image manager used by viewport/properties panels.
    pub reference_images: &'a mut crate::ui::reference_image::ReferenceImageManager,
    /// Batch transform UI state for multi-selection property editing.
    pub multi_transform_edit: &'a mut crate::app::state::MultiTransformSessionState,
    /// Frame timing data for scene stats panel.
    pub timings: &'a crate::app::FrameTimings,
}

impl<'a> TabViewer for SdfTabViewer<'a> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "Viewport".into(),
            Tab::ToolPanel => "Scene Panel".into(),
            Tab::InspectorPanel => "Inspector".into(),
            Tab::DrawerPanel => "Utilities".into(),
            Tab::NodeGraph => "Node Graph".into(),
            Tab::LightGraph => "Light Graph".into(),
            Tab::Properties => "Properties".into(),
            Tab::ReferenceImages => "Reference Images".into(),
            Tab::SceneTree => "Scene Tree".into(),
            Tab::RenderSettings => "Render Settings".into(),
            Tab::History => "History".into(),
            Tab::BrushSettings => "Brush Settings".into(),
            Tab::Lights => "Lights".into(),
            Tab::LightLinking => "Light Linking".into(),
            Tab::SceneStats => "Scene Stats".into(),
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
                    &self.node_graph_state.selected_set,
                    self.multi_transform_edit,
                    self.viewport.gizmo_state,
                    self.viewport.gizmo_mode,
                    self.viewport.gizmo_space,
                    self.viewport.gizmo_visible,
                    self.viewport.pivot_offset,
                    self.sculpt_state,
                    self.active_tool,
                    self.time,
                    &self.settings.render,
                    self.viewport.sculpt_count,
                    self.viewport.fps_info,
                    self.actions,
                    &self.settings.snap,
                    &self.viewport.selection_behavior,
                    self.viewport.isolation_label.as_deref(),
                    self.viewport.turntable_active,
                    self.viewport.last_sculpt_hit,
                    self.viewport.hover_world_pos,
                    self.viewport.cursor_over_geometry,
                    self.viewport.sculpt_brush_adjust,
                    self.active_light_ids,
                    self.viewport.soloed_light,
                    self.viewport.solo_label.as_deref(),
                    &*self.reference_images,
                    self.viewport.show_distance_readout,
                    self.viewport.measurement_mode,
                    self.viewport.measurement_points,
                );
                if let Some(pick) = vp_output.pending_pick {
                    *self.viewport.pending_pick = Some(pick);
                    *self.viewport.is_hover_pick = vp_output.is_hover_pick;
                    *self.viewport.sculpt_ctrl_held = vp_output.sculpt_ctrl_held;
                    *self.viewport.sculpt_shift_held = vp_output.sculpt_shift_held;
                    *self.viewport.sculpt_pressure = vp_output.sculpt_pressure;
                }
                *self.viewport.gizmo_drag_active = vp_output.gizmo_drag_active;
                *self.viewport.viewport_rect = Some(vp_output.viewport_rect);
                // Apply Ctrl+right-drag brush resize/strength adjustments
                if (vp_output.brush_radius_delta != 0.0 || vp_output.brush_strength_delta != 0.0)
                    && self.sculpt_state.is_active()
                {
                    let selected_mode = self.sculpt_state.selected_brush();
                    let profile = self.sculpt_state.selected_profile_mut();
                    profile.radius =
                        (profile.radius + vp_output.brush_radius_delta).clamp(0.05, 2.0);
                    profile.strength += vp_output.brush_strength_delta;
                    profile.clamp_strength_for_mode(selected_mode);
                }
            }
            Tab::ToolPanel => {
                let mut shell_context = crate::ui::primary_shell::PrimaryShellContext {
                    shell: self.primary_shell,
                    dock_state: None,
                    camera: self.camera,
                    scene: self.scene,
                    sculpt_state: self.sculpt_state,
                    selected: &mut self.node_graph_state.selected,
                    selected_set: &mut self.node_graph_state.selected_set,
                    renaming_node: self.scene_tree.renaming_node,
                    rename_buf: self.scene_tree.rename_buf,
                    scene_tree_drag: self.scene_tree.drag_state,
                    scene_tree_search: self.scene_tree.search_filter,
                    bake_progress: self.bake_progress,
                    actions: self.actions,
                    history: self.history,
                    active_light_ids: self.active_light_ids,
                    max_sculpt_resolution: self.settings.max_sculpt_resolution,
                    soloed_light: self.viewport.soloed_light,
                    material_library: self.material_library,
                    multi_transform_edit: self.multi_transform_edit,
                    gizmo_mode: self.viewport.gizmo_mode,
                    gizmo_space: self.viewport.gizmo_space,
                    selection_behavior: &self.viewport.selection_behavior,
                    reference_images: self.reference_images,
                    measurement_points: self.viewport.measurement_points,
                    show_distance_readout: self.viewport.show_distance_readout,
                    settings: self.settings,
                };
                crate::ui::primary_shell::draw_tool_panel_tab(ui, &mut shell_context);
            }
            Tab::InspectorPanel => {
                let mut shell_context = crate::ui::primary_shell::PrimaryShellContext {
                    shell: self.primary_shell,
                    dock_state: None,
                    camera: self.camera,
                    scene: self.scene,
                    sculpt_state: self.sculpt_state,
                    selected: &mut self.node_graph_state.selected,
                    selected_set: &mut self.node_graph_state.selected_set,
                    renaming_node: self.scene_tree.renaming_node,
                    rename_buf: self.scene_tree.rename_buf,
                    scene_tree_drag: self.scene_tree.drag_state,
                    scene_tree_search: self.scene_tree.search_filter,
                    bake_progress: self.bake_progress,
                    actions: self.actions,
                    history: self.history,
                    active_light_ids: self.active_light_ids,
                    max_sculpt_resolution: self.settings.max_sculpt_resolution,
                    soloed_light: self.viewport.soloed_light,
                    material_library: self.material_library,
                    multi_transform_edit: self.multi_transform_edit,
                    gizmo_mode: self.viewport.gizmo_mode,
                    gizmo_space: self.viewport.gizmo_space,
                    selection_behavior: &self.viewport.selection_behavior,
                    reference_images: self.reference_images,
                    measurement_points: self.viewport.measurement_points,
                    show_distance_readout: self.viewport.show_distance_readout,
                    settings: self.settings,
                };
                crate::ui::primary_shell::draw_inspector_panel_tab(ui, &mut shell_context);
            }
            Tab::DrawerPanel => {
                let mut shell_context = crate::ui::primary_shell::PrimaryShellContext {
                    shell: self.primary_shell,
                    dock_state: None,
                    camera: self.camera,
                    scene: self.scene,
                    sculpt_state: self.sculpt_state,
                    selected: &mut self.node_graph_state.selected,
                    selected_set: &mut self.node_graph_state.selected_set,
                    renaming_node: self.scene_tree.renaming_node,
                    rename_buf: self.scene_tree.rename_buf,
                    scene_tree_drag: self.scene_tree.drag_state,
                    scene_tree_search: self.scene_tree.search_filter,
                    bake_progress: self.bake_progress,
                    actions: self.actions,
                    history: self.history,
                    active_light_ids: self.active_light_ids,
                    max_sculpt_resolution: self.settings.max_sculpt_resolution,
                    soloed_light: self.viewport.soloed_light,
                    material_library: self.material_library,
                    multi_transform_edit: self.multi_transform_edit,
                    gizmo_mode: self.viewport.gizmo_mode,
                    gizmo_space: self.viewport.gizmo_space,
                    selection_behavior: &self.viewport.selection_behavior,
                    reference_images: self.reference_images,
                    measurement_points: self.viewport.measurement_points,
                    show_distance_readout: self.viewport.show_distance_readout,
                    settings: self.settings,
                };
                crate::ui::primary_shell::draw_drawer_panel_tab(ui, &mut shell_context);
            }
            Tab::NodeGraph => {
                node_graph::draw(ui, self.scene, self.node_graph_state, self.actions);
            }
            Tab::Properties => {
                properties::draw(
                    ui,
                    self.scene,
                    self.node_graph_state.selected,
                    &self.node_graph_state.selected_set,
                    self.sculpt_state,
                    self.bake_progress,
                    self.actions,
                    self.active_light_ids,
                    self.settings.max_sculpt_resolution,
                    self.viewport.soloed_light,
                    self.material_library,
                    self.multi_transform_edit,
                    self.viewport.gizmo_space,
                    &self.viewport.selection_behavior,
                );
                // Defensive: clear selection if the node was deleted by properties panel
                if let Some(sel) = self.node_graph_state.selected {
                    if !self.scene.nodes.contains_key(&sel) {
                        self.node_graph_state.clear_selection();
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
                    &mut self.node_graph_state.selected_set,
                    self.scene_tree.renaming_node,
                    self.scene_tree.rename_buf,
                    self.scene_tree.drag_state,
                    self.actions,
                    self.scene_tree.search_filter,
                    self.active_light_ids,
                    self.viewport.soloed_light,
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
                        self.node_graph_state.clear_selection();
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
            Tab::Lights => {
                crate::ui::lights_panel::draw(
                    ui,
                    self.scene,
                    &mut self.node_graph_state.selected,
                    self.actions,
                    self.active_light_ids,
                );
            }
            Tab::LightGraph => {
                light_graph::draw(
                    ui,
                    self.scene,
                    self.light_graph_state,
                    &mut self.node_graph_state.selected,
                    &mut self.node_graph_state.selected_set,
                    self.actions,
                );
            }
            Tab::LightLinking => {
                crate::ui::light_linking::draw(ui, self.scene, self.actions, self.active_light_ids);
            }
            Tab::ReferenceImages => {
                crate::ui::reference_image::draw_controls(ui, self.reference_images, self.actions);
            }
            Tab::SceneStats => {
                crate::ui::scene_stats::draw(ui, self.scene, self.timings);
            }
        }
    }
}
