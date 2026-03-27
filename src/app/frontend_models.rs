use std::path::Path;

use crate::app::reference_images::ReferenceImageStore;
use crate::app::state::{
    ExpertPanelKind, ExpertPanelRegistry, InteractionMode, MenuDropdownKind, MenuUiState,
    PanelBarEdge, PanelBarOrientation, PanelFrameworkState, PanelKind, PanelPointerInteractionKind,
    PanelResizeHandle, PanelSheetAnchor, PrimaryShellState, ScenePanelUiState, SceneSelectionState,
    WorkspaceRoute, WorkspaceUiState,
};
use crate::gizmo::{GizmoMode, GizmoSpace};
use crate::graph::history::History;
use crate::graph::presented_object::{
    collect_presented_base_wrapper_chain, collect_presented_selection, current_transform_owner,
    presented_children, presented_top_level_objects, resolve_presented_object, PresentedObjectKind,
    PresentedObjectRef,
};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::keymap::ActionBinding;
use crate::sculpt::BrushMode;
use crate::sculpt::SculptState;
use crate::settings::{
    EnvironmentBackgroundMode, EnvironmentSource, GroupRotateDirection, MultiAxisOrientation,
    MultiPivotMode, Settings,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelRow {
    pub host_id: NodeId,
    pub object_root_id: NodeId,
    pub label: String,
    pub kind_label: String,
    pub depth: usize,
    pub has_children: bool,
    pub expanded: bool,
    pub selected: bool,
    pub hidden: bool,
    pub locked: bool,
    pub renaming: bool,
    pub rename_value: String,
    pub dragging: bool,
    pub drop_allowed: bool,
    pub drop_target: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenePanelModel {
    pub rows: Vec<ScenePanelRow>,
    pub selection_count: usize,
    pub selected_host: Option<NodeId>,
    pub filter_query: String,
    pub drag_active: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorScalarFieldModel {
    pub value: f32,
    pub display_text: String,
    pub enabled: bool,
    pub mixed: bool,
    pub minimum: f32,
    pub maximum: f32,
    pub step: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorBoolFieldModel {
    pub value: bool,
    pub display_text: String,
    pub enabled: bool,
    pub mixed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorTransformModel {
    pub position: [InspectorScalarFieldModel; 3],
    pub rotation_deg: [InspectorScalarFieldModel; 3],
    pub scale: [InspectorScalarFieldModel; 3],
    pub can_scale: bool,
    pub multi_editing: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorMaterialModel {
    pub base_color: [InspectorScalarFieldModel; 3],
    pub roughness: InspectorScalarFieldModel,
    pub metallic: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorOperationModel {
    pub op_label: String,
    pub smooth_k: InspectorScalarFieldModel,
    pub steps: InspectorScalarFieldModel,
    pub color_blend: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorSculptModel {
    pub desired_resolution: InspectorScalarFieldModel,
    pub layer_intensity: InspectorScalarFieldModel,
    pub brush_radius: InspectorScalarFieldModel,
    pub brush_strength: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorLightModel {
    pub light_type_label: String,
    pub color: [InspectorScalarFieldModel; 3],
    pub intensity: InspectorScalarFieldModel,
    pub range: InspectorScalarFieldModel,
    pub cast_shadows: InspectorBoolFieldModel,
    pub volumetric: InspectorBoolFieldModel,
    pub volumetric_density: InspectorScalarFieldModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectorModel {
    pub title: String,
    pub name: String,
    pub kind_label: String,
    pub chips: Vec<String>,
    pub property_lines: Vec<String>,
    pub display_lines: Vec<String>,
    pub multi_selection_summary: Option<String>,
    pub transform: Option<InspectorTransformModel>,
    pub material: Option<InspectorMaterialModel>,
    pub operation: Option<InspectorOperationModel>,
    pub sculpt: Option<InspectorSculptModel>,
    pub light: Option<InspectorLightModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub label: String,
    pub direction_label: String,
    pub is_current: bool,
    pub jump_steps: usize,
    pub jump_enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceImageRow {
    pub label: String,
    pub plane_label: String,
    pub status_label: String,
    pub visible: bool,
    pub locked: bool,
    pub opacity: f32,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPanelEntry {
    pub kind: ExpertPanelKind,
    pub label: String,
    pub open: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettingsModel {
    pub show_grid: bool,
    pub show_node_labels: bool,
    pub show_bounding_box: bool,
    pub show_light_gizmos: bool,
    pub shadows_enabled: bool,
    pub ao_enabled: bool,
    pub export_resolution: u32,
    pub adaptive_export: bool,
    pub environment_is_hdri: bool,
    pub hdri_path_display: String,
    pub environment_rotation_degrees: f32,
    pub environment_exposure: f32,
    pub environment_bake_resolution: u32,
    pub background_is_procedural: bool,
    pub environment_background_blur: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UtilityModel {
    pub history_summary: String,
    pub reference_summary: String,
    pub history_rows: Vec<HistoryEntry>,
    pub reference_rows: Vec<ReferenceImageRow>,
    pub expert_panels: Vec<ExpertPanelEntry>,
    pub render_settings: RenderSettingsModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPaletteEntry {
    pub kind: ToolPaletteKind,
    pub label: String,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolPaletteKind {
    Select,
    Brush(BrushMode),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPaletteModel {
    pub visible: bool,
    pub select_tool: ToolPaletteEntry,
    pub brush_tools: Vec<ToolPaletteEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolPanelMode {
    Select,
    Sculpt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolContextModel {
    pub visible: bool,
    pub mode: ToolPanelMode,
    pub active_brush: Option<BrushMode>,
    pub select_tool: ToolPaletteEntry,
    pub brush_tools: Vec<ToolPaletteEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolPanelModel {
    pub title: String,
    pub mode: ToolPanelMode,
    pub summary: String,
    pub empty_state: String,
    pub show_sculpt_target_fields: bool,
    pub transform: Option<InspectorTransformModel>,
    pub material: Option<InspectorMaterialModel>,
    pub operation: Option<InspectorOperationModel>,
    pub sculpt: Option<InspectorSculptModel>,
    pub light: Option<InspectorLightModel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MenuStripKind {
    File,
    Edit,
    View,
    Settings,
    Help,
}

impl MenuStripKind {
    pub const ALL: [Self; 5] = [
        Self::File,
        Self::Edit,
        Self::View,
        Self::Settings,
        Self::Help,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Edit => "Edit",
            Self::View => "View",
            Self::Settings => "Settings",
            Self::Help => "Help",
        }
    }

    pub fn dropdown_kind(self) -> Option<MenuDropdownKind> {
        match self {
            Self::File => Some(MenuDropdownKind::File),
            Self::Edit => Some(MenuDropdownKind::Edit),
            Self::View => Some(MenuDropdownKind::View),
            Self::Settings => None,
            Self::Help => Some(MenuDropdownKind::Help),
        }
    }

    pub fn anchor_index(self) -> i32 {
        match self {
            Self::File => 0,
            Self::Edit => 1,
            Self::View => 2,
            Self::Settings => 3,
            Self::Help => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MenuCommandKind {
    NewScene,
    OpenProject,
    SaveProject,
    ImportMesh,
    ExportMesh,
    TakeScreenshot,
    AddReferenceImage,
    Undo,
    Redo,
    Copy,
    Paste,
    Duplicate,
    DeleteSelected,
    FrameAll,
    FocusSelected,
    CameraFront,
    CameraTop,
    CameraRight,
    ToggleOrtho,
    ToggleMeasure,
    ToggleTurntable,
    ToggleHelp,
    ToggleCommandPalette,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuStripItemModel {
    pub kind: MenuStripKind,
    pub label: String,
    pub active: bool,
    pub focused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuStripModel {
    pub visible: bool,
    pub items: Vec<MenuStripItemModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuCommandModel {
    pub command: MenuCommandKind,
    pub label: String,
    pub shortcut_label: String,
    pub enabled: bool,
    pub checked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuDropdownModel {
    pub visible: bool,
    pub kind: Option<MenuDropdownKind>,
    pub title: String,
    pub anchor_index: i32,
    pub highlighted_index: i32,
    pub items: Vec<MenuCommandModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsCardModel {
    pub visible: bool,
    pub multi_axis_orientation: MultiAxisOrientation,
    pub group_rotate_direction: GroupRotateDirection,
    pub multi_pivot_mode: MultiPivotMode,
    pub auto_save_enabled: bool,
    pub show_fps_overlay: bool,
    pub continuous_repaint: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelLauncherItemModel {
    pub kind: PanelKind,
    pub label: String,
    pub short_label: String,
    pub icon_key: Option<String>,
    pub active: bool,
    pub pinned: bool,
    pub show_drag_indicator: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelBarModel {
    pub visible: bool,
    pub edge: PanelBarEdge,
    pub orientation: PanelBarOrientation,
    pub items: Vec<PanelLauncherItemModel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivePanelContentModel {
    pub kind: PanelKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelFrameModel {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelSheetModel {
    pub kind: PanelKind,
    pub title: String,
    pub collapsed_title: String,
    pub collapsed_width: f32,
    pub collapsed_height: f32,
    pub pinned: bool,
    pub collapsed: bool,
    pub anchor: PanelSheetAnchor,
    pub frame: PanelFrameModel,
    pub movable: bool,
    pub resizable: bool,
    pub content: ActivePanelContentModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelFrameworkModel {
    pub bar: PanelBarModel,
    pub transient_panel: Option<PanelSheetModel>,
    pub pinned_panels: Vec<PanelSheetModel>,
    pub panel_interaction_active: bool,
    pub interaction_panel_kind: PanelKind,
    pub active_interaction_kind: PanelPointerInteractionKind,
    pub active_resize_handle: Option<PanelResizeHandle>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayLayoutModel {
    pub safe_area_left: f32,
    pub safe_area_top: f32,
    pub safe_area_right: f32,
    pub safe_area_bottom: f32,
    pub virtual_keyboard_x: f32,
    pub virtual_keyboard_y: f32,
    pub virtual_keyboard_width: f32,
    pub virtual_keyboard_height: f32,
    pub usable_x: f32,
    pub usable_y: f32,
    pub usable_width: f32,
    pub usable_height: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportStatusModel {
    pub interaction_label: String,
    pub transform_label: String,
    pub space_label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportFrameImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSummaryEntry {
    pub label: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspacePanelModel {
    pub visible: bool,
    pub route: WorkspaceRoute,
    pub route_label: String,
    pub selection_summary: String,
    pub detail_text: String,
    pub context_rows: Vec<WorkspaceSummaryEntry>,
    pub input_rows: Vec<WorkspaceSummaryEntry>,
    pub output_rows: Vec<WorkspaceSummaryEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShellSnapshot {
    pub overlay_layout: OverlayLayoutModel,
    pub menu_strip: MenuStripModel,
    pub menu_dropdown: MenuDropdownModel,
    pub settings_card: SettingsCardModel,
    pub tool_context: ToolContextModel,
    pub tool_palette: ToolPaletteModel,
    pub panel_framework: PanelFrameworkModel,
    pub scene_panel: ScenePanelModel,
    pub inspector: InspectorModel,
    pub tool_panel: ToolPanelModel,
    pub utility: UtilityModel,
    pub viewport_status: ViewportStatusModel,
    pub workspace: WorkspacePanelModel,
}

pub struct ShellSnapshotInputs<'a> {
    pub scene: &'a Scene,
    pub selection: &'a SceneSelectionState,
    pub scene_panel_ui: &'a ScenePanelUiState,
    pub primary_shell: &'a PrimaryShellState,
    pub menu_ui: &'a MenuUiState,
    pub panel_framework: &'a PanelFrameworkState,
    pub viewport_size_logical: [f32; 2],
    pub safe_area_insets_logical: [f32; 4],
    pub virtual_keyboard_position_logical: [f32; 2],
    pub virtual_keyboard_size_logical: [f32; 2],
    pub workspace: &'a WorkspaceUiState,
    pub history: &'a History,
    pub reference_images: &'a ReferenceImageStore,
    pub expert_panels: &'a ExpertPanelRegistry,
    pub file_actions_enabled: bool,
    pub settings: &'a Settings,
    pub sculpt_state: &'a SculptState,
    pub interaction_mode: InteractionMode,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    pub camera_is_ortho: bool,
    pub measurement_mode_active: bool,
    pub turntable_active: bool,
    pub help_visible: bool,
    pub command_palette_visible: bool,
    pub can_undo: bool,
    pub can_redo: bool,
    pub has_selection: bool,
    pub has_clipboard_node: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MenuCommandCheckState {
    pub ortho_enabled: bool,
    pub measurement_enabled: bool,
    pub turntable_enabled: bool,
    pub help_visible: bool,
    pub command_palette_visible: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MenuCommandAvailability {
    pub undo_enabled: bool,
    pub redo_enabled: bool,
    pub copy_enabled: bool,
    pub paste_enabled: bool,
    pub duplicate_enabled: bool,
    pub delete_enabled: bool,
    pub focus_selected_enabled: bool,
}

pub fn build_shell_snapshot(inputs: ShellSnapshotInputs<'_>) -> ShellSnapshot {
    let overlay_layout = build_overlay_layout_model(
        inputs.viewport_size_logical,
        inputs.safe_area_insets_logical,
        inputs.virtual_keyboard_position_logical,
        inputs.virtual_keyboard_size_logical,
    );
    let tool_context = build_tool_context_model(inputs.primary_shell);
    let inspector = build_inspector_model(
        inputs.scene,
        inputs.selection,
        inputs.settings,
        inputs.sculpt_state,
        inputs.interaction_mode,
        &inputs.gizmo_mode,
        &inputs.gizmo_space,
    );
    let tool_panel = build_tool_panel_model(
        inputs.scene,
        inputs.selection,
        inputs.sculpt_state,
        inputs.interaction_mode,
        &inspector,
        &tool_context,
    );
    let menu_strip = build_menu_strip_model(inputs.menu_ui);
    let menu_dropdown = build_menu_dropdown_model(
        inputs.menu_ui,
        inputs.file_actions_enabled,
        inputs.settings,
        MenuCommandCheckState {
            ortho_enabled: inputs.camera_is_ortho,
            measurement_enabled: inputs.measurement_mode_active,
            turntable_enabled: inputs.turntable_active,
            help_visible: inputs.help_visible,
            command_palette_visible: inputs.command_palette_visible,
        },
        MenuCommandAvailability {
            undo_enabled: inputs.can_undo,
            redo_enabled: inputs.can_redo,
            copy_enabled: inputs.has_selection,
            paste_enabled: inputs.has_clipboard_node,
            duplicate_enabled: inputs.has_selection,
            delete_enabled: inputs.has_selection,
            focus_selected_enabled: inputs.has_selection,
        },
    );
    let settings_card = build_settings_card_model(inputs.menu_ui, inputs.settings);

    ShellSnapshot {
        overlay_layout: overlay_layout.clone(),
        menu_strip,
        menu_dropdown,
        settings_card,
        tool_context: tool_context.clone(),
        tool_palette: build_tool_palette_model(&tool_context),
        panel_framework: build_panel_framework_model(inputs.panel_framework, &overlay_layout),
        scene_panel: build_scene_panel_model(inputs.scene, inputs.selection, inputs.scene_panel_ui),
        inspector,
        tool_panel,
        utility: build_utility_model(
            inputs.history,
            inputs.reference_images,
            inputs.expert_panels,
            inputs.settings,
        ),
        viewport_status: ViewportStatusModel {
            interaction_label: interaction_mode_label(inputs.interaction_mode).to_string(),
            transform_label: inputs.gizmo_mode.label().to_string(),
            space_label: inputs.gizmo_space.label().to_string(),
        },
        workspace: build_workspace_panel_model(
            inputs.scene,
            inputs.selection,
            inputs.workspace,
            inputs.primary_shell,
        ),
    }
}

pub fn build_tool_context_model(primary_shell: &PrimaryShellState) -> ToolContextModel {
    let (mode, active_brush) = match primary_shell.interaction_mode {
        InteractionMode::Sculpt(brush) => (ToolPanelMode::Sculpt, Some(brush)),
        InteractionMode::Select | InteractionMode::Measure => (ToolPanelMode::Select, None),
    };

    ToolContextModel {
        visible: primary_shell.tool_rail_visible,
        mode,
        active_brush,
        select_tool: ToolPaletteEntry {
            kind: ToolPaletteKind::Select,
            label: "Select".to_string(),
            active: matches!(
                primary_shell.interaction_mode,
                InteractionMode::Select | InteractionMode::Measure
            ),
        },
        brush_tools: BrushMode::ALL
            .into_iter()
            .map(|brush| ToolPaletteEntry {
                kind: ToolPaletteKind::Brush(brush),
                label: brush.label().to_string(),
                active: active_brush == Some(brush),
            })
            .collect(),
    }
}

pub fn build_tool_palette_model(tool_context: &ToolContextModel) -> ToolPaletteModel {
    ToolPaletteModel {
        visible: tool_context.visible,
        select_tool: tool_context.select_tool.clone(),
        brush_tools: tool_context.brush_tools.clone(),
    }
}

pub fn build_menu_strip_model(menu_ui: &MenuUiState) -> MenuStripModel {
    MenuStripModel {
        visible: menu_ui.strip_visible,
        items: MenuStripKind::ALL
            .into_iter()
            .map(|kind| MenuStripItemModel {
                kind,
                label: kind.label().to_string(),
                active: match kind.dropdown_kind() {
                    Some(dropdown_kind) => menu_ui.active_dropdown == Some(dropdown_kind),
                    None => menu_ui.settings_card_open,
                },
                focused: menu_ui.focused_launcher == Some(menu_launcher_kind(kind)),
            })
            .collect(),
    }
}

pub fn build_menu_dropdown_model(
    menu_ui: &MenuUiState,
    file_actions_enabled: bool,
    settings: &Settings,
    checks: MenuCommandCheckState,
    availability: MenuCommandAvailability,
) -> MenuDropdownModel {
    let Some(kind) = menu_ui.active_dropdown else {
        return MenuDropdownModel {
            visible: false,
            kind: None,
            title: String::new(),
            anchor_index: -1,
            highlighted_index: -1,
            items: Vec::new(),
        };
    };

    let strip_kind = match kind {
        MenuDropdownKind::File => MenuStripKind::File,
        MenuDropdownKind::Edit => MenuStripKind::Edit,
        MenuDropdownKind::View => MenuStripKind::View,
        MenuDropdownKind::Help => MenuStripKind::Help,
    };

    let items = menu_commands_for_kind(kind, file_actions_enabled, settings, checks, availability);
    let highlighted_index =
        resolve_menu_highlighted_index(menu_ui.highlighted_command_index, &items);

    MenuDropdownModel {
        visible: true,
        kind: Some(kind),
        title: kind.label().to_string(),
        anchor_index: strip_kind.anchor_index(),
        highlighted_index: highlighted_index.map(|index| index as i32).unwrap_or(-1),
        items,
    }
}

pub fn build_settings_card_model(menu_ui: &MenuUiState, settings: &Settings) -> SettingsCardModel {
    SettingsCardModel {
        visible: menu_ui.settings_card_open,
        multi_axis_orientation: settings.selection_behavior.multi_axis_orientation,
        group_rotate_direction: settings.selection_behavior.group_rotate_direction,
        multi_pivot_mode: settings.selection_behavior.multi_pivot_mode,
        auto_save_enabled: settings.auto_save_enabled,
        show_fps_overlay: settings.show_fps_overlay,
        continuous_repaint: settings.continuous_repaint,
    }
}

fn menu_launcher_kind(kind: MenuStripKind) -> crate::app::state::MenuLauncherKind {
    match kind {
        MenuStripKind::File => crate::app::state::MenuLauncherKind::File,
        MenuStripKind::Edit => crate::app::state::MenuLauncherKind::Edit,
        MenuStripKind::View => crate::app::state::MenuLauncherKind::View,
        MenuStripKind::Settings => crate::app::state::MenuLauncherKind::Settings,
        MenuStripKind::Help => crate::app::state::MenuLauncherKind::Help,
    }
}

fn resolve_menu_highlighted_index(
    preferred: Option<usize>,
    items: &[MenuCommandModel],
) -> Option<usize> {
    if let Some(index) = preferred {
        if items.get(index).is_some_and(|item| item.enabled) {
            return Some(index);
        }
    }
    items.iter().position(|item| item.enabled)
}

pub(crate) fn menu_commands_for_kind(
    kind: MenuDropdownKind,
    file_actions_enabled: bool,
    settings: &Settings,
    checks: MenuCommandCheckState,
    availability: MenuCommandAvailability,
) -> Vec<MenuCommandModel> {
    match kind {
        MenuDropdownKind::File => vec![
            menu_command(
                MenuCommandKind::NewScene,
                "New Scene",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::OpenProject,
                "Open Project",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::SaveProject,
                "Save Project",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ImportMesh,
                "Import Mesh",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ExportMesh,
                "Export Mesh",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::TakeScreenshot,
                "Screenshot",
                file_actions_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::AddReferenceImage,
                "Add Reference",
                file_actions_enabled,
                settings,
                checks,
            ),
        ],
        MenuDropdownKind::Edit => vec![
            menu_command(
                MenuCommandKind::Undo,
                "Undo",
                availability.undo_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::Redo,
                "Redo",
                availability.redo_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::Copy,
                "Copy",
                availability.copy_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::Paste,
                "Paste",
                availability.paste_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::Duplicate,
                "Duplicate",
                availability.duplicate_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::DeleteSelected,
                "Delete Selected",
                availability.delete_enabled,
                settings,
                checks,
            ),
        ],
        MenuDropdownKind::View => vec![
            menu_command(
                MenuCommandKind::FrameAll,
                "Frame All",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::FocusSelected,
                "Focus Selected",
                availability.focus_selected_enabled,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::CameraFront,
                "Front View",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::CameraTop,
                "Top View",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::CameraRight,
                "Right View",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ToggleOrtho,
                "Toggle Ortho",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ToggleMeasure,
                "Toggle Measure",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ToggleTurntable,
                "Toggle Turntable",
                true,
                settings,
                checks,
            ),
        ],
        MenuDropdownKind::Help => vec![
            menu_command(
                MenuCommandKind::ToggleHelp,
                "Toggle Help",
                true,
                settings,
                checks,
            ),
            menu_command(
                MenuCommandKind::ToggleCommandPalette,
                "Command Palette",
                true,
                settings,
                checks,
            ),
        ],
    }
}

fn menu_command(
    command: MenuCommandKind,
    label: &str,
    enabled: bool,
    settings: &Settings,
    checks: MenuCommandCheckState,
) -> MenuCommandModel {
    MenuCommandModel {
        command,
        label: label.to_string(),
        shortcut_label: menu_shortcut_label(command, settings),
        enabled,
        checked: menu_command_checked(command, checks),
    }
}

fn menu_command_checked(command: MenuCommandKind, checks: MenuCommandCheckState) -> bool {
    match command {
        MenuCommandKind::ToggleOrtho => checks.ortho_enabled,
        MenuCommandKind::ToggleMeasure => checks.measurement_enabled,
        MenuCommandKind::ToggleTurntable => checks.turntable_enabled,
        MenuCommandKind::ToggleHelp => checks.help_visible,
        MenuCommandKind::ToggleCommandPalette => checks.command_palette_visible,
        _ => false,
    }
}

fn menu_shortcut_label(command: MenuCommandKind, settings: &Settings) -> String {
    menu_action_binding(command)
        .and_then(|binding| settings.keymap.format_shortcut(binding))
        .unwrap_or_default()
}

fn menu_action_binding(command: MenuCommandKind) -> Option<ActionBinding> {
    match command {
        MenuCommandKind::NewScene => Some(ActionBinding::NewScene),
        MenuCommandKind::OpenProject => Some(ActionBinding::OpenProject),
        MenuCommandKind::SaveProject => Some(ActionBinding::SaveProject),
        MenuCommandKind::ImportMesh => None,
        MenuCommandKind::ExportMesh => Some(ActionBinding::ShowExportDialog),
        MenuCommandKind::TakeScreenshot => Some(ActionBinding::TakeScreenshot),
        MenuCommandKind::AddReferenceImage => None,
        MenuCommandKind::Undo => Some(ActionBinding::Undo),
        MenuCommandKind::Redo => Some(ActionBinding::Redo),
        MenuCommandKind::Copy => Some(ActionBinding::Copy),
        MenuCommandKind::Paste => Some(ActionBinding::Paste),
        MenuCommandKind::Duplicate => Some(ActionBinding::Duplicate),
        MenuCommandKind::DeleteSelected => Some(ActionBinding::DeleteSelected),
        MenuCommandKind::FrameAll => Some(ActionBinding::FrameAll),
        MenuCommandKind::FocusSelected => Some(ActionBinding::FocusSelected),
        MenuCommandKind::CameraFront => Some(ActionBinding::CameraFront),
        MenuCommandKind::CameraTop => Some(ActionBinding::CameraTop),
        MenuCommandKind::CameraRight => Some(ActionBinding::CameraRight),
        MenuCommandKind::ToggleOrtho => Some(ActionBinding::ToggleOrtho),
        MenuCommandKind::ToggleMeasure => Some(ActionBinding::ToggleMeasurementTool),
        MenuCommandKind::ToggleTurntable => Some(ActionBinding::ToggleTurntable),
        MenuCommandKind::ToggleHelp => Some(ActionBinding::ToggleHelp),
        MenuCommandKind::ToggleCommandPalette => Some(ActionBinding::ToggleCommandPalette),
    }
}

pub fn build_panel_framework_model(
    panel_framework: &PanelFrameworkState,
    overlay_layout: &OverlayLayoutModel,
) -> PanelFrameworkModel {
    let usable_rect = overlay_usable_rect(overlay_layout);
    let primary_bar = panel_framework
        .bar(crate::app::state::PanelBarId::PrimaryRight)
        .or_else(|| panel_framework.bars.first());
    let default_bar = crate::app::state::PanelBarState {
        id: crate::app::state::PanelBarId::PrimaryRight,
        edge: PanelBarEdge::Right,
        orientation: PanelBarOrientation::Vertical,
        items: PanelKind::ALL.to_vec(),
        active_transient: None,
        transient_rect: None,
    };
    let bar = primary_bar.unwrap_or(&default_bar);
    let transient_kind = bar.active_transient;
    let transient_frame = transient_kind
        .map(|_| panel_frame_model(panel_framework.resolved_transient_rect(bar.id, usable_rect)));

    let items = bar
        .items
        .iter()
        .copied()
        .map(|kind| PanelLauncherItemModel {
            kind,
            label: kind.label().to_string(),
            short_label: kind.short_label().to_string(),
            icon_key: Some(kind.icon_key().to_string()),
            active: transient_kind == Some(kind) || panel_framework.pinned_instance(kind).is_some(),
            pinned: panel_framework.pinned_instance(kind).is_some(),
            show_drag_indicator: transient_kind == Some(kind),
        })
        .collect::<Vec<_>>();

    let transient_panel =
        transient_kind
            .zip(transient_frame)
            .map(|(kind, frame)| PanelSheetModel {
                kind,
                title: kind.label().to_string(),
                collapsed_title: kind.short_label().to_string(),
                collapsed_width: PanelFrameworkState::COLLAPSED_PANEL_WIDTH,
                collapsed_height: PanelFrameworkState::COLLAPSED_PANEL_HEIGHT,
                pinned: false,
                collapsed: false,
                anchor: panel_framework.sheet_anchor_for_bar(bar.id),
                frame,
                movable: true,
                resizable: true,
                content: ActivePanelContentModel { kind },
            });

    let mut pinned_instances = panel_framework
        .pinned_instances
        .iter()
        .filter(|instance| instance.visible)
        .collect::<Vec<_>>();
    pinned_instances.sort_by_key(|instance| {
        panel_framework
            .focus_order
            .iter()
            .position(|focused_id| *focused_id == instance.id)
            .unwrap_or(usize::MAX)
    });

    let pinned_panels = pinned_instances
        .into_iter()
        .map(|instance| PanelSheetModel {
            kind: instance.kind,
            title: instance.kind.label().to_string(),
            collapsed_title: instance.kind.short_label().to_string(),
            collapsed_width: PanelFrameworkState::COLLAPSED_PANEL_WIDTH,
            collapsed_height: PanelFrameworkState::COLLAPSED_PANEL_HEIGHT,
            pinned: true,
            collapsed: instance.collapsed,
            anchor: panel_framework.sheet_anchor_for_bar(instance.anchor_bar),
            frame: panel_frame_model(
                panel_framework
                    .resolved_display_panel_rect(instance.kind, instance.anchor_bar, usable_rect)
                    .unwrap_or_else(|| {
                        panel_framework.resolved_transient_rect(instance.anchor_bar, usable_rect)
                    }),
            ),
            movable: true,
            resizable: !instance.collapsed,
            content: ActivePanelContentModel {
                kind: instance.kind,
            },
        })
        .collect();

    PanelFrameworkModel {
        bar: PanelBarModel {
            visible: true,
            edge: bar.edge,
            orientation: bar.orientation,
            items,
        },
        transient_panel,
        pinned_panels,
        panel_interaction_active: panel_framework.panel_interaction.is_some(),
        interaction_panel_kind: panel_framework
            .panel_interaction
            .map(|interaction| interaction.kind)
            .unwrap_or(PanelKind::Tool),
        active_interaction_kind: panel_framework
            .panel_interaction
            .map(|interaction| interaction.interaction)
            .unwrap_or(PanelPointerInteractionKind::Move),
        active_resize_handle: panel_framework.panel_interaction.and_then(|interaction| {
            match interaction.interaction {
                PanelPointerInteractionKind::Move => None,
                PanelPointerInteractionKind::Resize(handle) => Some(handle),
            }
        }),
    }
}

fn panel_frame_model(bounds: crate::app::ui_geometry::FloatingPanelBounds) -> PanelFrameModel {
    PanelFrameModel {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
    }
}

pub fn build_overlay_layout_model(
    viewport_size_logical: [f32; 2],
    safe_area_insets_logical: [f32; 4],
    virtual_keyboard_position_logical: [f32; 2],
    virtual_keyboard_size_logical: [f32; 2],
) -> OverlayLayoutModel {
    let viewport_width = viewport_size_logical[0].max(1.0);
    let viewport_height = viewport_size_logical[1].max(1.0);
    let safe_left = safe_area_insets_logical[0].max(0.0);
    let safe_top = safe_area_insets_logical[1].max(0.0);
    let safe_right = safe_area_insets_logical[2].max(0.0);
    let safe_bottom = safe_area_insets_logical[3].max(0.0);

    let usable_x = safe_left;
    let usable_y = safe_top;
    let usable_width = (viewport_width - safe_left - safe_right).max(1.0);
    let mut usable_height = (viewport_height - safe_top - safe_bottom).max(1.0);

    let keyboard_x = virtual_keyboard_position_logical[0].max(0.0);
    let keyboard_y = virtual_keyboard_position_logical[1].max(0.0);
    let keyboard_width = virtual_keyboard_size_logical[0].max(0.0);
    let keyboard_height = virtual_keyboard_size_logical[1].max(0.0);

    if keyboard_width > 0.0 && keyboard_height > 0.0 {
        let usable_bottom = usable_y + usable_height;
        let keyboard_top = keyboard_y;
        if keyboard_top.is_finite() && keyboard_top > usable_y && keyboard_top < usable_bottom {
            usable_height = (keyboard_top - usable_y).max(1.0);
        }
    }

    OverlayLayoutModel {
        safe_area_left: safe_left,
        safe_area_top: safe_top,
        safe_area_right: safe_right,
        safe_area_bottom: safe_bottom,
        virtual_keyboard_x: keyboard_x,
        virtual_keyboard_y: keyboard_y,
        virtual_keyboard_width: keyboard_width,
        virtual_keyboard_height: keyboard_height,
        usable_x,
        usable_y,
        usable_width,
        usable_height,
    }
}

fn overlay_usable_rect(
    overlay_layout: &OverlayLayoutModel,
) -> crate::app::ui_geometry::FloatingPanelBounds {
    crate::app::ui_geometry::FloatingPanelBounds::from_min_size(
        overlay_layout.usable_x,
        overlay_layout.usable_y,
        overlay_layout.usable_width.max(1.0),
        overlay_layout.usable_height.max(1.0),
    )
}

pub fn build_scene_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    scene_panel_ui: &ScenePanelUiState,
) -> ScenePanelModel {
    let mut rows = Vec::new();
    let selected_host = selection
        .selected
        .and_then(|selected_id| resolve_presented_object(scene, selected_id))
        .map(|object| object.host_id);
    let normalized_filter = scene_panel_ui.filter_query.trim().to_ascii_lowercase();
    for root in presented_top_level_objects(scene) {
        collect_scene_rows(
            scene,
            root,
            selection,
            scene_panel_ui,
            0,
            &normalized_filter,
            &mut rows,
        );
    }
    ScenePanelModel {
        rows,
        selection_count: selection.selected_set.len(),
        selected_host,
        filter_query: scene_panel_ui.filter_query.clone(),
        drag_active: scene_panel_ui.drag_source.is_some(),
    }
}

pub fn build_inspector_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    let presented_selection =
        collect_presented_selection(scene, selection.selected, &selection.selected_set);
    match (
        presented_selection.primary,
        presented_selection.ordered.len(),
    ) {
        (None, _) => InspectorModel {
            title: "No selection".to_string(),
            name: String::new(),
            kind_label: "NONE".to_string(),
            chips: Vec::new(),
            property_lines: vec!["Click an object in the scene list or viewport.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
            multi_selection_summary: None,
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        },
        (_, count) if count > 1 => build_multi_selection_inspector(
            scene,
            &presented_selection.ordered,
            settings,
            sculpt_state,
            interaction_mode,
            gizmo_mode,
            gizmo_space,
        ),
        (Some(object), _) => build_single_object_inspector(
            scene,
            object,
            settings,
            sculpt_state,
            interaction_mode,
            gizmo_mode,
            gizmo_space,
        ),
    }
}

fn build_tool_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    inspector: &InspectorModel,
    tool_context: &ToolContextModel,
) -> ToolPanelModel {
    let presented_selection =
        collect_presented_selection(scene, selection.selected, &selection.selected_set);

    match interaction_mode {
        InteractionMode::Sculpt(brush) => build_sculpt_tool_panel_model(
            scene,
            &presented_selection.ordered,
            sculpt_state,
            brush,
            tool_context,
        ),
        InteractionMode::Select | InteractionMode::Measure => build_select_tool_panel_model(
            scene,
            &presented_selection.ordered,
            inspector,
            tool_context,
        ),
    }
}

fn build_select_tool_panel_model(
    scene: &Scene,
    selected_objects: &[PresentedObjectRef],
    inspector: &InspectorModel,
    tool_context: &ToolContextModel,
) -> ToolPanelModel {
    let (summary, empty_state) = match selected_objects {
        [] => (String::new(), "No selection".to_string()),
        [object] => (
            scene
                .nodes
                .get(&object.host_id)
                .map(|node| node.name.clone())
                .unwrap_or_else(|| inspector.title.clone()),
            String::new(),
        ),
        objects => (format!("{} selected", objects.len()), String::new()),
    };

    let primary_object = selected_objects.first().copied();
    let single_selection = selected_objects.len() == 1;
    let material = single_selection
        .then(|| inspector.material.clone())
        .flatten();
    let operation = single_selection
        .then(|| inspector.operation.clone())
        .flatten();
    let light = single_selection.then(|| inspector.light.clone()).flatten();

    let (quick_material, quick_operation, quick_light) = match primary_object {
        Some(object) if matches!(object.kind, PresentedObjectKind::Light) => (None, None, light),
        Some(_) if operation.is_some() => (None, operation, None),
        Some(_) => (material, None, None),
        None => (None, None, None),
    };

    ToolPanelModel {
        title: "Tool".to_string(),
        mode: tool_context.mode,
        summary,
        empty_state,
        show_sculpt_target_fields: false,
        transform: inspector.transform.clone(),
        material: if selected_objects.len() == 1 {
            quick_material
        } else {
            None
        },
        operation: if selected_objects.len() == 1 {
            quick_operation
        } else {
            None
        },
        sculpt: None,
        light: if selected_objects.len() == 1 {
            quick_light
        } else {
            None
        },
    }
}

fn build_sculpt_tool_panel_model(
    scene: &Scene,
    selected_objects: &[PresentedObjectRef],
    sculpt_state: &SculptState,
    brush: BrushMode,
    tool_context: &ToolContextModel,
) -> ToolPanelModel {
    let sculpt_profile = sculpt_state.selected_profile();
    let sculpt_target = if selected_objects.len() == 1 {
        selected_objects
            .first()
            .and_then(|object| resolve_sculpt_tool_target(scene, *object))
    } else {
        None
    };

    let (desired_resolution, layer_intensity, show_sculpt_target_fields) = sculpt_target
        .and_then(|target_id| scene.nodes.get(&target_id))
        .and_then(|node| match &node.data {
            NodeData::Sculpt {
                desired_resolution,
                layer_intensity,
                ..
            } => Some((
                int_field(*desired_resolution as f32, 8.0, 512.0, 8.0),
                float_field(*layer_intensity, 0.0, 4.0, 0.01, 2),
                true,
            )),
            _ => None,
        })
        .unwrap_or_else(|| {
            (
                disabled_field("No sculpt target".to_string(), 8.0, 512.0, 8.0),
                disabled_field("No sculpt target".to_string(), 0.0, 4.0, 0.01),
                false,
            )
        });

    ToolPanelModel {
        title: "Tool".to_string(),
        mode: tool_context.mode,
        summary: format!("Brush: {}", brush.label()),
        empty_state: if show_sculpt_target_fields {
            String::new()
        } else {
            "No sculpt target selected".to_string()
        },
        show_sculpt_target_fields,
        transform: None,
        material: None,
        operation: None,
        sculpt: Some(InspectorSculptModel {
            desired_resolution,
            layer_intensity,
            brush_radius: float_field(sculpt_profile.radius, 0.05, 2.0, 0.01, 2),
            brush_strength: float_field(sculpt_profile.strength, -2.0, 2.0, 0.01, 2),
        }),
        light: None,
    }
}

fn resolve_sculpt_tool_target(scene: &Scene, object: PresentedObjectRef) -> Option<NodeId> {
    if let Some(sculpt_id) = object.attached_sculpt_id {
        return Some(sculpt_id);
    }

    match scene.nodes.get(&object.host_id).map(|node| &node.data) {
        Some(NodeData::Sculpt { .. }) => Some(object.host_id),
        _ => None,
    }
}

pub fn build_utility_model(
    history: &History,
    reference_images: &ReferenceImageStore,
    expert_panels: &ExpertPanelRegistry,
    settings: &Settings,
) -> UtilityModel {
    let undo_labels = history.undo_labels();
    let redo_labels = history.redo_labels();

    let mut history_rows = Vec::new();
    history_rows.extend(
        undo_labels
            .iter()
            .rev()
            .enumerate()
            .map(|(index, label)| HistoryEntry {
                label: label.clone(),
                direction_label: "Undo".to_string(),
                is_current: false,
                jump_steps: index + 1,
                jump_enabled: true,
            }),
    );
    history_rows.push(HistoryEntry {
        label: "Current state".to_string(),
        direction_label: String::new(),
        is_current: true,
        jump_steps: 0,
        jump_enabled: false,
    });
    history_rows.extend(
        redo_labels
            .iter()
            .enumerate()
            .map(|(index, label)| HistoryEntry {
                label: label.clone(),
                direction_label: "Redo".to_string(),
                is_current: false,
                jump_steps: index + 1,
                jump_enabled: true,
            }),
    );

    let visible_count = reference_images
        .images
        .iter()
        .filter(|image| image.visible)
        .count();

    UtilityModel {
        history_summary: format!(
            "{} undo | {} redo",
            history.undo_count(),
            history.redo_count()
        ),
        reference_summary: format!(
            "{} images | {} visible",
            reference_images.images.len(),
            visible_count
        ),
        history_rows,
        reference_rows: reference_images
            .images
            .iter()
            .map(|image| ReferenceImageRow {
                label: Path::new(&image.path)
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| image.path.clone()),
                plane_label: image.plane.label().to_string(),
                status_label: format!(
                    "{} | {}",
                    if image.visible { "Visible" } else { "Hidden" },
                    if image.locked { "Locked" } else { "Unlocked" }
                ),
                visible: image.visible,
                locked: image.locked,
                opacity: image.opacity,
                scale: image.scale,
            })
            .collect(),
        expert_panels: ExpertPanelKind::ALL
            .into_iter()
            .map(|kind| ExpertPanelEntry {
                kind,
                label: kind.label().to_string(),
                open: expert_panels.is_open(kind),
            })
            .collect(),
        render_settings: RenderSettingsModel {
            show_grid: settings.render.show_grid,
            show_node_labels: settings.render.show_node_labels,
            show_bounding_box: settings.render.show_bounding_box,
            show_light_gizmos: settings.render.show_light_gizmos,
            shadows_enabled: settings.render.shadows_enabled,
            ao_enabled: settings.render.ao_enabled,
            export_resolution: settings.export_resolution,
            adaptive_export: settings.adaptive_export,
            environment_is_hdri: matches!(
                settings.render.environment_source,
                EnvironmentSource::Hdri
            ),
            hdri_path_display: settings
                .render
                .hdri_path
                .clone()
                .unwrap_or_else(|| "No HDRI selected".to_string()),
            environment_rotation_degrees: settings.render.environment_rotation_degrees,
            environment_exposure: settings.render.environment_exposure,
            environment_bake_resolution: settings.render.environment_bake_resolution,
            background_is_procedural: matches!(
                settings.render.environment_background_mode,
                EnvironmentBackgroundMode::Procedural
            ),
            environment_background_blur: settings.render.environment_background_blur,
        },
    }
}

pub fn build_workspace_panel_model(
    scene: &Scene,
    selection: &SceneSelectionState,
    workspace: &WorkspaceUiState,
    primary_shell: &PrimaryShellState,
) -> WorkspacePanelModel {
    let mut context_rows = vec![WorkspaceSummaryEntry {
        label: "Route".to_string(),
        value: workspace.route.label().to_string(),
    }];
    let mut input_rows = Vec::new();
    let mut output_rows = Vec::new();
    let detail_text = match workspace.route {
        WorkspaceRoute::NodeGraph => {
            build_node_workspace_rows(
                scene,
                selection.selected,
                &mut context_rows,
                &mut input_rows,
                &mut output_rows,
            );
            "Graph scaffold for selected object structure. Routing and context are live; node editing comes next.".to_string()
        }
        WorkspaceRoute::LightGraph => {
            build_light_workspace_rows(
                scene,
                selection.selected,
                &mut context_rows,
                &mut input_rows,
                &mut output_rows,
            );
            "Lighting workspace scaffold. Scene light context is live; graph canvas and wiring are a follow-on slice.".to_string()
        }
    };

    WorkspacePanelModel {
        visible: !primary_shell.drawer_panel.is_hidden(),
        route: workspace.route,
        route_label: workspace.route.label().to_string(),
        selection_summary: selection
            .selected
            .and_then(|selected_id| resolve_presented_object(scene, selected_id))
            .and_then(|object| scene.nodes.get(&object.host_id))
            .map(|node| format!("Selected: {}", node.name))
            .unwrap_or_else(|| "No selection".to_string()),
        detail_text,
        context_rows,
        input_rows,
        output_rows,
    }
}

struct ScenePanelBuildContext<'a> {
    selection: &'a SceneSelectionState,
    scene_panel_ui: &'a ScenePanelUiState,
    filter_query: &'a str,
}

fn collect_scene_rows(
    scene: &Scene,
    object: PresentedObjectRef,
    selection: &SceneSelectionState,
    scene_panel_ui: &ScenePanelUiState,
    depth: usize,
    filter_query: &str,
    rows: &mut Vec<ScenePanelRow>,
) -> bool {
    let wrapper_chain = collect_presented_base_wrapper_chain(scene, object);
    let context = ScenePanelBuildContext {
        selection,
        scene_panel_ui,
        filter_query,
    };
    collect_scene_rows_for_object(scene, object, &wrapper_chain, &context, depth, rows)
}

fn collect_scene_rows_for_object(
    scene: &Scene,
    object: PresentedObjectRef,
    wrapper_chain: &[NodeId],
    context: &ScenePanelBuildContext<'_>,
    depth: usize,
    rows: &mut Vec<ScenePanelRow>,
) -> bool {
    if let Some((&wrapper_id, remaining_wrappers)) = wrapper_chain.split_first() {
        let Some(node) = scene.nodes.get(&wrapper_id) else {
            return false;
        };

        let mut child_rows = Vec::new();
        let descendant_matched = collect_scene_rows_for_object(
            scene,
            object,
            remaining_wrappers,
            context,
            depth + 1,
            &mut child_rows,
        );
        let matches_self = node_matches_filter(node, node_kind_label(node), context.filter_query);
        if !context.filter_query.is_empty() && !matches_self && !descendant_matched {
            return false;
        }

        let expanded = wrapper_row_expanded(
            context.scene_panel_ui,
            wrapper_id,
            depth,
            context.filter_query,
            descendant_matched,
        );
        rows.push(ScenePanelRow {
            host_id: wrapper_id,
            object_root_id: wrapper_id,
            label: node.name.clone(),
            kind_label: node_kind_label(node),
            depth,
            has_children: true,
            expanded,
            selected: context.selection.selected == Some(wrapper_id)
                || context.selection.selected_set.contains(&wrapper_id),
            hidden: scene.is_hidden(wrapper_id),
            locked: node.locked,
            renaming: context.scene_panel_ui.renaming_node == Some(wrapper_id),
            rename_value: if context.scene_panel_ui.renaming_node == Some(wrapper_id) {
                context.scene_panel_ui.rename_buffer.clone()
            } else {
                node.name.clone()
            },
            dragging: context.scene_panel_ui.drag_source == Some(wrapper_id),
            drop_allowed: context
                .scene_panel_ui
                .drag_source
                .is_some_and(|dragged| scene.is_valid_drop_target(wrapper_id, dragged)),
            drop_target: context.scene_panel_ui.drop_target == Some(wrapper_id),
        });
        if expanded {
            rows.extend(child_rows);
        }
        return true;
    }

    let Some(node) = scene.nodes.get(&object.host_id) else {
        return false;
    };

    let children = presented_children(scene, object);
    let has_children = !children.is_empty();
    let matches_self = node_matches_filter(
        node,
        object_kind_chip(object).to_string(),
        context.filter_query,
    );
    let manual_expanded = context.scene_panel_ui.is_expanded(object.host_id, depth);

    let mut child_rows = Vec::new();
    let mut descendant_matched = false;
    for child in children {
        descendant_matched |=
            collect_scene_rows_for_object(scene, child, &[], context, depth + 1, &mut child_rows);
    }

    if !context.filter_query.is_empty() && !matches_self && !descendant_matched {
        return false;
    }

    let expanded = has_children
        && if context.filter_query.is_empty() {
            manual_expanded
        } else {
            descendant_matched || manual_expanded
        };

    rows.push(ScenePanelRow {
        host_id: object.host_id,
        object_root_id: object.object_root_id,
        label: node.name.clone(),
        kind_label: object_kind_chip(object).to_string(),
        depth,
        has_children,
        expanded,
        selected: context.selection.selected_set.contains(&object.host_id)
            || context.selection.selected == Some(object.host_id),
        hidden: scene.is_hidden(object.object_root_id),
        locked: node.locked,
        renaming: context.scene_panel_ui.renaming_node == Some(object.host_id),
        rename_value: if context.scene_panel_ui.renaming_node == Some(object.host_id) {
            context.scene_panel_ui.rename_buffer.clone()
        } else {
            node.name.clone()
        },
        dragging: context.scene_panel_ui.drag_source == Some(object.object_root_id),
        drop_allowed: context
            .scene_panel_ui
            .drag_source
            .is_some_and(|dragged| scene.is_valid_drop_target(object.object_root_id, dragged)),
        drop_target: context.scene_panel_ui.drop_target == Some(object.object_root_id),
    });

    if expanded {
        rows.extend(child_rows);
    }

    true
}

fn wrapper_row_expanded(
    scene_panel_ui: &ScenePanelUiState,
    wrapper_id: NodeId,
    depth: usize,
    filter_query: &str,
    descendant_matched: bool,
) -> bool {
    let manual_expanded = scene_panel_ui.is_expanded(wrapper_id, depth);
    if filter_query.is_empty() {
        manual_expanded
    } else {
        descendant_matched || manual_expanded
    }
}

fn node_matches_filter(
    node: &crate::graph::scene::SceneNode,
    kind_label: String,
    filter_query: &str,
) -> bool {
    filter_query.is_empty()
        || node.name.to_ascii_lowercase().contains(filter_query)
        || kind_label.to_ascii_lowercase().contains(filter_query)
}

fn node_kind_label(node: &crate::graph::scene::SceneNode) -> String {
    match &node.data {
        NodeData::Primitive { kind, .. } => kind.base_name().to_ascii_uppercase(),
        NodeData::Transform { .. } => "TRANSFORM".to_string(),
        NodeData::Modifier { kind, .. } => kind.base_name().to_ascii_uppercase(),
        NodeData::Operation { op, .. } => op.base_name().to_ascii_uppercase(),
        NodeData::Sculpt { .. } => "SCULPT".to_string(),
        NodeData::Light { light_type, .. } => light_type.label().to_ascii_uppercase(),
    }
}

fn build_single_object_inspector(
    scene: &Scene,
    object: PresentedObjectRef,
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return InspectorModel {
            title: "Missing object".to_string(),
            name: String::new(),
            kind_label: "MISSING".to_string(),
            chips: Vec::new(),
            property_lines: vec!["The selected object is no longer present.".to_string()],
            display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
            multi_selection_summary: None,
            transform: None,
            material: None,
            operation: None,
            sculpt: None,
            light: None,
        };
    };

    let mut chips = vec![object_kind_chip(object).to_string()];
    if object.attached_sculpt_id.is_some() {
        chips.push("SCULPT".to_string());
    }
    if matches!(object.kind, PresentedObjectKind::Voxel) {
        chips.push("VOXEL".to_string());
    }
    if node.locked {
        chips.push("LOCKED".to_string());
    }

    let mut property_lines = match &node.data {
        NodeData::Primitive {
            kind,
            position,
            rotation,
            scale,
            material,
            ..
        } => vec![
            format!("Primitive: {}", kind.base_name()),
            format_vec3("Position", *position),
            format_vec3_degrees("Rotation", *rotation),
            format_vec3("Scale", *scale),
            format_material(material),
        ],
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            left,
            right,
        } => vec![
            format!("Operation: {}", op.base_name()),
            format!("Smooth K: {:.3}", smooth_k),
            format!("Steps: {:.0}", steps),
            format!("Color blend: {:.3}", color_blend),
            format_child(scene, "Left", *left),
            format_child(scene, "Right", *right),
        ],
        NodeData::Sculpt {
            input,
            position,
            rotation,
            material,
            layer_intensity,
            voxel_grid,
            desired_resolution,
        } => vec![
            if input.is_some() {
                "Sculpt layer: Attached".to_string()
            } else {
                "Sculpt object: Standalone".to_string()
            },
            format_vec3("Position", *position),
            format_vec3_degrees("Rotation", *rotation),
            format!("Resolution: {}^3", desired_resolution),
            format!("Detail size: {:.4}", voxel_grid.voxel_pitch()),
            format!("Layer intensity: {:.2}", layer_intensity),
            format_material(material),
        ],
        NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
            ..
        } => vec![
            format!("Light: {}", light_type.label()),
            format!("Color: {:.2}, {:.2}, {:.2}", color.x, color.y, color.z),
            format!("Intensity: {:.2}", intensity),
            format!("Range: {:.2}", range),
            format!("Shadows: {}", on_off(*cast_shadows)),
            format!("Volumetric: {}", on_off(*volumetric)),
            format!("Volumetric density: {:.2}", volumetric_density),
        ],
        NodeData::Transform { .. } | NodeData::Modifier { .. } => {
            vec!["This object is represented through its host wrapper stack.".to_string()]
        }
    };
    if object.attached_sculpt_id.is_some()
        && !matches!(node.data, NodeData::Sculpt { input: None, .. })
    {
        property_lines.push("Attached sculpt layer is active on this object.".to_string());
    }

    let operation = match &node.data {
        NodeData::Operation {
            op,
            smooth_k,
            steps,
            color_blend,
            ..
        } => Some(InspectorOperationModel {
            op_label: op.base_name().to_string(),
            smooth_k: float_field(*smooth_k, 0.0, 4.0, 0.01, 3),
            steps: int_field(*steps, 0.0, 32.0, 1.0),
            color_blend: float_field(*color_blend, -1.0, 1.0, 0.01, 2),
        }),
        _ => None,
    };

    let sculpt = match &node.data {
        NodeData::Sculpt {
            desired_resolution,
            layer_intensity,
            ..
        } => Some(InspectorSculptModel {
            desired_resolution: int_field(*desired_resolution as f32, 8.0, 512.0, 8.0),
            layer_intensity: float_field(*layer_intensity, 0.0, 4.0, 0.01, 2),
            brush_radius: float_field(sculpt_state.selected_profile().radius, 0.05, 2.0, 0.01, 2),
            brush_strength: float_field(
                sculpt_state.selected_profile().strength,
                -2.0,
                2.0,
                0.01,
                2,
            ),
        }),
        _ => None,
    };

    let light = match &node.data {
        NodeData::Light {
            light_type,
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
            ..
        } => Some(InspectorLightModel {
            light_type_label: light_type.label().to_string(),
            color: [
                float_field(color.x, 0.0, 10.0, 0.01, 2),
                float_field(color.y, 0.0, 10.0, 0.01, 2),
                float_field(color.z, 0.0, 10.0, 0.01, 2),
            ],
            intensity: float_field(*intensity, 0.0, 20.0, 0.05, 2),
            range: float_field(*range, 0.01, 50.0, 0.05, 2),
            cast_shadows: bool_field(*cast_shadows),
            volumetric: bool_field(*volumetric),
            volumetric_density: float_field(*volumetric_density, 0.0, 1.0, 0.01, 2),
        }),
        _ => None,
    };

    InspectorModel {
        title: node.name.clone(),
        name: node.name.clone(),
        kind_label: object_kind_chip(object).to_string(),
        chips,
        property_lines,
        display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
        multi_selection_summary: None,
        transform: build_transform_model(scene, &[object]),
        material: build_material_model(scene, &[object]),
        operation,
        sculpt,
        light,
    }
}

fn build_multi_selection_inspector(
    scene: &Scene,
    objects: &[PresentedObjectRef],
    settings: &Settings,
    sculpt_state: &SculptState,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> InspectorModel {
    InspectorModel {
        title: format!("{} objects selected", objects.len()),
        name: String::new(),
        kind_label: "MULTI".to_string(),
        chips: vec!["MULTI".to_string()],
        property_lines: objects
            .iter()
            .filter_map(|object| scene.nodes.get(&object.host_id))
            .map(|node| format!("• {}", node.name))
            .collect(),
        display_lines: display_lines(settings, interaction_mode, gizmo_mode, gizmo_space),
        multi_selection_summary: Some(format!("{} selected", objects.len())),
        transform: build_transform_model(scene, objects).map(|mut model| {
            model.multi_editing = true;
            model
        }),
        material: build_material_model(scene, objects),
        operation: None,
        sculpt: Some(InspectorSculptModel {
            desired_resolution: disabled_field("Single selection".to_string(), 8.0, 512.0, 8.0),
            layer_intensity: disabled_field("Single selection".to_string(), 0.0, 4.0, 0.01),
            brush_radius: float_field(sculpt_state.selected_profile().radius, 0.05, 2.0, 0.01, 2),
            brush_strength: float_field(
                sculpt_state.selected_profile().strength,
                -2.0,
                2.0,
                0.01,
                2,
            ),
        }),
        light: None,
    }
}

fn build_transform_model(
    scene: &Scene,
    objects: &[PresentedObjectRef],
) -> Option<InspectorTransformModel> {
    let targets = objects
        .iter()
        .filter_map(|object| editable_transform_target(scene, *object))
        .collect::<Vec<_>>();
    if targets.len() != objects.len() || targets.is_empty() {
        return None;
    }

    let mut positions = [Vec::new(), Vec::new(), Vec::new()];
    let mut rotations = [Vec::new(), Vec::new(), Vec::new()];
    let mut scales = [Vec::new(), Vec::new(), Vec::new()];
    let mut can_scale = true;

    for target_id in targets {
        let node = scene.nodes.get(&target_id)?;
        match &node.data {
            NodeData::Primitive {
                position,
                rotation,
                scale,
                ..
            } => {
                positions[0].push(position.x);
                positions[1].push(position.y);
                positions[2].push(position.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                scales[0].push(scale.x);
                scales[1].push(scale.y);
                scales[2].push(scale.z);
            }
            NodeData::Sculpt {
                position, rotation, ..
            } => {
                positions[0].push(position.x);
                positions[1].push(position.y);
                positions[2].push(position.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                can_scale = false;
            }
            NodeData::Transform {
                translation,
                rotation,
                scale,
                ..
            } => {
                positions[0].push(translation.x);
                positions[1].push(translation.y);
                positions[2].push(translation.z);
                rotations[0].push(rotation.x.to_degrees());
                rotations[1].push(rotation.y.to_degrees());
                rotations[2].push(rotation.z.to_degrees());
                scales[0].push(scale.x);
                scales[1].push(scale.y);
                scales[2].push(scale.z);
            }
            _ => return None,
        }
    }

    Some(InspectorTransformModel {
        position: [
            float_field_from_values(&positions[0], -20.0, 20.0, 0.01, 3),
            float_field_from_values(&positions[1], -20.0, 20.0, 0.01, 3),
            float_field_from_values(&positions[2], -20.0, 20.0, 0.01, 3),
        ],
        rotation_deg: [
            float_field_from_values(&rotations[0], -180.0, 180.0, 1.0, 1),
            float_field_from_values(&rotations[1], -180.0, 180.0, 1.0, 1),
            float_field_from_values(&rotations[2], -180.0, 180.0, 1.0, 1),
        ],
        scale: [
            float_field_from_values(&scales[0], 0.01, 10.0, 0.01, 3),
            float_field_from_values(&scales[1], 0.01, 10.0, 0.01, 3),
            float_field_from_values(&scales[2], 0.01, 10.0, 0.01, 3),
        ],
        can_scale,
        multi_editing: objects.len() > 1,
    })
}

fn build_material_model(
    scene: &Scene,
    objects: &[PresentedObjectRef],
) -> Option<InspectorMaterialModel> {
    let mut red = Vec::new();
    let mut green = Vec::new();
    let mut blue = Vec::new();
    let mut roughness = Vec::new();
    let mut metallic = Vec::new();

    for object in objects {
        let node = scene.nodes.get(&object.host_id)?;
        let material = node.data.material()?;
        red.push(material.base_color.x);
        green.push(material.base_color.y);
        blue.push(material.base_color.z);
        roughness.push(material.roughness);
        metallic.push(material.metallic);
    }

    if red.is_empty() {
        return None;
    }

    Some(InspectorMaterialModel {
        base_color: [
            float_field_from_values(&red, 0.0, 1.0, 0.01, 2),
            float_field_from_values(&green, 0.0, 1.0, 0.01, 2),
            float_field_from_values(&blue, 0.0, 1.0, 0.01, 2),
        ],
        roughness: float_field_from_values(&roughness, 0.0, 1.0, 0.01, 2),
        metallic: float_field_from_values(&metallic, 0.0, 1.0, 0.01, 2),
    })
}

fn editable_transform_target(scene: &Scene, object: PresentedObjectRef) -> Option<NodeId> {
    match object.kind {
        PresentedObjectKind::Parametric => current_transform_owner(scene, object.host_id),
        PresentedObjectKind::Voxel => Some(object.host_id),
        PresentedObjectKind::Light => match scene
            .nodes
            .get(&object.object_root_id)
            .map(|node| &node.data)
        {
            Some(NodeData::Transform { .. }) => Some(object.object_root_id),
            _ => Some(object.host_id),
        },
    }
}

fn build_node_workspace_rows(
    scene: &Scene,
    selected: Option<NodeId>,
    context_rows: &mut Vec<WorkspaceSummaryEntry>,
    input_rows: &mut Vec<WorkspaceSummaryEntry>,
    output_rows: &mut Vec<WorkspaceSummaryEntry>,
) {
    let Some(selected_id) = selected else {
        context_rows.push(WorkspaceSummaryEntry {
            label: "Selection".to_string(),
            value: "Choose an object to inspect graph context".to_string(),
        });
        return;
    };
    let Some(object) = resolve_presented_object(scene, selected_id) else {
        return;
    };
    let Some(node) = scene.nodes.get(&object.object_root_id) else {
        return;
    };

    context_rows.push(WorkspaceSummaryEntry {
        label: "Selected host".to_string(),
        value: scene
            .nodes
            .get(&object.host_id)
            .map(|node| node.name.clone())
            .unwrap_or_else(|| "Missing".to_string()),
    });
    context_rows.push(WorkspaceSummaryEntry {
        label: "Object root".to_string(),
        value: node.name.clone(),
    });
    context_rows.push(WorkspaceSummaryEntry {
        label: "Kind".to_string(),
        value: object_kind_chip(object).to_string(),
    });

    match &node.data {
        NodeData::Primitive { .. } | NodeData::Light { .. } => {
            input_rows.push(WorkspaceSummaryEntry {
                label: "Inputs".to_string(),
                value: "Leaf object".to_string(),
            })
        }
        NodeData::Operation { left, right, .. } => {
            input_rows.push(WorkspaceSummaryEntry {
                label: "Left".to_string(),
                value: child_name(scene, *left),
            });
            input_rows.push(WorkspaceSummaryEntry {
                label: "Right".to_string(),
                value: child_name(scene, *right),
            });
        }
        NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. }
        | NodeData::Sculpt { input, .. } => input_rows.push(WorkspaceSummaryEntry {
            label: "Input".to_string(),
            value: child_name(scene, *input),
        }),
    }

    let parents = parent_names(scene, object.object_root_id);
    if parents.is_empty() {
        output_rows.push(WorkspaceSummaryEntry {
            label: "Outputs".to_string(),
            value: "Top-level object".to_string(),
        });
    } else {
        for (index, parent) in parents.into_iter().enumerate() {
            output_rows.push(WorkspaceSummaryEntry {
                label: format!("Parent {}", index + 1),
                value: parent,
            });
        }
    }
}

fn build_light_workspace_rows(
    scene: &Scene,
    selected: Option<NodeId>,
    context_rows: &mut Vec<WorkspaceSummaryEntry>,
    input_rows: &mut Vec<WorkspaceSummaryEntry>,
    output_rows: &mut Vec<WorkspaceSummaryEntry>,
) {
    context_rows.push(WorkspaceSummaryEntry {
        label: "Scene lights".to_string(),
        value: scene
            .nodes
            .values()
            .filter(|node| matches!(node.data, NodeData::Light { .. }))
            .count()
            .to_string(),
    });

    match selected.and_then(|selected_id| resolve_presented_object(scene, selected_id)) {
        Some(object) if matches!(object.kind, PresentedObjectKind::Light) => {
            context_rows.push(WorkspaceSummaryEntry {
                label: "Selected light".to_string(),
                value: scene
                    .nodes
                    .get(&object.host_id)
                    .map(|node| node.name.clone())
                    .unwrap_or_else(|| "Missing".to_string()),
            });
        }
        _ => {
            context_rows.push(WorkspaceSummaryEntry {
                label: "Selection".to_string(),
                value: "Select a light to inspect routing context".to_string(),
            });
        }
    }

    input_rows.push(WorkspaceSummaryEntry {
        label: "Light source".to_string(),
        value: "Direct scene light".to_string(),
    });
    output_rows.push(WorkspaceSummaryEntry {
        label: "Affected geometry".to_string(),
        value: scene
            .nodes
            .values()
            .filter(|node| node.data.material().is_some())
            .count()
            .to_string(),
    });
}

fn parent_names(scene: &Scene, child_id: NodeId) -> Vec<String> {
    scene
        .nodes
        .values()
        .filter(|node| node.data.children().any(|candidate| candidate == child_id))
        .map(|node| node.name.clone())
        .collect()
}

fn child_name(scene: &Scene, child: Option<NodeId>) -> String {
    child
        .and_then(|child_id| scene.nodes.get(&child_id))
        .map(|node| node.name.clone())
        .unwrap_or_else(|| "(empty)".to_string())
}

fn float_field(
    value: f32,
    minimum: f32,
    maximum: f32,
    step: f32,
    precision: usize,
) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value,
        display_text: format!("{value:.precision$}"),
        enabled: true,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn int_field(value: f32, minimum: f32, maximum: f32, step: f32) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value,
        display_text: format!("{value:.0}"),
        enabled: true,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn disabled_field(
    display_text: String,
    minimum: f32,
    maximum: f32,
    step: f32,
) -> InspectorScalarFieldModel {
    InspectorScalarFieldModel {
        value: minimum,
        display_text,
        enabled: false,
        mixed: false,
        minimum,
        maximum,
        step,
    }
}

fn bool_field(value: bool) -> InspectorBoolFieldModel {
    InspectorBoolFieldModel {
        value,
        display_text: if value { "On" } else { "Off" }.to_string(),
        enabled: true,
        mixed: false,
    }
}

fn float_field_from_values(
    values: &[f32],
    minimum: f32,
    maximum: f32,
    step: f32,
    precision: usize,
) -> InspectorScalarFieldModel {
    let average = if values.is_empty() {
        minimum
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    };
    let mixed = values
        .first()
        .is_some_and(|first| values.iter().any(|value| (*value - *first).abs() > 0.0001));
    InspectorScalarFieldModel {
        value: average,
        display_text: if mixed {
            "Mixed".to_string()
        } else {
            format!("{average:.precision$}")
        },
        enabled: true,
        mixed,
        minimum,
        maximum,
        step,
    }
}

fn display_lines(
    settings: &Settings,
    interaction_mode: InteractionMode,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
) -> Vec<String> {
    vec![
        format!("Grid: {}", on_off(settings.render.show_grid)),
        format!("Node labels: {}", on_off(settings.render.show_node_labels)),
        format!(
            "Bounding box: {}",
            on_off(settings.render.show_bounding_box)
        ),
        format!(
            "Light gizmos: {}",
            on_off(settings.render.show_light_gizmos)
        ),
        format!("Shadows: {}", on_off(settings.render.shadows_enabled)),
        format!("Ambient occlusion: {}", on_off(settings.render.ao_enabled)),
        format!("Interaction: {}", interaction_mode_label(interaction_mode)),
        format!("Transform: {}", gizmo_mode.label()),
        format!("Space: {}", gizmo_space.label()),
    ]
}

fn object_kind_chip(object: PresentedObjectRef) -> &'static str {
    match object.kind {
        PresentedObjectKind::Parametric => "OBJECT",
        PresentedObjectKind::Voxel => "VOXEL",
        PresentedObjectKind::Light => "LIGHT",
    }
}

fn interaction_mode_label(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Select => "Select",
        InteractionMode::Measure => "Measure",
        InteractionMode::Sculpt(brush) => brush.label(),
    }
}

fn on_off(value: bool) -> &'static str {
    if value {
        "On"
    } else {
        "Off"
    }
}

fn format_vec3(label: &str, value: glam::Vec3) -> String {
    format!("{label}: {:.3}, {:.3}, {:.3}", value.x, value.y, value.z)
}

fn format_vec3_degrees(label: &str, value: glam::Vec3) -> String {
    format!(
        "{label}: {:.1}, {:.1}, {:.1} deg",
        value.x.to_degrees(),
        value.y.to_degrees(),
        value.z.to_degrees()
    )
}

fn format_material(material: &crate::graph::scene::MaterialParams) -> String {
    format!(
        "Material: color {:.2}/{:.2}/{:.2}, roughness {:.2}, metallic {:.2}",
        material.base_color.x,
        material.base_color.y,
        material.base_color.z,
        material.roughness,
        material.metallic
    )
}

fn format_child(scene: &Scene, label: &str, child: Option<NodeId>) -> String {
    format!("{label}: {}", child_name(scene, child))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::{
        MenuDropdownKind, MenuUiState, PanelBarId, PanelFrameworkState, PanelKind,
    };
    use crate::graph::scene::{CsgOp, LightType, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
    use crate::settings::{GroupRotateDirection, MultiAxisOrientation, MultiPivotMode};
    use glam::Vec3;

    fn overlay_layout(width: f32, height: f32) -> OverlayLayoutModel {
        build_overlay_layout_model([width, height], [0.0; 4], [0.0; 2], [0.0; 2])
    }

    fn tool_context_for_mode(interaction_mode: InteractionMode) -> ToolContextModel {
        let mut shell = PrimaryShellState::default();
        shell.interaction_mode = interaction_mode;
        build_tool_context_model(&shell)
    }

    #[test]
    fn scene_panel_marks_selected_row() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);

        let model = build_scene_panel_model(&scene, &selection, &ScenePanelUiState::default());

        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.selected));
    }

    #[test]
    fn scene_panel_filter_expands_matching_ancestor_rows() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let transform = scene.create_transform(Some(sphere));
        let mut ui = ScenePanelUiState::default();
        ui.filter_query = "sphere".to_string();

        let model = build_scene_panel_model(&scene, &SceneSelectionState::default(), &ui);

        let parent_row = model
            .rows
            .iter()
            .find(|row| row.object_root_id == transform)
            .expect("transform row");
        assert!(parent_row.expanded);
        assert!(model
            .rows
            .iter()
            .any(|row| row.host_id == sphere && row.depth == parent_row.depth + 1));
    }

    #[test]
    fn utility_maps_environment_settings() {
        let mut settings = Settings::default();
        settings.render.environment_source = EnvironmentSource::Hdri;
        settings.render.hdri_path = Some("studio.hdr".to_string());
        settings.render.environment_background_mode = EnvironmentBackgroundMode::Procedural;

        let model = build_utility_model(
            &History::new(),
            &ReferenceImageStore::default(),
            &ExpertPanelRegistry::default(),
            &settings,
        );

        assert!(model.render_settings.environment_is_hdri);
        assert_eq!(model.render_settings.hdri_path_display, "studio.hdr");
        assert!(model.render_settings.background_is_procedural);
    }

    #[test]
    fn tool_palette_marks_select_mode_active() {
        let model =
            build_tool_palette_model(&build_tool_context_model(&PrimaryShellState::default()));

        assert!(model.select_tool.active);
        assert!(model.brush_tools.iter().all(|tool| !tool.active));
    }

    #[test]
    fn tool_palette_marks_active_brush() {
        let mut shell = PrimaryShellState::default();
        shell.interaction_mode = InteractionMode::Sculpt(BrushMode::Flatten);

        let model = build_tool_palette_model(&build_tool_context_model(&shell));

        assert!(!model.select_tool.active);
        let active_labels = model
            .brush_tools
            .iter()
            .filter(|tool| tool.active)
            .map(|tool| tool.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(active_labels, vec!["Flatten"]);
    }

    #[test]
    fn menu_strip_marks_settings_item_active_when_settings_card_open() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.open_settings_card();

        let model = build_menu_strip_model(&menu_ui);

        assert!(model
            .items
            .iter()
            .any(|item| item.kind == MenuStripKind::Settings && item.active));
        assert!(model
            .items
            .iter()
            .filter(|item| item.active)
            .all(|item| item.kind == MenuStripKind::Settings));
    }

    #[test]
    fn menu_strip_visibility_comes_from_menu_state() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.strip_visible = false;

        let model = build_menu_strip_model(&menu_ui);

        assert!(!model.visible);
    }

    #[test]
    fn menu_dropdown_file_commands_follow_file_action_enablement() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.open_dropdown(MenuDropdownKind::File);

        let disabled = build_menu_dropdown_model(
            &menu_ui,
            false,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability::default(),
        );
        assert!(disabled.visible);
        assert!(disabled.items.iter().all(|item| !item.enabled));

        let enabled = build_menu_dropdown_model(
            &menu_ui,
            true,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability::default(),
        );
        assert!(enabled.items.iter().all(|item| item.enabled));
    }

    #[test]
    fn menu_dropdown_exposes_shortcut_labels_from_keymap() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.open_dropdown(MenuDropdownKind::Edit);

        let model = build_menu_dropdown_model(
            &menu_ui,
            true,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability::default(),
        );
        let undo = model
            .items
            .iter()
            .find(|item| item.command == MenuCommandKind::Undo)
            .expect("undo menu command");
        let copy = model
            .items
            .iter()
            .find(|item| item.command == MenuCommandKind::Copy)
            .expect("copy menu command");

        assert_eq!(undo.shortcut_label, "Ctrl+Z");
        assert_eq!(copy.shortcut_label, "Ctrl+C");
    }

    #[test]
    fn menu_dropdown_edit_and_view_enablement_uses_runtime_availability() {
        let mut edit_menu = MenuUiState::default();
        edit_menu.open_dropdown(MenuDropdownKind::Edit);

        let disabled_edit = build_menu_dropdown_model(
            &edit_menu,
            true,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability::default(),
        );

        assert!(disabled_edit
            .items
            .iter()
            .any(|item| item.command == MenuCommandKind::Undo && !item.enabled));
        assert!(disabled_edit
            .items
            .iter()
            .any(|item| item.command == MenuCommandKind::Paste && !item.enabled));
        assert!(disabled_edit
            .items
            .iter()
            .any(|item| item.command == MenuCommandKind::DeleteSelected && !item.enabled));

        let enabled_edit = build_menu_dropdown_model(
            &edit_menu,
            true,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability {
                undo_enabled: true,
                redo_enabled: true,
                copy_enabled: true,
                paste_enabled: true,
                duplicate_enabled: true,
                delete_enabled: true,
                focus_selected_enabled: true,
            },
        );

        assert!(enabled_edit.items.iter().all(|item| item.enabled));

        let mut view_menu = MenuUiState::default();
        view_menu.open_dropdown(MenuDropdownKind::View);
        let disabled_focus = build_menu_dropdown_model(
            &view_menu,
            true,
            &Settings::default(),
            MenuCommandCheckState::default(),
            MenuCommandAvailability::default(),
        );
        assert!(disabled_focus
            .items
            .iter()
            .any(|item| item.command == MenuCommandKind::FocusSelected && !item.enabled));
    }

    #[test]
    fn menu_dropdown_marks_toggle_commands_checked_from_runtime_state() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.open_dropdown(MenuDropdownKind::View);

        let model = build_menu_dropdown_model(
            &menu_ui,
            true,
            &Settings::default(),
            MenuCommandCheckState {
                ortho_enabled: true,
                measurement_enabled: false,
                turntable_enabled: true,
                help_visible: false,
                command_palette_visible: false,
            },
            MenuCommandAvailability::default(),
        );

        let ortho = model
            .items
            .iter()
            .find(|item| item.command == MenuCommandKind::ToggleOrtho)
            .expect("ortho command");
        let measure = model
            .items
            .iter()
            .find(|item| item.command == MenuCommandKind::ToggleMeasure)
            .expect("measure command");
        let turntable = model
            .items
            .iter()
            .find(|item| item.command == MenuCommandKind::ToggleTurntable)
            .expect("turntable command");

        assert!(ortho.checked);
        assert!(!measure.checked);
        assert!(turntable.checked);
    }

    #[test]
    fn settings_card_model_reflects_current_settings_values() {
        let mut menu_ui = MenuUiState::default();
        menu_ui.open_settings_card();
        let mut settings = Settings::default();
        settings.selection_behavior.multi_axis_orientation = MultiAxisOrientation::ActiveObject;
        settings.selection_behavior.group_rotate_direction = GroupRotateDirection::Inverted;
        settings.selection_behavior.multi_pivot_mode = MultiPivotMode::ActiveObject;
        settings.auto_save_enabled = false;
        settings.show_fps_overlay = false;
        settings.continuous_repaint = true;

        let model = build_settings_card_model(&menu_ui, &settings);

        assert!(model.visible);
        assert_eq!(
            model.multi_axis_orientation,
            MultiAxisOrientation::ActiveObject
        );
        assert_eq!(model.group_rotate_direction, GroupRotateDirection::Inverted);
        assert_eq!(model.multi_pivot_mode, MultiPivotMode::ActiveObject);
        assert!(!model.auto_save_enabled);
        assert!(!model.show_fps_overlay);
        assert!(model.continuous_repaint);
    }

    #[test]
    fn panel_framework_model_marks_open_transient_panel_active() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));

        assert_eq!(
            model.transient_panel.as_ref().map(|panel| panel.kind),
            Some(PanelKind::ObjectProperties)
        );
        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::ObjectProperties && item.active));
    }

    #[test]
    fn panel_framework_model_orders_tool_first() {
        let model = build_panel_framework_model(
            &PanelFrameworkState::default(),
            &overlay_layout(1200.0, 800.0),
        );

        let order = model
            .bar
            .items
            .iter()
            .map(|item| item.kind)
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            vec![
                PanelKind::Tool,
                PanelKind::ObjectProperties,
                PanelKind::RenderSettings,
                PanelKind::Scene,
                PanelKind::History,
                PanelKind::ReferenceImages,
            ]
        );
    }

    #[test]
    fn panel_framework_model_opens_tool_panel_with_expected_title() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::Tool, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));
        let panel = model.transient_panel.expect("tool panel");

        assert_eq!(panel.kind, PanelKind::Tool);
        assert_eq!(panel.title, "Tool");
    }

    #[test]
    fn panel_framework_model_marks_pinned_panel_without_duplication() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        state.toggle_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));

        assert!(model.transient_panel.is_none());
        assert_eq!(model.pinned_panels.len(), 1);
        assert_eq!(model.pinned_panels[0].kind, PanelKind::RenderSettings);
        assert!(model.pinned_panels[0].movable);
        assert!(model.pinned_panels[0].resizable);
        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::RenderSettings && item.pinned && item.active));
        assert!(model
            .bar
            .items
            .iter()
            .all(|item| item.kind != PanelKind::RenderSettings || !item.show_drag_indicator));
    }

    #[test]
    fn panel_framework_model_uses_collapsed_frame_for_collapsed_pinned_panel() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);
        state.pin_panel(PanelKind::RenderSettings);
        state.toggle_pinned_collapsed(PanelKind::RenderSettings);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));
        let panel = model
            .pinned_panels
            .iter()
            .find(|panel| panel.kind == PanelKind::RenderSettings)
            .expect("collapsed pinned panel");

        assert!(panel.collapsed);
        assert!((panel.collapsed_width - PanelFrameworkState::COLLAPSED_PANEL_WIDTH).abs() < 0.01);
        assert!(
            (panel.collapsed_height - PanelFrameworkState::COLLAPSED_PANEL_HEIGHT).abs() < 0.01
        );
        assert!((panel.frame.width - PanelFrameworkState::COLLAPSED_PANEL_WIDTH).abs() < 0.01);
        assert!((panel.frame.height - PanelFrameworkState::COLLAPSED_PANEL_HEIGHT).abs() < 0.01);
        assert!(!panel.resizable);
    }

    #[test]
    fn panel_framework_model_resolves_default_transient_frame() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));
        let panel = model.transient_panel.expect("transient panel");

        assert!((panel.frame.width - 390.0).abs() < 0.01);
        assert!((panel.frame.x - 632.0).abs() < 0.01);
        assert!((panel.frame.y - 20.0).abs() < 0.01);
        assert!(panel.movable);
        assert!(panel.resizable);
    }

    #[test]
    fn panel_framework_model_clamps_remembered_transient_frame() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state
            .bar_mut(PanelBarId::PrimaryRight)
            .expect("primary bar")
            .transient_rect = Some(crate::app::ui_geometry::FloatingPanelBounds::from_min_size(
            900.0, 700.0, 390.0, 420.0,
        ));

        let model = build_panel_framework_model(&state, &overlay_layout(1000.0, 600.0));
        let panel = model.transient_panel.expect("transient panel");

        assert!((panel.frame.x - 590.0).abs() < 0.01);
        assert!((panel.frame.y - 160.0).abs() < 0.01);
        assert!((panel.frame.height - 420.0).abs() < 0.01);
    }

    #[test]
    fn panel_framework_model_marks_only_active_transient_launcher_with_drag_hint() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::RenderSettings, PanelBarId::PrimaryRight);

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));

        assert!(model
            .bar
            .items
            .iter()
            .any(|item| item.kind == PanelKind::RenderSettings && item.show_drag_indicator));
        assert!(model
            .bar
            .items
            .iter()
            .all(|item| item.kind == PanelKind::RenderSettings || !item.show_drag_indicator));
    }

    #[test]
    fn panel_framework_model_exposes_active_resize_interaction_metadata() {
        let mut state = PanelFrameworkState::default();
        state.open_panel(PanelKind::ObjectProperties, PanelBarId::PrimaryRight);
        state.begin_panel_interaction(
            PanelKind::ObjectProperties,
            PanelBarId::PrimaryRight,
            PanelPointerInteractionKind::Resize(PanelResizeHandle::BottomRight),
        );

        let model = build_panel_framework_model(&state, &overlay_layout(1200.0, 800.0));

        assert!(model.panel_interaction_active);
        assert_eq!(model.interaction_panel_kind, PanelKind::ObjectProperties);
        assert_eq!(
            model.active_interaction_kind,
            PanelPointerInteractionKind::Resize(PanelResizeHandle::BottomRight)
        );
        assert_eq!(
            model.active_resize_handle,
            Some(PanelResizeHandle::BottomRight)
        );
    }

    #[test]
    fn overlay_layout_model_shrinks_usable_height_for_keyboard() {
        let model = build_overlay_layout_model(
            [1200.0, 900.0],
            [16.0, 24.0, 18.0, 20.0],
            [0.0, 620.0],
            [1200.0, 280.0],
        );

        assert!((model.usable_x - 16.0).abs() < 0.01);
        assert!((model.usable_y - 24.0).abs() < 0.01);
        assert!((model.usable_width - 1166.0).abs() < 0.01);
        assert!((model.usable_height - 596.0).abs() < 0.01);
    }

    #[test]
    fn tool_context_treats_measure_as_select_family() {
        let context = tool_context_for_mode(InteractionMode::Measure);

        assert_eq!(context.mode, ToolPanelMode::Select);
        assert!(context.select_tool.active);
        assert!(context.brush_tools.iter().all(|tool| !tool.active));
    }

    #[test]
    fn inspector_model_keeps_no_selection_state_available() {
        let model = build_inspector_model(
            &Scene::new(),
            &SceneSelectionState::default(),
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        assert_eq!(model.title, "No selection");
        assert!(model
            .property_lines
            .iter()
            .any(|line| line.contains("Click an object")));
    }

    #[test]
    fn tool_panel_select_mode_shows_no_selection_empty_state() {
        let model = build_tool_panel_model(
            &Scene::new(),
            &SceneSelectionState::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &build_inspector_model(
                &Scene::new(),
                &SceneSelectionState::default(),
                &Settings::default(),
                &SculptState::new_inactive(),
                InteractionMode::Select,
                &GizmoMode::Translate,
                &GizmoSpace::Local,
            ),
            &tool_context_for_mode(InteractionMode::Select),
        );

        assert_eq!(model.mode, ToolPanelMode::Select);
        assert_eq!(model.title, "Tool");
        assert_eq!(model.empty_state, "No selection");
        assert!(model.transform.is_none());
        assert!(model.material.is_none());
        assert!(model.operation.is_none());
        assert!(model.light.is_none());
    }

    #[test]
    fn tool_panel_select_mode_uses_material_for_single_primitive() {
        let scene = Scene::new();
        let selected_id = scene
            .nodes
            .iter()
            .find(|(_, node)| matches!(node.data, NodeData::Primitive { .. }))
            .map(|(id, _)| *id)
            .expect("default primitive");
        let mut selection = SceneSelectionState::default();
        selection.select_single(selected_id);
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &inspector,
            &tool_context_for_mode(InteractionMode::Select),
        );

        assert_eq!(model.mode, ToolPanelMode::Select);
        assert_eq!(model.summary, inspector.title);
        assert!(model.transform.is_some());
        assert!(model.material.is_some());
        assert!(model.operation.is_none());
        assert!(model.light.is_none());
    }

    #[test]
    fn tool_panel_select_mode_uses_operation_quick_section() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let box_id = scene.create_primitive(SdfPrimitive::Box);
        let operation = scene.create_operation(CsgOp::Union, Some(sphere), Some(box_id));
        let mut selection = SceneSelectionState::default();
        selection.select_single(operation);
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &inspector,
            &tool_context_for_mode(InteractionMode::Select),
        );

        assert!(model.operation.is_some());
        assert!(model.material.is_none());
        assert!(model.light.is_none());
    }

    #[test]
    fn tool_panel_select_mode_uses_light_quick_section() {
        let mut scene = Scene::new();
        let (_light_id, light_transform_id) = scene.create_light(LightType::Point);
        let mut selection = SceneSelectionState::default();
        selection.select_single(light_transform_id);
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &inspector,
            &tool_context_for_mode(InteractionMode::Select),
        );

        assert!(model.transform.is_some());
        assert!(model.light.is_some());
        assert!(model.material.is_none());
        assert!(model.operation.is_none());
    }

    #[test]
    fn tool_panel_select_mode_keeps_multi_selection_compact() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let box_id = scene.create_primitive(SdfPrimitive::Box);
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);
        selection.selected_set.insert(box_id);
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &SculptState::new_inactive(),
            InteractionMode::Select,
            &inspector,
            &tool_context_for_mode(InteractionMode::Select),
        );

        assert_eq!(model.summary, "2 selected");
        assert!(model.transform.is_some());
        assert!(model.material.is_none());
        assert!(model.operation.is_none());
        assert!(model.light.is_none());
    }

    #[test]
    fn tool_panel_sculpt_mode_shows_brush_without_target() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);
        let sculpt_state = SculptState::new_inactive();
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &sculpt_state,
            InteractionMode::Sculpt(BrushMode::Smooth),
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &sculpt_state,
            InteractionMode::Sculpt(BrushMode::Smooth),
            &inspector,
            &tool_context_for_mode(InteractionMode::Sculpt(BrushMode::Smooth)),
        );

        assert_eq!(model.mode, ToolPanelMode::Sculpt);
        assert_eq!(model.summary, "Brush: Smooth");
        assert_eq!(model.empty_state, "No sculpt target selected");
        assert!(model.sculpt.is_some());
        assert!(!model.show_sculpt_target_fields);
    }

    #[test]
    fn tool_panel_sculpt_mode_shows_target_fields_for_attached_sculpt() {
        let mut scene = Scene::new();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt_id = scene.create_sculpt(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::splat(0.7),
            VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );
        let mut selection = SceneSelectionState::default();
        selection.select_single(sphere);
        let sculpt_state = SculptState::new_inactive();
        let inspector = build_inspector_model(
            &scene,
            &selection,
            &Settings::default(),
            &sculpt_state,
            InteractionMode::Sculpt(BrushMode::Add),
            &GizmoMode::Translate,
            &GizmoSpace::Local,
        );

        let model = build_tool_panel_model(
            &scene,
            &selection,
            &sculpt_state,
            InteractionMode::Sculpt(BrushMode::Add),
            &inspector,
            &tool_context_for_mode(InteractionMode::Sculpt(BrushMode::Add)),
        );

        assert_eq!(model.summary, "Brush: Add");
        assert_eq!(model.empty_state, "");
        assert!(model.show_sculpt_target_fields);
        let sculpt = model.sculpt.expect("sculpt tool section");
        assert_eq!(sculpt.desired_resolution.display_text, "8");
        assert_ne!(sculpt_id, sphere);
    }
}
