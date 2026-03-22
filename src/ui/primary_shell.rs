use std::collections::HashSet;

use eframe::egui;
use egui_dock::DockState;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink, WorkspacePreset};
use crate::app::state::{
    InteractionMode, MultiTransformSessionState, PrimaryShellContextTab, PrimaryShellDrawerTab,
    PrimaryShellState, ShellSnapAnchor, ShellWindowState,
};
use crate::graph::history::History;
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive};
use crate::material_preset::MaterialLibrary;
use crate::sculpt::{BrushMode, SculptState};
use crate::settings::{SelectionBehaviorSettings, Settings};
use crate::ui::dock::Tab;
use crate::ui::gizmo::GizmoSpace;
use crate::ui::reference_image::ReferenceImageManager;
use crate::ui::{
    brush_settings, history_panel, properties, reference_image, render_settings, scene_tree,
};

const SHELL_MARGIN: f32 = 12.0;
const LAUNCHER_HEIGHT: f32 = 40.0;
const WINDOW_SNAP_THRESHOLD: f32 = 24.0;
const TOOL_MIN_SIZE: egui::Vec2 = egui::vec2(260.0, 260.0);
const INSPECTOR_MIN_SIZE: egui::Vec2 = egui::vec2(280.0, 320.0);
const DRAWER_MIN_SIZE: egui::Vec2 = egui::vec2(360.0, 200.0);

pub struct PrimaryShellContext<'a> {
    pub shell: &'a mut PrimaryShellState,
    pub dock_state: Option<&'a mut DockState<Tab>>,
    pub scene: &'a mut Scene,
    pub sculpt_state: &'a mut SculptState,
    pub selected: &'a mut Option<NodeId>,
    pub selected_set: &'a mut HashSet<NodeId>,
    pub renaming_node: &'a mut Option<NodeId>,
    pub rename_buf: &'a mut String,
    pub scene_tree_drag: &'a mut Option<NodeId>,
    pub scene_tree_search: &'a mut String,
    pub bake_progress: Option<(u32, u32)>,
    pub actions: &'a mut ActionSink,
    pub history: &'a History,
    pub active_light_ids: &'a HashSet<NodeId>,
    pub max_sculpt_resolution: u32,
    pub soloed_light: Option<NodeId>,
    pub material_library: &'a mut MaterialLibrary,
    pub multi_transform_edit: &'a mut MultiTransformSessionState,
    pub gizmo_space: &'a GizmoSpace,
    pub selection_behavior: &'a SelectionBehaviorSettings,
    pub reference_images: &'a mut ReferenceImageManager,
    pub measurement_points: &'a mut Vec<Vec3>,
    pub show_distance_readout: &'a mut bool,
    pub settings: &'a mut Settings,
}

pub fn draw(ctx: &egui::Context, viewport_rect: egui::Rect, shell: PrimaryShellContext<'_>) {
    let mut shell = shell;
    sync_context_tab(shell.shell, shell.scene, *shell.selected);

    let launcher_rect = draw_launcher_strip(ctx, viewport_rect, &mut shell);
    let window_bounds = shell_window_bounds(viewport_rect, launcher_rect);

    draw_tool_panel_window(ctx, window_bounds, &mut shell);
    draw_inspector_panel_window(ctx, window_bounds, &mut shell);
    draw_drawer_window(ctx, window_bounds, &launcher_rect, &mut shell);
}

pub fn draw_tool_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_tool_panel_contents(ui, shell);
}

pub fn draw_inspector_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    sync_context_tab(shell.shell, shell.scene, *shell.selected);
    draw_inspector_panel_contents(ui, shell);
}

fn shell_frame(ctx: &egui::Context) -> egui::Frame {
    egui::Frame::window(&ctx.style())
        .fill(egui::Color32::from_rgba_premultiplied(26, 28, 34, 235))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_premultiplied(110, 120, 140, 180),
        ))
        .rounding(egui::Rounding::same(10.0))
        .inner_margin(egui::Margin::same(10.0))
}

fn draw_launcher_strip(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) -> egui::Rect {
    let bar_rect = egui::Rect::from_min_size(
        egui::pos2(
            viewport_rect.min.x + SHELL_MARGIN,
            viewport_rect.max.y - LAUNCHER_HEIGHT - SHELL_MARGIN,
        ),
        egui::vec2((viewport_rect.width() - SHELL_MARGIN * 2.0).max(140.0), LAUNCHER_HEIGHT),
    );

    egui::Area::new(egui::Id::new("primary_shell_launcher_strip"))
        .order(egui::Order::Foreground)
        .fixed_pos(bar_rect.min)
        .show(ctx, |ui| {
            shell_frame(ctx).show(ui, |ui| {
                ui.set_width(bar_rect.width());
                ui.horizontal_wrapped(|ui| {
                    drawer_tab_button(ui, shell, PrimaryShellDrawerTab::Items, "Items");
                    drawer_tab_button(ui, shell, PrimaryShellDrawerTab::History, "History");
                    drawer_tab_button(ui, shell, PrimaryShellDrawerTab::Reference, "Reference");
                    drawer_tab_button(ui, shell, PrimaryShellDrawerTab::Advanced, "Advanced");
                    ui.separator();
                    if ui
                        .selectable_label(shell.shell.tool_panel.open, "Tool")
                        .clicked()
                    {
                        shell.shell.toggle_tool_panel();
                    }
                    if ui
                        .selectable_label(shell.shell.inspector_panel.open, "Inspector")
                        .clicked()
                    {
                        shell.shell.toggle_inspector_panel();
                    }
                    ui.separator();
                    if ui.small_button("Reset Layout").clicked() {
                        shell.shell.reset_layout();
                    }
                });
            });
        });

    bar_rect
}

fn draw_tool_panel_window(
    ctx: &egui::Context,
    bounds: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let mut tool_panel = std::mem::replace(
        &mut shell.shell.tool_panel,
        ShellWindowState::snapped_default(false, egui::Vec2::ZERO, None),
    );
    let default_pos = default_panel_position(
        tool_panel.snap_anchor,
        tool_panel.size,
        bounds,
        egui::pos2(bounds.min.x, bounds.min.y),
    );
    let window_id = egui::Id::new(("primary_shell_tool_panel", shell.shell.layout_revision));
    let mut dock_clicked = false;
    let mut reset_clicked = false;

    show_shell_window(
        ctx,
        window_id,
        "Tool Panel",
        &mut tool_panel,
        bounds,
        default_pos,
        TOOL_MIN_SIZE,
        |ui| {
            shell_window_actions(ui, true, &mut dock_clicked, &mut reset_clicked);
            draw_tool_panel_contents(ui, shell);
        },
    );

    if dock_clicked {
        if let Some(dock_state) = shell.dock_state.as_deref_mut() {
            toggle_dock_tab(dock_state, Tab::ToolPanel);
            tool_panel.open = false;
        }
    }

    shell.shell.tool_panel = tool_panel;
    if reset_clicked {
        shell.shell.reset_layout();
    }
}

fn draw_inspector_panel_window(
    ctx: &egui::Context,
    bounds: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let mut inspector_panel = std::mem::replace(
        &mut shell.shell.inspector_panel,
        ShellWindowState::snapped_default(false, egui::Vec2::ZERO, None),
    );
    let default_pos = default_panel_position(
        inspector_panel.snap_anchor,
        inspector_panel.size,
        bounds,
        egui::pos2(bounds.max.x - inspector_panel.size.x, bounds.min.y),
    );
    let window_id =
        egui::Id::new(("primary_shell_inspector_panel", shell.shell.layout_revision));
    let mut dock_clicked = false;
    let mut reset_clicked = false;

    show_shell_window(
        ctx,
        window_id,
        "Inspector",
        &mut inspector_panel,
        bounds,
        default_pos,
        INSPECTOR_MIN_SIZE,
        |ui| {
            shell_window_actions(ui, true, &mut dock_clicked, &mut reset_clicked);
            draw_inspector_panel_contents(ui, shell);
        },
    );

    if dock_clicked {
        if let Some(dock_state) = shell.dock_state.as_deref_mut() {
            toggle_dock_tab(dock_state, Tab::InspectorPanel);
            inspector_panel.open = false;
        }
    }

    shell.shell.inspector_panel = inspector_panel;
    if reset_clicked {
        shell.shell.reset_layout();
    }
}

fn draw_drawer_window(
    ctx: &egui::Context,
    bounds: egui::Rect,
    launcher_rect: &egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let mut drawer_panel = std::mem::replace(
        &mut shell.shell.drawer_panel,
        ShellWindowState::snapped_default(false, egui::Vec2::ZERO, None),
    );
    let default_pos = default_panel_position(
        drawer_panel.snap_anchor,
        drawer_panel.size,
        bounds,
        egui::pos2(
            bounds.center().x - drawer_panel.size.x * 0.5,
            launcher_rect.min.y - drawer_panel.size.y - 8.0,
        ),
    );
    let window_id = egui::Id::new(("primary_shell_drawer_panel", shell.shell.layout_revision));
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let dockable = drawer_dock_tab(shell.shell.active_drawer_tab).is_some();

    show_shell_window(
        ctx,
        window_id,
        drawer_window_title(shell.shell.active_drawer_tab),
        &mut drawer_panel,
        bounds,
        default_pos,
        DRAWER_MIN_SIZE,
        |ui| {
            shell_window_actions(ui, dockable, &mut dock_clicked, &mut reset_clicked);
            draw_drawer_contents(ui, shell);
        },
    );

    if dock_clicked {
        if let (Some(dock_state), Some(tab)) = (
            shell.dock_state.as_deref_mut(),
            drawer_dock_tab(shell.shell.active_drawer_tab),
        ) {
            toggle_dock_tab(dock_state, tab);
            drawer_panel.open = false;
        }
    }

    shell.shell.drawer_panel = drawer_panel;
    if reset_clicked {
        shell.shell.reset_layout();
    }
}

#[allow(clippy::too_many_arguments)]
fn show_shell_window(
    ctx: &egui::Context,
    id: egui::Id,
    title: &str,
    state: &mut ShellWindowState,
    bounds: egui::Rect,
    default_pos: egui::Pos2,
    min_size: egui::Vec2,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    if !state.open {
        return;
    }

    state.size = clamp_window_size(state.size, bounds.size(), min_size);
    let current_pos = current_panel_position(state, bounds, default_pos);

    if let Some(window) = egui::Window::new(title)
        .id(id)
        .open(&mut state.open)
        .collapsible(false)
        .resizable(true)
        .default_pos(default_pos)
        .current_pos(current_pos)
        .default_size(state.size)
        .min_size(min_size)
        .constrain_to(bounds)
        .frame(shell_frame(ctx))
        .show(ctx, |ui| add_contents(ui))
    {
        let rect = window.response.rect;
        state.size = clamp_window_size(rect.size(), bounds.size(), min_size);
        state.position = Some(clamp_window_position(rect.min, state.size, bounds));
        state.snap_anchor = detect_snap_anchor(rect, bounds);
    }
}

fn shell_window_actions(
    ui: &mut egui::Ui,
    dockable: bool,
    dock_clicked: &mut bool,
    reset_clicked: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.small("Floating shell window");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Reset").clicked() {
                *reset_clicked = true;
            }
            if dockable && ui.small_button("Dock").clicked() {
                *dock_clicked = true;
            }
        });
    });
    ui.separator();
}

pub fn draw_tool_panel_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.label(egui::RichText::new("Interaction").small().strong());
    ui.horizontal_wrapped(|ui| {
        interaction_mode_button(ui, shell, InteractionMode::Select, "Select");
        interaction_mode_button(ui, shell, InteractionMode::Measure, "Measure");
        for brush in BrushMode::ALL {
            interaction_mode_button(ui, shell, InteractionMode::Sculpt(brush), brush.label());
        }
        ui.separator();
        if ui
            .selectable_label(*shell.show_distance_readout, "Distance")
            .clicked()
        {
            shell.actions.push(Action::ToggleDistanceReadout);
        }
    });

    if shell.shell.interaction_mode == InteractionMode::Measure {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.small(format!("{} point(s) placed", shell.measurement_points.len()));
            if ui.small_button("Clear Points").clicked() {
                shell.measurement_points.clear();
            }
        });
    }

    ui.separator();

    egui::CollapsingHeader::new("Create")
        .default_open(true)
        .show(ui, |ui| {
            let button_size = egui::vec2(74.0, 28.0);
            ui.horizontal_wrapped(|ui| {
                for primitive in SdfPrimitive::ALL {
                    if ui
                        .add(egui::Button::new(primitive.base_name()).min_size(button_size))
                        .clicked()
                    {
                        shell.actions.push(Action::CreatePrimitive(primitive.clone()));
                    }
                }
            });
        });

    egui::CollapsingHeader::new("Guided Boolean")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(selected_id) = *shell.selected {
                let selected_name = shell
                    .scene
                    .nodes
                    .get(&selected_id)
                    .map(|node| node.name.as_str())
                    .unwrap_or("Selected node");
                ui.small(format!("Base: {selected_name}"));
                ui.horizontal_wrapped(|ui| {
                    boolean_menu_button(ui, "Union +", CsgOp::Union, shell.actions);
                    boolean_menu_button(ui, "Subtract +", CsgOp::Subtract, shell.actions);
                    boolean_menu_button(ui, "Intersect +", CsgOp::Intersect, shell.actions);
                });
            } else {
                ui.weak("Select a node to add a guided boolean operand.");
            }
        });

    egui::CollapsingHeader::new("Wrap Selection")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(selected_id) = *shell.selected {
                if ui.button("Add Transform").clicked() {
                    shell
                        .actions
                        .push(Action::InsertTransformAbove { target: selected_id });
                }
                ui.menu_button("Add Modifier", |ui| {
                    for modifier in ModifierKind::ALL {
                        if ui.button(modifier.base_name()).clicked() {
                            shell.actions.push(Action::InsertModifierAbove {
                                target: selected_id,
                                kind: modifier.clone(),
                            });
                            ui.close_menu();
                        }
                    }
                });
            } else {
                ui.weak("Select a node to wrap it with a transform or modifier.");
            }
        });

    egui::CollapsingHeader::new("Scene")
        .default_open(true)
        .show(ui, |ui| {
            if ui.button("Frame All").clicked() {
                shell.actions.push(Action::FrameAll);
            }
        });
}

fn interaction_mode_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    mode: InteractionMode,
    label: &str,
) {
    let active = shell.shell.interaction_mode == mode;
    if ui.selectable_label(active, label).clicked() && !active {
        shell.actions.push(Action::SetInteractionMode(mode));
    }
}

fn boolean_menu_button(
    ui: &mut egui::Ui,
    label: &str,
    op: CsgOp,
    actions: &mut ActionSink,
) {
    ui.menu_button(label, |ui| {
        for primitive in SdfPrimitive::ALL {
            if ui.button(primitive.base_name()).clicked() {
                actions.push(Action::ShellCreateBooleanPrimitive {
                    op: op.clone(),
                    primitive: primitive.clone(),
                });
                ui.close_menu();
            }
        }
    });
}

fn sync_context_tab(
    shell: &mut PrimaryShellState,
    scene: &Scene,
    selected: Option<NodeId>,
) {
    let selected_node = selected.and_then(|node_id| scene.nodes.get(&node_id));
    let selected_is_light =
        selected_node.is_some_and(|node| matches!(node.data, NodeData::Light { .. }));
    let selected_has_material = selected_node.is_some_and(|node| {
        matches!(node.data, NodeData::Primitive { .. } | NodeData::Sculpt { .. })
    });

    match shell.interaction_mode {
        InteractionMode::Sculpt(_) => {
            if shell.active_context_tab == PrimaryShellContextTab::Selection {
                shell.active_context_tab = PrimaryShellContextTab::Sculpt;
            }
        }
        InteractionMode::Measure => {
            if shell.active_context_tab == PrimaryShellContextTab::Sculpt {
                shell.active_context_tab = PrimaryShellContextTab::Selection;
            }
        }
        InteractionMode::Select => {
            if selected_is_light && shell.active_context_tab == PrimaryShellContextTab::Selection {
                shell.active_context_tab = PrimaryShellContextTab::Light;
            } else if selected_has_material
                && shell.active_context_tab == PrimaryShellContextTab::Selection
            {
                shell.active_context_tab = PrimaryShellContextTab::Material;
            } else if shell.active_context_tab == PrimaryShellContextTab::Sculpt {
                shell.active_context_tab = PrimaryShellContextTab::Selection;
            }
        }
    }

    if shell.active_context_tab == PrimaryShellContextTab::Light && !selected_is_light {
        shell.active_context_tab = PrimaryShellContextTab::Selection;
    }
    if shell.active_context_tab == PrimaryShellContextTab::Material && !selected_has_material {
        shell.active_context_tab = PrimaryShellContextTab::Selection;
    }
}

fn draw_inspector_panel_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.horizontal_wrapped(|ui| {
        context_tab_button(ui, shell, PrimaryShellContextTab::Sculpt, "Sculpt");
        context_tab_button(ui, shell, PrimaryShellContextTab::Selection, "Selection");
        context_tab_button(ui, shell, PrimaryShellContextTab::Material, "Material");
        context_tab_button(ui, shell, PrimaryShellContextTab::Light, "Light");
        context_tab_button(ui, shell, PrimaryShellContextTab::Node, "Node");
    });
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| match shell.shell.active_context_tab {
            PrimaryShellContextTab::Sculpt => {
                brush_settings::draw(ui, shell.sculpt_state);
            }
            PrimaryShellContextTab::Selection => {
                if shell.shell.interaction_mode == InteractionMode::Measure {
                    draw_measurement_panel(ui, shell);
                } else {
                    draw_properties_panel(ui, shell);
                }
            }
            PrimaryShellContextTab::Light => {
                let selected_light_node =
                    (*shell.selected).and_then(|node_id| shell.scene.nodes.get(&node_id));
                let missing_light_selection = match selected_light_node {
                    Some(node) => !matches!(node.data, NodeData::Light { .. }),
                    None => true,
                };
                if missing_light_selection {
                    ui.weak("Select a light node to focus the light inspector.");
                }
                draw_properties_panel(ui, shell);
            }
            PrimaryShellContextTab::Material | PrimaryShellContextTab::Node => {
                draw_properties_panel(ui, shell);
            }
        });
}

fn draw_measurement_panel(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.strong("Measurement");
    ui.small("Measure mode is active. Tap two points in the viewport.");
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("Exit Measure").clicked() {
            shell
                .actions
                .push(Action::SetInteractionMode(InteractionMode::Select));
        }
        if ui.button("Clear Points").clicked() {
            shell.measurement_points.clear();
        }
    });
    ui.separator();

    match shell.measurement_points.as_slice() {
        [] => {
            ui.weak("No points selected yet.");
        }
        [p0] => {
            ui.monospace(format!("P1: {:.3}, {:.3}, {:.3}", p0.x, p0.y, p0.z));
            ui.weak("Select a second point to complete the measurement.");
        }
        [p0, p1, ..] => {
            ui.monospace(format!("P1: {:.3}, {:.3}, {:.3}", p0.x, p0.y, p0.z));
            ui.monospace(format!("P2: {:.3}, {:.3}, {:.3}", p1.x, p1.y, p1.z));
            ui.separator();
            ui.label(format!("Distance: {:.4}", p0.distance(*p1)));
        }
    }
}

fn draw_properties_panel(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    properties::draw(
        ui,
        shell.scene,
        *shell.selected,
        shell.selected_set,
        shell.sculpt_state,
        shell.bake_progress,
        shell.actions,
        shell.active_light_ids,
        shell.max_sculpt_resolution,
        shell.soloed_light,
        shell.material_library,
        shell.multi_transform_edit,
        shell.gizmo_space,
        shell.selection_behavior,
    );
}

fn context_tab_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellContextTab,
    label: &str,
) {
    if ui
        .selectable_label(shell.shell.active_context_tab == tab, label)
        .clicked()
    {
        shell.shell.active_context_tab = tab;
    }
}

fn draw_drawer_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    match shell.shell.active_drawer_tab {
        PrimaryShellDrawerTab::Items => {
            scene_tree::draw(
                ui,
                shell.scene,
                shell.selected,
                shell.selected_set,
                shell.renaming_node,
                shell.rename_buf,
                shell.scene_tree_drag,
                shell.actions,
                shell.scene_tree_search,
                shell.active_light_ids,
                shell.soloed_light,
            );
        }
        PrimaryShellDrawerTab::History => {
            history_panel::draw(ui, shell.history, shell.actions);
        }
        PrimaryShellDrawerTab::Reference => {
            reference_image::draw_controls(ui, shell.reference_images, shell.actions);
        }
        PrimaryShellDrawerTab::Advanced => {
            draw_advanced_drawer(ui, shell);
        }
    }
}

fn draw_advanced_drawer(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.strong("Advanced");
    ui.small(
        "Use docked expert panels when you need graph editing or denser desktop workflows.",
    );
    ui.separator();

    ui.label(egui::RichText::new("Expert Panels").small().strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Node Graph").clicked() {
            if let Some(dock_state) = shell.dock_state.as_deref_mut() {
                toggle_dock_tab(dock_state, Tab::NodeGraph);
            }
        }
        if ui.button("Properties").clicked() {
            if let Some(dock_state) = shell.dock_state.as_deref_mut() {
                toggle_dock_tab(dock_state, Tab::Properties);
            }
        }
        if ui.button("Render Settings").clicked() {
            if let Some(dock_state) = shell.dock_state.as_deref_mut() {
                toggle_dock_tab(dock_state, Tab::RenderSettings);
            }
        }
        if ui.button("Scene Stats").clicked() {
            if let Some(dock_state) = shell.dock_state.as_deref_mut() {
                toggle_dock_tab(dock_state, Tab::SceneStats);
            }
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Expert Workspaces").small().strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Modeling").clicked() {
            shell
                .actions
                .push(Action::SetWorkspace(WorkspacePreset::Modeling));
        }
        if ui.button("Sculpting").clicked() {
            shell
                .actions
                .push(Action::SetWorkspace(WorkspacePreset::Sculpting));
        }
        if ui.button("Rendering").clicked() {
            shell
                .actions
                .push(Action::SetWorkspace(WorkspacePreset::Rendering));
        }
    });

    ui.separator();
    render_settings::draw(ui, shell.settings, shell.actions);
}

fn drawer_tab_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellDrawerTab,
    label: &str,
) {
    let active = shell.shell.drawer_panel.open && shell.shell.active_drawer_tab == tab;
    if ui.selectable_label(active, label).clicked() {
        shell.shell.toggle_drawer(tab);
    }
}

fn drawer_window_title(tab: PrimaryShellDrawerTab) -> &'static str {
    match tab {
        PrimaryShellDrawerTab::Items => "Items",
        PrimaryShellDrawerTab::History => "History",
        PrimaryShellDrawerTab::Reference => "Reference",
        PrimaryShellDrawerTab::Advanced => "Advanced",
    }
}

fn drawer_dock_tab(tab: PrimaryShellDrawerTab) -> Option<Tab> {
    match tab {
        PrimaryShellDrawerTab::Items => Some(Tab::SceneTree),
        PrimaryShellDrawerTab::History => Some(Tab::History),
        PrimaryShellDrawerTab::Reference => Some(Tab::ReferenceImages),
        PrimaryShellDrawerTab::Advanced => Some(Tab::NodeGraph),
    }
}

fn shell_window_bounds(viewport_rect: egui::Rect, launcher_rect: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_max(
        viewport_rect.min + egui::vec2(SHELL_MARGIN, SHELL_MARGIN),
        egui::pos2(
            viewport_rect.max.x - SHELL_MARGIN,
            launcher_rect.min.y - SHELL_MARGIN,
        ),
    )
}

fn default_panel_position(
    snap_anchor: Option<ShellSnapAnchor>,
    size: egui::Vec2,
    bounds: egui::Rect,
    fallback: egui::Pos2,
) -> egui::Pos2 {
    let clamped_fallback = clamp_window_position(fallback, size, bounds);
    match snap_anchor {
        Some(anchor) => snapped_panel_position(anchor, clamped_fallback, size, bounds),
        None => clamped_fallback,
    }
}

fn current_panel_position(
    state: &ShellWindowState,
    bounds: egui::Rect,
    default_pos: egui::Pos2,
) -> egui::Pos2 {
    let size = clamp_window_size(state.size, bounds.size(), TOOL_MIN_SIZE.min(INSPECTOR_MIN_SIZE));
    let base_pos = state.position.unwrap_or(default_pos);
    let clamped_pos = clamp_window_position(base_pos, size, bounds);
    match state.snap_anchor {
        Some(anchor) => snapped_panel_position(anchor, clamped_pos, size, bounds),
        None => clamped_pos,
    }
}

fn snapped_panel_position(
    anchor: ShellSnapAnchor,
    current_pos: egui::Pos2,
    size: egui::Vec2,
    bounds: egui::Rect,
) -> egui::Pos2 {
    match anchor {
        ShellSnapAnchor::Left => egui::pos2(bounds.min.x, current_pos.y),
        ShellSnapAnchor::Right => egui::pos2(bounds.max.x - size.x, current_pos.y),
        ShellSnapAnchor::Bottom => egui::pos2(current_pos.x, bounds.max.y - size.y),
    }
}

fn clamp_window_position(
    position: egui::Pos2,
    size: egui::Vec2,
    bounds: egui::Rect,
) -> egui::Pos2 {
    let max_x = (bounds.max.x - size.x).max(bounds.min.x);
    let max_y = (bounds.max.y - size.y).max(bounds.min.y);
    egui::pos2(
        position.x.clamp(bounds.min.x, max_x),
        position.y.clamp(bounds.min.y, max_y),
    )
}

fn clamp_window_size(size: egui::Vec2, max_size: egui::Vec2, min_size: egui::Vec2) -> egui::Vec2 {
    egui::vec2(
        size.x.clamp(min_size.x, max_size.x.max(min_size.x)),
        size.y.clamp(min_size.y, max_size.y.max(min_size.y)),
    )
}

fn detect_snap_anchor(rect: egui::Rect, bounds: egui::Rect) -> Option<ShellSnapAnchor> {
    let candidates = [
        (
            (rect.min.x - bounds.min.x).abs(),
            ShellSnapAnchor::Left,
        ),
        (
            (bounds.max.x - rect.max.x).abs(),
            ShellSnapAnchor::Right,
        ),
        (
            (bounds.max.y - rect.max.y).abs(),
            ShellSnapAnchor::Bottom,
        ),
    ];
    let (distance, anchor) = candidates
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;
    if distance <= WINDOW_SNAP_THRESHOLD {
        Some(anchor)
    } else {
        None
    }
}

fn toggle_dock_tab(dock_state: &mut DockState<Tab>, tab: Tab) {
    if let Some(location) = dock_state.find_tab(&tab) {
        dock_state.remove_tab(location);
    } else {
        dock_state.push_to_focused_leaf(tab);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_window_position_keeps_panel_inside_bounds() {
        let bounds = egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(210.0, 220.0));
        let position = clamp_window_position(egui::pos2(-40.0, 400.0), egui::vec2(80.0, 60.0), bounds);
        assert_eq!(position, egui::pos2(10.0, 160.0));
    }

    #[test]
    fn detect_snap_anchor_matches_left_edge() {
        let bounds = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(300.0, 200.0));
        let rect = egui::Rect::from_min_size(egui::pos2(8.0, 24.0), egui::vec2(120.0, 80.0));
        assert_eq!(detect_snap_anchor(rect, bounds), Some(ShellSnapAnchor::Left));
    }

    #[test]
    fn detect_snap_anchor_matches_right_edge() {
        let bounds = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(300.0, 200.0));
        let rect = egui::Rect::from_min_size(egui::pos2(176.0, 24.0), egui::vec2(120.0, 80.0));
        assert_eq!(detect_snap_anchor(rect, bounds), Some(ShellSnapAnchor::Right));
    }

    #[test]
    fn detect_snap_anchor_matches_bottom_edge() {
        let bounds = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(300.0, 200.0));
        let rect = egui::Rect::from_min_size(egui::pos2(90.0, 128.0), egui::vec2(120.0, 68.0));
        assert_eq!(detect_snap_anchor(rect, bounds), Some(ShellSnapAnchor::Bottom));
    }

    #[test]
    fn detect_snap_anchor_returns_none_when_far_from_edges() {
        let bounds = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(300.0, 200.0));
        let rect = egui::Rect::from_min_size(egui::pos2(90.0, 40.0), egui::vec2(120.0, 80.0));
        assert_eq!(detect_snap_anchor(rect, bounds), None);
    }
}
