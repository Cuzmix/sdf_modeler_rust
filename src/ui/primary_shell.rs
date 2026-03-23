use std::collections::HashSet;

use eframe::egui;
use egui_dock::DockState;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink, WorkspacePreset};
use crate::app::state::{
    InteractionMode, MultiTransformSessionState, PrimaryShellContextTab, PrimaryShellDrawerTab,
    PrimaryShellState, ShellPanelKind,
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
const DRAWER_GAP_FROM_LAUNCHER: f32 = 8.0;
const TOOL_DEFAULT_SIZE: egui::Vec2 = egui::vec2(320.0, 420.0);
const INSPECTOR_DEFAULT_SIZE: egui::Vec2 = egui::vec2(360.0, 480.0);
const DRAWER_DEFAULT_SIZE: egui::Vec2 = egui::vec2(560.0, 260.0);
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

    draw_tool_panel_window(ctx, viewport_rect, &mut shell);
    draw_inspector_panel_window(ctx, viewport_rect, &mut shell);
    draw_drawer_window(ctx, viewport_rect, &launcher_rect, &mut shell);
}

pub fn draw_tool_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_docked_shell_tab_header(ui, ShellPanelKind::Tool, shell.actions);
    draw_tool_panel_contents(ui, shell);
}

pub fn draw_inspector_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    sync_context_tab(shell.shell, shell.scene, *shell.selected);
    draw_docked_shell_tab_header(ui, ShellPanelKind::Inspector, shell.actions);
    draw_inspector_panel_contents(ui, shell);
}

pub fn draw_drawer_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_docked_shell_tab_header(ui, ShellPanelKind::Drawer, shell.actions);
    draw_drawer_tab_controls(ui, shell);
    ui.separator();
    draw_drawer_contents(ui, shell);
}

fn launcher_frame(ctx: &egui::Context) -> egui::Frame {
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
            launcher_frame(ctx).show(ui, |ui| {
                ui.set_width(bar_rect.width());
                ui.horizontal_wrapped(|ui| {
                    drawer_tab_button(
                        ui,
                        shell,
                        PrimaryShellDrawerTab::Items,
                        "Items",
                    );
                    drawer_tab_button(
                        ui,
                        shell,
                        PrimaryShellDrawerTab::History,
                        "History",
                    );
                    drawer_tab_button(
                        ui,
                        shell,
                        PrimaryShellDrawerTab::Reference,
                        "Reference",
                    );
                    drawer_tab_button(
                        ui,
                        shell,
                        PrimaryShellDrawerTab::Advanced,
                        "Advanced",
                    );
                    ui.separator();
                    let tool_active = panel_is_active(shell.shell, ShellPanelKind::Tool);
                    if ui.selectable_label(tool_active, "Tool").clicked() {
                        toggle_tool_panel_from_launcher(shell.shell, shell.actions);
                    }
                    let inspector_active = panel_is_active(shell.shell, ShellPanelKind::Inspector);
                    if ui.selectable_label(inspector_active, "Inspector").clicked() {
                        toggle_inspector_panel_from_launcher(shell.shell, shell.actions);
                    }
                    ui.separator();
                    if ui.small_button("Reset Layout").clicked() {
                        shell.actions.push(Action::ResetPrimaryShellLayout);
                    }
                });
            });
        });

    bar_rect
}

fn draw_tool_panel_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.tool_panel.is_floating() {
        return;
    }
    let window_id = shell_window_id(
        "primary_shell_tool_panel",
        shell.shell.layout_revision,
        shell.shell.tool_panel.floating_revision,
    );
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let default_rect = shell
        .shell
        .tool_panel
        .last_floating_rect
        .unwrap_or_else(|| egui::Rect::from_min_size(default_tool_panel_position(viewport_rect), TOOL_DEFAULT_SIZE));
    let mut open = true;

    show_shell_window(
        ctx,
        window_id,
        "Tool Panel",
        &mut open,
        default_rect.min,
        default_rect.size(),
        TOOL_MIN_SIZE,
        |ui| {
            shell_window_actions(
                ui,
                shell.dock_state.is_some(),
                &mut dock_clicked,
                &mut reset_clicked,
            );
            draw_tool_panel_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell.shell.tool_panel.remember_floating_rect(current_rect);

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Tool,
            rect: current_rect,
        });
    } else if !open {
        shell.shell.tool_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

fn draw_inspector_panel_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.inspector_panel.is_floating() {
        return;
    }
    let window_id = shell_window_id(
        "primary_shell_inspector_panel",
        shell.shell.layout_revision,
        shell.shell.inspector_panel.floating_revision,
    );
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let default_rect = shell.shell.inspector_panel.last_floating_rect.unwrap_or_else(|| {
        egui::Rect::from_min_size(default_inspector_panel_position(viewport_rect), INSPECTOR_DEFAULT_SIZE)
    });
    let mut open = true;

    show_shell_window(
        ctx,
        window_id,
        "Inspector",
        &mut open,
        default_rect.min,
        default_rect.size(),
        INSPECTOR_MIN_SIZE,
        |ui| {
            shell_window_actions(
                ui,
                shell.dock_state.is_some(),
                &mut dock_clicked,
                &mut reset_clicked,
            );
            draw_inspector_panel_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell.shell.inspector_panel.remember_floating_rect(current_rect);

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Inspector,
            rect: current_rect,
        });
    } else if !open {
        shell.shell.inspector_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

fn draw_drawer_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    launcher_rect: &egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.drawer_panel.is_floating() {
        return;
    }
    let window_id = shell_window_id(
        "primary_shell_drawer_panel",
        shell.shell.layout_revision,
        shell.shell.drawer_panel.floating_revision,
    );
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let dockable = shell.dock_state.is_some();
    let default_rect = shell.shell.drawer_panel.last_floating_rect.unwrap_or_else(|| {
        egui::Rect::from_min_size(
            default_drawer_panel_position(viewport_rect, launcher_rect),
            DRAWER_DEFAULT_SIZE,
        )
    });
    let mut open = true;

    show_shell_window(
        ctx,
        window_id,
        drawer_window_title(shell.shell.active_drawer_tab),
        &mut open,
        default_rect.min,
        default_rect.size(),
        DRAWER_MIN_SIZE,
        |ui| {
            shell_window_actions(ui, dockable, &mut dock_clicked, &mut reset_clicked);
            draw_drawer_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell.shell.drawer_panel.remember_floating_rect(current_rect);

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Drawer,
            rect: current_rect,
        });
    } else if !open {
        shell.shell.drawer_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

#[allow(clippy::too_many_arguments)]
fn show_shell_window(
    ctx: &egui::Context,
    id: egui::Id,
    title: &str,
    open: &mut bool,
    default_pos: egui::Pos2,
    default_size: egui::Vec2,
    min_size: egui::Vec2,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    if !*open {
        return;
    }

    egui::Window::new(title)
        .id(id)
        .open(open)
        .collapsible(true)
        .resizable(true)
        .default_pos(default_pos)
        .default_size(default_size)
        .min_size(min_size)
        .show(ctx, |ui| add_contents(ui));
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

fn draw_docked_shell_tab_header(ui: &mut egui::Ui, panel: ShellPanelKind, actions: &mut ActionSink) {
    ui.horizontal(|ui| {
        ui.small("Docked shell panel");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Reset").clicked() {
                actions.push(Action::ResetPrimaryShellLayout);
            }
            if ui.small_button("Undock").clicked() {
                actions.push(Action::UndockShellPanel(panel));
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

fn draw_drawer_tab_controls(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.horizontal_wrapped(|ui| {
        context_tab_button_for_drawer(ui, shell, PrimaryShellDrawerTab::Items, "Items");
        context_tab_button_for_drawer(ui, shell, PrimaryShellDrawerTab::History, "History");
        context_tab_button_for_drawer(ui, shell, PrimaryShellDrawerTab::Reference, "Reference");
        context_tab_button_for_drawer(ui, shell, PrimaryShellDrawerTab::Advanced, "Advanced");
    });
}

fn context_tab_button_for_drawer(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellDrawerTab,
    label: &str,
) {
    if ui
        .selectable_label(shell.shell.active_drawer_tab == tab, label)
        .clicked()
    {
        shell.shell.active_drawer_tab = tab;
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
            shell.actions.push(Action::ToggleDockTab(Tab::NodeGraph));
        }
        if ui.button("Properties").clicked() {
            shell.actions.push(Action::ToggleDockTab(Tab::Properties));
        }
        if ui.button("Render Settings").clicked() {
            shell.actions.push(Action::ToggleDockTab(Tab::RenderSettings));
        }
        if ui.button("Scene Stats").clicked() {
            shell.actions.push(Action::ToggleDockTab(Tab::SceneStats));
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
    let active =
        shell.shell.active_drawer_tab == tab && panel_is_active(shell.shell, ShellPanelKind::Drawer);
    if ui.selectable_label(active, label).clicked() {
        toggle_drawer_from_launcher(shell.shell, tab, shell.actions);
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

fn panel_is_active(shell: &PrimaryShellState, panel: ShellPanelKind) -> bool {
    let panel_state = shell.panel(panel);
    panel_state.is_floating() || panel_state.is_docked()
}

fn toggle_tool_panel_from_launcher(shell: &mut PrimaryShellState, actions: &mut ActionSink) {
    if shell.tool_panel.is_floating() {
        shell.tool_panel.hide();
    } else if shell.tool_panel.is_docked() {
        actions.push(Action::HideShellPanel(ShellPanelKind::Tool));
    } else {
        shell.tool_panel.show_floating(None);
    }
}

fn toggle_inspector_panel_from_launcher(shell: &mut PrimaryShellState, actions: &mut ActionSink) {
    if shell.inspector_panel.is_floating() {
        shell.inspector_panel.hide();
    } else if shell.inspector_panel.is_docked() {
        actions.push(Action::HideShellPanel(ShellPanelKind::Inspector));
    } else {
        shell.inspector_panel.show_floating(None);
    }
}

fn toggle_drawer_from_launcher(
    shell: &mut PrimaryShellState,
    tab: PrimaryShellDrawerTab,
    actions: &mut ActionSink,
) {
    if shell.active_drawer_tab != tab {
        shell.active_drawer_tab = tab;
        if shell.drawer_panel.is_hidden() {
            shell.drawer_panel.show_floating(None);
        }
        return;
    }

    if shell.drawer_panel.is_floating() {
        shell.drawer_panel.hide();
    } else if shell.drawer_panel.is_docked() {
        actions.push(Action::HideShellPanel(ShellPanelKind::Drawer));
    } else {
        shell.drawer_panel.show_floating(None);
    }
}

fn shell_window_id(name: &'static str, layout_revision: u64, floating_revision: u64) -> egui::Id {
    egui::Id::new((name, layout_revision, floating_revision))
}

fn default_tool_panel_position(viewport_rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        viewport_rect.min.x + SHELL_MARGIN,
        viewport_rect.min.y + SHELL_MARGIN,
    )
}

fn default_inspector_panel_position(viewport_rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        (viewport_rect.max.x - INSPECTOR_DEFAULT_SIZE.x - SHELL_MARGIN)
            .max(viewport_rect.min.x + SHELL_MARGIN),
        viewport_rect.min.y + SHELL_MARGIN,
    )
}

fn default_drawer_panel_position(viewport_rect: egui::Rect, launcher_rect: &egui::Rect) -> egui::Pos2 {
    egui::pos2(
        (viewport_rect.center().x - DRAWER_DEFAULT_SIZE.x * 0.5)
            .max(viewport_rect.min.x + SHELL_MARGIN),
        launcher_rect.min.y - DRAWER_DEFAULT_SIZE.y - DRAWER_GAP_FROM_LAUNCHER,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_window_id_changes_with_layout_revision() {
        let initial_id = shell_window_id("primary_shell_tool_panel", 0, 0);
        let reset_id = shell_window_id("primary_shell_tool_panel", 1, 0);
        let undocked_id = shell_window_id("primary_shell_tool_panel", 1, 1);

        assert_ne!(initial_id, reset_id);
        assert_ne!(reset_id, undocked_id);
    }

    #[test]
    fn panel_is_active_detects_floating_and_docked_panels() {
        let mut shell = PrimaryShellState::default();
        assert!(panel_is_active(&shell, ShellPanelKind::Tool));
        assert!(!panel_is_active(&shell, ShellPanelKind::Drawer));

        shell.drawer_panel.dock();
        assert!(panel_is_active(&shell, ShellPanelKind::Drawer));
    }

    #[test]
    fn launcher_toggle_closes_tool_panel_when_floating() {
        let mut shell = PrimaryShellState::default();
        let mut actions = ActionSink::new();

        toggle_tool_panel_from_launcher(&mut shell, &mut actions);

        assert!(shell.tool_panel.is_hidden());
        assert!(actions.is_empty());
    }

    #[test]
    fn launcher_toggle_removes_tool_panel_when_docked() {
        let mut shell = PrimaryShellState::default();
        shell.tool_panel.dock();
        let mut actions = ActionSink::new();

        toggle_tool_panel_from_launcher(&mut shell, &mut actions);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions.first(),
            Some(Action::HideShellPanel(ShellPanelKind::Tool))
        ));
    }

    #[test]
    fn launcher_switches_drawer_tab_without_undocking_existing_drawer() {
        let mut shell = PrimaryShellState::default();
        shell.drawer_panel.dock();
        shell.active_drawer_tab = PrimaryShellDrawerTab::Items;
        let mut actions = ActionSink::new();

        toggle_drawer_from_launcher(&mut shell, PrimaryShellDrawerTab::Advanced, &mut actions);

        assert_eq!(shell.active_drawer_tab, PrimaryShellDrawerTab::Advanced);
        assert!(shell.drawer_panel.is_docked());
        assert!(actions.is_empty());
    }

    #[test]
    fn launcher_toggle_removes_docked_drawer_panel_for_active_tab() {
        let mut shell = PrimaryShellState::default();
        shell.drawer_panel.dock();
        shell.active_drawer_tab = PrimaryShellDrawerTab::Reference;
        let mut actions = ActionSink::new();

        toggle_drawer_from_launcher(&mut shell, PrimaryShellDrawerTab::Reference, &mut actions);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions.first(),
            Some(Action::HideShellPanel(ShellPanelKind::Drawer))
        ));
    }
}
