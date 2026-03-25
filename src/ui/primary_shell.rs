use std::collections::HashSet;

use egui_dock::DockState;

use crate::app::actions::{Action, ActionSink, WorkspacePreset};
use crate::app::reference_images::ReferenceImageStore;
use crate::app::state::{
    ExpertPanelRegistry, InteractionMode, MultiTransformSessionState, PrimaryShellInspectorTab,
    PrimaryShellState, PrimaryShellUtilityTab, SculptUtilityControl, SculptUtilityDragState,
    ShellPanelKind,
};
use crate::app::ui_geometry::FloatingPanelBounds;
use crate::gpu::camera::Camera;
use crate::graph::history::History;
use crate::graph::presented_object::{
    collect_presented_selection, presented_top_level_objects, resolve_presented_object,
    PresentedObjectKind, PresentedObjectRef,
};
use crate::graph::scene::{CsgOp, NodeId, Scene, SdfPrimitive};
use crate::material_preset::MaterialLibrary;
use crate::sculpt::{BrushMode, FalloffMode, SculptBrushProfile, SculptState};
use crate::settings::{SelectionBehaviorSettings, Settings};
use crate::ui::dock::Tab;
use crate::ui::egui_compat::{corner_radius, outside_stroke};
use crate::ui::gizmo::{self, GizmoMode, GizmoSpace};
use crate::ui::{
    brush_settings, chips, chrome, history_panel, presented_object_actions, presented_properties,
    presented_scene_tree, reference_image,
};

const SHELL_MARGIN: f32 = 12.0;
const SCENE_DEFAULT_WIDTH: f32 = 324.0;
const INSPECTOR_DEFAULT_SIZE: egui::Vec2 = egui::vec2(360.0, 500.0);
const UTILITY_DEFAULT_SIZE: egui::Vec2 = egui::vec2(360.0, 280.0);
const BRUSH_ADVANCED_DEFAULT_SIZE: egui::Vec2 = egui::vec2(280.0, 320.0);
const SCENE_MIN_SIZE: egui::Vec2 = egui::vec2(300.0, 420.0);
const INSPECTOR_MIN_SIZE: egui::Vec2 = egui::vec2(300.0, 360.0);
const UTILITY_MIN_SIZE: egui::Vec2 = egui::vec2(280.0, 220.0);
const TOP_STRIP_HEIGHT: f32 = 42.0;
const TOOL_RAIL_HEIGHT: f32 = 46.0;
const TOOL_RAIL_WIDTH: f32 = 720.0;
const UTILITY_STRIP_HEIGHT: f32 = 42.0;
const UTILITY_STRIP_WIDTH: f32 = 520.0;
const SELECTION_POPUP_OFFSET: egui::Vec2 = egui::vec2(16.0, -16.0);
const SCULPT_STRIP_BUTTON_SIZE: egui::Vec2 = egui::vec2(52.0, 52.0);
const SCULPT_STRIP_GAP: f32 = 8.0;
const SCULPT_DRAG_SENSITIVITY: f32 = 0.01;
const FALLOFF_DRAG_STEP: f32 = 26.0;
const SCENE_PANEL_WINDOW_KEY: &str = "primary_shell_scene_panel_v2";
const INSPECTOR_PANEL_WINDOW_KEY: &str = "primary_shell_subject_inspector_v2";

pub struct PrimaryShellContext<'a> {
    pub shell: &'a mut PrimaryShellState,
    pub dock_state: Option<&'a mut DockState<Tab>>,
    pub camera: &'a Camera,
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
    pub expert_panels: &'a ExpertPanelRegistry,
    pub gizmo_mode: &'a GizmoMode,
    pub gizmo_space: &'a GizmoSpace,
    pub selection_behavior: &'a SelectionBehaviorSettings,
    pub reference_images: &'a mut ReferenceImageStore,
    pub measurement_points: &'a mut Vec<glam::Vec3>,
    pub show_distance_readout: &'a mut bool,
    pub settings: &'a mut Settings,
}

pub fn draw(ctx: &egui::Context, viewport_rect: egui::Rect, shell: PrimaryShellContext<'_>) {
    let mut shell = shell;
    sync_inspector_tab(shell.shell);
    clear_finished_sculpt_strip_drag(ctx, shell.shell);

    draw_scene_panel_window(ctx, viewport_rect, &mut shell);
    draw_subject_inspector_window(ctx, viewport_rect, &mut shell);
    draw_utility_panel_window(ctx, viewport_rect, &mut shell);

    if shell.shell.selection_context_strip_visible {
        draw_selection_context_strip(ctx, viewport_rect, &mut shell);
    }
    if shell.shell.tool_rail_visible {
        draw_tool_rail(ctx, viewport_rect, &mut shell);
    }
    if shell.shell.utility_strip_visible {
        draw_utility_strip(ctx, viewport_rect, &mut shell);
    }
    if matches!(shell.shell.interaction_mode, InteractionMode::Sculpt(_))
        && shell.shell.sculpt_utility_strip_visible
    {
        draw_sculpt_utility_strip(ctx, viewport_rect, &mut shell);
        draw_brush_advanced_popup(ctx, viewport_rect, &mut shell);
    } else {
        shell.shell.sculpt_utility_drag = None;
        shell.shell.brush_advanced_open = false;
    }
    if shell.shell.selection_popup_visible {
        draw_selection_popup(ctx, viewport_rect, &mut shell);
    }
}

fn bounds_to_rect(bounds: FloatingPanelBounds) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::pos2(bounds.x, bounds.y),
        egui::vec2(bounds.width, bounds.height),
    )
}

fn rect_to_bounds(rect: egui::Rect) -> FloatingPanelBounds {
    FloatingPanelBounds::from_min_size(rect.min.x, rect.min.y, rect.width(), rect.height())
}

pub fn draw_tool_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_scene_panel_docked_header(ui, shell.actions);
    draw_scene_panel_contents(ui, shell);
}

pub fn draw_inspector_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    sync_inspector_tab(shell.shell);
    draw_subject_inspector_docked_header(ui, shell.actions);
    draw_subject_inspector_contents(ui, shell);
}

pub fn draw_drawer_panel_tab(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_docked_shell_tab_header(
        ui,
        ShellPanelKind::Drawer,
        "Docked utilities",
        shell.actions,
    );
    draw_utility_tab_controls(ui, shell);
    ui.separator();
    draw_utility_panel_contents(ui, shell);
}

fn draw_scene_panel_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.tool_panel.is_floating() {
        return;
    }

    let window_id = shell_window_id(
        SCENE_PANEL_WINDOW_KEY,
        shell.shell.layout_revision,
        shell.shell.tool_panel.floating_revision,
    );
    let default_rect = shell
        .shell
        .tool_panel
        .last_floating_rect
        .map(bounds_to_rect)
        .unwrap_or_else(|| default_scene_panel_rect(viewport_rect));
    let mut open = true;
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let mut hide_clicked = false;

    show_chrome_window(
        ctx,
        window_id,
        "Scene Panel",
        &mut open,
        default_rect.min,
        default_rect.size(),
        SCENE_MIN_SIZE,
        |ui| {
            draw_scene_panel_window_header(
                ui,
                presented_top_level_objects(shell.scene).len(),
                shell.dock_state.is_some(),
                &mut dock_clicked,
                &mut reset_clicked,
                &mut hide_clicked,
            );
            draw_scene_panel_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell
        .shell
        .tool_panel
        .remember_floating_rect(rect_to_bounds(current_rect));

    if hide_clicked {
        open = false;
    }

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Tool,
            rect: rect_to_bounds(current_rect),
        });
    } else if !open {
        shell.shell.tool_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

fn draw_subject_inspector_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.inspector_panel.is_floating() {
        return;
    }

    let window_id = shell_window_id(
        INSPECTOR_PANEL_WINDOW_KEY,
        shell.shell.layout_revision,
        shell.shell.inspector_panel.floating_revision,
    );
    let default_rect = shell
        .shell
        .inspector_panel
        .last_floating_rect
        .map(bounds_to_rect)
        .unwrap_or_else(|| default_inspector_panel_rect(viewport_rect));
    let mut open = true;
    let mut dock_clicked = false;
    let mut reset_clicked = false;
    let mut hide_clicked = false;

    show_chrome_window(
        ctx,
        window_id,
        "Subject Inspector",
        &mut open,
        default_rect.min,
        default_rect.size(),
        INSPECTOR_MIN_SIZE,
        |ui| {
            draw_subject_inspector_window_header(
                ui,
                shell.dock_state.is_some(),
                &mut dock_clicked,
                &mut reset_clicked,
                &mut hide_clicked,
            );
            draw_subject_inspector_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell
        .shell
        .inspector_panel
        .remember_floating_rect(rect_to_bounds(current_rect));

    if hide_clicked {
        open = false;
    }

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Inspector,
            rect: rect_to_bounds(current_rect),
        });
    } else if !open {
        shell.shell.inspector_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

fn draw_utility_panel_window(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.drawer_panel.is_floating() {
        return;
    }

    let window_id = shell_window_id(
        "primary_shell_utility_panel",
        shell.shell.layout_revision,
        shell.shell.drawer_panel.floating_revision,
    );
    let default_rect = shell
        .shell
        .drawer_panel
        .last_floating_rect
        .map(bounds_to_rect)
        .unwrap_or_else(|| {
            egui::Rect::from_min_size(
                default_utility_panel_position(viewport_rect),
                UTILITY_DEFAULT_SIZE,
            )
        });
    let mut open = true;
    let mut dock_clicked = false;
    let mut reset_clicked = false;

    show_shell_window(
        ctx,
        window_id,
        utility_panel_title(shell.shell.active_utility_tab),
        &mut open,
        default_rect.min,
        default_rect.size(),
        UTILITY_MIN_SIZE,
        |ui| {
            shell_window_actions(
                ui,
                shell.dock_state.is_some(),
                "Secondary utility panel",
                &mut dock_clicked,
                &mut reset_clicked,
            );
            draw_utility_tab_controls(ui, shell);
            ui.separator();
            draw_utility_panel_contents(ui, shell);
        },
    );

    let current_rect = ctx
        .memory(|memory| memory.area_rect(window_id))
        .unwrap_or(default_rect);
    shell
        .shell
        .drawer_panel
        .remember_floating_rect(rect_to_bounds(current_rect));

    if dock_clicked && shell.dock_state.is_some() {
        shell.actions.push(Action::DockShellPanel {
            panel: ShellPanelKind::Drawer,
            rect: rect_to_bounds(current_rect),
        });
    } else if !open {
        shell.shell.drawer_panel.hide();
    }
    if reset_clicked {
        shell.actions.push(Action::ResetPrimaryShellLayout);
    }
}

fn draw_scene_panel_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    presented_scene_tree::draw(
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

fn draw_subject_inspector_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    draw_subject_summary_band(ui, shell);
    ui.add_space(8.0);

    ui.horizontal(|ui| {
        inspector_tab_button(
            ui,
            shell,
            PrimaryShellInspectorTab::Properties,
            "Properties",
        );
        inspector_tab_button(ui, shell, PrimaryShellInspectorTab::Display, "Display");
    });
    ui.add_space(8.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| match shell.shell.active_inspector_tab {
            PrimaryShellInspectorTab::Properties => draw_properties_panel(ui, shell),
            PrimaryShellInspectorTab::Display => draw_display_panel(ui, shell),
        });
}

fn draw_subject_summary_band(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    let presented_selection =
        collect_presented_selection(shell.scene, *shell.selected, shell.selected_set);

    chrome::card_frame().show(ui, |ui| {
        match (
            presented_selection.primary,
            presented_selection.ordered.len(),
        ) {
            (Some(object), 1) => {
                let host_node = shell.scene.nodes.get(&object.host_id);
                let object_name = host_node.map(|node| node.name.as_str()).unwrap_or("Object");
                let is_locked = host_node.is_some_and(|node| node.locked);

                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(object_name).strong());
                    chips::draw_chip(
                        ui,
                        object_kind_chip_label(object),
                        egui::Color32::from_rgb(42, 46, 60),
                        egui::Color32::from_rgb(220, 226, 240),
                    );
                    if object.attached_sculpt_id.is_some() {
                        chips::draw_chip(
                            ui,
                            "SCULPT",
                            egui::Color32::from_rgb(98, 67, 24),
                            egui::Color32::from_rgb(255, 220, 140),
                        );
                    }
                    if matches!(object.kind, PresentedObjectKind::Voxel) {
                        chips::draw_chip(
                            ui,
                            "VOXEL",
                            egui::Color32::from_rgb(28, 72, 92),
                            egui::Color32::from_rgb(180, 232, 255),
                        );
                    }
                    if matches!(object.kind, PresentedObjectKind::Light) {
                        chips::draw_chip(
                            ui,
                            "LIGHT",
                            egui::Color32::from_rgb(64, 58, 26),
                            egui::Color32::from_rgb(252, 230, 144),
                        );
                    }
                    if is_locked {
                        chips::draw_chip(
                            ui,
                            "LOCKED",
                            egui::Color32::from_rgb(80, 56, 22),
                            egui::Color32::from_rgb(255, 214, 148),
                        );
                    }
                });

                ui.add_space(6.0);

                if let Some((done, total)) = shell.bake_progress {
                    let progress = done as f32 / total.max(1) as f32;
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .text(format!("Baking... {:.0}%", progress * 100.0)),
                    );
                } else {
                    ui.horizontal_wrapped(|ui| {
                        if chrome::action_button(ui, "Duplicate", false).clicked() {
                            shell
                                .actions
                                .push(Action::DuplicatePresentedObject(object.host_id));
                        }
                        if matches!(object.kind, PresentedObjectKind::Voxel) {
                            if chrome::action_button(ui, "Enter Sculpt", false).clicked() {
                                shell.actions.push(Action::EnterSculptMode);
                            }
                        } else if let Some(sculpt_id) = object.attached_sculpt_id {
                            if chrome::action_button(ui, "Enter Sculpt", false).clicked() {
                                shell.actions.push(Action::EnterSculptMode);
                            }
                            if chrome::action_button(ui, "Convert to Voxel", false).clicked() {
                                presented_object_actions::push_convert_to_voxel_action(
                                    shell.scene,
                                    object,
                                    sculpt_id,
                                    shell.actions,
                                );
                            }
                            if chrome::action_button(ui, "Remove Sculpt Layer", false).clicked() {
                                shell.actions.push(Action::RemoveAttachedSculpt {
                                    host: object.host_id,
                                });
                            }
                        } else if object.supports_add_sculpt()
                            && chrome::action_button(ui, "Add Sculpt Layer", false).clicked()
                        {
                            presented_object_actions::push_add_sculpt_layer_action(
                                shell.scene,
                                object,
                                shell.actions,
                            );
                        }

                        if chrome::action_button(ui, "Delete", false)
                            .on_hover_text(
                                "Remove this object and its hidden internal wrapper nodes",
                            )
                            .clicked()
                        {
                            shell
                                .actions
                                .push(Action::DeletePresentedObject(object.object_root_id));
                        }
                    });
                }
            }
            (_, count) if count > 1 => {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(format!("{count} Objects Selected")).strong());
                    chips::draw_chip(
                        ui,
                        "MULTI",
                        egui::Color32::from_rgb(52, 58, 76),
                        egui::Color32::from_rgb(214, 224, 248),
                    );
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if chrome::action_button(ui, "Delete", false).clicked() {
                        for object in &presented_selection.ordered {
                            shell
                                .actions
                                .push(Action::DeletePresentedObject(object.object_root_id));
                        }
                    }
                });
            }
            _ => {
                ui.label(egui::RichText::new("No selection").strong());
                ui.small("Select an object in the viewport or scene panel.");
            }
        }
    });
}

fn draw_properties_panel(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    presented_properties::draw(
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

fn draw_display_panel(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    let before = shell.settings.render.clone();

    show_inspector_card(
        ui,
        "inspector_display_viewport",
        "Viewport",
        true,
        None,
        |ui| {
            ui.checkbox(&mut shell.settings.render.show_grid, "Grid");
            ui.checkbox(&mut shell.settings.render.show_node_labels, "Node Labels");
            ui.checkbox(&mut shell.settings.render.show_bounding_box, "Bounding Box");
            ui.checkbox(&mut shell.settings.render.show_light_gizmos, "Light Gizmos");

            let mut distance_readout = *shell.show_distance_readout;
            if ui
                .checkbox(&mut distance_readout, "Distance Readout")
                .changed()
            {
                shell.actions.push(Action::ToggleDistanceReadout);
            }
        },
    );

    show_inspector_card(
        ui,
        "inspector_display_planes",
        "Construction Planes",
        false,
        None,
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Axis");
                ui.selectable_value(&mut shell.settings.render.cross_section_axis, 0, "X");
                ui.selectable_value(&mut shell.settings.render.cross_section_axis, 1, "Y");
                ui.selectable_value(&mut shell.settings.render.cross_section_axis, 2, "Z");
            });
            ui.add(
                egui::Slider::new(
                    &mut shell.settings.render.cross_section_position,
                    -5.0..=5.0,
                )
                .text("Plane Offset"),
            );
        },
    );

    show_inspector_card(
        ui,
        "inspector_display_preview",
        "Render Preview",
        false,
        None,
        |ui| {
            ui.checkbox(&mut shell.settings.render.shadows_enabled, "Shadows");
            ui.checkbox(&mut shell.settings.render.ao_enabled, "Ambient Occlusion");
        },
    );

    if shell.shell.interaction_mode == InteractionMode::Measure
        || !shell.measurement_points.is_empty()
    {
        show_inspector_card(
            ui,
            "inspector_display_measurement",
            "Measurement",
            true,
            Some("Tap two points in the viewport to measure distance."),
            |ui| draw_measurement_panel(ui, shell),
        );
    }

    show_inspector_card(
        ui,
        "inspector_display_status",
        "Status",
        false,
        None,
        |ui| {
            ui.small(format!(
                "Interaction: {}",
                interaction_mode_label(shell.shell.interaction_mode)
            ));
            ui.small(format!("Transform: {}", shell.gizmo_mode.label()));
            ui.small(format!("Space: {}", shell.gizmo_space.label()));
        },
    );

    if shell.settings.render != before {
        shell.actions.push(Action::SettingsChanged);
    }
}

fn draw_measurement_panel(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
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
    ui.add_space(6.0);

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

fn draw_utility_tab_controls(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.horizontal_wrapped(|ui| {
        utility_tab_button(ui, shell, PrimaryShellUtilityTab::History, "History");
        utility_tab_button(ui, shell, PrimaryShellUtilityTab::Reference, "Reference");
        utility_tab_button(ui, shell, PrimaryShellUtilityTab::Advanced, "Advanced");
    });
}

fn draw_utility_panel_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    match shell.shell.active_utility_tab {
        PrimaryShellUtilityTab::History => history_panel::draw(ui, shell.history, shell.actions),
        PrimaryShellUtilityTab::Reference => {
            reference_image::draw_controls(ui, shell.reference_images, shell.actions)
        }
        PrimaryShellUtilityTab::Advanced => draw_advanced_utilities(ui, shell),
    }
}

fn draw_advanced_utilities(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    ui.strong("Advanced");
    ui.small("Expert panels and dense desktop workflows live here.");
    ui.separator();

    ui.label(egui::RichText::new("Expert Panels").small().strong());
    ui.horizontal_wrapped(|ui| {
        for panel in crate::app::state::ExpertPanelKind::ALL {
            let is_open = shell.expert_panels.is_open(panel);
            if ui.selectable_label(is_open, panel.label()).clicked() {
                shell.actions.push(Action::ToggleExpertPanel(panel));
            }
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Workspaces").small().strong());
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
}

fn draw_selection_context_strip(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let strip_size = egui::vec2(
        480.0_f32.min(viewport_rect.width() - 160.0).max(260.0),
        TOP_STRIP_HEIGHT,
    );
    let strip_pos = egui::pos2(
        viewport_rect.center().x - strip_size.x * 0.5,
        viewport_rect.min.y + SHELL_MARGIN,
    );

    egui::Area::new(egui::Id::new("primary_shell_selection_context_strip"))
        .order(egui::Order::Foreground)
        .fixed_pos(strip_pos)
        .show(ctx, |ui| {
            chrome_frame(ctx).show(ui, |ui| {
                ui.set_min_width(strip_size.x);
                draw_selection_context_contents(ui, shell);
            });
        });
}

fn draw_selection_context_contents(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    if let Some(object) = selected_object(shell.scene, *shell.selected) {
        let object_name = shell
            .scene
            .nodes
            .get(&object.host_id)
            .map(|node| node.name.as_str())
            .unwrap_or("Selected Object");

        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(object_name).strong());
            chips::draw_chip(
                ui,
                object_kind_chip_label(object),
                egui::Color32::from_rgb(42, 46, 60),
                egui::Color32::from_rgb(220, 226, 240),
            );
            if object.attached_sculpt_id.is_some() {
                chips::draw_chip(
                    ui,
                    "SCULPT",
                    egui::Color32::from_rgb(98, 67, 24),
                    egui::Color32::from_rgb(255, 220, 140),
                );
            }
            if matches!(object.kind, PresentedObjectKind::Voxel) {
                chips::draw_chip(
                    ui,
                    "VOXEL",
                    egui::Color32::from_rgb(28, 72, 92),
                    egui::Color32::from_rgb(180, 232, 255),
                );
            }
            chips::draw_chip(
                ui,
                interaction_mode_label(shell.shell.interaction_mode),
                egui::Color32::from_rgb(38, 56, 88),
                egui::Color32::from_rgb(196, 220, 255),
            );
            chips::draw_chip(
                ui,
                shell.gizmo_mode.label(),
                egui::Color32::from_rgb(46, 52, 60),
                egui::Color32::from_rgb(220, 220, 220),
            );
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("No object selected").strong());
            ui.small("Select an object or create one from the scene panel.");
            chips::draw_chip(
                ui,
                interaction_mode_label(shell.shell.interaction_mode),
                egui::Color32::from_rgb(38, 56, 88),
                egui::Color32::from_rgb(196, 220, 255),
            );
        });
    }
}

fn draw_tool_rail(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let rail_size = egui::vec2(
        TOOL_RAIL_WIDTH
            .min(viewport_rect.width() - 120.0)
            .max(540.0),
        TOOL_RAIL_HEIGHT,
    );
    let rail_pos = egui::pos2(
        viewport_rect.center().x - rail_size.x * 0.5,
        viewport_rect.max.y - rail_size.y - SHELL_MARGIN,
    );

    egui::Area::new(egui::Id::new("primary_shell_tool_rail"))
        .order(egui::Order::Foreground)
        .fixed_pos(rail_pos)
        .show(ctx, |ui| {
            chrome_frame(ctx).show(ui, |ui| {
                ui.set_min_width(rail_size.x);
                ui.horizontal_wrapped(|ui| {
                    interaction_mode_button(ui, shell, InteractionMode::Select, "Select");
                    interaction_mode_button(ui, shell, InteractionMode::Measure, "Measure");
                    ui.separator();
                    gizmo_mode_button(ui, shell, GizmoMode::Translate);
                    gizmo_mode_button(ui, shell, GizmoMode::Rotate);
                    gizmo_mode_button(ui, shell, GizmoMode::Scale);
                    ui.separator();
                    for brush in BrushMode::ALL {
                        interaction_mode_button(
                            ui,
                            shell,
                            InteractionMode::Sculpt(brush),
                            brush.label(),
                        );
                    }
                    ui.separator();
                    draw_distance_toggle(ui, shell);
                    if ui.small_button("Command").clicked() {
                        shell.actions.push(Action::ToggleCommandPalette);
                    }
                });
            });
        });
}

fn draw_utility_strip(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let strip_size = egui::vec2(
        UTILITY_STRIP_WIDTH.min(viewport_rect.width() - 120.0),
        UTILITY_STRIP_HEIGHT,
    );
    let strip_pos = egui::pos2(
        viewport_rect.min.x + SHELL_MARGIN,
        viewport_rect.max.y - strip_size.y - SHELL_MARGIN,
    );

    egui::Area::new(egui::Id::new("primary_shell_utility_strip"))
        .order(egui::Order::Foreground)
        .fixed_pos(strip_pos)
        .show(ctx, |ui| {
            chrome_frame(ctx).show(ui, |ui| {
                ui.set_min_width(strip_size.x);
                ui.horizontal_wrapped(|ui| {
                    panel_toggle_button(ui, shell, ShellPanelKind::Tool, "Scene");
                    panel_toggle_button(ui, shell, ShellPanelKind::Inspector, "Inspector");
                    ui.separator();
                    utility_strip_button(ui, shell, PrimaryShellUtilityTab::History, "History");
                    utility_strip_button(ui, shell, PrimaryShellUtilityTab::Reference, "Reference");
                    utility_strip_button(ui, shell, PrimaryShellUtilityTab::Advanced, "Advanced");
                    ui.separator();
                    ui.menu_button("Workspace", |ui| {
                        if ui.button("Modeling").clicked() {
                            shell
                                .actions
                                .push(Action::SetWorkspace(WorkspacePreset::Modeling));
                            ui.close();
                        }
                        if ui.button("Sculpting").clicked() {
                            shell
                                .actions
                                .push(Action::SetWorkspace(WorkspacePreset::Sculpting));
                            ui.close();
                        }
                        if ui.button("Rendering").clicked() {
                            shell
                                .actions
                                .push(Action::SetWorkspace(WorkspacePreset::Rendering));
                            ui.close();
                        }
                    });
                    if ui.small_button("Reset").clicked() {
                        shell.actions.push(Action::ResetPrimaryShellLayout);
                    }
                });
            });
        });
}

fn draw_selection_popup(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let Some(object) = selected_object(shell.scene, *shell.selected) else {
        return;
    };
    if matches!(object.kind, PresentedObjectKind::Light) {
        return;
    }
    let Some(anchor) = selection_popup_anchor(shell, viewport_rect) else {
        return;
    };

    egui::Area::new(egui::Id::new("primary_shell_selection_popup"))
        .order(egui::Order::Foreground)
        .fixed_pos(anchor + SELECTION_POPUP_OFFSET)
        .show(ctx, |ui| {
            chrome_frame(ctx).show(ui, |ui| {
                ui.vertical(|ui| {
                    if ui.button("Duplicate").clicked() {
                        shell
                            .actions
                            .push(Action::DuplicatePresentedObject(object.host_id));
                    }
                    ui.separator();
                    draw_boolean_popup_action(ui, shell, "Union", CsgOp::Union);
                    draw_boolean_popup_action(ui, shell, "Subtract", CsgOp::Subtract);
                    draw_boolean_popup_action(ui, shell, "Intersect", CsgOp::Intersect);
                    ui.separator();
                    if object.supports_add_sculpt() {
                        if ui.button("Add Sculpt Layer").clicked() {
                            presented_object_actions::push_add_sculpt_layer_action(
                                shell.scene,
                                object,
                                shell.actions,
                            );
                        }
                    } else if let Some(sculpt_id) = object.attached_sculpt_id {
                        if ui.button("Enter Sculpt").clicked() {
                            shell.actions.push(Action::EnterSculptMode);
                        }
                        if ui.button("Convert to Voxel Object").clicked() {
                            presented_object_actions::push_convert_to_voxel_action(
                                shell.scene,
                                object,
                                sculpt_id,
                                shell.actions,
                            );
                        }
                    }
                });
            });
        });
}

fn draw_boolean_popup_action(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    label: &str,
    op: CsgOp,
) {
    ui.menu_button(label, |ui| {
        for primitive in SdfPrimitive::ALL {
            if ui.button(primitive.base_name()).clicked() {
                shell.actions.push(Action::ShellCreateBooleanPrimitive {
                    op: op.clone(),
                    primitive: primitive.clone(),
                });
                ui.close();
            }
        }
    });
}

fn draw_sculpt_utility_strip(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    let strip_pos = egui::pos2(
        sculpt_strip_x(shell, viewport_rect),
        viewport_rect.center().y - 120.0,
    );

    egui::Area::new(egui::Id::new("primary_shell_sculpt_utility_strip"))
        .order(egui::Order::Foreground)
        .fixed_pos(strip_pos)
        .show(ctx, |ui| {
            chrome_frame(ctx).show(ui, |ui| {
                ui.vertical(|ui| {
                    draw_sculpt_drag_control(
                        ui,
                        shell,
                        SculptUtilityControl::Radius,
                        "R",
                        radius_value_label(shell.sculpt_state),
                    );
                    ui.add_space(SCULPT_STRIP_GAP);
                    draw_sculpt_drag_control(
                        ui,
                        shell,
                        SculptUtilityControl::Strength,
                        "S",
                        strength_value_label(shell.sculpt_state),
                    );
                    ui.add_space(SCULPT_STRIP_GAP);
                    draw_sculpt_drag_control(
                        ui,
                        shell,
                        SculptUtilityControl::Falloff,
                        "F",
                        falloff_value_label(shell.sculpt_state),
                    );
                    ui.add_space(SCULPT_STRIP_GAP);
                    let advanced_label = if shell.shell.brush_advanced_open {
                        "Hide"
                    } else {
                        "More"
                    };
                    if ui.button(advanced_label).clicked() {
                        shell.shell.brush_advanced_open = !shell.shell.brush_advanced_open;
                    }
                });
            });
        });
}

fn draw_brush_advanced_popup(
    ctx: &egui::Context,
    viewport_rect: egui::Rect,
    shell: &mut PrimaryShellContext<'_>,
) {
    if !shell.shell.brush_advanced_open {
        return;
    }

    let mut open = true;
    egui::Window::new("Brush Settings")
        .id(shell_window_id(
            "primary_shell_brush_advanced_popup",
            shell.shell.layout_revision,
            0,
        ))
        .open(&mut open)
        .collapsible(true)
        .resizable(true)
        .default_pos(egui::pos2(
            sculpt_strip_x(shell, viewport_rect) + SCULPT_STRIP_BUTTON_SIZE.x + 16.0,
            viewport_rect.center().y - BRUSH_ADVANCED_DEFAULT_SIZE.y * 0.5,
        ))
        .default_size(BRUSH_ADVANCED_DEFAULT_SIZE)
        .min_size(egui::vec2(220.0, 240.0))
        .show(ctx, |ui| {
            brush_settings::draw_brush_advanced_controls(ui, shell.sculpt_state);
        });

    if !open {
        shell.shell.brush_advanced_open = false;
    }
}

fn draw_sculpt_drag_control(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    control: SculptUtilityControl,
    short_label: &str,
    value_label: String,
) {
    let (rect, response) =
        ui.allocate_exact_size(SCULPT_STRIP_BUTTON_SIZE, egui::Sense::click_and_drag());
    let fill = if response.dragged()
        || shell
            .shell
            .sculpt_utility_drag
            .as_ref()
            .is_some_and(|drag| drag.control == control)
    {
        egui::Color32::from_rgb(56, 84, 122)
    } else if response.hovered() {
        egui::Color32::from_rgb(48, 52, 62)
    } else {
        egui::Color32::from_rgb(34, 36, 42)
    };
    ui.painter().rect(
        rect,
        corner_radius(8.0),
        fill,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(86, 90, 104)),
        outside_stroke(),
    );
    ui.painter().text(
        rect.center_top() + egui::vec2(0.0, 10.0),
        egui::Align2::CENTER_TOP,
        short_label,
        egui::TextStyle::Button.resolve(ui.style()),
        egui::Color32::WHITE,
    );
    ui.painter().text(
        rect.center_bottom() - egui::vec2(0.0, 10.0),
        egui::Align2::CENTER_BOTTOM,
        value_label,
        egui::TextStyle::Small.resolve(ui.style()),
        egui::Color32::from_rgb(188, 196, 214),
    );

    if response.drag_started() {
        let initial_value = sculpt_control_initial_value(shell.sculpt_state, control);
        let pointer_pos = response.interact_pointer_pos().unwrap_or(rect.center());
        shell.shell.sculpt_utility_drag = Some(SculptUtilityDragState {
            control,
            anchor_pos: [pointer_pos.x, pointer_pos.y],
            initial_value,
        });
    }

    if response.clicked() && matches!(control, SculptUtilityControl::Falloff) {
        cycle_falloff(shell.sculpt_state);
    }

    if response.dragged() {
        if let (Some(drag), Some(pointer_pos)) = (
            shell.shell.sculpt_utility_drag.clone(),
            response.interact_pointer_pos(),
        ) {
            if drag.control == control {
                apply_sculpt_drag(shell.sculpt_state, drag, pointer_pos);
            }
        }
    }
}

fn clear_finished_sculpt_strip_drag(ctx: &egui::Context, shell: &mut PrimaryShellState) {
    if !ctx.input(|input| input.pointer.primary_down()) {
        shell.sculpt_utility_drag = None;
    }
}

fn apply_sculpt_drag(
    sculpt_state: &mut SculptState,
    drag: SculptUtilityDragState,
    pointer_pos: egui::Pos2,
) {
    let delta_x = pointer_pos.x - drag.anchor_pos[0];
    let brush = sculpt_state.selected_brush();
    let profile = sculpt_state.selected_profile_mut();

    match drag.control {
        SculptUtilityControl::Radius => {
            profile.radius =
                (drag.initial_value + delta_x * SCULPT_DRAG_SENSITIVITY).clamp(0.05, 2.0);
        }
        SculptUtilityControl::Strength => {
            let (min_strength, max_strength) = SculptBrushProfile::strength_limits(brush);
            profile.strength = (drag.initial_value + delta_x * SCULPT_DRAG_SENSITIVITY)
                .clamp(min_strength, max_strength);
        }
        SculptUtilityControl::Falloff => {
            let initial_index = drag.initial_value.round() as i32;
            let next_index =
                (initial_index + (delta_x / FALLOFF_DRAG_STEP).round() as i32).clamp(0, 3);
            profile.falloff_mode = match next_index {
                0 => FalloffMode::Smooth,
                1 => FalloffMode::Linear,
                2 => FalloffMode::Sharp,
                _ => FalloffMode::Flat,
            };
        }
    }
}

fn sculpt_control_initial_value(sculpt_state: &SculptState, control: SculptUtilityControl) -> f32 {
    let profile = sculpt_state.selected_profile();
    match control {
        SculptUtilityControl::Radius => profile.radius,
        SculptUtilityControl::Strength => profile.strength,
        SculptUtilityControl::Falloff => match profile.falloff_mode {
            FalloffMode::Smooth => 0.0,
            FalloffMode::Linear => 1.0,
            FalloffMode::Sharp => 2.0,
            FalloffMode::Flat => 3.0,
        },
    }
}

fn cycle_falloff(sculpt_state: &mut SculptState) {
    let profile = sculpt_state.selected_profile_mut();
    profile.falloff_mode = match profile.falloff_mode {
        FalloffMode::Smooth => FalloffMode::Linear,
        FalloffMode::Linear => FalloffMode::Sharp,
        FalloffMode::Sharp => FalloffMode::Flat,
        FalloffMode::Flat => FalloffMode::Smooth,
    };
}

fn radius_value_label(sculpt_state: &SculptState) -> String {
    format!("{:.2}", sculpt_state.selected_profile().radius)
}

fn strength_value_label(sculpt_state: &SculptState) -> String {
    format!("{:.2}", sculpt_state.selected_profile().strength)
}

fn falloff_value_label(sculpt_state: &SculptState) -> String {
    match sculpt_state.selected_profile().falloff_mode {
        FalloffMode::Smooth => "Smooth".into(),
        FalloffMode::Linear => "Linear".into(),
        FalloffMode::Sharp => "Sharp".into(),
        FalloffMode::Flat => "Flat".into(),
    }
}

fn draw_distance_toggle(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>) {
    if ui
        .selectable_label(*shell.show_distance_readout, "Distance")
        .clicked()
    {
        shell.actions.push(Action::ToggleDistanceReadout);
    }
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

fn gizmo_mode_button(ui: &mut egui::Ui, shell: &mut PrimaryShellContext<'_>, mode: GizmoMode) {
    let active = *shell.gizmo_mode == mode;
    if ui.selectable_label(active, mode.label()).clicked() && !active {
        shell
            .actions
            .push(Action::SetInteractionMode(InteractionMode::Select));
        shell.actions.push(Action::SetGizmoMode(mode));
    }
}

fn panel_toggle_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    panel: ShellPanelKind,
    label: &str,
) {
    let active = panel_is_active(shell.shell, panel);
    if ui.selectable_label(active, label).clicked() {
        match panel {
            ShellPanelKind::Tool => toggle_scene_panel_from_strip(shell.shell, shell.actions),
            ShellPanelKind::Inspector => {
                toggle_inspector_panel_from_strip(shell.shell, shell.actions)
            }
            ShellPanelKind::Drawer => toggle_utility_panel_from_strip(
                shell.shell,
                shell.shell.active_utility_tab,
                shell.actions,
            ),
        }
    }
}

fn utility_strip_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellUtilityTab,
    label: &str,
) {
    let active = shell.shell.active_utility_tab == tab
        && panel_is_active(shell.shell, ShellPanelKind::Drawer);
    if ui.selectable_label(active, label).clicked() {
        toggle_utility_panel_from_strip(shell.shell, tab, shell.actions);
    }
}

fn inspector_tab_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellInspectorTab,
    label: &str,
) {
    if chrome::tab_pill(ui, label, shell.shell.active_inspector_tab == tab).clicked() {
        shell.shell.active_inspector_tab = tab;
    }
}

fn utility_tab_button(
    ui: &mut egui::Ui,
    shell: &mut PrimaryShellContext<'_>,
    tab: PrimaryShellUtilityTab,
    label: &str,
) {
    if ui
        .selectable_label(shell.shell.active_utility_tab == tab, label)
        .clicked()
    {
        shell.shell.active_utility_tab = tab;
    }
}

fn toggle_scene_panel_from_strip(shell: &mut PrimaryShellState, actions: &mut ActionSink) {
    if shell.tool_panel.is_floating() {
        shell.tool_panel.hide();
    } else if shell.tool_panel.is_docked() {
        actions.push(Action::HideShellPanel(ShellPanelKind::Tool));
    } else {
        shell.tool_panel.show_floating(None);
    }
}

fn toggle_inspector_panel_from_strip(shell: &mut PrimaryShellState, actions: &mut ActionSink) {
    if shell.inspector_panel.is_floating() {
        shell.inspector_panel.hide();
    } else if shell.inspector_panel.is_docked() {
        actions.push(Action::HideShellPanel(ShellPanelKind::Inspector));
    } else {
        shell.inspector_panel.show_floating(None);
    }
}

fn toggle_utility_panel_from_strip(
    shell: &mut PrimaryShellState,
    tab: PrimaryShellUtilityTab,
    actions: &mut ActionSink,
) {
    if shell.active_utility_tab != tab {
        shell.active_utility_tab = tab;
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

fn selected_object(scene: &Scene, selected: Option<NodeId>) -> Option<PresentedObjectRef> {
    selected.and_then(|selected_id| resolve_presented_object(scene, selected_id))
}

fn object_kind_chip_label(object: PresentedObjectRef) -> &'static str {
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

fn selection_popup_anchor(
    shell: &PrimaryShellContext<'_>,
    viewport_rect: egui::Rect,
) -> Option<egui::Pos2> {
    let selection = gizmo::collect_gizmo_selection(
        shell.scene,
        *shell.selected,
        shell.selected_set,
        shell.selection_behavior,
    )?;
    let aspect = viewport_rect.width() / viewport_rect.height().max(1.0);
    let view_proj = shell.camera.projection_matrix(aspect) * shell.camera.view_matrix();
    gizmo::world_to_screen(selection.center_world(), &view_proj, viewport_rect)
}

fn sculpt_strip_x(shell: &PrimaryShellContext<'_>, viewport_rect: egui::Rect) -> f32 {
    if shell.shell.tool_panel.is_floating() {
        shell
            .shell
            .tool_panel
            .last_floating_rect
            .map(|rect| rect.right() + 8.0)
            .unwrap_or(viewport_rect.min.x + SHELL_MARGIN)
    } else {
        viewport_rect.min.x + SHELL_MARGIN
    }
}

fn panel_is_active(shell: &PrimaryShellState, panel: ShellPanelKind) -> bool {
    let panel_state = shell.panel(panel);
    panel_state.is_floating() || panel_state.is_docked()
}

fn sync_inspector_tab(shell: &mut PrimaryShellState) {
    if shell.interaction_mode == InteractionMode::Measure {
        shell.active_inspector_tab = PrimaryShellInspectorTab::Display;
    }
}

fn utility_panel_title(tab: PrimaryShellUtilityTab) -> &'static str {
    match tab {
        PrimaryShellUtilityTab::History => "History",
        PrimaryShellUtilityTab::Reference => "Reference",
        PrimaryShellUtilityTab::Advanced => "Advanced",
    }
}

fn shell_window_id(name: &'static str, layout_revision: u64, floating_revision: u64) -> egui::Id {
    egui::Id::new((name, layout_revision, floating_revision))
}

fn default_scene_panel_rect(viewport_rect: egui::Rect) -> egui::Rect {
    let top = viewport_rect.min.y + SHELL_MARGIN + TOP_STRIP_HEIGHT + 6.0;
    let bottom =
        viewport_rect.max.y - TOOL_RAIL_HEIGHT - UTILITY_STRIP_HEIGHT - (SHELL_MARGIN * 2.0) - 12.0;
    let height = (bottom - top).max(SCENE_MIN_SIZE.y);
    egui::Rect::from_min_size(
        egui::pos2(viewport_rect.min.x + SHELL_MARGIN, top),
        egui::vec2(SCENE_DEFAULT_WIDTH, height),
    )
}

fn default_inspector_panel_rect(viewport_rect: egui::Rect) -> egui::Rect {
    let top = viewport_rect.min.y + SHELL_MARGIN + TOP_STRIP_HEIGHT + 6.0;
    let bottom = viewport_rect.max.y - TOOL_RAIL_HEIGHT - SHELL_MARGIN - 12.0;
    let height = (bottom - top).max(INSPECTOR_MIN_SIZE.y);
    egui::Rect::from_min_size(
        egui::pos2(
            (viewport_rect.max.x - INSPECTOR_DEFAULT_SIZE.x - SHELL_MARGIN)
                .max(viewport_rect.min.x + SHELL_MARGIN),
            top,
        ),
        egui::vec2(INSPECTOR_DEFAULT_SIZE.x, height),
    )
}

fn default_utility_panel_position(viewport_rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        viewport_rect.min.x + SHELL_MARGIN,
        viewport_rect.max.y - UTILITY_DEFAULT_SIZE.y - UTILITY_STRIP_HEIGHT - SHELL_MARGIN - 8.0,
    )
}

fn chrome_frame(ctx: &egui::Context) -> egui::Frame {
    chrome::surface_frame(ctx)
}

#[allow(clippy::too_many_arguments)]
fn show_chrome_window(
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
        .title_bar(false)
        .collapsible(false)
        .resizable(true)
        .default_pos(default_pos)
        .default_size(default_size)
        .min_size(min_size)
        .frame(egui::Frame::new())
        .show(ctx, |ui| {
            chrome::surface_frame(ctx).show(ui, |ui| add_contents(ui));
        });
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

fn draw_scene_panel_window_header(
    ui: &mut egui::Ui,
    object_count: usize,
    dockable: bool,
    dock_clicked: &mut bool,
    reset_clicked: &mut bool,
    hide_clicked: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Scene").strong());
        chips::draw_chip(
            ui,
            &format!("{object_count} objects"),
            egui::Color32::from_rgb(42, 46, 60),
            egui::Color32::from_rgb(220, 226, 240),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if chrome::action_button(ui, "Hide", false).clicked() {
                *hide_clicked = true;
            }
            if chrome::action_button(ui, "Reset", false).clicked() {
                *reset_clicked = true;
            }
            if dockable && chrome::action_button(ui, "Dock", false).clicked() {
                *dock_clicked = true;
            }
        });
    });
    ui.add_space(8.0);
}

fn draw_scene_panel_docked_header(ui: &mut egui::Ui, actions: &mut ActionSink) {
    chrome::tree_row_frame(false, false).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Scene").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if chrome::action_button(ui, "Undock", false).clicked() {
                    actions.push(Action::UndockShellPanel(ShellPanelKind::Tool));
                }
                if chrome::action_button(ui, "Reset", false).clicked() {
                    actions.push(Action::ResetPrimaryShellLayout);
                }
            });
        });
    });
    ui.add_space(8.0);
}

fn draw_subject_inspector_window_header(
    ui: &mut egui::Ui,
    dockable: bool,
    dock_clicked: &mut bool,
    reset_clicked: &mut bool,
    hide_clicked: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Inspector").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if chrome::action_button(ui, "Hide", false).clicked() {
                *hide_clicked = true;
            }
            if chrome::action_button(ui, "Reset", false).clicked() {
                *reset_clicked = true;
            }
            if dockable && chrome::action_button(ui, "Dock", false).clicked() {
                *dock_clicked = true;
            }
        });
    });
    ui.add_space(8.0);
}

fn draw_subject_inspector_docked_header(ui: &mut egui::Ui, actions: &mut ActionSink) {
    chrome::card_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Inspector").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if chrome::action_button(ui, "Undock", false).clicked() {
                    actions.push(Action::UndockShellPanel(ShellPanelKind::Inspector));
                }
                if chrome::action_button(ui, "Reset", false).clicked() {
                    actions.push(Action::ResetPrimaryShellLayout);
                }
            });
        });
    });
    ui.add_space(8.0);
}

fn show_inspector_card(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash,
    title: &str,
    default_open: bool,
    subtitle: Option<&str>,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    chrome::card_frame().show(ui, |ui| {
        egui::CollapsingHeader::new(title)
            .default_open(default_open)
            .id_salt(id_source)
            .show(ui, |ui| {
                if let Some(subtitle) = subtitle {
                    ui.small(subtitle);
                    ui.add_space(6.0);
                }
                add_contents(ui);
            });
    });
    ui.add_space(8.0);
}

fn shell_window_actions(
    ui: &mut egui::Ui,
    dockable: bool,
    status: &str,
    dock_clicked: &mut bool,
    reset_clicked: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.small(status);
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

fn draw_docked_shell_tab_header(
    ui: &mut egui::Ui,
    panel: ShellPanelKind,
    label: &str,
    actions: &mut ActionSink,
) {
    ui.horizontal(|ui| {
        ui.small(label);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_window_id_changes_with_layout_revision() {
        let initial_id = shell_window_id(SCENE_PANEL_WINDOW_KEY, 0, 0);
        let reset_id = shell_window_id(SCENE_PANEL_WINDOW_KEY, 1, 0);
        let undocked_id = shell_window_id(SCENE_PANEL_WINDOW_KEY, 1, 1);

        assert_ne!(initial_id, reset_id);
        assert_ne!(reset_id, undocked_id);
    }

    #[test]
    fn default_scene_panel_rect_is_left_biased_vertical_rail() {
        let viewport_rect =
            egui::Rect::from_min_size(egui::pos2(20.0, 10.0), egui::vec2(1440.0, 900.0));
        let rect = default_scene_panel_rect(viewport_rect);

        assert_eq!(rect.min.x, viewport_rect.min.x + SHELL_MARGIN);
        assert!(rect.width() <= 340.0);
        assert!(rect.height() > rect.width());
        assert!(rect.min.y < viewport_rect.center().y);
    }

    #[test]
    fn default_inspector_panel_rect_is_right_biased_vertical_rail() {
        let viewport_rect =
            egui::Rect::from_min_size(egui::pos2(20.0, 10.0), egui::vec2(1440.0, 900.0));
        let rect = default_inspector_panel_rect(viewport_rect);

        assert!(rect.max.x <= viewport_rect.max.x - SHELL_MARGIN + 0.001);
        assert!(rect.min.x > viewport_rect.center().x);
        assert!(rect.width() >= INSPECTOR_MIN_SIZE.x);
        assert!(rect.width() <= 380.0);
        assert!(rect.height() > rect.width());
        assert!(rect.min.y < viewport_rect.center().y);
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
    fn strip_toggle_closes_scene_panel_when_floating() {
        let mut shell = PrimaryShellState::default();
        let mut actions = ActionSink::new();

        toggle_scene_panel_from_strip(&mut shell, &mut actions);

        assert!(shell.tool_panel.is_hidden());
        assert!(actions.is_empty());
    }

    #[test]
    fn utility_toggle_switches_tab_without_undocking_existing_panel() {
        let mut shell = PrimaryShellState::default();
        shell.drawer_panel.dock();
        shell.active_utility_tab = PrimaryShellUtilityTab::History;
        let mut actions = ActionSink::new();

        toggle_utility_panel_from_strip(&mut shell, PrimaryShellUtilityTab::Advanced, &mut actions);

        assert_eq!(shell.active_utility_tab, PrimaryShellUtilityTab::Advanced);
        assert!(shell.drawer_panel.is_docked());
        assert!(actions.is_empty());
    }

    #[test]
    fn sync_inspector_tab_forces_display_in_measure_mode() {
        let mut shell = PrimaryShellState::default();
        shell.interaction_mode = InteractionMode::Measure;
        shell.active_inspector_tab = PrimaryShellInspectorTab::Properties;

        sync_inspector_tab(&mut shell);

        assert_eq!(
            shell.active_inspector_tab,
            PrimaryShellInspectorTab::Display
        );
    }
}
