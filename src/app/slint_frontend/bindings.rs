use std::rc::Rc;
use std::sync::atomic::Ordering;

use slint::VecModel;

use crate::app::frontend_models::ShellSnapshot;
use crate::app::{BakeStatus, ExportStatus, ImportStatus, SdfApp};
use crate::gizmo::ViewportGizmoOverlay;

use super::{
    HistoryRowView, ImportDialogState, InspectorPanelState, ReferenceRowView, RenderSettingsState,
    ScenePanelState, SceneRowView, SlintHostWindow, TopBarState, UtilityPanelState,
    ViewportPathView,
};

pub(super) fn apply_shell_snapshot(window: &SlintHostWindow, snapshot: &ShellSnapshot) {
    window.set_top_bar_state(build_top_bar_snapshot(window, snapshot));
    window.set_scene_panel_state(build_scene_panel_state(window, snapshot));
    window.set_inspector_panel_state(build_inspector_panel_state(snapshot));
    window.set_utility_panel_state(build_utility_panel_snapshot(window, snapshot));
}

pub(super) fn apply_runtime_ui_state(window: &SlintHostWindow, app: &SdfApp) {
    let mut top_bar_state = window.get_top_bar_state();
    top_bar_state.toast_text = app
        .ui
        .toasts
        .last()
        .map(|toast| toast.message.clone())
        .unwrap_or_default()
        .into();
    top_bar_state.task_status = format_task_status(app).into();
    top_bar_state.gizmo_translate_active =
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Translate);
    top_bar_state.gizmo_rotate_active =
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Rotate);
    top_bar_state.gizmo_scale_active =
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Scale);
    top_bar_state.gizmo_local_space = matches!(app.gizmo.space, crate::gizmo::GizmoSpace::Local);
    top_bar_state.camera_is_ortho = app.doc.camera.orthographic;
    top_bar_state.measurement_mode_active = app.ui.measurement_mode;
    top_bar_state.turntable_active = app.ui.turntable_active;
    window.set_top_bar_state(top_bar_state);

    let mut scene_panel_state = window.get_scene_panel_state();
    scene_panel_state.scene_filter = app.ui.scene_tree_search.clone().into();
    window.set_scene_panel_state(scene_panel_state);

    let mut utility_panel_state = window.get_utility_panel_state();
    utility_panel_state.import_dialog = build_import_dialog_state(app);
    window.set_utility_panel_state(utility_panel_state);
}

pub(super) fn apply_gizmo_overlay(
    window: &SlintHostWindow,
    overlay: Option<&ViewportGizmoOverlay>,
) {
    let (viewbox_width, viewbox_height, paths) = if let Some(overlay) = overlay {
        (
            overlay.viewbox_size[0],
            overlay.viewbox_size[1],
            overlay
                .paths
                .iter()
                .map(viewport_path_view)
                .collect::<Vec<_>>(),
        )
    } else {
        (1.0, 1.0, Vec::new())
    };

    window.set_viewport_gizmo_viewbox_width(viewbox_width);
    window.set_viewport_gizmo_viewbox_height(viewbox_height);
    window.set_viewport_gizmo_paths(Rc::new(VecModel::from(paths)).into());
}

fn build_top_bar_snapshot(window: &SlintHostWindow, snapshot: &ShellSnapshot) -> TopBarState {
    let mut state = window.get_top_bar_state();
    state.viewport_status = format!(
        "{} / {} / {}",
        snapshot.viewport_status.interaction_label,
        snapshot.viewport_status.transform_label,
        snapshot.viewport_status.space_label
    )
    .into();
    state
}

fn build_scene_panel_state(window: &SlintHostWindow, snapshot: &ShellSnapshot) -> ScenePanelState {
    let current_state = window.get_scene_panel_state();
    let scene_rows = snapshot
        .scene_panel
        .rows
        .iter()
        .map(scene_row_view)
        .collect::<Vec<_>>();
    let history_rows = snapshot
        .utility
        .history_rows
        .iter()
        .map(history_row_view)
        .collect::<Vec<_>>();

    ScenePanelState {
        selection_summary: selection_summary(snapshot).into(),
        selected_name: snapshot.inspector.name.clone().into(),
        scene_filter: current_state.scene_filter,
        scene_rows: Rc::new(VecModel::from(scene_rows)).into(),
        history_rows: Rc::new(VecModel::from(history_rows)).into(),
    }
}

fn build_inspector_panel_state(snapshot: &ShellSnapshot) -> InspectorPanelState {
    let transform = snapshot.inspector.transform.as_ref();
    let material = snapshot.inspector.material.as_ref();
    let operation = snapshot.inspector.operation.as_ref();
    let sculpt = snapshot.inspector.sculpt.as_ref();
    let light = snapshot.inspector.light.as_ref();

    InspectorPanelState {
        title: snapshot.inspector.title.clone().into(),
        chips: snapshot.inspector.chips.join(" | ").into(),
        display: join_lines(&snapshot.inspector.display_lines).into(),
        property_summary: join_lines(&snapshot.inspector.property_lines).into(),
        has_transform: transform.is_some(),
        transform_pos_x: transform.map_or(0.0, |model| model.position[0]),
        transform_pos_y: transform.map_or(0.0, |model| model.position[1]),
        transform_pos_z: transform.map_or(0.0, |model| model.position[2]),
        transform_rot_x: transform.map_or(0.0, |model| model.rotation_deg[0]),
        transform_rot_y: transform.map_or(0.0, |model| model.rotation_deg[1]),
        transform_rot_z: transform.map_or(0.0, |model| model.rotation_deg[2]),
        selected_scale_x: transform.map_or(1.0, |model| model.scale[0]),
        selected_scale_y: transform.map_or(1.0, |model| model.scale[1]),
        selected_scale_z: transform.map_or(1.0, |model| model.scale[2]),
        can_scale: transform.is_some_and(|model| model.can_scale),
        has_material: material.is_some(),
        material_color_r: material.map_or(0.0, |model| model.base_color[0]),
        material_color_g: material.map_or(0.0, |model| model.base_color[1]),
        material_color_b: material.map_or(0.0, |model| model.base_color[2]),
        material_roughness: material.map_or(0.0, |model| model.roughness),
        material_metallic: material.map_or(0.0, |model| model.metallic),
        has_operation: operation.is_some(),
        operation_label: operation
            .map(|model| model.op_label.clone())
            .unwrap_or_default()
            .into(),
        operation_smooth_k: operation.map_or(0.0, |model| model.smooth_k),
        operation_steps: operation.map_or(0.0, |model| model.steps),
        operation_color_blend: operation.map_or(0.0, |model| model.color_blend),
        has_sculpt: sculpt.is_some(),
        sculpt_resolution: sculpt.map_or(0, |model| model.desired_resolution as i32),
        sculpt_layer_intensity: sculpt.map_or(0.0, |model| model.layer_intensity),
        sculpt_brush_radius: sculpt.map_or(0.0, |model| model.brush_radius),
        sculpt_brush_strength: sculpt.map_or(0.0, |model| model.brush_strength),
        has_light: light.is_some(),
        light_label: light
            .map(|model| model.light_type_label.clone())
            .unwrap_or_default()
            .into(),
        light_color_r: light.map_or(0.0, |model| model.color[0]),
        light_color_g: light.map_or(0.0, |model| model.color[1]),
        light_color_b: light.map_or(0.0, |model| model.color[2]),
        light_intensity: light.map_or(0.0, |model| model.intensity),
        light_range: light.map_or(0.0, |model| model.range),
        light_cast_shadows: light.is_some_and(|model| model.cast_shadows),
        light_volumetric: light.is_some_and(|model| model.volumetric),
        light_volumetric_density: light.map_or(0.0, |model| model.volumetric_density),
    }
}

fn build_utility_panel_snapshot(
    window: &SlintHostWindow,
    snapshot: &ShellSnapshot,
) -> UtilityPanelState {
    let reference_rows = snapshot
        .utility
        .reference_rows
        .iter()
        .map(reference_row_view)
        .collect::<Vec<_>>();
    let render = &snapshot.utility.render_settings;

    UtilityPanelState {
        reference_rows: Rc::new(VecModel::from(reference_rows)).into(),
        render_settings: RenderSettingsState {
            show_grid: render.show_grid,
            show_node_labels: render.show_node_labels,
            show_bounding_box: render.show_bounding_box,
            show_light_gizmos: render.show_light_gizmos,
            shadows_enabled: render.shadows_enabled,
            ao_enabled: render.ao_enabled,
            export_resolution: render.export_resolution as i32,
            adaptive_export: render.adaptive_export,
        },
        import_dialog: window.get_utility_panel_state().import_dialog,
    }
}

fn build_import_dialog_state(app: &SdfApp) -> ImportDialogState {
    if let Some(dialog) = app.ui.import_dialog.as_ref() {
        ImportDialogState {
            visible: true,
            summary: format_import_summary(dialog).into(),
            resolution: dialog.resolution as i32,
        }
    } else {
        ImportDialogState {
            visible: false,
            summary: "".into(),
            resolution: 0,
        }
    }
}

fn viewport_path_view(path: &crate::gizmo::ViewportGizmoPath) -> ViewportPathView {
    ViewportPathView {
        commands: path.commands.clone().into(),
        stroke: color_from_rgba(path.stroke_rgba),
        fill: color_from_rgba(path.fill_rgba),
        stroke_width: path.stroke_width,
    }
}

fn color_from_rgba(rgba: [u8; 4]) -> slint::Color {
    slint::Color::from_argb_u8(rgba[3], rgba[0], rgba[1], rgba[2])
}

fn selection_summary(snapshot: &ShellSnapshot) -> String {
    let row_count = snapshot.scene_panel.rows.len();
    match snapshot.scene_panel.selection_count {
        0 => format!("No selection | {row_count} visible rows"),
        1 => format!("{} | {row_count} visible rows", snapshot.inspector.title),
        count => format!("{count} selected | {row_count} visible rows"),
    }
}

fn format_task_status(app: &SdfApp) -> String {
    if let Some(status) = format_bake_status(&app.async_state.bake_status) {
        return status;
    }
    if let Some(status) = format_export_status(&app.async_state.export_status) {
        return status;
    }
    if let Some(status) = format_import_status(&app.async_state.import_status) {
        return status;
    }
    if app.async_state.sculpt_dragging {
        return "Sculpting".to_string();
    }
    if app.ui.measurement_mode {
        return format!("Measure {} pts", app.ui.measurement_points.len());
    }
    String::new()
}

fn format_bake_status(status: &BakeStatus) -> Option<String> {
    match status {
        BakeStatus::Idle => None,
        BakeStatus::InProgress {
            progress,
            total,
            flatten,
            ..
        } => Some(format!(
            "{} {}%",
            if *flatten { "Flatten bake" } else { "Bake" },
            progress_percent(progress.load(Ordering::Relaxed), *total)
        )),
    }
}

fn format_export_status(status: &ExportStatus) -> Option<String> {
    match status {
        ExportStatus::Idle => None,
        ExportStatus::InProgress {
            progress,
            total,
            resolution,
            ..
        } => Some(format!(
            "Export {}% | {} res",
            progress_percent(progress.load(Ordering::Relaxed), *total),
            resolution
        )),
    }
}

fn format_import_status(status: &ImportStatus) -> Option<String> {
    match status {
        ImportStatus::Idle => None,
        ImportStatus::InProgress {
            progress,
            total,
            filename,
            ..
        } => Some(format!(
            "Import {}% | {}",
            progress_percent(progress.load(Ordering::Relaxed), *total),
            filename
        )),
    }
}

fn progress_percent(progress: u32, total: u32) -> u32 {
    if total == 0 {
        return 0;
    }
    progress.saturating_mul(100).min(total.saturating_mul(100)) / total
}

fn scene_row_view(row: &crate::app::frontend_models::ScenePanelRow) -> SceneRowView {
    SceneRowView {
        label: scene_row_label(row).into(),
        selected: row.selected,
        hidden: row.hidden,
        locked: row.locked,
    }
}

fn scene_row_label(row: &crate::app::frontend_models::ScenePanelRow) -> String {
    let indent = "  ".repeat(row.depth);
    let visibility = if row.hidden { "[hidden] " } else { "" };
    let locked = if row.locked { "[locked] " } else { "" };
    let selection = if row.selected { "> " } else { "  " };
    format!("{selection}{indent}{visibility}{locked}{}", row.label)
}

fn history_row_view(entry: &crate::app::frontend_models::HistoryEntry) -> HistoryRowView {
    HistoryRowView {
        label: entry.label.clone().into(),
        is_undo: entry.is_undo,
    }
}

fn reference_row_view(entry: &crate::app::frontend_models::ReferenceImageRow) -> ReferenceRowView {
    ReferenceRowView {
        label: entry.label.clone().into(),
        plane_label: entry.plane_label.clone().into(),
        visible: entry.visible,
        locked: entry.locked,
        opacity: entry.opacity,
        scale: entry.scale,
    }
}

fn join_lines(lines: &[String]) -> String {
    lines.join("\n")
}

fn format_import_summary(dialog: &crate::app::state::ImportDialog) -> String {
    format!(
        "{}\nVertices: {}\nTriangles: {}\nBounds: {:.2}, {:.2}, {:.2}\nAuto resolution: {}",
        dialog.filename,
        dialog.vertex_count,
        dialog.triangle_count,
        dialog.bounds_size.x,
        dialog.bounds_size.y,
        dialog.bounds_size.z,
        dialog.auto_resolution
    )
}
