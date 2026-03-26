use std::rc::Rc;
use std::sync::atomic::Ordering;

use slint::VecModel;

use crate::app::frontend_models::ShellSnapshot;
use crate::app::{BakeStatus, ExportStatus, ImportStatus, SdfApp};
use crate::gizmo::ViewportGizmoOverlay;

use super::{HistoryRowView, ReferenceRowView, SceneRowView, SlintHostWindow, ViewportPathView};

pub(super) fn apply_shell_snapshot(window: &SlintHostWindow, snapshot: &ShellSnapshot) {
    let scene_rows = snapshot
        .scene_panel
        .rows
        .iter()
        .map(scene_row_view)
        .collect::<Vec<_>>();
    window.set_scene_rows(Rc::new(VecModel::from(scene_rows)).into());
    window.set_selected_name(snapshot.inspector.name.clone().into());

    window.set_inspector_title(snapshot.inspector.title.clone().into());
    window.set_inspector_chips(snapshot.inspector.chips.join(" | ").into());
    window.set_inspector_property_summary(join_lines(&snapshot.inspector.property_lines).into());
    window.set_inspector_display(join_lines(&snapshot.inspector.display_lines).into());

    let transform = snapshot.inspector.transform.as_ref();
    window.set_has_transform(transform.is_some());
    window.set_transform_pos_x(transform.map_or(0.0, |model| model.position[0]));
    window.set_transform_pos_y(transform.map_or(0.0, |model| model.position[1]));
    window.set_transform_pos_z(transform.map_or(0.0, |model| model.position[2]));
    window.set_transform_rot_x(transform.map_or(0.0, |model| model.rotation_deg[0]));
    window.set_transform_rot_y(transform.map_or(0.0, |model| model.rotation_deg[1]));
    window.set_transform_rot_z(transform.map_or(0.0, |model| model.rotation_deg[2]));
    window.set_selected_scale_x(transform.map_or(1.0, |model| model.scale[0]));
    window.set_selected_scale_y(transform.map_or(1.0, |model| model.scale[1]));
    window.set_selected_scale_z(transform.map_or(1.0, |model| model.scale[2]));
    window.set_can_scale(transform.is_some_and(|model| model.can_scale));

    let material = snapshot.inspector.material.as_ref();
    window.set_has_material(material.is_some());
    window.set_material_color_r(material.map_or(0.0, |model| model.base_color[0]));
    window.set_material_color_g(material.map_or(0.0, |model| model.base_color[1]));
    window.set_material_color_b(material.map_or(0.0, |model| model.base_color[2]));
    window.set_material_roughness(material.map_or(0.0, |model| model.roughness));
    window.set_material_metallic(material.map_or(0.0, |model| model.metallic));

    let operation = snapshot.inspector.operation.as_ref();
    window.set_has_operation(operation.is_some());
    window.set_operation_label(
        operation
            .map(|model| model.op_label.clone())
            .unwrap_or_default()
            .into(),
    );
    window.set_operation_smooth_k(operation.map_or(0.0, |model| model.smooth_k));
    window.set_operation_steps(operation.map_or(0.0, |model| model.steps));
    window.set_operation_color_blend(operation.map_or(0.0, |model| model.color_blend));

    let sculpt = snapshot.inspector.sculpt.as_ref();
    window.set_has_sculpt(sculpt.is_some());
    window.set_sculpt_resolution(sculpt.map_or(0, |model| model.desired_resolution as i32));
    window.set_sculpt_layer_intensity(sculpt.map_or(0.0, |model| model.layer_intensity));
    window.set_sculpt_brush_radius(sculpt.map_or(0.0, |model| model.brush_radius));
    window.set_sculpt_brush_strength(sculpt.map_or(0.0, |model| model.brush_strength));

    let light = snapshot.inspector.light.as_ref();
    window.set_has_light(light.is_some());
    window.set_light_label(
        light
            .map(|model| model.light_type_label.clone())
            .unwrap_or_default()
            .into(),
    );
    window.set_light_color_r(light.map_or(0.0, |model| model.color[0]));
    window.set_light_color_g(light.map_or(0.0, |model| model.color[1]));
    window.set_light_color_b(light.map_or(0.0, |model| model.color[2]));
    window.set_light_intensity(light.map_or(0.0, |model| model.intensity));
    window.set_light_range(light.map_or(0.0, |model| model.range));
    window.set_light_cast_shadows(light.is_some_and(|model| model.cast_shadows));
    window.set_light_volumetric(light.is_some_and(|model| model.volumetric));
    window.set_light_volumetric_density(light.map_or(0.0, |model| model.volumetric_density));

    let history_rows = snapshot
        .utility
        .history_rows
        .iter()
        .map(history_row_view)
        .collect::<Vec<_>>();
    window.set_history_rows(Rc::new(VecModel::from(history_rows)).into());

    let reference_rows = snapshot
        .utility
        .reference_rows
        .iter()
        .map(reference_row_view)
        .collect::<Vec<_>>();
    window.set_reference_rows(Rc::new(VecModel::from(reference_rows)).into());

    let render = &snapshot.utility.render_settings;
    window.set_render_show_grid(render.show_grid);
    window.set_render_show_node_labels(render.show_node_labels);
    window.set_render_show_bounding_box(render.show_bounding_box);
    window.set_render_show_light_gizmos(render.show_light_gizmos);
    window.set_render_shadows_enabled(render.shadows_enabled);
    window.set_render_ao_enabled(render.ao_enabled);
    window.set_export_resolution(render.export_resolution as i32);
    window.set_adaptive_export(render.adaptive_export);

    window.set_viewport_status(
        format!(
            "{} / {} / {}",
            snapshot.viewport_status.interaction_label,
            snapshot.viewport_status.transform_label,
            snapshot.viewport_status.space_label
        )
        .into(),
    );
    window.set_selection_summary(selection_summary(snapshot).into());
}

pub(super) fn apply_runtime_ui_state(window: &SlintHostWindow, app: &SdfApp) {
    let toast_text = app
        .ui
        .toasts
        .last()
        .map(|toast| toast.message.clone())
        .unwrap_or_default();
    window.set_toast_text(toast_text.into());
    window.set_task_status(format_task_status(app).into());
    window.set_scene_filter(app.ui.scene_tree_search.clone().into());
    window.set_gizmo_translate_active(
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Translate),
    );
    window.set_gizmo_rotate_active(
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Rotate),
    );
    window.set_gizmo_scale_active(
        app.gizmo.gizmo_visible && matches!(app.gizmo.mode, crate::gizmo::GizmoMode::Scale),
    );
    window.set_gizmo_local_space(matches!(app.gizmo.space, crate::gizmo::GizmoSpace::Local));
    window.set_camera_is_ortho(app.doc.camera.orthographic);
    window.set_measurement_mode_active(app.ui.measurement_mode);
    window.set_turntable_active(app.ui.turntable_active);

    if let Some(dialog) = app.ui.import_dialog.as_ref() {
        window.set_show_import_dialog(true);
        window.set_import_summary(format_import_summary(dialog).into());
        window.set_import_resolution(dialog.resolution as i32);
    } else {
        window.set_show_import_dialog(false);
        window.set_import_summary("".into());
        window.set_import_resolution(0);
    }
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
