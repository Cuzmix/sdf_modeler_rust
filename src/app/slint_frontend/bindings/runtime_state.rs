use std::sync::atomic::Ordering;

use crate::app::slint_frontend::{ImportDialogState, SlintHostWindow};
use crate::app::{BakeStatus, ExportStatus, ImportStatus, SdfApp};

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
    top_bar_state.file_actions_enabled =
        !matches!(app.async_state.bake_status, BakeStatus::InProgress { .. })
            && !matches!(
                app.async_state.export_status,
                ExportStatus::InProgress { .. }
            )
            && !matches!(
                app.async_state.import_status,
                ImportStatus::InProgress { .. }
            );
    window.set_top_bar_state(top_bar_state);

    let mut scene_panel_state = window.get_scene_panel_state();
    scene_panel_state.scene_filter = app.ui.scene_panel.filter_query.clone().into();
    window.set_scene_panel_state(scene_panel_state);

    let mut utility_panel_state = window.get_utility_panel_state();
    utility_panel_state.import_dialog = build_import_dialog_state(app);
    window.set_utility_panel_state(utility_panel_state);
}

fn build_import_dialog_state(app: &SdfApp) -> ImportDialogState {
    if let Some(dialog) = app.ui.import_dialog.as_ref() {
        ImportDialogState {
            visible: true,
            summary: format_import_summary(dialog).into(),
            resolution: dialog.resolution as i32,
            confirm_enabled: true,
        }
    } else {
        ImportDialogState {
            visible: false,
            summary: "".into(),
            resolution: 0,
            confirm_enabled: false,
        }
    }
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
