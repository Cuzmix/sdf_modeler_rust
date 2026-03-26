use std::rc::Rc;

use slint::VecModel;

use crate::app::slint_frontend::{SlintHostWindow, ViewportPathView};
use crate::gizmo::{ViewportGizmoOverlay, ViewportGizmoPath};

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

fn viewport_path_view(path: &ViewportGizmoPath) -> ViewportPathView {
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
