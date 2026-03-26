#![allow(dead_code)]

#[path = "gizmo/selection.rs"]
mod selection;
#[path = "gizmo/viewport.rs"]
mod viewport;

use crate::graph::scene::NodeId;

pub(crate) use selection::{current_transform_target, GizmoSelection};
pub(crate) use viewport::{
    build_viewport_gizmo_overlay, run_viewport_gizmo_interaction, GizmoInputSnapshot,
    ViewportGizmoOverlay, ViewportGizmoPath,
};

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Translate => "Move",
            Self::Rotate => "Rotate",
            Self::Scale => "Scale",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GizmoSpace {
    Local,
    World,
}

impl GizmoSpace {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Local => "Local",
            Self::World => "World",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum GizmoState {
    #[default]
    Idle,
    PendingStart {
        axis: viewport::GizmoAxis,
    },
    DraggingSingle {
        axis: viewport::GizmoAxis,
        node_id: NodeId,
        rotation_drag: viewport::ScreenRotationDragState,
    },
    DraggingMulti {
        axis: viewport::GizmoAxis,
        drag_session: viewport::GizmoDragSession,
    },
}
