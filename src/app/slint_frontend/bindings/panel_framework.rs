use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{
    PanelBarModel, PanelFrameModel, PanelLauncherItemModel, PanelSheetModel, ShellSnapshot,
};
use crate::app::slint_frontend::{
    PanelAnchorView, PanelBarViewState, PanelEdgeView, PanelFrameView, PanelFrameworkViewState,
    PanelKindView, PanelLauncherItemView, PanelOrientationView,
    PanelPointerInteractionKind as PanelPointerInteractionKindView,
    PanelResizeHandle as PanelResizeHandleView, PanelSheetViewState,
};

pub(super) fn build_panel_framework_state(snapshot: &ShellSnapshot) -> PanelFrameworkViewState {
    PanelFrameworkViewState {
        bar: panel_bar_state(&snapshot.panel_framework.bar),
        transient_panel: panel_sheet_state(snapshot.panel_framework.transient_panel.as_ref()),
        pinned_panels: Rc::new(VecModel::from(
            snapshot
                .panel_framework
                .pinned_panels
                .iter()
                .map(|panel| panel_sheet_state(Some(panel)))
                .collect::<Vec<_>>(),
        ))
        .into(),
        panel_interaction_active: snapshot.panel_framework.panel_interaction_active,
        interaction_panel_kind: panel_kind_view(snapshot.panel_framework.interaction_panel_kind),
        active_interaction_kind: panel_pointer_interaction_kind_view(
            snapshot.panel_framework.active_interaction_kind,
        ),
        active_resize_handle: panel_resize_handle_view(
            snapshot.panel_framework.active_resize_handle,
        ),
    }
}

fn panel_bar_state(model: &PanelBarModel) -> PanelBarViewState {
    PanelBarViewState {
        visible: model.visible,
        edge: match model.edge {
            crate::app::state::PanelBarEdge::Left => PanelEdgeView::Left,
            crate::app::state::PanelBarEdge::Right => PanelEdgeView::Right,
            crate::app::state::PanelBarEdge::Top => PanelEdgeView::Top,
            crate::app::state::PanelBarEdge::Bottom => PanelEdgeView::Bottom,
        },
        orientation: match model.orientation {
            crate::app::state::PanelBarOrientation::Vertical => PanelOrientationView::Vertical,
            crate::app::state::PanelBarOrientation::Horizontal => PanelOrientationView::Horizontal,
        },
        items: Rc::new(VecModel::from(
            model
                .items
                .iter()
                .map(panel_launcher_item_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
    }
}

fn panel_launcher_item_view(model: &PanelLauncherItemModel) -> PanelLauncherItemView {
    PanelLauncherItemView {
        label: model.label.clone().into(),
        short_label: model.short_label.clone().into(),
        icon_key: model.icon_key.clone().unwrap_or_default().into(),
        active: model.active,
        pinned: model.pinned,
        show_drag_indicator: model.show_drag_indicator,
        kind: panel_kind_view(model.kind),
    }
}

fn panel_sheet_state(model: Option<&PanelSheetModel>) -> PanelSheetViewState {
    let Some(model) = model else {
        return PanelSheetViewState {
            visible: false,
            title: "".into(),
            collapsed_title: "".into(),
            collapsed_width: 0.0,
            collapsed_height: 0.0,
            pinned: false,
            collapsed: false,
            kind: PanelKindView::ObjectProperties,
            anchor: PanelAnchorView::LeftOfBar,
            movable: false,
            resizable: false,
            frame: PanelFrameView {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
        };
    };

    PanelSheetViewState {
        visible: true,
        title: model.title.clone().into(),
        collapsed_title: model.collapsed_title.clone().into(),
        collapsed_width: model.collapsed_width,
        collapsed_height: model.collapsed_height,
        pinned: model.pinned,
        collapsed: model.collapsed,
        kind: panel_kind_view(model.kind),
        anchor: match model.anchor {
            crate::app::state::PanelSheetAnchor::Left => PanelAnchorView::LeftOfBar,
            crate::app::state::PanelSheetAnchor::Right => PanelAnchorView::RightOfBar,
            crate::app::state::PanelSheetAnchor::Above => PanelAnchorView::AboveBar,
            crate::app::state::PanelSheetAnchor::Below => PanelAnchorView::BelowBar,
        },
        movable: model.movable,
        resizable: model.resizable,
        frame: panel_frame_view(model.frame),
    }
}

fn panel_frame_view(model: PanelFrameModel) -> PanelFrameView {
    PanelFrameView {
        x: model.x,
        y: model.y,
        width: model.width,
        height: model.height,
    }
}

fn panel_kind_view(kind: crate::app::state::PanelKind) -> PanelKindView {
    match kind {
        crate::app::state::PanelKind::Tool => PanelKindView::Tool,
        crate::app::state::PanelKind::ObjectProperties => PanelKindView::ObjectProperties,
        crate::app::state::PanelKind::RenderSettings => PanelKindView::RenderSettings,
        crate::app::state::PanelKind::Scene => PanelKindView::Scene,
        crate::app::state::PanelKind::History => PanelKindView::History,
        crate::app::state::PanelKind::ReferenceImages => PanelKindView::ReferenceImages,
    }
}

fn panel_pointer_interaction_kind_view(
    kind: crate::app::state::PanelPointerInteractionKind,
) -> PanelPointerInteractionKindView {
    match kind {
        crate::app::state::PanelPointerInteractionKind::Move => {
            PanelPointerInteractionKindView::Move
        }
        crate::app::state::PanelPointerInteractionKind::Resize(_) => {
            PanelPointerInteractionKindView::Resize
        }
    }
}

fn panel_resize_handle_view(
    handle: Option<crate::app::state::PanelResizeHandle>,
) -> PanelResizeHandleView {
    match handle {
        Some(crate::app::state::PanelResizeHandle::Top) => PanelResizeHandleView::Top,
        Some(crate::app::state::PanelResizeHandle::Right) => PanelResizeHandleView::Right,
        Some(crate::app::state::PanelResizeHandle::Bottom) => PanelResizeHandleView::Bottom,
        Some(crate::app::state::PanelResizeHandle::Left) => PanelResizeHandleView::Left,
        Some(crate::app::state::PanelResizeHandle::TopLeft) => PanelResizeHandleView::TopLeft,
        Some(crate::app::state::PanelResizeHandle::TopRight) => PanelResizeHandleView::TopRight,
        Some(crate::app::state::PanelResizeHandle::BottomLeft) => {
            PanelResizeHandleView::BottomLeft
        }
        Some(crate::app::state::PanelResizeHandle::BottomRight) => {
            PanelResizeHandleView::BottomRight
        }
        None => PanelResizeHandleView::None,
    }
}
