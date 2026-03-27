use std::cell::RefCell;
use std::rc::Rc;

use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{
    PanelHeaderAction, PanelKindView, PanelPointerAction,
    PanelPointerInteractionKind as PanelPointerInteractionKindView, PanelPointerPhase,
    PanelResizeHandle as PanelResizeHandleView, SlintHostWindow,
};
use crate::app::state::{PanelBarId, PanelKind, PanelPointerInteractionKind, PanelResizeHandle};
use crate::app::ui_geometry::FloatingPanelBounds;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let launcher_context = context.clone();
    window.on_panel_launcher_action(move |kind| {
        mutate_host_and_tick(&launcher_context, move |host_state| {
            host_state.queue_action(Action::TogglePanel(
                panel_kind(kind),
                PanelBarId::PrimaryRight,
            ));
        });
    });

    let header_context = context.clone();
    window.on_panel_header_action(move |kind, action| {
        mutate_host_and_tick(&header_context, move |host_state| {
            let action = match action {
                PanelHeaderAction::Close => Action::ClosePanel(panel_kind(kind)),
                PanelHeaderAction::Pin => Action::PinPanel(panel_kind(kind)),
                PanelHeaderAction::Unpin => Action::UnpinPanel(panel_kind(kind)),
                PanelHeaderAction::ToggleCollapsed => {
                    Action::TogglePinnedPanelCollapsed(panel_kind(kind))
                }
            };
            host_state.queue_action(action);
        });
    });

    let dismiss_context = context.clone();
    window.on_dismiss_transient_panels(move || {
        mutate_host_and_tick(&dismiss_context, move |host_state| {
            host_state.queue_action(Action::DismissTransientPanels);
        });
    });

    let drag_context = context.clone();
    let drag_last_position = Rc::new(RefCell::new(None::<[f32; 2]>));
    let drag_last_position_for_callback = drag_last_position.clone();
    window.on_panel_pointer_action(move |kind, action| {
        let usable_rect = drag_context
            .window_weak
            .upgrade()
            .map(|window| {
                FloatingPanelBounds::from_min_size(
                    window.get_overlay_usable_x(),
                    window.get_overlay_usable_y(),
                    window.get_overlay_usable_width(),
                    window.get_overlay_usable_height(),
                )
            })
            .unwrap_or_else(|| FloatingPanelBounds::from_min_size(0.0, 0.0, 960.0, 540.0));
        let drag_last_position = drag_last_position_for_callback.clone();
        mutate_host_and_tick(&drag_context, move |host_state| {
            handle_panel_pointer_action(
                host_state,
                panel_kind(kind),
                action,
                usable_rect,
                &drag_last_position,
            );
        });
    });
}

fn panel_kind(kind: PanelKindView) -> PanelKind {
    match kind {
        PanelKindView::Tool => PanelKind::Tool,
        PanelKindView::ObjectProperties => PanelKind::ObjectProperties,
        PanelKindView::RenderSettings => PanelKind::RenderSettings,
        PanelKindView::Scene => PanelKind::Scene,
        PanelKindView::History => PanelKind::History,
        PanelKindView::ReferenceImages => PanelKind::ReferenceImages,
    }
}

fn handle_panel_pointer_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    kind: PanelKind,
    action: PanelPointerAction,
    usable_rect: FloatingPanelBounds,
    drag_last_position: &Rc<RefCell<Option<[f32; 2]>>>,
) {
    match action.phase {
        PanelPointerPhase::Down => {
            let is_visible = host_state
                .app
                .ui
                .panel_framework
                .pinned_instance(kind)
                .is_some()
                || host_state
                    .app
                    .ui
                    .panel_framework
                    .active_transient(PanelBarId::PrimaryRight)
                    == Some(kind);
            if !is_visible {
                return;
            }
            *drag_last_position.borrow_mut() = Some([action.x, action.y]);
            host_state.queue_action(Action::BeginPanelInteraction {
                kind,
                bar_id: PanelBarId::PrimaryRight,
                interaction: panel_pointer_interaction(action.interaction, action.handle),
            });
        }
        PanelPointerPhase::Move => {
            let Some(previous) = *drag_last_position.borrow() else {
                return;
            };
            let delta_x = action.x - previous[0];
            let delta_y = action.y - previous[1];
            *drag_last_position.borrow_mut() = Some([action.x, action.y]);
            if delta_x.abs() <= f32::EPSILON && delta_y.abs() <= f32::EPSILON {
                return;
            }
            host_state.queue_action(Action::UpdatePanelInteraction {
                kind,
                bar_id: PanelBarId::PrimaryRight,
                delta_x,
                delta_y,
                usable_rect,
            });
        }
        PanelPointerPhase::Up => {
            *drag_last_position.borrow_mut() = None;
            host_state.queue_action(Action::EndPanelInteraction {
                kind,
                bar_id: PanelBarId::PrimaryRight,
            });
        }
        PanelPointerPhase::Cancel => {
            *drag_last_position.borrow_mut() = None;
            host_state.queue_action(Action::CancelPanelInteraction {
                kind,
                bar_id: PanelBarId::PrimaryRight,
            });
        }
    }
}

fn panel_pointer_interaction(
    interaction: PanelPointerInteractionKindView,
    handle: PanelResizeHandleView,
) -> PanelPointerInteractionKind {
    match interaction {
        PanelPointerInteractionKindView::Move => PanelPointerInteractionKind::Move,
        PanelPointerInteractionKindView::Resize => {
            PanelPointerInteractionKind::Resize(panel_resize_handle(handle))
        }
    }
}

fn panel_resize_handle(handle: PanelResizeHandleView) -> PanelResizeHandle {
    match handle {
        PanelResizeHandleView::Top => PanelResizeHandle::Top,
        PanelResizeHandleView::Right => PanelResizeHandle::Right,
        PanelResizeHandleView::Bottom => PanelResizeHandle::Bottom,
        PanelResizeHandleView::Left => PanelResizeHandle::Left,
        PanelResizeHandleView::TopLeft => PanelResizeHandle::TopLeft,
        PanelResizeHandleView::TopRight => PanelResizeHandle::TopRight,
        PanelResizeHandleView::BottomLeft => PanelResizeHandle::BottomLeft,
        PanelResizeHandleView::BottomRight => PanelResizeHandle::BottomRight,
        PanelResizeHandleView::None => PanelResizeHandle::Top,
    }
}
