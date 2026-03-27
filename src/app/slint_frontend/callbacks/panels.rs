use std::cell::RefCell;
use std::rc::Rc;

use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{
    PanelDragAction, PanelDragPhase, PanelHeaderAction, PanelKindView, SlintHostWindow,
};
use crate::app::state::{PanelBarId, PanelKind};

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
    window.on_panel_drag_action(move |kind, action| {
        let viewport_size = drag_context
            .window_weak
            .upgrade()
            .map(|window| [window.get_viewport_width(), window.get_viewport_height()])
            .unwrap_or([960.0, 540.0]);
        let drag_last_position = drag_last_position_for_callback.clone();
        mutate_host_and_tick(&drag_context, move |host_state| {
            handle_panel_drag_action(
                host_state,
                panel_kind(kind),
                action,
                viewport_size,
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
    }
}

fn handle_panel_drag_action(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    kind: PanelKind,
    action: PanelDragAction,
    viewport_size: [f32; 2],
    drag_last_position: &Rc<RefCell<Option<[f32; 2]>>>,
) {
    match action.phase {
        PanelDragPhase::Down => {
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
            host_state.queue_action(Action::BeginPanelDrag {
                kind,
                bar_id: PanelBarId::PrimaryRight,
            });
        }
        PanelDragPhase::Move => {
            let Some(previous) = *drag_last_position.borrow() else {
                return;
            };
            let delta_x = action.x - previous[0];
            let delta_y = action.y - previous[1];
            *drag_last_position.borrow_mut() = Some([action.x, action.y]);
            if delta_x.abs() <= f32::EPSILON && delta_y.abs() <= f32::EPSILON {
                return;
            }
            host_state.queue_action(Action::DragPanel {
                kind,
                bar_id: PanelBarId::PrimaryRight,
                delta_x,
                delta_y,
                viewport_width: viewport_size[0],
                viewport_height: viewport_size[1],
            });
        }
        PanelDragPhase::Up => {
            *drag_last_position.borrow_mut() = None;
            host_state.queue_action(Action::EndPanelDrag {
                kind,
                bar_id: PanelBarId::PrimaryRight,
            });
        }
        PanelDragPhase::Cancel => {
            *drag_last_position.borrow_mut() = None;
            host_state.queue_action(Action::CancelPanelDrag {
                kind,
                bar_id: PanelBarId::PrimaryRight,
            });
        }
    }
}
