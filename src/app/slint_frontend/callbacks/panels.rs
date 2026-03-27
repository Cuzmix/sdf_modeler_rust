use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{PanelHeaderAction, PanelKindView, SlintHostWindow};
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
}

fn panel_kind(kind: PanelKindView) -> PanelKind {
    match kind {
        PanelKindView::ObjectProperties => PanelKind::ObjectProperties,
        PanelKindView::RenderSettings => PanelKind::RenderSettings,
    }
}
