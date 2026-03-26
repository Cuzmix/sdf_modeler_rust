use super::context::CallbackContext;
use super::mutation::mutate_host_and_tick;
use crate::app::actions::Action;
use crate::app::slint_frontend::{SlintHostWindow, WorkspaceAction};
use crate::app::state::ExpertPanelKind;

pub(super) fn install(window: &SlintHostWindow, context: &CallbackContext) {
    let context = context.clone();
    window.on_workspace_action(move |action| {
        mutate_host_and_tick(&context, move |host_state| match action {
            WorkspaceAction::ShowNodeGraph => {
                host_state.queue_action(Action::ToggleExpertPanel(ExpertPanelKind::NodeGraph));
            }
            WorkspaceAction::ShowLightGraph => {
                host_state.queue_action(Action::ToggleExpertPanel(ExpertPanelKind::LightGraph));
            }
            WorkspaceAction::CloseWorkspace => {
                host_state.queue_action(Action::HideShellPanel(
                    crate::app::state::ShellPanelKind::Drawer,
                ));
            }
            WorkspaceAction::FocusSelected => {
                host_state.queue_action(Action::FocusSelected);
            }
            WorkspaceAction::FrameAll => {
                host_state.queue_action(Action::FrameAll);
            }
        });
    });
}
