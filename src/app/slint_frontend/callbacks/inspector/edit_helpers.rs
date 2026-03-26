use super::super::context::CallbackContext;
use super::super::vector_axes::axis_value;
use crate::app::slint_frontend::InspectorEditMode;
use crate::app::slint_frontend::SlintHostWindow;

pub(super) fn apply_vector_edit<ReadCurrent, Apply>(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    axis: i32,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
    apply: Apply,
) where
    ReadCurrent: Fn(&SlintHostWindow) -> [f32; 3],
    Apply: Fn(&mut crate::app::SdfApp, usize, f32),
{
    let Some(window) = context.window_weak.upgrade() else {
        return;
    };
    let component = axis.max(0) as usize;
    let current = axis_value(axis, read_current(&window));
    let next = match mode {
        InspectorEditMode::Nudge => current + value,
        InspectorEditMode::Set | InspectorEditMode::Toggle => value,
    };
    apply(&mut host_state.app, component, next);
}

pub(super) fn apply_scalar_edit<ReadCurrent, Apply>(
    host_state: &mut crate::app::slint_frontend::host_state::SlintHostState,
    context: &CallbackContext,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
    apply: Apply,
) where
    ReadCurrent: Fn(&SlintHostWindow) -> f32,
    Apply: Fn(&mut crate::app::SdfApp, f32),
{
    let next = apply_scalar_value(context, mode, value, read_current);
    apply(&mut host_state.app, next);
}

pub(super) fn apply_scalar_value<ReadCurrent>(
    context: &CallbackContext,
    mode: InspectorEditMode,
    value: f32,
    read_current: ReadCurrent,
) -> f32
where
    ReadCurrent: Fn(&SlintHostWindow) -> f32,
{
    let Some(window) = context.window_weak.upgrade() else {
        return value;
    };
    match mode {
        InspectorEditMode::Nudge => read_current(&window) + value,
        InspectorEditMode::Set | InspectorEditMode::Toggle => value,
    }
}
