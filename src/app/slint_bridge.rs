use super::actions::{Action, ActionSink};
use super::backend_frame::FrameInputSnapshot;
use super::frontend_models::ShellSnapshot;
use super::viewport_interaction::{PointerButtonSnapshot, ViewportInputSnapshot};
use crate::keymap::KeyboardModifiers;

pub(super) const POINTER_BUTTON_OTHER: i32 = 0;
pub(super) const POINTER_BUTTON_PRIMARY: i32 = 1;
pub(super) const POINTER_BUTTON_SECONDARY: i32 = 2;
pub(super) const POINTER_BUTTON_MIDDLE: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlintUiEvent {
    FrameAll,
    Undo,
    Redo,
    SelectPreviousSceneRow,
    SelectNextSceneRow,
}

#[derive(Clone, Debug)]
pub(super) struct SlintViewportInputState {
    viewport_size_logical: [f32; 2],
    pixels_per_point: f32,
    pointer_inside: bool,
    pointer_position_logical: Option<[f32; 2]>,
    pointer_delta_logical: [f32; 2],
    wheel_delta_logical: [f32; 2],
    primary: PointerButtonSnapshot,
    secondary: PointerButtonSnapshot,
    middle: PointerButtonSnapshot,
    modifiers: KeyboardModifiers,
    pressure: f32,
    double_clicked: bool,
}

impl Default for SlintViewportInputState {
    fn default() -> Self {
        Self {
            viewport_size_logical: [960.0, 540.0],
            pixels_per_point: 1.0,
            pointer_inside: false,
            pointer_position_logical: None,
            pointer_delta_logical: [0.0, 0.0],
            wheel_delta_logical: [0.0, 0.0],
            primary: PointerButtonSnapshot::default(),
            secondary: PointerButtonSnapshot::default(),
            middle: PointerButtonSnapshot::default(),
            modifiers: KeyboardModifiers::default(),
            pressure: 0.0,
            double_clicked: false,
        }
    }
}

impl SlintViewportInputState {
    pub(super) fn set_viewport_geometry(
        &mut self,
        width_logical: f32,
        height_logical: f32,
        pixels_per_point: f32,
    ) {
        self.viewport_size_logical = [width_logical.max(1.0), height_logical.max(1.0)];
        self.pixels_per_point = pixels_per_point.max(1.0);
        if let Some(position) = self.pointer_position_logical {
            self.pointer_inside = self.contains_point(position);
        }
    }

    pub(super) fn handle_pointer_down(
        &mut self,
        x: f32,
        y: f32,
        button: i32,
        modifiers: KeyboardModifiers,
        is_touch: bool,
    ) {
        self.update_pointer_position(x, y);
        self.modifiers = modifiers;
        self.pressure = if is_touch { 1.0 } else { 0.0 };
        match button {
            POINTER_BUTTON_PRIMARY => {
                self.primary.down = true;
                self.primary.pressed = true;
            }
            POINTER_BUTTON_SECONDARY => {
                self.secondary.down = true;
                self.secondary.pressed = true;
            }
            POINTER_BUTTON_MIDDLE => {
                self.middle.down = true;
                self.middle.pressed = true;
            }
            _ => {}
        }
    }

    pub(super) fn handle_pointer_up(
        &mut self,
        x: f32,
        y: f32,
        button: i32,
        modifiers: KeyboardModifiers,
        is_touch: bool,
    ) {
        self.update_pointer_position(x, y);
        self.modifiers = modifiers;
        self.pressure = if is_touch { 1.0 } else { 0.0 };
        match button {
            POINTER_BUTTON_PRIMARY => {
                self.primary.down = false;
                self.primary.released = true;
            }
            POINTER_BUTTON_SECONDARY => {
                self.secondary.down = false;
                self.secondary.released = true;
            }
            POINTER_BUTTON_MIDDLE => {
                self.middle.down = false;
                self.middle.released = true;
            }
            _ => {}
        }
    }

    pub(super) fn handle_pointer_move(
        &mut self,
        x: f32,
        y: f32,
        modifiers: KeyboardModifiers,
        is_touch: bool,
    ) {
        self.update_pointer_position(x, y);
        self.modifiers = modifiers;
        self.pressure = if is_touch { 1.0 } else { 0.0 };
    }

    pub(super) fn handle_pointer_cancel(&mut self) {
        self.pointer_inside = false;
        self.pointer_position_logical = None;
        self.pointer_delta_logical = [0.0, 0.0];
        self.primary.down = false;
        self.primary.released = true;
        self.secondary.down = false;
        self.secondary.released = true;
        self.middle.down = false;
        self.middle.released = true;
        self.pressure = 0.0;
    }

    pub(super) fn handle_pointer_exit(&mut self) {
        self.pointer_inside = false;
        self.pointer_position_logical = None;
    }

    pub(super) fn handle_scroll(
        &mut self,
        delta_x: f32,
        delta_y: f32,
        modifiers: KeyboardModifiers,
    ) {
        self.modifiers = modifiers;
        self.wheel_delta_logical[0] += delta_x;
        self.wheel_delta_logical[1] += delta_y;
    }

    pub(super) fn handle_double_click(&mut self, x: f32, y: f32) {
        self.update_pointer_position(x, y);
        self.double_clicked = true;
    }

    pub(super) fn pointer_primary_down(&self) -> bool {
        self.primary.down
    }

    pub(super) fn needs_continuous_ticks(&self) -> bool {
        self.primary.down || self.secondary.down || self.middle.down
    }

    pub(super) fn take_snapshot(&mut self, now_seconds: f64) -> ViewportInputSnapshot {
        let snapshot = ViewportInputSnapshot {
            viewport_size_physical: [
                (self.viewport_size_logical[0] * self.pixels_per_point).max(1.0) as u32,
                (self.viewport_size_logical[1] * self.pixels_per_point).max(1.0) as u32,
            ],
            pixels_per_point: self.pixels_per_point,
            now_seconds,
            pointer_inside: self.pointer_inside,
            pointer_position_physical: self.pointer_position_logical.map(|position| {
                [
                    position[0] * self.pixels_per_point,
                    position[1] * self.pixels_per_point,
                ]
            }),
            pointer_delta_physical: [
                self.pointer_delta_logical[0] * self.pixels_per_point,
                self.pointer_delta_logical[1] * self.pixels_per_point,
            ],
            wheel_delta_logical: self.wheel_delta_logical,
            primary: self.primary,
            secondary: self.secondary,
            middle: self.middle,
            modifiers: self.modifiers,
            pressure: self.pressure,
            double_clicked: self.double_clicked,
        };
        self.pointer_delta_logical = [0.0, 0.0];
        self.wheel_delta_logical = [0.0, 0.0];
        self.primary.pressed = false;
        self.primary.released = false;
        self.secondary.pressed = false;
        self.secondary.released = false;
        self.middle.pressed = false;
        self.middle.released = false;
        self.double_clicked = false;
        if !self.primary.down && !self.secondary.down && !self.middle.down {
            self.pressure = 0.0;
        }
        snapshot
    }

    fn update_pointer_position(&mut self, x: f32, y: f32) {
        let next = [x, y];
        if let Some(previous) = self.pointer_position_logical {
            self.pointer_delta_logical[0] += next[0] - previous[0];
            self.pointer_delta_logical[1] += next[1] - previous[1];
        }
        self.pointer_position_logical = Some(next);
        self.pointer_inside = self.contains_point(next);
    }

    fn contains_point(&self, position: [f32; 2]) -> bool {
        position[0] >= 0.0
            && position[1] >= 0.0
            && position[0] <= self.viewport_size_logical[0]
            && position[1] <= self.viewport_size_logical[1]
    }
}

pub(super) fn capture_frame_input(
    now_seconds: f64,
    viewport_state: &SlintViewportInputState,
) -> FrameInputSnapshot {
    FrameInputSnapshot {
        now_seconds,
        pointer_primary_down: viewport_state.pointer_primary_down(),
        is_dragging_ui: false,
    }
}

pub(super) fn dispatch_event(
    event: SlintUiEvent,
    snapshot: Option<&ShellSnapshot>,
    actions: &mut ActionSink,
) {
    match event {
        SlintUiEvent::FrameAll => actions.push(Action::FrameAll),
        SlintUiEvent::Undo => actions.push(Action::Undo),
        SlintUiEvent::Redo => actions.push(Action::Redo),
        SlintUiEvent::SelectPreviousSceneRow => {
            let Some(shell) = snapshot else {
                return;
            };
            let row_count = shell.scene_panel.rows.len();
            if row_count == 0 {
                return;
            }
            let current_index = selected_scene_index(shell).unwrap_or(0);
            let target_index = if current_index == 0 {
                row_count - 1
            } else {
                current_index - 1
            };
            actions.push(Action::Select(Some(
                shell.scene_panel.rows[target_index].host_id,
            )));
        }
        SlintUiEvent::SelectNextSceneRow => {
            let Some(shell) = snapshot else {
                return;
            };
            let row_count = shell.scene_panel.rows.len();
            if row_count == 0 {
                return;
            }
            let current_index = selected_scene_index(shell).unwrap_or(row_count - 1);
            let target_index = (current_index + 1) % row_count;
            actions.push(Action::Select(Some(
                shell.scene_panel.rows[target_index].host_id,
            )));
        }
    }
}

fn selected_scene_index(snapshot: &ShellSnapshot) -> Option<usize> {
    let selected_host = snapshot.scene_panel.selected_host?;
    snapshot
        .scene_panel
        .rows
        .iter()
        .position(|row| row.host_id == selected_host)
}
