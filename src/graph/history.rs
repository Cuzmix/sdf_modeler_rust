use super::scene::{NodeId, Scene};

const MAX_UNDO_DEPTH: usize = 50;

#[derive(Clone)]
struct Snapshot {
    scene: Scene,
    selected: Option<NodeId>,
}

pub struct History {
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,
    pending_snapshot: Option<Snapshot>,
    was_dragging: bool,
}

impl History {
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pending_snapshot: None,
            was_dragging: false,
        }
    }

    /// Call at the START of each frame. Captures the "before" state.
    pub fn begin_frame(&mut self, scene: &Scene, selected: Option<NodeId>) {
        if self.pending_snapshot.is_none() {
            self.pending_snapshot = Some(Snapshot {
                scene: scene.clone(),
                selected,
            });
        }
    }

    /// Call at the END of each frame. If scene changed and user is not
    /// mid-drag, commit the pending snapshot to the undo stack.
    pub fn end_frame(
        &mut self,
        scene: &Scene,
        selected: Option<NodeId>,
        is_anything_dragged: bool,
    ) {
        let drag_just_ended = self.was_dragging && !is_anything_dragged;
        self.was_dragging = is_anything_dragged;

        if let Some(ref snapshot) = self.pending_snapshot {
            let changed =
                !scene.content_eq(&snapshot.scene) || selected != snapshot.selected;

            if changed && (!is_anything_dragged || drag_just_ended) {
                self.push_undo(snapshot.clone());
                self.redo_stack.clear();
                self.pending_snapshot = None;
            } else if !changed && !is_anything_dragged {
                // No change, refresh snapshot for next mutation window
                self.pending_snapshot = None;
            }
        }
    }

    fn push_undo(&mut self, snapshot: Snapshot) {
        self.undo_stack.push(snapshot);
        if self.undo_stack.len() > MAX_UNDO_DEPTH {
            self.undo_stack.remove(0);
        }
    }

    /// Undo: restore top of undo stack, push current state to redo.
    pub fn undo(
        &mut self,
        current_scene: &Scene,
        current_selected: Option<NodeId>,
    ) -> Option<(Scene, Option<NodeId>)> {
        let snapshot = self.undo_stack.pop()?;
        self.redo_stack.push(Snapshot {
            scene: current_scene.clone(),
            selected: current_selected,
        });
        self.pending_snapshot = None;
        Some((snapshot.scene, snapshot.selected))
    }

    /// Redo: restore top of redo stack, push current state to undo.
    pub fn redo(
        &mut self,
        current_scene: &Scene,
        current_selected: Option<NodeId>,
    ) -> Option<(Scene, Option<NodeId>)> {
        let snapshot = self.redo_stack.pop()?;
        self.undo_stack.push(Snapshot {
            scene: current_scene.clone(),
            selected: current_selected,
        });
        self.pending_snapshot = None;
        Some((snapshot.scene, snapshot.selected))
    }
}
