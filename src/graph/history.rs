use super::scene::{NodeId, Scene};

const MAX_UNDO_DEPTH: usize = 50;

#[derive(Clone)]
struct Snapshot {
    scene: Scene,
    selected: Option<NodeId>,
    label: String,
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

    /// Get labels of the undo stack (oldest first).
    pub fn undo_labels(&self) -> Vec<String> {
        self.undo_stack.iter().map(|s| s.label.clone()).collect()
    }

    /// Get labels of the redo stack (next-to-redo first).
    pub fn redo_labels(&self) -> Vec<String> {
        self.redo_stack.iter().rev().map(|s| s.label.clone()).collect()
    }

    pub fn undo_count(&self) -> usize { self.undo_stack.len() }
    pub fn redo_count(&self) -> usize { self.redo_stack.len() }

    /// Call at the START of each frame. Captures the "before" state.
    pub fn begin_frame(&mut self, scene: &Scene, selected: Option<NodeId>) {
        if self.pending_snapshot.is_none() {
            self.pending_snapshot = Some(Snapshot {
                scene: scene.clone(),
                selected,
                label: String::new(),
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
                // Auto-detect a label based on what changed
                let label = Self::detect_change_label(&snapshot.scene, scene, snapshot.selected, selected);
                let mut snap = snapshot.clone();
                snap.label = label;
                self.push_undo(snap);
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
            label: "Undo".into(),
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
            label: "Redo".into(),
        });
        self.pending_snapshot = None;
        Some((snapshot.scene, snapshot.selected))
    }

    /// Detect what kind of change happened between two scenes.
    fn detect_change_label(old: &Scene, new: &Scene, old_sel: Option<NodeId>, new_sel: Option<NodeId>) -> String {
        let old_count = old.nodes.len();
        let new_count = new.nodes.len();

        if new_count > old_count {
            // Node(s) added
            for (&id, node) in &new.nodes {
                if !old.nodes.contains_key(&id) {
                    return format!("Add {}", node.name);
                }
            }
            return "Add node".into();
        }

        if new_count < old_count {
            // Node(s) removed
            for (&id, node) in &old.nodes {
                if !new.nodes.contains_key(&id) {
                    return format!("Delete {}", node.name);
                }
            }
            return "Delete node".into();
        }

        // Same count — check for property/structure changes
        if old_sel != new_sel {
            return "Select".into();
        }

        // Check if structure changed (parent/child connections)
        if old.structure_key() != new.structure_key() {
            return "Restructure".into();
        }

        "Edit".into()
    }
}
