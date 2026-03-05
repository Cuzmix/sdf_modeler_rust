use crate::graph::scene::NodeId;

/// Maximum number of undo entries to keep per sculpt session.
/// At 96³ resolution, each snapshot is ~3.5 MB, so 50 entries ≈ 175 MB.
const MAX_UNDO_ENTRIES: usize = 50;

/// Per-stroke undo/redo history for sculpt mode.
///
/// Captures a snapshot of the voxel grid data at the start of each stroke
/// (mouse-down → mouse-up). Ctrl+Z undoes one stroke at a time.
pub struct SculptHistory {
    /// Which sculpt node this history tracks. Cleared when switching nodes.
    node_id: Option<NodeId>,
    /// Stack of pre-stroke grid snapshots (oldest at index 0).
    undo_stack: Vec<Vec<f32>>,
    /// Stack of post-undo grid snapshots for redo.
    redo_stack: Vec<Vec<f32>>,
    /// Snapshot captured at the start of the current stroke (before any dabs).
    /// `Some(...)` while a stroke is in progress, `None` between strokes.
    pending_snapshot: Option<Vec<f32>>,
}

impl SculptHistory {
    pub fn new() -> Self {
        Self {
            node_id: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pending_snapshot: None,
        }
    }

    /// Set which sculpt node this history is tracking. Clears all history
    /// if the node changes.
    pub fn set_node(&mut self, id: NodeId) {
        if self.node_id != Some(id) {
            self.clear();
            self.node_id = Some(id);
        }
    }

    /// Called at the start of a stroke (first dab, when `last_sculpt_hit` is None).
    /// Captures a snapshot of the grid data before any modifications.
    pub fn begin_stroke(&mut self, data: &[f32]) {
        // Only capture if we don't already have a pending snapshot
        // (guards against double-calls within the same stroke)
        if self.pending_snapshot.is_none() {
            self.pending_snapshot = Some(data.to_vec());
        }
    }

    /// Called at the end of a stroke (mouse released). Pushes the pending
    /// snapshot onto the undo stack. Any new stroke after an undo clears
    /// the redo stack.
    pub fn end_stroke(&mut self) {
        if let Some(snapshot) = self.pending_snapshot.take() {
            self.undo_stack.push(snapshot);
            // New stroke invalidates redo history
            self.redo_stack.clear();
            // Cap the undo stack
            while self.undo_stack.len() > MAX_UNDO_ENTRIES {
                self.undo_stack.remove(0);
            }
        }
    }

    /// Undo one stroke. Returns the grid data to restore, or None if nothing
    /// to undo. The caller must also provide the *current* grid data so it
    /// can be pushed onto the redo stack.
    pub fn undo(&mut self, current_data: &[f32]) -> Option<Vec<f32>> {
        let snapshot = self.undo_stack.pop()?;
        // Save current state for redo
        self.redo_stack.push(current_data.to_vec());
        Some(snapshot)
    }

    /// Redo one stroke. Returns the grid data to restore, or None if nothing
    /// to redo. The caller must also provide the *current* grid data so it
    /// can be pushed back onto the undo stack.
    pub fn redo(&mut self, current_data: &[f32]) -> Option<Vec<f32>> {
        let snapshot = self.redo_stack.pop()?;
        // Save current state for undo
        self.undo_stack.push(current_data.to_vec());
        Some(snapshot)
    }

    /// Clear all history (called on node switch, scene load, etc.).
    pub fn clear(&mut self) {
        self.node_id = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.pending_snapshot = None;
    }

    /// Whether there are any undo entries available.
    #[allow(dead_code)]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether there are any redo entries available.
    #[allow(dead_code)]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}
