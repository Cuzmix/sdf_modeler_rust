use super::scene::{NodeData, NodeId, Scene};

const MAX_UNDO_DEPTH: usize = 50;

#[derive(Clone)]
enum SnapshotState {
    Scene(Scene),
    Sculpt {
        node_id: NodeId,
        voxel_data: Vec<f32>,
    },
}

#[derive(Clone)]
struct Snapshot {
    state: SnapshotState,
    selected: Option<NodeId>,
    label: String,
}

impl Snapshot {
    fn scene(scene: &Scene, selected: Option<NodeId>, label: impl Into<String>) -> Self {
        Self {
            state: SnapshotState::Scene(scene.clone()),
            selected,
            label: label.into(),
        }
    }

    fn sculpt(
        node_id: NodeId,
        voxel_data: &[f32],
        selected: Option<NodeId>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            state: SnapshotState::Sculpt {
                node_id,
                voxel_data: voxel_data.to_vec(),
            },
            selected,
            label: label.into(),
        }
    }

    fn into_restore(self) -> RestoreSnapshot {
        let state = match self.state {
            SnapshotState::Scene(scene) => RestoreState::Scene(scene),
            SnapshotState::Sculpt {
                node_id,
                voxel_data,
            } => RestoreState::Sculpt {
                node_id,
                voxel_data,
            },
        };
        RestoreSnapshot {
            state,
            selected: self.selected,
        }
    }
}

pub enum RestoreState {
    Scene(Scene),
    Sculpt {
        node_id: NodeId,
        voxel_data: Vec<f32>,
    },
}

pub struct RestoreSnapshot {
    pub state: RestoreState,
    pub selected: Option<NodeId>,
}

pub struct History {
    undo_stack: Vec<Snapshot>,
    redo_stack: Vec<Snapshot>,
    pending_snapshot: Option<Snapshot>,
    pending_sculpt_snapshot: Option<Snapshot>,
    was_dragging: bool,
}

impl History {
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pending_snapshot: None,
            pending_sculpt_snapshot: None,
            was_dragging: false,
        }
    }

    /// Get labels of the undo stack (oldest first).
    pub fn undo_labels(&self) -> Vec<String> {
        self.undo_stack.iter().map(|s| s.label.clone()).collect()
    }

    /// Get labels of the redo stack (next-to-redo first).
    pub fn redo_labels(&self) -> Vec<String> {
        self.redo_stack
            .iter()
            .rev()
            .map(|s| s.label.clone())
            .collect()
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Call at the START of each frame. Captures the "before" state.
    pub fn begin_frame(&mut self, scene: &Scene, selected: Option<NodeId>) {
        if self.pending_snapshot.is_none() {
            self.pending_snapshot = Some(Snapshot::scene(scene, selected, String::new()));
        }
    }

    pub fn begin_sculpt_stroke(
        &mut self,
        scene: &Scene,
        node_id: NodeId,
        selected: Option<NodeId>,
        label: impl Into<String>,
    ) {
        if self.pending_sculpt_snapshot.is_some() {
            return;
        }
        if let Some(snapshot) = Self::capture_sculpt_snapshot(scene, node_id, selected, label) {
            self.pending_sculpt_snapshot = Some(snapshot);
        }
    }

    pub fn end_sculpt_stroke(&mut self, scene: &Scene) {
        let Some(snapshot) = self.pending_sculpt_snapshot.take() else {
            return;
        };
        let SnapshotState::Sculpt {
            node_id,
            voxel_data: before_data,
        } = &snapshot.state
        else {
            return;
        };
        let Some(current_data) = Self::capture_sculpt_data(scene, *node_id) else {
            self.pending_snapshot = None;
            return;
        };
        if !Self::voxel_data_eq(before_data, &current_data) {
            self.push_undo(snapshot);
            self.redo_stack.clear();
        }
        // Scene history should not also commit a full-scene snapshot for this stroke.
        self.pending_snapshot = None;
    }

    pub fn discard_pending_sculpt_stroke(&mut self) {
        self.pending_sculpt_snapshot = None;
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

        if self.pending_sculpt_snapshot.is_some() {
            return;
        }

        if let Some(ref snapshot) = self.pending_snapshot {
            let SnapshotState::Scene(before_scene) = &snapshot.state else {
                return;
            };
            let changed = !scene.content_eq(before_scene) || selected != snapshot.selected;

            if changed && (!is_anything_dragged || drag_just_ended) {
                // Auto-detect a label based on what changed
                let label =
                    Self::detect_change_label(before_scene, scene, snapshot.selected, selected);
                self.push_undo(Snapshot::scene(before_scene, snapshot.selected, label));
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
    ) -> Option<RestoreSnapshot> {
        let snapshot = self.undo_stack.pop()?;
        let redo_snapshot = Self::capture_current_snapshot_for(
            &snapshot.state,
            current_scene,
            current_selected,
            snapshot.label.clone(),
        )?;
        self.redo_stack.push(redo_snapshot);
        self.pending_snapshot = None;
        self.pending_sculpt_snapshot = None;
        Some(snapshot.into_restore())
    }

    /// Redo: restore top of redo stack, push current state to undo.
    pub fn redo(
        &mut self,
        current_scene: &Scene,
        current_selected: Option<NodeId>,
    ) -> Option<RestoreSnapshot> {
        let snapshot = self.redo_stack.pop()?;
        let undo_snapshot = Self::capture_current_snapshot_for(
            &snapshot.state,
            current_scene,
            current_selected,
            snapshot.label.clone(),
        )?;
        self.undo_stack.push(undo_snapshot);
        self.pending_snapshot = None;
        self.pending_sculpt_snapshot = None;
        Some(snapshot.into_restore())
    }

    fn capture_current_snapshot_for(
        state: &SnapshotState,
        current_scene: &Scene,
        current_selected: Option<NodeId>,
        label: impl Into<String>,
    ) -> Option<Snapshot> {
        let label = label.into();
        match state {
            SnapshotState::Scene(_) => {
                Some(Snapshot::scene(current_scene, current_selected, label))
            }
            SnapshotState::Sculpt { node_id, .. } => Some(Snapshot::sculpt(
                *node_id,
                &Self::capture_sculpt_data(current_scene, *node_id)?,
                current_selected,
                label,
            )),
        }
    }

    fn capture_sculpt_snapshot(
        scene: &Scene,
        node_id: NodeId,
        selected: Option<NodeId>,
        label: impl Into<String>,
    ) -> Option<Snapshot> {
        Some(Snapshot::sculpt(
            node_id,
            &Self::capture_sculpt_data(scene, node_id)?,
            selected,
            label,
        ))
    }

    fn capture_sculpt_data(scene: &Scene, node_id: NodeId) -> Option<Vec<f32>> {
        scene.nodes.get(&node_id).and_then(|node| match &node.data {
            NodeData::Sculpt { voxel_grid, .. } => Some(voxel_grid.data.clone()),
            _ => None,
        })
    }

    fn voxel_data_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }

    /// Detect what kind of change happened between two scenes.
    fn detect_change_label(
        old: &Scene,
        new: &Scene,
        old_sel: Option<NodeId>,
        new_sel: Option<NodeId>,
    ) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{NodeData, Scene, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
    use glam::Vec3;

    /// Find the first Primitive node ID in a scene.
    fn find_primitive_id(scene: &Scene) -> u64 {
        *scene
            .nodes
            .iter()
            .find(|(_, n)| matches!(n.data, NodeData::Primitive { .. }))
            .expect("scene should have a primitive")
            .0
    }

    /// Create an empty scene (no default sphere) for predictable testing.
    fn empty_scene() -> Scene {
        Scene {
            nodes: std::collections::HashMap::new(),
            next_id: 0,
            name_counters: std::collections::HashMap::new(),
            hidden_nodes: std::collections::HashSet::new(),
            light_masks: std::collections::HashMap::new(),
        }
    }

    fn unwrap_scene_restore(restore: RestoreSnapshot) -> (Scene, Option<NodeId>) {
        match restore.state {
            RestoreState::Scene(scene) => (scene, restore.selected),
            RestoreState::Sculpt { .. } => panic!("expected a scene restore"),
        }
    }

    fn unwrap_sculpt_restore(restore: RestoreSnapshot) -> (NodeId, Vec<f32>, Option<NodeId>) {
        match restore.state {
            RestoreState::Sculpt {
                node_id,
                voxel_data,
            } => (node_id, voxel_data, restore.selected),
            RestoreState::Scene(_) => panic!("expected a sculpt restore"),
        }
    }

    fn sculpt_scene() -> (Scene, NodeId) {
        let mut scene = empty_scene();
        let input_id = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt_id = scene.create_sculpt(
            input_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::splat(0.5),
            VoxelGrid::new_displacement(4, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );
        (scene, sculpt_id)
    }

    #[test]
    fn new_history_is_empty() {
        let history = History::new();
        assert_eq!(history.undo_count(), 0);
        assert_eq!(history.redo_count(), 0);
        assert!(history.undo_labels().is_empty());
        assert!(history.redo_labels().is_empty());
    }

    #[test]
    fn push_and_undo_restores_previous_state() {
        let mut history = History::new();
        let scene_before = empty_scene();
        let mut scene_after = empty_scene();
        scene_after.create_primitive(SdfPrimitive::Sphere);

        // Simulate a frame where a node is added
        history.begin_frame(&scene_before, None);
        history.end_frame(&scene_after, None, false);

        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);

        // Undo should restore the empty scene
        let result = unwrap_scene_restore(history.undo(&scene_after, None).unwrap());
        let (restored_scene, restored_selected) = result;
        assert_eq!(restored_scene.nodes.len(), 0);
        assert_eq!(restored_selected, None);
        assert_eq!(history.undo_count(), 0);
        assert_eq!(history.redo_count(), 1);
    }

    #[test]
    fn redo_restores_undone_state() {
        let mut history = History::new();
        let scene_before = empty_scene();
        let mut scene_after = empty_scene();
        scene_after.create_primitive(SdfPrimitive::Box);

        // Push a change
        history.begin_frame(&scene_before, None);
        history.end_frame(&scene_after, None, false);

        // Undo it
        let (restored, _) = unwrap_scene_restore(history.undo(&scene_after, None).unwrap());
        assert_eq!(restored.nodes.len(), 0);

        // Redo should bring back the scene with the box
        let (redone_scene, _) = unwrap_scene_restore(history.redo(&restored, None).unwrap());
        assert_eq!(redone_scene.nodes.len(), 1);
        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn undo_on_empty_returns_none() {
        let mut history = History::new();
        let scene = empty_scene();
        assert!(history.undo(&scene, None).is_none());
    }

    #[test]
    fn redo_on_empty_returns_none() {
        let mut history = History::new();
        let scene = empty_scene();
        assert!(history.redo(&scene, None).is_none());
    }

    #[test]
    fn new_change_clears_redo_stack() {
        let mut history = History::new();
        let scene_v0 = empty_scene();
        let mut scene_v1 = empty_scene();
        scene_v1.create_primitive(SdfPrimitive::Sphere);
        let mut scene_v2 = scene_v1.clone();
        scene_v2.create_primitive(SdfPrimitive::Box);

        // Push two changes
        history.begin_frame(&scene_v0, None);
        history.end_frame(&scene_v1, None, false);
        history.begin_frame(&scene_v1, None);
        history.end_frame(&scene_v2, None, false);
        assert_eq!(history.undo_count(), 2);

        // Undo one
        let (restored, _) = unwrap_scene_restore(history.undo(&scene_v2, None).unwrap());
        assert_eq!(history.redo_count(), 1);

        // Make a new change from the restored state — redo should be cleared
        let mut scene_v3 = restored.clone();
        scene_v3.create_primitive(SdfPrimitive::Cylinder);
        history.begin_frame(&restored, None);
        history.end_frame(&scene_v3, None, false);

        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn capacity_limit_evicts_oldest() {
        let mut history = History::new();

        // Push MAX_UNDO_DEPTH + 5 changes to exceed capacity
        for i in 0..(MAX_UNDO_DEPTH + 5) {
            let mut scene_before = empty_scene();
            scene_before.next_id = i as u64;
            let mut scene_after = empty_scene();
            scene_after.next_id = (i + 1) as u64;
            scene_after.create_primitive(SdfPrimitive::Sphere);

            history.begin_frame(&scene_before, None);
            history.end_frame(&scene_after, None, false);
        }

        assert_eq!(history.undo_count(), MAX_UNDO_DEPTH);
    }

    #[test]
    fn no_snapshot_during_drag() {
        let mut history = History::new();
        let scene_before = empty_scene();
        let mut scene_during_drag = empty_scene();
        scene_during_drag.create_primitive(SdfPrimitive::Sphere);

        // Begin frame, then end while dragging — should NOT push undo
        history.begin_frame(&scene_before, None);
        history.end_frame(&scene_during_drag, None, true);
        assert_eq!(history.undo_count(), 0);
    }

    #[test]
    fn snapshot_committed_when_drag_ends() {
        let mut history = History::new();
        let scene_before = empty_scene();
        let mut scene_after = empty_scene();
        scene_after.create_primitive(SdfPrimitive::Sphere);

        // Frame 1: start dragging (change detected but dragging)
        history.begin_frame(&scene_before, None);
        history.end_frame(&scene_after, None, true);
        assert_eq!(history.undo_count(), 0);

        // Frame 2: drag ends (was_dragging=true, is_anything_dragged=false)
        history.begin_frame(&scene_after, None);
        // The pending snapshot was consumed or refreshed; simulate drag-end
        // by calling end_frame with the changed scene but dragging=false.
        // We need to re-inject the pre-drag state as pending snapshot.
        // Actually, let's test the real flow more carefully:
        // After frame 1, pending_snapshot is still Some (change + dragging keeps it).
        // Frame 2 begin_frame won't overwrite because pending is still Some.
        // Frame 2 end_frame with drag_just_ended should commit.
        history.end_frame(&scene_after, None, false);
        assert_eq!(history.undo_count(), 1);
    }

    #[test]
    fn selection_change_tracked() {
        let mut history = History::new();
        let scene = Scene::new(); // Has one sphere with id=0

        // Change selection
        history.begin_frame(&scene, None);
        history.end_frame(&scene, Some(0), false);
        assert_eq!(history.undo_count(), 1);

        // Undo should restore no selection
        let (_, restored_sel) = unwrap_scene_restore(history.undo(&scene, Some(0)).unwrap());
        assert_eq!(restored_sel, None);
    }

    #[test]
    fn undo_redo_labels_ordered_correctly() {
        let mut history = History::new();
        let scene_v0 = empty_scene();
        let mut scene_v1 = empty_scene();
        scene_v1.create_primitive(SdfPrimitive::Sphere);
        let mut scene_v2 = scene_v1.clone();
        scene_v2.create_primitive(SdfPrimitive::Box);

        // Push two changes
        history.begin_frame(&scene_v0, None);
        history.end_frame(&scene_v1, None, false);
        history.begin_frame(&scene_v1, None);
        history.end_frame(&scene_v2, None, false);

        let undo_labels = history.undo_labels();
        assert_eq!(undo_labels.len(), 2);
        assert_eq!(undo_labels[0], "Add Sphere");
        assert_eq!(undo_labels[1], "Add Box");

        // Undo one — should appear in redo labels
        history.undo(&scene_v2, None);
        let redo_labels = history.redo_labels();
        assert_eq!(redo_labels.len(), 1);
    }

    #[test]
    fn detect_label_add_node() {
        let scene_before = empty_scene();
        let mut scene_after = empty_scene();
        scene_after.create_primitive(SdfPrimitive::Cylinder);

        let label = History::detect_change_label(&scene_before, &scene_after, None, None);
        assert_eq!(label, "Add Cylinder");
    }

    #[test]
    fn detect_label_delete_node() {
        let mut scene_before = empty_scene();
        let node_id = scene_before.create_primitive(SdfPrimitive::Torus);
        let mut scene_after = scene_before.clone();
        scene_after.remove_node(node_id);

        let label = History::detect_change_label(&scene_before, &scene_after, None, None);
        assert_eq!(label, "Delete Torus");
    }

    #[test]
    fn detect_label_selection_change() {
        let scene = Scene::new();
        let label = History::detect_change_label(&scene, &scene, None, Some(0));
        assert_eq!(label, "Select");
    }

    #[test]
    fn detect_label_edit_fallback() {
        let scene = Scene::new();
        let mut modified = scene.clone();
        let prim_id = find_primitive_id(&scene);
        // Change a property without adding/removing nodes or changing selection
        if let Some(node) = modified.nodes.get_mut(&prim_id) {
            if let crate::graph::scene::NodeData::Primitive {
                ref mut material, ..
            } = node.data
            {
                material.base_color = glam::Vec3::new(1.0, 0.0, 0.0);
            }
        }
        let label = History::detect_change_label(&scene, &modified, None, None);
        assert_eq!(label, "Edit");
    }

    #[test]
    fn multiple_undo_redo_roundtrip() {
        let mut history = History::new();
        let scene_v0 = empty_scene();
        let mut scene_v1 = empty_scene();
        let id1 = scene_v1.create_primitive(SdfPrimitive::Sphere);
        let mut scene_v2 = scene_v1.clone();
        scene_v2.create_primitive(SdfPrimitive::Box);

        // Push v0→v1, v1→v2
        history.begin_frame(&scene_v0, None);
        history.end_frame(&scene_v1, None, false);
        history.begin_frame(&scene_v1, Some(id1));
        history.end_frame(&scene_v2, Some(id1), false);

        // Undo twice, redo twice should end at original state
        let (s1, sel1) = unwrap_scene_restore(history.undo(&scene_v2, Some(id1)).unwrap());
        assert_eq!(s1.nodes.len(), 1);
        let (s0, sel0) = unwrap_scene_restore(history.undo(&s1, sel1).unwrap());
        assert_eq!(s0.nodes.len(), 0);
        assert_eq!(sel0, None);

        let (r1, _) = unwrap_scene_restore(history.redo(&s0, sel0).unwrap());
        assert_eq!(r1.nodes.len(), 1);
        let (r2, _) = unwrap_scene_restore(history.redo(&r1, sel1).unwrap());
        assert_eq!(r2.nodes.len(), 2);
    }

    #[test]
    fn sculpt_stroke_joins_global_undo_timeline() {
        let mut history = History::new();
        let (mut scene, sculpt_id) = sculpt_scene();

        history.begin_sculpt_stroke(&scene, sculpt_id, Some(sculpt_id), "Sculpt Add");
        if let Some(node) = scene.nodes.get_mut(&sculpt_id) {
            if let NodeData::Sculpt {
                ref mut voxel_grid, ..
            } = node.data
            {
                voxel_grid.data[0] = -0.5;
            }
        }
        history.end_sculpt_stroke(&scene);

        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.undo_labels(), vec!["Sculpt Add".to_string()]);

        let (restored_id, restored_data, restored_sel) =
            unwrap_sculpt_restore(history.undo(&scene, Some(sculpt_id)).unwrap());
        assert_eq!(restored_id, sculpt_id);
        assert_eq!(restored_sel, Some(sculpt_id));
        assert_eq!(restored_data[0], 0.0);

        let mut restored_scene = scene.clone();
        if let Some(node) = restored_scene.nodes.get_mut(&sculpt_id) {
            if let NodeData::Sculpt {
                ref mut voxel_grid, ..
            } = node.data
            {
                voxel_grid.data = restored_data;
            }
        }

        let (_, redone_data, _) =
            unwrap_sculpt_restore(history.redo(&restored_scene, Some(sculpt_id)).unwrap());
        assert_eq!(redone_data[0], -0.5);
    }

    // ── Property edit undo coalescing ────────────────────────────────

    /// Simulate a multi-frame slider drag editing a primitive's position.
    /// The entire drag should produce exactly ONE undo entry.
    #[test]
    fn property_slider_drag_coalesced_to_single_undo() {
        let mut history = History::new();
        let original_scene = Scene::new();
        let node_id = find_primitive_id(&original_scene);

        // Frame 0: no changes yet
        history.begin_frame(&original_scene, Some(node_id));
        history.end_frame(&original_scene, Some(node_id), false);
        assert_eq!(history.undo_count(), 0);

        // Frame 1: drag starts — slider changes position to 1.0
        history.begin_frame(&original_scene, Some(node_id));
        let mut scene_mid_drag = original_scene.clone();
        if let Some(n) = scene_mid_drag.nodes.get_mut(&node_id) {
            if let NodeData::Primitive {
                ref mut position, ..
            } = n.data
            {
                position.x = 1.0;
            }
        }
        // is_anything_dragged=true (slider widget being dragged)
        history.end_frame(&scene_mid_drag, Some(node_id), true);
        assert_eq!(history.undo_count(), 0, "should not commit during drag");

        // Frame 2: drag continues — position moves to 3.0
        history.begin_frame(&scene_mid_drag, Some(node_id));
        let mut scene_drag_2 = scene_mid_drag.clone();
        if let Some(n) = scene_drag_2.nodes.get_mut(&node_id) {
            if let NodeData::Primitive {
                ref mut position, ..
            } = n.data
            {
                position.x = 3.0;
            }
        }
        history.end_frame(&scene_drag_2, Some(node_id), true);
        assert_eq!(history.undo_count(), 0, "should not commit during drag");

        // Frame 3: drag ends — mouse released, position stays at 3.0
        history.begin_frame(&scene_drag_2, Some(node_id));
        history.end_frame(&scene_drag_2, Some(node_id), false);
        assert_eq!(
            history.undo_count(),
            1,
            "should commit exactly one entry on drag end"
        );

        // Undo should restore to original position (0.0)
        let (restored, _) =
            unwrap_scene_restore(history.undo(&scene_drag_2, Some(node_id)).unwrap());
        let restored_pos = match &restored.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        };
        assert!(
            (restored_pos.x - 0.0).abs() < 0.001,
            "undo should restore position to original (0.0), got {}",
            restored_pos.x
        );
    }

    /// Verify that a gizmo drag (modifying transform) over multiple frames
    /// produces exactly one undo entry when the drag ends.
    #[test]
    fn gizmo_drag_coalesced_to_single_undo() {
        let mut history = History::new();
        let original_scene = Scene::new();
        let node_id = find_primitive_id(&original_scene);

        // Frame 0: idle
        history.begin_frame(&original_scene, Some(node_id));
        history.end_frame(&original_scene, Some(node_id), false);

        // Frames 1-5: gizmo drag moves position incrementally
        let mut scene = original_scene.clone();
        for frame in 1..=5 {
            history.begin_frame(&scene, Some(node_id));
            if let Some(n) = scene.nodes.get_mut(&node_id) {
                if let NodeData::Primitive {
                    ref mut position, ..
                } = n.data
                {
                    position.y = frame as f32 * 0.5;
                }
            }
            // Viewport response is being dragged → is_anything_dragged=true
            history.end_frame(&scene, Some(node_id), true);
        }
        assert_eq!(history.undo_count(), 0, "no commits during gizmo drag");

        // Frame 6: drag ends
        history.begin_frame(&scene, Some(node_id));
        history.end_frame(&scene, Some(node_id), false);
        assert_eq!(
            history.undo_count(),
            1,
            "exactly one entry after gizmo drag"
        );

        // Undo restores to original
        let (restored, _) = unwrap_scene_restore(history.undo(&scene, Some(node_id)).unwrap());
        let restored_pos = match &restored.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        };
        assert!(
            restored_pos.y.abs() < 0.001,
            "undo should restore y to 0.0, got {}",
            restored_pos.y
        );
    }

    /// Verify that a color edit (no drag, instant click) creates one undo entry.
    #[test]
    fn color_preset_click_creates_one_undo_entry() {
        let mut history = History::new();
        let original_scene = Scene::new();
        let node_id = find_primitive_id(&original_scene);

        // Frame 0: idle
        history.begin_frame(&original_scene, Some(node_id));
        history.end_frame(&original_scene, Some(node_id), false);

        // Frame 1: user clicks a color preset (no drag, instant change)
        history.begin_frame(&original_scene, Some(node_id));
        let mut scene = original_scene.clone();
        if let Some(n) = scene.nodes.get_mut(&node_id) {
            if let NodeData::Primitive {
                ref mut material, ..
            } = n.data
            {
                material.base_color = glam::Vec3::new(1.0, 0.0, 0.0); // red preset
            }
        }
        history.end_frame(&scene, Some(node_id), false);
        assert_eq!(history.undo_count(), 1, "instant edit = one undo entry");

        // Undo restores original color
        let (restored, _) = unwrap_scene_restore(history.undo(&scene, Some(node_id)).unwrap());
        let restored_color = match &restored.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { material, .. } => material.base_color,
            _ => panic!("expected primitive"),
        };
        assert_ne!(
            restored_color,
            glam::Vec3::new(1.0, 0.0, 0.0),
            "undo should restore to non-red color"
        );
    }

    /// Verify that two consecutive drags create two separate undo entries.
    #[test]
    fn consecutive_drags_create_separate_undo_entries() {
        let mut history = History::new();
        let original_scene = Scene::new();
        let node_id = find_primitive_id(&original_scene);

        // Frame 0: idle
        history.begin_frame(&original_scene, Some(node_id));
        history.end_frame(&original_scene, Some(node_id), false);

        // Drag 1: move X to 2.0
        history.begin_frame(&original_scene, Some(node_id));
        let mut scene_after_drag1 = original_scene.clone();
        if let Some(n) = scene_after_drag1.nodes.get_mut(&node_id) {
            if let NodeData::Primitive {
                ref mut position, ..
            } = n.data
            {
                position.x = 2.0;
            }
        }
        history.end_frame(&scene_after_drag1, Some(node_id), true);
        // Drag 1 ends
        history.begin_frame(&scene_after_drag1, Some(node_id));
        history.end_frame(&scene_after_drag1, Some(node_id), false);
        assert_eq!(history.undo_count(), 1, "first drag = one undo entry");

        // Drag 2: move Y to 5.0
        history.begin_frame(&scene_after_drag1, Some(node_id));
        let mut scene_after_drag2 = scene_after_drag1.clone();
        if let Some(n) = scene_after_drag2.nodes.get_mut(&node_id) {
            if let NodeData::Primitive {
                ref mut position, ..
            } = n.data
            {
                position.y = 5.0;
            }
        }
        history.end_frame(&scene_after_drag2, Some(node_id), true);
        // Drag 2 ends
        history.begin_frame(&scene_after_drag2, Some(node_id));
        history.end_frame(&scene_after_drag2, Some(node_id), false);
        assert_eq!(history.undo_count(), 2, "two drags = two undo entries");

        // Undo twice restores original
        let (s1, _) =
            unwrap_scene_restore(history.undo(&scene_after_drag2, Some(node_id)).unwrap());
        let pos1 = match &s1.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        };
        assert!(
            (pos1.x - 2.0).abs() < 0.001,
            "first undo restores drag1 state"
        );
        assert!(pos1.y.abs() < 0.001, "first undo restores drag1 state");

        let (s0, _) = unwrap_scene_restore(history.undo(&s1, Some(node_id)).unwrap());
        let pos0 = match &s0.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        };
        assert!(pos0.x.abs() < 0.001, "second undo restores original");
    }
}
