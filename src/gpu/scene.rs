use std::collections::HashMap;

use glam::Vec3;

// ── Primitive and Operation types ───────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SdfPrimitive {
    Sphere = 0,
    Box = 1,
    Cylinder = 2,
    Torus = 3,
    Plane = 4,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SdfOperation {
    Union = 0,
    SmoothUnion = 1,
    Subtract = 2,
    Intersect = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdfTransform {
    Translate = 0,
    Rotate = 1,
    Scale = 2,
}

// ── Node-based Scene Graph ──────────────────────────────────────

pub type NodeId = u64;

#[derive(Clone)]
pub struct PrimitiveData {
    pub primitive: SdfPrimitive,
    pub position: Vec3,
    pub scale: Vec3,
    pub color: Vec3,
}

impl PrimitiveData {
    pub fn new(primitive: SdfPrimitive) -> Self {
        let (default_scale, default_color) = match primitive {
            SdfPrimitive::Sphere => (Vec3::splat(0.5), Vec3::new(0.8, 0.55, 0.4)),
            SdfPrimitive::Box => (Vec3::splat(0.4), Vec3::new(0.4, 0.6, 0.8)),
            SdfPrimitive::Cylinder => (Vec3::new(0.3, 0.5, 0.3), Vec3::new(0.6, 0.8, 0.4)),
            SdfPrimitive::Torus => (Vec3::new(0.4, 0.15, 0.4), Vec3::new(0.8, 0.4, 0.6)),
            SdfPrimitive::Plane => (Vec3::ONE, Vec3::new(0.25, 0.25, 0.3)),
        };

        Self {
            primitive,
            position: Vec3::new(0.0, 0.5, 0.0),
            scale: default_scale,
            color: default_color,
        }
    }
}

#[derive(Clone)]
pub struct OperationData {
    pub operation: SdfOperation,
    pub smooth_k: f32,
    pub left: NodeId,
    pub right: NodeId,
}

#[derive(Clone)]
pub struct TransformData {
    pub transform: SdfTransform,
    pub input: NodeId,
    pub offset: Vec3, // translate offset / rotation angles / scale factors
}

#[derive(Clone)]
pub enum NodeData {
    Primitive(PrimitiveData),
    Operation(OperationData),
    Transform(TransformData),
}

#[derive(Clone)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub data: NodeData,
}

// ── Scene ───────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Scene {
    nodes: HashMap<NodeId, SceneNode>,
    root: Option<NodeId>,
    next_id: u64,
    pub selected: Option<NodeId>,
    node_counter: u32,
    gpu_index_to_node_id: Vec<Option<NodeId>>,
}

impl Scene {
    pub fn default_scene() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            root: None,
            next_id: 1,
            selected: None,
            node_counter: 0,
            gpu_index_to_node_id: Vec::new(),
        };
        scene.add_primitive(SdfPrimitive::Sphere);
        scene
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn root_id(&self) -> Option<NodeId> {
        self.root
    }

    pub fn get_node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    /// Map a GPU array index (from pick result) back to a NodeId.
    pub fn node_id_from_gpu_index(&self, gpu_idx: usize) -> Option<NodeId> {
        self.gpu_index_to_node_id.get(gpu_idx).copied().flatten()
    }

    /// Get the position of the selected node (primitive position or translate offset).
    /// Returns None for operations, rotate, and scale (no gizmo).
    pub fn selected_primitive_position(&self) -> Option<Vec3> {
        let sel_id = self.selected?;
        let node = self.nodes.get(&sel_id)?;
        match &node.data {
            NodeData::Primitive(prim) => Some(prim.position),
            NodeData::Transform(tr) if tr.transform == SdfTransform::Translate => Some(tr.offset),
            _ => None,
        }
    }

    /// Set the position of the selected node (primitive position or translate offset).
    pub fn set_selected_position(&mut self, pos: Vec3) {
        if let Some(sel_id) = self.selected {
            if let Some(node) = self.nodes.get_mut(&sel_id) {
                match &mut node.data {
                    NodeData::Primitive(ref mut prim) => prim.position = pos,
                    NodeData::Transform(ref mut tr)
                        if tr.transform == SdfTransform::Translate =>
                    {
                        tr.offset = pos;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Add a new primitive. If a root exists, wrap in a SmoothUnion operation.
    pub fn add_primitive(&mut self, primitive: SdfPrimitive) -> NodeId {
        self.node_counter += 1;
        let name = match primitive {
            SdfPrimitive::Sphere => format!("Sphere {}", self.node_counter),
            SdfPrimitive::Box => format!("Box {}", self.node_counter),
            SdfPrimitive::Cylinder => format!("Cylinder {}", self.node_counter),
            SdfPrimitive::Torus => format!("Torus {}", self.node_counter),
            SdfPrimitive::Plane => format!("Plane {}", self.node_counter),
        };

        let prim_id = self.alloc_id();
        self.nodes.insert(
            prim_id,
            SceneNode {
                id: prim_id,
                name,
                data: NodeData::Primitive(PrimitiveData::new(primitive)),
            },
        );

        match self.root {
            None => {
                self.root = Some(prim_id);
            }
            Some(old_root) => {
                let op_id = self.alloc_id();
                self.nodes.insert(
                    op_id,
                    SceneNode {
                        id: op_id,
                        name: format!("SmoothUnion {}", self.node_counter),
                        data: NodeData::Operation(OperationData {
                            operation: SdfOperation::SmoothUnion,
                            smooth_k: 0.1,
                            left: old_root,
                            right: prim_id,
                        }),
                    },
                );
                self.root = Some(op_id);
            }
        }

        self.selected = Some(prim_id);
        prim_id
    }

    /// Wrap the target node in a transform. The transform replaces the target
    /// in its parent, and the target becomes the transform's input.
    pub fn wrap_in_transform(&mut self, target_id: NodeId, transform: SdfTransform) -> NodeId {
        self.node_counter += 1;
        let name = match transform {
            SdfTransform::Translate => format!("Translate {}", self.node_counter),
            SdfTransform::Rotate => format!("Rotate {}", self.node_counter),
            SdfTransform::Scale => format!("Scale {}", self.node_counter),
        };

        let default_offset = match transform {
            SdfTransform::Translate => Vec3::ZERO,
            SdfTransform::Rotate => Vec3::ZERO,
            SdfTransform::Scale => Vec3::ONE,
        };

        let tr_id = self.alloc_id();
        self.nodes.insert(
            tr_id,
            SceneNode {
                id: tr_id,
                name,
                data: NodeData::Transform(TransformData {
                    transform,
                    input: target_id,
                    offset: default_offset,
                }),
            },
        );

        // Rewire parent to point to transform instead of target
        let parent = self.find_parent(target_id);
        match parent {
            None => {
                if self.root == Some(target_id) {
                    self.root = Some(tr_id);
                }
            }
            Some(parent_id) => {
                let pn = self.nodes.get_mut(&parent_id).unwrap();
                match &mut pn.data {
                    NodeData::Operation(ref mut op) => {
                        if op.left == target_id {
                            op.left = tr_id;
                        } else {
                            op.right = tr_id;
                        }
                    }
                    NodeData::Transform(ref mut parent_tr) => {
                        parent_tr.input = tr_id;
                    }
                    _ => {}
                }
            }
        }

        self.selected = Some(tr_id);
        tr_id
    }

    /// Remove the selected node.
    /// - Transform: unwrap (promote input to parent).
    /// - Primitive/Operation: walk up through transform chain, remove from operation parent.
    pub fn remove_selected(&mut self) {
        let Some(sel_id) = self.selected else {
            return;
        };

        // If selected is a transform, unwrap it (promote its input)
        let is_transform = matches!(
            self.nodes.get(&sel_id).map(|n| &n.data),
            Some(NodeData::Transform(_))
        );

        if is_transform {
            let input_id = match &self.nodes[&sel_id].data {
                NodeData::Transform(tr) => tr.input,
                _ => unreachable!(),
            };

            let parent = self.find_parent(sel_id);
            match parent {
                None => {
                    self.root = Some(input_id);
                }
                Some(parent_id) => {
                    let pn = self.nodes.get_mut(&parent_id).unwrap();
                    match &mut pn.data {
                        NodeData::Operation(ref mut op) => {
                            if op.left == sel_id {
                                op.left = input_id;
                            } else {
                                op.right = input_id;
                            }
                        }
                        NodeData::Transform(ref mut tr) => {
                            tr.input = input_id;
                        }
                        _ => {}
                    }
                }
            }
            self.nodes.remove(&sel_id);
            self.selected = None;
            return;
        }

        // For primitives/operations: walk up through transform chain to find
        // the structural parent (operation or root).
        let mut current = sel_id;
        let mut transforms_to_remove = Vec::new();

        loop {
            match self.find_parent(current) {
                None => {
                    // current is root — remove everything
                    self.root = None;
                    self.remove_subtree(sel_id);
                    for t in transforms_to_remove {
                        self.nodes.remove(&t);
                    }
                    break;
                }
                Some(parent_id) => {
                    let parent_is_transform = matches!(
                        self.nodes.get(&parent_id).map(|n| &n.data),
                        Some(NodeData::Transform(_))
                    );

                    if parent_is_transform {
                        transforms_to_remove.push(parent_id);
                        current = parent_id;
                        continue;
                    }

                    // Parent is an operation — do normal sibling promotion
                    if let NodeData::Operation(op) = &self.nodes[&parent_id].data {
                        let sibling = if op.left == current {
                            op.right
                        } else {
                            op.left
                        };

                        let grandparent = self.find_parent(parent_id);
                        match grandparent {
                            None => {
                                self.root = Some(sibling);
                            }
                            Some(gp_id) => {
                                let gp = self.nodes.get_mut(&gp_id).unwrap();
                                match &mut gp.data {
                                    NodeData::Operation(ref mut gp_op) => {
                                        if gp_op.left == parent_id {
                                            gp_op.left = sibling;
                                        } else {
                                            gp_op.right = sibling;
                                        }
                                    }
                                    NodeData::Transform(ref mut gp_tr) => {
                                        gp_tr.input = sibling;
                                    }
                                    _ => {}
                                }
                            }
                        }

                        self.remove_subtree(sel_id);
                        for t in transforms_to_remove {
                            self.nodes.remove(&t);
                        }
                        self.nodes.remove(&parent_id);
                    }
                    break;
                }
            }
        }

        self.selected = None;
    }

    /// Disconnect a child from its parent operation.
    /// Removes the parent op, promotes the sibling, deletes the disconnected child subtree.
    pub fn disconnect(&mut self, parent_op_id: NodeId, is_left_child: bool) {
        let Some(parent_node) = self.nodes.get(&parent_op_id) else {
            return;
        };
        let (child_to_remove, sibling_to_promote) = match &parent_node.data {
            NodeData::Operation(op) => {
                if is_left_child {
                    (op.left, op.right)
                } else {
                    (op.right, op.left)
                }
            }
            _ => return, // only operations support wire disconnect
        };

        // Rewire grandparent to point to sibling
        let grandparent = self.find_parent(parent_op_id);
        match grandparent {
            None => {
                self.root = Some(sibling_to_promote);
            }
            Some(gp_id) => {
                let gp_node = self.nodes.get_mut(&gp_id).unwrap();
                match &mut gp_node.data {
                    NodeData::Operation(ref mut op) => {
                        if op.left == parent_op_id {
                            op.left = sibling_to_promote;
                        } else {
                            op.right = sibling_to_promote;
                        }
                    }
                    NodeData::Transform(ref mut tr) => {
                        tr.input = sibling_to_promote;
                    }
                    _ => {}
                }
            }
        }

        self.nodes.remove(&parent_op_id);
        self.remove_subtree(child_to_remove);
        self.selected = None;
    }

    /// Recursively remove a node and all its descendants.
    pub fn remove_subtree(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.remove(&node_id) {
            match node.data {
                NodeData::Operation(op) => {
                    self.remove_subtree(op.left);
                    self.remove_subtree(op.right);
                }
                NodeData::Transform(tr) => {
                    self.remove_subtree(tr.input);
                }
                NodeData::Primitive(_) => {}
            }
        }
    }

    /// Detach a child from its parent, collapsing the parent structure.
    /// For operations: promotes sibling. For transforms: recursively detaches up the chain.
    pub fn detach_from_parent(&mut self, child_id: NodeId) -> Option<NodeId> {
        let parent_id = self.find_parent(child_id)?;

        // Need to read parent data before mutating
        let parent_is_transform = matches!(
            self.nodes.get(&parent_id).map(|n| &n.data),
            Some(NodeData::Transform(_))
        );

        if parent_is_transform {
            // Transform has only one child. Remove it, then recursively
            // detach the transform from ITS parent.
            let result = self.detach_from_parent(parent_id);
            self.nodes.remove(&parent_id);
            return result;
        }

        // Parent is an operation — promote sibling
        let sibling_id = {
            let parent = &self.nodes[&parent_id];
            match &parent.data {
                NodeData::Operation(op) => {
                    if op.left == child_id {
                        op.right
                    } else {
                        op.left
                    }
                }
                _ => return None,
            }
        };

        let grandparent = self.find_parent(parent_id);
        match grandparent {
            None => {
                self.root = Some(sibling_id);
            }
            Some(gp_id) => {
                let gp_node = self.nodes.get_mut(&gp_id).unwrap();
                match &mut gp_node.data {
                    NodeData::Operation(ref mut op) => {
                        if op.left == parent_id {
                            op.left = sibling_id;
                        } else {
                            op.right = sibling_id;
                        }
                    }
                    NodeData::Transform(ref mut tr) => {
                        tr.input = sibling_id;
                    }
                    _ => {}
                }
            }
        }

        self.nodes.remove(&parent_id);
        Some(sibling_id)
    }

    /// Check if `potential_ancestor` is an ancestor of `target` (or equal to it).
    fn is_ancestor_of(&self, potential_ancestor: NodeId, target: NodeId) -> bool {
        if potential_ancestor == target {
            return true;
        }
        if let Some(node) = self.nodes.get(&potential_ancestor) {
            match &node.data {
                NodeData::Operation(op) => {
                    return self.is_ancestor_of(op.left, target)
                        || self.is_ancestor_of(op.right, target);
                }
                NodeData::Transform(tr) => {
                    return self.is_ancestor_of(tr.input, target);
                }
                _ => {}
            }
        }
        false
    }

    /// Rewire: set `target_op`'s left or right child to `new_child_id`.
    /// Returns Ok(()) if valid, Err if it would create a cycle.
    pub fn rewire(
        &mut self,
        target_op_id: NodeId,
        is_left_slot: bool,
        new_child_id: NodeId,
    ) -> Result<(), &'static str> {
        // Validate target is an operation
        if !matches!(
            self.nodes.get(&target_op_id).map(|n| &n.data),
            Some(NodeData::Operation(_))
        ) {
            return Err("target is not an operation");
        }

        // Check cycle: new_child must not be an ancestor of target_op
        if self.is_ancestor_of(new_child_id, target_op_id) {
            return Err("would create a cycle");
        }

        // Get the old child at the target slot
        let old_child = {
            let op = match &self.nodes[&target_op_id].data {
                NodeData::Operation(op) => op,
                _ => unreachable!(),
            };
            if is_left_slot {
                op.left
            } else {
                op.right
            }
        };

        if old_child == new_child_id {
            return Ok(());
        }

        // Detach new_child from its current parent
        let parent_of_new = self.find_parent(new_child_id);
        if let Some(p) = parent_of_new {
            if p != target_op_id {
                self.detach_from_parent(new_child_id);
            }
        }

        // Set the new child on the target operation
        let op = match &mut self.nodes.get_mut(&target_op_id).unwrap().data {
            NodeData::Operation(op) => op,
            _ => unreachable!(),
        };
        if is_left_slot {
            op.left = new_child_id;
        } else {
            op.right = new_child_id;
        }

        // Remove the old child
        if old_child != new_child_id {
            self.remove_subtree(old_child);
        }

        Ok(())
    }

    /// Rewire a transform's single input to a new child.
    pub fn rewire_transform(
        &mut self,
        transform_id: NodeId,
        new_input_id: NodeId,
    ) -> Result<(), &'static str> {
        let old_input = match self.nodes.get(&transform_id).map(|n| &n.data) {
            Some(NodeData::Transform(tr)) => tr.input,
            _ => return Err("target is not a transform"),
        };

        if self.is_ancestor_of(new_input_id, transform_id) {
            return Err("would create a cycle");
        }

        if old_input == new_input_id {
            return Ok(());
        }

        // Detach new_input from its current parent
        let parent_of_new = self.find_parent(new_input_id);
        if let Some(p) = parent_of_new {
            if p != transform_id {
                self.detach_from_parent(new_input_id);
            }
        }

        // Set new input
        if let Some(node) = self.nodes.get_mut(&transform_id) {
            if let NodeData::Transform(ref mut tr) = node.data {
                tr.input = new_input_id;
            }
        }

        // Remove old input subtree
        self.remove_subtree(old_input);

        Ok(())
    }

    /// Find the parent node that has `child_id` as a child.
    fn find_parent(&self, child_id: NodeId) -> Option<NodeId> {
        for (id, node) in &self.nodes {
            match &node.data {
                NodeData::Operation(op) => {
                    if op.left == child_id || op.right == child_id {
                        return Some(*id);
                    }
                }
                NodeData::Transform(tr) => {
                    if tr.input == child_id {
                        return Some(*id);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Flatten the tree to GPU format via post-order traversal (RPN).
    /// Transforms are pushed BEFORE their child (pre-order) since they are
    /// parameter lookups, not instructions.
    pub fn flatten_for_gpu(&mut self) -> (Vec<SdfNodeGpu>, SceneInfoGpu) {
        let mut gpu_nodes = Vec::new();
        self.gpu_index_to_node_id.clear();
        let mut selected_gpu_idx: i32 = -1;

        if let Some(root_id) = self.root {
            flatten_recursive(
                &self.nodes,
                root_id,
                self.selected,
                &mut gpu_nodes,
                &mut self.gpu_index_to_node_id,
                &mut selected_gpu_idx,
            );
        }

        let info = SceneInfoGpu {
            node_count: gpu_nodes.len() as u32,
            selected_idx: selected_gpu_idx,
            _pad: [0; 2],
        };

        (gpu_nodes, info)
    }
}

/// Post-order recursive flattening (free function to avoid borrow issues).
/// Transforms are pre-order (pushed before child).
fn flatten_recursive(
    nodes: &HashMap<NodeId, SceneNode>,
    node_id: NodeId,
    selected: Option<NodeId>,
    out: &mut Vec<SdfNodeGpu>,
    gpu_map: &mut Vec<Option<NodeId>>,
    selected_gpu_idx: &mut i32,
) {
    let node = &nodes[&node_id];

    match &node.data {
        NodeData::Primitive(prim) => {
            let gpu_idx = out.len();
            let is_selected = selected == Some(node_id);
            if is_selected {
                *selected_gpu_idx = gpu_idx as i32;
            }
            out.push(SdfNodeGpu {
                type_op: [
                    0.0, // 0 = primitive
                    prim.primitive as u32 as f32,
                    0.0,
                    if is_selected { 1.0 } else { 0.0 },
                ],
                position: [prim.position.x, prim.position.y, prim.position.z, 0.0],
                scale: [prim.scale.x, prim.scale.y, prim.scale.z, 0.0],
                color: [prim.color.x, prim.color.y, prim.color.z, 1.0],
                _reserved: [0.0; 4],
            });
            gpu_map.push(Some(node_id));
        }
        NodeData::Operation(op) => {
            let left = op.left;
            let right = op.right;
            let operation = op.operation;
            let smooth_k = op.smooth_k;

            // Post-order: left, right, then operation
            flatten_recursive(nodes, left, selected, out, gpu_map, selected_gpu_idx);
            flatten_recursive(nodes, right, selected, out, gpu_map, selected_gpu_idx);

            out.push(SdfNodeGpu {
                type_op: [
                    1.0, // 1 = operation
                    operation as u32 as f32,
                    smooth_k,
                    0.0,
                ],
                position: [0.0; 4],
                scale: [0.0; 4],
                color: [0.0; 4],
                _reserved: [0.0; 4],
            });
            gpu_map.push(None); // operations don't have a NodeId for picking
        }
        NodeData::Transform(tr) => {
            // Pre-order: push transform BEFORE child (it's a parameter lookup)
            let gpu_idx = out.len();
            let is_selected = selected == Some(node_id);
            if is_selected {
                *selected_gpu_idx = gpu_idx as i32;
            }
            out.push(SdfNodeGpu {
                type_op: [
                    2.0, // 2 = transform
                    tr.transform as u32 as f32,
                    0.0,
                    if is_selected { 1.0 } else { 0.0 },
                ],
                position: [tr.offset.x, tr.offset.y, tr.offset.z, 0.0],
                scale: [0.0; 4],
                color: [0.0; 4],
                _reserved: [0.0; 4],
            });
            gpu_map.push(Some(node_id));

            // Then emit child
            flatten_recursive(nodes, tr.input, selected, out, gpu_map, selected_gpu_idx);
        }
    }
}

// ── Tree items for UI ─────────────────────────────────────────

pub struct TreeItemData {
    pub depth: i32,
    pub name: String,
    pub node_type: i32,
    pub is_selected: bool,
    pub node_id: NodeId,
}

impl Scene {
    pub fn build_tree_items(&self) -> Vec<TreeItemData> {
        let mut items = Vec::new();
        if let Some(root_id) = self.root {
            self.build_tree_recursive(root_id, 0, &mut items);
        }
        items
    }

    fn build_tree_recursive(&self, node_id: NodeId, depth: i32, out: &mut Vec<TreeItemData>) {
        let node = &self.nodes[&node_id];
        match &node.data {
            NodeData::Primitive(prim) => {
                out.push(TreeItemData {
                    depth,
                    name: node.name.clone(),
                    node_type: prim.primitive as i32,
                    is_selected: self.selected == Some(node_id),
                    node_id,
                });
            }
            NodeData::Operation(op) => {
                out.push(TreeItemData {
                    depth,
                    name: node.name.clone(),
                    node_type: 10 + op.operation as i32,
                    is_selected: self.selected == Some(node_id),
                    node_id,
                });
                self.build_tree_recursive(op.left, depth + 1, out);
                self.build_tree_recursive(op.right, depth + 1, out);
            }
            NodeData::Transform(tr) => {
                out.push(TreeItemData {
                    depth,
                    name: node.name.clone(),
                    node_type: 20 + tr.transform as i32,
                    is_selected: self.selected == Some(node_id),
                    node_id,
                });
                self.build_tree_recursive(tr.input, depth + 1, out);
            }
        }
    }
}

// ── Graph layout for node editor ──────────────────────────────

pub struct GraphNodeData {
    pub node_id: NodeId,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub name: String,
    pub node_type: i32,
    pub is_selected: bool,
    // Port positions (canvas-space)
    pub out_port_x: f32,
    pub out_port_y: f32,
    // Two-input ports (operations)
    pub in_port_top_x: f32,
    pub in_port_top_y: f32,
    pub in_port_bot_x: f32,
    pub in_port_bot_y: f32,
    pub has_input_ports: bool,
    // Single-input port (transforms)
    pub in_port_single_x: f32,
    pub in_port_single_y: f32,
    pub has_single_input: bool,
}

pub struct GraphWireData {
    pub parent_node_id: NodeId,
    pub child_node_id: NodeId,
    pub is_left_child: bool,
    pub start: (f32, f32), // child output port
    pub end: (f32, f32),   // parent input port
}

pub struct GraphLayoutData {
    pub nodes: Vec<GraphNodeData>,
    pub wires: Vec<GraphWireData>,
    pub connections_svg: String,
    pub canvas_width: f32,
    pub canvas_height: f32,
}

const GRAPH_NODE_W: f32 = 130.0;
const GRAPH_NODE_H: f32 = 36.0;
const GRAPH_COL_SPACING: f32 = 180.0;
const GRAPH_ROW_SPACING: f32 = 50.0;
const GRAPH_PADDING: f32 = 20.0;

impl Scene {
    /// Compute a left-to-right auto-layout for the node graph panel.
    pub fn build_graph_layout(&self) -> GraphLayoutData {
        let Some(root_id) = self.root else {
            return GraphLayoutData {
                nodes: Vec::new(),
                wires: Vec::new(),
                connections_svg: String::new(),
                canvas_width: 0.0,
                canvas_height: 0.0,
            };
        };

        let mut columns: HashMap<NodeId, i32> = HashMap::new();
        let mut y_positions: HashMap<NodeId, f32> = HashMap::new();
        let mut leaf_counter: f32 = 0.0;

        self.assign_columns(root_id, &mut columns);
        self.assign_y_positions(root_id, &mut y_positions, &mut leaf_counter);

        let mut nodes = Vec::new();
        let mut max_x: f32 = 0.0;
        let mut max_y: f32 = 0.0;

        for (&nid, &col) in &columns {
            let node = &self.nodes[&nid];
            let row_y = y_positions[&nid];
            let px = GRAPH_PADDING + col as f32 * GRAPH_COL_SPACING;
            let py = GRAPH_PADDING + row_y * GRAPH_ROW_SPACING;

            let (name, node_type, has_input_ports, has_single_input) = match &node.data {
                NodeData::Primitive(prim) => {
                    (node.name.clone(), prim.primitive as i32, false, false)
                }
                NodeData::Operation(op) => {
                    (node.name.clone(), 10 + op.operation as i32, true, false)
                }
                NodeData::Transform(tr) => {
                    (node.name.clone(), 20 + tr.transform as i32, false, true)
                }
            };

            nodes.push(GraphNodeData {
                node_id: nid,
                x: px,
                y: py,
                width: GRAPH_NODE_W,
                height: GRAPH_NODE_H,
                name,
                node_type,
                is_selected: self.selected == Some(nid),
                out_port_x: px + GRAPH_NODE_W,
                out_port_y: py + GRAPH_NODE_H / 2.0,
                in_port_top_x: px,
                in_port_top_y: py + GRAPH_NODE_H / 3.0,
                in_port_bot_x: px,
                in_port_bot_y: py + GRAPH_NODE_H * 2.0 / 3.0,
                has_input_ports,
                in_port_single_x: px,
                in_port_single_y: py + GRAPH_NODE_H / 2.0,
                has_single_input,
            });

            if px + GRAPH_NODE_W > max_x {
                max_x = px + GRAPH_NODE_W;
            }
            if py + GRAPH_NODE_H > max_y {
                max_y = py + GRAPH_NODE_H;
            }
        }

        let mut wires = Vec::new();
        let connections_svg =
            self.build_connections_svg(root_id, &columns, &y_positions, &mut wires);

        GraphLayoutData {
            nodes,
            wires,
            connections_svg,
            canvas_width: max_x + GRAPH_PADDING,
            canvas_height: max_y + GRAPH_PADDING,
        }
    }

    /// Post-order: leaf = column 0, operation = 1 + max(children), transform = child + 1.
    fn assign_columns(&self, node_id: NodeId, columns: &mut HashMap<NodeId, i32>) {
        let node = &self.nodes[&node_id];
        match &node.data {
            NodeData::Primitive(_) => {
                columns.insert(node_id, 0);
            }
            NodeData::Operation(op) => {
                self.assign_columns(op.left, columns);
                self.assign_columns(op.right, columns);
                let left_col = columns[&op.left];
                let right_col = columns[&op.right];
                columns.insert(node_id, 1 + left_col.max(right_col));
            }
            NodeData::Transform(tr) => {
                self.assign_columns(tr.input, columns);
                let child_col = columns[&tr.input];
                columns.insert(node_id, child_col + 1);
            }
        }
    }

    /// In-order: leaves get sequential rows, operations center between children,
    /// transforms share their child's y position.
    fn assign_y_positions(
        &self,
        node_id: NodeId,
        y_positions: &mut HashMap<NodeId, f32>,
        leaf_counter: &mut f32,
    ) {
        let node = &self.nodes[&node_id];
        match &node.data {
            NodeData::Primitive(_) => {
                y_positions.insert(node_id, *leaf_counter);
                *leaf_counter += 1.0;
            }
            NodeData::Operation(op) => {
                self.assign_y_positions(op.left, y_positions, leaf_counter);
                self.assign_y_positions(op.right, y_positions, leaf_counter);
                let left_y = y_positions[&op.left];
                let right_y = y_positions[&op.right];
                y_positions.insert(node_id, (left_y + right_y) / 2.0);
            }
            NodeData::Transform(tr) => {
                self.assign_y_positions(tr.input, y_positions, leaf_counter);
                let child_y = y_positions[&tr.input];
                y_positions.insert(node_id, child_y);
            }
        }
    }

    /// Build SVG bezier path string for all connections, also populating wire data.
    fn build_connections_svg(
        &self,
        node_id: NodeId,
        columns: &HashMap<NodeId, i32>,
        y_positions: &HashMap<NodeId, f32>,
        wires: &mut Vec<GraphWireData>,
    ) -> String {
        let mut svg = String::new();
        self.build_connections_recursive(node_id, columns, y_positions, &mut svg, wires);
        svg
    }

    fn build_connections_recursive(
        &self,
        node_id: NodeId,
        columns: &HashMap<NodeId, i32>,
        y_positions: &HashMap<NodeId, f32>,
        svg: &mut String,
        wires: &mut Vec<GraphWireData>,
    ) {
        let node = &self.nodes[&node_id];
        match &node.data {
            NodeData::Operation(op) => {
                self.build_connections_recursive(op.left, columns, y_positions, svg, wires);
                self.build_connections_recursive(op.right, columns, y_positions, svg, wires);

                let parent_col = columns[&node_id];
                let parent_y = y_positions[&node_id];
                let parent_px = GRAPH_PADDING + parent_col as f32 * GRAPH_COL_SPACING;
                let parent_py = GRAPH_PADDING + parent_y * GRAPH_ROW_SPACING;

                let input_x = parent_px;
                let input_y_left = parent_py + GRAPH_NODE_H / 3.0;
                let input_y_right = parent_py + GRAPH_NODE_H * 2.0 / 3.0;

                for (i, (child_id, input_y)) in
                    [(op.left, input_y_left), (op.right, input_y_right)]
                        .iter()
                        .enumerate()
                {
                    let child_col = columns[child_id];
                    let child_y_pos = y_positions[child_id];
                    let child_px = GRAPH_PADDING + child_col as f32 * GRAPH_COL_SPACING;
                    let child_py = GRAPH_PADDING + child_y_pos * GRAPH_ROW_SPACING;

                    let x1 = child_px + GRAPH_NODE_W;
                    let y1 = child_py + GRAPH_NODE_H / 2.0;
                    let x2 = input_x;
                    let y2 = *input_y;

                    wires.push(GraphWireData {
                        parent_node_id: node_id,
                        child_node_id: *child_id,
                        is_left_child: i == 0,
                        start: (x1, y1),
                        end: (x2, y2),
                    });

                    let dx = (x2 - x1).abs();
                    let cx1 = x1 + dx / 3.0;
                    let cx2 = x2 - dx / 3.0;

                    use std::fmt::Write;
                    let _ = write!(
                        svg,
                        "M {:.1} {:.1} C {:.1} {:.1} {:.1} {:.1} {:.1} {:.1} ",
                        x1, y1, cx1, y1, cx2, y2, x2, y2
                    );
                }
            }
            NodeData::Transform(tr) => {
                self.build_connections_recursive(tr.input, columns, y_positions, svg, wires);

                let tr_col = columns[&node_id];
                let tr_y = y_positions[&node_id];
                let tr_px = GRAPH_PADDING + tr_col as f32 * GRAPH_COL_SPACING;
                let tr_py = GRAPH_PADDING + tr_y * GRAPH_ROW_SPACING;

                let child_col = columns[&tr.input];
                let child_y_pos = y_positions[&tr.input];
                let child_px = GRAPH_PADDING + child_col as f32 * GRAPH_COL_SPACING;
                let child_py = GRAPH_PADDING + child_y_pos * GRAPH_ROW_SPACING;

                let x1 = child_px + GRAPH_NODE_W;
                let y1 = child_py + GRAPH_NODE_H / 2.0;
                let x2 = tr_px; // single input at left center
                let y2 = tr_py + GRAPH_NODE_H / 2.0;

                wires.push(GraphWireData {
                    parent_node_id: node_id,
                    child_node_id: tr.input,
                    is_left_child: true,
                    start: (x1, y1),
                    end: (x2, y2),
                });

                let dx = (x2 - x1).abs();
                let cx1 = x1 + dx / 3.0;
                let cx2 = x2 - dx / 3.0;

                use std::fmt::Write;
                let _ = write!(
                    svg,
                    "M {:.1} {:.1} C {:.1} {:.1} {:.1} {:.1} {:.1} {:.1} ",
                    x1, y1, cx1, y1, cx2, y2, x2, y2
                );
            }
            NodeData::Primitive(_) => {} // leaves have no children to connect
        }
    }
}

// ── Pick result ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug)]
pub enum PickResult {
    Background,
    Floor,
    Node(usize),
    GizmoAxis(GizmoAxis),
}

// ── GPU-side structs (bytemuck) ─────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfNodeGpu {
    pub type_op: [f32; 4],
    pub position: [f32; 4],
    pub scale: [f32; 4],
    pub color: [f32; 4],
    pub _reserved: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneInfoGpu {
    pub node_count: u32,
    pub selected_idx: i32,
    pub _pad: [u32; 2],
}
