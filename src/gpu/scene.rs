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

// ── Node-based Scene Graph ──────────────────────────────────────

pub type NodeId = u64;

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

pub struct OperationData {
    pub operation: SdfOperation,
    pub smooth_k: f32,
    pub left: NodeId,
    pub right: NodeId,
}

pub enum NodeData {
    Primitive(PrimitiveData),
    Operation(OperationData),
}

pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub data: NodeData,
}

// ── Scene ───────────────────────────────────────────────────────

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

    /// Get the position of the selected primitive node.
    pub fn selected_primitive_position(&self) -> Option<Vec3> {
        let sel_id = self.selected?;
        let node = self.nodes.get(&sel_id)?;
        match &node.data {
            NodeData::Primitive(prim) => Some(prim.position),
            _ => None,
        }
    }

    /// Set the position of the selected primitive node.
    pub fn set_selected_position(&mut self, pos: Vec3) {
        if let Some(sel_id) = self.selected {
            if let Some(node) = self.nodes.get_mut(&sel_id) {
                if let NodeData::Primitive(ref mut prim) = node.data {
                    prim.position = pos;
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

    /// Remove the selected primitive and its parent operation.
    pub fn remove_selected(&mut self) {
        let Some(sel_id) = self.selected else {
            return;
        };

        let parent = self.find_parent(sel_id);

        match parent {
            None => {
                // Selected is root
                if self.root == Some(sel_id) {
                    self.root = None;
                    self.nodes.remove(&sel_id);
                }
            }
            Some(parent_id) => {
                // Find the sibling
                let sibling_id = {
                    let parent_node = &self.nodes[&parent_id];
                    match &parent_node.data {
                        NodeData::Operation(op) => {
                            if op.left == sel_id {
                                op.right
                            } else {
                                op.left
                            }
                        }
                        _ => unreachable!("parent must be an operation"),
                    }
                };

                // Find grandparent and rewire
                let grandparent = self.find_parent(parent_id);
                match grandparent {
                    None => {
                        // Parent was root — sibling becomes root
                        self.root = Some(sibling_id);
                    }
                    Some(gp_id) => {
                        let gp_node = self.nodes.get_mut(&gp_id).unwrap();
                        if let NodeData::Operation(ref mut op) = gp_node.data {
                            if op.left == parent_id {
                                op.left = sibling_id;
                            } else {
                                op.right = sibling_id;
                            }
                        }
                    }
                }

                self.nodes.remove(&sel_id);
                self.nodes.remove(&parent_id);
            }
        }

        self.selected = None;
    }

    /// Find the operation node that has `child_id` as a child.
    fn find_parent(&self, child_id: NodeId) -> Option<NodeId> {
        for (id, node) in &self.nodes {
            if let NodeData::Operation(op) = &node.data {
                if op.left == child_id || op.right == child_id {
                    return Some(*id);
                }
            }
        }
        None
    }

    /// Flatten the tree to GPU format via post-order traversal (RPN).
    /// Also updates `gpu_index_to_node_id` for pick result mapping.
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
