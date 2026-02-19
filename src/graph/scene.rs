use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub type NodeId = u64;

#[derive(Clone, Debug)]
pub enum SdfPrimitive {
    Sphere,
    Box,
    Cylinder,
    Torus,
    Plane,
}

#[derive(Clone, Debug)]
pub enum CsgOp {
    Union,
    SmoothUnion,
    Subtract,
    Intersect,
}

#[derive(Clone, Debug)]
pub enum NodeData {
    Primitive {
        kind: SdfPrimitive,
        position: Vec3,
        scale: Vec3,
        color: Vec3,
    },
    Operation {
        op: CsgOp,
        smooth_k: f32,
        left: NodeId,
        right: NodeId,
    },
}

#[derive(Clone, Debug)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub data: NodeData,
}

pub struct Scene {
    pub nodes: HashMap<NodeId, SceneNode>,
    pub root: Option<NodeId>,
    next_id: u64,
    name_counters: HashMap<String, u32>,
}

impl Scene {
    pub fn new() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            root: None,
            next_id: 0,
            name_counters: HashMap::new(),
        };
        let id = scene.create_sphere();
        scene.root = Some(id);
        scene
    }

    fn next_name(&mut self, base: &str) -> String {
        let counter = self.name_counters.entry(base.to_string()).or_insert(0);
        *counter += 1;
        if *counter == 1 {
            base.to_string()
        } else {
            format!("{} {}", base, counter)
        }
    }

    pub fn add_node(&mut self, name: String, data: NodeData) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, SceneNode { id, name, data });
        id
    }

    pub fn remove_node(&mut self, id: NodeId) -> Option<SceneNode> {
        let node = self.nodes.remove(&id);
        if self.root == Some(id) {
            self.root = None;
        }
        // Disconnect any operations referencing this node
        let ops_to_fix: Vec<NodeId> = self
            .nodes
            .values()
            .filter_map(|n| match &n.data {
                NodeData::Operation { left, right, .. }
                    if *left == id || *right == id =>
                {
                    Some(n.id)
                }
                _ => None,
            })
            .collect();
        for op_id in ops_to_fix {
            self.remove_node(op_id);
        }
        node
    }

    // --- Primitive factories ---

    pub fn create_sphere(&mut self) -> NodeId {
        let name = self.next_name("Sphere");
        self.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                scale: Vec3::ONE,
                color: Vec3::new(0.8, 0.3, 0.2),
            },
        )
    }

    pub fn create_box(&mut self) -> NodeId {
        let name = self.next_name("Box");
        self.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Box,
                position: Vec3::new(2.0, 0.0, 0.0),
                scale: Vec3::ONE,
                color: Vec3::new(0.2, 0.5, 0.8),
            },
        )
    }

    pub fn create_cylinder(&mut self) -> NodeId {
        let name = self.next_name("Cylinder");
        self.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Cylinder,
                position: Vec3::new(0.0, 0.0, 2.0),
                scale: Vec3::new(0.5, 1.0, 0.5),
                color: Vec3::new(0.2, 0.8, 0.3),
            },
        )
    }

    pub fn create_torus(&mut self) -> NodeId {
        let name = self.next_name("Torus");
        self.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Torus,
                position: Vec3::new(-2.0, 0.0, 0.0),
                scale: Vec3::new(1.0, 0.3, 1.0),
                color: Vec3::new(0.8, 0.6, 0.2),
            },
        )
    }

    // --- Operation factories ---

    pub fn create_union(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let name = self.next_name("Union");
        self.add_node(
            name,
            NodeData::Operation {
                op: CsgOp::Union,
                smooth_k: 0.0,
                left,
                right,
            },
        )
    }

    pub fn create_smooth_union(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let name = self.next_name("Smooth Union");
        self.add_node(
            name,
            NodeData::Operation {
                op: CsgOp::SmoothUnion,
                smooth_k: 0.5,
                left,
                right,
            },
        )
    }

    pub fn create_subtract(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let name = self.next_name("Subtract");
        self.add_node(
            name,
            NodeData::Operation {
                op: CsgOp::Subtract,
                smooth_k: 0.0,
                left,
                right,
            },
        )
    }

    pub fn create_intersect(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let name = self.next_name("Intersect");
        self.add_node(
            name,
            NodeData::Operation {
                op: CsgOp::Intersect,
                smooth_k: 0.0,
                left,
                right,
            },
        )
    }

    // --- Topology mutation ---

    pub fn set_left_child(&mut self, op_id: NodeId, child_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { left, .. } = &mut node.data {
                *left = child_id;
            }
        }
    }

    pub fn set_right_child(&mut self, op_id: NodeId, child_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { right, .. } = &mut node.data {
                *right = child_id;
            }
        }
    }

    // --- Graph analysis ---

    /// Hash of graph topology only (not parameter values).
    pub fn structure_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.root.hash(&mut hasher);
        self.nodes.len().hash(&mut hasher);
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[&id];
            id.hash(&mut hasher);
            match &node.data {
                NodeData::Primitive { kind, .. } => {
                    0u8.hash(&mut hasher);
                    std::mem::discriminant(kind).hash(&mut hasher);
                }
                NodeData::Operation {
                    op, left, right, ..
                } => {
                    1u8.hash(&mut hasher);
                    std::mem::discriminant(op).hash(&mut hasher);
                    left.hash(&mut hasher);
                    right.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Post-order traversal from root. Returns nodes in evaluation order.
    pub fn topo_order(&self) -> Vec<NodeId> {
        let Some(root) = self.root else {
            return Vec::new();
        };
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        self.topo_visit(root, &mut visited, &mut result);
        result
    }

    fn topo_visit(&self, id: NodeId, visited: &mut HashSet<NodeId>, result: &mut Vec<NodeId>) {
        if !visited.insert(id) {
            return;
        }
        let Some(node) = self.nodes.get(&id) else {
            return;
        };
        if let NodeData::Operation { left, right, .. } = &node.data {
            self.topo_visit(*left, visited, result);
            self.topo_visit(*right, visited, result);
        }
        result.push(id);
    }
}
