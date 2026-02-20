use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use super::voxel::VoxelGrid;

pub type NodeId = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SdfPrimitive {
    Sphere,
    Box,
    Cylinder,
    Torus,
    Plane,
    Cone,
    Capsule,
}

impl SdfPrimitive {
    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Box => "Box",
            Self::Cylinder => "Cylinder",
            Self::Torus => "Torus",
            Self::Plane => "Plane",
            Self::Cone => "Cone",
            Self::Capsule => "Capsule",
        }
    }

    pub fn default_position(&self) -> Vec3 {
        match self {
            Self::Sphere => Vec3::ZERO,
            Self::Box => Vec3::new(2.0, 0.0, 0.0),
            Self::Cylinder => Vec3::new(0.0, 0.0, 2.0),
            Self::Torus => Vec3::new(-2.0, 0.0, 0.0),
            Self::Cone => Vec3::new(0.0, 0.0, -2.0),
            Self::Capsule => Vec3::new(-2.0, 0.0, 2.0),
            Self::Plane => Vec3::ZERO,
        }
    }

    pub fn default_scale(&self) -> Vec3 {
        match self {
            Self::Sphere | Self::Box | Self::Plane => Vec3::ONE,
            Self::Torus => Vec3::new(1.0, 0.3, 1.0),
            Self::Cylinder | Self::Cone => Vec3::new(0.5, 1.0, 0.5),
            Self::Capsule => Vec3::new(0.3, 1.0, 0.3),
        }
    }

    pub fn default_color(&self) -> Vec3 {
        match self {
            Self::Sphere => Vec3::new(0.8, 0.3, 0.2),
            Self::Box => Vec3::new(0.2, 0.5, 0.8),
            Self::Cylinder => Vec3::new(0.2, 0.8, 0.3),
            Self::Torus => Vec3::new(0.8, 0.6, 0.2),
            Self::Cone => Vec3::new(0.7, 0.3, 0.7),
            Self::Capsule => Vec3::new(0.3, 0.7, 0.7),
            Self::Plane => Vec3::new(0.5, 0.5, 0.5),
        }
    }

    pub fn gpu_type_id(&self) -> f32 {
        match self {
            Self::Sphere => 0.0,
            Self::Box => 1.0,
            Self::Cylinder => 2.0,
            Self::Torus => 3.0,
            Self::Plane => 4.0,
            Self::Cone => 5.0,
            Self::Capsule => 6.0,
        }
    }

    pub fn sdf_function_name(&self) -> &'static str {
        match self {
            Self::Sphere => "sdf_sphere",
            Self::Box => "sdf_box",
            Self::Cylinder => "sdf_cylinder",
            Self::Torus => "sdf_torus",
            Self::Plane => "sdf_plane",
            Self::Cone => "sdf_cone",
            Self::Capsule => "sdf_capsule",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Sphere => "[Sph]",
            Self::Box => "[Box]",
            Self::Cylinder => "[Cyl]",
            Self::Torus => "[Tor]",
            Self::Plane => "[Pln]",
            Self::Cone => "[Con]",
            Self::Capsule => "[Cap]",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CsgOp {
    Union,
    SmoothUnion,
    Subtract,
    Intersect,
}

impl CsgOp {
    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Union => "Union",
            Self::SmoothUnion => "Smooth Union",
            Self::Subtract => "Subtract",
            Self::Intersect => "Intersect",
        }
    }

    pub fn default_smooth_k(&self) -> f32 {
        match self {
            Self::SmoothUnion => 0.5,
            _ => 0.0,
        }
    }

    pub fn gpu_op_id(&self) -> f32 {
        match self {
            Self::Union => 10.0,
            Self::SmoothUnion => 11.0,
            Self::Subtract => 12.0,
            Self::Intersect => 13.0,
        }
    }

    pub fn wgsl_function_name(&self) -> &'static str {
        match self {
            Self::Union => "op_union",
            Self::SmoothUnion => "op_smooth_union",
            Self::Subtract => "op_subtract",
            Self::Intersect => "op_intersect",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Union => "[Uni]",
            Self::SmoothUnion => "[SmU]",
            Self::Subtract => "[Sub]",
            Self::Intersect => "[Int]",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeData {
    Primitive {
        kind: SdfPrimitive,
        position: Vec3,
        rotation: Vec3,
        scale: Vec3,
        color: Vec3,
        #[serde(default)]
        voxel_grid: Option<VoxelGrid>,
    },
    Operation {
        op: CsgOp,
        smooth_k: f32,
        left: NodeId,
        right: NodeId,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub data: NodeData,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Scene {
    pub nodes: HashMap<NodeId, SceneNode>,
    pub root: Option<NodeId>,
    pub(crate) next_id: u64,
    pub(crate) name_counters: HashMap<String, u32>,
}

fn voxel_grids_eq(a: &Option<VoxelGrid>, b: &Option<VoxelGrid>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(ga), Some(gb)) => {
            ga.resolution == gb.resolution
                && ga.bounds_min == gb.bounds_min
                && ga.bounds_max == gb.bounds_max
                && ga.data.len() == gb.data.len()
                && ga.data
                    .iter()
                    .zip(gb.data.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}

impl Scene {
    pub fn new() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            root: None,
            next_id: 0,
            name_counters: HashMap::new(),
        };
        let id = scene.create_primitive(SdfPrimitive::Sphere);
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

    // --- Factories ---

    pub fn create_primitive(&mut self, kind: SdfPrimitive) -> NodeId {
        let name = self.next_name(kind.base_name());
        self.add_node(
            name,
            NodeData::Primitive {
                position: kind.default_position(),
                rotation: Vec3::ZERO,
                scale: kind.default_scale(),
                color: kind.default_color(),
                kind,
                voxel_grid: None,
            },
        )
    }

    pub fn create_operation(&mut self, op: CsgOp, left: NodeId, right: NodeId) -> NodeId {
        let name = self.next_name(op.base_name());
        self.add_node(
            name,
            NodeData::Operation {
                smooth_k: op.default_smooth_k(),
                op,
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
                NodeData::Primitive { kind, voxel_grid, .. } => {
                    0u8.hash(&mut hasher);
                    std::mem::discriminant(kind).hash(&mut hasher);
                    voxel_grid.is_some().hash(&mut hasher);
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

    // --- Reachability ---

    /// Returns the set of all NodeIds reachable from root via DFS.
    pub fn reachable_from_root(&self) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        if let Some(root) = self.root {
            self.visit_reachable(root, &mut visited);
        }
        visited
    }

    fn visit_reachable(&self, id: NodeId, visited: &mut HashSet<NodeId>) {
        if !visited.insert(id) {
            return;
        }
        if let Some(node) = self.nodes.get(&id) {
            if let NodeData::Operation { left, right, .. } = &node.data {
                self.visit_reachable(*left, visited);
                self.visit_reachable(*right, visited);
            }
        }
    }

    /// Deep equality check (topology + parameters). Used by undo system.
    pub fn content_eq(&self, other: &Scene) -> bool {
        if self.root != other.root || self.nodes.len() != other.nodes.len() {
            return false;
        }
        for (id, node) in &self.nodes {
            let Some(other_node) = other.nodes.get(id) else {
                return false;
            };
            if node.name != other_node.name {
                return false;
            }
            match (&node.data, &other_node.data) {
                (
                    NodeData::Primitive { kind: k1, position: p1, rotation: r1, scale: s1, color: c1, voxel_grid: v1 },
                    NodeData::Primitive { kind: k2, position: p2, rotation: r2, scale: s2, color: c2, voxel_grid: v2 },
                ) => {
                    if std::mem::discriminant(k1) != std::mem::discriminant(k2)
                        || p1 != p2 || r1 != r2 || s1 != s2 || c1 != c2
                        || !voxel_grids_eq(v1, v2)
                    {
                        return false;
                    }
                }
                (
                    NodeData::Operation { op: o1, smooth_k: k1, left: l1, right: r1 },
                    NodeData::Operation { op: o2, smooth_k: k2, left: l2, right: r2 },
                ) => {
                    if std::mem::discriminant(o1) != std::mem::discriminant(o2)
                        || k1 != k2 || l1 != l2 || r1 != r2
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }
}
