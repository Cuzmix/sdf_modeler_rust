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
pub enum TransformKind {
    Translate,
    Rotate,
    Scale,
}

impl TransformKind {
    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Translate => "Translate",
            Self::Rotate => "Rotate",
            Self::Scale => "Scale",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Translate => "[Trl]",
            Self::Rotate => "[Rot]",
            Self::Scale => "[Scl]",
        }
    }

    pub fn default_value(&self) -> Vec3 {
        match self {
            Self::Translate => Vec3::ZERO,
            Self::Rotate => Vec3::ZERO,
            Self::Scale => Vec3::ONE,
        }
    }

    pub fn gpu_type_id(&self) -> f32 {
        match self {
            Self::Translate => 21.0,
            Self::Rotate => 22.0,
            Self::Scale => 23.0,
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
        /// Legacy: kept for v2 save file migration only. Always None at runtime.
        #[serde(default, skip_serializing)]
        voxel_grid: Option<VoxelGrid>,
    },
    Operation {
        op: CsgOp,
        smooth_k: f32,
        left: Option<NodeId>,
        right: Option<NodeId>,
    },
    Sculpt {
        input: Option<NodeId>,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
        #[serde(default = "crate::graph::voxel::default_resolution")]
        desired_resolution: u32,
    },
    Transform {
        kind: TransformKind,
        input: Option<NodeId>,
        value: Vec3,
    },
}

impl NodeData {
    /// Iterate over child node IDs (0-2 children depending on variant).
    pub fn children(&self) -> impl Iterator<Item = NodeId> {
        let (a, b) = match self {
            NodeData::Operation { left, right, .. } => (*left, *right),
            NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => (*input, None),
            NodeData::Primitive { .. } => (None, None),
        };
        a.into_iter().chain(b)
    }

    /// For geometry nodes (Primitive/Sculpt), return local bounding sphere (center, radius).
    /// Returns None for non-geometry nodes.
    pub fn geometry_local_sphere(&self) -> Option<([f32; 3], f32)> {
        match self {
            NodeData::Primitive { position, scale, .. } => {
                let r = scale.x.max(scale.y).max(scale.z);
                Some(([position.x, position.y, position.z], r))
            }
            NodeData::Sculpt { position, voxel_grid, .. } => {
                let mid = [
                    position.x + (voxel_grid.bounds_min.x + voxel_grid.bounds_max.x) * 0.5,
                    position.y + (voxel_grid.bounds_min.y + voxel_grid.bounds_max.y) * 0.5,
                    position.z + (voxel_grid.bounds_min.z + voxel_grid.bounds_max.z) * 0.5,
                ];
                let r = ((voxel_grid.bounds_max.x - voxel_grid.bounds_min.x) * 0.5)
                    .max((voxel_grid.bounds_max.y - voxel_grid.bounds_min.y) * 0.5)
                    .max((voxel_grid.bounds_max.z - voxel_grid.bounds_min.z) * 0.5);
                Some((mid, r))
            }
            _ => None,
        }
    }
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
    pub(crate) next_id: u64,
    pub(crate) name_counters: HashMap<String, u32>,
}

impl Scene {
    pub fn new() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
        };
        scene.create_primitive(SdfPrimitive::Sphere);
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
        // Null out any references to this node (instead of cascade-deleting)
        let to_patch: Vec<(NodeId, bool, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| match &n.data {
                NodeData::Operation { left, right, .. } => {
                    let is_left = *left == Some(id);
                    let is_right = *right == Some(id);
                    if is_left || is_right {
                        Some((n.id, is_left, is_right, false))
                    } else {
                        None
                    }
                }
                NodeData::Sculpt { input, .. } if *input == Some(id) => {
                    Some((n.id, false, false, true))
                }
                NodeData::Transform { input, .. } if *input == Some(id) => {
                    Some((n.id, false, false, true))
                }
                _ => None,
            })
            .collect();
        for (parent_id, is_left, is_right, is_single_input) in to_patch {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = None;
                        }
                        if is_right {
                            *right = None;
                        }
                    }
                    NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. }
                        if is_single_input =>
                    {
                        *input = None;
                    }
                    _ => {}
                }
            }
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
                left: Some(left),
                right: Some(right),
            },
        )
    }

    pub fn create_sculpt(
        &mut self,
        input: NodeId,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
    ) -> NodeId {
        let name = self.next_name("Sculpt");
        let desired_resolution = voxel_grid.resolution;
        self.add_node(
            name,
            NodeData::Sculpt {
                input: Some(input),
                position,
                rotation,
                color,
                voxel_grid,
                desired_resolution,
            },
        )
    }

    pub fn create_transform(&mut self, kind: TransformKind, input: Option<NodeId>) -> NodeId {
        let name = self.next_name(kind.base_name());
        let value = kind.default_value();
        self.add_node(name, NodeData::Transform { kind, input, value })
    }

    /// Insert a Transform modifier above `target_id`.
    /// Creates a Transform node with `input = target_id` and rewires all parents.
    pub fn insert_transform_above(&mut self, target_id: NodeId, kind: TransformKind) -> NodeId {
        let transform_id = self.create_transform(kind, Some(target_id));

        // Rewire all parents that referenced target_id
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| {
                if n.id == transform_id {
                    return None;
                }
                match &n.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(target_id);
                        let is_right = *right == Some(target_id);
                        if is_left || is_right {
                            Some((n.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Transform { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(transform_id);
                        }
                        if is_right {
                            *right = Some(transform_id);
                        }
                    }
                    NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => {
                        *input = Some(transform_id);
                    }
                    _ => {}
                }
            }
        }

        transform_id
    }

    /// Insert a Sculpt modifier above `target_id`.
    /// Creates a Sculpt node with `input = target_id` and rewires all parents.
    pub fn insert_sculpt_above(
        &mut self,
        target_id: NodeId,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
    ) -> NodeId {
        let sculpt_id = self.create_sculpt(target_id, position, rotation, color, voxel_grid);

        // Rewire all parents that referenced target_id
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| {
                if n.id == sculpt_id {
                    return None;
                }
                match &n.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(target_id);
                        let is_right = *right == Some(target_id);
                        if is_left || is_right {
                            Some((n.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Transform { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(sculpt_id);
                        }
                        if is_right {
                            *right = Some(sculpt_id);
                        }
                    }
                    NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => {
                        *input = Some(sculpt_id);
                    }
                    _ => {}
                }
            }
        }

        sculpt_id
    }

    // --- Topology mutation ---

    pub fn set_left_child(&mut self, op_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { left, .. } = &mut node.data {
                *left = child_id;
            }
        }
    }

    pub fn set_right_child(&mut self, op_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { right, .. } = &mut node.data {
                *right = child_id;
            }
        }
    }

    pub fn set_sculpt_input(&mut self, node_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            match &mut node.data {
                NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => {
                    *input = child_id;
                }
                _ => {}
            }
        }
    }

    pub fn swap_children(&mut self, op_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { left, right, .. } = &mut node.data {
                std::mem::swap(left, right);
            }
        }
    }

    // --- Graph analysis ---

    /// Hash of graph topology only (not parameter values).
    pub fn structure_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
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
                NodeData::Sculpt {
                    input, voxel_grid, ..
                } => {
                    2u8.hash(&mut hasher);
                    input.hash(&mut hasher);
                    voxel_grid.resolution.hash(&mut hasher);
                }
                NodeData::Transform { kind, input, .. } => {
                    3u8.hash(&mut hasher);
                    std::mem::discriminant(kind).hash(&mut hasher);
                    input.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Lightweight hash of all mutable node parameters (positions, rotations,
    /// scales, colors, smooth_k, etc.). Skips voxel data — voxel changes are
    /// tracked explicitly via dirty flags at brush/undo/redo sites.
    pub fn data_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[&id];
            id.hash(&mut hasher);
            node.name.hash(&mut hasher);
            match &node.data {
                NodeData::Primitive {
                    position, rotation, scale, color, ..
                } => {
                    position.x.to_bits().hash(&mut hasher);
                    position.y.to_bits().hash(&mut hasher);
                    position.z.to_bits().hash(&mut hasher);
                    rotation.x.to_bits().hash(&mut hasher);
                    rotation.y.to_bits().hash(&mut hasher);
                    rotation.z.to_bits().hash(&mut hasher);
                    scale.x.to_bits().hash(&mut hasher);
                    scale.y.to_bits().hash(&mut hasher);
                    scale.z.to_bits().hash(&mut hasher);
                    color.x.to_bits().hash(&mut hasher);
                    color.y.to_bits().hash(&mut hasher);
                    color.z.to_bits().hash(&mut hasher);
                }
                NodeData::Operation { smooth_k, .. } => {
                    smooth_k.to_bits().hash(&mut hasher);
                }
                NodeData::Sculpt {
                    position, rotation, color, desired_resolution, ..
                } => {
                    position.x.to_bits().hash(&mut hasher);
                    position.y.to_bits().hash(&mut hasher);
                    position.z.to_bits().hash(&mut hasher);
                    rotation.x.to_bits().hash(&mut hasher);
                    rotation.y.to_bits().hash(&mut hasher);
                    rotation.z.to_bits().hash(&mut hasher);
                    color.x.to_bits().hash(&mut hasher);
                    color.y.to_bits().hash(&mut hasher);
                    color.z.to_bits().hash(&mut hasher);
                    desired_resolution.hash(&mut hasher);
                }
                NodeData::Transform { value, .. } => {
                    value.x.to_bits().hash(&mut hasher);
                    value.y.to_bits().hash(&mut hasher);
                    value.z.to_bits().hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Returns nodes not referenced as a child by any other node.
    pub fn top_level_nodes(&self) -> Vec<NodeId> {
        let referenced: HashSet<NodeId> = self.nodes.values()
            .flat_map(|n| n.data.children())
            .collect();
        let mut top: Vec<NodeId> = self.nodes.keys()
            .filter(|id| !referenced.contains(id))
            .cloned()
            .collect();
        top.sort();
        top
    }

    /// Post-order traversal from all top-level nodes. Returns nodes in evaluation order.
    pub fn topo_order(&self) -> Vec<NodeId> {
        let tops = self.top_level_nodes();
        if tops.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        for &root in &tops {
            self.topo_visit(root, &mut visited, &mut result);
        }
        result
    }

    fn topo_visit(&self, id: NodeId, visited: &mut HashSet<NodeId>, result: &mut Vec<NodeId>) {
        if !visited.insert(id) { return; }
        let Some(node) = self.nodes.get(&id) else { return; };
        for child_id in node.data.children() {
            self.topo_visit(child_id, visited, result);
        }
        result.push(id);
    }

    // --- Tree traversal helpers ---

    /// Build a parent map: child_id → parent_id.
    pub fn build_parent_map(&self) -> HashMap<NodeId, NodeId> {
        let mut map = HashMap::new();
        for node in self.nodes.values() {
            for child_id in node.data.children() {
                map.insert(child_id, node.id);
            }
        }
        map
    }

    /// Walk up from a leaf through ancestor transforms and compute world-space
    /// bounding sphere (center, radius).
    fn walk_transforms_sphere(
        &self,
        center: [f32; 3],
        extent: f32,
        leaf_id: NodeId,
        parent_map: &HashMap<NodeId, NodeId>,
    ) -> ([f32; 3], f32) {
        let mut wc = center;
        let mut wr = extent;
        let mut current = leaf_id;
        while let Some(&pid) = parent_map.get(&current) {
            if let Some(parent) = self.nodes.get(&pid) {
                match &parent.data {
                    NodeData::Transform { kind: TransformKind::Translate, value, .. } => {
                        wc[0] += value.x;
                        wc[1] += value.y;
                        wc[2] += value.z;
                    }
                    NodeData::Transform { kind: TransformKind::Scale, value, .. } => {
                        let s = value.x.abs().max(value.y.abs()).max(value.z.abs());
                        wr *= s;
                        wc[0] *= value.x;
                        wc[1] *= value.y;
                        wc[2] *= value.z;
                    }
                    NodeData::Transform { kind: TransformKind::Rotate, .. } => {
                        let dist = (wc[0] * wc[0] + wc[1] * wc[1] + wc[2] * wc[2]).sqrt();
                        wr += dist;
                        wc = [0.0, 0.0, 0.0];
                    }
                    _ => {}
                }
            }
            current = pid;
        }
        (wc, wr)
    }

    /// Check if any node in the subtree rooted at `root` is a Sculpt node.
    pub fn subtree_has_sculpt(&self, root: NodeId) -> bool {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(node) = self.nodes.get(&id) {
                if matches!(node.data, NodeData::Sculpt { .. }) {
                    return true;
                }
                stack.extend(node.data.children());
            }
        }
        false
    }

    /// Collect all node IDs in the subtree rooted at `root`.
    pub fn collect_subtree(&self, root: NodeId) -> HashSet<NodeId> {
        let mut set = HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !set.insert(id) { continue; }
            if let Some(node) = self.nodes.get(&id) {
                stack.extend(node.data.children());
            }
        }
        set
    }

    /// Compute world-space bounding sphere (center, radius) for a subtree.
    /// `parent_map` should be pre-built via `build_parent_map()`.
    pub fn compute_subtree_sphere(
        &self,
        root: NodeId,
        parent_map: &HashMap<NodeId, NodeId>,
    ) -> ([f32; 3], f32) {
        let subtree_nodes = self.collect_subtree(root);
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        let mut has_geom = false;
        let mut max_smooth_k: f32 = 0.0;

        for &nid in &subtree_nodes {
            if let Some(node) = self.nodes.get(&nid) {
                if let NodeData::Operation { smooth_k, .. } = &node.data {
                    max_smooth_k = max_smooth_k.max(*smooth_k);
                }
                if let Some((center, extent)) = node.data.geometry_local_sphere() {
                    let (wc, wr) = self.walk_transforms_sphere(center, extent, nid, parent_map);
                    for i in 0..3 {
                        bmin[i] = bmin[i].min(wc[i] - wr);
                        bmax[i] = bmax[i].max(wc[i] + wr);
                    }
                    has_geom = true;
                }
            }
        }

        if !has_geom {
            return ([0.0; 3], 1.0);
        }

        let pad = max_smooth_k + 0.5;
        let center = [
            (bmin[0] + bmax[0]) * 0.5,
            (bmin[1] + bmax[1]) * 0.5,
            (bmin[2] + bmax[2]) * 0.5,
        ];
        let half = [
            (bmax[0] - bmin[0]) * 0.5 + pad,
            (bmax[1] - bmin[1]) * 0.5 + pad,
            (bmax[2] - bmin[2]) * 0.5 + pad,
        ];
        let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
        (center, radius)
    }

    /// Flatten a subtree into a single standalone Sculpt node.
    /// Replaces the entire subtree rooted at `subtree_root` with a new Sculpt node
    /// containing the pre-baked `voxel_grid`. Returns the new node's ID.
    pub fn flatten_subtree(
        &mut self,
        subtree_root: NodeId,
        voxel_grid: VoxelGrid,
        center: Vec3,
        color: Vec3,
    ) -> NodeId {
        // 1. Collect all nodes in the subtree (to delete later)
        let subtree_ids = self.collect_subtree(subtree_root);

        // 2. Find parents that reference subtree_root and record their rewiring info
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| {
                if subtree_ids.contains(&n.id) {
                    return None; // Skip nodes that are part of the subtree
                }
                match &n.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(subtree_root);
                        let is_right = *right == Some(subtree_root);
                        if is_left || is_right {
                            Some((n.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. } if *input == Some(subtree_root) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Transform { input, .. } if *input == Some(subtree_root) => {
                        Some((n.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        // 3. Add the new standalone Sculpt node (input: None)
        let desired_resolution = voxel_grid.resolution;
        let name = self.next_name("Sculpt");
        let new_id = self.add_node(
            name,
            NodeData::Sculpt {
                input: None,
                position: center,
                rotation: Vec3::ZERO,
                color,
                voxel_grid,
                desired_resolution,
            },
        );

        // 4. Rewire parents: subtree_root → new_id
        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(new_id);
                        }
                        if is_right {
                            *right = Some(new_id);
                        }
                    }
                    NodeData::Sculpt { input, .. } | NodeData::Transform { input, .. } => {
                        *input = Some(new_id);
                    }
                    _ => {}
                }
            }
        }

        // 5. Delete all subtree nodes (now orphaned)
        for id in subtree_ids {
            self.nodes.remove(&id);
        }

        new_id
    }

    /// Compute a conservative world-space AABB encompassing all scene geometry.
    /// Walks the tree to properly compose transforms with their children's bounds.
    pub fn compute_bounds(&self) -> ([f32; 3], [f32; 3]) {
        let parent_map = self.build_parent_map();
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        let mut has_geometry = false;

        for node in self.nodes.values() {
            if let Some((center, extent)) = node.data.geometry_local_sphere() {
                let (wc, wr) = self.walk_transforms_sphere(center, extent, node.id, &parent_map);
                for i in 0..3 {
                    bmin[i] = bmin[i].min(wc[i] - wr);
                    bmax[i] = bmax[i].max(wc[i] + wr);
                }
                has_geometry = true;
            }
        }

        if !has_geometry {
            return ([-5.0; 3], [5.0; 3]);
        }

        for i in 0..3 {
            bmin[i] -= 1.5;
            bmax[i] += 1.5;
        }
        (bmin, bmax)
    }

    /// Deep equality check (topology + parameters). Used by undo system.
    pub fn content_eq(&self, other: &Scene) -> bool {
        if self.nodes.len() != other.nodes.len() {
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
                    NodeData::Primitive {
                        kind: k1,
                        position: p1,
                        rotation: r1,
                        scale: s1,
                        color: c1,
                        ..
                    },
                    NodeData::Primitive {
                        kind: k2,
                        position: p2,
                        rotation: r2,
                        scale: s2,
                        color: c2,
                        ..
                    },
                ) => {
                    if std::mem::discriminant(k1) != std::mem::discriminant(k2)
                        || p1 != p2
                        || r1 != r2
                        || s1 != s2
                        || c1 != c2
                    {
                        return false;
                    }
                }
                (
                    NodeData::Operation {
                        op: o1,
                        smooth_k: k1,
                        left: l1,
                        right: r1,
                    },
                    NodeData::Operation {
                        op: o2,
                        smooth_k: k2,
                        left: l2,
                        right: r2,
                    },
                ) => {
                    if std::mem::discriminant(o1) != std::mem::discriminant(o2)
                        || k1 != k2
                        || l1 != l2
                        || r1 != r2
                    {
                        return false;
                    }
                }
                (
                    NodeData::Sculpt {
                        input: i1,
                        position: p1,
                        rotation: r1,
                        color: c1,
                        voxel_grid: v1,
                        desired_resolution: dr1,
                    },
                    NodeData::Sculpt {
                        input: i2,
                        position: p2,
                        rotation: r2,
                        color: c2,
                        voxel_grid: v2,
                        desired_resolution: dr2,
                    },
                ) => {
                    if i1 != i2
                        || p1 != p2
                        || r1 != r2
                        || c1 != c2
                        || dr1 != dr2
                        || !v1.content_eq(v2)
                    {
                        return false;
                    }
                }
                (
                    NodeData::Transform {
                        kind: k1,
                        input: i1,
                        value: v1,
                    },
                    NodeData::Transform {
                        kind: k2,
                        input: i2,
                        value: v2,
                    },
                ) => {
                    if std::mem::discriminant(k1) != std::mem::discriminant(k2)
                        || i1 != i2
                        || v1 != v2
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
