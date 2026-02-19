use glam::Vec3;
use std::collections::HashMap;

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
}

impl Scene {
    pub fn new() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            root: None,
            next_id: 0,
        };
        let id = scene.add_node(
            "Sphere".to_string(),
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                scale: Vec3::ONE,
                color: Vec3::new(0.8, 0.3, 0.2),
            },
        );
        scene.root = Some(id);
        scene
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
        node
    }
}
