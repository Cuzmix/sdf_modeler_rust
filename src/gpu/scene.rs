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

// ── Scene Node ──────────────────────────────────────────────────

pub struct SdfNode {
    pub name: String,
    pub primitive: SdfPrimitive,
    pub operation: SdfOperation,
    pub position: Vec3,
    pub scale: Vec3,
    pub color: Vec3,
    pub smooth_k: f32,
}

impl SdfNode {
    pub fn new(primitive: SdfPrimitive, name: impl Into<String>) -> Self {
        let (default_scale, default_color) = match primitive {
            SdfPrimitive::Sphere => (Vec3::splat(0.5), Vec3::new(0.8, 0.55, 0.4)),
            SdfPrimitive::Box => (Vec3::splat(0.4), Vec3::new(0.4, 0.6, 0.8)),
            SdfPrimitive::Cylinder => (Vec3::new(0.3, 0.5, 0.3), Vec3::new(0.6, 0.8, 0.4)),
            SdfPrimitive::Torus => (Vec3::new(0.4, 0.15, 0.4), Vec3::new(0.8, 0.4, 0.6)),
            SdfPrimitive::Plane => (Vec3::ONE, Vec3::new(0.25, 0.25, 0.3)),
        };

        Self {
            name: name.into(),
            primitive,
            operation: SdfOperation::SmoothUnion,
            position: Vec3::new(0.0, 0.5, 0.0),
            scale: default_scale,
            color: default_color,
            smooth_k: 0.1,
        }
    }

}

// ── Scene ───────────────────────────────────────────────────────

pub struct Scene {
    pub nodes: Vec<SdfNode>,
    pub selected: Option<usize>,
    node_counter: u32, // for auto-naming
}

impl Scene {
    pub fn default_scene() -> Self {
        let mut scene = Self {
            nodes: Vec::new(),
            selected: None,
            node_counter: 0,
        };
        scene.add_node(SdfPrimitive::Sphere);
        scene
    }

    pub fn add_node(&mut self, primitive: SdfPrimitive) -> usize {
        self.node_counter += 1;
        let name = match primitive {
            SdfPrimitive::Sphere => format!("Sphere {}", self.node_counter),
            SdfPrimitive::Box => format!("Box {}", self.node_counter),
            SdfPrimitive::Cylinder => format!("Cylinder {}", self.node_counter),
            SdfPrimitive::Torus => format!("Torus {}", self.node_counter),
            SdfPrimitive::Plane => format!("Plane {}", self.node_counter),
        };
        self.nodes.push(SdfNode::new(primitive, name));
        let idx = self.nodes.len() - 1;
        self.selected = Some(idx);
        idx
    }

    pub fn remove_selected(&mut self) {
        if let Some(idx) = self.selected {
            if idx < self.nodes.len() {
                self.nodes.remove(idx);
            }
            self.selected = None;
        }
    }

    /// Pack all nodes into GPU-ready structs.
    pub fn pack_nodes(&self) -> Vec<SdfNodeGpu> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let selected = if self.selected == Some(i) { 1.0 } else { 0.0 };
                SdfNodeGpu {
                    type_op: [
                        node.primitive as u32 as f32,
                        node.operation as u32 as f32,
                        node.smooth_k,
                        selected,
                    ],
                    position: [node.position.x, node.position.y, node.position.z, 0.0],
                    scale: [node.scale.x, node.scale.y, node.scale.z, 0.0],
                    color: [node.color.x, node.color.y, node.color.z, 1.0],
                    _reserved: [0.0; 4],
                }
            })
            .collect()
    }

    /// Pack scene metadata for the GPU.
    pub fn pack_info(&self) -> SceneInfoGpu {
        SceneInfoGpu {
            node_count: self.nodes.len() as u32,
            selected_idx: self.selected.map_or(-1, |i| i as i32),
            _pad: [0; 2],
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
