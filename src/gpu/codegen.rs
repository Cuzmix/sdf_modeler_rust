use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SdfNodeGpu {
    pub type_op: [f32; 4],
    pub position: [f32; 4],
    pub scale: [f32; 4],
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// Shader template: everything before scene_sdf
// ---------------------------------------------------------------------------

const SHADER_PRELUDE: &str = r#"
struct Camera {
    inv_view_proj: mat4x4f,
    eye: vec4f,
    viewport: vec4f,
    time: f32,
}

struct SdfNode {
    type_op: vec4f,
    position: vec4f,
    scale: vec4f,
    color: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> nodes: array<SdfNode>;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4f(x, y, 0.0, 1.0);
    return out;
}

// --- SDF Primitives ---

fn sdf_sphere(p: vec3f, s: vec3f) -> f32 {
    return length(p) - s.x;
}

fn sdf_box(p: vec3f, s: vec3f) -> f32 {
    let q = abs(p) - s;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_cylinder(p: vec3f, s: vec3f) -> f32 {
    let d = abs(vec2f(length(p.xz), p.y)) - vec2f(s.x, s.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdf_torus(p: vec3f, s: vec3f) -> f32 {
    let q = vec2f(length(p.xz) - s.x, p.y);
    return length(q) - s.y;
}

fn sdf_plane(p: vec3f, s: vec3f) -> f32 {
    return p.y;
}

// --- CSG Operations (operate on vec2f: x=distance, y=material_id) ---

fn op_union(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if a.x < b.x { return a; } else { return b; }
}

fn op_smooth_union(a: vec2f, b: vec2f, k: f32) -> vec2f {
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(k, 0.0001), 0.0, 1.0);
    let d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    let mat = select(b.y, a.y, a.x < b.x);
    return vec2f(d, mat);
}

fn op_subtract(a: vec2f, b: vec2f, k: f32) -> vec2f {
    let d = max(a.x, -b.x);
    return vec2f(d, a.y);
}

fn op_intersect(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if a.x > b.x { return a; } else { return b; }
}
"#;

// ---------------------------------------------------------------------------
// Shader template: everything after scene_sdf
// ---------------------------------------------------------------------------

const SHADER_POSTLUDE: &str = r#"
fn ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    var t = 0.0;
    var mat_id = -1.0;
    for (var i = 0; i < 128; i++) {
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        if hit.x < 0.001 {
            mat_id = hit.y;
            break;
        }
        t += hit.x * 0.8;
        mat_id = hit.y;
        if t > 50.0 { break; }
    }
    return vec2f(t, mat_id);
}

fn calc_normal(p: vec3f) -> vec3f {
    let e = vec2f(0.001, 0.0);
    return normalize(vec3f(
        scene_sdf(p + e.xyy).x - scene_sdf(p - e.xyy).x,
        scene_sdf(p + e.yxy).x - scene_sdf(p - e.yxy).x,
        scene_sdf(p + e.yyx).x - scene_sdf(p - e.yyx).x,
    ));
}

fn get_node_color(mat_id: i32) -> vec3f {
    if mat_id < 0 { return vec3f(0.5); }
    return nodes[mat_id].color.xyz;
}

fn is_selected(mat_id: i32) -> bool {
    if mat_id < 0 { return false; }
    return nodes[mat_id].color.w > 0.5;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let local = frag_coord.xy - camera.viewport.xy;
    let uv = local / camera.viewport.zw * 2.0 - 1.0;

    let ndc = vec4f(uv.x, -uv.y, 1.0, 1.0);
    let world = camera.inv_view_proj * ndc;
    let rd = normalize(world.xyz / world.w - camera.eye.xyz);
    let ro = camera.eye.xyz;

    let hit = ray_march(ro, rd);
    let t = hit.x;
    let mat_id = i32(hit.y + 0.5);

    if t > 49.0 {
        let bg = mix(vec3f(0.08, 0.08, 0.12), vec3f(0.02, 0.02, 0.04), uv.y * 0.5 + 0.5);
        return vec4f(bg, 1.0);
    }

    let p = ro + rd * t;
    let n = calc_normal(p);
    let light_dir = normalize(vec3f(1.0, 2.0, 3.0));
    let h = normalize(light_dir - rd);

    let ambient = 0.08;
    let diffuse = max(dot(n, light_dir), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 32.0);

    let albedo = get_node_color(mat_id);
    var color = albedo * (ambient + diffuse * 0.9) + vec3f(1.0) * specular * 0.5;

    if is_selected(mat_id) {
        let rim = 1.0 - max(dot(n, -rd), 0.0);
        let rim_factor = pow(rim, 2.0) * 0.6;
        color = mix(color, vec3f(1.0, 0.8, 0.2), rim_factor);
    }

    return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

pub fn generate_shader(scene: &Scene) -> String {
    let scene_sdf = generate_scene_sdf(scene);
    format!("{}\n{}\n{}", SHADER_PRELUDE, scene_sdf, SHADER_POSTLUDE)
}

fn generate_scene_sdf(scene: &Scene) -> String {
    let order = scene.topo_order();
    if order.is_empty() {
        return "fn scene_sdf(p: vec3f) -> vec2f {\n    return vec2f(1e10, -1.0);\n}"
            .to_string();
    }

    let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut lines = Vec::new();
    lines.push("fn scene_sdf(p: vec3f) -> vec2f {".to_string());

    for (i, &node_id) in order.iter().enumerate() {
        let Some(node) = scene.nodes.get(&node_id) else {
            continue;
        };
        match &node.data {
            NodeData::Primitive { kind, .. } => {
                let sdf_fn = match kind {
                    SdfPrimitive::Sphere => "sdf_sphere",
                    SdfPrimitive::Box => "sdf_box",
                    SdfPrimitive::Cylinder => "sdf_cylinder",
                    SdfPrimitive::Torus => "sdf_torus",
                    SdfPrimitive::Plane => "sdf_plane",
                };
                lines.push(format!(
                    "    let n{i} = vec2f({sdf_fn}(p - nodes[{i}].position.xyz, nodes[{i}].scale.xyz), f32({i}));"
                ));
            }
            NodeData::Operation { op, left, right, .. } => {
                let li = idx_map.get(left).copied().unwrap_or(0);
                let ri = idx_map.get(right).copied().unwrap_or(0);
                let op_fn = match op {
                    CsgOp::Union => "op_union",
                    CsgOp::SmoothUnion => "op_smooth_union",
                    CsgOp::Subtract => "op_subtract",
                    CsgOp::Intersect => "op_intersect",
                };
                lines.push(format!(
                    "    let n{i} = {op_fn}(n{li}, n{ri}, nodes[{i}].type_op.y);"
                ));
            }
        }
    }

    let root_idx = order.len() - 1;
    lines.push(format!("    return n{root_idx};"));
    lines.push("}".to_string());

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// GPU buffer builder
// ---------------------------------------------------------------------------

pub fn build_node_buffer(scene: &Scene, selected: Option<NodeId>) -> Vec<SdfNodeGpu> {
    let order = scene.topo_order();
    let mut buffer = Vec::with_capacity(order.len().max(1));

    for &node_id in &order {
        let Some(node) = scene.nodes.get(&node_id) else {
            buffer.push(SdfNodeGpu::zeroed());
            continue;
        };
        let is_sel = if selected == Some(node_id) { 1.0 } else { 0.0 };

        match &node.data {
            NodeData::Primitive {
                kind,
                position,
                scale,
                color,
            } => {
                let type_val = match kind {
                    SdfPrimitive::Sphere => 0.0,
                    SdfPrimitive::Box => 1.0,
                    SdfPrimitive::Cylinder => 2.0,
                    SdfPrimitive::Torus => 3.0,
                    SdfPrimitive::Plane => 4.0,
                };
                buffer.push(SdfNodeGpu {
                    type_op: [type_val, 0.0, 0.0, 0.0],
                    position: [position.x, position.y, position.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, 0.0],
                    color: [color.x, color.y, color.z, is_sel],
                });
            }
            NodeData::Operation { op, smooth_k, .. } => {
                let op_val = match op {
                    CsgOp::Union => 10.0,
                    CsgOp::SmoothUnion => 11.0,
                    CsgOp::Subtract => 12.0,
                    CsgOp::Intersect => 13.0,
                };
                buffer.push(SdfNodeGpu {
                    type_op: [op_val, *smooth_k, 0.0, 0.0],
                    position: [0.0; 4],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                });
            }
        }
    }

    // Ensure at least one element (avoid zero-sized buffer)
    if buffer.is_empty() {
        buffer.push(SdfNodeGpu::zeroed());
    }

    buffer
}
