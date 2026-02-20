use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::graph::scene::{NodeData, NodeId, Scene};

/// 128-byte GPU node (8 x vec4f). Expanded for rotation + future growth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SdfNodeGpu {
    pub type_op: [f32; 4],   // [type_val, smooth_k, 0, 0]
    pub position: [f32; 4],  // [x, y, z, 0]
    pub rotation: [f32; 4],  // [rx, ry, rz, 0] (radians)
    pub scale: [f32; 4],     // [sx, sy, sz, 0]
    pub color: [f32; 4],     // [r, g, b, is_selected]
    pub extra0: [f32; 4],    // reserved: future material
    pub extra1: [f32; 4],    // reserved: future modifiers
    pub extra2: [f32; 4],    // reserved: future flags
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
    rotation: vec4f,
    scale: vec4f,
    color: vec4f,
    extra0: vec4f,
    extra1: vec4f,
    extra2: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> nodes: array<SdfNode>;
@group(1) @binding(1) var<storage, read> voxel_data: array<f32>;

fn sdf_voxel_grid(local_p: vec3f, node_idx: u32) -> f32 {
    let offset = u32(nodes[node_idx].extra0.x);
    let res    = u32(nodes[node_idx].extra0.y);
    let bmin   = nodes[node_idx].extra1.xyz;
    let bmax   = nodes[node_idx].extra2.xyz;

    let size = bmax - bmin;
    let norm = (local_p - bmin) / size;
    let max_c = f32(res - 1u);
    let gc = clamp(norm * max_c, vec3f(0.0), vec3f(max_c));

    let i0 = vec3<u32>(vec3f(floor(gc.x), floor(gc.y), floor(gc.z)));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(res - 1u));
    let f = fract(gc);

    let r2 = res * res;
    let c000 = voxel_data[offset + i0.z * r2 + i0.y * res + i0.x];
    let c100 = voxel_data[offset + i0.z * r2 + i0.y * res + i1.x];
    let c010 = voxel_data[offset + i0.z * r2 + i1.y * res + i0.x];
    let c110 = voxel_data[offset + i0.z * r2 + i1.y * res + i1.x];
    let c001 = voxel_data[offset + i1.z * r2 + i0.y * res + i0.x];
    let c101 = voxel_data[offset + i1.z * r2 + i0.y * res + i1.x];
    let c011 = voxel_data[offset + i1.z * r2 + i1.y * res + i0.x];
    let c111 = voxel_data[offset + i1.z * r2 + i1.y * res + i1.x];

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    let c0  = mix(c00, c10, f.y);
    let c1  = mix(c01, c11, f.y);
    return mix(c0, c1, f.z);
}

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

// --- Euler rotation (XYZ order) ---

fn rotate_euler(p: vec3f, r: vec3f) -> vec3f {
    var q = p;
    // Rotate X
    let cx = cos(r.x); let sx = sin(r.x);
    q = vec3f(q.x, cx * q.y - sx * q.z, sx * q.y + cx * q.z);
    // Rotate Y
    let cy = cos(r.y); let sy = sin(r.y);
    q = vec3f(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
    // Rotate Z
    let cz = cos(r.z); let sz = sin(r.z);
    q = vec3f(cz * q.x - sz * q.y, sz * q.x + cz * q.y, q.z);
    return q;
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

fn sdf_cone(p: vec3f, s: vec3f) -> f32 {
    // Cone with radius s.x and height s.y, tip at origin pointing up
    let q = vec2f(length(p.xz), p.y);
    let tip = vec2f(0.0, s.y);
    let base = vec2f(s.x, 0.0);
    let ab = base - tip;
    let aq = q - tip;
    let t = clamp(dot(aq, ab) / dot(ab, ab), 0.0, 1.0);
    let closest = tip + ab * t;
    let d_side = length(q - closest);
    // Inside/outside sign
    let cross2d = ab.x * aq.y - ab.y * aq.x;
    let sign_val = select(1.0, -1.0, cross2d < 0.0 && q.y > 0.0 && q.y < s.y);
    return d_side * sign_val;
}

fn sdf_capsule(p: vec3f, s: vec3f) -> f32 {
    // Capsule along Y axis: radius s.x, half-height s.y
    let h = s.y;
    let r = s.x;
    let py = clamp(p.y, -h, h);
    return length(p - vec3f(0.0, py, 0.0)) - r;
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
// --- Rendering quality constants ---
const SHADOW_STEPS: i32 = 32;
const SHADOW_PENUMBRA_K: f32 = 8.0;
const AO_SAMPLES: i32 = 5;
const AO_STEP: f32 = 0.08;
const AO_DECAY: f32 = 0.95;

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

fn soft_shadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    for (var i = 0; i < SHADOW_STEPS; i++) {
        let h = scene_sdf(ro + rd * t).x;
        if h < 0.001 {
            return 0.0;
        }
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.2);
        if t > maxt { break; }
    }
    return clamp(res, 0.0, 1.0);
}

fn calc_ao(p: vec3f, n: vec3f) -> f32 {
    var occ = 0.0;
    var weight = 1.0;
    for (var i = 1; i <= AO_SAMPLES; i++) {
        let dist = AO_STEP * f32(i);
        let d = scene_sdf(p + n * dist).x;
        occ += (dist - d) * weight;
        weight *= AO_DECAY;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
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

    // Sky gradient
    if t > 49.0 {
        let sky_t = uv.y * 0.5 + 0.5;
        let bg = mix(vec3f(0.10, 0.10, 0.16), vec3f(0.02, 0.02, 0.05), sky_t);
        return vec4f(bg, 1.0);
    }

    let p = ro + rd * t;
    let n = calc_normal(p);

    // Key light
    let key_dir = normalize(vec3f(1.0, 2.0, 3.0));
    let key_h = normalize(key_dir - rd);
    let key_diff = max(dot(n, key_dir), 0.0);
    let key_spec = pow(max(dot(n, key_h), 0.0), 32.0);
    let shadow = soft_shadow(p + n * 0.01, key_dir, 0.02, 20.0, SHADOW_PENUMBRA_K);

    // Fill light (opposite side, dimmer, no shadow)
    let fill_dir = normalize(vec3f(-1.0, 0.5, -1.0));
    let fill_diff = max(dot(n, fill_dir), 0.0) * 0.25;

    // Ambient occlusion
    let ao = calc_ao(p, n);

    let albedo = get_node_color(mat_id);
    let ambient = 0.06 * ao;
    var color = albedo * (ambient + key_diff * shadow * 0.85 + fill_diff)
              + vec3f(1.0) * key_spec * shadow * 0.4;

    if is_selected(mat_id) {
        let rim = 1.0 - max(dot(n, -rd), 0.0);
        let rim_factor = pow(rim, 2.0) * 0.6;
        color = mix(color, vec3f(1.0, 0.8, 0.2), rim_factor);
    }

    return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Pick compute shader postlude
// ---------------------------------------------------------------------------

const PICK_COMPUTE_POSTLUDE: &str = r#"
fn pick_ray_march(ro: vec3f, rd: vec3f) -> vec2f {
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
        if t > 50.0 {
            mat_id = -1.0;
            break;
        }
    }
    return vec2f(t, mat_id);
}

struct PickInput {
    mouse_pos: vec2f,
    _pad: vec2f,
}

@group(2) @binding(0) var<uniform> pick_in: PickInput;
@group(2) @binding(1) var<storage, read_write> pick_out: array<f32, 8>;

@compute @workgroup_size(1)
fn cs_pick() {
    // Generate ray from mouse position (same logic as fs_main)
    let uv = pick_in.mouse_pos / camera.viewport.zw * 2.0 - 1.0;
    let ndc = vec4f(uv.x, -uv.y, 1.0, 1.0);
    let world = camera.inv_view_proj * ndc;
    let rd = normalize(world.xyz / world.w - camera.eye.xyz);
    let ro = camera.eye.xyz;

    let hit = pick_ray_march(ro, rd);
    let t = hit.x;
    let mat_id = hit.y;

    let p = ro + rd * t;

    pick_out[0] = mat_id;
    pick_out[1] = t;
    pick_out[2] = p.x;
    pick_out[3] = p.y;
    pick_out[4] = p.z;
    pick_out[5] = 0.0;
    pick_out[6] = 0.0;
    pick_out[7] = 0.0;
}
"#;

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

pub fn generate_shader(scene: &Scene) -> String {
    let scene_sdf = generate_scene_sdf(scene);
    format!("{}\n{}\n{}", SHADER_PRELUDE, scene_sdf, SHADER_POSTLUDE)
}

pub fn generate_pick_shader(scene: &Scene) -> String {
    let scene_sdf = generate_scene_sdf(scene);
    format!("{}\n{}\n{}", SHADER_PRELUDE, scene_sdf, PICK_COMPUTE_POSTLUDE)
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
                lines.push(format!(
                    "    let lp{i} = rotate_euler(p - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
                ));
                let sdf_fn = kind.sdf_function_name();
                lines.push(format!(
                    "    let n{i} = vec2f({sdf_fn}(lp{i}, nodes[{i}].scale.xyz), f32({i}));"
                ));
            }
            NodeData::Operation { op, left, right, .. } => {
                let li = idx_map.get(left).copied().unwrap_or(0);
                let ri = idx_map.get(right).copied().unwrap_or(0);
                let op_fn = op.wgsl_function_name();
                lines.push(format!(
                    "    let n{i} = {op_fn}(n{li}, n{ri}, nodes[{i}].type_op.y);"
                ));
            }
            NodeData::Sculpt { .. } => {
                lines.push(format!(
                    "    let lp{i} = rotate_euler(p - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
                ));
                lines.push(format!(
                    "    let n{i} = vec2f(sdf_voxel_grid(lp{i}, {i}u), f32({i}));"
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

/// Build the concatenated voxel data buffer.
/// Returns (flat_data, offset_map) where offset_map maps NodeId → offset in flat_data (f32 elements).
pub fn build_voxel_buffer(scene: &Scene) -> (Vec<f32>, HashMap<NodeId, u32>) {
    let order = scene.topo_order();
    let mut flat_data = Vec::new();
    let mut offsets = HashMap::new();

    for &node_id in &order {
        if let Some(node) = scene.nodes.get(&node_id) {
            if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                let offset = flat_data.len() as u32;
                offsets.insert(node_id, offset);
                flat_data.extend_from_slice(&voxel_grid.data);
            }
        }
    }

    (flat_data, offsets)
}

pub fn build_node_buffer(
    scene: &Scene,
    selected: Option<NodeId>,
    voxel_offsets: &HashMap<NodeId, u32>,
) -> Vec<SdfNodeGpu> {
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
                rotation,
                scale,
                color,
                ..
            } => {
                buffer.push(SdfNodeGpu {
                    type_op: [kind.gpu_type_id(), 0.0, 0.0, 0.0],
                    position: [position.x, position.y, position.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [scale.x, scale.y, scale.z, 0.0],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                });
            }
            NodeData::Operation { op, smooth_k, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [op.gpu_op_id(), *smooth_k, 0.0, 0.0],
                    position: [0.0; 4],
                    rotation: [0.0; 4],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
                });
            }
            NodeData::Sculpt {
                position,
                rotation,
                color,
                voxel_grid,
                ..
            } => {
                let offset = voxel_offsets.get(&node_id).copied().unwrap_or(0);
                buffer.push(SdfNodeGpu {
                    type_op: [20.0, 0.0, 0.0, 0.0],
                    position: [position.x, position.y, position.z, 0.0],
                    rotation: [rotation.x, rotation.y, rotation.z, 0.0],
                    scale: [1.0, 1.0, 1.0, 0.0],
                    color: [color.x, color.y, color.z, is_sel],
                    extra0: [offset as f32, voxel_grid.resolution as f32, 0.0, 0.0],
                    extra1: [voxel_grid.bounds_min.x, voxel_grid.bounds_min.y, voxel_grid.bounds_min.z, 0.0],
                    extra2: [voxel_grid.bounds_max.x, voxel_grid.bounds_max.y, voxel_grid.bounds_max.z, 0.0],
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
