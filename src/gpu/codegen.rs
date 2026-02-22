use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode, TransformKind};
use crate::settings::RenderConfig;

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
    quality_mode: f32,
    _pad: vec2f,
    scene_min: vec4f,
    scene_max: vec4f,
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

    // Distance to AABB (0 when inside the box)
    let clamped = clamp(local_p, bmin, bmax);
    let box_dist = length(local_p - clamped);

    // Far from box: skip expensive trilinear interpolation entirely
    if box_dist > 0.1 {
        return box_dist;
    }

    // Near or inside: trilinear interp at clamped point + box_dist for continuity.
    // When inside, clamped == local_p and box_dist == 0 (same as before).
    // When outside near boundary, samples boundary voxels and adds box_dist.
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
    return mix(c0, c1, f.z) + box_dist;
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

// --- Ray-AABB intersection (for scene-level bounding) ---

fn ray_aabb(ro: vec3f, inv_rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
    let t1 = (bmin - ro) * inv_rd;
    let t2 = (bmax - ro) * inv_rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(t_near, t_far);
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
    if k < 0.0001 {
        return vec2f(max(a.x, -b.x), a.y);
    }
    let h = clamp(0.5 - 0.5 * (a.x + b.x) / k, 0.0, 1.0);
    let d = mix(a.x, -b.x, h) + k * h * (1.0 - h);
    return vec2f(d, a.y);
}

fn op_intersect(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if k < 0.0001 {
        if a.x > b.x { return a; } else { return b; }
    }
    let h = clamp(0.5 - 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    let d = mix(b.x, a.x, h) + k * h * (1.0 - h);
    return vec2f(d, select(b.y, a.y, a.x > b.x));
}
"#;

// ---------------------------------------------------------------------------
// Shader template: everything after scene_sdf
// ---------------------------------------------------------------------------

const SHADER_POSTLUDE: &str = r#"
// --- Rendering quality constants ---
const SHADOW_STEPS: i32 = /*SHADOW_STEPS*/;
const SHADOW_PENUMBRA_K: f32 = /*SHADOW_PENUMBRA_K*/;
const AO_SAMPLES: i32 = /*AO_SAMPLES*/;
const AO_STEP: f32 = /*AO_STEP*/;
const AO_DECAY: f32 = /*AO_DECAY*/;

fn ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    let fast = camera.quality_mode > 0.5;
    let eff_steps = select(/*MARCH_MAX_STEPS*/, /*MARCH_MAX_STEPS*/ / 2, fast);
    let eps = select(/*MARCH_EPSILON*/, /*MARCH_EPSILON*/ * 4.0, fast);
    let step_mult = select(/*MARCH_STEP_MULT*/, min(/*MARCH_STEP_MULT*/ * 1.5, 1.0), fast);

    // Scene-level ray-AABB: skip empty space before the scene
    let inv_rd = 1.0 / rd;
    let aabb = ray_aabb(ro, inv_rd, camera.scene_min.xyz, camera.scene_max.xyz);
    let max_dist = min(aabb.y + 1.0, /*MARCH_MAX_DIST*/);
    if aabb.x > aabb.y || aabb.y < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    var t = max(aabb.x - 0.01, 0.0);

    // Enhanced sphere tracing (Keinert over-relaxation)
    var omega = select(1.2, 1.0, fast);
    var prev_d = 1e10;
    var prev_step = 0.0;
    var mat_id = -1.0;
    for (var i = 0; i < /*MARCH_MAX_STEPS*/; i++) {
        if i >= eff_steps { break; }
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        let d = hit.x;
        let step = d * omega * step_mult;
        // Overshoot detection: if combined radii can't bridge the step, undo
        if omega > 1.0 && prev_d + d < prev_step {
            t -= prev_step;
            omega = 1.0;
            prev_d = 1e10;
            prev_step = 0.0;
            continue;
        }
        if d < eps {
            mat_id = hit.y;
            break;
        }
        t += step;
        prev_step = step;
        prev_d = d;
        if t > max_dist { break; }
    }
    if mat_id < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    return vec2f(t, mat_id);
}

fn calc_normal(p: vec3f, t: f32) -> vec3f {
    // Tetrahedron technique: 4 SDF evals instead of 6.
    // Distance-adaptive epsilon: larger at distance (reduces aliasing), tighter up close (more detail).
    let e = clamp(0.001 * t, 0.0005, 0.05);
    let k = vec2f(1.0, -1.0);
    return normalize(
        k.xyy * scene_sdf(p + k.xyy * e).x +
        k.yyx * scene_sdf(p + k.yyx * e).x +
        k.yxy * scene_sdf(p + k.yxy * e).x +
        k.xxx * scene_sdf(p + k.xxx * e).x
    );
}

fn soft_shadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
    // iq's improved soft shadows with triangulation (reduced light leaking)
    var res = 1.0;
    var t = mint;
    var ph = 1e20;
    for (var i = 0; i < SHADOW_STEPS; i++) {
        let h = scene_sdf(ro + rd * t).x;
        if h < 0.005 {
            return 0.0;
        }
        let y = h * h / (2.0 * ph);
        let d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0001, t - y));
        ph = h;
        t += h;
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
    return clamp(1.0 - /*AO_INTENSITY*/ * occ, 0.0, 1.0);
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
    if t > /*SKY_CUTOFF*/ {
        let sky_t = uv.y * 0.5 + 0.5;
        let bg = mix(vec3f(/*SKY_HORIZON*/), vec3f(/*SKY_ZENITH*/), sky_t);
        return vec4f(bg, 1.0);
    }

    let p = ro + rd * t;
    let n = calc_normal(p, t);

    // Key light
    let key_dir = normalize(vec3f(/*KEY_LIGHT_DIR*/));
    let key_h = normalize(key_dir - rd);
    let key_diff = max(dot(n, key_dir), 0.0);
    let key_spec = pow(max(dot(n, key_h), 0.0), /*KEY_SPEC_POWER*/);
    /*SHADOW_LINE*/

    // Fill light (opposite side, dimmer, no shadow)
    let fill_dir = normalize(vec3f(/*FILL_LIGHT_DIR*/));
    let fill_diff = max(dot(n, fill_dir), 0.0) * /*FILL_INTENSITY*/;

    // Ambient occlusion
    /*AO_LINE*/

    let albedo = get_node_color(mat_id);
    // Hemispherical sky light (iq outdoor lighting)
    let sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
    // Colored shadow tinting: blue shift in shadows (skylight fill)
    let shadow_col = pow(vec3f(shadow), vec3f(1.0, 1.2, 1.5));
    // Sun uses shadow only; sky/fill use AO only (iq's rule)
    var color = albedo * (sky * /*AMBIENT*/ * ao + key_diff * shadow_col * /*KEY_DIFFUSE*/ + fill_diff * ao)
              + vec3f(1.0) * key_spec * shadow * /*KEY_SPEC_INTENSITY*/;

    if is_selected(mat_id) {
        let rim = 1.0 - max(dot(n, -rd), 0.0);
        let rim_factor = pow(rim, 2.0) * 0.6;
        color = mix(color, vec3f(1.0, 0.8, 0.2), rim_factor);
    }

    /*FOG_LINE*/

    return vec4f(pow(color, vec3f(1.0 / /*GAMMA*/)), 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Pick compute shader postlude
// ---------------------------------------------------------------------------

const PICK_COMPUTE_POSTLUDE: &str = r#"
fn pick_ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    let inv_rd = 1.0 / rd;
    let aabb = ray_aabb(ro, inv_rd, camera.scene_min.xyz, camera.scene_max.xyz);
    let max_dist = min(aabb.y + 1.0, /*MARCH_MAX_DIST*/);
    if aabb.x > aabb.y || aabb.y < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    var t = max(aabb.x - 0.01, 0.0);
    var mat_id = -1.0;
    for (var i = 0; i < /*MARCH_MAX_STEPS*/; i++) {
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        if hit.x < /*MARCH_EPSILON*/ {
            mat_id = hit.y;
            break;
        }
        t += hit.x * /*MARCH_STEP_MULT*/;
        if t > max_dist { break; }
    }
    if mat_id < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
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
// Brush compute shader (standalone — no scene/camera bindings needed)
// ---------------------------------------------------------------------------

pub const BRUSH_COMPUTE_SHADER: &str = r#"
struct BrushParams {
    center_local: vec3f,
    radius: f32,
    strength: f32,
    sign_val: f32,
    grid_offset: u32,
    grid_resolution: u32,
    bounds_min: vec3f,
    _pad0: f32,
    bounds_max: vec3f,
    _pad1: f32,
    min_voxel: vec3<u32>,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> brush: BrushParams;
@group(0) @binding(1) var<storage, read_write> voxel_data: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn cs_brush(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_pos = brush.min_voxel + gid;
    let res = brush.grid_resolution;
    if any(grid_pos >= vec3<u32>(res)) { return; }

    let res_f = f32(res - 1u);
    let norm = vec3f(grid_pos) / res_f;
    let world_pos = brush.bounds_min + norm * (brush.bounds_max - brush.bounds_min);
    let dist = length(world_pos - brush.center_local);

    if dist < brush.radius {
        let falloff = 1.0 - dist / brush.radius;
        let delta = brush.sign_val * brush.strength * falloff * falloff;
        let idx = brush.grid_offset + grid_pos.z * res * res + grid_pos.y * res + grid_pos.x;
        voxel_data[idx] += delta;
    }
}
"#;

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// Format f32 as a WGSL literal (always includes decimal point).
fn format_f32(v: f32) -> String {
    let s = format!("{}", v);
    if s.contains('.') { s } else { format!("{}.0", s) }
}

/// Format [f32; 3] as WGSL vec3 components: "x, y, z".
fn format_vec3(v: [f32; 3]) -> String {
    format!("{}, {}, {}", format_f32(v[0]), format_f32(v[1]), format_f32(v[2]))
}

/// Apply the 4 raymarching placeholders shared by render and pick shaders.
fn apply_march_placeholders(src: &str, config: &RenderConfig) -> String {
    src.replace("/*MARCH_MAX_STEPS*/", &config.march_max_steps.to_string())
        .replace("/*MARCH_EPSILON*/", &format_f32(config.march_epsilon))
        .replace("/*MARCH_STEP_MULT*/", &format_f32(config.march_step_multiplier))
        .replace("/*MARCH_MAX_DIST*/", &format_f32(config.march_max_distance))
}

/// Info about sculpt nodes for texture binding.
pub struct SculptTexInfo {
    pub node_id: NodeId,
    pub tex_idx: usize,
    pub resolution: u32,
}

/// Count sculpt nodes in topo order and return their info for texture creation.
pub fn collect_sculpt_tex_info(scene: &Scene) -> Vec<SculptTexInfo> {
    let order = scene.topo_order();
    let mut infos = Vec::new();
    for &node_id in &order {
        if let Some(node) = scene.nodes.get(&node_id) {
            if let NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                infos.push(SculptTexInfo {
                    node_id,
                    tex_idx: infos.len(),
                    resolution: voxel_grid.resolution,
                });
            }
        }
    }
    infos
}

/// Generate WGSL texture declarations and per-sculpt sampling functions.
/// Returns (wgsl_code, node_id→tex_idx map).
fn generate_voxel_texture_decls(scene: &Scene) -> (String, HashMap<NodeId, usize>) {
    let infos = collect_sculpt_tex_info(scene);
    if infos.is_empty() {
        return (String::new(), HashMap::new());
    }

    let mut lines = Vec::new();
    lines.push("@group(2) @binding(0) var voxel_sampler: sampler;".to_string());

    let mut tex_map = HashMap::new();
    for info in &infos {
        let i = info.tex_idx;
        let binding = i + 1;
        lines.push(format!(
            "@group(2) @binding({binding}) var voxel_tex_{i}: texture_3d<f32>;"
        ));
        tex_map.insert(info.node_id, i);
    }

    lines.push(String::new());

    // Per-sculpt sampling function
    for info in &infos {
        let i = info.tex_idx;
        lines.push(format!("fn sdf_voxel_tex_{i}(local_p: vec3f, node_idx: u32) -> f32 {{"));
        lines.push("    let bmin = nodes[node_idx].extra1.xyz;".to_string());
        lines.push("    let bmax = nodes[node_idx].extra2.xyz;".to_string());
        lines.push("    let clamped = clamp(local_p, bmin, bmax);".to_string());
        lines.push("    let box_dist = length(local_p - clamped);".to_string());
        lines.push("    if box_dist > 0.1 { return box_dist; }".to_string());
        lines.push("    let uv = (local_p - bmin) / (bmax - bmin);".to_string());
        lines.push(format!(
            "    return textureSampleLevel(voxel_tex_{i}, voxel_sampler, uv, 0.0).x + box_dist;"
        ));
        lines.push("}".to_string());
        lines.push(String::new());
    }

    (lines.join("\n"), tex_map)
}

pub fn generate_shader(scene: &Scene, config: &RenderConfig) -> String {
    let (tex_decls, sculpt_tex_map) = generate_voxel_texture_decls(scene);
    let tex_map = if sculpt_tex_map.is_empty() { None } else { Some(&sculpt_tex_map) };
    let scene_sdf = generate_scene_sdf(scene, tex_map);

    let shadow_line = if config.shadows_enabled {
        format!(
            "    var shadow = 1.0;\n    if camera.quality_mode < 0.5 {{ shadow = soft_shadow(p + n * {}, key_dir, {}, {}, SHADOW_PENUMBRA_K); }}",
            format_f32(config.shadow_bias),
            format_f32(config.shadow_mint),
            format_f32(config.shadow_maxt),
        )
    } else {
        "    let shadow = 1.0;".to_string()
    };

    let ao_line = if config.ao_enabled {
        "    var ao = 1.0;\n    if camera.quality_mode < 0.5 { ao = calc_ao(p, n); }".to_string()
    } else {
        "    let ao = 1.0;".to_string()
    };

    let fog_line = if config.fog_enabled {
        format!(
            "    {{ let fog_amt = 1.0 - exp(-t * {}); let sun_s = pow(max(dot(rd, key_dir), 0.0), 8.0); let fog_c = mix(vec3f({}), vec3f(1.0, 0.9, 0.7), sun_s); color = mix(color, fog_c, fog_amt); }}",
            format_f32(config.fog_density),
            format_vec3(config.fog_color),
        )
    } else {
        "".to_string()
    };

    let sky_cutoff = config.march_max_distance - 1.0;

    let postlude = apply_march_placeholders(SHADER_POSTLUDE, config)
        .replace("/*SHADOW_STEPS*/", &config.shadow_steps.to_string())
        .replace("/*SHADOW_PENUMBRA_K*/", &format_f32(config.shadow_penumbra_k))
        .replace("/*AO_SAMPLES*/", &config.ao_samples.to_string())
        .replace("/*AO_STEP*/", &format_f32(config.ao_step))
        .replace("/*AO_DECAY*/", &format_f32(config.ao_decay))
        .replace("/*AO_INTENSITY*/", &format_f32(config.ao_intensity))
        .replace("/*KEY_LIGHT_DIR*/", &format_vec3(config.key_light_dir))
        .replace("/*KEY_DIFFUSE*/", &format_f32(config.key_diffuse))
        .replace("/*KEY_SPEC_POWER*/", &format_f32(config.key_spec_power))
        .replace("/*KEY_SPEC_INTENSITY*/", &format_f32(config.key_spec_intensity))
        .replace("/*FILL_LIGHT_DIR*/", &format_vec3(config.fill_light_dir))
        .replace("/*FILL_INTENSITY*/", &format_f32(config.fill_intensity))
        .replace("/*AMBIENT*/", &format_f32(config.ambient))
        .replace("/*SKY_HORIZON*/", &format_vec3(config.sky_horizon))
        .replace("/*SKY_ZENITH*/", &format_vec3(config.sky_zenith))
        .replace("/*SKY_CUTOFF*/", &format_f32(sky_cutoff))
        .replace("/*GAMMA*/", &format_f32(config.gamma))
        .replace("/*SHADOW_LINE*/", &shadow_line)
        .replace("/*AO_LINE*/", &ao_line)
        .replace("/*FOG_LINE*/", &fog_line);

    format!("{}\n{}\n{}\n{}", SHADER_PRELUDE, tex_decls, scene_sdf, postlude)
}

pub fn generate_pick_shader(scene: &Scene, config: &RenderConfig) -> String {
    let scene_sdf = generate_scene_sdf(scene, None);
    let pick_postlude = apply_march_placeholders(PICK_COMPUTE_POSTLUDE, config);
    format!("{}\n{}\n{}", SHADER_PRELUDE, scene_sdf, pick_postlude)
}

/// Walk up from `node_id` through ancestors, collecting all Transform ancestors.
/// Returns chain from innermost (closest to leaf) to outermost.
fn get_transform_chain(
    node_id: NodeId,
    parent_map: &HashMap<NodeId, NodeId>,
    scene: &Scene,
    idx_map: &HashMap<NodeId, usize>,
) -> Vec<(usize, TransformKind)> {
    let mut chain = Vec::new();
    let mut current = node_id;
    while let Some(&parent_id) = parent_map.get(&current) {
        if let Some(node) = scene.nodes.get(&parent_id) {
            if let NodeData::Transform { kind, .. } = &node.data {
                if let Some(&idx) = idx_map.get(&parent_id) {
                    chain.push((idx, kind.clone()));
                }
            }
        }
        current = parent_id;
    }
    chain
}

/// Emit WGSL code for a transform chain. Returns the final point variable name.
/// Chain is innermost-first; we process outermost-first (reversed).
fn emit_transform_chain(
    lines: &mut Vec<String>,
    node_idx: usize,
    chain: &[(usize, TransformKind)],
) -> String {
    if chain.is_empty() {
        return "p".to_string();
    }
    let mut current_var = "p".to_string();
    for (step, (transform_idx, kind)) in chain.iter().rev().enumerate() {
        let new_var = format!("tp{}_{}", node_idx, step);
        match kind {
            TransformKind::Translate => {
                lines.push(format!(
                    "    let {new_var} = {current_var} - nodes[{transform_idx}].position.xyz;"
                ));
            }
            TransformKind::Rotate => {
                lines.push(format!(
                    "    let {new_var} = rotate_euler({current_var}, nodes[{transform_idx}].rotation.xyz);"
                ));
            }
            TransformKind::Scale => {
                lines.push(format!(
                    "    let {new_var} = {current_var} / nodes[{transform_idx}].scale.xyz;"
                ));
            }
        }
        current_var = new_var;
    }
    current_var
}

/// Emit WGSL for a single node. Extracted so both cheap and expensive phases can use it.
/// `sculpt_tex_map`: if Some, sculpt nodes use `sdf_voxel_tex_N` (texture path);
///                    if None, they use `sdf_voxel_grid` (storage buffer path).
fn emit_node_wgsl(
    lines: &mut Vec<String>,
    i: usize,
    node_id: NodeId,
    node: &SceneNode,
    parent_map: &HashMap<NodeId, NodeId>,
    scene: &Scene,
    idx_map: &HashMap<NodeId, usize>,
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) {
    match &node.data {
        NodeData::Primitive { kind, .. } => {
            let chain = get_transform_chain(node_id, parent_map, scene, idx_map);
            let point_var = emit_transform_chain(lines, i, &chain);
            lines.push(format!(
                "    let lp{i} = rotate_euler({point_var} - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
            ));
            let sdf_fn = kind.sdf_function_name();
            lines.push(format!(
                "    let n{i} = vec2f({sdf_fn}(lp{i}, nodes[{i}].scale.xyz), f32({i}));"
            ));
        }
        NodeData::Operation { op, left, right, .. } => {
            let li = left.and_then(|id| idx_map.get(&id).copied());
            let ri = right.and_then(|id| idx_map.get(&id).copied());
            match (li, ri) {
                (Some(li), Some(ri)) => {
                    let op_fn = op.wgsl_function_name();
                    lines.push(format!(
                        "    let n{i} = {op_fn}(n{li}, n{ri}, nodes[{i}].type_op.y);"
                    ));
                }
                (Some(ci), None) | (None, Some(ci)) => {
                    lines.push(format!("    let n{i} = n{ci};"));
                }
                (None, None) => {
                    lines.push(format!("    let n{i} = vec2f(1e10, -1.0);"));
                }
            }
        }
        NodeData::Sculpt { .. } => {
            let chain = get_transform_chain(node_id, parent_map, scene, idx_map);
            let point_var = emit_transform_chain(lines, i, &chain);
            lines.push(format!(
                "    let lp{i} = rotate_euler({point_var} - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
            ));
            // Use texture path if available, otherwise storage buffer
            let sdf_call = if let Some(tex_map) = sculpt_tex_map {
                if let Some(&tex_idx) = tex_map.get(&node_id) {
                    format!("sdf_voxel_tex_{tex_idx}(lp{i}, {i}u)")
                } else {
                    format!("sdf_voxel_grid(lp{i}, {i}u)")
                }
            } else {
                format!("sdf_voxel_grid(lp{i}, {i}u)")
            };
            lines.push(format!(
                "    let n{i} = vec2f({sdf_call}, f32({i}));"
            ));
        }
        NodeData::Transform { kind, input, .. } => {
            let child_idx = input.and_then(|id| idx_map.get(&id).copied());
            if let Some(ci) = child_idx {
                match kind {
                    TransformKind::Scale => {
                        lines.push(format!(
                            "    let n{i} = vec2f(n{ci}.x * min(nodes[{i}].scale.x, min(nodes[{i}].scale.y, nodes[{i}].scale.z)), n{ci}.y);"
                        ));
                    }
                    _ => {
                        lines.push(format!("    let n{i} = n{ci};"));
                    }
                }
            } else {
                lines.push(format!("    let n{i} = vec2f(1e10, -1.0);"));
            }
        }
    }
}

fn generate_scene_sdf(
    scene: &Scene,
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) -> String {
    let order = scene.topo_order();
    if order.is_empty() {
        return "fn scene_sdf(p: vec3f) -> vec2f {\n    return vec2f(1e10, -1.0);\n}"
            .to_string();
    }

    let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let parent_map = scene.build_parent_map();

    let tops = scene.top_level_nodes();

    generate_scene_sdf_flat(scene, &order, &idx_map, &parent_map, &tops, sculpt_tex_map)
}

/// Original flat codegen (no expensive subtrees).
fn generate_scene_sdf_flat(
    scene: &Scene,
    order: &[NodeId],
    idx_map: &HashMap<NodeId, usize>,
    parent_map: &HashMap<NodeId, NodeId>,
    tops: &[NodeId],
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) -> String {
    let mut lines = Vec::new();
    lines.push("fn scene_sdf(p: vec3f) -> vec2f {".to_string());

    for (i, &node_id) in order.iter().enumerate() {
        let Some(node) = scene.nodes.get(&node_id) else { continue; };
        emit_node_wgsl(&mut lines, i, node_id, node, parent_map, scene, idx_map, sculpt_tex_map);
    }

    let top_indices: Vec<usize> = tops
        .iter()
        .filter_map(|id| idx_map.get(id).copied())
        .collect();
    match top_indices.len() {
        0 => lines.push("    return vec2f(1e10, -1.0);".to_string()),
        1 => lines.push(format!("    return n{};", top_indices[0])),
        _ => {
            lines.push(format!("    var result = n{};", top_indices[0]));
            for &idx in &top_indices[1..] {
                lines.push(format!("    result = op_union(result, n{idx}, 0.0);"));
            }
            lines.push("    return result;".to_string());
        }
    }
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
            NodeData::Transform { kind, value, .. } => {
                buffer.push(SdfNodeGpu {
                    type_op: [kind.gpu_type_id(), 0.0, 0.0, 0.0],
                    position: if matches!(kind, TransformKind::Translate) {
                        [value.x, value.y, value.z, 0.0]
                    } else {
                        [0.0; 4]
                    },
                    rotation: if matches!(kind, TransformKind::Rotate) {
                        [value.x, value.y, value.z, 0.0]
                    } else {
                        [0.0; 4]
                    },
                    scale: if matches!(kind, TransformKind::Scale) {
                        [value.x, value.y, value.z, 0.0]
                    } else {
                        [1.0, 1.0, 1.0, 0.0]
                    },
                    color: [0.0, 0.0, 0.0, is_sel],
                    extra0: [0.0; 4],
                    extra1: [0.0; 4],
                    extra2: [0.0; 4],
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
