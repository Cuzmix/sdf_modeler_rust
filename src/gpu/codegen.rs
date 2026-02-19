use std::collections::HashMap;

use super::scene::{NodeData, NodeId, Scene, SdfOperation, SdfPrimitive, SdfTransform};

/// WGSL prelude: structs, bindings, vertex shader, SDF primitives, boolean operations.
pub const WGSL_PRELUDE: &str = r#"// SDF Raymarching — expression builder codegen

// ── Camera Uniform (group 0) ───────────────────────────────────
struct Camera {
    view:          mat4x4f,
    projection:    mat4x4f,
    inv_view_proj: mat4x4f,
    eye:           vec3f,
    _pad1:         f32,
    resolution:    vec2f,
    time:          f32,
    _pad2:         f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

// ── Scene Data (group 1) ───────────────────────────────────────
struct SdfNodeGpu {
    type_op:   vec4f,
    position:  vec4f,
    scale:     vec4f,
    color:     vec4f,
    _reserved: vec4f,
};

struct SceneInfo {
    node_count:   u32,
    selected_idx: i32,
    _pad0:        u32,
    _pad1:        u32,
};

@group(1) @binding(0) var<storage, read> nodes: array<SdfNodeGpu>;
@group(1) @binding(1) var<uniform> scene_info: SceneInfo;

// ── Pick Pass (group 2) ──────────────────────────────────────────
struct PickInfo {
    click_ndc: vec2f,
    _pad:      vec2f,
};
@group(2) @binding(0) var<uniform> pick_info: PickInfo;

// ── Vertex ──────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    out.uv = pos[vi] * 0.5 + 0.5;
    return out;
}

// ── SDF Primitives ──────────────────────────────────────────────

fn sdf_sphere(p: vec3f, s: vec3f) -> f32 {
    return length(p) - s.x;
}

fn sdf_box(p: vec3f, s: vec3f) -> f32 {
    let q = abs(p) - s;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_cylinder(p: vec3f, s: vec3f) -> f32 {
    let d = vec2f(length(p.xz) - s.x, abs(p.y) - s.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdf_torus(p: vec3f, s: vec3f) -> f32 {
    let q = vec2f(length(p.xz) - s.x, p.y);
    return length(q) - s.y;
}

fn sdf_plane_y(p: vec3f) -> f32 {
    return p.y;
}

// ── Boolean Operations ──────────────────────────────────────────

fn op_union(a: f32, b: f32) -> f32 {
    return min(a, b);
}

fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / max(k, 0.0001), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn op_subtract(a: f32, b: f32) -> f32 {
    return max(a, -b);
}

fn op_intersect(a: f32, b: f32) -> f32 {
    return max(a, b);
}

// ── Transform Helpers ──────────────────────────────────────────

fn rotate_xyz(p: vec3f, angles: vec3f) -> vec3f {
    let cx = cos(angles.x); let sx = sin(angles.x);
    let cy = cos(angles.y); let sy = sin(angles.y);
    let cz = cos(angles.z); let sz = sin(angles.z);
    var q = p;
    q = vec3f(q.x, cx*q.y - sx*q.z, sx*q.y + cx*q.z); // X
    q = vec3f(cy*q.x + sy*q.z, q.y, -sy*q.x + cy*q.z); // Y
    q = vec3f(cz*q.x - sz*q.y, sz*q.x + cz*q.y, q.z); // Z
    return q;
}

"#;

/// WGSL postlude: raymarching, lighting, gizmo, fragment shaders.
pub const WGSL_POSTLUDE: &str = r#"
// ── Raymarching ─────────────────────────────────────────────────
const MAX_STEPS: i32    = 96;
const MAX_DIST: f32     = 50.0;
const SURFACE_DIST: f32 = 0.0005;

struct HitInfo {
    t:        f32,
    mat_id:   f32,
};

fn raymarch(ro: vec3f, rd: vec3f) -> HitInfo {
    var t = 0.0;
    var mat_id = -1.0;
    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        let d = hit.x;
        mat_id = hit.y;
        if d < SURFACE_DIST {
            return HitInfo(t, mat_id);
        }
        if t > MAX_DIST {
            break;
        }
        // Conservative step (0.9x) to avoid overshooting at surface creases
        t += d * 0.9;
    }
    return HitInfo(-1.0, -1.0);
}

// ── Normal via gradient ─────────────────────────────────────────
fn calc_normal(p: vec3f) -> vec3f {
    let e = 0.001;
    let n = vec3f(
        scene_dist(p + vec3f(e, 0.0, 0.0)) - scene_dist(p - vec3f(e, 0.0, 0.0)),
        scene_dist(p + vec3f(0.0, e, 0.0)) - scene_dist(p - vec3f(0.0, e, 0.0)),
        scene_dist(p + vec3f(0.0, 0.0, e)) - scene_dist(p - vec3f(0.0, 0.0, e)),
    );
    return normalize(n);
}

// ── Soft Shadow ─────────────────────────────────────────────────
fn soft_shadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    for (var i = 0; i < 16; i++) {
        let d = scene_dist(ro + rd * t);
        if d < SURFACE_DIST {
            return 0.0;
        }
        res = min(res, k * d / t);
        t += clamp(d, 0.02, 0.2);
        if t > maxt {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// ── Ambient Occlusion ───────────────────────────────────────────
fn calc_ao(p: vec3f, n: vec3f) -> f32 {
    var occ = 0.0;
    var scale = 1.0;
    for (var i = 1; i <= 5; i++) {
        let step = f32(i) * 0.05;
        let d = scene_dist(p + n * step);
        occ += (step - d) * scale;
        scale *= 0.6;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

// ── Gizmo (screen-space lines) ─────────────────────────────────

// Project world point to screen pixel coordinates
fn world_to_screen(p: vec3f) -> vec3f {
    let clip = camera.projection * camera.view * vec4f(p, 1.0);
    if clip.w <= 0.0 { return vec3f(-1000.0, -1000.0, -1.0); } // behind camera
    let ndc = clip.xy / clip.w;
    return vec3f((ndc * 0.5 + 0.5) * camera.resolution, clip.w);
}

// Distance from 2D point to 2D line segment
fn dist_to_segment_2d(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(pa - ba * h);
}

// Gizmo axis colors
fn gizmo_axis_color(axis: u32) -> vec3f {
    if axis == 1u { return vec3f(1.0, 0.2, 0.2); } // X = red
    if axis == 2u { return vec3f(0.2, 1.0, 0.2); } // Y = green
    return vec3f(0.4, 0.4, 1.0);                     // Z = blue
}

// Compute gizmo endpoints in screen space.
// Returns axis hit (0=none, 1=X, 2=Y, 3=Z) and pixel distance.
fn gizmo_hit_axis(pixel: vec2f, center: vec3f, threshold: f32) -> vec2f {
    let cam_dist = length(camera.eye - center);
    let gizmo_len = cam_dist * 0.25;

    let cs = world_to_screen(center).xy;
    let xe = world_to_screen(center + vec3f(gizmo_len, 0.0, 0.0)).xy;
    let ye = world_to_screen(center + vec3f(0.0, gizmo_len, 0.0)).xy;
    let ze = world_to_screen(center + vec3f(0.0, 0.0, gizmo_len)).xy;

    let dx = dist_to_segment_2d(pixel, cs, xe);
    let dy = dist_to_segment_2d(pixel, cs, ye);
    let dz = dist_to_segment_2d(pixel, cs, ze);

    // Find closest axis within threshold
    var min_d = threshold;
    var axis = 0.0;
    if dz < min_d { min_d = dz; axis = 3.0; }
    if dy < min_d { min_d = dy; axis = 2.0; }
    if dx < min_d { min_d = dx; axis = 1.0; }
    return vec2f(axis, min_d);
}

// ── Lighting ────────────────────────────────────────────────────
fn shade(p: vec3f, n: vec3f, rd: vec3f, mat_id: f32) -> vec3f {
    // Material color
    var albedo: vec3f;
    var is_selected = false;

    if mat_id < 0.0 {
        // Floor: checkerboard
        let checker = step(0.0, sin(p.x * 3.14159 * 2.0) * sin(p.z * 3.14159 * 2.0));
        albedo = mix(vec3f(0.15, 0.15, 0.18), vec3f(0.25, 0.25, 0.3), checker);
    } else {
        let idx = u32(mat_id);
        albedo = nodes[idx].color.xyz;
        is_selected = nodes[idx].type_op.w > 0.5;
    }

    // Directional light
    let light_dir = normalize(vec3f(0.6, 0.8, 0.4));
    let light_col = vec3f(1.0, 0.95, 0.9);

    // Diffuse
    let ndl = max(dot(n, light_dir), 0.0);

    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir - rd);
    let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
    let spec_strength = select(0.5, 0.05, mat_id < 0.0);

    // Shadow
    let shadow = soft_shadow(p + n * 0.01, light_dir, 0.01, 10.0, 16.0);

    // Ambient occlusion
    let ao = calc_ao(p, n);

    // Ambient
    let ambient = vec3f(0.12, 0.14, 0.18);

    // Combine
    var col = albedo * (ambient * ao + light_col * ndl * shadow);
    col += light_col * spec * spec_strength * shadow;

    // Selection rim highlight
    if is_selected {
        let rim = 1.0 - max(dot(n, -rd), 0.0);
        let rim_intensity = pow(rim, 3.0) * 0.6;
        col += vec3f(0.3, 0.6, 1.0) * rim_intensity;
    }

    return col;
}

// ── Fragment ────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let ndc = in.uv * 2.0 - 1.0;

    let clip_near = vec4f(ndc.x, ndc.y, 0.0, 1.0);
    let clip_far  = vec4f(ndc.x, ndc.y, 1.0, 1.0);

    let world_near = camera.inv_view_proj * clip_near;
    let world_far  = camera.inv_view_proj * clip_far;

    let ro = world_near.xyz / world_near.w;
    let rd = normalize(world_far.xyz / world_far.w - ro);

    let hit = raymarch(ro, rd);

    // Sky background
    var col: vec3f;
    var scene_t = hit.t;

    if hit.t < 0.0 {
        let sky_t = rd.y * 0.5 + 0.5;
        col = mix(vec3f(0.08, 0.08, 0.12), vec3f(0.15, 0.18, 0.25), sky_t);
        scene_t = 1e10;
    } else {
        let p = ro + rd * hit.t;
        let n = calc_normal(p);
        col = shade(p, n, rd, hit.mat_id);
    }

    // Gizmo overlay (screen-space lines, always on top)
    if scene_info.selected_idx >= 0 {
        let sel_idx = u32(scene_info.selected_idx);
        let gizmo_center = nodes[sel_idx].position.xyz;
        let pixel = in.uv * camera.resolution;
        let line_width = 2.0;  // visual line width in pixels
        let aa_width = 1.5;    // anti-alias feather
        let g = gizmo_hit_axis(pixel, gizmo_center, line_width + aa_width);
        if g.x > 0.0 {
            let axis_col = gizmo_axis_color(u32(g.x));
            let alpha = 1.0 - smoothstep(line_width - 0.5, line_width + aa_width, g.y);
            col = mix(col, axis_col, alpha);
        }
    }

    // Gamma correction
    col = pow(col, vec3f(1.0 / 2.2));

    return vec4f(col, 1.0);
}

// ── Pick Fragment (outputs encoded node ID) ─────────────────────
@fragment
fn pick_fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Use click NDC instead of fragment UV for ray generation
    let ndc = pick_info.click_ndc;

    let clip_near = vec4f(ndc.x, ndc.y, 0.0, 1.0);
    let clip_far  = vec4f(ndc.x, ndc.y, 1.0, 1.0);

    let world_near = camera.inv_view_proj * clip_near;
    let world_far  = camera.inv_view_proj * clip_far;

    let ro = world_near.xyz / world_near.w;
    let rd = normalize(world_far.xyz / world_far.w - ro);

    let hit = raymarch(ro, rd);

    // Encode: 0=background, 1=floor, 2+=node(index+2), 253=X, 254=Y, 255=Z
    var id = 0u;
    if hit.t >= 0.0 {
        if hit.mat_id < 0.0 {
            id = 1u;  // floor
        } else {
            id = u32(hit.mat_id) + 2u;  // node
        }
    }

    // Check gizmo lines (takes priority over scene)
    if scene_info.selected_idx >= 0 {
        let sel_idx = u32(scene_info.selected_idx);
        let gizmo_center = nodes[sel_idx].position.xyz;
        // Convert click NDC to pixel coordinates
        let click_pixel = (pick_info.click_ndc * 0.5 + 0.5) * camera.resolution;
        let pick_threshold = 8.0; // wider hitbox for picking
        let g = gizmo_hit_axis(click_pixel, gizmo_center, pick_threshold);
        if g.x > 0.0 {
            // 253=X, 254=Y, 255=Z
            id = 252u + u32(g.x);
        }
    }

    return vec4f(f32(id) / 255.0, 0.0, 0.0, 1.0);
}
"#;

/// Compose a complete WGSL shader from prelude + generated scene_sdf + postlude.
pub fn compose_shader(scene: &Scene) -> String {
    let mut wgsl = String::with_capacity(8192);
    wgsl.push_str(WGSL_PRELUDE);
    wgsl.push_str("// ── Scene Evaluation (codegen) ──────────────────────────────\n\n");
    generate_scene_sdf(scene, &mut wgsl);
    wgsl.push_str(WGSL_POSTLUDE);
    wgsl
}

/// Compute a structure key that changes only when graph topology changes.
/// Excludes parameter values (position, scale, color, smooth_k).
pub fn structure_key(scene: &Scene) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    if let Some(root_id) = scene.root_id() {
        hash_topology(scene, root_id, &mut hasher);
    }
    hasher.finish()
}

fn hash_topology(scene: &Scene, node_id: NodeId, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    node_id.hash(hasher);
    if let Some(node) = scene.get_node(node_id) {
        match &node.data {
            NodeData::Primitive(prim) => {
                0u8.hash(hasher);
                (prim.primitive as u32).hash(hasher);
            }
            NodeData::Operation(op) => {
                1u8.hash(hasher);
                (op.operation as u32).hash(hasher);
                hash_topology(scene, op.left, hasher);
                hash_topology(scene, op.right, hasher);
            }
            NodeData::Transform(tr) => {
                2u8.hash(hasher);
                (tr.transform as u32).hash(hasher);
                hash_topology(scene, tr.input, hasher);
            }
        }
    }
}

/// Generate the scene_sdf() and scene_dist() WGSL functions.
fn generate_scene_sdf(scene: &Scene, out: &mut String) {
    out.push_str("fn scene_sdf(p: vec3f) -> vec2f {\n");

    if let Some(root_id) = scene.root_id() {
        let mut ctx = CodegenCtx {
            scene,
            counter: 0,
            next_gpu_idx: 0,
            node_var: HashMap::new(),
        };
        let root_var = ctx.emit_node(root_id, "p", out);

        // Hard union with floor plane
        out.push_str("    let floor_d = sdf_plane_y(p);\n");
        out.push_str(&format!(
            "    if (floor_d < d{root_var}) {{ return vec2f(floor_d, -1.0); }}\n"
        ));
        out.push_str(&format!(
            "    return vec2f(d{root_var}, m{root_var});\n"
        ));
    } else {
        // Empty scene: just floor
        out.push_str("    let floor_d = sdf_plane_y(p);\n");
        out.push_str("    return vec2f(floor_d, -1.0);\n");
    }

    out.push_str("}\n\n");
    out.push_str("fn scene_dist(p: vec3f) -> f32 {\n");
    out.push_str("    return scene_sdf(p).x;\n");
    out.push_str("}\n");
}

struct CodegenCtx<'a> {
    scene: &'a Scene,
    counter: usize,
    next_gpu_idx: usize,
    node_var: HashMap<NodeId, usize>,
}

impl<'a> CodegenCtx<'a> {
    /// Emit WGSL for a node. Returns the variable index (N in dN/mN).
    /// `p_expr` is the current query point expression (e.g., "p").
    fn emit_node(&mut self, node_id: NodeId, p_expr: &str, out: &mut String) -> usize {
        // DAG sharing: if already emitted, reuse variable
        if let Some(&var_idx) = self.node_var.get(&node_id) {
            return var_idx;
        }

        let node = self.scene.get_node(node_id).expect("node must exist");
        let var_idx = self.counter;
        self.counter += 1;
        self.node_var.insert(node_id, var_idx);

        match &node.data {
            NodeData::Primitive(prim) => {
                // Primitives push gpu_idx when visited (matches flatten_for_gpu)
                let gpu_idx = self.next_gpu_idx;
                self.next_gpu_idx += 1;

                let sdf_fn = match prim.primitive {
                    SdfPrimitive::Sphere => "sdf_sphere",
                    SdfPrimitive::Box => "sdf_box",
                    SdfPrimitive::Cylinder => "sdf_cylinder",
                    SdfPrimitive::Torus => "sdf_torus",
                    SdfPrimitive::Plane => "sdf_plane_y",
                };

                if matches!(prim.primitive, SdfPrimitive::Plane) {
                    out.push_str(&format!(
                        "    let d{var_idx} = {sdf_fn}({p_expr} - nodes[{gpu_idx}].position.xyz);\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "    let d{var_idx} = {sdf_fn}({p_expr} - nodes[{gpu_idx}].position.xyz, nodes[{gpu_idx}].scale.xyz);\n"
                    ));
                }
                out.push_str(&format!(
                    "    let m{var_idx}: f32 = f32({gpu_idx});\n"
                ));
            }
            NodeData::Operation(op) => {
                let left = op.left;
                let right = op.right;
                let operation = op.operation;

                // Post-order: emit children first
                let left_var = self.emit_node(left, p_expr, out);
                let right_var = self.emit_node(right, p_expr, out);

                // Operation pushed AFTER both children (matches flatten_for_gpu)
                let gpu_idx = self.next_gpu_idx;
                self.next_gpu_idx += 1;

                let op_call = match operation {
                    SdfOperation::Union => {
                        format!("op_union(d{left_var}, d{right_var})")
                    }
                    SdfOperation::SmoothUnion => {
                        format!(
                            "op_smooth_union(d{left_var}, d{right_var}, nodes[{gpu_idx}].type_op.z)"
                        )
                    }
                    SdfOperation::Subtract => {
                        format!("op_subtract(d{left_var}, d{right_var})")
                    }
                    SdfOperation::Intersect => {
                        format!("op_intersect(d{left_var}, d{right_var})")
                    }
                };

                out.push_str(&format!("    let d{var_idx} = {op_call};\n"));

                // Material: subtract keeps A's material, others pick closer
                let mat_expr = match operation {
                    SdfOperation::Subtract => format!("m{left_var}"),
                    _ => format!(
                        "select(m{left_var}, m{right_var}, d{right_var} < d{left_var})"
                    ),
                };
                out.push_str(&format!("    let m{var_idx} = {mat_expr};\n"));
            }
            NodeData::Transform(tr) => {
                let input = tr.input;
                let transform = tr.transform;

                // Pre-order: assign gpu_idx BEFORE child (matches flatten_for_gpu)
                let gpu_idx = self.next_gpu_idx;
                self.next_gpu_idx += 1;

                // Transform modifies the query point
                let new_p = format!("p{var_idx}");
                match transform {
                    SdfTransform::Translate => {
                        out.push_str(&format!(
                            "    let {new_p} = {p_expr} - nodes[{gpu_idx}].position.xyz;\n"
                        ));
                    }
                    SdfTransform::Rotate => {
                        out.push_str(&format!(
                            "    let {new_p} = rotate_xyz({p_expr}, nodes[{gpu_idx}].position.xyz);\n"
                        ));
                    }
                    SdfTransform::Scale => {
                        out.push_str(&format!(
                            "    let {new_p} = {p_expr} / nodes[{gpu_idx}].position.xyz;\n"
                        ));
                    }
                }

                // Emit child with the transformed query point
                let child_var = self.emit_node(input, &new_p, out);

                // Alias distance and material from child
                match transform {
                    SdfTransform::Scale => {
                        // Scale requires distance correction: d * min(sx, sy, sz)
                        out.push_str(&format!(
                            "    let d{var_idx} = d{child_var} * min(nodes[{gpu_idx}].position.x, min(nodes[{gpu_idx}].position.y, nodes[{gpu_idx}].position.z));\n"
                        ));
                    }
                    _ => {
                        out.push_str(&format!("    let d{var_idx} = d{child_var};\n"));
                    }
                }
                out.push_str(&format!("    let m{var_idx} = m{child_var};\n"));
            }
        }

        var_idx
    }
}
