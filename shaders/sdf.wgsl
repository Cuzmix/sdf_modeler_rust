// Phase 3: Dynamic SDF Raymarching — scene driven by storage buffer

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
    type_op:   vec4f,   // x=primitive_type, y=operation, z=smooth_k, w=selected
    position:  vec4f,   // xyz=position
    scale:     vec4f,   // xyz=scale
    color:     vec4f,   // xyz=rgb, w=alpha
    _reserved: vec4f,
};

struct SceneInfo {
    node_count:   u32,
    selected_idx: i32,
    _pad0:        u32,
    _pad1:        u32,
};

const MAX_NODES: u32 = 64u;
@group(1) @binding(0) var<uniform> nodes: array<SdfNodeGpu, 64>;
@group(1) @binding(1) var<uniform> scene_info: SceneInfo;

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

// ── Scene Evaluation ────────────────────────────────────────────

// Returns (distance, node_index_as_float). node_index = -1 for floor.
fn scene_sdf(p: vec3f) -> vec2f {
    // Evaluate scene nodes separately (operations apply between nodes only)
    var scene_dist = 1e10;
    var scene_mat = -1.0;

    let count = scene_info.node_count;
    for (var i = 0u; i < count; i++) {
        let node = nodes[i];
        let prim_type = u32(node.type_op.x);
        let op_type   = u32(node.type_op.y);
        let smooth_k  = node.type_op.z;

        // Transform to local space
        let local = p - node.position.xyz;

        // Evaluate primitive
        var d: f32;
        switch prim_type {
            case 0u: { d = sdf_sphere(local, node.scale.xyz); }
            case 1u: { d = sdf_box(local, node.scale.xyz); }
            case 2u: { d = sdf_cylinder(local, node.scale.xyz); }
            case 3u: { d = sdf_torus(local, node.scale.xyz); }
            default: { d = sdf_plane_y(local); }
        }

        // First node: use directly
        if i == 0u {
            scene_dist = d;
            scene_mat = 0.0;
            continue;
        }

        // Combine with previous scene nodes
        let prev = scene_dist;
        switch op_type {
            case 0u: { scene_dist = op_union(scene_dist, d); }
            case 1u: { scene_dist = op_smooth_union(scene_dist, d, smooth_k); }
            case 2u: { scene_dist = op_subtract(scene_dist, d); }
            case 3u: { scene_dist = op_intersect(scene_dist, d); }
            default: { scene_dist = op_union(scene_dist, d); }
        }

        // Track closest node for material
        if d < prev {
            scene_mat = f32(i);
        }
    }

    // Combine scene with ground plane using hard union
    let floor_dist = sdf_plane_y(p);
    if count == 0u || floor_dist < scene_dist {
        return vec2f(floor_dist, -1.0);
    }
    return vec2f(scene_dist, scene_mat);
}

fn scene_dist(p: vec3f) -> f32 {
    return scene_sdf(p).x;
}

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

    if hit.t < 0.0 {
        let sky_t = rd.y * 0.5 + 0.5;
        let bg = mix(vec3f(0.08, 0.08, 0.12), vec3f(0.15, 0.18, 0.25), sky_t);
        return vec4f(bg, 1.0);
    }

    let p = ro + rd * hit.t;
    let n = calc_normal(p);

    var col = shade(p, n, rd, hit.mat_id);

    // Gamma correction
    col = pow(col, vec3f(1.0 / 2.2));

    return vec4f(col, 1.0);
}
