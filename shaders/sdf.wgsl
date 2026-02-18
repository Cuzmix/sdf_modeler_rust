// Phase 2: SDF Raymarching with camera uniforms

// ── Camera Uniform ──────────────────────────────────────────────
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

// ── Vertex ──────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle covering [-1,1] clip space
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    // UV: [0,1] range, y=0 at bottom
    out.uv = pos[vi] * 0.5 + 0.5;
    return out;
}

// ── SDF Primitives ──────────────────────────────────────────────
fn sdf_sphere(p: vec3f, center: vec3f, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn sdf_plane(p: vec3f, normal: vec3f, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

// ── Scene ───────────────────────────────────────────────────────
// Returns (distance, material_id): 0 = sphere, 1 = floor
fn scene_sdf(p: vec3f) -> vec2f {
    let sphere = sdf_sphere(p, vec3f(0.0, 0.5, 0.0), 0.5);
    let floor  = sdf_plane(p, vec3f(0.0, 1.0, 0.0), 0.0);

    if sphere < floor {
        return vec2f(sphere, 0.0);
    }
    return vec2f(floor, 1.0);
}

fn scene_dist(p: vec3f) -> f32 {
    return scene_sdf(p).x;
}

// ── Raymarching ─────────────────────────────────────────────────
const MAX_STEPS: i32    = 64;
const MAX_DIST: f32     = 50.0;
const SURFACE_DIST: f32 = 0.001;

struct HitInfo {
    t:           f32,   // distance along ray (-1 = miss)
    material_id: f32,
};

fn raymarch(ro: vec3f, rd: vec3f) -> HitInfo {
    var t = 0.0;
    var mat_id = 0.0;
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
        t += d;
    }
    return HitInfo(-1.0, 0.0);
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
    // Material colors
    var albedo: vec3f;
    if mat_id < 0.5 {
        // Sphere: warm clay
        albedo = vec3f(0.8, 0.55, 0.4);
    } else {
        // Floor: checkerboard
        let checker = step(0.0, sin(p.x * 3.14159 * 2.0) * sin(p.z * 3.14159 * 2.0));
        albedo = mix(vec3f(0.15, 0.15, 0.18), vec3f(0.25, 0.25, 0.3), checker);
    }

    // Directional light
    let light_dir = normalize(vec3f(0.6, 0.8, 0.4));
    let light_col = vec3f(1.0, 0.95, 0.9);

    // Diffuse
    let ndl = max(dot(n, light_dir), 0.0);

    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir - rd);
    let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
    let spec_strength = select(0.5, 0.05, mat_id > 0.5);

    // Shadow
    let shadow = soft_shadow(p + n * 0.01, light_dir, 0.01, 10.0, 16.0);

    // Ambient occlusion
    let ao = calc_ao(p, n);

    // Ambient
    let ambient = vec3f(0.12, 0.14, 0.18);

    // Combine
    var col = albedo * (ambient * ao + light_col * ndl * shadow);
    col += light_col * spec * spec_strength * shadow;

    return col;
}

// ── Fragment ────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Pixel coordinate to NDC [-1, 1]
    let ndc = in.uv * 2.0 - 1.0;

    // Generate ray via inverse view-projection
    let clip_near = vec4f(ndc.x, ndc.y, 0.0, 1.0);
    let clip_far  = vec4f(ndc.x, ndc.y, 1.0, 1.0);

    let world_near = camera.inv_view_proj * clip_near;
    let world_far  = camera.inv_view_proj * clip_far;

    let ro = world_near.xyz / world_near.w;
    let rd = normalize(world_far.xyz / world_far.w - ro);

    // Raymarch
    let hit = raymarch(ro, rd);

    if hit.t < 0.0 {
        // Background gradient: dark blue to dark gray
        let sky_t = rd.y * 0.5 + 0.5;
        let bg = mix(vec3f(0.08, 0.08, 0.12), vec3f(0.15, 0.18, 0.25), sky_t);
        return vec4f(bg, 1.0);
    }

    let p = ro + rd * hit.t;
    let n = calc_normal(p);

    var col = shade(p, n, rd, hit.material_id);

    // Gamma correction
    col = pow(col, vec3f(1.0 / 2.2));

    return vec4f(col, 1.0);
}
