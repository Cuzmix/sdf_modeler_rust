struct Camera {
    inv_view_proj: mat4x4f,
    eye: vec4f,
    viewport: vec4f,
    time: f32,
    quality_mode: f32,
    grid_enabled: f32,
    selected_idx: f32,
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

    // Trilinear interp at clamped point + box_dist for continuity.
    // When inside, clamped == local_p and box_dist == 0 (same as before).
    // When outside, samples boundary voxels and adds box_dist.
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

// Displacement-only grid sampling (for differential SDF sculpt nodes with analytical child).
// Returns 0.0 outside the grid (neutral displacement).
fn disp_voxel_grid(local_p: vec3f, node_idx: u32) -> f32 {
    let offset = u32(nodes[node_idx].extra0.x);
    let res    = u32(nodes[node_idx].extra0.y);
    let bmin   = nodes[node_idx].extra1.xyz;
    let bmax   = nodes[node_idx].extra2.xyz;

    let norm = (local_p - bmin) / (bmax - bmin);
    // Outside grid: no displacement
    if any(norm < vec3f(0.0)) || any(norm > vec3f(1.0)) {
        return 0.0;
    }

    let max_c = f32(res - 1u);
    let gc = norm * max_c;

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

fn sdf_ellipsoid(p: vec3f, s: vec3f) -> f32 {
    let k0 = length(p / s);
    let k1 = length(p / (s * s));
    return select(k0 * (k0 - 1.0) / k1, length(p) - min(s.x, min(s.y, s.z)), k1 < 0.0001);
}

fn sdf_hex_prism(p: vec3f, s: vec3f) -> f32 {
    let k = vec3f(-0.8660254038, 0.5, 0.57735026919);
    let pa = abs(p);
    let dot2 = 2.0 * min(k.x * pa.x + k.y * pa.z, 0.0);
    let rx = pa.x - dot2 * k.x;
    let rz = pa.z - dot2 * k.y;
    let clampedRx = clamp(rx, -k.z * s.x, k.z * s.x);
    let d1 = length(vec2f(rx - clampedRx, rz - s.x)) * sign(rz - s.x);
    let d2 = pa.y - s.y;
    return min(max(d1, d2), 0.0) + length(max(vec2f(d1, d2), vec2f(0.0)));
}

fn sdf_pyramid(p: vec3f, s: vec3f) -> f32 {
    let h = s.y;
    let b = s.x;
    let m2 = h * h + 0.25;
    var xz = vec2f(abs(p.x), abs(p.z));
    if xz.y > xz.x { xz = vec2f(xz.y, xz.x); }
    xz -= vec2f(0.5) * b;
    let q = vec3f(xz.y, h * p.y - 0.5 * xz.x, h * xz.x + 0.5 * p.y);
    let ss = max(-q.x, 0.0);
    let t = clamp((q.y - 0.5 * xz.y) / (m2 + 0.25), 0.0, 1.0);
    let a = m2 * (q.x + ss) * (q.x + ss) + q.y * q.y;
    let bb = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    let d2 = select(bb, a, min(q.y, -q.x * m2 - q.y * 0.5) > 0.0);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
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
