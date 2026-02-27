// Ray-AABB intersection and CSG (Constructive Solid Geometry) boolean operations.
// CSG ops take vec2f(distance, material_id) pairs and a smoothness parameter k.

// Ray-AABB intersection — returns vec2f(t_near, t_far).
// Used by raymarchers to skip empty space before the scene bounding box.
fn ray_aabb(ro: vec3f, inv_rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
    let t1 = (bmin - ro) * inv_rd;
    let t2 = (bmax - ro) * inv_rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(t_near, t_far);
}

// Hard union: picks the closer surface.
fn op_union(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if a.x < b.x { return a; } else { return b; }
}

// Smooth union: blends two surfaces with smoothness radius k.
fn op_smooth_union(a: vec2f, b: vec2f, k: f32) -> vec2f {
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(k, 0.0001), 0.0, 1.0);
    let d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    let mat = select(b.y, a.y, a.x < b.x);
    return vec2f(d, mat);
}

// Subtraction: carves shape b out of shape a. Supports smooth blending when k > 0.
fn op_subtract(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if k < 0.0001 {
        return vec2f(max(a.x, -b.x), a.y);
    }
    let h = clamp(0.5 - 0.5 * (a.x + b.x) / k, 0.0, 1.0);
    let d = mix(a.x, -b.x, h) + k * h * (1.0 - h);
    return vec2f(d, a.y);
}

// Intersection: keeps only the overlapping region. Supports smooth blending when k > 0.
fn op_intersect(a: vec2f, b: vec2f, k: f32) -> vec2f {
    if k < 0.0001 {
        if a.x > b.x { return a; } else { return b; }
    }
    let h = clamp(0.5 - 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    let d = mix(b.x, a.x, h) + k * h * (1.0 - h);
    return vec2f(d, select(b.y, a.y, a.x > b.x));
}
