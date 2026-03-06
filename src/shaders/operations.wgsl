// Ray-AABB intersection and CSG (Constructive Solid Geometry) boolean operations.
// CSG ops take vec4f(distance, mat_a, mat_b, blend_factor) pairs and a smoothness parameter k.
// mat_a = primary material, mat_b = secondary material (-1 if none), blend_factor = interpolation weight.

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

// Hard union: picks the closer surface. No blending.
fn op_union(a: vec4f, b: vec4f, k: f32) -> vec4f {
    if a.x < b.x {
        return vec4f(a.x, a.y, -1.0, 0.0);
    } else {
        return vec4f(b.x, b.y, -1.0, 0.0);
    }
}

// Smooth union: blends two surfaces with smoothness radius k.
// Stores both material IDs and the blend factor h for material interpolation.
fn op_smooth_union(a: vec4f, b: vec4f, k: f32) -> vec4f {
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(k, 0.0001), 0.0, 1.0);
    let d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    return vec4f(d, a.y, b.y, 1.0 - h);
}

// Subtraction: carves shape b out of shape a. Supports smooth blending when k > 0.
// When smooth (k > 0), the carved boundary blends toward the subtractor's material.
fn op_subtract(a: vec4f, b: vec4f, k: f32) -> vec4f {
    if k < 0.0001 {
        return vec4f(max(a.x, -b.x), a.y, -1.0, 0.0);
    }
    let h = clamp(0.5 - 0.5 * (a.x + b.x) / k, 0.0, 1.0);
    let d = mix(a.x, -b.x, h) + k * h * (1.0 - h);
    return vec4f(d, a.y, b.y, h);
}

// Intersection: keeps only the overlapping region. Supports smooth blending when k > 0.
// When smooth (k > 0), the intersection boundary blends both materials.
fn op_intersect(a: vec4f, b: vec4f, k: f32) -> vec4f {
    if k < 0.0001 {
        if a.x > b.x {
            return vec4f(a.x, a.y, -1.0, 0.0);
        } else {
            return vec4f(b.x, b.y, -1.0, 0.0);
        }
    }
    let h = clamp(0.5 - 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    let d = mix(b.x, a.x, h) + k * h * (1.0 - h);
    return vec4f(d, a.y, b.y, h);
}

// Chamfer union: creates a 45° beveled edge at the union boundary.
// The bevel width is controlled by k.
fn op_chamfer_union(a: vec4f, b: vec4f, k: f32) -> vec4f {
    let d = min(min(a.x, b.x), (a.x - k + b.x) * sqrt(0.5));
    // Blend factor: how close we are to the chamfer surface vs the original surfaces
    let chamfer_d = (a.x - k + b.x) * sqrt(0.5);
    let h = clamp(1.0 - (d - chamfer_d) / max(k, 0.0001), 0.0, 1.0) * 0.5;
    if a.x < b.x {
        return vec4f(d, a.y, b.y, h);
    } else {
        return vec4f(d, b.y, a.y, h);
    }
}

// Chamfer subtraction: carves shape b from shape a with a beveled edge.
fn op_chamfer_subtract(a: vec4f, b: vec4f, k: f32) -> vec4f {
    let d = max(max(a.x, -b.x), (a.x + k - b.x) * sqrt(0.5));
    let chamfer_d = (a.x + k - b.x) * sqrt(0.5);
    let h = clamp(1.0 - (chamfer_d - d) / max(k, 0.0001), 0.0, 1.0) * 0.5;
    return vec4f(d, a.y, b.y, h);
}

// Chamfer intersection: keeps only the overlap with a beveled edge.
fn op_chamfer_intersect(a: vec4f, b: vec4f, k: f32) -> vec4f {
    let d = max(max(a.x, b.x), (a.x - k + b.x) * sqrt(0.5));
    let chamfer_d = (a.x - k + b.x) * sqrt(0.5);
    let h = clamp(1.0 - (chamfer_d - d) / max(k, 0.0001), 0.0, 1.0) * 0.5;
    if a.x > b.x {
        return vec4f(d, a.y, b.y, h);
    } else {
        return vec4f(d, b.y, a.y, h);
    }
}
