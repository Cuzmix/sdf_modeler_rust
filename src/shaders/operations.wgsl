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
// Preserves the winner's full material info (mat_a, mat_b, blend_factor).
fn op_union(a: vec4f, b: vec4f, k: f32) -> vec4f {
    if a.x < b.x {
        return a;
    } else {
        return b;
    }
}

// Smooth union: blends two surfaces with smoothness radius k.
// color_k controls material blend independently (-1 = follow k).
fn op_smooth_union(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(k, 0.0001), 0.0, 1.0);
    let d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    let ch = clamp(0.5 + 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
    return vec4f(d, a.y, b.y, 1.0 - ch);
}

// Subtraction: carves shape b out of shape a. Supports smooth blending when k > 0.
// color_k controls material blend independently (-1 = follow k).
fn op_subtract(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    if k < 0.0001 {
        return vec4f(max(a.x, -b.x), a.y, -1.0, 0.0);
    }
    let ck = select(k, color_k, color_k >= 0.0);
    let h = clamp(0.5 - 0.5 * (a.x + b.x) / k, 0.0, 1.0);
    let d = mix(a.x, -b.x, h) + k * h * (1.0 - h);
    let ch = clamp(0.5 - 0.5 * (a.x + b.x) / ck, 0.0, 1.0);
    return vec4f(d, a.y, b.y, ch);
}

// Intersection: keeps only the overlapping region. Supports smooth blending when k > 0.
// color_k controls material blend independently (-1 = follow k).
fn op_intersect(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    if k < 0.0001 {
        if a.x > b.x {
            return vec4f(a.x, a.y, -1.0, 0.0);
        } else {
            return vec4f(b.x, b.y, -1.0, 0.0);
        }
    }
    let ck = select(k, color_k, color_k >= 0.0);
    let h = clamp(0.5 - 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    let d = mix(b.x, a.x, h) + k * h * (1.0 - h);
    let ch = clamp(0.5 - 0.5 * (b.x - a.x) / ck, 0.0, 1.0);
    return vec4f(d, a.y, b.y, ch);
}

// Chamfer union: creates a 45° beveled edge at the union boundary.
// The bevel width is controlled by k.
fn op_chamfer_union(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    let d = min(min(a.x, b.x), (a.x - k + b.x) * sqrt(0.5));
    // Same blend convention as smooth union: 0 = pure a, 1 = pure b
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
    return vec4f(d, a.y, b.y, 1.0 - h);
}

// Chamfer subtraction: carves shape b from shape a with a beveled edge.
fn op_chamfer_subtract(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    let d = max(max(a.x, -b.x), (a.x + k - b.x) * sqrt(0.5));
    // Same blend convention as smooth subtract: 0 = pure a, increasing = toward b
    let h = clamp(0.5 - 0.5 * (a.x + b.x) / max(ck, 0.0001), 0.0, 1.0);
    return vec4f(d, a.y, b.y, h);
}

// Chamfer intersection: keeps only the overlap with a beveled edge.
fn op_chamfer_intersect(a: vec4f, b: vec4f, k: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    let d = max(max(a.x, b.x), (a.x - k + b.x) * sqrt(0.5));
    // Same blend convention as smooth intersect: 0 = pure a, increasing = toward b
    let h = clamp(0.5 - 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
    return vec4f(d, a.y, b.y, h);
}

// GLSL-style modulo (x - y * floor(x/y)), differs from WGSL % for negative values.
fn glsl_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

// Staircase union: creates a step-like pattern at the union boundary.
// k = staircase radius, n = number of steps. Based on Mercury hg_sdf.
fn op_stairs_union(a: vec4f, b: vec4f, k: f32, n: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    let s = k / max(n, 1.0);
    let u = b.x - k;
    let stair_d = 0.5 * (u + a.x + abs(glsl_mod(u - a.x + s, 2.0 * s) - s));
    let d = min(min(a.x, b.x), stair_d);
    // Same blend convention as smooth union: 0 = pure a, 1 = pure b
    let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
    return vec4f(d, a.y, b.y, 1.0 - h);
}

// Staircase subtraction: carves shape b from shape a with a stepped edge.
fn op_stairs_subtract(a: vec4f, b: vec4f, k: f32, n: f32, color_k: f32) -> vec4f {
    let neg_a = vec4f(-a.x, a.y, a.z, a.w);
    let result = op_stairs_union(neg_a, b, k, n, color_k);
    return vec4f(-result.x, result.y, result.z, result.w);
}

// 2D 45-degree rotation helper for column operations.
fn rotate_45(p: vec2f) -> vec2f {
    let c = sqrt(0.5); // cos(45°) = sin(45°) = 1/√2
    return vec2f(p.x * c - p.y * c, p.x * c + p.y * c);
}

// 1D modular repeat: mirrors the coordinate into [-size, size].
fn p_mod1(p: f32, size: f32) -> f32 {
    return glsl_mod(p + size, 2.0 * size) - size;
}

// Columns union: creates repeating cylindrical columns at the union boundary.
// k = column region radius, n = number of columns. Based on Mercury hg_sdf.
fn op_columns_union(a: vec4f, b: vec4f, k: f32, n: f32, color_k: f32) -> vec4f {
    let ck = select(k, color_k, color_k >= 0.0);
    if a.x < k && b.x < k {
        var p = vec2f(a.x, b.x);
        let column_radius = k * sqrt(2.0) / ((max(n, 1.0) - 1.0) * 2.0 + sqrt(2.0));
        p = rotate_45(p);
        p.x -= sqrt(2.0) / 2.0 * k;
        p.x += column_radius * sqrt(2.0);
        if glsl_mod(n, 2.0) == 1.0 {
            p.y += column_radius;
        }
        p.y = p_mod1(p.y, column_radius * 2.0);
        let col_d = min(length(p) - column_radius, p.x);
        let d = min(min(a.x, b.x), col_d);
        // Same blend convention as smooth union: 0 = pure a, 1 = pure b
        let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
        return vec4f(d, a.y, b.y, 1.0 - h);
    } else {
        // Outside column region — hard union fallback
        let h = clamp(0.5 + 0.5 * (b.x - a.x) / max(ck, 0.0001), 0.0, 1.0);
        let d = min(a.x, b.x);
        return vec4f(d, a.y, b.y, 1.0 - h);
    }
}

// Columns subtraction: carves shape b from shape a with columnar edge detail.
fn op_columns_subtract(a: vec4f, b: vec4f, k: f32, n: f32, color_k: f32) -> vec4f {
    let neg_a = vec4f(-a.x, a.y, a.z, a.w);
    let result = op_columns_union(neg_a, b, k, n, color_k);
    return vec4f(-result.x, result.y, result.z, result.w);
}
