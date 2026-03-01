// SDF modifier functions — these transform a point before SDF evaluation.
// Each takes a point and modifier parameters, returns the modified point.

fn twist_point(p: vec3f, rate: f32) -> vec3f {
    let c = cos(rate * p.y);
    let s = sin(rate * p.y);
    return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

fn bend_point(p: vec3f, k: f32) -> vec3f {
    let c = cos(k * p.x);
    let s = sin(k * p.x);
    return vec3f(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

fn taper_point(p: vec3f, factor: f32) -> vec3f {
    let s = 1.0 / (1.0 + factor * p.y);
    return vec3f(p.x * s, p.y, p.z * s);
}

fn elongate_point(p: vec3f, h: vec3f) -> vec3f {
    return p - clamp(p, -h, h);
}

fn mirror_point(p: vec3f, axes: vec3f) -> vec3f {
    return select(p, abs(p), axes > vec3f(0.5));
}

fn repeat_point(p: vec3f, s: vec3f) -> vec3f {
    var q = p;
    if s.x > 0.0 { q.x = q.x - s.x * round(q.x / s.x); }
    if s.y > 0.0 { q.y = q.y - s.y * round(q.y / s.y); }
    if s.z > 0.0 { q.z = q.z - s.z * round(q.z / s.z); }
    return q;
}

fn finite_repeat_point(p: vec3f, s: vec3f, c: vec3f) -> vec3f {
    var q = p;
    if s.x > 0.0 { q.x = q.x - s.x * clamp(round(q.x / s.x), -c.x, c.x); }
    if s.y > 0.0 { q.y = q.y - s.y * clamp(round(q.y / s.y), -c.y, c.y); }
    if s.z > 0.0 { q.z = q.z - s.z * clamp(round(q.z / s.z), -c.z, c.z); }
    return q;
}

// Radial repeat: rotate point to nearest angular sector.
// count = number of copies, axis = 0(X), 1(Y), 2(Z).
fn radial_repeat_point(p: vec3f, count: f32, axis: f32) -> vec3f {
    let n = max(count, 1.0);
    let sector = 6.28318530718 / n; // TAU / n
    // Extract the 2D plane perpendicular to the chosen axis
    var a: f32;
    var r: f32;
    if axis < 0.5 {
        // X axis: repeat in YZ plane
        a = atan2(p.z, p.y);
        r = length(vec2f(p.y, p.z));
    } else if axis < 1.5 {
        // Y axis: repeat in XZ plane
        a = atan2(p.z, p.x);
        r = length(vec2f(p.x, p.z));
    } else {
        // Z axis: repeat in XY plane
        a = atan2(p.y, p.x);
        r = length(vec2f(p.x, p.y));
    }
    // Snap angle to nearest sector
    a = a - sector * round(a / sector);
    let ca = cos(a);
    let sa = sin(a);
    if axis < 0.5 {
        return vec3f(p.x, r * ca, r * sa);
    } else if axis < 1.5 {
        return vec3f(r * ca, p.y, r * sa);
    } else {
        return vec3f(r * ca, r * sa, p.z);
    }
}
