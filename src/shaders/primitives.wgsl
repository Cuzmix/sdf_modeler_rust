// SDF primitive shape functions.
// Each takes a point p and scale/size vector s, returns signed distance.
// Based on iq's distance functions: https://iquilezles.org/articles/distfunctions/

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
    let d2 = select(min(a, bb), 0.0, min(q.y, -q.x * m2 - q.y * 0.5) > 0.0);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}
