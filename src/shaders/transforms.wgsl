// Geometric transform utilities used by the SDF codegen pipeline.

// Euler rotation (XYZ order) — applies rotation angles (radians) to a point.
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
