// GPU object picking via compute shader.
// Casts a single ray from mouse position, returns the closest hit node ID and position.
//
// This file contains placeholder markers like /*MARCH_MAX_STEPS*/ that are
// replaced at runtime by shader_templates.rs with values from RenderConfig.
// Do not remove or rename these markers.
//
// Depends on: bindings.wgsl, operations.wgsl (ray_aabb), and the codegen-generated
// scene_sdf() function.

// PERFORMANCE CRITICAL: keep simple, avoid branches.
fn pick_ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    let inv_rd = 1.0 / rd;
    let aabb = ray_aabb(ro, inv_rd, camera.scene_min.xyz, camera.scene_max.xyz);
    let max_dist = min(aabb.y + 1.0, /*MARCH_MAX_DIST*/);
    if aabb.x > aabb.y || aabb.y < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    var t = max(aabb.x - 0.01, 0.0);
    var mat_id = -1.0;
    for (var i = 0; i < /*MARCH_MAX_STEPS*/; i++) {
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        if hit.x < /*MARCH_EPSILON*/ {
            mat_id = hit.y;
            break;
        }
        t += hit.x * /*MARCH_STEP_MULT*/;
        if t > max_dist { break; }
    }
    if mat_id < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    return vec2f(t, mat_id);
}

struct PickInput {
    mouse_pos: vec2f,
    _pad: vec2f,
}

@group(2) @binding(0) var<uniform> pick_in: PickInput;
@group(2) @binding(1) var<storage, read_write> pick_out: array<f32, 8>;

@compute @workgroup_size(1)
fn cs_pick() {
    // Generate ray from mouse position (same logic as fs_main)
    let uv = pick_in.mouse_pos / camera.viewport.zw * 2.0 - 1.0;
    let ndc = vec4f(uv.x, -uv.y, 1.0, 1.0);
    let world = camera.inv_view_proj * ndc;
    let rd = normalize(world.xyz / world.w - camera.eye.xyz);
    let ro = camera.eye.xyz;

    let hit = pick_ray_march(ro, rd);
    let t = hit.x;
    let mat_id = hit.y;

    let p = ro + rd * t;

    pick_out[0] = mat_id;
    pick_out[1] = t;
    pick_out[2] = p.x;
    pick_out[3] = p.y;
    pick_out[4] = p.z;
    pick_out[5] = 0.0;
    pick_out[6] = 0.0;
    pick_out[7] = 0.0;
}
