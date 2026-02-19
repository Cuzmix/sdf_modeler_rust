struct Camera {
    inv_view_proj: mat4x4f,
    eye: vec4f,
    viewport: vec4f,
    time: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;

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

fn sdf_sphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn scene_sdf(p: vec3f) -> f32 {
    return sdf_sphere(p, 1.0);
}

fn calc_normal(p: vec3f) -> vec3f {
    let e = vec2f(0.001, 0.0);
    return normalize(vec3f(
        scene_sdf(p + e.xyy) - scene_sdf(p - e.xyy),
        scene_sdf(p + e.yxy) - scene_sdf(p - e.yxy),
        scene_sdf(p + e.yyx) - scene_sdf(p - e.yyx),
    ));
}

fn ray_march(ro: vec3f, rd: vec3f) -> f32 {
    var t = 0.0;
    for (var i = 0; i < 96; i++) {
        let p = ro + rd * t;
        let d = scene_sdf(p);
        if d < 0.001 {
            break;
        }
        t += d * 0.9;
        if t > 50.0 {
            break;
        }
    }
    return t;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let local = frag_coord.xy - camera.viewport.xy;
    let uv = local / camera.viewport.zw * 2.0 - 1.0;

    let ndc = vec4f(uv.x, -uv.y, 1.0, 1.0);
    let world = camera.inv_view_proj * ndc;
    let rd = normalize(world.xyz / world.w - camera.eye.xyz);
    let ro = camera.eye.xyz;

    let t = ray_march(ro, rd);

    // Background gradient
    if t > 49.0 {
        let bg = mix(vec3f(0.08, 0.08, 0.12), vec3f(0.02, 0.02, 0.04), uv.y * 0.5 + 0.5);
        return vec4f(bg, 1.0);
    }

    // Blinn-Phong shading
    let p = ro + rd * t;
    let n = calc_normal(p);
    let light_dir = normalize(vec3f(1.0, 2.0, 3.0));
    let h = normalize(light_dir - rd);

    let ambient = 0.08;
    let diffuse = max(dot(n, light_dir), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 32.0);

    let albedo = vec3f(0.8, 0.3, 0.2);
    let color = albedo * (ambient + diffuse * 0.9) + vec3f(1.0) * specular * 0.5;

    return vec4f(pow(color, vec3f(1.0 / 2.2)), 1.0);
}
