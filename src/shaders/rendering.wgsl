// Main render pipeline: raymarching, lighting, and fragment shader.
//
// This file contains placeholder markers like /*MARCH_MAX_STEPS*/ that are
// replaced at runtime by shader_templates.rs with values from RenderConfig.
// Do not remove or rename these markers.
//
// Depends on: bindings.wgsl, operations.wgsl (ray_aabb), and the codegen-generated
// scene_sdf() function.

// --- Rendering quality constants ---
const SHADOW_STEPS: i32 = /*SHADOW_STEPS*/;
const SHADOW_PENUMBRA_K: f32 = /*SHADOW_PENUMBRA_K*/;
const AO_SAMPLES: i32 = /*AO_SAMPLES*/;
const AO_STEP: f32 = /*AO_STEP*/;
const AO_DECAY: f32 = /*AO_DECAY*/;

// PERFORMANCE CRITICAL: keep simple, avoid branches.
// This is the inner loop of the renderer — runs per-pixel, up to MARCH_MAX_STEPS iterations.
fn ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    let fast = camera.quality_mode > 0.5;
    let eff_steps = select(/*MARCH_MAX_STEPS*/, /*MARCH_MAX_STEPS*/ / 2, fast);
    let eps = select(/*MARCH_EPSILON*/, /*MARCH_EPSILON*/ * 4.0, fast);
    let step_mult = select(/*MARCH_STEP_MULT*/, min(/*MARCH_STEP_MULT*/ * 1.5, 1.0), fast);

    // Scene-level ray-AABB: skip empty space before the scene
    let inv_rd = 1.0 / rd;
    let aabb = ray_aabb(ro, inv_rd, camera.scene_min.xyz, camera.scene_max.xyz);
    let max_dist = min(aabb.y + 1.0, /*MARCH_MAX_DIST*/);
    if aabb.x > aabb.y || aabb.y < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    var t = max(aabb.x - 0.01, 0.0);

    // Enhanced sphere tracing (Keinert over-relaxation)
    var omega = select(1.2, 1.0, fast);
    var prev_d = 1e10;
    var prev_step = 0.0;
    var mat_id = -1.0;
    for (var i = 0; i < /*MARCH_MAX_STEPS*/; i++) {
        if i >= eff_steps { break; }
        let p = ro + rd * t;
        let hit = scene_sdf(p);
        let d = hit.x;
        let step = d * omega * step_mult;
        // Overshoot detection: if combined radii can't bridge the step, undo
        if omega > 1.0 && prev_d + d < prev_step {
            t -= prev_step;
            omega = 1.0;
            prev_d = 1e10;
            prev_step = 0.0;
            continue;
        }
        if d < eps {
            mat_id = hit.y;
            break;
        }
        t += step;
        prev_step = step;
        prev_d = d;
        if t > max_dist { break; }
    }
    if mat_id < 0.0 {
        return vec2f(/*MARCH_MAX_DIST*/ + 1.0, -1.0);
    }
    return vec2f(t, mat_id);
}

// PERFORMANCE CRITICAL: keep simple, avoid branches.
fn calc_normal(p: vec3f, t: f32) -> vec3f {
    // Tetrahedron technique: 4 SDF evals instead of 6.
    // Distance-adaptive epsilon: larger at distance (reduces aliasing), tighter up close (more detail).
    let e = clamp(0.001 * t, 0.0005, 0.05);
    let k = vec2f(1.0, -1.0);
    return normalize(
        k.xyy * scene_sdf(p + k.xyy * e).x +
        k.yyx * scene_sdf(p + k.yyx * e).x +
        k.yxy * scene_sdf(p + k.yxy * e).x +
        k.xxx * scene_sdf(p + k.xxx * e).x
    );
}

fn soft_shadow(ro: vec3f, rd: vec3f, mint: f32, maxt: f32, k: f32) -> f32 {
    // iq's improved soft shadows with triangulation (reduced light leaking)
    var res = 1.0;
    var t = mint;
    var ph = 1e20;
    for (var i = 0; i < SHADOW_STEPS; i++) {
        let h = scene_sdf(ro + rd * t).x;
        if h < 0.005 {
            return 0.0;
        }
        let y = h * h / (2.0 * ph);
        let d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0001, t - y));
        ph = h;
        t += h;
        if t > maxt { break; }
    }
    return clamp(res, 0.0, 1.0);
}

fn calc_ao(p: vec3f, n: vec3f) -> f32 {
    var occ = 0.0;
    var weight = 1.0;
    for (var i = 1; i <= AO_SAMPLES; i++) {
        let dist = AO_STEP * f32(i);
        let d = scene_sdf(p + n * dist).x;
        occ += (dist - d) * weight;
        weight *= AO_DECAY;
    }
    return clamp(1.0 - /*AO_INTENSITY*/ * occ, 0.0, 1.0);
}

fn get_node_color(mat_id: i32) -> vec3f {
    if mat_id < 0 { return vec3f(0.5); }
    return nodes[mat_id].color.xyz;
}

fn is_selected(mat_id: i32) -> bool {
    if mat_id < 0 { return false; }
    return nodes[mat_id].color.w > 0.5;
}

// Material helpers: Primitive packs emissive in extra1/extra2, Sculpt uses spare .w slots
fn get_node_emissive(mat_id: i32) -> vec3f {
    if mat_id < 0 { return vec3f(0.0); }
    let is_sculpt = nodes[mat_id].type_op.x > 19.5;
    if is_sculpt {
        // Sculpt: extra0.zw = emissive.xy, extra1.w = emissive.z
        return vec3f(nodes[mat_id].extra0.z, nodes[mat_id].extra0.w, nodes[mat_id].extra1.w);
    }
    // Primitive: extra1.xyz = emissive
    return nodes[mat_id].extra1.xyz;
}

fn get_node_emissive_intensity(mat_id: i32) -> f32 {
    if mat_id < 0 { return 0.0; }
    let is_sculpt = nodes[mat_id].type_op.x > 19.5;
    if is_sculpt {
        // Sculpt: type_op.y = emissive_intensity
        return nodes[mat_id].type_op.y;
    }
    // Primitive: extra1.w = emissive_intensity
    return nodes[mat_id].extra1.w;
}

fn get_node_fresnel(mat_id: i32) -> f32 {
    if mat_id < 0 { return 0.04; }
    let is_sculpt = nodes[mat_id].type_op.x > 19.5;
    if is_sculpt {
        // Sculpt: extra2.w = fresnel
        return nodes[mat_id].extra2.w;
    }
    // Primitive: extra2.x = fresnel
    return nodes[mat_id].extra2.x;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let local = frag_coord.xy - camera.viewport.xy;
    let uv = local / camera.viewport.zw * 2.0 - 1.0;

    let ndc = vec4f(uv.x, -uv.y, 1.0, 1.0);
    let world = camera.inv_view_proj * ndc;
    let rd = normalize(world.xyz / world.w - camera.eye.xyz);
    let ro = camera.eye.xyz;

    let hit = ray_march(ro, rd);
    let t = hit.x;
    let mat_id = i32(hit.y + 0.5);
    let sdf_miss = t > /*SKY_CUTOFF*/;

    // Precompute grid plane hit in uniform control flow (fwidth requires this)
    let t_grid = -(ro.y / rd.y);
    let gp = ro + rd * t_grid;
    let dwdx = fwidth(gp.x);
    let dwdz = fwidth(gp.z);

    // Sky gradient (base color when no SDF hit)
    let sky_grad_t = uv.y * 0.5 + 0.5;
    let bg_sky = mix(vec3f(/*SKY_HORIZON*/), vec3f(/*SKY_ZENITH*/), sky_grad_t);

    // --- Compute base color: sky or shaded SDF ---
    var color = bg_sky;

    if !sdf_miss {
        let p = ro + rd * t;
        let n = calc_normal(p, t);

        // Material properties
        let mat_metallic = select(0.0, nodes[mat_id].type_op.z, mat_id >= 0);
        let mat_roughness = select(0.5, nodes[mat_id].type_op.w, mat_id >= 0);
        let mat_fresnel = get_node_fresnel(mat_id);

        // Key light
        let key_dir = normalize(vec3f(/*KEY_LIGHT_DIR*/));
        let key_h = normalize(key_dir - rd);
        let key_diff = max(dot(n, key_dir), 0.0);
        let spec_power = max(4.0, /*KEY_SPEC_POWER*/ * (1.0 - mat_roughness) * (1.0 - mat_roughness));
        let key_spec_raw = pow(max(dot(n, key_h), 0.0), spec_power);
        // Schlick Fresnel approximation
        let VdotH = max(dot(-rd, key_h), 0.0);
        let F = mat_fresnel + (1.0 - mat_fresnel) * pow(1.0 - VdotH, 5.0);
        let key_spec = key_spec_raw * F;
        /*SHADOW_LINE*/

        // Fill light
        let fill_dir = normalize(vec3f(/*FILL_LIGHT_DIR*/));
        let fill_diff = max(dot(n, fill_dir), 0.0) * /*FILL_INTENSITY*/;

        // Ambient occlusion
        /*AO_LINE*/

        let albedo = get_node_color(mat_id);
        let diff_factor = 1.0 - mat_metallic * 0.8;
        let spec_tint = mix(vec3f(1.0), albedo, mat_metallic);
        // Hemispherical sky light (iq outdoor lighting)
        let sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        let shadow_col = pow(vec3f(shadow), vec3f(1.0, 1.2, 1.5));
        color = albedo * diff_factor * (sky * /*AMBIENT*/ * ao + key_diff * shadow_col * /*KEY_DIFFUSE*/ + fill_diff * ao)
                  + spec_tint * key_spec * shadow * /*KEY_SPEC_INTENSITY*/;

        // Emissive contribution (added before tonemapping for natural overbright bloom)
        let emissive_col = get_node_emissive(mat_id);
        let emissive_int = get_node_emissive_intensity(mat_id);
        color += emissive_col * emissive_int;

        /*FOG_LINE*/

        /*TONEMAP_LINE*/
        color = pow(color, vec3f(1.0 / /*GAMMA*/));
    }

    // --- Grid overlay: transparent blend AFTER base color ---
    // Condition matches TypeGPU: grid enabled, ray not parallel, grid in front,
    // AND (SDF missed OR grid closer than SDF hit)
    if camera.grid_enabled > 0.5 && abs(rd.y) > 0.0001 && t_grid > 0.0 && (sdf_miss || t_grid < t) {
        // Fine grid lines (1 unit spacing)
        let dist_fx = 0.5 - abs(fract(gp.x) - 0.5);
        let dist_fz = 0.5 - abs(fract(gp.z) - 0.5);
        let fine_x = 1.0 - smoothstep(0.0, dwdx * 1.5, dist_fx);
        let fine_z = 1.0 - smoothstep(0.0, dwdz * 1.5, dist_fz);
        let fine_line = max(fine_x, fine_z);

        // Coarse grid lines (5 unit spacing)
        let dist_cx = (0.5 - abs(fract(gp.x / 5.0) - 0.5)) * 5.0;
        let dist_cz = (0.5 - abs(fract(gp.z / 5.0) - 0.5)) * 5.0;
        let coarse_x = 1.0 - smoothstep(0.0, dwdx * 1.5, dist_cx);
        let coarse_z = 1.0 - smoothstep(0.0, dwdz * 1.5, dist_cz);
        let coarse_line = max(coarse_x, coarse_z);

        // Axis lines
        let axis_x = 1.0 - smoothstep(0.0, dwdz * 2.0, abs(gp.z));
        let axis_z = 1.0 - smoothstep(0.0, dwdx * 2.0, abs(gp.x));

        // Distance fade
        let fade = clamp(1.0 - t_grid / 80.0, 0.0, 1.0);
        let grid_i = max(fine_line * 0.3, coarse_line * 0.6);
        let grid_col = vec3f(grid_i + axis_x * 0.7, grid_i, grid_i + axis_z * 0.7);
        let alpha = max(max(grid_i, axis_x * 0.8), axis_z * 0.8) * fade;

        // Transparent blend over base color
        color = mix(color, grid_col, alpha);
    }

    // Selection flag in alpha: 0 = selected, 1 = not selected.
    // The blit shader reads this to draw post-process outline edges.
    let sel_alpha = select(1.0, 0.0, !sdf_miss && is_selected(mat_id));
    return vec4f(color, sel_alpha);
}
