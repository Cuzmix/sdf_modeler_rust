use crate::settings::RenderConfig;

// ---------------------------------------------------------------------------
// Shader template: everything before scene_sdf
// ---------------------------------------------------------------------------

pub(crate) const SHADER_PRELUDE: &str = include_str!("../shaders/prelude.wgsl");

// ---------------------------------------------------------------------------
// Shader template: everything after scene_sdf
// ---------------------------------------------------------------------------

pub(crate) const SHADER_POSTLUDE: &str = r#"
// --- Rendering quality constants ---
const SHADOW_STEPS: i32 = /*SHADOW_STEPS*/;
const SHADOW_PENUMBRA_K: f32 = /*SHADOW_PENUMBRA_K*/;
const AO_SAMPLES: i32 = /*AO_SAMPLES*/;
const AO_STEP: f32 = /*AO_STEP*/;
const AO_DECAY: f32 = /*AO_DECAY*/;

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

    // Selection outline: precompute in uniform control flow (fwidth requires this)
    let sel_f = select(0.0, 1.0, is_selected(mat_id));
    let fwidth_sel = fwidth(sel_f);

    // Sky gradient (base color when no SDF hit)
    let sky_grad_t = uv.y * 0.5 + 0.5;
    let bg_sky = mix(vec3f(/*SKY_HORIZON*/), vec3f(/*SKY_ZENITH*/), sky_grad_t);

    // --- Compute base color: sky or shaded SDF ---
    var color = bg_sky;
    var outline_a = 0.0;

    if !sdf_miss {
        let p = ro + rd * t;
        let n = calc_normal(p, t);

        // Material properties
        let mat_metallic = select(0.0, nodes[mat_id].type_op.z, mat_id >= 0);
        let mat_roughness = select(0.5, nodes[mat_id].type_op.w, mat_id >= 0);

        // Key light
        let key_dir = normalize(vec3f(/*KEY_LIGHT_DIR*/));
        let key_h = normalize(key_dir - rd);
        let key_diff = max(dot(n, key_dir), 0.0);
        let spec_power = max(4.0, /*KEY_SPEC_POWER*/ * (1.0 - mat_roughness) * (1.0 - mat_roughness));
        let key_spec = pow(max(dot(n, key_h), 0.0), spec_power);
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

        // Selection outline via selected_sdf distance check
        if !is_selected(mat_id) {
            let sel_d = selected_sdf(p);
            let pixel_world = t * 2.0 / camera.viewport.w;
            outline_a = 1.0 - smoothstep(0.0, pixel_world * /*OUTLINE_THICKNESS*/, sel_d);
        } else {
            let ndotv = max(dot(n, -rd), 0.0);
            outline_a = 1.0 - smoothstep(0.0, 0.15, ndotv);
        }

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

    // Selection outline: selected_sdf for object boundaries, fwidth for sky boundary
    let final_outline = max(outline_a, fwidth_sel);
    if final_outline > 0.01 {
        color = mix(color, vec3f(/*OUTLINE_COLOR*/), final_outline);
    }

    return vec4f(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Pick compute shader postlude
// ---------------------------------------------------------------------------

pub(crate) const PICK_COMPUTE_POSTLUDE: &str = r#"
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
"#;

// ---------------------------------------------------------------------------
// Brush compute shader (standalone -- no scene/camera bindings needed)
// ---------------------------------------------------------------------------

pub(crate) const BRUSH_COMPUTE_SHADER: &str = include_str!("../shaders/brush.wgsl");

// ---------------------------------------------------------------------------
// Composite compute shader entry point
// ---------------------------------------------------------------------------

pub(crate) const COMPOSITE_COMPUTE_ENTRY: &str = include_str!("../shaders/composite_entry.wgsl");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format f32 as a WGSL literal (always includes decimal point).
pub(crate) fn format_f32(v: f32) -> String {
    let s = format!("{}", v);
    if s.contains('.') { s } else { format!("{}.0", s) }
}

/// Format [f32; 3] as WGSL vec3 components: "x, y, z".
pub(crate) fn format_vec3(v: [f32; 3]) -> String {
    format!("{}, {}, {}", format_f32(v[0]), format_f32(v[1]), format_f32(v[2]))
}

/// Apply the 4 raymarching placeholders shared by render and pick shaders.
pub(crate) fn apply_march_placeholders(src: &str, config: &RenderConfig) -> String {
    src.replace("/*MARCH_MAX_STEPS*/", &config.march_max_steps.to_string())
        .replace("/*MARCH_EPSILON*/", &format_f32(config.march_epsilon))
        .replace("/*MARCH_STEP_MULT*/", &format_f32(config.march_step_multiplier))
        .replace("/*MARCH_MAX_DIST*/", &format_f32(config.march_max_distance))
}

/// Build the SHADER_POSTLUDE with all placeholder replacements.
pub(crate) fn build_postlude(config: &RenderConfig) -> String {
    let shadow_line = if config.shadows_enabled {
        format!(
            "    var shadow = 1.0;\n    if camera.quality_mode < 0.5 {{ shadow = soft_shadow(p + n * {}, key_dir, {}, {}, SHADOW_PENUMBRA_K); }}",
            format_f32(config.shadow_bias),
            format_f32(config.shadow_mint),
            format_f32(config.shadow_maxt),
        )
    } else {
        "    let shadow = 1.0;".to_string()
    };

    let ao_line = if config.ao_enabled {
        "    var ao = 1.0;\n    if camera.quality_mode < 0.5 { ao = calc_ao(p, n); }".to_string()
    } else {
        "    let ao = 1.0;".to_string()
    };

    let fog_line = if config.fog_enabled {
        format!(
            "    {{ let fog_amt = 1.0 - exp(-t * {}); let sun_s = pow(max(dot(rd, key_dir), 0.0), 8.0); let fog_c = mix(vec3f({}), vec3f(1.0, 0.9, 0.7), sun_s); color = mix(color, fog_c, fog_amt); }}",
            format_f32(config.fog_density),
            format_vec3(config.fog_color),
        )
    } else {
        "".to_string()
    };

    let sky_cutoff = config.march_max_distance - 1.0;

    apply_march_placeholders(SHADER_POSTLUDE, config)
        .replace("/*SHADOW_STEPS*/", &config.shadow_steps.to_string())
        .replace("/*SHADOW_PENUMBRA_K*/", &format_f32(config.shadow_penumbra_k))
        .replace("/*AO_SAMPLES*/", &config.ao_samples.to_string())
        .replace("/*AO_STEP*/", &format_f32(config.ao_step))
        .replace("/*AO_DECAY*/", &format_f32(config.ao_decay))
        .replace("/*AO_INTENSITY*/", &format_f32(config.ao_intensity))
        .replace("/*KEY_LIGHT_DIR*/", &format_vec3(config.key_light_dir))
        .replace("/*KEY_DIFFUSE*/", &format_f32(config.key_diffuse))
        .replace("/*KEY_SPEC_POWER*/", &format_f32(config.key_spec_power))
        .replace("/*KEY_SPEC_INTENSITY*/", &format_f32(config.key_spec_intensity))
        .replace("/*FILL_LIGHT_DIR*/", &format_vec3(config.fill_light_dir))
        .replace("/*FILL_INTENSITY*/", &format_f32(config.fill_intensity))
        .replace("/*AMBIENT*/", &format_f32(config.ambient))
        .replace("/*SKY_HORIZON*/", &format_vec3(config.sky_horizon))
        .replace("/*SKY_ZENITH*/", &format_vec3(config.sky_zenith))
        .replace("/*SKY_CUTOFF*/", &format_f32(sky_cutoff))
        .replace("/*GAMMA*/", &format_f32(config.gamma))
        .replace("/*SHADOW_LINE*/", &shadow_line)
        .replace("/*AO_LINE*/", &ao_line)
        .replace("/*FOG_LINE*/", &fog_line)
        .replace("/*OUTLINE_COLOR*/", &format_vec3(config.outline_color))
        .replace("/*OUTLINE_THICKNESS*/", &format_f32(config.outline_thickness))
        .replace("/*TONEMAP_LINE*/", if config.tonemapping_aces {
            "    { let ta = color * (2.51 * color + vec3f(0.03)); let tb = color * (2.43 * color + vec3f(0.59)) + vec3f(0.14); color = clamp(ta / tb, vec3f(0.0), vec3f(1.0)); }"
        } else {
            ""
        })
}
