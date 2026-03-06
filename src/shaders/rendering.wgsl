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

// Step count for debug heatmap (written by ray_march, read by shading)
var<private> debug_step_count: f32;

// PERFORMANCE CRITICAL: keep simple, avoid branches.
// This is the inner loop of the renderer — runs per-pixel, up to MARCH_MAX_STEPS iterations.
fn ray_march(ro: vec3f, rd: vec3f) -> vec4f {
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
    var mat_a = -1.0;
    var mat_b = -1.0;
    var blend = 0.0;
    var steps_taken = 0;
    for (var i = 0; i < /*MARCH_MAX_STEPS*/; i++) {
        if i >= eff_steps { break; }
        steps_taken = i + 1;
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
            mat_a = hit.y;
            mat_b = hit.z;
            blend = hit.w;
            break;
        }
        t += step;
        prev_step = step;
        prev_d = d;
        if t > max_dist { break; }
    }
    debug_step_count = f32(steps_taken) / f32(eff_steps);
    if mat_a < 0.0 {
        return vec4f(/*MARCH_MAX_DIST*/ + 1.0, -1.0, -1.0, 0.0);
    }
    return vec4f(t, mat_a, mat_b, blend);
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

// --- Cook-Torrance GGX BRDF functions (Filament reference) ---
const PI: f32 = 3.14159265359;

// GGX (Trowbridge-Reitz) normal distribution function
fn D_GGX(NoH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith GGX correlated geometric attenuation (height-correlated form)
// The (4 * NoV * NoL) denominator is cancelled in this form.
fn G_SmithGGX(NoV: f32, NoL: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ggx_v = NoL * sqrt(a2 + NoV * NoV * (1.0 - a2));
    let ggx_l = NoV * sqrt(a2 + NoL * NoL * (1.0 - a2));
    return 0.5 / max(ggx_v + ggx_l, 0.0001);
}

// Schlick Fresnel approximation (vec3f for metallic F0 = albedo)
fn F_Schlick_vec3(VoH: f32, f0: vec3f) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow(1.0 - VoH, 5.0);
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

// Material blending helpers: interpolate properties between two materials at CSG boundaries.
// When mat_b < 0 or blend_factor <= 0, returns mat_a's property unchanged (no blending).
fn get_blended_color(mat_a: i32, mat_b: i32, blend_factor: f32) -> vec3f {
    let color_a = get_node_color(mat_a);
    if mat_b < 0 || blend_factor <= 0.0 {
        return color_a;
    }
    let color_b = get_node_color(mat_b);
    return mix(color_a, color_b, blend_factor);
}

fn get_blended_roughness(mat_a: i32, mat_b: i32, blend_factor: f32) -> f32 {
    let rough_a = select(0.5, nodes[mat_a].type_op.w, mat_a >= 0);
    if mat_b < 0 || blend_factor <= 0.0 {
        return rough_a;
    }
    let rough_b = select(0.5, nodes[mat_b].type_op.w, mat_b >= 0);
    return mix(rough_a, rough_b, blend_factor);
}

fn get_blended_metallic(mat_a: i32, mat_b: i32, blend_factor: f32) -> f32 {
    let metal_a = select(0.0, nodes[mat_a].type_op.z, mat_a >= 0);
    if mat_b < 0 || blend_factor <= 0.0 {
        return metal_a;
    }
    let metal_b = select(0.0, nodes[mat_b].type_op.z, mat_b >= 0);
    return mix(metal_a, metal_b, blend_factor);
}

fn get_blended_fresnel(mat_a: i32, mat_b: i32, blend_factor: f32) -> f32 {
    let fresnel_a = get_node_fresnel(mat_a);
    if mat_b < 0 || blend_factor <= 0.0 {
        return fresnel_a;
    }
    let fresnel_b = get_node_fresnel(mat_b);
    return mix(fresnel_a, fresnel_b, blend_factor);
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
    let mat_b_id = i32(hit.z + 0.5);
    let blend_factor = hit.w;
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
    let shading_mode = camera.scene_min.w;

    // Cross-Section mode: 2D slice heatmap, no raymarching needed
    if shading_mode > 5.5 {
        let axis = i32(camera.cross_section.x + 0.5);
        let slice_pos = camera.cross_section.y;

        // Compute world position on the slice plane for this pixel.
        // Cast a ray and find where it intersects the axis-aligned slice plane.
        let plane_normal = select(select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), axis == 1), vec3f(0.0, 0.0, 1.0), axis == 2);
        let denom = dot(rd, plane_normal);

        if abs(denom) > 0.0001 {
            let plane_t = (slice_pos - dot(ro, plane_normal)) / denom;
            if plane_t > 0.0 {
                let world_pos = ro + rd * plane_t;
                let d = scene_sdf(world_pos).x;

                // Color by distance: blue (inside, d<0), red (outside, d>0), white (surface, d≈0)
                let surface_band = smoothstep(0.0, 0.02, abs(d));
                let inside_color = vec3f(0.1, 0.2, 0.8);  // blue
                let outside_color = vec3f(0.8, 0.15, 0.1); // red
                let surface_color = vec3f(1.0, 1.0, 1.0);  // white

                // Smooth gradient with distance-based intensity
                let intensity = exp(-abs(d) * 2.0);
                let base = select(inside_color, outside_color, d > 0.0);
                // Blend toward darker as distance increases, white at surface
                let dist_color = mix(base * 0.3, base, intensity);
                color = mix(surface_color, dist_color, surface_band);

                // Contour lines at regular intervals for readability
                let contour_spacing = 0.5;
                let contour = 1.0 - smoothstep(0.0, 0.015, abs(fract(d / contour_spacing + 0.5) - 0.5) * contour_spacing);
                color = mix(color, vec3f(0.0), contour * 0.3);
            }
        }
        return vec4f(color, 1.0);
    }

    // Step Heatmap applies to ALL pixels (including misses) to show full cost
    if shading_mode > 4.5 {
        let ratio = debug_step_count;
        let r = smoothstep(0.3, 0.7, ratio);
        let g = 1.0 - smoothstep(0.5, 1.0, ratio);
        color = vec3f(r, g, 0.0);
    }

    if !sdf_miss && !(shading_mode > 4.5) {
        let p = ro + rd * t;
        let n = calc_normal(p, t);

        if shading_mode > 3.5 {
            // Matcap: procedural ceramic sphere look from view-space normals
            // Compute camera basis from rd
            let cam_fwd = normalize(-rd);
            let world_up = vec3f(0.0, 1.0, 0.0);
            let cam_right = normalize(cross(world_up, cam_fwd));
            let cam_up = cross(cam_fwd, cam_right);
            let vn = vec3f(dot(n, cam_right), dot(n, cam_up), dot(n, cam_fwd));
            // Procedural matcap: warm clay with rim darkening + specular highlight
            let ndotv = max(dot(n, -rd), 0.0);
            let rim = 1.0 - ndotv;
            let rim2 = rim * rim;
            let base_col = vec3f(0.85, 0.78, 0.72);
            let rim_col = vec3f(0.35, 0.25, 0.2);
            let highlight = pow(max(vn.z, 0.0), 32.0) * 0.6;
            color = mix(base_col, rim_col, rim2) + vec3f(highlight);
            color = pow(color, vec3f(1.0 / /*GAMMA*/));
        } else if shading_mode > 2.5 {
            // Normals: map normal XYZ from [-1,1] to [0,1] as RGB
            color = n * 0.5 + 0.5;
        } else if shading_mode > 1.5 {
            // Clay: uniform gray, hemisphere + key diffuse only
            let clay = vec3f(0.7, 0.7, 0.72);
            let sky_l = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
            let key_dir = normalize(camera.key_light.xyz);
            let key_diff = max(dot(n, key_dir), 0.0);
            color = clay * (sky_l * 0.4 + key_diff * 0.6 + 0.15);
            color = pow(color, vec3f(1.0 / /*GAMMA*/));
        } else if shading_mode > 0.5 {
            // Solid: per-node colors, flat diffuse, no shadows/AO
            let albedo = get_blended_color(mat_id, mat_b_id, blend_factor);
            let key_dir = normalize(camera.key_light.xyz);
            let key_diff = max(dot(n, key_dir), 0.0);
            let fill_dir = normalize(camera.fill_light.xyz);
            let fill_diff = max(dot(n, fill_dir), 0.0) * camera.fill_light.w;
            color = albedo * (key_diff * 0.7 + fill_diff + 0.2);
            color = pow(color, vec3f(1.0 / /*GAMMA*/));
        } else {
        // Full: existing PBR pipeline

        // Material properties (blended at smooth CSG boundaries)
        let mat_metallic = get_blended_metallic(mat_id, mat_b_id, blend_factor);
        let mat_roughness = get_blended_roughness(mat_id, mat_b_id, blend_factor);
        let mat_fresnel = get_blended_fresnel(mat_id, mat_b_id, blend_factor);

        let albedo = get_blended_color(mat_id, mat_b_id, blend_factor);
        let view_dir = -rd;
        let NoV = max(dot(n, view_dir), 0.0001);

        // Dielectric F0 from fresnel parameter; metallic F0 = albedo
        let f0 = mix(vec3f(mat_fresnel), albedo, mat_metallic);
        // Clamp roughness to avoid division artifacts at perfectly smooth
        let roughness = clamp(mat_roughness, 0.045, 1.0);

        // Lambertian diffuse (energy-conserving: metals have no diffuse)
        let diffuse_color = albedo * (1.0 - mat_metallic) / PI;

        // Key light — Cook-Torrance GGX BRDF
        let key_dir = normalize(camera.key_light.xyz);
        let key_h = normalize(key_dir + view_dir);
        let NoL_key = max(dot(n, key_dir), 0.0);
        let NoH_key = max(dot(n, key_h), 0.0);
        let VoH_key = max(dot(view_dir, key_h), 0.0);

        let D_key = D_GGX(NoH_key, roughness);
        let G_key = G_SmithGGX(NoV, NoL_key, roughness);
        let F_key = F_Schlick_vec3(VoH_key, f0);
        let specular_key = D_key * G_key * F_key;

        /*SHADOW_LINE*/

        // Fill light — same BRDF
        let fill_dir = normalize(camera.fill_light.xyz);
        let fill_h = normalize(fill_dir + view_dir);
        let NoL_fill = max(dot(n, fill_dir), 0.0);
        let NoH_fill = max(dot(n, fill_h), 0.0);
        let VoH_fill = max(dot(view_dir, fill_h), 0.0);

        let D_fill = D_GGX(NoH_fill, roughness);
        let G_fill = G_SmithGGX(NoV, NoL_fill, roughness);
        let F_fill = F_Schlick_vec3(VoH_fill, f0);
        let specular_fill = D_fill * G_fill * F_fill;

        // Ambient occlusion
        /*AO_LINE*/

        // Hemispherical sky light (iq outdoor lighting)
        let sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        let shadow_col = pow(vec3f(shadow), vec3f(1.0, 1.2, 1.5));

        // Accumulate lighting — parameters from uniform buffer (no shader rebuild needed)
        let key_diffuse_i = camera.key_light.w;
        let key_spec_i = camera.key_color_spec.w;
        let key_col = camera.key_color_spec.xyz;
        let fill_i = camera.fill_light.w;
        let fill_col = camera.fill_color_ambient.xyz;
        let ambient_i = camera.fill_color_ambient.w;

        let key_contribution = (diffuse_color + specular_key * key_spec_i) * NoL_key * key_diffuse_i * key_col;
        let fill_contribution = (diffuse_color + specular_fill * key_spec_i) * NoL_fill * fill_i * fill_col;
        color = key_contribution * shadow_col
              + fill_contribution * ao
              + diffuse_color * sky * ambient_i * ao;

        // --- Scene lights (up to 8 positioned lights from Light nodes) ---
        let scene_light_count = i32(camera.scene_light_info.x + 0.5);
        for (var li = 0; li < 8; li++) {
            if li >= scene_light_count { break; }
            let base = li * 4;
            let sl_pos_type = camera.scene_lights[base];
            let sl_dir_int = camera.scene_lights[base + 1];
            let sl_col_range = camera.scene_lights[base + 2];
            let sl_params = camera.scene_lights[base + 3];

            let sl_type = i32(sl_pos_type.w + 0.5); // 0=point, 1=spot, 2=directional
            let sl_intensity = sl_dir_int.w;
            let sl_color = sl_col_range.xyz;
            let sl_range = sl_col_range.w;

            // Compute light direction and attenuation
            var sl_light_dir: vec3f;
            var sl_atten = 1.0;
            if sl_type == 2 {
                // Directional: constant direction, no attenuation
                sl_light_dir = normalize(-sl_dir_int.xyz);
            } else {
                // Point/Spot: direction from surface to light
                let to_light = sl_pos_type.xyz - p;
                let dist = length(to_light);
                sl_light_dir = to_light / max(dist, 0.0001);
                // Inverse square falloff with range-based windowing
                let dist_ratio = dist / max(sl_range, 0.001);
                let window = max(1.0 - dist_ratio * dist_ratio, 0.0);
                sl_atten = (window * window) / max(dist * dist, 0.01);
            }

            // Spot cone falloff
            if sl_type == 1 {
                let cos_half_angle = sl_params.x;
                let cos_theta = dot(-sl_light_dir, normalize(sl_dir_int.xyz));
                // Smooth falloff from inner to outer cone
                let inner_cos = mix(1.0, cos_half_angle, 0.8); // inner cone at 80% of angle
                sl_atten *= clamp((cos_theta - cos_half_angle) / max(inner_cos - cos_half_angle, 0.0001), 0.0, 1.0);
            }

            // Cook-Torrance BRDF for this light
            let sl_h = normalize(sl_light_dir + view_dir);
            let sl_NoL = max(dot(n, sl_light_dir), 0.0);
            let sl_NoH = max(dot(n, sl_h), 0.0);
            let sl_VoH = max(dot(view_dir, sl_h), 0.0);
            let sl_D = D_GGX(sl_NoH, roughness);
            let sl_G = G_SmithGGX(NoV, sl_NoL, roughness);
            let sl_F = F_Schlick_vec3(sl_VoH, f0);
            let sl_specular = sl_D * sl_G * sl_F;

            color += (diffuse_color + sl_specular) * sl_NoL * sl_intensity * sl_atten * sl_color;
        }

        // Environment reflection (sky gradient sampled along reflected ray)
        /*ENV_REFL_LINE*/

        // Emissive contribution (added before tonemapping for natural overbright bloom)
        let emissive_col = get_node_emissive(mat_id);
        let emissive_int = get_node_emissive_intensity(mat_id);
        color += emissive_col * emissive_int;

        // Subsurface scattering (thickness-based approximation)
        /*SSS_LINE*/

        /*FOG_LINE*/

        /*TONEMAP_LINE*/
        color = pow(color, vec3f(1.0 / /*GAMMA*/));
        } // end Full shading
    }

    // --- 3D brush preview: ring outline on surface ---
    if camera.brush_pos.w > 0.0 && !sdf_miss {
        let p = ro + rd * t;
        let brush_dist = length(p - camera.brush_pos.xyz) - camera.brush_pos.w;
        let ring_thickness = max(0.002 * t, 0.003);
        let ring = 1.0 - smoothstep(0.0, ring_thickness * 2.0, abs(brush_dist));
        color = mix(color, vec3f(0.3, 0.8, 0.3), ring * 0.6);
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
