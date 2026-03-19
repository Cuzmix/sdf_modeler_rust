use crate::settings::RenderConfig;

// ---------------------------------------------------------------------------
// WGSL shader modules — each include_str!() loads a focused shader file.
// See src/shaders/ for the individual files.
// ---------------------------------------------------------------------------

const BINDINGS: &str = include_str!("../shaders/bindings.wgsl");
const VOXEL_SAMPLING: &str = include_str!("../shaders/voxel_sampling.wgsl");
const VERTEX: &str = include_str!("../shaders/vertex.wgsl");
const TRANSFORMS: &str = include_str!("../shaders/transforms.wgsl");
const PRIMITIVES: &str = include_str!("../shaders/primitives.wgsl");
const MODIFIERS: &str = include_str!("../shaders/modifiers.wgsl");
const NOISE: &str = include_str!("../shaders/noise.wgsl");
const OPERATIONS: &str = include_str!("../shaders/operations.wgsl");
const RENDERING: &str = include_str!("../shaders/rendering.wgsl");
pub(crate) const PICK: &str = include_str!("../shaders/pick.wgsl");

// ---------------------------------------------------------------------------
// Shader assembly — combine modules into complete shader preludes.
// ---------------------------------------------------------------------------

/// Prelude for render (vertex+fragment) shader: all SDF library + vertex shader.
/// Vertex shader is included here but NOT in compute_prelude().
pub(crate) fn render_prelude() -> String {
    [
        BINDINGS,
        VOXEL_SAMPLING,
        VERTEX,
        TRANSFORMS,
        PRIMITIVES,
        MODIFIERS,
        NOISE,
        OPERATIONS,
    ]
    .join("\n")
}

/// Prelude for compute shaders (pick, composite): SDF library without vertex shader.
/// Compute shaders would error with an orphan @vertex entry point.
pub(crate) fn compute_prelude() -> String {
    [
        BINDINGS,
        VOXEL_SAMPLING,
        TRANSFORMS,
        PRIMITIVES,
        MODIFIERS,
        NOISE,
        OPERATIONS,
    ]
    .join("\n")
}

// ---------------------------------------------------------------------------
// Brush compute shader (standalone — no scene/camera bindings needed)
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
    if s.contains('.') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Format [f32; 3] as WGSL vec3 components: "x, y, z".
pub(crate) fn format_vec3(v: [f32; 3]) -> String {
    format!(
        "{}, {}, {}",
        format_f32(v[0]),
        format_f32(v[1]),
        format_f32(v[2])
    )
}

/// Apply the 4 raymarching placeholders shared by render and pick shaders.
pub(crate) fn apply_march_placeholders(src: &str, config: &RenderConfig) -> String {
    src.replace("/*MARCH_MAX_STEPS*/", &config.march_max_steps.to_string())
        .replace("/*MARCH_EPSILON*/", &format_f32(config.march_epsilon))
        .replace(
            "/*MARCH_STEP_MULT*/",
            &format_f32(config.march_step_multiplier),
        )
        .replace("/*MARCH_MAX_DIST*/", &format_f32(config.march_max_distance))
}

/// Build the render postlude with all placeholder replacements.
pub(crate) fn build_postlude(config: &RenderConfig) -> String {
    // Shadows are now computed per-light in the scene light loop (rendering.wgsl).
    // The SHADOW_LINE placeholder is kept empty — shadow config is passed via
    // SHADOWS_ENABLED, SHADOW_BIAS, SHADOW_MINT, SHADOW_MAXT constants.
    let shadow_line = "";

    let ao_line = if config.ao_enabled {
        "    var ao = 1.0;\n    if camera.quality_mode < 0.5 { ao = calc_ao(p, n); }".to_string()
    } else {
        "    let ao = 1.0;".to_string()
    };

    let sss_line = if config.sss_enabled {
        format!(
            "    {{ let sss_d = scene_sdf(p + primary_light_dir * 0.05).x + scene_sdf(p + primary_light_dir * 0.1).x + scene_sdf(p + primary_light_dir * 0.2).x; let sss_thick = max(0.0, sss_d); let sss_trans = exp(-sss_thick * {}) * clamp(dot(-rd, primary_light_dir), 0.0, 1.0) * 0.5; color += albedo * vec3f({}) * sss_trans; }}",
            format_f32(config.sss_strength),
            format_vec3(config.sss_color),
        )
    } else {
        "".to_string()
    };

    let fog_line = if config.fog_enabled {
        format!(
            "    {{ let fog_amt = 1.0 - exp(-t * {}); let sun_s = pow(max(dot(rd, primary_light_dir), 0.0), 8.0); let fog_c = mix(vec3f({}), vec3f(1.0, 0.9, 0.7), sun_s); color = mix(color, fog_c, fog_amt); }}",
            format_f32(config.fog_density),
            format_vec3(config.fog_color),
        )
    } else {
        "".to_string()
    };

    let sky_cutoff = config.march_max_distance - 1.0;

    apply_march_placeholders(RENDERING, config)
        .replace("/*SHADOW_STEPS*/", &config.shadow_steps.to_string())
        .replace("/*SHADOW_PENUMBRA_K*/", &format_f32(config.shadow_penumbra_k))
        .replace("/*SHADOW_BIAS*/", &format_f32(config.shadow_bias))
        .replace("/*SHADOW_MINT*/", &format_f32(config.shadow_mint))
        .replace("/*SHADOW_MAXT*/", &format_f32(config.shadow_maxt))
        .replace("/*SHADOWS_ENABLED*/", if config.shadows_enabled { "true" } else { "false" })
        .replace("/*AO_SAMPLES*/", &config.ao_samples.to_string())
        .replace("/*AO_STEP*/", &format_f32(config.ao_step))
        .replace("/*AO_DECAY*/", &format_f32(config.ao_decay))
        .replace("/*AO_INTENSITY*/", &format_f32(config.ao_intensity))
        .replace("/*SKY_CUTOFF*/", &format_f32(sky_cutoff))
        .replace("/*GAMMA*/", &format_f32(config.gamma))
        .replace("/*SHADOW_LINE*/", shadow_line)
        .replace("/*AO_LINE*/", &ao_line)
        .replace("/*SSS_LINE*/", &sss_line)
        .replace("/*FOG_LINE*/", &fog_line)
        .replace("/*OUTLINE_COLOR*/", &format_vec3(config.outline_color))
        .replace("/*OUTLINE_THICKNESS*/", &format_f32(config.outline_thickness))
        .replace("/*TONEMAP_LINE*/", if config.tonemapping_aces {
            "    { let ta = color * (2.51 * color + vec3f(0.03)); let tb = color * (2.43 * color + vec3f(0.59)) + vec3f(0.14); color = clamp(ta / tb, vec3f(0.0), vec3f(1.0)); }"
        } else {
            ""
        })
}
