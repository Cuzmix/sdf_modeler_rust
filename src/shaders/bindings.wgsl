// GPU uniform and storage buffer bindings for the SDF rendering pipeline.
// Camera holds view/projection data and rendering settings.
// SdfNode holds per-node SDF parameters packed into vec4f slots.

struct Camera {
    inv_view_proj: mat4x4f,
    eye: vec4f,
    viewport: vec4f,
    time: f32,
    quality_mode: f32,
    grid_enabled: f32,
    selected_idx: f32,
    scene_min: vec4f,
    scene_max: vec4f,
    brush_pos: vec4f,
    cross_section: vec4f,
    // Ambient lighting (global key/fill migrated to scene Directional nodes)
    ambient_info: vec4f,    // x = ambient_intensity, yzw = unused
    // Scene lights (up to 8, packed as 4 vec4f per light)
    scene_light_info: vec4f,   // x = count, y = volumetric_count, z = volumetric_steps, w = unused
    scene_lights: array<vec4f, 32>, // 8 lights × 4 vec4f
    // Per-light volumetric params (1 vec4f per light)
    scene_light_vol: array<vec4f, 8>, // [volumetric_on, density, has_cookie, cookie_idx]
}

struct SdfNode {
    type_op: vec4f,
    position: vec4f,
    rotation: vec4f,
    scale: vec4f,
    color: vec4f,
    extra0: vec4f,
    extra1: vec4f,
    extra2: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> nodes: array<SdfNode>;
@group(1) @binding(1) var<storage, read> voxel_data: array<f32>;
