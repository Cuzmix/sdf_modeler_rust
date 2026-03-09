struct BrushParams {
    center_local: vec3f,
    radius: f32,
    strength: f32,
    sign_val: f32,
    grid_offset: u32,
    grid_resolution: u32,
    bounds_min: vec3f,
    _pad0: f32,
    bounds_max: vec3f,
    _pad1: f32,
    min_voxel: vec3<u32>,
    _pad2: u32,
    brush_mode: f32,
    falloff_mode: f32,
    smooth_iterations: u32,
    flatten_ref: f32,
    surface_constraint: f32,
    _pad3x: f32,
    _pad3y: f32,
    _pad3z: f32,
    view_dir_local: vec3f,
    _pad4: f32,
}

@group(0) @binding(0) var<uniform> brush: BrushParams;
@group(0) @binding(1) var<storage, read_write> voxel_data: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn cs_brush(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_pos = brush.min_voxel + gid;
    let res = brush.grid_resolution;
    if any(grid_pos >= vec3<u32>(res)) { return; }

    let res_f = f32(res - 1u);
    let norm = vec3f(grid_pos) / res_f;
    let world_pos = brush.bounds_min + norm * (brush.bounds_max - brush.bounds_min);
    let offset = world_pos - brush.center_local;
    let dist = length(offset);

    if dist >= brush.radius { return; }

    // Branchless falloff selection (0=Smooth, 1=Linear, 2=Sharp, 3=Flat)
    let nt = dist / brush.radius;
    let fm = brush.falloff_mode;
    let isLinear = step(0.5, fm) * (1.0 - step(1.5, fm));
    let isSharp  = step(1.5, fm) * (1.0 - step(2.5, fm));
    let isFlat   = step(2.5, fm);
    let isSmooth = 1.0 - isLinear - isSharp - isFlat;

    // Front-face attenuation (disabled when view_dir_local is zero).
    let view_len2 = dot(brush.view_dir_local, brush.view_dir_local);
    let has_front_face = view_len2 > 1e-6 && dist > 1e-6;
    let offset_dir = offset * (1.0 / max(dist, 1e-6));
    let view_dir = brush.view_dir_local * inverseSqrt(max(view_len2, 1e-6));
    let ndotv = dot(offset_dir, view_dir);
    let hemi = clamp(0.5 - 0.5 * ndotv, 0.0, 1.0);
    let front_face = select(1.0, hemi * hemi, has_front_face);

    let falloff = ((1.0 - nt * nt * (3.0 - 2.0 * nt)) * isSmooth
                + (1.0 - nt) * isLinear
                + (1.0 - nt) * (1.0 - nt) * isSharp
                + 1.0 * isFlat) * front_face;

    let idx = brush.grid_offset + grid_pos.z * res * res + grid_pos.y * res + grid_pos.x;
    let cur = voxel_data[idx];

    // Surface constraint: attenuate brush near surface only
    let sf = select(1.0,
        1.0 - clamp(abs(cur) / (brush.radius * brush.surface_constraint), 0.0, 1.0),
        brush.surface_constraint > 0.0);

    // Branchless mode selection (0=Add, 1=Carve, 3=Flatten, 4=Inflate; 2=Smooth never dispatched)
    let bm = brush.brush_mode;
    let isAdd     = 1.0 - step(0.5, bm);
    let isCarve   = step(0.5, bm) * (1.0 - step(1.5, bm));
    let isFlatten = step(2.5, bm) * (1.0 - step(3.5, bm));
    let isInflate = step(3.5, bm) * (1.0 - step(4.5, bm));

    let add_delta     = -1.0 * brush.strength * falloff * sf;
    let carve_delta   =  1.0 * brush.strength * falloff * sf;
    let flatten_delta = (brush.flatten_ref - cur) * falloff * brush.strength * sf;
    // Inflate: Add with implicit surface constraint (only near-surface voxels)
    let inflate_sf = 1.0 - clamp(abs(cur) / (brush.radius * 0.5), 0.0, 1.0);
    let inflate_delta = -1.0 * brush.strength * falloff * inflate_sf;

    voxel_data[idx] += add_delta * isAdd + carve_delta * isCarve + flatten_delta * isFlatten + inflate_delta * isInflate;
}

