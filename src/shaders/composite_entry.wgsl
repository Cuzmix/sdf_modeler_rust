struct CompositeParams {
    bounds_min: vec4f,
    bounds_max: vec4f,
    resolution: u32,
    update_min_x: u32,
    update_min_y: u32,
    update_min_z: u32,
    update_max_x: u32,
    update_max_y: u32,
    update_max_z: u32,
    _pad: u32,
}

@group(3) @binding(0) var<uniform> comp_params: CompositeParams;
@group(3) @binding(1) var comp_sdf_out: texture_storage_3d<r32float, write>;
@group(3) @binding(2) var comp_mat_out: texture_storage_3d<r32uint, write>;
@group(3) @binding(3) var comp_normal_out: texture_storage_3d<rgba8snorm, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let update_min = vec3u(comp_params.update_min_x, comp_params.update_min_y, comp_params.update_min_z);
    let update_max = vec3u(comp_params.update_max_x, comp_params.update_max_y, comp_params.update_max_z);
    let voxel = update_min + gid;
    let res = comp_params.resolution;
    if any(voxel >= vec3u(res)) || any(voxel > update_max) { return; }

    let norm = vec3f(voxel) / f32(res - 1u);
    let world_pos = comp_params.bounds_min.xyz + norm * (comp_params.bounds_max.xyz - comp_params.bounds_min.xyz);

    let hit = scene_sdf(world_pos);

    textureStore(comp_sdf_out, voxel, vec4f(hit.x, 0.0, 0.0, 0.0));
    textureStore(comp_mat_out, voxel, vec4u(u32(max(hit.y + 1.0, 0.0)), 0u, 0u, 0u));

    // Precompute normal using tetrahedron technique (4 SDF evals)
    let e = 0.002;
    let k = vec2f(1.0, -1.0);
    let n = normalize(
        k.xyy * scene_sdf(world_pos + k.xyy * e).x +
        k.yyx * scene_sdf(world_pos + k.yyx * e).x +
        k.yxy * scene_sdf(world_pos + k.yxy * e).x +
        k.xxx * scene_sdf(world_pos + k.xxx * e).x
    );
    textureStore(comp_normal_out, voxel, vec4f(n, 0.0));
}
