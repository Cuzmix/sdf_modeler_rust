// Voxel grid sampling functions for sculpted SDF nodes.
// sdf_voxel_grid: trilinear interpolation of a full SDF voxel grid (storage buffer path).
// disp_voxel_grid: displacement-only sampling for differential sculpt nodes.

fn mix_voxel_corners(
    c000: f32,
    c100: f32,
    c010: f32,
    c110: f32,
    c001: f32,
    c101: f32,
    c011: f32,
    c111: f32,
    f: vec3f,
) -> f32 {
    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    let c0  = mix(c00, c10, f.y);
    let c1  = mix(c01, c11, f.y);
    return mix(c0, c1, f.z);
}

fn sdf_voxel_grid(local_p: vec3f, node_idx: u32) -> f32 {
    let offset = u32(nodes[node_idx].extra0.x);
    let res    = u32(nodes[node_idx].extra0.y);
    let bmin   = nodes[node_idx].extra1.xyz;
    let bmax   = nodes[node_idx].extra2.xyz;

    // Distance to AABB (0 when inside the box)
    let clamped = clamp(local_p, bmin, bmax);
    let box_dist = length(local_p - clamped);

    // Trilinear interpolation at the clamped point + box_dist for continuity.
    // When inside, clamped == local_p and box_dist == 0 (same as before).
    // When outside, samples boundary voxels and adds box_dist.
    let size = bmax - bmin;
    let norm = (local_p - bmin) / size;
    let max_c = f32(res - 1u);
    let gc = clamp(norm * max_c, vec3f(0.0), vec3f(max_c));

    let i0 = vec3<u32>(vec3f(floor(gc.x), floor(gc.y), floor(gc.z)));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(res - 1u));
    let f = fract(gc);

    let r2 = res * res;
    let c000 = voxel_data[offset + i0.z * r2 + i0.y * res + i0.x];
    let c100 = voxel_data[offset + i0.z * r2 + i0.y * res + i1.x];
    let c010 = voxel_data[offset + i0.z * r2 + i1.y * res + i0.x];
    let c110 = voxel_data[offset + i0.z * r2 + i1.y * res + i1.x];
    let c001 = voxel_data[offset + i1.z * r2 + i0.y * res + i0.x];
    let c101 = voxel_data[offset + i1.z * r2 + i0.y * res + i1.x];
    let c011 = voxel_data[offset + i1.z * r2 + i1.y * res + i0.x];
    let c111 = voxel_data[offset + i1.z * r2 + i1.y * res + i1.x];

    return mix_voxel_corners(c000, c100, c010, c110, c001, c101, c011, c111, f) + box_dist;
}

// Displacement-only grid sampling (for differential SDF sculpt nodes with analytical child).
// Returns 0.0 outside the grid (neutral displacement).
fn disp_voxel_grid(local_p: vec3f, node_idx: u32) -> f32 {
    let offset = u32(nodes[node_idx].extra0.x);
    let res    = u32(nodes[node_idx].extra0.y);
    let bmin   = nodes[node_idx].extra1.xyz;
    let bmax   = nodes[node_idx].extra2.xyz;

    let norm = (local_p - bmin) / (bmax - bmin);
    // Outside grid: no displacement
    if any(norm < vec3f(0.0)) || any(norm > vec3f(1.0)) {
        return 0.0;
    }

    let max_c = f32(res - 1u);
    let gc = norm * max_c;

    let i0 = vec3<u32>(vec3f(floor(gc.x), floor(gc.y), floor(gc.z)));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(res - 1u));
    let f = fract(gc);

    let r2 = res * res;
    let c000 = voxel_data[offset + i0.z * r2 + i0.y * res + i0.x];
    let c100 = voxel_data[offset + i0.z * r2 + i0.y * res + i1.x];
    let c010 = voxel_data[offset + i0.z * r2 + i1.y * res + i0.x];
    let c110 = voxel_data[offset + i0.z * r2 + i1.y * res + i1.x];
    let c001 = voxel_data[offset + i1.z * r2 + i0.y * res + i0.x];
    let c101 = voxel_data[offset + i1.z * r2 + i0.y * res + i1.x];
    let c011 = voxel_data[offset + i1.z * r2 + i1.y * res + i0.x];
    let c111 = voxel_data[offset + i1.z * r2 + i1.y * res + i1.x];

    return mix_voxel_corners(c000, c100, c010, c110, c001, c101, c011, c111, f);
}
