struct BlitParams {
    viewport: vec4f,
}

@group(0) @binding(0) var<uniform> params: BlitParams;
@group(0) @binding(1) var blit_sampler: sampler;
@group(0) @binding(2) var blit_texture: texture_2d<f32>;

@vertex fn vs_blit(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs_blit(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let uv = (frag_coord.xy - params.viewport.xy) / params.viewport.zw;
    return textureSampleLevel(blit_texture, blit_sampler, uv, 0.0);
}
