struct BlitParams {
    viewport: vec4f,
    outline_color: vec4f,  // (r, g, b, width_in_pixels)
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
    let scene = textureSampleLevel(blit_texture, blit_sampler, uv, 0.0);

    // Post-process selection outline via edge detection on alpha channel.
    // Alpha encodes selection: 0 = selected, 1 = not selected.
    let w = i32(params.outline_color.w);
    if w <= 0 {
        return vec4f(scene.rgb, 1.0);
    }

    let tex_size = vec2<i32>(textureDimensions(blit_texture));
    let px = vec2<i32>(uv * vec2f(tex_size));
    let center_sel = textureLoad(blit_texture, clamp(px, vec2(0), tex_size - 1), 0).a < 0.5;

    var is_edge = false;
    for (var i = 1; i <= 8; i++) {
        if i > w { break; }
        let r = textureLoad(blit_texture, clamp(px + vec2(i, 0), vec2(0), tex_size - 1), 0).a < 0.5;
        let l = textureLoad(blit_texture, clamp(px + vec2(-i, 0), vec2(0), tex_size - 1), 0).a < 0.5;
        let u = textureLoad(blit_texture, clamp(px + vec2(0, -i), vec2(0), tex_size - 1), 0).a < 0.5;
        let d = textureLoad(blit_texture, clamp(px + vec2(0, i), vec2(0), tex_size - 1), 0).a < 0.5;
        if r != center_sel || l != center_sel || u != center_sel || d != center_sel {
            is_edge = true;
            break;
        }
    }

    let outline = select(0.0, 1.0, is_edge);
    let final_color = mix(scene.rgb, params.outline_color.rgb, outline);
    return vec4f(final_color, 1.0);
}
