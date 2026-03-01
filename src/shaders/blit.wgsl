struct BlitParams {
    viewport: vec4f,
    outline_color: vec4f,  // (r, g, b, width_in_pixels)
    bloom_params: vec4f,   // (threshold, intensity, radius, enabled)
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
    var color = scene.rgb;

    // --- Bloom: star-pattern multi-tap with brightness threshold ---
    if params.bloom_params.w > 0.5 {
        let threshold = params.bloom_params.x;
        let intensity = params.bloom_params.y;
        let radius = params.bloom_params.z;
        let tex_size = vec2f(textureDimensions(blit_texture));
        let inv_ts = 1.0 / tex_size;

        var bloom = vec3f(0.0);
        // 4 directions: horizontal, vertical, diagonal
        let d0 = vec2f(1.0, 0.0);
        let d1 = vec2f(0.0, 1.0);
        let d2 = vec2f(0.707, 0.707);
        let d3 = vec2f(0.707, -0.707);

        for (var i = 1; i <= 6; i++) {
            let w = 1.0 / f32(i);
            let r = f32(i) * radius;
            // Sample in 4 directions (positive + negative = 8 taps per step)
            let s0p = textureSampleLevel(blit_texture, blit_sampler, uv + d0 * r * inv_ts, 0.0).rgb;
            let s0n = textureSampleLevel(blit_texture, blit_sampler, uv - d0 * r * inv_ts, 0.0).rgb;
            let s1p = textureSampleLevel(blit_texture, blit_sampler, uv + d1 * r * inv_ts, 0.0).rgb;
            let s1n = textureSampleLevel(blit_texture, blit_sampler, uv - d1 * r * inv_ts, 0.0).rgb;
            let s2p = textureSampleLevel(blit_texture, blit_sampler, uv + d2 * r * inv_ts, 0.0).rgb;
            let s2n = textureSampleLevel(blit_texture, blit_sampler, uv - d2 * r * inv_ts, 0.0).rgb;
            let s3p = textureSampleLevel(blit_texture, blit_sampler, uv + d3 * r * inv_ts, 0.0).rgb;
            let s3n = textureSampleLevel(blit_texture, blit_sampler, uv - d3 * r * inv_ts, 0.0).rgb;
            bloom += max(vec3f(0.0), s0p - threshold) * w;
            bloom += max(vec3f(0.0), s0n - threshold) * w;
            bloom += max(vec3f(0.0), s1p - threshold) * w;
            bloom += max(vec3f(0.0), s1n - threshold) * w;
            bloom += max(vec3f(0.0), s2p - threshold) * w;
            bloom += max(vec3f(0.0), s2n - threshold) * w;
            bloom += max(vec3f(0.0), s3p - threshold) * w;
            bloom += max(vec3f(0.0), s3n - threshold) * w;
        }
        color += bloom * intensity / 16.0;
    }

    // Post-process selection outline via edge detection on alpha channel.
    // Alpha encodes selection: 0 = selected, 1 = not selected.
    let w = i32(params.outline_color.w);
    if w <= 0 {
        return vec4f(color, 1.0);
    }

    let tex_size_i = vec2<i32>(textureDimensions(blit_texture));
    let px = vec2<i32>(uv * vec2f(tex_size_i));
    let center_sel = textureLoad(blit_texture, clamp(px, vec2(0), tex_size_i - 1), 0).a < 0.5;

    var is_edge = false;
    for (var i = 1; i <= 8; i++) {
        if i > w { break; }
        let r = textureLoad(blit_texture, clamp(px + vec2(i, 0), vec2(0), tex_size_i - 1), 0).a < 0.5;
        let l = textureLoad(blit_texture, clamp(px + vec2(-i, 0), vec2(0), tex_size_i - 1), 0).a < 0.5;
        let u = textureLoad(blit_texture, clamp(px + vec2(0, -i), vec2(0), tex_size_i - 1), 0).a < 0.5;
        let d = textureLoad(blit_texture, clamp(px + vec2(0, i), vec2(0), tex_size_i - 1), 0).a < 0.5;
        if r != center_sel || l != center_sel || u != center_sel || d != center_sel {
            is_edge = true;
            break;
        }
    }

    let outline = select(0.0, 1.0, is_edge);
    let final_color = mix(color, params.outline_color.rgb, outline);
    return vec4f(final_color, 1.0);
}
