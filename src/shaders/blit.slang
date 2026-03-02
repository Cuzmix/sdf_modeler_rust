// Blit shader — Slang version of blit.wgsl
// Compiles to WGSL via the Slang compiler (feature = "slang").

struct BlitParams {
    float4 viewport;
    float4 outline_color;  // (r, g, b, width_in_pixels)
    float4 bloom_params;   // (threshold, intensity, radius, enabled)
};

[[vk::binding(0, 0)]] ConstantBuffer<BlitParams> params;
[[vk::binding(1, 0)]] SamplerState blit_sampler;
[[vk::binding(2, 0)]] Texture2D<float4> blit_texture;

[shader("vertex")]
float4 vs_blit(uint vi : SV_VertexID) : SV_Position {
    float x = float(int(vi & 1u)) * 4.0 - 1.0;
    float y = float(int(vi >> 1u)) * 4.0 - 1.0;
    return float4(x, y, 0.0, 1.0);
}

[shader("fragment")]
float4 fs_blit(float4 frag_coord : SV_Position) : SV_Target {
    float2 uv = (frag_coord.xy - params.viewport.xy) / params.viewport.zw;
    float4 scene = blit_texture.SampleLevel(blit_sampler, uv, 0.0);
    float3 color = scene.rgb;

    // --- Bloom: star-pattern multi-tap with brightness threshold ---
    if (params.bloom_params.w > 0.5) {
        float threshold = params.bloom_params.x;
        float intensity = params.bloom_params.y;
        float radius = params.bloom_params.z;

        uint tw, th;
        blit_texture.GetDimensions(tw, th);
        float2 tex_size = float2(tw, th);
        float2 inv_ts = 1.0 / tex_size;

        float3 bloom = float3(0.0, 0.0, 0.0);
        float2 d0 = float2(1.0, 0.0);
        float2 d1 = float2(0.0, 1.0);
        float2 d2 = float2(0.707, 0.707);
        float2 d3 = float2(0.707, -0.707);

        for (int i = 1; i <= 6; i++) {
            float w = 1.0 / float(i);
            float r = float(i) * radius;
            float3 s0p = blit_texture.SampleLevel(blit_sampler, uv + d0 * r * inv_ts, 0.0).rgb;
            float3 s0n = blit_texture.SampleLevel(blit_sampler, uv - d0 * r * inv_ts, 0.0).rgb;
            float3 s1p = blit_texture.SampleLevel(blit_sampler, uv + d1 * r * inv_ts, 0.0).rgb;
            float3 s1n = blit_texture.SampleLevel(blit_sampler, uv - d1 * r * inv_ts, 0.0).rgb;
            float3 s2p = blit_texture.SampleLevel(blit_sampler, uv + d2 * r * inv_ts, 0.0).rgb;
            float3 s2n = blit_texture.SampleLevel(blit_sampler, uv - d2 * r * inv_ts, 0.0).rgb;
            float3 s3p = blit_texture.SampleLevel(blit_sampler, uv + d3 * r * inv_ts, 0.0).rgb;
            float3 s3n = blit_texture.SampleLevel(blit_sampler, uv - d3 * r * inv_ts, 0.0).rgb;
            bloom += max(float3(0.0, 0.0, 0.0), s0p - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s0n - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s1p - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s1n - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s2p - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s2n - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s3p - threshold) * w;
            bloom += max(float3(0.0, 0.0, 0.0), s3n - threshold) * w;
        }
        color += bloom * intensity / 16.0;
    }

    // Post-process selection outline via edge detection on alpha channel.
    int outline_w = int(params.outline_color.w);
    if (outline_w <= 0) {
        return float4(color, 1.0);
    }

    uint tex_w, tex_h;
    blit_texture.GetDimensions(tex_w, tex_h);
    int2 tex_size_i = int2(tex_w, tex_h);
    int2 px = int2(uv * float2(tex_size_i));
    bool center_sel = blit_texture.Load(int3(clamp(px, int2(0, 0), tex_size_i - 1), 0)).a < 0.5;

    bool is_edge = false;
    for (int i = 1; i <= 8; i++) {
        if (i > outline_w) break;
        bool r_sel = blit_texture.Load(int3(clamp(px + int2(i, 0), int2(0, 0), tex_size_i - 1), 0)).a < 0.5;
        bool l_sel = blit_texture.Load(int3(clamp(px + int2(-i, 0), int2(0, 0), tex_size_i - 1), 0)).a < 0.5;
        bool u_sel = blit_texture.Load(int3(clamp(px + int2(0, -i), int2(0, 0), tex_size_i - 1), 0)).a < 0.5;
        bool d_sel = blit_texture.Load(int3(clamp(px + int2(0, i), int2(0, 0), tex_size_i - 1), 0)).a < 0.5;
        if (r_sel != center_sel || l_sel != center_sel || u_sel != center_sel || d_sel != center_sel) {
            is_edge = true;
            break;
        }
    }

    float outline = is_edge ? 1.0 : 0.0;
    float3 final_color = lerp(color, params.outline_color.rgb, outline);
    return float4(final_color, 1.0);
}
