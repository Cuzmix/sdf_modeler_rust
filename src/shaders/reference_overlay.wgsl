struct OverlayVertexInput {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
};

struct OverlayVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var overlay_sampler: sampler;
@group(0) @binding(1) var overlay_texture: texture_2d<f32>;

@vertex
fn vs_main(input: OverlayVertexInput) -> OverlayVertexOutput {
    var output: OverlayVertexOutput;
    output.position = vec4f(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: OverlayVertexOutput) -> @location(0) vec4f {
    let sampled = textureSample(overlay_texture, overlay_sampler, input.uv);
    return sampled * input.color;
}
