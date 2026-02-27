// Full-screen triangle vertex shader.
// Draws a single oversized triangle that covers the viewport (3 vertices, no vertex buffer).
// Used by the render pipeline only — not included in compute shaders.

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4f(x, y, 0.0, 1.0);
    return out;
}
