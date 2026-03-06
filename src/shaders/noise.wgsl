// ---------------------------------------------------------------------------
// Noise functions for procedural SDF effects (domain warping, displacement).
// All functions are pure, deterministic, and GPU-friendly (no divergent branching).
// ---------------------------------------------------------------------------

// Hash-based pseudo-random: maps vec3f to vec3f.
// Uses large primes and fract() for uniform distribution.
fn hash33(p: vec3f) -> vec3f {
    var q = vec3f(
        dot(p, vec3f(127.1, 311.7, 74.7)),
        dot(p, vec3f(269.5, 183.3, 246.1)),
        dot(p, vec3f(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123) * 2.0 - 1.0;
}

// Quintic interpolation curve: 6t^5 - 15t^4 + 10t^3
// Provides C2 continuity (continuous second derivative) for smooth noise.
fn quintic(t: vec3f) -> vec3f {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D gradient noise using trilinear interpolation of hashed gradients.
// Returns a scalar in approximately [-1, 1].
fn noise3d(p: vec3f) -> f32 {
    let cell = floor(p);
    let local = p - cell;

    let fade = quintic(local);

    // Hash gradients at 8 cube corners and dot with distance vectors
    let g000 = dot(hash33(cell + vec3f(0.0, 0.0, 0.0)), local - vec3f(0.0, 0.0, 0.0));
    let g100 = dot(hash33(cell + vec3f(1.0, 0.0, 0.0)), local - vec3f(1.0, 0.0, 0.0));
    let g010 = dot(hash33(cell + vec3f(0.0, 1.0, 0.0)), local - vec3f(0.0, 1.0, 0.0));
    let g110 = dot(hash33(cell + vec3f(1.0, 1.0, 0.0)), local - vec3f(1.0, 1.0, 0.0));
    let g001 = dot(hash33(cell + vec3f(0.0, 0.0, 1.0)), local - vec3f(0.0, 0.0, 1.0));
    let g101 = dot(hash33(cell + vec3f(1.0, 0.0, 1.0)), local - vec3f(1.0, 0.0, 1.0));
    let g011 = dot(hash33(cell + vec3f(0.0, 1.0, 1.0)), local - vec3f(0.0, 1.0, 1.0));
    let g111 = dot(hash33(cell + vec3f(1.0, 1.0, 1.0)), local - vec3f(1.0, 1.0, 1.0));

    // Trilinear interpolation with quintic fade
    let mix_x0 = mix(g000, g100, fade.x);
    let mix_x1 = mix(g010, g110, fade.x);
    let mix_x2 = mix(g001, g101, fade.x);
    let mix_x3 = mix(g011, g111, fade.x);

    let mix_y0 = mix(mix_x0, mix_x1, fade.y);
    let mix_y1 = mix(mix_x2, mix_x3, fade.y);

    return mix(mix_y0, mix_y1, fade.z);
}

// Fractal Brownian Motion returning 3D displacement vector.
// Each axis samples noise at a different offset to produce independent displacements.
// Lacunarity = 2.0, persistence = 0.5. Max 8 octaves (clamped).
fn fbm_noise(p: vec3f, frequency: f32, amplitude: f32, octaves: i32) -> vec3f {
    let max_octaves = min(octaves, 8);
    var result = vec3f(0.0);
    var freq = frequency;
    var amp = amplitude;

    // Offsets to decorrelate the three displacement axes
    let offset_y = vec3f(31.416, 67.281, 11.513);
    let offset_z = vec3f(73.156, 19.874, 53.129);

    for (var i = 0; i < max_octaves; i = i + 1) {
        let sample_pos = p * freq;
        result.x += noise3d(sample_pos) * amp;
        result.y += noise3d(sample_pos + offset_y) * amp;
        result.z += noise3d(sample_pos + offset_z) * amp;

        freq *= 2.0;
        amp *= 0.5;
    }

    return result;
}
