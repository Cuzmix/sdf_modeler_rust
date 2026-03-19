const PI: f32 = 3.14159265359;
const IRRADIANCE_SAMPLE_COUNT: u32 = 64u;
const PREFILTER_SAMPLE_COUNT: u32 = 64u;
const BRDF_SAMPLE_COUNT: u32 = 128u;

struct EnvironmentBakeUniform {
    face_and_flags: vec4<u32>,
    params: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_zenith: vec4<f32>,
    solid_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> environment_bake: EnvironmentBakeUniform;
@group(1) @binding(0) var environment_source_sampler: sampler;
@group(1) @binding(1) var environment_source_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_environment_bake(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn face_direction(face: u32, frag_coord: vec2<f32>, resolution: f32) -> vec3<f32> {
    let uv = (2.0 * (frag_coord / resolution)) - vec2<f32>(1.0, 1.0);
    let u = uv.x;
    let v = uv.y;

    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -v, -u)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -v, u)); }
        case 2u: { return normalize(vec3<f32>(u, 1.0, v)); }
        case 3u: { return normalize(vec3<f32>(u, -1.0, -v)); }
        case 4u: { return normalize(vec3<f32>(u, -v, 1.0)); }
        default: { return normalize(vec3<f32>(-u, -v, -1.0)); }
    }
}

fn rotate_environment_direction(direction: vec3<f32>, rotation_radians: f32) -> vec3<f32> {
    let s = sin(rotation_radians);
    let c = cos(rotation_radians);
    return vec3<f32>(
        c * direction.x + s * direction.z,
        direction.y,
        -s * direction.x + c * direction.z,
    );
}

fn sample_procedural_environment(direction: vec3<f32>) -> vec3<f32> {
    let background_mode = environment_bake.face_and_flags.w;
    if background_mode == 1u {
        return environment_bake.solid_color.xyz;
    }

    let t = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(environment_bake.sky_horizon.xyz, environment_bake.sky_zenith.xyz, t);
}

fn direction_to_equirect_uv(direction: vec3<f32>) -> vec2<f32> {
    let dir = normalize(direction);
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = fract(phi / (2.0 * PI) + 0.5);
    let v = clamp(theta / PI, 0.0, 1.0);
    return vec2<f32>(u, v);
}

fn sample_environment(direction: vec3<f32>) -> vec3<f32> {
    let rotated = rotate_environment_direction(direction, environment_bake.params.y);
    let source_mode = environment_bake.face_and_flags.y;
    let hdri_available = environment_bake.face_and_flags.z;
    var base_color = sample_procedural_environment(rotated);
    if source_mode == 1u && hdri_available == 1u {
        let uv = direction_to_equirect_uv(rotated);
        base_color = textureSampleLevel(environment_source_texture, environment_source_sampler, uv, 0.0).xyz;
    }
    return base_color * environment_bake.params.z;
}

fn tangent_frame(normal: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn radical_inverse_vdc(bits_value: u32) -> f32 {
    var bits = bits_value;
    bits = (bits >> 16u) | (bits << 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), radical_inverse_vdc(index));
}

fn cosine_sample_hemisphere(xi: vec2<f32>) -> vec3<f32> {
    let r = sqrt(xi.x);
    let phi = 2.0 * PI * xi.y;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - xi.x));
    return vec3<f32>(x, y, z);
}

fn importance_sample_ggx(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let a2 = a * a;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let half_vector_tangent =
        vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let frame = tangent_frame(normal);
    return normalize(frame * half_vector_tangent);
}

fn geometry_schlick_ggx(no_x: f32, roughness: f32) -> f32 {
    let k = roughness * roughness * 0.5;
    return no_x / max(no_x * (1.0 - k) + k, 0.0001);
}

fn geometry_smith_ibl(no_v: f32, no_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(no_v, roughness) * geometry_schlick_ggx(no_l, roughness);
}

fn integrate_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let frame = tangent_frame(normal);
    var irradiance = vec3<f32>(0.0, 0.0, 0.0);
    var weight = 0.0;

    for (var sample_index = 0u; sample_index < IRRADIANCE_SAMPLE_COUNT; sample_index = sample_index + 1u) {
        let xi = hammersley(sample_index, IRRADIANCE_SAMPLE_COUNT);
        let hemisphere = cosine_sample_hemisphere(xi);
        let sample_dir = normalize(frame * hemisphere);
        let no_l = max(dot(normal, sample_dir), 0.0);
        if no_l <= 0.0 {
            continue;
        }
        irradiance = irradiance + sample_environment(sample_dir) * no_l;
        weight = weight + no_l;
    }

    if weight > 0.0 {
        return irradiance / weight;
    }
    return sample_environment(normal);
}

fn prefilter_environment(reflection_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let normal = normalize(reflection_dir);
    let view_dir = normal;
    var prefiltered = vec3<f32>(0.0, 0.0, 0.0);
    var weight = 0.0;

    for (var sample_index = 0u; sample_index < PREFILTER_SAMPLE_COUNT; sample_index = sample_index + 1u) {
        let xi = hammersley(sample_index, PREFILTER_SAMPLE_COUNT);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        let light_dir = normalize(2.0 * dot(view_dir, half_vector) * half_vector - view_dir);
        let no_l = max(dot(normal, light_dir), 0.0);
        if no_l <= 0.0 {
            continue;
        }
        prefiltered = prefiltered + sample_environment(light_dir) * no_l;
        weight = weight + no_l;
    }

    if weight > 0.0 {
        return prefiltered / weight;
    }
    return sample_environment(reflection_dir);
}

fn integrate_brdf(no_v: f32, roughness: f32) -> vec2<f32> {
    let view_dir = vec3<f32>(sqrt(max(0.0, 1.0 - no_v * no_v)), 0.0, no_v);
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    var scale = 0.0;
    var bias = 0.0;

    for (var sample_index = 0u; sample_index < BRDF_SAMPLE_COUNT; sample_index = sample_index + 1u) {
        let xi = hammersley(sample_index, BRDF_SAMPLE_COUNT);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        let light_dir = normalize(2.0 * dot(view_dir, half_vector) * half_vector - view_dir);
        let no_l = max(light_dir.z, 0.0);
        let no_h = max(half_vector.z, 0.0);
        let vo_h = max(dot(view_dir, half_vector), 0.0);
        if no_l <= 0.0 {
            continue;
        }

        let geometry = geometry_smith_ibl(no_v, no_l, roughness);
        let visibility = (geometry * vo_h) / max(no_h * no_v, 0.0001);
        let fresnel = pow(1.0 - vo_h, 5.0);
        scale = scale + (1.0 - fresnel) * visibility;
        bias = bias + fresnel * visibility;
    }

    return vec2<f32>(
        scale / f32(BRDF_SAMPLE_COUNT),
        bias / f32(BRDF_SAMPLE_COUNT),
    );
}

@fragment
fn fs_source_cube(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let direction = face_direction(environment_bake.face_and_flags.x, frag_coord.xy, environment_bake.params.w);
    return vec4<f32>(sample_environment(direction), 1.0);
}

@fragment
fn fs_irradiance_cube(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let normal = face_direction(environment_bake.face_and_flags.x, frag_coord.xy, environment_bake.params.w);
    return vec4<f32>(integrate_irradiance(normal), 1.0);
}

@fragment
fn fs_prefiltered_cube(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let reflection_dir =
        face_direction(environment_bake.face_and_flags.x, frag_coord.xy, environment_bake.params.w);
    return vec4<f32>(
        prefilter_environment(reflection_dir, environment_bake.params.x),
        1.0,
    );
}

@fragment
fn fs_brdf_lut(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec2<f32> {
    let uv = frag_coord.xy / environment_bake.params.w;
    let no_v = clamp(uv.x, 0.0001, 0.9999);
    let roughness = clamp(uv.y, 0.0001, 0.9999);
    return integrate_brdf(no_v, roughness);
}
