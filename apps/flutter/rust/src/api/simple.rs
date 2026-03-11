#[flutter_rust_bridge::frb(sync)]
pub fn ping() -> String {
    "pong".to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn bridge_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn render_preview_frame(width: u32, height: u32, time_seconds: f32) -> Vec<u8> {
    let safe_width = width.max(1).min(1024);
    let safe_height = height.max(1).min(1024);
    let mut pixels = vec![0_u8; (safe_width * safe_height * 4) as usize];

    let aspect = safe_width as f32 / safe_height as f32;
    let camera_radius = 2.6_f32;
    let camera_yaw = time_seconds * 0.6_f32;
    let camera_position = [
        camera_radius * camera_yaw.cos(),
        0.7_f32,
        camera_radius * camera_yaw.sin(),
    ];
    let target = [0.0_f32, 0.0_f32, 0.0_f32];

    let forward = normalize(sub(target, camera_position));
    let world_up = [0.0_f32, 1.0_f32, 0.0_f32];
    let right = normalize(cross(forward, world_up));
    let up = normalize(cross(right, forward));

    for y in 0..safe_height {
        for x in 0..safe_width {
            let pixel_index = ((y * safe_width + x) * 4) as usize;

            let ndc_x = (2.0_f32 * ((x as f32 + 0.5_f32) / safe_width as f32) - 1.0_f32) * aspect;
            let ndc_y = 1.0_f32 - 2.0_f32 * ((y as f32 + 0.5_f32) / safe_height as f32);

            let ray_direction = normalize(add(
                forward,
                add(scale(right, ndc_x * 0.9_f32), scale(up, ndc_y * 0.9_f32)),
            ));

            let color = shade_scene(camera_position, ray_direction, time_seconds);

            pixels[pixel_index] = (color[0].clamp(0.0_f32, 1.0_f32) * 255.0_f32) as u8;
            pixels[pixel_index + 1] = (color[1].clamp(0.0_f32, 1.0_f32) * 255.0_f32) as u8;
            pixels[pixel_index + 2] = (color[2].clamp(0.0_f32, 1.0_f32) * 255.0_f32) as u8;
            pixels[pixel_index + 3] = 255;
        }
    }

    pixels
}

fn shade_scene(ray_origin: [f32; 3], ray_direction: [f32; 3], time_seconds: f32) -> [f32; 3] {
    const MAX_STEPS: usize = 72;
    const MAX_DISTANCE: f32 = 20.0_f32;
    const HIT_EPSILON: f32 = 0.0015_f32;

    let mut traveled = 0.0_f32;
    for _ in 0..MAX_STEPS {
        let sample_position = add(ray_origin, scale(ray_direction, traveled));
        let scene_distance = sdf_scene(sample_position, time_seconds);

        if scene_distance < HIT_EPSILON {
            let normal = estimate_normal(sample_position, time_seconds);
            let light_direction = normalize([0.6_f32, 0.8_f32, 0.35_f32]);
            let diffuse = dot(normal, light_direction).max(0.0_f32);
            let view = scale(ray_direction, -1.0_f32);
            let half_vector = normalize(add(light_direction, view));
            let specular = dot(normal, half_vector).max(0.0_f32).powf(28.0_f32);

            let base_color = [0.12_f32, 0.62_f32, 0.90_f32];
            let ambient = 0.10_f32;
            return [
                base_color[0] * (ambient + diffuse * 0.85_f32) + specular * 0.20_f32,
                base_color[1] * (ambient + diffuse * 0.85_f32) + specular * 0.20_f32,
                base_color[2] * (ambient + diffuse * 0.85_f32) + specular * 0.20_f32,
            ];
        }

        traveled += scene_distance;
        if traveled > MAX_DISTANCE {
            break;
        }
    }

    let horizon = 0.5_f32 * (ray_direction[1] + 1.0_f32);
    [
        0.04_f32 * (1.0_f32 - horizon) + 0.10_f32 * horizon,
        0.05_f32 * (1.0_f32 - horizon) + 0.12_f32 * horizon,
        0.08_f32 * (1.0_f32 - horizon) + 0.18_f32 * horizon,
    ]
}

fn sdf_scene(point: [f32; 3], time_seconds: f32) -> f32 {
    let rotating_point = rotate_y(point, time_seconds * 0.9_f32);
    let sphere = length(point) - 0.78_f32;
    let box_shape = sdf_box(
        add(
            rotating_point,
            [
                0.24_f32 * time_seconds.sin(),
                -0.10_f32,
                0.24_f32 * time_seconds.cos(),
            ],
        ),
        [0.34_f32, 0.34_f32, 0.34_f32],
    );
    sphere.min(box_shape)
}

fn estimate_normal(point: [f32; 3], time_seconds: f32) -> [f32; 3] {
    let e = 0.002_f32;
    let dx = sdf_scene(add(point, [e, 0.0_f32, 0.0_f32]), time_seconds)
        - sdf_scene(add(point, [-e, 0.0_f32, 0.0_f32]), time_seconds);
    let dy = sdf_scene(add(point, [0.0_f32, e, 0.0_f32]), time_seconds)
        - sdf_scene(add(point, [0.0_f32, -e, 0.0_f32]), time_seconds);
    let dz = sdf_scene(add(point, [0.0_f32, 0.0_f32, e]), time_seconds)
        - sdf_scene(add(point, [0.0_f32, 0.0_f32, -e]), time_seconds);
    normalize([dx, dy, dz])
}

fn sdf_box(point: [f32; 3], half_extents: [f32; 3]) -> f32 {
    let q = [
        point[0].abs() - half_extents[0],
        point[1].abs() - half_extents[1],
        point[2].abs() - half_extents[2],
    ];

    let outside = [q[0].max(0.0_f32), q[1].max(0.0_f32), q[2].max(0.0_f32)];
    let outside_distance = length(outside);
    let inside_distance = q[0].max(q[1].max(q[2])).min(0.0_f32);
    outside_distance + inside_distance
}

fn rotate_y(point: [f32; 3], angle: f32) -> [f32; 3] {
    let c = angle.cos();
    let s = angle.sin();
    [point[0] * c + point[2] * s, point[1], -point[0] * s + point[2] * c]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v).max(1.0e-6_f32);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f32; 3], factor: f32) -> [f32; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Keep default utilities enabled for logging/error integration.
    flutter_rust_bridge::setup_default_user_utils();
}
