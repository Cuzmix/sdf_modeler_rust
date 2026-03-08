use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

/// Stable in-process viewport image bridge for the Slint shell.
///
/// This renderer is intentionally CPU-side so we avoid unstable Slint/WGPU API
/// coupling while the migration keeps egui/wgpu as the primary viewport runtime.
pub struct EmbeddedViewportRenderer {
    width: u32,
    height: u32,
}

impl EmbeddedViewportRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }

    pub fn render(&self, elapsed_seconds: f32) -> Image {
        let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(self.width, self.height);
        let pixels = pixel_buffer.make_mut_slice();

        let width_f32 = self.width as f32;
        let height_f32 = self.height as f32;

        let sphere_center_x = 0.06 * (elapsed_seconds * 0.7).sin();
        let sphere_center_y = 0.06 + 0.02 * (elapsed_seconds * 0.35).cos();
        let sphere_radius = 0.34_f32;

        let light_x = -0.55_f32;
        let light_y = 0.45_f32;
        let light_z = 0.70_f32;
        let light_len = (light_x * light_x + light_y * light_y + light_z * light_z).sqrt();
        let lx = light_x / light_len;
        let ly = light_y / light_len;
        let lz = light_z / light_len;

        for y in 0..self.height {
            let y_ratio = if self.height > 1 {
                y as f32 / (height_f32 - 1.0)
            } else {
                0.0
            };

            for x in 0..self.width {
                let x_ratio = if self.width > 1 {
                    x as f32 / (width_f32 - 1.0)
                } else {
                    0.0
                };

                let x_ndc = x_ratio * 2.0 - 1.0;
                let y_ndc = y_ratio * 2.0 - 1.0;

                let mut red = 0.10 + 0.10 * y_ratio;
                let mut green = 0.11 + 0.10 * y_ratio;
                let mut blue = 0.13 + 0.14 * y_ratio;

                let horizon_y = 0.08_f32;
                if y_ndc >= horizon_y {
                    let depth = ((y_ndc - horizon_y) / (1.0 - horizon_y)).clamp(0.0, 1.0);
                    let perspective = 0.26 + depth * 1.75;
                    let grid_x = x_ndc / perspective;

                    let grid_spacing = 0.25_f32;
                    let phase_x = (grid_x / grid_spacing).rem_euclid(1.0);
                    let vertical_line = (phase_x - 0.5).abs() < 0.018;

                    let mut horizontal_line = false;
                    for depth_step in 0..10 {
                        let t = depth_step as f32 / 9.0;
                        let line_y = horizon_y + t * t * (1.0 - horizon_y);
                        let thickness = 0.002 + (1.0 - t) * 0.003;
                        if (y_ndc - line_y).abs() <= thickness {
                            horizontal_line = true;
                            break;
                        }
                    }

                    let line_strength = match (vertical_line, horizontal_line) {
                        (true, true) => 0.65,
                        (true, false) => 0.48,
                        (false, true) => 0.38,
                        _ => 0.0,
                    };

                    if line_strength > 0.0 {
                        red = blend(red, 0.42, line_strength);
                        green = blend(green, 0.44, line_strength);
                        blue = blend(blue, 0.50, line_strength);
                    }
                }

                let dx = x_ndc - sphere_center_x;
                let dy = y_ndc - sphere_center_y;
                let dist_sq = dx * dx + dy * dy;
                let sphere_radius_sq = sphere_radius * sphere_radius;
                if dist_sq <= sphere_radius_sq {
                    let z = (sphere_radius_sq - dist_sq).sqrt();
                    let nx = dx / sphere_radius;
                    let ny = -dy / sphere_radius;
                    let nz = z / sphere_radius;

                    let ndotl = (nx * lx + ny * ly + nz * lz).max(0.0);
                    let ambient = 0.22;
                    let diffuse = ambient + ndotl * 0.75;
                    let reflect_z = (2.0 * ndotl * nz - lz).max(0.0);
                    let specular = reflect_z.powf(24.0) * 0.20;
                    let rim = (1.0 - nz).powf(2.8) * 0.10;

                    let base_red = 0.78;
                    let base_green = 0.63;
                    let base_blue = 0.56;

                    red = (base_red * diffuse + specular + rim).clamp(0.0, 1.0);
                    green = (base_green * diffuse + specular * 0.8 + rim).clamp(0.0, 1.0);
                    blue = (base_blue * diffuse + specular * 0.6 + rim).clamp(0.0, 1.0);
                }

                let index = (y as usize) * (self.width as usize) + (x as usize);
                pixels[index] = Rgba8Pixel {
                    r: to_u8(red),
                    g: to_u8(green),
                    b: to_u8(blue),
                    a: 255,
                };
            }
        }

        Image::from_rgba8(pixel_buffer)
    }
}

fn blend(base: f32, overlay: f32, alpha: f32) -> f32 {
    base * (1.0 - alpha) + overlay * alpha
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
