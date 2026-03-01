use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

const ORBIT_SENSITIVITY: f32 = 0.005;
const PAN_SPEED_FACTOR: f32 = 0.002;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;
const MAX_DISTANCE: f32 = 100.0;
const PITCH_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;

#[derive(Serialize, Deserialize)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    #[serde(default)]
    pub roll: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.4,
            roll: 0.0,
            distance: 5.0,
            target: Vec3::ZERO,
            fov: 45.0_f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();
        self.target + Vec3::new(x, y, z)
    }

    fn up(&self) -> Vec3 {
        if self.pitch.cos() >= 0.0 { Vec3::Y } else { Vec3::NEG_Y }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, self.up());
        if self.roll == 0.0 {
            view
        } else {
            Mat4::from_rotation_z(self.roll) * view
        }
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, 0.01, 100.0)
    }

    pub fn to_uniform(
        &self,
        viewport: [f32; 4],
        time: f32,
        quality_mode: f32,
        grid_enabled: bool,
        scene_bounds: ([f32; 3], [f32; 3]),
        selected_idx: f32,
    ) -> CameraUniform {
        let aspect = viewport[2] / viewport[3].max(1.0);
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let inv_vp = (proj * view).inverse();

        CameraUniform {
            inv_view_proj: inv_vp.to_cols_array(),
            eye: self.eye().extend(1.0).to_array(),
            viewport,
            time,
            quality_mode,
            grid_enabled: if grid_enabled { 1.0 } else { 0.0 },
            selected_idx,
            scene_min: [scene_bounds.0[0], scene_bounds.0[1], scene_bounds.0[2], 0.0],
            scene_max: [scene_bounds.1[0], scene_bounds.1[1], scene_bounds.1[2], 0.0],
        }
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * ORBIT_SENSITIVITY;
        self.pitch += dy * ORBIT_SENSITIVITY;
    }

    pub fn clamp_pitch(&mut self) {
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn roll_by(&mut self, delta: f32, sensitivity: f32) {
        self.roll += delta * sensitivity;
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(self.up()).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * PAN_SPEED_FACTOR;
        self.target -= right * dx * speed;
        self.target += up * dy * speed;
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * ZOOM_SENSITIVITY;
        self.distance = self.distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn set_front(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.roll = 0.0;
    }

    pub fn set_top(&mut self) {
        self.yaw = 0.0;
        self.pitch = std::f32::consts::FRAC_PI_2;
        self.roll = 0.0;
    }

    pub fn set_right(&mut self) {
        self.yaw = std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
        self.roll = 0.0;
    }

    pub fn focus_on(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.distance = (radius / (self.fov * 0.5).tan()).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 4],
    pub viewport: [f32; 4],
    pub time: f32,
    pub quality_mode: f32,
    pub grid_enabled: f32,
    pub selected_idx: f32,
    pub scene_min: [f32; 4],
    pub scene_max: [f32; 4],
}
