use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.4,
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

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, 0.01, 100.0)
    }

    pub fn to_uniform(&self, viewport: [f32; 4], time: f32) -> CameraUniform {
        let aspect = viewport[2] / viewport[3].max(1.0);
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let inv_vp = (proj * view).inverse();

        CameraUniform {
            inv_view_proj: inv_vp.to_cols_array(),
            eye: self.eye().extend(1.0).to_array(),
            viewport,
            time,
            _pad: [0.0; 3],
        }
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch += dy * 0.005;
        self.pitch = self
            .pitch
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * 0.002;
        self.target -= right * dx * speed;
        self.target += up * dy * speed;
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * 0.001;
        self.distance = self.distance.clamp(0.1, 100.0);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 4],
    pub viewport: [f32; 4],
    pub time: f32,
    pub _pad: [f32; 3],
}
