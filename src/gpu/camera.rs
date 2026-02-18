use glam::{Mat4, Vec3};

/// GPU-side camera uniform data. Must match the WGSL `Camera` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view: [f32; 16],
    pub projection: [f32; 16],
    pub inv_view_proj: [f32; 16],
    pub eye: [f32; 3],
    pub _pad1: f32,
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad2: f32,
}

/// Orbit camera that revolves around a target point.
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            azimuth: 0.5,            // ~30° horizontal
            elevation: 0.4,          // ~23° up
            fov_y: 45.0_f32.to_radians(),
            near: 0.01,
            far: 100.0,
        }
    }

    /// Camera world position computed from spherical coordinates.
    pub fn eye(&self) -> Vec3 {
        let cos_el = self.elevation.cos();
        let sin_el = self.elevation.sin();
        let cos_az = self.azimuth.cos();
        let sin_az = self.azimuth.sin();

        self.target
            + Vec3::new(
                cos_el * sin_az,
                sin_el,
                cos_el * cos_az,
            ) * self.distance
    }

    /// View matrix (world → camera).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// Perspective projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Pack camera data into the GPU uniform struct.
    pub fn uniforms(&self, width: u32, height: u32, time: f32) -> CameraUniforms {
        let aspect = width as f32 / height.max(1) as f32;
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        let vp = proj * view;
        let inv_vp = vp.inverse();
        let eye = self.eye();

        CameraUniforms {
            view: view.to_cols_array(),
            projection: proj.to_cols_array(),
            inv_view_proj: inv_vp.to_cols_array(),
            eye: eye.to_array(),
            _pad1: 0.0,
            resolution: [width as f32, height as f32],
            time,
            _pad2: 0.0,
        }
    }

    /// Orbit: rotate azimuth/elevation from mouse pixel delta.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.azimuth -= dx * sensitivity;
        self.elevation += dy * sensitivity;

        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.elevation = self.elevation.clamp(-limit, limit);
    }

    /// Pan: translate target in the screen-aligned plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.002 * self.distance;
        let view = self.view_matrix();

        // Extract right and up vectors from the view matrix
        let right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x);
        let up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y);

        self.target -= right * dx * sensitivity;
        self.target += up * dy * sensitivity;
    }

    /// Zoom: adjust distance (multiplicative for smooth feel).
    pub fn zoom(&mut self, delta: f32) {
        let factor = 1.0 + delta * 0.001;
        self.distance = (self.distance * factor).clamp(0.1, 100.0);
    }
}
