use crate::gpu::camera::CameraUniform;

pub struct PendingPick {
    pub mouse_pos: [f32; 2],
    pub camera_uniform: CameraUniform,
    /// Whether Ctrl was held during the click (for multi-select toggle).
    pub ctrl_held: bool,
}

pub struct PickResult {
    pub material_id: i32,
    pub distance: f32,
    pub world_pos: [f32; 3],
}
