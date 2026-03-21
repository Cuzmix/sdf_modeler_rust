use crate::gpu::camera::CameraUniform;

pub struct PendingPick {
    pub mouse_pos: [f32; 2],
    pub camera_uniform: CameraUniform,
    /// Whether additive selection was requested during the click.
    pub additive_select_held: bool,
}

pub struct PickResult {
    pub material_id: i32,
    pub distance: f32,
    pub world_pos: [f32; 3],
}
