use crate::gpu::camera::CameraUniform;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PendingPickIntent {
    Selection,
    SculptStroke,
    SculptHover,
    SculptTargetSwitch,
}

impl PendingPickIntent {
    pub fn is_sculpt_intent(self) -> bool {
        matches!(self, Self::SculptStroke | Self::SculptHover)
    }
}

pub struct PendingPick {
    pub mouse_pos: [f32; 2],
    pub camera_uniform: CameraUniform,
    pub intent: PendingPickIntent,
    /// Whether additive selection was requested during the click.
    pub additive_select_held: bool,
}

pub struct PickResult {
    pub material_id: i32,
    pub distance: f32,
    pub world_pos: [f32; 3],
}
