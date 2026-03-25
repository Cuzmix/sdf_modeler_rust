use std::path::Path;

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPlane {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

impl RefPlane {
    pub const ALL: [Self; 6] = [
        Self::Front,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Top,
        Self::Bottom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Front => "Front",
            Self::Back => "Back",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Top => "Top",
            Self::Bottom => "Bottom",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceImageEntry {
    pub path: String,
    pub plane: RefPlane,
    pub offset: Vec3,
    pub scale: f32,
    pub opacity: f32,
    pub locked: bool,
    pub visible: bool,
    pub size_px: [f32; 2],
}

#[derive(Default)]
pub struct ReferenceImageStore {
    pub images: Vec<ReferenceImageEntry>,
}

impl ReferenceImageStore {
    pub fn add_loaded(&mut self, path: &Path, width: u32, height: u32) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("Image has invalid dimensions".to_string());
        }
        self.images.push(ReferenceImageEntry {
            path: path.to_string_lossy().into_owned(),
            plane: RefPlane::Front,
            offset: Vec3::ZERO,
            scale: 2.0,
            opacity: 0.6,
            locked: false,
            visible: true,
            size_px: [width as f32, height as f32],
        });
        Ok(())
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.images.len() {
            self.images.remove(index);
        }
    }

    pub fn toggle_visibility(&mut self, index: usize) {
        if let Some(image) = self.images.get_mut(index) {
            image.visible = !image.visible;
        }
    }

    pub fn toggle_all_visibility(&mut self) {
        let any_visible = self.images.iter().any(|image| image.visible);
        for image in &mut self.images {
            image.visible = !any_visible;
        }
    }
}
