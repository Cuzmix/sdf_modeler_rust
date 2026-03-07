use std::path::Path;

use eframe::egui;
use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::gpu::camera::Camera;

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

pub struct ReferenceImage {
    pub texture: egui::TextureHandle,
    pub path: String,
    pub plane: RefPlane,
    pub offset: Vec3,
    pub scale: f32,
    pub opacity: f32,
    pub locked: bool,
    pub visible: bool,
    pub size_px: egui::Vec2,
}

impl ReferenceImage {
    pub fn texture_id(&self) -> egui::TextureId {
        self.texture.id()
    }
}

#[derive(Default)]
pub struct ReferenceImageManager {
    pub images: Vec<ReferenceImage>,
}

impl ReferenceImageManager {
    pub fn add_from_path(&mut self, ctx: &egui::Context, path: &Path) -> Result<(), String> {
        let dynamic_image = image::open(path)
            .map_err(|e| format!("Failed to open image '{}': {}", path.display(), e))?;
        let rgba = dynamic_image.to_rgba8();
        let (width, height) = rgba.dimensions();
        if width == 0 || height == 0 {
            return Err("Image has invalid dimensions".to_string());
        }
        let pixels = rgba.into_raw();
        let color_image =
            egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels);
        let texture_name = format!("reference:{}", path.display());
        let texture = ctx.load_texture(texture_name, color_image, egui::TextureOptions::LINEAR);
        self.images.push(ReferenceImage {
            texture,
            path: path.to_string_lossy().into_owned(),
            plane: RefPlane::Front,
            offset: Vec3::ZERO,
            scale: 2.0,
            opacity: 0.6,
            locked: false,
            visible: true,
            size_px: egui::vec2(width as f32, height as f32),
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

pub fn draw_controls(
    ui: &mut egui::Ui,
    manager: &mut ReferenceImageManager,
    actions: &mut ActionSink,
) {
    ui.separator();
    ui.heading("Reference Images");

    ui.horizontal(|ui| {
        if ui.button("Add Reference Image").clicked() {
            actions.push(Action::AddReferenceImage);
        }
        if ui.button("Toggle All (Alt+R)").clicked() {
            actions.push(Action::ToggleAllReferenceImages);
        }
    });

    if manager.images.is_empty() {
        ui.weak("No reference images loaded.");
        return;
    }

    let mut remove_index: Option<usize> = None;
    for (index, image) in manager.images.iter_mut().enumerate() {
        ui.separator();

        let file_name = Path::new(&image.path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&image.path);
        ui.horizontal(|ui| {
            ui.label(file_name);
            if ui
                .small_button(if image.visible { "Hide" } else { "Show" })
                .clicked()
            {
                actions.push(Action::ToggleReferenceImageVisibility(index));
            }
            if ui.small_button("Remove").clicked() {
                remove_index = Some(index);
            }
        });

        ui.checkbox(&mut image.locked, "Locked");

        ui.add_enabled_ui(!image.locked, |ui| {
            egui::ComboBox::from_label("Plane")
                .selected_text(image.plane.label())
                .show_ui(ui, |ui| {
                    for plane in RefPlane::ALL {
                        ui.selectable_value(&mut image.plane, plane, plane.label());
                    }
                });

            ui.horizontal(|ui| {
                ui.label("Opacity");
                ui.add(egui::Slider::new(&mut image.opacity, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Scale");
                ui.add(egui::Slider::new(&mut image.scale, 0.05..=20.0).logarithmic(true));
            });

            ui.label("Offset");
            ui.horizontal(|ui| {
                ui.label("X");
                ui.add(egui::DragValue::new(&mut image.offset.x).speed(0.01));
                ui.label("Y");
                ui.add(egui::DragValue::new(&mut image.offset.y).speed(0.01));
                ui.label("Z");
                ui.add(egui::DragValue::new(&mut image.offset.z).speed(0.01));
            });
        });
    }

    if let Some(index) = remove_index {
        actions.push(Action::RemoveReferenceImage(index));
    }
}

fn quad_world_corners(image: &ReferenceImage) -> [Vec3; 4] {
    let aspect = if image.size_px.y > 0.0 {
        image.size_px.x / image.size_px.y
    } else {
        1.0
    };
    let half_h = image.scale * 0.5;
    let half_w = half_h * aspect.max(1e-4);
    let c = image.offset;

    match image.plane {
        RefPlane::Front | RefPlane::Back => [
            Vec3::new(c.x - half_w, c.y + half_h, c.z),
            Vec3::new(c.x + half_w, c.y + half_h, c.z),
            Vec3::new(c.x + half_w, c.y - half_h, c.z),
            Vec3::new(c.x - half_w, c.y - half_h, c.z),
        ],
        RefPlane::Left | RefPlane::Right => [
            Vec3::new(c.x, c.y + half_h, c.z + half_w),
            Vec3::new(c.x, c.y + half_h, c.z - half_w),
            Vec3::new(c.x, c.y - half_h, c.z - half_w),
            Vec3::new(c.x, c.y - half_h, c.z + half_w),
        ],
        RefPlane::Top | RefPlane::Bottom => [
            Vec3::new(c.x - half_w, c.y, c.z - half_h),
            Vec3::new(c.x + half_w, c.y, c.z - half_h),
            Vec3::new(c.x + half_w, c.y, c.z + half_h),
            Vec3::new(c.x - half_w, c.y, c.z + half_h),
        ],
    }
}

pub fn draw_overlay(
    painter: &egui::Painter,
    camera: &Camera,
    rect: egui::Rect,
    manager: &ReferenceImageManager,
) {
    if manager.images.is_empty() {
        return;
    }
    let aspect = (rect.width() / rect.height().max(1.0)).max(1e-5);
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    for image in &manager.images {
        if !image.visible {
            continue;
        }
        let corners = quad_world_corners(image);
        let projected: Option<[egui::Pos2; 4]> = corners
            .iter()
            .map(|&corner| crate::ui::gizmo::world_to_screen(corner, &view_proj, rect))
            .collect::<Option<Vec<_>>>()
            .and_then(|pts| pts.try_into().ok());
        let Some(projected) = projected else {
            continue;
        };

        let tint = egui::Color32::from_white_alpha((image.opacity.clamp(0.0, 1.0) * 255.0) as u8);
        let mut mesh = egui::Mesh::with_texture(image.texture_id());
        let base = mesh.vertices.len() as u32;
        mesh.vertices.push(egui::epaint::Vertex {
            pos: projected[0],
            uv: egui::pos2(0.0, 0.0),
            color: tint,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: projected[1],
            uv: egui::pos2(1.0, 0.0),
            color: tint,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: projected[2],
            uv: egui::pos2(1.0, 1.0),
            color: tint,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: projected[3],
            uv: egui::pos2(0.0, 1.0),
            color: tint,
        });
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        painter.add(egui::Shape::mesh(mesh));
    }
}

