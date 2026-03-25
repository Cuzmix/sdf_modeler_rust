use std::path::Path;

use glam::Vec3;

use crate::app::actions::{Action, ActionSink};
use crate::app::reference_images::{RefPlane, ReferenceImageEntry, ReferenceImageStore};
use crate::gpu::camera::Camera;

pub struct LoadedReferenceImage {
    pub width: u32,
    pub height: u32,
    pub color_image: egui::ColorImage,
}

#[derive(Default)]
pub struct EguiReferenceImageCache {
    textures: Vec<egui::TextureHandle>,
}

impl EguiReferenceImageCache {
    pub fn push_loaded(&mut self, ctx: &egui::Context, path: &Path, loaded: LoadedReferenceImage) {
        let texture_name = format!("reference:{}", path.display());
        let texture = ctx.load_texture(
            texture_name,
            loaded.color_image,
            egui::TextureOptions::LINEAR,
        );
        self.textures.push(texture);
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.textures.len() {
            let _ = self.textures.remove(index);
        }
    }

    pub fn texture_id(&self, index: usize) -> Option<egui::TextureId> {
        self.textures.get(index).map(|texture| texture.id())
    }
}

pub fn load_reference_image(path: &Path) -> Result<LoadedReferenceImage, String> {
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
    Ok(LoadedReferenceImage {
        width,
        height,
        color_image,
    })
}

pub fn draw_controls(ui: &mut egui::Ui, store: &mut ReferenceImageStore, actions: &mut ActionSink) {
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

    if store.images.is_empty() {
        ui.weak("No reference images loaded.");
        return;
    }

    let mut remove_index: Option<usize> = None;
    for (index, image) in store.images.iter_mut().enumerate() {
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

fn quad_world_corners(image: &ReferenceImageEntry) -> [Vec3; 4] {
    let aspect = if image.size_px[1] > 0.0 {
        image.size_px[0] / image.size_px[1]
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
    store: &ReferenceImageStore,
    cache: &EguiReferenceImageCache,
) {
    if store.images.is_empty() {
        return;
    }
    let aspect = (rect.width() / rect.height().max(1.0)).max(1e-5);
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    for (index, image) in store.images.iter().enumerate() {
        if !image.visible {
            continue;
        }
        let Some(texture_id) = cache.texture_id(index) else {
            continue;
        };

        let tint = egui::Color32::from_white_alpha((image.opacity.clamp(0.0, 1.0) * 255.0) as u8);
        let mut mesh = egui::Mesh::with_texture(texture_id);
        let corners = quad_world_corners(image);
        let tess = 20usize;
        let inv_tess = 1.0 / tess as f32;

        for y in 0..tess {
            let v0 = y as f32 * inv_tess;
            let v1 = (y + 1) as f32 * inv_tess;
            for x in 0..tess {
                let u0 = x as f32 * inv_tess;
                let u1 = (x + 1) as f32 * inv_tess;

                let p00 = bilerp_quad(corners, u0, v0);
                let p10 = bilerp_quad(corners, u1, v0);
                let p11 = bilerp_quad(corners, u1, v1);
                let p01 = bilerp_quad(corners, u0, v1);

                let s00 = crate::ui::gizmo::world_to_screen(p00, &view_proj, rect);
                let s10 = crate::ui::gizmo::world_to_screen(p10, &view_proj, rect);
                let s11 = crate::ui::gizmo::world_to_screen(p11, &view_proj, rect);
                let s01 = crate::ui::gizmo::world_to_screen(p01, &view_proj, rect);
                let (Some(s00), Some(s10), Some(s11), Some(s01)) = (s00, s10, s11, s01) else {
                    continue;
                };

                let base = mesh.vertices.len() as u32;
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: s00,
                    uv: egui::pos2(u0, v0),
                    color: tint,
                });
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: s10,
                    uv: egui::pos2(u1, v0),
                    color: tint,
                });
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: s11,
                    uv: egui::pos2(u1, v1),
                    color: tint,
                });
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: s01,
                    uv: egui::pos2(u0, v1),
                    color: tint,
                });
                mesh.indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base,
                    base + 2,
                    base + 3,
                ]);
            }
        }

        if !mesh.indices.is_empty() {
            painter.add(egui::Shape::mesh(mesh));
        }
    }
}

fn bilerp_quad(corners: [Vec3; 4], u: f32, v: f32) -> Vec3 {
    let top = corners[0].lerp(corners[1], u);
    let bottom = corners[3].lerp(corners[2], u);
    top.lerp(bottom, v)
}
