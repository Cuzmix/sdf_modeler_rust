pub(super) struct SlintViewportTexture {
    pub(super) view: wgpu::TextureView,
    pub(super) size: (u32, u32),
}

use super::SlintHostState;
use crate::app::slint_frontend::SlintHostWindow;

impl SlintHostState {
    pub(in crate::app::slint_frontend) fn render_viewport_if_needed(&mut self) {
        if !self.viewport_dirty {
            return;
        }
        let Some(texture) = self.viewport_texture.as_ref() else {
            return;
        };
        self.app.render_viewport_texture(
            &texture.view,
            self.viewport_size.0.max(1),
            self.viewport_size.1.max(1),
        );
        self.viewport_dirty = false;
    }

    pub(in crate::app::slint_frontend) fn release_viewport_texture(&mut self) {
        self.viewport_texture = None;
        self.viewport_dirty = true;
    }

    pub(super) fn ensure_viewport_texture(&mut self, window: &SlintHostWindow) {
        let width = self.viewport_size.0.max(1);
        let height = self.viewport_size.1.max(1);
        if self
            .viewport_texture
            .as_ref()
            .is_some_and(|texture| texture.size == (width, height))
        {
            return;
        }

        let texture = self
            .app
            .gpu
            .render_context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Slint Viewport Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.app.gpu.render_context.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image = match slint::Image::try_from(texture.clone()) {
            Ok(image) => image,
            Err(error) => {
                log::error!("Failed to import viewport texture into Slint: {error}");
                return;
            }
        };
        window.set_viewport_image(image.clone());
        self.viewport_texture = Some(SlintViewportTexture {
            view,
            size: (width, height),
        });
        self.viewport_dirty = true;
    }
}
