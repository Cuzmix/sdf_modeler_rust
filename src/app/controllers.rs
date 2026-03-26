use crate::graph::presented_object::{
    current_transform_owner, resolve_presented_object, PresentedObjectKind,
};
use crate::graph::scene::{NodeData, NodeId};
use crate::settings::{EnvironmentBackgroundMode, EnvironmentSource};

use super::SdfApp;

impl SdfApp {
    pub(super) fn set_selected_position_component(&mut self, axis: usize, value: f32) {
        let mut changed = false;
        for target_id in self.selected_transform_targets() {
            let Some(node) = self.doc.scene.nodes.get_mut(&target_id) else {
                continue;
            };
            match &mut node.data {
                NodeData::Primitive { position, .. } | NodeData::Sculpt { position, .. } => {
                    set_vec3_component(position, axis, value);
                    changed = true;
                }
                NodeData::Transform { translation, .. } => {
                    set_vec3_component(translation, axis, value);
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_rotation_deg_component(&mut self, axis: usize, degrees: f32) {
        let radians = degrees.to_radians();
        let mut changed = false;
        for target_id in self.selected_transform_targets() {
            let Some(node) = self.doc.scene.nodes.get_mut(&target_id) else {
                continue;
            };
            match &mut node.data {
                NodeData::Primitive { rotation, .. }
                | NodeData::Sculpt { rotation, .. }
                | NodeData::Transform { rotation, .. } => {
                    set_vec3_component(rotation, axis, radians);
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_scale_component(&mut self, axis: usize, value: f32) {
        let mut changed = false;
        for target_id in self.selected_transform_targets() {
            let Some(node) = self.doc.scene.nodes.get_mut(&target_id) else {
                continue;
            };
            match &mut node.data {
                NodeData::Primitive { scale, .. } | NodeData::Transform { scale, .. } => {
                    set_vec3_component(scale, axis, value.max(0.001));
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_material_color_component(&mut self, axis: usize, value: f32) {
        let mut changed = false;
        for target_id in self.selected_material_target_ids() {
            let Some(node) = self.doc.scene.nodes.get_mut(&target_id) else {
                continue;
            };
            let Some(material) = node.data.material_mut() else {
                continue;
            };
            set_vec3_component(&mut material.base_color, axis, value.clamp(0.0, 1.0));
            changed = true;
        }
        if changed {
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_material_roughness(&mut self, value: f32) {
        self.edit_selected_material(|material| material.roughness = value.clamp(0.0, 1.0));
    }

    pub(super) fn set_selected_material_metallic(&mut self, value: f32) {
        self.edit_selected_material(|material| material.metallic = value.clamp(0.0, 1.0));
    }

    pub(super) fn set_selected_operation_smooth_k(&mut self, value: f32) {
        self.edit_selected_operation(|smooth_k, _, _| *smooth_k = value.max(0.0));
    }

    pub(super) fn set_selected_operation_steps(&mut self, value: f32) {
        self.edit_selected_operation(|_, steps, _| *steps = value.max(0.0));
    }

    pub(super) fn set_selected_operation_color_blend(&mut self, value: f32) {
        self.edit_selected_operation(|_, _, color_blend| *color_blend = value.max(-1.0));
    }

    pub(super) fn set_selected_sculpt_resolution(&mut self, value: u32) {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        if let NodeData::Sculpt {
            desired_resolution, ..
        } = &mut node.data
        {
            *desired_resolution = value.max(8);
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_sculpt_layer_intensity(&mut self, value: f32) {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        if let NodeData::Sculpt {
            layer_intensity, ..
        } = &mut node.data
        {
            *layer_intensity = value.clamp(0.0, 4.0);
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_brush_radius(&mut self, value: f32) {
        let selected_mode = self.doc.sculpt_state.selected_brush();
        let profile = self.doc.sculpt_state.selected_profile_mut();
        profile.radius = value.clamp(0.05, 2.0);
        profile.clamp_strength_for_mode(selected_mode);
    }

    pub(super) fn set_selected_brush_strength(&mut self, value: f32) {
        let selected_mode = self.doc.sculpt_state.selected_brush();
        let profile = self.doc.sculpt_state.selected_profile_mut();
        profile.strength = value;
        profile.clamp_strength_for_mode(selected_mode);
    }

    pub(super) fn set_selected_light_color_component(&mut self, axis: usize, value: f32) {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        if let NodeData::Light { color, .. } = &mut node.data {
            set_vec3_component(color, axis, value.clamp(0.0, 10.0));
            self.gpu.buffer_dirty = true;
        }
    }

    pub(super) fn set_selected_light_intensity(&mut self, value: f32) {
        self.edit_selected_light(|_, intensity, _, _, _, _| *intensity = value.max(0.0));
    }

    pub(super) fn set_selected_light_range(&mut self, value: f32) {
        self.edit_selected_light(|_, _, range, _, _, _| *range = value.max(0.01));
    }

    pub(super) fn set_selected_light_cast_shadows(&mut self, value: bool) {
        self.edit_selected_light(|_, _, _, cast_shadows, _, _| *cast_shadows = value);
    }

    pub(super) fn set_selected_light_volumetric(&mut self, value: bool) {
        self.edit_selected_light(|_, _, _, _, volumetric, _| *volumetric = value);
    }

    pub(super) fn set_selected_light_volumetric_density(&mut self, value: f32) {
        self.edit_selected_light(|_, _, _, _, _, volumetric_density| {
            *volumetric_density = value.clamp(0.0, 1.0);
        });
    }

    pub(super) fn set_render_show_grid(&mut self, value: bool) {
        self.settings.render.show_grid = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_render_show_node_labels(&mut self, value: bool) {
        self.settings.render.show_node_labels = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_render_show_bounding_box(&mut self, value: bool) {
        self.settings.render.show_bounding_box = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_render_show_light_gizmos(&mut self, value: bool) {
        self.settings.render.show_light_gizmos = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_render_shadows_enabled(&mut self, value: bool) {
        self.settings.render.shadows_enabled = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_render_ao_enabled(&mut self, value: bool) {
        self.settings.render.ao_enabled = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_environment_source(&mut self, value: EnvironmentSource) {
        self.settings.render.environment_source = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn pick_environment_hdri(&mut self) {
        match crate::desktop_dialogs::environment_hdri_dialog() {
            crate::desktop_dialogs::FileDialogSelection::Selected(path) => {
                self.settings.render.hdri_path = Some(path.to_string_lossy().to_string());
                self.settings.render.environment_source = EnvironmentSource::Hdri;
                if self.settings.render.environment_bake_resolution == 0 {
                    self.settings.render.environment_bake_resolution = 512;
                }
                self.commit_render_settings_edit();
            }
            crate::desktop_dialogs::FileDialogSelection::Cancelled => {}
            crate::desktop_dialogs::FileDialogSelection::Unsupported => {
                self.ui.toasts.push(crate::app::Toast {
                    message: "HDRI picking is not supported on this platform yet".to_string(),
                    is_error: true,
                    created: crate::compat::Instant::now(),
                    duration: crate::compat::Duration::from_secs(4),
                });
            }
        }
    }

    pub(super) fn set_environment_rotation_degrees(&mut self, value: f32) {
        self.settings.render.environment_rotation_degrees = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_environment_exposure(&mut self, value: f32) {
        self.settings.render.environment_exposure = value.clamp(-8.0, 8.0);
        self.commit_render_settings_edit();
    }

    pub(super) fn set_environment_bake_resolution(&mut self, value: u32) {
        self.settings.render.environment_bake_resolution = value.min(4096);
        self.commit_render_settings_edit();
    }

    pub(super) fn set_environment_background_mode(&mut self, value: EnvironmentBackgroundMode) {
        self.settings.render.environment_background_mode = value;
        self.commit_render_settings_edit();
    }

    pub(super) fn set_environment_background_blur(&mut self, value: f32) {
        self.settings.render.environment_background_blur = value.clamp(0.0, 1.0);
        self.commit_render_settings_edit();
    }

    pub(super) fn set_export_resolution(&mut self, value: u32) {
        let max_res = self.settings.max_export_resolution.max(16);
        self.settings.export_resolution = value.clamp(16, max_res);
        self.settings.save();
    }

    pub(super) fn set_adaptive_export(&mut self, value: bool) {
        self.settings.adaptive_export = value;
        self.settings.save();
    }

    pub(super) fn toggle_reference_image_lock(&mut self, index: usize) {
        let Some(reference) = self.ui.reference_images.images.get_mut(index) else {
            return;
        };
        reference.locked = !reference.locked;
    }

    pub(super) fn cycle_reference_image_plane(&mut self, index: usize) {
        let Some(reference) = self.ui.reference_images.images.get_mut(index) else {
            return;
        };
        reference.plane = match reference.plane {
            crate::app::reference_images::RefPlane::Front => {
                crate::app::reference_images::RefPlane::Back
            }
            crate::app::reference_images::RefPlane::Back => {
                crate::app::reference_images::RefPlane::Left
            }
            crate::app::reference_images::RefPlane::Left => {
                crate::app::reference_images::RefPlane::Right
            }
            crate::app::reference_images::RefPlane::Right => {
                crate::app::reference_images::RefPlane::Top
            }
            crate::app::reference_images::RefPlane::Top => {
                crate::app::reference_images::RefPlane::Bottom
            }
            crate::app::reference_images::RefPlane::Bottom => {
                crate::app::reference_images::RefPlane::Front
            }
        };
    }

    pub(super) fn set_reference_image_opacity(&mut self, index: usize, value: f32) {
        let Some(reference) = self.ui.reference_images.images.get_mut(index) else {
            return;
        };
        reference.opacity = value.clamp(0.0, 1.0);
    }

    pub(super) fn set_reference_image_scale(&mut self, index: usize, value: f32) {
        let Some(reference) = self.ui.reference_images.images.get_mut(index) else {
            return;
        };
        reference.scale = value.clamp(0.05, 20.0);
    }

    fn selected_transform_targets(&self) -> Vec<NodeId> {
        let mut targets = self
            .ordered_selected_ids()
            .into_iter()
            .filter_map(|selected_id| {
                let object = resolve_presented_object(&self.doc.scene, selected_id)?;
                match object.kind {
                    PresentedObjectKind::Parametric => {
                        current_transform_owner(&self.doc.scene, object.host_id)
                    }
                    PresentedObjectKind::Voxel => Some(object.host_id),
                    PresentedObjectKind::Light => match self
                        .doc
                        .scene
                        .nodes
                        .get(&object.object_root_id)
                        .map(|node| &node.data)
                    {
                        Some(NodeData::Transform { .. }) => Some(object.object_root_id),
                        _ => Some(object.host_id),
                    },
                }
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    fn selected_material_target_ids(&self) -> Vec<NodeId> {
        let mut ids = self
            .ordered_selected_ids()
            .into_iter()
            .filter(|id| {
                self.doc
                    .scene
                    .nodes
                    .get(id)
                    .and_then(|node| node.data.material())
                    .is_some()
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn ordered_selected_ids(&self) -> Vec<NodeId> {
        let mut ids = self
            .ui
            .selection
            .selected_set
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if ids.is_empty() {
            if let Some(selected_id) = self.ui.selection.selected {
                ids.push(selected_id);
            }
        } else if let Some(selected_id) = self.ui.selection.selected {
            ids.push(selected_id);
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn edit_selected_material<F>(&mut self, edit: F)
    where
        F: FnOnce(&mut crate::graph::scene::MaterialParams),
    {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        let Some(material) = node.data.material_mut() else {
            return;
        };
        edit(material);
        self.gpu.buffer_dirty = true;
    }

    fn edit_selected_operation<F>(&mut self, edit: F)
    where
        F: FnOnce(&mut f32, &mut f32, &mut f32),
    {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        let NodeData::Operation {
            smooth_k,
            steps,
            color_blend,
            ..
        } = &mut node.data
        else {
            return;
        };
        edit(smooth_k, steps, color_blend);
        self.gpu.buffer_dirty = true;
    }

    fn edit_selected_light<F>(&mut self, edit: F)
    where
        F: FnOnce(&mut glam::Vec3, &mut f32, &mut f32, &mut bool, &mut bool, &mut f32),
    {
        let Some(selected_id) = self.ui.selection.selected else {
            return;
        };
        let Some(node) = self.doc.scene.nodes.get_mut(&selected_id) else {
            return;
        };
        let NodeData::Light {
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
            ..
        } = &mut node.data
        else {
            return;
        };
        edit(
            color,
            intensity,
            range,
            cast_shadows,
            volumetric,
            volumetric_density,
        );
        self.gpu.buffer_dirty = true;
    }

    fn commit_render_settings_edit(&mut self) {
        self.settings.save();
        self.gpu.current_structure_key = 0;
        self.gpu.buffer_dirty = true;
    }
}

fn set_vec3_component(value: &mut glam::Vec3, axis: usize, next: f32) {
    match axis {
        0 => value.x = next,
        1 => value.y = next,
        2 => value.z = next,
        _ => {}
    }
}
