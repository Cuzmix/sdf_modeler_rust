use glam::Vec3;

use crate::graph::scene::{NodeData, NodeId};
use crate::graph::voxel::{self, VoxelGrid};

use super::SdfApp;

const DETAIL_STEP_FACTOR: f32 = 1.25;
const MIN_SCULPT_RESOLUTION: u32 = 16;
const EXPAND_VOLUME_FACTOR: f32 = 1.35;

fn next_detail_resolution(current_resolution: u32, increase: bool, max_resolution: u32) -> u32 {
    let max_resolution = max_resolution.max(MIN_SCULPT_RESOLUTION);
    if increase {
        let mut new_resolution = ((current_resolution as f32) * DETAIL_STEP_FACTOR).round() as u32;
        new_resolution = new_resolution.max(current_resolution.saturating_add(1));
        new_resolution.clamp(MIN_SCULPT_RESOLUTION, max_resolution)
    } else {
        let mut new_resolution = ((current_resolution as f32) / DETAIL_STEP_FACTOR).round() as u32;
        new_resolution = new_resolution
            .min(current_resolution.saturating_sub(1))
            .max(MIN_SCULPT_RESOLUTION);
        new_resolution.clamp(MIN_SCULPT_RESOLUTION, max_resolution)
    }
}

fn expanded_volume_bounds(bounds_min: Vec3, bounds_max: Vec3) -> (Vec3, Vec3) {
    let center = (bounds_min + bounds_max) * 0.5;
    let half_extent = (bounds_max - bounds_min) * 0.5 * EXPAND_VOLUME_FACTOR;
    (center - half_extent, center + half_extent)
}

fn remesh_resolution_for_detail(
    bounds_min: Vec3,
    bounds_max: Vec3,
    target_pitch: f32,
    max_resolution: u32,
) -> (u32, bool) {
    let ideal_resolution = voxel::resolution_for_voxel_pitch(bounds_min, bounds_max, target_pitch);
    let max_resolution = max_resolution.max(MIN_SCULPT_RESOLUTION);
    let new_resolution = ideal_resolution.clamp(MIN_SCULPT_RESOLUTION, max_resolution);
    (new_resolution, new_resolution < ideal_resolution)
}

impl SdfApp {
    pub(super) fn increase_sculpt_detail(&mut self, node_id: NodeId) {
        let Some((current_resolution, bounds_min, bounds_max)) = self.sculpt_grid_info(node_id)
        else {
            return;
        };
        let new_resolution = next_detail_resolution(
            current_resolution,
            true,
            self.settings.max_sculpt_resolution,
        );
        if new_resolution == current_resolution {
            self.push_sculpt_toast("Sculpt is already at the maximum detail size", false);
            return;
        }

        self.resample_sculpt_node(node_id, new_resolution, bounds_min, bounds_max);
        let detail_state = self.doc.sculpt_state.detail_state_mut();
        detail_state.last_pre_expand_detail_size = None;
        detail_state.detail_limited_after_growth = false;
    }

    pub(super) fn decrease_sculpt_detail(&mut self, node_id: NodeId) {
        let Some((current_resolution, bounds_min, bounds_max)) = self.sculpt_grid_info(node_id)
        else {
            return;
        };
        let new_resolution = next_detail_resolution(
            current_resolution,
            false,
            self.settings.max_sculpt_resolution,
        );
        if new_resolution == current_resolution {
            self.push_sculpt_toast("Sculpt is already at the minimum detail size", false);
            return;
        }

        self.resample_sculpt_node(node_id, new_resolution, bounds_min, bounds_max);
        let detail_state = self.doc.sculpt_state.detail_state_mut();
        detail_state.last_pre_expand_detail_size = None;
        detail_state.detail_limited_after_growth = false;
    }

    pub(super) fn expand_sculpt_volume(&mut self, node_id: NodeId) {
        let Some((current_resolution, bounds_min, bounds_max)) = self.sculpt_grid_info(node_id)
        else {
            return;
        };
        let current_pitch = voxel::voxel_pitch_for_bounds(bounds_min, bounds_max, current_resolution);
        let (new_bounds_min, new_bounds_max) = expanded_volume_bounds(bounds_min, bounds_max);

        self.resample_sculpt_node(node_id, current_resolution, new_bounds_min, new_bounds_max);
        let detail_state = self.doc.sculpt_state.detail_state_mut();
        detail_state.last_pre_expand_detail_size = Some(current_pitch);
        detail_state.detail_limited_after_growth = true;
    }

    pub(super) fn remesh_sculpt_at_current_detail(&mut self, node_id: NodeId) {
        let Some((current_resolution, bounds_min, bounds_max)) = self.sculpt_grid_info(node_id)
        else {
            return;
        };
        let target_pitch = self
            .doc
            .sculpt_state
            .detail_state()
            .last_pre_expand_detail_size
            .unwrap_or_else(|| voxel::voxel_pitch_for_bounds(bounds_min, bounds_max, current_resolution));
        let (new_resolution, limited) = remesh_resolution_for_detail(
            bounds_min,
            bounds_max,
            target_pitch,
            self.settings.max_sculpt_resolution,
        );

        if new_resolution != current_resolution {
            self.resample_sculpt_node(node_id, new_resolution, bounds_min, bounds_max);
        }

        let detail_state = self.doc.sculpt_state.detail_state_mut();
        if limited {
            detail_state.last_pre_expand_detail_size = Some(target_pitch);
        } else {
            detail_state.last_pre_expand_detail_size = None;
        }
        detail_state.detail_limited_after_growth = limited;
        if limited {
            self.push_sculpt_toast("Detail remesh hit the sculpt resolution cap", false);
        }
    }

    pub(super) fn fit_sculpt_volume(&mut self, node_id: NodeId) {
        let Some((current_resolution, _, _)) = self.sculpt_grid_info(node_id) else {
            return;
        };
        let Some((new_bounds_min, new_bounds_max)) = self.fitted_sculpt_bounds(node_id) else {
            self.push_sculpt_toast("Could not find sculpt surface to fit the volume", false);
            return;
        };
        self.resample_sculpt_node(node_id, current_resolution, new_bounds_min, new_bounds_max);
        let detail_state = self.doc.sculpt_state.detail_state_mut();
        detail_state.last_pre_expand_detail_size = None;
        detail_state.detail_limited_after_growth = false;
    }

    fn sculpt_grid_info(&self, node_id: NodeId) -> Option<(u32, Vec3, Vec3)> {
        let node = self.doc.scene.nodes.get(&node_id)?;
        let NodeData::Sculpt { voxel_grid, .. } = &node.data else {
            return None;
        };
        Some((
            voxel_grid.resolution,
            voxel_grid.bounds_min,
            voxel_grid.bounds_max,
        ))
    }

    fn resample_sculpt_node(
        &mut self,
        node_id: NodeId,
        resolution: u32,
        bounds_min: Vec3,
        bounds_max: Vec3,
    ) {
        self.finalize_pending_grab_repair();
        self.doc.sculpt_state.clear_stroke_state();

        let Some(new_grid) = self
            .doc
            .scene
            .nodes
            .get(&node_id)
            .and_then(|node| match &node.data {
                NodeData::Sculpt { voxel_grid, .. } => {
                    Some(voxel_grid.resampled_to(resolution, bounds_min, bounds_max))
                }
                _ => None,
            })
        else {
            return;
        };

        if let Some(node) = self.doc.scene.nodes.get_mut(&node_id) {
            if let NodeData::Sculpt {
                voxel_grid,
                desired_resolution,
                ..
            } = &mut node.data
            {
                *voxel_grid = new_grid;
                *desired_resolution = resolution;
            }
        }

        self.async_state.last_sculpt_hit = None;
        self.async_state.lazy_brush_pos = None;
        self.async_state.hover_world_pos = None;
        self.async_state.cursor_over_geometry = false;
        self.clear_sculpt_runtime_cache();
        self.gpu.buffer_dirty = true;
        if self.settings.render.composite_volume_enabled {
            self.perf.composite_full_update_needed = true;
        }
    }

    fn fitted_sculpt_bounds(&self, node_id: NodeId) -> Option<(Vec3, Vec3)> {
        let node = self.doc.scene.nodes.get(&node_id)?;
        let NodeData::Sculpt {
            position,
            rotation,
            voxel_grid,
            input,
            ..
        } = &node.data
        else {
            return None;
        };

        let threshold = voxel_grid.voxel_pitch() * 1.5;
        let padding = Vec3::splat(voxel_grid.voxel_pitch() * 2.0);
        let min_extent = Vec3::splat(voxel_grid.voxel_pitch() * 4.0);
        let mut found = false;
        let mut local_min = Vec3::splat(f32::MAX);
        let mut local_max = Vec3::splat(f32::MIN);

        for z in 0..voxel_grid.resolution {
            for y in 0..voxel_grid.resolution {
                for x in 0..voxel_grid.resolution {
                    let local_pos = voxel_grid.grid_to_world(x as f32, y as f32, z as f32);
                    let idx = VoxelGrid::index(x, y, z, voxel_grid.resolution);
                    let value = if voxel_grid.is_displacement {
                        if let Some(child_id) = input {
                            let world_pos =
                                *position + crate::sculpt::inverse_rotate_euler(local_pos, *rotation);
                            voxel::evaluate_sdf_tree(&self.doc.scene, *child_id, world_pos)
                                + voxel_grid.data[idx]
                        } else {
                            voxel_grid.data[idx]
                        }
                    } else {
                        voxel_grid.data[idx]
                    };

                    if value < 0.0 || value.abs() <= threshold {
                        found = true;
                        local_min = local_min.min(local_pos);
                        local_max = local_max.max(local_pos);
                    }
                }
            }
        }

        if !found {
            return None;
        }

        let mut bounds_min = local_min - padding;
        let mut bounds_max = local_max + padding;
        let current_center = (bounds_min + bounds_max) * 0.5;
        let current_extent = (bounds_max - bounds_min).max(min_extent);
        bounds_min = current_center - current_extent * 0.5;
        bounds_max = current_center + current_extent * 0.5;
        Some((bounds_min, bounds_max))
    }

    fn push_sculpt_toast(&mut self, message: &str, is_error: bool) {
        self.ui.toasts.push(super::Toast {
            message: message.to_string(),
            is_error,
            created: crate::compat::Instant::now(),
            duration: crate::compat::Duration::from_secs(4),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_detail_resolution_moves_up_and_down_within_limits() {
        assert_eq!(next_detail_resolution(64, true, 256), 80);
        assert_eq!(next_detail_resolution(64, false, 256), 51);
        assert_eq!(next_detail_resolution(16, false, 256), 16);
        assert_eq!(next_detail_resolution(250, true, 256), 256);
    }

    #[test]
    fn expanded_volume_bounds_preserve_center_and_grow_extent() {
        let bounds_min = Vec3::new(-1.0, -2.0, -3.0);
        let bounds_max = Vec3::new(1.0, 2.0, 3.0);
        let (expanded_min, expanded_max) = expanded_volume_bounds(bounds_min, bounds_max);

        assert_eq!((expanded_min + expanded_max) * 0.5, Vec3::ZERO);
        assert!((expanded_max.x - expanded_min.x) > (bounds_max.x - bounds_min.x));
        assert!((expanded_max.y - expanded_min.y) > (bounds_max.y - bounds_min.y));
        assert!((expanded_max.z - expanded_min.z) > (bounds_max.z - bounds_min.z));
    }

    #[test]
    fn remesh_resolution_for_detail_reports_cap_limiting() {
        let bounds_min = Vec3::splat(-2.0);
        let bounds_max = Vec3::splat(2.0);
        let target_pitch = 0.05;

        let (clamped_resolution, limited) =
            remesh_resolution_for_detail(bounds_min, bounds_max, target_pitch, 64);
        assert_eq!(clamped_resolution, 64);
        assert!(limited);

        let (full_resolution, limited) =
            remesh_resolution_for_detail(bounds_min, bounds_max, target_pitch, 256);
        assert!(full_resolution > 64);
        assert!(!limited);
    }
}
