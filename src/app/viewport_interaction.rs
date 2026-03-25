use glam::{Vec3, Vec4};

use crate::app::actions::{Action, ActionSink};
use crate::app::state::{ViewportInteractionState, ViewportPrimaryDragMode};
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{NodeData, Scene};
use crate::keymap::KeyboardModifiers;
use crate::sculpt::{self, SculptState};
use crate::settings::RenderConfig;

use super::backend_frame::ViewportUiFeedback;
use super::SdfApp;

const CLICK_DISTANCE_THRESHOLD: f32 = 6.0;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PointerButtonSnapshot {
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewportInputSnapshot {
    pub viewport_size_physical: [u32; 2],
    pub pixels_per_point: f32,
    pub now_seconds: f64,
    pub pointer_inside: bool,
    pub pointer_position_physical: Option<[f32; 2]>,
    pub pointer_delta_physical: [f32; 2],
    pub wheel_delta_logical: [f32; 2],
    pub primary: PointerButtonSnapshot,
    pub secondary: PointerButtonSnapshot,
    pub middle: PointerButtonSnapshot,
    pub modifiers: KeyboardModifiers,
    pub pressure: f32,
    pub double_clicked: bool,
}

impl Default for ViewportInputSnapshot {
    fn default() -> Self {
        Self {
            viewport_size_physical: [1, 1],
            pixels_per_point: 1.0,
            now_seconds: 0.0,
            pointer_inside: false,
            pointer_position_physical: None,
            pointer_delta_physical: [0.0, 0.0],
            wheel_delta_logical: [0.0, 0.0],
            primary: PointerButtonSnapshot::default(),
            secondary: PointerButtonSnapshot::default(),
            middle: PointerButtonSnapshot::default(),
            modifiers: KeyboardModifiers::default(),
            pressure: 0.0,
            double_clicked: false,
        }
    }
}

pub(crate) struct ViewportInteractionContext<'a> {
    pub state: &'a mut ViewportInteractionState,
    pub camera: &'a mut Camera,
    pub scene: &'a Scene,
    pub sculpt_state: &'a SculptState,
    pub last_sculpt_hit: Option<Vec3>,
    pub render_config: &'a RenderConfig,
    pub allow_selection_pick: bool,
}

impl SdfApp {
    pub(super) fn run_viewport_interaction(
        &mut self,
        input: &ViewportInputSnapshot,
        actions: &mut ActionSink,
    ) -> ViewportUiFeedback {
        run_viewport_interaction_core(
            ViewportInteractionContext {
                state: &mut self.ui.viewport_interaction,
                camera: &mut self.doc.camera,
                scene: &self.doc.scene,
                sculpt_state: &self.doc.sculpt_state,
                last_sculpt_hit: self.async_state.last_sculpt_hit,
                render_config: &self.settings.render,
                allow_selection_pick: true,
            },
            input,
            actions,
        )
    }
}

pub(crate) fn run_viewport_interaction_core(
    context: ViewportInteractionContext<'_>,
    input: &ViewportInputSnapshot,
    actions: &mut ActionSink,
) -> ViewportUiFeedback {
    let ViewportInteractionContext {
        state,
        camera,
        scene,
        sculpt_state,
        last_sculpt_hit,
        render_config,
        allow_selection_pick,
    } = context;
    let mut feedback = ViewportUiFeedback::default();
    let sculpt_active = sculpt_state.is_active();
    let pointer_pos = input.pointer_position_physical;
    let pointer_delta_logical = [
        input.pointer_delta_physical[0] / input.pixels_per_point.max(1.0),
        input.pointer_delta_physical[1] / input.pixels_per_point.max(1.0),
    ];

    if input.primary.pressed {
        state.primary_press_origin_physical = pointer_pos;
        state.primary_drag_distance = 0.0;
        state.primary_drag_mode = if sculpt_active {
            let stroke_confirmed = last_sculpt_hit.is_some();
            match pointer_pos {
                Some(pos)
                    if !in_safety_border(
                        pos,
                        input.viewport_size_physical,
                        render_config.sculpt_safety_border,
                    ) && (stroke_confirmed
                        || cursor_in_sculpt_bounds(
                            pos,
                            sculpt_state,
                            scene,
                            camera,
                            input.viewport_size_physical,
                        )) =>
                {
                    ViewportPrimaryDragMode::Sculpt
                }
                _ => ViewportPrimaryDragMode::Orbit,
            }
        } else {
            ViewportPrimaryDragMode::Orbit
        };
    }

    if input.primary.down {
        state.primary_drag_distance += glam::Vec2::from_array(input.pointer_delta_physical).length();
    }

    if input.pointer_inside && input.wheel_delta_logical[1] != 0.0 {
        camera.zoom(input.wheel_delta_logical[1]);
    }

    if input.middle.down
        && (pointer_delta_logical[0] != 0.0 || pointer_delta_logical[1] != 0.0)
    {
        apply_orbit_drag(camera, render_config, pointer_delta_logical, input.modifiers);
    }

    if input.secondary.down
        && (pointer_delta_logical[0] != 0.0 || pointer_delta_logical[1] != 0.0)
    {
        if sculpt_active && input.modifiers.ctrl {
            feedback.brush_radius_delta = pointer_delta_logical[0] * 0.005 * camera.distance;
            feedback.brush_strength_delta = -pointer_delta_logical[1] * 0.002;
        } else {
            camera.pan(pointer_delta_logical[0], pointer_delta_logical[1]);
        }
    }

    if sculpt_active {
        if input.double_clicked {
            if let Some(pos) = pointer_pos {
                if in_safety_border(
                    pos,
                    input.viewport_size_physical,
                    render_config.sculpt_safety_border,
                ) {
                    actions.push(Action::FrameAll);
                }
            }
        }

        if input.primary.down
            && matches!(state.primary_drag_mode, ViewportPrimaryDragMode::Orbit)
            && (pointer_delta_logical[0] != 0.0 || pointer_delta_logical[1] != 0.0)
        {
            apply_orbit_drag(camera, render_config, pointer_delta_logical, input.modifiers);
        } else if input.primary.down
            && matches!(state.primary_drag_mode, ViewportPrimaryDragMode::Sculpt)
            && input.pointer_inside
        {
            if let Some(pos) = pointer_pos {
                feedback.pending_pick = Some(PendingPick {
                    mouse_pos: pos,
                    camera_uniform: build_pick_uniform(
                        camera,
                        scene,
                        input.viewport_size_physical,
                        input.now_seconds as f32,
                    ),
                    additive_select_held: false,
                });
                feedback.sculpt_ctrl_held = input.modifiers.ctrl;
                feedback.sculpt_shift_held = input.modifiers.shift;
                feedback.sculpt_pressure = input.pressure;
            }
        } else if !input.primary.down && input.pointer_inside {
            if let Some(pos) = pointer_pos {
                feedback.pending_pick = Some(PendingPick {
                    mouse_pos: pos,
                    camera_uniform: build_pick_uniform(
                        camera,
                        scene,
                        input.viewport_size_physical,
                        input.now_seconds as f32,
                    ),
                    additive_select_held: false,
                });
                feedback.is_hover_pick = true;
            }
        }
    } else {
        if input.primary.down
            && (pointer_delta_logical[0] != 0.0 || pointer_delta_logical[1] != 0.0)
        {
            apply_orbit_drag(camera, render_config, pointer_delta_logical, input.modifiers);
        }

        if allow_selection_pick
            && input.primary.released
            && state.primary_drag_distance <= CLICK_DISTANCE_THRESHOLD
            && input.pointer_inside
        {
            if let Some(pos) = pointer_pos {
                feedback.pending_pick = Some(PendingPick {
                    mouse_pos: pos,
                    camera_uniform: build_pick_uniform(
                        camera,
                        scene,
                        input.viewport_size_physical,
                        input.now_seconds as f32,
                    ),
                    additive_select_held: input.modifiers.shift,
                });
            }
        }
    }

    if input.primary.released {
        state.primary_press_origin_physical = None;
        state.primary_drag_distance = 0.0;
        state.primary_drag_mode = ViewportPrimaryDragMode::None;
    }

    state.last_pointer_pos_physical = pointer_pos;

    feedback
}

fn apply_orbit_drag(
    camera: &mut Camera,
    render_config: &RenderConfig,
    delta_logical: [f32; 2],
    modifiers: KeyboardModifiers,
) {
    if modifiers.ctrl && modifiers.alt {
        let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
        camera.roll_by(sign * delta_logical[0], render_config.roll_sensitivity);
    } else {
        camera.orbit(delta_logical[0], delta_logical[1]);
        if render_config.clamp_orbit_pitch {
            camera.clamp_pitch();
        }
    }
}

fn build_pick_uniform(
    camera: &Camera,
    scene: &Scene,
    viewport_size_physical: [u32; 2],
    time: f32,
) -> CameraUniform {
    let scene_bounds = scene.compute_bounds();
    camera.to_uniform(
        [
            0.0,
            0.0,
            viewport_size_physical[0].max(1) as f32,
            viewport_size_physical[1].max(1) as f32,
        ],
        time,
        0.0,
        false,
        scene_bounds,
        -1.0,
        0.0,
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [[0.0; 4]; 32],
        [[0.0; 4]; 8],
    )
}

pub(crate) fn in_safety_border(
    pointer_pos_physical: [f32; 2],
    viewport_size_physical: [u32; 2],
    fraction: f32,
) -> bool {
    if fraction <= 0.0 {
        return false;
    }
    let width = viewport_size_physical[0].max(1) as f32;
    let height = viewport_size_physical[1].max(1) as f32;
    let border = width.min(height) * fraction;
    pointer_pos_physical[0] < border
        || pointer_pos_physical[0] > width - border
        || pointer_pos_physical[1] < border
        || pointer_pos_physical[1] > height - border
}

fn ray_aabb(origin: Vec3, dir: Vec3, box_min: Vec3, box_max: Vec3) -> (f32, f32) {
    let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    (t_enter, t_exit)
}

pub(crate) fn cursor_in_sculpt_bounds(
    pointer_pos_physical: [f32; 2],
    sculpt_state: &SculptState,
    scene: &Scene,
    camera: &Camera,
    viewport_size_physical: [u32; 2],
) -> bool {
    use crate::graph::voxel;

    let node_id = match sculpt_state {
        SculptState::Active { node_id, .. } => *node_id,
        _ => return false,
    };
    let node = match scene.nodes.get(&node_id) {
        Some(node) => node,
        None => return false,
    };
    let (position, rotation, voxel_grid, input_child) = match &node.data {
        NodeData::Sculpt {
            position,
            rotation,
            voxel_grid,
            input,
            ..
        } => (*position, *rotation, voxel_grid, *input),
        _ => return false,
    };

    let viewport_width = viewport_size_physical[0].max(1) as f32;
    let viewport_height = viewport_size_physical[1].max(1) as f32;
    let ndc_x = pointer_pos_physical[0] / viewport_width * 2.0 - 1.0;
    let ndc_y = 1.0 - pointer_pos_physical[1] / viewport_height * 2.0;
    let aspect = viewport_width / viewport_height.max(1.0);
    let inv_vp = (camera.projection_matrix(aspect) * camera.view_matrix()).inverse();
    let near = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let ray_origin = near.truncate() / near.w;
    let ray_dir = ((far.truncate() / far.w) - ray_origin).normalize_or_zero();
    if ray_dir.length_squared() <= 1e-8 {
        return false;
    }

    let local_origin = sculpt::inverse_rotate_euler(ray_origin - position, rotation);
    let local_dir = sculpt::inverse_rotate_euler(ray_dir, rotation).normalize_or_zero();
    let (t_enter, t_exit) = ray_aabb(
        local_origin,
        local_dir,
        voxel_grid.bounds_min,
        voxel_grid.bounds_max,
    );
    if t_enter >= t_exit {
        return false;
    }

    let cell_size = (voxel_grid.bounds_max - voxel_grid.bounds_min) / voxel_grid.resolution as f32;
    let min_step = cell_size.x.min(cell_size.y).min(cell_size.z);
    let threshold = min_step * 0.5;
    let mut t = t_enter.max(0.0);
    while t < t_exit {
        let local_pos = local_origin + local_dir * t;
        let displacement = voxel_grid.sample(local_pos);
        let sdf = if voxel_grid.is_displacement {
            if let Some(child_id) = input_child {
                let world_pos = position + sculpt::inverse_rotate_euler(local_pos, rotation);
                voxel::evaluate_sdf_tree(scene, child_id, world_pos) + displacement
            } else {
                displacement
            }
        } else {
            displacement
        };

        if sdf <= threshold {
            return true;
        }
        t += min_step.max(sdf.abs() * 0.9);
    }

    false
}
