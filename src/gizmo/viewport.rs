use std::collections::HashSet;
use std::fmt::Write;

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

use crate::gpu::camera::Camera;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::keymap::KeyboardModifiers;
use crate::settings::{GroupRotateDirection, SelectionBehaviorSettings, SnapConfig};

use super::selection::{
    apply_rotation_delta, collect_gizmo_selection, collect_transform_wrapper_targets,
    compute_axis_directions, gizmo_center_for_single_transform, multi_selection_axis_directions,
    set_node_position, set_node_rotation, set_node_scale, transform_for_node, GizmoSelection,
    GizmoTarget, NodeTransform,
};
use super::{GizmoMode, GizmoSpace, GizmoState};

const AXIS_LENGTH: f32 = 1.2;
const HIT_THRESHOLD: f32 = 12.0;
const ARROW_SIZE: f32 = 8.0;
const AXIS_STROKE_WIDTH: f32 = 2.5;
const SCALE_BOX_SIZE: f32 = 6.0;
const RING_RADIUS: f32 = 1.0;
const RING_SEGMENTS: usize = 48;
const RING_HIT_THRESHOLD: f32 = 14.0;
const TRANSLATE_SENSITIVITY: f32 = 0.003;
const SCALE_SENSITIVITY: f32 = 0.005;
const ROTATE_MIN_RADIUS_PX: f32 = 12.0;
const PIVOT_INDICATOR_RADIUS: f32 = 4.0;
const PIVOT_SEGMENTS: usize = 24;

const COLOR_X: [u8; 4] = [220, 60, 60, 255];
const COLOR_Y: [u8; 4] = [60, 200, 60, 255];
const COLOR_Z: [u8; 4] = [60, 100, 240, 255];
const COLOR_X_HOVER: [u8; 4] = [255, 120, 120, 255];
const COLOR_Y_HOVER: [u8; 4] = [120, 255, 120, 255];
const COLOR_Z_HOVER: [u8; 4] = [120, 160, 255, 255];
const PIVOT_COLOR: [u8; 4] = [255, 200, 50, 255];
const PIVOT_LINK_COLOR: [u8; 4] = [255, 200, 50, 100];

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    fn color(&self) -> [u8; 4] {
        match self {
            Self::X => COLOR_X,
            Self::Y => COLOR_Y,
            Self::Z => COLOR_Z,
        }
    }

    fn hover_color(&self) -> [u8; 4] {
        match self {
            Self::X => COLOR_X_HOVER,
            Self::Y => COLOR_Y_HOVER,
            Self::Z => COLOR_Z_HOVER,
        }
    }

    fn euler_index(&self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

const ALL_AXES: [GizmoAxis; 3] = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ScreenRotationDragState {
    previous_cursor_angle: Option<f32>,
    accumulated_angle: f32,
    applied_angle: f32,
    rotation_sign: f32,
}

impl ScreenRotationDragState {
    fn new(origin_screen: Vec2, pointer_screen: Option<Vec2>, rotation_sign: f32) -> Self {
        Self {
            previous_cursor_angle: pointer_screen
                .and_then(|pointer| pointer_angle_around_origin(origin_screen, pointer)),
            accumulated_angle: 0.0,
            applied_angle: 0.0,
            rotation_sign,
        }
    }

    fn update(&mut self, origin_screen: Vec2, pointer_screen: Vec2) -> Option<f32> {
        let current_angle = pointer_angle_around_origin(origin_screen, pointer_screen)?;
        let delta_angle = self
            .previous_cursor_angle
            .map(|previous| wrap_signed_angle_delta(current_angle - previous))
            .unwrap_or(0.0);
        self.previous_cursor_angle = Some(current_angle);
        self.accumulated_angle += delta_angle;
        Some(delta_angle)
    }

    #[cfg(test)]
    fn accumulated_angle(&self) -> f32 {
        self.accumulated_angle
    }

    fn target_angle(&self, snap_increment_radians: Option<f32>) -> f32 {
        let signed_accumulated = self.accumulated_angle * self.rotation_sign;
        snap_increment_radians
            .map(|snap| snap_value(signed_accumulated, snap))
            .unwrap_or(signed_accumulated)
    }

    fn consume_applied_delta(&mut self, snap_increment_radians: Option<f32>) -> f32 {
        let target_angle = self.target_angle(snap_increment_radians);
        let delta = target_angle - self.applied_angle;
        self.applied_angle = target_angle;
        delta
    }

    fn consume_applied_total(&mut self, snap_increment_radians: Option<f32>) -> f32 {
        let target_angle = self.target_angle(snap_increment_radians);
        self.applied_angle = target_angle;
        target_angle
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GizmoDragSession {
    start_center_world: Vec3,
    start_origin_screen: Vec2,
    axis_directions: [Vec3; 3],
    targets: Vec<GizmoTarget>,
    accumulated_drag_delta: Vec2,
    rotation_drag: ScreenRotationDragState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GizmoInputSnapshot {
    pub viewport_size_physical: [u32; 2],
    pub pointer_inside: bool,
    pub pointer_position_physical: Option<[f32; 2]>,
    pub pointer_delta_physical: [f32; 2],
    pub primary_down: bool,
    pub primary_pressed: bool,
    pub primary_released: bool,
    pub modifiers: KeyboardModifiers,
}

impl GizmoInputSnapshot {
    fn pointer_screen(self) -> Option<Vec2> {
        self.pointer_position_physical.map(Vec2::from_array)
    }

    fn pointer_delta(self) -> Vec2 {
        Vec2::from_array(self.pointer_delta_physical)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GizmoInteractionResult {
    pub consumed_pointer: bool,
    pub drag_active: bool,
    pub requested_transform_wrappers: Vec<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewportGizmoPath {
    pub commands: String,
    pub stroke_rgba: [u8; 4],
    pub fill_rgba: [u8; 4],
    pub stroke_width: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewportGizmoOverlay {
    pub viewbox_size: [f32; 2],
    pub paths: Vec<ViewportGizmoPath>,
}

struct ResolvedGizmoFrame {
    selection: Option<GizmoSelection>,
    multi_selection_active: bool,
    single_node_id: Option<NodeId>,
    single_transform: Option<NodeTransform>,
    center_world: Vec3,
    origin_screen: Vec2,
    axes: [Vec3; 3],
    axis_screens: [Vec2; 3],
    view_proj: Mat4,
    viewport_size_physical: [u32; 2],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_viewport_gizmo_overlay(
    camera: &Camera,
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
    gizmo_state: &GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: Vec3,
    viewport_size_physical: [u32; 2],
    selection_behavior: &SelectionBehaviorSettings,
    gizmo_visible: bool,
    input: Option<&GizmoInputSnapshot>,
) -> Option<ViewportGizmoOverlay> {
    let frame = resolve_gizmo_frame(
        camera,
        scene,
        selected,
        selected_set,
        gizmo_state,
        gizmo_mode,
        gizmo_space,
        pivot_offset,
        viewport_size_physical,
        selection_behavior,
        gizmo_visible,
    )?;

    let pointer_screen = input
        .copied()
        .filter(|snapshot| snapshot.pointer_inside)
        .and_then(GizmoInputSnapshot::pointer_screen);
    let hovered_axis = if active_axis(gizmo_state).is_none() {
        pointer_screen.and_then(|pointer| hover_axis_for_frame(pointer, gizmo_mode, &frame))
    } else {
        None
    };
    let active_axis = active_axis(gizmo_state);
    let colors = [
        axis_color(&GizmoAxis::X, active_axis.as_ref(), hovered_axis.as_ref()),
        axis_color(&GizmoAxis::Y, active_axis.as_ref(), hovered_axis.as_ref()),
        axis_color(&GizmoAxis::Z, active_axis.as_ref(), hovered_axis.as_ref()),
    ];

    let mut paths = Vec::new();
    match gizmo_mode {
        GizmoMode::Translate => {
            for (index, color) in colors.iter().enumerate() {
                paths.push(ViewportGizmoPath {
                    commands: line_path(frame.origin_screen, frame.axis_screens[index]),
                    stroke_rgba: *color,
                    fill_rgba: [0, 0, 0, 0],
                    stroke_width: AXIS_STROKE_WIDTH,
                });
                if let Some(commands) =
                    arrow_head_path(frame.origin_screen, frame.axis_screens[index])
                {
                    paths.push(ViewportGizmoPath {
                        commands,
                        stroke_rgba: [0, 0, 0, 0],
                        fill_rgba: *color,
                        stroke_width: 0.0,
                    });
                }
            }
        }
        GizmoMode::Scale => {
            for (index, color) in colors.iter().enumerate() {
                paths.push(ViewportGizmoPath {
                    commands: line_path(frame.origin_screen, frame.axis_screens[index]),
                    stroke_rgba: *color,
                    fill_rgba: [0, 0, 0, 0],
                    stroke_width: AXIS_STROKE_WIDTH,
                });
                paths.push(ViewportGizmoPath {
                    commands: square_path(frame.axis_screens[index], SCALE_BOX_SIZE),
                    stroke_rgba: *color,
                    fill_rgba: *color,
                    stroke_width: 1.0,
                });
            }
        }
        GizmoMode::Rotate => {
            for (index, color) in colors.iter().enumerate() {
                if let Some(commands) = ring_path(
                    frame.center_world,
                    frame.axes[index],
                    &frame.view_proj,
                    frame.viewport_size_physical,
                ) {
                    paths.push(ViewportGizmoPath {
                        commands,
                        stroke_rgba: *color,
                        fill_rgba: [0, 0, 0, 0],
                        stroke_width: AXIS_STROKE_WIDTH,
                    });
                }
            }
        }
    }

    if !frame.multi_selection_active && pivot_offset.length_squared() > 1e-6 {
        if let Some(single_transform) = frame.single_transform.as_ref() {
            if let Some(node_origin_screen) = world_to_screen(
                single_transform.position,
                &frame.view_proj,
                frame.viewport_size_physical,
            ) {
                paths.push(ViewportGizmoPath {
                    commands: line_path(node_origin_screen, frame.origin_screen),
                    stroke_rgba: PIVOT_LINK_COLOR,
                    fill_rgba: [0, 0, 0, 0],
                    stroke_width: 1.0,
                });
                paths.push(ViewportGizmoPath {
                    commands: circle_path(
                        node_origin_screen,
                        PIVOT_INDICATOR_RADIUS,
                        PIVOT_SEGMENTS,
                    ),
                    stroke_rgba: PIVOT_COLOR,
                    fill_rgba: [0, 0, 0, 0],
                    stroke_width: 1.5,
                });
            }
        }
    }

    Some(ViewportGizmoOverlay {
        viewbox_size: [
            frame.viewport_size_physical[0].max(1) as f32,
            frame.viewport_size_physical[1].max(1) as f32,
        ],
        paths,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_viewport_gizmo_interaction(
    input: &GizmoInputSnapshot,
    camera: &Camera,
    scene: &mut Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: &mut Vec3,
    snap_config: &SnapConfig,
    selection_behavior: &SelectionBehaviorSettings,
    gizmo_visible: bool,
) -> GizmoInteractionResult {
    if !gizmo_visible {
        *gizmo_state = GizmoState::Idle;
        return GizmoInteractionResult::default();
    }

    if matches!(gizmo_state, GizmoState::PendingStart { .. }) && !input.primary_down {
        *gizmo_state = GizmoState::Idle;
    }

    let Some(frame) = resolve_gizmo_frame(
        camera,
        scene,
        selected,
        selected_set,
        gizmo_state,
        gizmo_mode,
        gizmo_space,
        *pivot_offset,
        input.viewport_size_physical,
        selection_behavior,
        gizmo_visible,
    ) else {
        *gizmo_state = GizmoState::Idle;
        return GizmoInteractionResult::default();
    };

    let pointer_screen = input.pointer_screen();
    let hovered_axis = pointer_screen
        .filter(|_| active_axis(gizmo_state).is_none())
        .and_then(|pointer| hover_axis_for_frame(pointer, gizmo_mode, &frame));
    let mut result = GizmoInteractionResult::default();
    let mut started_drag_this_frame = false;

    if input.primary_pressed
        || matches!(gizmo_state, GizmoState::PendingStart { .. } if input.primary_down)
    {
        let axis_to_start = match gizmo_state {
            GizmoState::PendingStart { axis } => Some(axis.clone()),
            _ => hovered_axis.clone(),
        };

        if let Some(axis) = axis_to_start {
            if let Some(pending_targets) =
                pending_transform_wrapper_targets(scene, selected, selected_set)
            {
                *gizmo_state = GizmoState::PendingStart { axis };
                result.consumed_pointer = true;
                result.drag_active = true;
                result.requested_transform_wrappers = pending_targets;
                return result;
            }

            let axis_index = axis.euler_index();
            if frame.multi_selection_active {
                if let Some(selection) = frame.selection.as_ref() {
                    let drag_axes = multi_selection_axis_directions(
                        selection.reference_rotation_world,
                        selection_behavior,
                    );
                    let drag_center = selection.base_center_world;
                    let drag_origin_screen = world_to_screen(
                        drag_center,
                        &frame.view_proj,
                        frame.viewport_size_physical,
                    )
                    .unwrap_or(frame.origin_screen);
                    let drag_axis_dir = drag_axes[axis_index];
                    let rotation_sign = if matches!(gizmo_mode, GizmoMode::Rotate) {
                        rotation_drag_sign(
                            drag_axis_dir,
                            camera.eye() - drag_center,
                            true,
                            selection_behavior.group_rotate_direction,
                        )
                    } else {
                        1.0
                    };
                    *gizmo_state = GizmoState::DraggingMulti {
                        axis,
                        drag_session: build_drag_session(
                            selection,
                            drag_axes,
                            drag_origin_screen,
                            pointer_screen,
                            rotation_sign,
                        ),
                    };
                    result.consumed_pointer = true;
                    result.drag_active = true;
                    started_drag_this_frame = true;
                }
            } else if let Some(node_id) = frame.single_node_id {
                let rotation_sign = if matches!(gizmo_mode, GizmoMode::Rotate) {
                    rotation_drag_sign(
                        frame.axes[axis_index],
                        camera.eye() - frame.center_world,
                        false,
                        selection_behavior.group_rotate_direction,
                    )
                } else {
                    1.0
                };
                *gizmo_state = GizmoState::DraggingSingle {
                    axis,
                    node_id,
                    rotation_drag: ScreenRotationDragState::new(
                        frame.origin_screen,
                        pointer_screen,
                        rotation_sign,
                    ),
                };
                result.consumed_pointer = true;
                result.drag_active = true;
                started_drag_this_frame = true;
            }
        }
    }

    if input.primary_down && !started_drag_this_frame {
        match gizmo_state {
            GizmoState::DraggingSingle {
                axis,
                node_id,
                rotation_drag,
            } => {
                if let Some(single_transform) = frame.single_transform.as_ref() {
                    let axis_index = axis.euler_index();
                    let axis_dir = frame.axes[axis_index];
                    let allow_pivot_drag = input.modifiers.alt && !frame.multi_selection_active;

                    if allow_pivot_drag {
                        handle_pivot_drag(
                            input,
                            axis_dir,
                            frame.center_world,
                            frame.origin_screen,
                            camera,
                            &frame.view_proj,
                            frame.viewport_size_physical,
                            pivot_offset,
                            single_transform.rotation,
                        );
                    } else {
                        match gizmo_mode {
                            GizmoMode::Translate => {
                                handle_translate_drag(
                                    input,
                                    axis_dir,
                                    frame.center_world,
                                    frame.origin_screen,
                                    camera,
                                    &frame.view_proj,
                                    frame.viewport_size_physical,
                                    scene,
                                    *node_id,
                                );
                                if input.modifiers.ctrl {
                                    snap_position(
                                        scene,
                                        *node_id,
                                        axis_index,
                                        snap_config.translate_snap,
                                    );
                                }
                            }
                            GizmoMode::Scale => {
                                handle_scale_drag(
                                    input,
                                    axis,
                                    axis_dir,
                                    frame.center_world,
                                    frame.origin_screen,
                                    camera,
                                    &frame.view_proj,
                                    frame.viewport_size_physical,
                                    scene,
                                    *node_id,
                                    *pivot_offset,
                                    single_transform.rotation,
                                );
                                if input.modifiers.ctrl {
                                    snap_scale(scene, *node_id, axis_index, snap_config.scale_snap);
                                }
                            }
                            GizmoMode::Rotate => {
                                handle_rotate_drag(
                                    input,
                                    axis_dir,
                                    frame.origin_screen,
                                    rotation_drag,
                                    input
                                        .modifiers
                                        .ctrl
                                        .then_some(snap_config.rotate_snap.to_radians()),
                                    scene,
                                    *node_id,
                                    *pivot_offset,
                                    single_transform.rotation,
                                    gizmo_space,
                                );
                            }
                        }
                    }
                    result.consumed_pointer = true;
                    result.drag_active = true;
                }
            }
            GizmoState::DraggingMulti { axis, drag_session } => {
                let axis_index = axis.euler_index();
                let axis_dir = drag_session.axis_directions[axis_index];
                match gizmo_mode {
                    GizmoMode::Translate => handle_multi_translate_drag(
                        input,
                        drag_session,
                        axis_dir,
                        camera,
                        &frame.view_proj,
                        frame.viewport_size_physical,
                        scene,
                        snap_config,
                        input.modifiers.ctrl,
                    ),
                    GizmoMode::Rotate => handle_multi_rotate_drag(
                        input,
                        drag_session,
                        axis_dir,
                        scene,
                        snap_config,
                        input.modifiers.ctrl,
                    ),
                    GizmoMode::Scale => handle_multi_scale_drag(
                        input,
                        drag_session,
                        axis,
                        axis_dir,
                        &frame.view_proj,
                        frame.viewport_size_physical,
                        scene,
                        snap_config,
                        input.modifiers.ctrl,
                    ),
                }
                result.consumed_pointer = true;
                result.drag_active = true;
            }
            GizmoState::Idle | GizmoState::PendingStart { .. } => {}
        }
    }

    if input.primary_released && !matches!(gizmo_state, GizmoState::Idle) {
        *gizmo_state = GizmoState::Idle;
        result.consumed_pointer = true;
    }

    result
}

#[allow(clippy::too_many_arguments)]
fn resolve_gizmo_frame(
    camera: &Camera,
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
    gizmo_state: &GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: Vec3,
    viewport_size_physical: [u32; 2],
    selection_behavior: &SelectionBehaviorSettings,
    gizmo_visible: bool,
) -> Option<ResolvedGizmoFrame> {
    if !gizmo_visible {
        return None;
    }

    let width = viewport_size_physical[0].max(1) as f32;
    let height = viewport_size_physical[1].max(1) as f32;
    let aspect = width / height.max(1.0);
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    let mut selected_count = selected_set.len();
    if let Some(primary_selected) = selected {
        if !selected_set.contains(&primary_selected) {
            selected_count += 1;
        }
    }

    let multi_drag_active = matches!(gizmo_state, GizmoState::DraggingMulti { .. });
    let multi_selection_active = match gizmo_state {
        GizmoState::DraggingSingle { .. } => false,
        GizmoState::DraggingMulti { .. } => true,
        GizmoState::PendingStart { .. } | GizmoState::Idle => selected_count > 1,
    };

    let selection = if multi_selection_active {
        collect_gizmo_selection(scene, selected, selected_set, selection_behavior)
    } else {
        None
    };

    let (single_node_id, single_transform, center_world, axes) = if multi_selection_active {
        if let Some(selection) = selection.as_ref() {
            if matches!(gizmo_mode, GizmoMode::Scale)
                && !selection.supports_scale()
                && !multi_drag_active
            {
                return None;
            }
            (
                None,
                None,
                selection.base_center_world,
                multi_selection_axis_directions(
                    selection.reference_rotation_world,
                    selection_behavior,
                ),
            )
        } else if let GizmoState::DraggingMulti { drag_session, .. } = gizmo_state {
            (
                None,
                None,
                drag_session.start_center_world,
                drag_session.axis_directions,
            )
        } else {
            return None;
        }
    } else {
        let node_id = match gizmo_state {
            GizmoState::DraggingSingle { node_id, .. } => *node_id,
            _ => super::current_transform_target(scene, selected?),
        };
        let (position, rotation, has_scale) = transform_for_node(scene, node_id)?;
        if matches!(gizmo_mode, GizmoMode::Scale) && !has_scale {
            return None;
        }
        let transform = NodeTransform {
            position,
            rotation,
            scale: Vec3::ONE,
            has_scale,
        };
        (
            Some(node_id),
            Some(transform.clone()),
            gizmo_center_for_single_transform(&transform, pivot_offset),
            compute_axis_directions(rotation, gizmo_space),
        )
    };

    let origin_screen = world_to_screen(center_world, &view_proj, viewport_size_physical)?;
    let mut axis_screens = [origin_screen; 3];
    for (index, axis) in axes.iter().enumerate() {
        let end = center_world + *axis * AXIS_LENGTH;
        axis_screens[index] =
            world_to_screen(end, &view_proj, viewport_size_physical).unwrap_or(origin_screen);
    }

    Some(ResolvedGizmoFrame {
        selection,
        multi_selection_active,
        single_node_id,
        single_transform,
        center_world,
        origin_screen,
        axes,
        axis_screens,
        view_proj,
        viewport_size_physical,
    })
}

fn pending_transform_wrapper_targets(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
) -> Option<Vec<NodeId>> {
    let targets = collect_transform_wrapper_targets(scene, selected, selected_set);
    if targets.is_empty() {
        None
    } else {
        Some(targets)
    }
}

fn hover_axis_for_frame(
    pointer_screen: Vec2,
    gizmo_mode: &GizmoMode,
    frame: &ResolvedGizmoFrame,
) -> Option<GizmoAxis> {
    match gizmo_mode {
        GizmoMode::Translate | GizmoMode::Scale => hit_test_axes(
            pointer_screen,
            frame.origin_screen,
            &frame.axis_screens,
            HIT_THRESHOLD,
        ),
        GizmoMode::Rotate => hit_test_rings(
            pointer_screen,
            frame.center_world,
            &frame.axes,
            &frame.view_proj,
            frame.viewport_size_physical,
        ),
    }
}

fn active_axis(gizmo_state: &GizmoState) -> Option<GizmoAxis> {
    match gizmo_state {
        GizmoState::PendingStart { axis }
        | GizmoState::DraggingSingle { axis, .. }
        | GizmoState::DraggingMulti { axis, .. } => Some(axis.clone()),
        GizmoState::Idle => None,
    }
}

fn axis_color(
    axis: &GizmoAxis,
    active_axis: Option<&GizmoAxis>,
    hovered_axis: Option<&GizmoAxis>,
) -> [u8; 4] {
    if active_axis == Some(axis) || hovered_axis == Some(axis) {
        axis.hover_color()
    } else {
        axis.color()
    }
}

fn world_to_screen(
    world_pos: Vec3,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
) -> Option<Vec2> {
    let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
    if clip.w <= 0.0 || !clip.w.is_finite() {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() {
        return None;
    }
    let width = viewport_size_physical[0].max(1) as f32;
    let height = viewport_size_physical[1].max(1) as f32;
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * width,
        (-ndc.y * 0.5 + 0.5) * height,
    ))
}

fn point_to_segment_dist(point: Vec2, seg_start: Vec2, seg_end: Vec2) -> f32 {
    let ab = seg_end - seg_start;
    let ap = point - seg_start;
    let len_sq = ab.length_squared();
    if len_sq < 0.001 {
        return point.distance(seg_start);
    }
    let t = (ab.dot(ap) / len_sq).clamp(0.0, 1.0);
    point.distance(seg_start + ab * t)
}

fn pointer_angle_around_origin(origin_screen: Vec2, pointer_screen: Vec2) -> Option<f32> {
    let from_center = pointer_screen - origin_screen;
    if from_center.length_squared() < ROTATE_MIN_RADIUS_PX * ROTATE_MIN_RADIUS_PX {
        return None;
    }
    Some(from_center.y.atan2(from_center.x))
}

fn wrap_signed_angle_delta(angle_delta: f32) -> f32 {
    use std::f32::consts::{PI, TAU};

    let mut wrapped = angle_delta;
    while wrapped > PI {
        wrapped -= TAU;
    }
    while wrapped < -PI {
        wrapped += TAU;
    }
    wrapped
}

fn screen_axis_dir(
    origin: Vec3,
    axis_dir: Vec3,
    origin_screen: Vec2,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
) -> Vec2 {
    let end = world_to_screen(origin + axis_dir, view_proj, viewport_size_physical)
        .unwrap_or(origin_screen);
    let direction = end - origin_screen;
    let length = direction.length();
    if length > 0.1 {
        direction / length
    } else {
        Vec2::ZERO
    }
}

fn hit_test_axes(
    pointer_screen: Vec2,
    origin_screen: Vec2,
    axis_screens: &[Vec2; 3],
    threshold: f32,
) -> Option<GizmoAxis> {
    let dx = point_to_segment_dist(pointer_screen, origin_screen, axis_screens[0]);
    let dy = point_to_segment_dist(pointer_screen, origin_screen, axis_screens[1]);
    let dz = point_to_segment_dist(pointer_screen, origin_screen, axis_screens[2]);
    let min = dx.min(dy).min(dz);
    if min > threshold {
        None
    } else if (min - dx).abs() < 0.01 {
        Some(GizmoAxis::X)
    } else if (min - dy).abs() < 0.01 {
        Some(GizmoAxis::Y)
    } else {
        Some(GizmoAxis::Z)
    }
}

fn ring_tangent_bitangent(axis_dir: Vec3) -> (Vec3, Vec3) {
    let up = if axis_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = axis_dir.cross(up).normalize_or_zero();
    let bitangent = axis_dir.cross(tangent).normalize_or_zero();
    (tangent, bitangent)
}

fn hit_test_rings(
    pointer_screen: Vec2,
    center_world: Vec3,
    axes: &[Vec3; 3],
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
) -> Option<GizmoAxis> {
    let mut best_dist = f32::MAX;
    let mut best_axis = None;

    for (index, axis) in ALL_AXES.iter().enumerate() {
        let distance = ring_distance_to_point(
            pointer_screen,
            center_world,
            axes[index],
            view_proj,
            viewport_size_physical,
        );
        if distance < best_dist {
            best_dist = distance;
            best_axis = Some(axis.clone());
        }
    }

    if best_dist > RING_HIT_THRESHOLD {
        None
    } else {
        best_axis
    }
}

fn ring_distance_to_point(
    pointer_screen: Vec2,
    center_world: Vec3,
    axis_dir: Vec3,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
) -> f32 {
    let (tangent, bitangent) = ring_tangent_bitangent(axis_dir);
    let mut min_dist = f32::MAX;
    let mut previous: Option<Vec2> = None;

    for index in 0..=RING_SEGMENTS {
        let angle = index as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        let world_point =
            center_world + (tangent * angle.cos() + bitangent * angle.sin()) * RING_RADIUS;
        if let Some(screen_point) = world_to_screen(world_point, view_proj, viewport_size_physical)
        {
            if let Some(previous_point) = previous {
                min_dist = min_dist.min(point_to_segment_dist(
                    pointer_screen,
                    previous_point,
                    screen_point,
                ));
            }
            previous = Some(screen_point);
        } else {
            previous = None;
        }
    }

    min_dist
}

fn rotation_drag_sign(
    axis_dir: Vec3,
    view_dir: Vec3,
    multi_selection: bool,
    group_rotate_direction: GroupRotateDirection,
) -> f32 {
    let facing_sign = if axis_dir.dot(view_dir) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    if multi_selection {
        match group_rotate_direction {
            GroupRotateDirection::Standard => facing_sign,
            GroupRotateDirection::Inverted => -facing_sign,
        }
    } else {
        facing_sign
    }
}

fn build_drag_session(
    selection: &GizmoSelection,
    axis_directions: [Vec3; 3],
    origin_screen: Vec2,
    start_pointer_screen: Option<Vec2>,
    rotation_sign: f32,
) -> GizmoDragSession {
    GizmoDragSession {
        start_center_world: selection.base_center_world,
        start_origin_screen: origin_screen,
        axis_directions,
        targets: selection.targets.to_vec(),
        accumulated_drag_delta: Vec2::ZERO,
        rotation_drag: ScreenRotationDragState::new(
            origin_screen,
            start_pointer_screen,
            rotation_sign,
        ),
    }
}

fn apply_position_delta(scene: &mut Scene, node_id: NodeId, delta: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive { position, .. } => *position += delta,
        NodeData::Sculpt { position, .. } => *position += delta,
        NodeData::Transform { translation, .. } => *translation += delta,
        _ => {}
    }
}

fn apply_world_translation_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    world_delta: Vec3,
) {
    for target in targets {
        let new_world_position = target.world_position + world_delta;
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(new_world_position);
        set_node_position(scene, target.node_id, new_local_position);
    }
}

fn apply_world_rotation_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    pivot_world: Vec3,
    delta_quat: Quat,
) {
    for target in targets {
        let rotated_world_position =
            pivot_world + delta_quat * (target.world_position - pivot_world);
        let rotated_world_orientation = delta_quat * target.world_rotation;
        let local_world_rotation =
            target.parent_world_rotation.inverse() * rotated_world_orientation;
        let local_rotation_quat = local_world_rotation.inverse();
        let local_rotation =
            quat_to_euler_stable(local_rotation_quat, target.local_transform.rotation);
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(rotated_world_position);

        set_node_position(scene, target.node_id, new_local_position);
        set_node_rotation(scene, target.node_id, local_rotation);
    }
}

fn apply_axis_scale_to_targets(
    scene: &mut Scene,
    targets: &[GizmoTarget],
    pivot_world: Vec3,
    axis_dir: Vec3,
    axis: &GizmoAxis,
    factor: f32,
) {
    for target in targets {
        let offset = target.world_position - pivot_world;
        let parallel = axis_dir * offset.dot(axis_dir);
        let perpendicular = offset - parallel;
        let scaled_world_position = pivot_world + perpendicular + parallel * factor;
        let new_local_position = target
            .parent_world_inverse
            .transform_point3(scaled_world_position);
        set_node_position(scene, target.node_id, new_local_position);

        if !target.local_transform.has_scale {
            continue;
        }

        let mut scaled_local = target.local_transform.scale;
        match axis {
            GizmoAxis::X => scaled_local.x = (scaled_local.x * factor).max(0.01),
            GizmoAxis::Y => scaled_local.y = (scaled_local.y * factor).max(0.01),
            GizmoAxis::Z => scaled_local.z = (scaled_local.z * factor).max(0.01),
        }
        set_node_scale(scene, target.node_id, scaled_local);
    }
}

fn snap_target_scales(scene: &mut Scene, targets: &[GizmoTarget], axis_index: usize, snap: f32) {
    for target in targets {
        if target.local_transform.has_scale {
            snap_scale(scene, target.node_id, axis_index, snap);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_translate_drag(
    input: &GizmoInputSnapshot,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Vec2,
    camera: &Camera,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
    scene: &mut Scene,
    drag_node: NodeId,
) {
    let axis_screen_dir = screen_axis_dir(
        origin,
        axis_dir,
        origin_screen,
        view_proj,
        viewport_size_physical,
    );
    let projected = input.pointer_delta().dot(axis_screen_dir);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let world_delta = axis_dir * projected * world_scale;
    apply_position_delta(scene, drag_node, world_delta);
}

#[allow(clippy::too_many_arguments)]
fn handle_scale_drag(
    input: &GizmoInputSnapshot,
    axis: &GizmoAxis,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Vec2,
    _camera: &Camera,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
    scene: &mut Scene,
    drag_node: NodeId,
    pivot_offset: Vec3,
    node_rotation: Vec3,
) {
    let axis_screen_dir = screen_axis_dir(
        origin,
        axis_dir,
        origin_screen,
        view_proj,
        viewport_size_physical,
    );
    let projected = input.pointer_delta().dot(axis_screen_dir);
    let factor = 1.0 + projected * SCALE_SENSITIVITY;

    if let Some(node) = scene.nodes.get_mut(&drag_node) {
        let (scale, position) = match &mut node.data {
            NodeData::Primitive {
                scale, position, ..
            } => (scale, position),
            NodeData::Transform {
                scale, translation, ..
            } => (scale, translation),
            _ => return,
        };

        if pivot_offset.length_squared() > 1e-6 {
            let pivot_world = *position + inverse_rotate_euler(pivot_offset, node_rotation);
            let offset = *position - pivot_world;
            let mut scale_vector = Vec3::ONE;
            match axis {
                GizmoAxis::X => scale_vector.x = factor,
                GizmoAxis::Y => scale_vector.y = factor,
                GizmoAxis::Z => scale_vector.z = factor,
            }
            *position = pivot_world + offset * scale_vector;
        }

        match axis {
            GizmoAxis::X => scale.x = (scale.x * factor).max(0.01),
            GizmoAxis::Y => scale.y = (scale.y * factor).max(0.01),
            GizmoAxis::Z => scale.z = (scale.z * factor).max(0.01),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_rotate_drag(
    input: &GizmoInputSnapshot,
    axis_dir: Vec3,
    origin_screen: Vec2,
    rotation_drag: &mut ScreenRotationDragState,
    snap_increment_radians: Option<f32>,
    scene: &mut Scene,
    drag_node: NodeId,
    pivot_offset: Vec3,
    node_rotation: Vec3,
    gizmo_space: &GizmoSpace,
) {
    let Some(pointer_screen) = input.pointer_screen() else {
        return;
    };
    if rotation_drag
        .update(origin_screen, pointer_screen)
        .is_none()
    {
        return;
    }
    let angle_delta = rotation_drag.consume_applied_delta(snap_increment_radians);
    if angle_delta.abs() <= f32::EPSILON {
        return;
    }

    let delta_quaternion = Quat::from_axis_angle(axis_dir, angle_delta);
    if let Some(node) = scene.nodes.get_mut(&drag_node) {
        let (rotation, position) = match &mut node.data {
            NodeData::Primitive {
                rotation, position, ..
            } => (rotation, position),
            NodeData::Sculpt {
                rotation, position, ..
            } => (rotation, position),
            NodeData::Transform {
                rotation,
                translation,
                ..
            } => (rotation, translation),
            _ => return,
        };
        if pivot_offset.length_squared() > 1e-6 {
            let pivot_world = *position + inverse_rotate_euler(pivot_offset, node_rotation);
            let offset = *position - pivot_world;
            *position = pivot_world + delta_quaternion * offset;
        }
        apply_rotation_delta(rotation, angle_delta, axis_dir, gizmo_space);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_multi_translate_drag(
    input: &GizmoInputSnapshot,
    drag_session: &mut GizmoDragSession,
    axis_dir: Vec3,
    camera: &Camera,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
    scene: &mut Scene,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let axis_screen_dir = screen_axis_dir(
        drag_session.start_center_world,
        axis_dir,
        drag_session.start_origin_screen,
        view_proj,
        viewport_size_physical,
    );
    drag_session.accumulated_drag_delta += input.pointer_delta();
    let projected = drag_session.accumulated_drag_delta.dot(axis_screen_dir);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let mut projected_world = projected * world_scale;
    if ctrl_held {
        projected_world = snap_value(projected_world, snap_config.translate_snap);
    }
    apply_world_translation_to_targets(scene, &drag_session.targets, axis_dir * projected_world);
}

fn handle_multi_rotate_drag(
    input: &GizmoInputSnapshot,
    drag_session: &mut GizmoDragSession,
    axis_dir: Vec3,
    scene: &mut Scene,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let Some(pointer_screen) = input.pointer_screen() else {
        return;
    };
    if drag_session
        .rotation_drag
        .update(drag_session.start_origin_screen, pointer_screen)
        .is_none()
    {
        return;
    }
    let snap_increment_radians = ctrl_held.then_some(snap_config.rotate_snap.to_radians());
    let total_angle = drag_session
        .rotation_drag
        .consume_applied_total(snap_increment_radians);
    if total_angle.abs() <= f32::EPSILON {
        return;
    }

    let delta_quaternion = Quat::from_axis_angle(axis_dir, total_angle);
    apply_world_rotation_to_targets(
        scene,
        &drag_session.targets,
        drag_session.start_center_world,
        delta_quaternion,
    );
}

#[allow(clippy::too_many_arguments)]
fn handle_multi_scale_drag(
    input: &GizmoInputSnapshot,
    drag_session: &mut GizmoDragSession,
    axis: &GizmoAxis,
    axis_dir: Vec3,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
    scene: &mut Scene,
    snap_config: &SnapConfig,
    ctrl_held: bool,
) {
    let axis_screen_dir = screen_axis_dir(
        drag_session.start_center_world,
        axis_dir,
        drag_session.start_origin_screen,
        view_proj,
        viewport_size_physical,
    );
    drag_session.accumulated_drag_delta += input.pointer_delta();
    let projected = drag_session.accumulated_drag_delta.dot(axis_screen_dir);
    let factor = 1.0 + projected * SCALE_SENSITIVITY;

    apply_axis_scale_to_targets(
        scene,
        &drag_session.targets,
        drag_session.start_center_world,
        axis_dir,
        axis,
        factor,
    );
    if ctrl_held {
        snap_target_scales(
            scene,
            &drag_session.targets,
            axis.euler_index(),
            snap_config.scale_snap,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_pivot_drag(
    input: &GizmoInputSnapshot,
    axis_dir: Vec3,
    origin: Vec3,
    origin_screen: Vec2,
    camera: &Camera,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
    pivot_offset: &mut Vec3,
    node_rotation: Vec3,
) {
    let axis_screen_dir = screen_axis_dir(
        origin,
        axis_dir,
        origin_screen,
        view_proj,
        viewport_size_physical,
    );
    let projected = input.pointer_delta().dot(axis_screen_dir);
    let world_scale = camera.distance * TRANSLATE_SENSITIVITY;
    let world_delta = axis_dir * projected * world_scale;
    *pivot_offset += inverse_rotate_euler(world_delta, node_rotation);
}

fn snap_value(value: f32, snap: f32) -> f32 {
    (value / snap).round() * snap
}

fn snap_position(scene: &mut Scene, node_id: NodeId, axis_index: usize, snap: f32) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    let position = match &mut node.data {
        NodeData::Primitive { position, .. } => position,
        NodeData::Sculpt { position, .. } => position,
        NodeData::Transform { translation, .. } => translation,
        _ => return,
    };
    match axis_index {
        0 => position.x = snap_value(position.x, snap),
        1 => position.y = snap_value(position.y, snap),
        _ => position.z = snap_value(position.z, snap),
    }
}

fn snap_scale(scene: &mut Scene, node_id: NodeId, axis_index: usize, snap: f32) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    let scale = match &mut node.data {
        NodeData::Primitive { scale, .. } => scale,
        NodeData::Transform { scale, .. } => scale,
        _ => return,
    };
    match axis_index {
        0 => scale.x = snap_value(scale.x, snap).max(0.01),
        1 => scale.y = snap_value(scale.y, snap).max(0.01),
        _ => scale.z = snap_value(scale.z, snap).max(0.01),
    }
}

fn inverse_rotate_euler(point: Vec3, rotation: Vec3) -> Vec3 {
    let mut rotated = point;
    let (sz, cz) = rotation.z.sin_cos();
    rotated = Vec3::new(
        cz * rotated.x + sz * rotated.y,
        -sz * rotated.x + cz * rotated.y,
        rotated.z,
    );
    let (sy, cy) = rotation.y.sin_cos();
    rotated = Vec3::new(
        cy * rotated.x - sy * rotated.z,
        rotated.y,
        sy * rotated.x + cy * rotated.z,
    );
    let (sx, cx) = rotation.x.sin_cos();
    Vec3::new(
        rotated.x,
        cx * rotated.y + sx * rotated.z,
        -sx * rotated.y + cx * rotated.z,
    )
}

fn euler_to_quat(rotation: Vec3) -> Quat {
    Quat::from_euler(glam::EulerRot::ZYX, rotation.z, rotation.y, rotation.x)
}

fn quat_to_euler_stable(quaternion: Quat, previous: Vec3) -> Vec3 {
    use std::f32::consts::{PI, TAU};

    fn wrap_near(angle: f32, reference: f32) -> f32 {
        let mut wrapped = angle;
        while wrapped - reference > PI {
            wrapped -= TAU;
        }
        while wrapped - reference < -PI {
            wrapped += TAU;
        }
        wrapped
    }

    fn normalize_near(value: Vec3, previous: Vec3) -> Vec3 {
        Vec3::new(
            wrap_near(value.x, previous.x),
            wrap_near(value.y, previous.y),
            wrap_near(value.z, previous.z),
        )
    }

    let (rotation_z, rotation_y, rotation_x) = quaternion.to_euler(glam::EulerRot::ZYX);
    let primary = normalize_near(Vec3::new(rotation_x, rotation_y, rotation_z), previous);
    let alternate = normalize_near(
        Vec3::new(rotation_x + PI, PI - rotation_y, rotation_z + PI),
        previous,
    );

    if (primary - previous).length_squared() <= (alternate - previous).length_squared() {
        primary
    } else {
        alternate
    }
}

fn svg_move_to(buffer: &mut String, point: Vec2) {
    let _ = write!(buffer, "M {:.3} {:.3}", point.x, point.y);
}

fn svg_line_to(buffer: &mut String, point: Vec2) {
    let _ = write!(buffer, " L {:.3} {:.3}", point.x, point.y);
}

fn line_path(start: Vec2, end: Vec2) -> String {
    let mut commands = String::new();
    svg_move_to(&mut commands, start);
    svg_line_to(&mut commands, end);
    commands
}

fn polygon_path(points: &[Vec2]) -> String {
    let mut commands = String::new();
    if let Some(first) = points.first().copied() {
        svg_move_to(&mut commands, first);
        for point in points.iter().copied().skip(1) {
            svg_line_to(&mut commands, point);
        }
        commands.push_str(" Z");
    }
    commands
}

fn arrow_head_path(start: Vec2, end: Vec2) -> Option<String> {
    let direction = end - start;
    let length = direction.length();
    if length < 1.0 {
        return None;
    }
    let direction = direction / length;
    let perpendicular = Vec2::new(-direction.y, direction.x);
    let tip = end;
    let left = end - direction * ARROW_SIZE + perpendicular * ARROW_SIZE * 0.5;
    let right = end - direction * ARROW_SIZE - perpendicular * ARROW_SIZE * 0.5;
    Some(polygon_path(&[tip, left, right]))
}

fn square_path(center: Vec2, size: f32) -> String {
    let half = size * 0.5;
    polygon_path(&[
        center + Vec2::new(-half, -half),
        center + Vec2::new(half, -half),
        center + Vec2::new(half, half),
        center + Vec2::new(-half, half),
    ])
}

fn circle_path(center: Vec2, radius: f32, segments: usize) -> String {
    let mut points = Vec::with_capacity(segments);
    for index in 0..segments {
        let angle = index as f32 / segments as f32 * std::f32::consts::TAU;
        points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
    }
    polygon_path(&points)
}

fn ring_path(
    center_world: Vec3,
    axis_dir: Vec3,
    view_proj: &Mat4,
    viewport_size_physical: [u32; 2],
) -> Option<String> {
    let (tangent, bitangent) = ring_tangent_bitangent(axis_dir);
    let mut commands = String::new();
    let mut previous_visible = false;

    for index in 0..=RING_SEGMENTS {
        let angle = index as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        let world_point =
            center_world + (tangent * angle.cos() + bitangent * angle.sin()) * RING_RADIUS;
        if let Some(screen_point) = world_to_screen(world_point, view_proj, viewport_size_physical)
        {
            if previous_visible {
                svg_line_to(&mut commands, screen_point);
            } else {
                if !commands.is_empty() {
                    commands.push(' ');
                }
                svg_move_to(&mut commands, screen_point);
            }
            previous_visible = true;
        } else {
            previous_visible = false;
        }
    }

    if commands.is_empty() {
        None
    } else {
        Some(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::SdfPrimitive;
    use std::collections::{HashMap, HashSet};

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        }
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let delta = actual - expected;
        assert!(
            delta.length() < 1e-4,
            "expected {expected:?}, got {actual:?}, delta={delta:?}"
        );
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        let delta = (actual - expected).abs();
        assert!(
            delta < 1e-4,
            "expected {expected}, got {actual}, delta={delta}"
        );
    }

    fn primitive_position(scene: &Scene, node_id: NodeId) -> Vec3 {
        match &scene.nodes.get(&node_id).unwrap().data {
            NodeData::Primitive { position, .. } => *position,
            other => panic!("expected primitive, got {other:?}"),
        }
    }

    #[test]
    fn group_translation_moves_all_selected_targets() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Sphere);

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *position = Vec3::new(-1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }

        let selection = collect_gizmo_selection(
            &scene,
            Some(left),
            &HashSet::from([left, right]),
            &SelectionBehaviorSettings::default(),
        )
        .unwrap();
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Vec2::ZERO,
            None,
            1.0,
        );

        apply_world_translation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::new(0.0, 2.0, 0.0),
        );

        assert_vec3_close(primitive_position(&scene, left), Vec3::new(-1.0, 2.0, 0.0));
        assert_vec3_close(primitive_position(&scene, right), Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn group_rotation_uses_shared_pivot() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Sphere);

        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *position = Vec3::new(-1.0, 0.0, 0.0);
        }
        if let Some(NodeData::Primitive { position, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *position = Vec3::new(1.0, 0.0, 0.0);
        }

        let selection = collect_gizmo_selection(
            &scene,
            Some(left),
            &HashSet::from([left, right]),
            &SelectionBehaviorSettings::default(),
        )
        .unwrap();
        let drag_session = build_drag_session(
            &selection,
            [Vec3::X, Vec3::Y, Vec3::Z],
            Vec2::ZERO,
            None,
            1.0,
        );

        apply_world_rotation_to_targets(
            &mut scene,
            &drag_session.targets,
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        );

        assert_vec3_close(primitive_position(&scene, left), Vec3::new(0.0, -1.0, 0.0));
        assert_vec3_close(primitive_position(&scene, right), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn rotation_drag_sign_respects_group_direction_setting() {
        assert_eq!(
            rotation_drag_sign(Vec3::Z, Vec3::Z, false, GroupRotateDirection::Standard),
            1.0
        );
        assert_eq!(
            rotation_drag_sign(Vec3::Z, Vec3::Z, true, GroupRotateDirection::Standard),
            1.0
        );
        assert_eq!(
            rotation_drag_sign(Vec3::Z, Vec3::Z, true, GroupRotateDirection::Inverted),
            -1.0
        );
        assert_eq!(
            rotation_drag_sign(Vec3::Z, -Vec3::Z, true, GroupRotateDirection::Standard),
            -1.0
        );
        assert_eq!(
            rotation_drag_sign(Vec3::Z, -Vec3::Z, true, GroupRotateDirection::Inverted),
            1.0
        );
    }

    #[test]
    fn screen_rotation_drag_tracks_full_turn_without_depth_terms() {
        let origin = Vec2::ZERO;
        let radius = 100.0;
        let mut drag = ScreenRotationDragState::new(origin, Some(Vec2::new(radius, 0.0)), 1.0);

        for angle in [
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
            0.0,
        ] {
            let pointer = Vec2::new(radius * angle.cos(), radius * angle.sin());
            let _ = drag.update(origin, pointer);
        }

        assert_f32_close(drag.accumulated_angle(), std::f32::consts::TAU);
    }

    #[test]
    fn snapped_rotation_consumes_incremental_applied_delta() {
        let origin = Vec2::ZERO;
        let radius = 100.0;
        let mut drag = ScreenRotationDragState::new(origin, Some(Vec2::new(radius, 0.0)), 1.0);
        let snap = 15.0_f32.to_radians();

        let pointer_a = Vec2::new(
            radius * (10.0_f32.to_radians()).cos(),
            radius * (10.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_a);
        let delta_a = drag.consume_applied_delta(Some(snap));
        assert_f32_close(delta_a, 15.0_f32.to_radians());

        let pointer_b = Vec2::new(
            radius * (14.0_f32.to_radians()).cos(),
            radius * (14.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_b);
        let delta_b = drag.consume_applied_delta(Some(snap));
        assert_f32_close(delta_b, 0.0);

        let pointer_c = Vec2::new(
            radius * (28.0_f32.to_radians()).cos(),
            radius * (28.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_c);
        let delta_c = drag.consume_applied_delta(Some(snap));
        assert_f32_close(delta_c, 15.0_f32.to_radians());
    }

    #[test]
    fn snapped_rotation_consumes_total_applied_angle_for_baseline_flows() {
        let origin = Vec2::ZERO;
        let radius = 100.0;
        let mut drag = ScreenRotationDragState::new(origin, Some(Vec2::new(radius, 0.0)), 1.0);
        let snap = 15.0_f32.to_radians();

        let pointer_a = Vec2::new(
            radius * (10.0_f32.to_radians()).cos(),
            radius * (10.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_a);
        let total_a = drag.consume_applied_total(Some(snap));
        assert_f32_close(total_a, 15.0_f32.to_radians());

        let pointer_b = Vec2::new(
            radius * (14.0_f32.to_radians()).cos(),
            radius * (14.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_b);
        let total_b = drag.consume_applied_total(Some(snap));
        assert_f32_close(total_b, 15.0_f32.to_radians());

        let pointer_c = Vec2::new(
            radius * (28.0_f32.to_radians()).cos(),
            radius * (28.0_f32.to_radians()).sin(),
        );
        let _ = drag.update(origin, pointer_c);
        let total_c = drag.consume_applied_total(Some(snap));
        assert_f32_close(total_c, 30.0_f32.to_radians());
    }
}
