#![allow(dead_code)]

use std::collections::HashSet;

use glam::{Mat4, Quat, Vec3};

use crate::graph::presented_object::{current_transform_owner, resolve_presented_object};
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::{MultiAxisOrientation, MultiPivotMode, SelectionBehaviorSettings};

use super::GizmoSpace;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeTransform {
    pub(crate) position: Vec3,
    pub(crate) rotation: Vec3,
    pub(crate) scale: Vec3,
    pub(crate) has_scale: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GizmoTarget {
    pub(crate) node_id: NodeId,
    pub(crate) local_transform: NodeTransform,
    pub(crate) parent_world_inverse: Mat4,
    pub(crate) parent_world_rotation: Quat,
    pub(crate) world_position: Vec3,
    pub(crate) world_rotation: Quat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GizmoSelection {
    pub(crate) targets: Vec<GizmoTarget>,
    pub(crate) base_center_world: Vec3,
    pub(crate) reference_rotation_world: Quat,
    pub(crate) reference_target_id: NodeId,
    pub(crate) pivot_mode: MultiPivotMode,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MultiTransformReadout {
    pub position_delta: Vec3,
    pub rotation_delta_rad: Vec3,
    pub scale_factor: Vec3,
    pub scale_enabled: bool,
}

impl Default for GizmoSelection {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            base_center_world: Vec3::ZERO,
            reference_rotation_world: Quat::IDENTITY,
            reference_target_id: 0,
            pivot_mode: MultiPivotMode::SelectionCenter,
        }
    }
}

impl GizmoSelection {
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn supports_scale(&self) -> bool {
        self.targets
            .iter()
            .all(|target| target.local_transform.has_scale)
    }

    pub fn center_world(&self) -> Vec3 {
        self.base_center_world
    }
}

fn rotate_euler(point: Vec3, rotation: Vec3) -> Vec3 {
    let mut rotated = point;
    let (sx, cx) = rotation.x.sin_cos();
    rotated = Vec3::new(
        rotated.x,
        cx * rotated.y - sx * rotated.z,
        sx * rotated.y + cx * rotated.z,
    );
    let (sy, cy) = rotation.y.sin_cos();
    rotated = Vec3::new(
        cy * rotated.x + sy * rotated.z,
        rotated.y,
        -sy * rotated.x + cy * rotated.z,
    );
    let (sz, cz) = rotation.z.sin_cos();
    Vec3::new(
        cz * rotated.x - sz * rotated.y,
        sz * rotated.x + cz * rotated.y,
        rotated.z,
    )
}

pub(crate) fn inverse_rotate_euler(point: Vec3, rotation: Vec3) -> Vec3 {
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

fn local_to_world_rotation(rotation: Vec3) -> Quat {
    euler_to_quat(rotation).inverse()
}

fn transform_local_to_parent_matrix(transform: &NodeTransform) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        transform.scale,
        local_to_world_rotation(transform.rotation),
        transform.position,
    )
}

fn extract_node_transform(scene: &Scene, node_id: NodeId) -> Option<NodeTransform> {
    match scene.nodes.get(&node_id).map(|node| &node.data) {
        Some(NodeData::Primitive {
            position,
            rotation,
            scale,
            ..
        }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: *scale,
            has_scale: true,
        }),
        Some(NodeData::Sculpt {
            position, rotation, ..
        }) => Some(NodeTransform {
            position: *position,
            rotation: *rotation,
            scale: Vec3::ONE,
            has_scale: false,
        }),
        Some(NodeData::Transform {
            translation,
            rotation,
            scale,
            ..
        }) => Some(NodeTransform {
            position: *translation,
            rotation: *rotation,
            scale: *scale,
            has_scale: true,
        }),
        _ => None,
    }
}

fn build_parent_world_transform(
    scene: &Scene,
    node_id: NodeId,
    parent_map: &std::collections::HashMap<NodeId, NodeId>,
) -> (Mat4, Mat4, Quat) {
    let mut transform_chain = Vec::new();
    let mut current = node_id;

    while let Some(&parent_id) = parent_map.get(&current) {
        if let Some(NodeData::Transform {
            translation,
            rotation,
            scale,
            ..
        }) = scene.nodes.get(&parent_id).map(|node| &node.data)
        {
            transform_chain.push(NodeTransform {
                position: *translation,
                rotation: *rotation,
                scale: *scale,
                has_scale: true,
            });
        }
        current = parent_id;
    }

    let mut parent_world_matrix = Mat4::IDENTITY;
    let mut parent_world_rotation = Quat::IDENTITY;
    for transform in transform_chain.iter().rev() {
        parent_world_matrix *= transform_local_to_parent_matrix(transform);
        parent_world_rotation *= local_to_world_rotation(transform.rotation);
    }

    (
        parent_world_matrix,
        parent_world_matrix.inverse(),
        parent_world_rotation,
    )
}

fn build_gizmo_target(
    scene: &Scene,
    node_id: NodeId,
    parent_map: &std::collections::HashMap<NodeId, NodeId>,
) -> Option<GizmoTarget> {
    let local_transform = extract_node_transform(scene, node_id)?;
    let (parent_world_matrix, parent_world_inverse, parent_world_rotation) =
        build_parent_world_transform(scene, node_id, parent_map);
    let world_position = parent_world_matrix.transform_point3(local_transform.position);
    let world_rotation = parent_world_rotation * local_to_world_rotation(local_transform.rotation);

    Some(GizmoTarget {
        node_id,
        local_transform,
        parent_world_inverse,
        parent_world_rotation,
        world_position,
        world_rotation,
    })
}

pub(crate) fn current_transform_target(scene: &Scene, node_id: NodeId) -> NodeId {
    let Some(object) = resolve_presented_object(scene, node_id) else {
        return node_id;
    };
    if node_id != object.host_id {
        return node_id;
    }
    current_transform_owner(scene, node_id).unwrap_or(node_id)
}

pub(crate) fn collect_transform_wrapper_targets(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
) -> Vec<NodeId> {
    let mut requested_targets = Vec::new();
    let mut seen_targets = HashSet::new();

    let mut selection_ids = Vec::new();
    if let Some(primary_selected) = selected {
        selection_ids.push(primary_selected);
    }
    let mut extra_ids: Vec<_> = selected_set
        .iter()
        .copied()
        .filter(|node_id| Some(*node_id) != selected)
        .collect();
    extra_ids.sort_unstable();
    selection_ids.extend(extra_ids);

    for node_id in selection_ids {
        let Some(object) = resolve_presented_object(scene, node_id) else {
            continue;
        };
        if node_id != object.host_id {
            continue;
        }
        if !matches!(
            object.kind,
            crate::graph::presented_object::PresentedObjectKind::Parametric
        ) || object.attached_sculpt_id.is_none()
        {
            continue;
        }
        if current_transform_owner(scene, object.host_id) != Some(object.host_id) {
            continue;
        }
        if seen_targets.insert(object.object_root_id) {
            requested_targets.push(object.object_root_id);
        }
    }

    requested_targets
}

pub(crate) fn collect_gizmo_selection(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
    selection_behavior: &SelectionBehaviorSettings,
) -> Option<GizmoSelection> {
    let mut ordered_ids = Vec::new();
    if let Some(primary_selected) = selected {
        ordered_ids.push(current_transform_target(scene, primary_selected));
    }

    let mut extra_ids: Vec<_> = selected_set
        .iter()
        .copied()
        .filter(|node_id| Some(*node_id) != selected)
        .collect();
    extra_ids.sort_unstable();
    ordered_ids.extend(
        extra_ids
            .into_iter()
            .map(|node_id| current_transform_target(scene, node_id)),
    );

    let mut unique_ids = Vec::new();
    let mut seen_ids = HashSet::new();
    for node_id in ordered_ids {
        if seen_ids.insert(node_id) {
            unique_ids.push(node_id);
        }
    }
    if unique_ids.is_empty() {
        return None;
    }

    let parent_map = scene.build_parent_map();
    let unique_id_set: HashSet<_> = unique_ids.iter().copied().collect();
    let mut targets = Vec::new();

    for node_id in unique_ids {
        let mut current = node_id;
        let mut has_selected_ancestor = false;
        while let Some(&parent_id) = parent_map.get(&current) {
            if unique_id_set.contains(&parent_id) {
                has_selected_ancestor = true;
                break;
            }
            current = parent_id;
        }
        if has_selected_ancestor {
            continue;
        }
        if let Some(target) = build_gizmo_target(scene, node_id, &parent_map) {
            targets.push(target);
        }
    }

    if targets.is_empty() {
        return None;
    }

    let reference_target = selected
        .map(|selected_id| current_transform_target(scene, selected_id))
        .and_then(|selected_id| targets.iter().find(|target| target.node_id == selected_id))
        .unwrap_or(&targets[0]);
    let reference_target_id = reference_target.node_id;

    let mut selection_center_world = Vec3::ZERO;
    for target in &targets {
        selection_center_world += target.world_position;
    }
    selection_center_world /= targets.len() as f32;

    let base_center_world = match selection_behavior.multi_pivot_mode {
        MultiPivotMode::SelectionCenter => selection_center_world,
        MultiPivotMode::ActiveObject => reference_target.world_position,
    };

    let reference_rotation_world = match selection_behavior.multi_axis_orientation {
        MultiAxisOrientation::WorldZero => Quat::IDENTITY,
        MultiAxisOrientation::ActiveObject => reference_target.world_rotation,
    };

    Some(GizmoSelection {
        targets,
        base_center_world,
        reference_rotation_world,
        reference_target_id,
        pivot_mode: selection_behavior.multi_pivot_mode,
    })
}

fn current_targets_from_baseline(
    scene: &Scene,
    baseline: &GizmoSelection,
) -> Option<Vec<GizmoTarget>> {
    let parent_map = scene.build_parent_map();
    baseline
        .targets
        .iter()
        .map(|target| build_gizmo_target(scene, target.node_id, &parent_map))
        .collect()
}

fn find_target_by_id(targets: &[GizmoTarget], target_id: NodeId) -> Option<&GizmoTarget> {
    targets.iter().find(|target| target.node_id == target_id)
}

fn rotation_delta_quat_from_euler(
    rotation_delta_rad: Vec3,
    gizmo_space: &GizmoSpace,
    reference_rotation_world: Quat,
) -> Quat {
    let frame_delta = local_to_world_rotation(rotation_delta_rad);
    match gizmo_space {
        GizmoSpace::World => frame_delta,
        GizmoSpace::Local => {
            reference_rotation_world * frame_delta * reference_rotation_world.inverse()
        }
    }
}

fn rotation_delta_euler_from_world_quat(
    rotation_delta_world: Quat,
    gizmo_space: &GizmoSpace,
    reference_rotation_world: Quat,
    previous_rotation_delta_rad: Vec3,
) -> Vec3 {
    let frame_delta = match gizmo_space {
        GizmoSpace::World => rotation_delta_world,
        GizmoSpace::Local => {
            reference_rotation_world.inverse() * rotation_delta_world * reference_rotation_world
        }
    };
    quat_to_euler_stable(frame_delta.inverse(), previous_rotation_delta_rad)
}

pub(crate) fn derive_multi_transform_readout(
    scene: &Scene,
    baseline: &GizmoSelection,
    gizmo_space: &GizmoSpace,
    previous_rotation_delta_rad: Vec3,
) -> Option<MultiTransformReadout> {
    let current_targets = current_targets_from_baseline(scene, baseline)?;
    if current_targets.is_empty() {
        return None;
    }

    let mut current_selection_center_world = Vec3::ZERO;
    for target in &current_targets {
        current_selection_center_world += target.world_position;
    }
    current_selection_center_world /= current_targets.len() as f32;

    let current_pivot_world = match baseline.pivot_mode {
        MultiPivotMode::SelectionCenter => current_selection_center_world,
        MultiPivotMode::ActiveObject => {
            find_target_by_id(&current_targets, baseline.reference_target_id)
                .map(|target| target.world_position)
                .unwrap_or(current_targets[0].world_position)
        }
    };
    let position_delta = current_pivot_world - baseline.base_center_world;

    let reference_current = find_target_by_id(&current_targets, baseline.reference_target_id)
        .unwrap_or(&current_targets[0]);
    let reference_baseline = find_target_by_id(&baseline.targets, baseline.reference_target_id)
        .unwrap_or(&baseline.targets[0]);
    let rotation_delta_world =
        reference_current.world_rotation * reference_baseline.world_rotation.inverse();
    let rotation_delta_rad = rotation_delta_euler_from_world_quat(
        rotation_delta_world,
        gizmo_space,
        baseline.reference_rotation_world,
        previous_rotation_delta_rad,
    );

    let scale_enabled = baseline.supports_scale();
    let scale_factor = if scale_enabled {
        let current_scale = reference_current.local_transform.scale;
        let baseline_scale = reference_baseline.local_transform.scale;
        Vec3::new(
            current_scale.x / baseline_scale.x.max(0.01),
            current_scale.y / baseline_scale.y.max(0.01),
            current_scale.z / baseline_scale.z.max(0.01),
        )
    } else {
        Vec3::ONE
    };

    Some(MultiTransformReadout {
        position_delta,
        rotation_delta_rad,
        scale_factor,
        scale_enabled,
    })
}

pub(crate) fn compute_axis_directions(node_rotation: Vec3, gizmo_space: &GizmoSpace) -> [Vec3; 3] {
    match gizmo_space {
        GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
        GizmoSpace::Local => [
            inverse_rotate_euler(Vec3::X, node_rotation),
            inverse_rotate_euler(Vec3::Y, node_rotation),
            inverse_rotate_euler(Vec3::Z, node_rotation),
        ],
    }
}

pub(crate) fn compute_world_axis_directions(
    reference_rotation_world: Quat,
    gizmo_space: &GizmoSpace,
) -> [Vec3; 3] {
    match gizmo_space {
        GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
        GizmoSpace::Local => [
            (reference_rotation_world * Vec3::X).normalize_or_zero(),
            (reference_rotation_world * Vec3::Y).normalize_or_zero(),
            (reference_rotation_world * Vec3::Z).normalize_or_zero(),
        ],
    }
}

pub(crate) fn multi_selection_axis_directions(
    reference_rotation_world: Quat,
    selection_behavior: &SelectionBehaviorSettings,
) -> [Vec3; 3] {
    match selection_behavior.multi_axis_orientation {
        MultiAxisOrientation::WorldZero => [Vec3::X, Vec3::Y, Vec3::Z],
        MultiAxisOrientation::ActiveObject => {
            compute_world_axis_directions(reference_rotation_world, &GizmoSpace::Local)
        }
    }
}

pub(crate) fn gizmo_center_for_single_transform(
    transform: &NodeTransform,
    pivot_offset: Vec3,
) -> Vec3 {
    transform.position + rotate_euler(pivot_offset, transform.rotation)
}

pub(crate) fn single_transform(
    scene: &Scene,
    selected: Option<NodeId>,
) -> Option<(NodeId, Vec3, Vec3, bool)> {
    let node_id = current_transform_target(scene, selected?);
    transform_for_node(scene, node_id)
        .map(|(position, rotation, has_scale)| (node_id, position, rotation, has_scale))
}

pub(crate) fn transform_for_node(scene: &Scene, node_id: NodeId) -> Option<(Vec3, Vec3, bool)> {
    let transform = extract_node_transform(scene, node_id)?;
    Some((transform.position, transform.rotation, transform.has_scale))
}

pub(crate) fn set_node_position(scene: &mut Scene, node_id: NodeId, position: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            position: node_position,
            ..
        } => *node_position = position,
        NodeData::Sculpt {
            position: node_position,
            ..
        } => *node_position = position,
        NodeData::Transform { translation, .. } => *translation = position,
        _ => {}
    }
}

pub(crate) fn set_node_rotation(scene: &mut Scene, node_id: NodeId, rotation: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        NodeData::Sculpt {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        NodeData::Transform {
            rotation: node_rotation,
            ..
        } => *node_rotation = rotation,
        _ => {}
    }
}

pub(crate) fn set_node_scale(scene: &mut Scene, node_id: NodeId, scale: Vec3) {
    let Some(node) = scene.nodes.get_mut(&node_id) else {
        return;
    };
    match &mut node.data {
        NodeData::Primitive {
            scale: node_scale, ..
        } => *node_scale = scale,
        NodeData::Transform {
            scale: node_scale, ..
        } => *node_scale = scale,
        _ => {}
    }
}

pub(crate) fn apply_rotation_delta(
    rotation: &mut Vec3,
    angle_delta: f32,
    axis_direction: Vec3,
    gizmo_space: &GizmoSpace,
) {
    let previous = *rotation;
    let delta_quaternion = Quat::from_axis_angle(axis_direction, angle_delta);
    let current = euler_to_quat(*rotation);
    let updated = match gizmo_space {
        GizmoSpace::Local => current * delta_quaternion,
        GizmoSpace::World => delta_quaternion * current,
    };
    *rotation = quat_to_euler_stable(updated, previous);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{ModifierKind, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
    use std::collections::HashMap;

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        }
    }

    fn sculpt_grid() -> VoxelGrid {
        VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    fn primitive_position(scene: &Scene, node_id: NodeId) -> Vec3 {
        match &scene.nodes[&node_id].data {
            NodeData::Primitive { position, .. } => *position,
            _ => panic!("expected primitive"),
        }
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let delta = actual - expected;
        assert!(
            delta.length() <= 0.0005,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn collect_gizmo_selection_uses_transform_owner_for_attached_sculpt_host() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let (transform_id, _sculpt_id) = scene.insert_sculpt_layer_above(
            primitive_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );

        let selection = collect_gizmo_selection(
            &scene,
            Some(primitive_id),
            &HashSet::from([primitive_id]),
            &SelectionBehaviorSettings::default(),
        )
        .expect("selection should resolve");

        assert_eq!(selection.target_count(), 1);
        assert_eq!(selection.targets[0].node_id, transform_id);
    }

    #[test]
    fn collect_transform_wrapper_targets_requests_missing_attached_sculpt_wrapper() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier_id = scene.insert_modifier_above(primitive_id, ModifierKind::Noise);
        let sculpt_id = scene.insert_sculpt_above(
            modifier_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );

        let requested = collect_transform_wrapper_targets(
            &scene,
            Some(primitive_id),
            &HashSet::from([primitive_id, sculpt_id]),
        );

        assert_eq!(requested, vec![sculpt_id]);
    }

    #[test]
    fn derive_multi_transform_readout_tracks_rotation_and_translation() {
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

        let baseline = collect_gizmo_selection(
            &scene,
            Some(left),
            &HashSet::from([left, right]),
            &SelectionBehaviorSettings::default(),
        )
        .expect("baseline");

        set_node_position(&mut scene, left, Vec3::new(0.0, -1.0, 0.0));
        set_node_position(&mut scene, right, Vec3::new(0.0, 1.0, 0.0));
        if let Some(NodeData::Primitive { rotation, .. }) =
            scene.nodes.get_mut(&left).map(|node| &mut node.data)
        {
            *rotation = Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        }
        if let Some(NodeData::Primitive { rotation, .. }) =
            scene.nodes.get_mut(&right).map(|node| &mut node.data)
        {
            *rotation = Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        }

        let readout =
            derive_multi_transform_readout(&scene, &baseline, &GizmoSpace::Local, Vec3::ZERO)
                .expect("readout");

        assert_vec3_close(readout.position_delta, Vec3::new(0.0, 0.0, 0.0));
        assert_vec3_close(
            readout.rotation_delta_rad,
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
        );
    }
}
