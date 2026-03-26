#![allow(dead_code)]

use std::collections::HashSet;

use super::scene::{NodeData, NodeId, Scene};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentedObjectKind {
    Parametric,
    Voxel,
    Light,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentedObjectRef {
    pub host_id: NodeId,
    pub object_root_id: NodeId,
    pub attached_sculpt_id: Option<NodeId>,
    pub kind: PresentedObjectKind,
}

impl PresentedObjectRef {
    pub const fn supports_add_sculpt(self) -> bool {
        matches!(self.kind, PresentedObjectKind::Parametric) && self.attached_sculpt_id.is_none()
    }

    pub const fn can_remove_attached_sculpt(self) -> bool {
        self.attached_sculpt_id.is_some()
    }

    pub const fn render_highlight_id(self) -> NodeId {
        match self.attached_sculpt_id {
            Some(sculpt_id) => sculpt_id,
            None => self.host_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresentedSelection {
    pub primary: Option<PresentedObjectRef>,
    pub ordered: Vec<PresentedObjectRef>,
}

fn first_transform_wrapper(scene: &Scene, object: PresentedObjectRef) -> Option<NodeId> {
    collect_presented_wrapper_chain(scene, object)
        .into_iter()
        .find(|wrapper_id| {
            matches!(
                scene.nodes.get(wrapper_id).map(|node| &node.data),
                Some(NodeData::Transform { .. })
            )
        })
}

fn attached_sculpt_base_root(scene: &Scene, object: PresentedObjectRef) -> Option<NodeId> {
    let sculpt_id = object.attached_sculpt_id?;
    match scene.nodes.get(&sculpt_id).map(|node| &node.data) {
        Some(NodeData::Sculpt {
            input: Some(child_id),
            ..
        }) => Some(*child_id),
        _ => Some(object.host_id),
    }
}

fn resolve_host_base(scene: &Scene, start_id: NodeId) -> Option<(NodeId, PresentedObjectKind)> {
    let mut current = start_id;
    loop {
        let node = scene.nodes.get(&current)?;
        match &node.data {
            NodeData::Transform {
                input: Some(child_id),
                ..
            }
            | NodeData::Modifier {
                input: Some(child_id),
                ..
            }
            | NodeData::Sculpt {
                input: Some(child_id),
                ..
            } => {
                current = *child_id;
            }
            NodeData::Primitive { .. } | NodeData::Operation { .. } => {
                return Some((current, PresentedObjectKind::Parametric));
            }
            NodeData::Sculpt { input: None, .. } => {
                return Some((current, PresentedObjectKind::Voxel));
            }
            NodeData::Light { .. } => {
                return Some((current, PresentedObjectKind::Light));
            }
            NodeData::Transform { input: None, .. } | NodeData::Modifier { input: None, .. } => {
                return None;
            }
        }
    }
}

pub fn resolve_presented_object(scene: &Scene, start_id: NodeId) -> Option<PresentedObjectRef> {
    let (host_id, kind) = resolve_host_base(scene, start_id)?;
    let parent_map = scene.build_parent_map();
    let mut current = host_id;
    let mut object_root_id = host_id;
    let mut attached_sculpt_id = None;

    while let Some(&parent_id) = parent_map.get(&current) {
        let parent = scene.nodes.get(&parent_id)?;
        match &parent.data {
            NodeData::Transform {
                input: Some(input_id),
                ..
            }
            | NodeData::Modifier {
                input: Some(input_id),
                ..
            } if *input_id == current => {
                object_root_id = parent_id;
                current = parent_id;
            }
            NodeData::Sculpt {
                input: Some(input_id),
                ..
            } if *input_id == current => {
                if matches!(kind, PresentedObjectKind::Parametric) && attached_sculpt_id.is_none() {
                    attached_sculpt_id = Some(parent_id);
                }
                object_root_id = parent_id;
                current = parent_id;
            }
            _ => break,
        }
    }

    Some(PresentedObjectRef {
        host_id,
        object_root_id,
        attached_sculpt_id,
        kind,
    })
}

pub fn resolve_host_selection(scene: &Scene, selected: Option<NodeId>) -> Option<NodeId> {
    selected
        .and_then(|selected_id| resolve_presented_object(scene, selected_id))
        .map(|presented| presented.host_id)
}

pub fn current_transform_owner(scene: &Scene, start_id: NodeId) -> Option<NodeId> {
    let object = resolve_presented_object(scene, start_id)?;
    match object.kind {
        PresentedObjectKind::Voxel | PresentedObjectKind::Light => Some(object.host_id),
        PresentedObjectKind::Parametric if object.attached_sculpt_id.is_some() => {
            first_transform_wrapper(scene, object).or(Some(object.host_id))
        }
        PresentedObjectKind::Parametric => Some(object.host_id),
    }
}

pub fn object_transform_wrapper(scene: &Scene, start_id: NodeId) -> Option<NodeId> {
    let object = resolve_presented_object(scene, start_id)?;
    if !matches!(object.kind, PresentedObjectKind::Parametric)
        || object.attached_sculpt_id.is_none()
    {
        return None;
    }
    first_transform_wrapper(scene, object)
}

pub fn presented_wrap_target(scene: &Scene, start_id: NodeId) -> Option<NodeId> {
    let object = resolve_presented_object(scene, start_id)?;
    if matches!(object.kind, PresentedObjectKind::Light) {
        return None;
    }
    attached_sculpt_base_root(scene, object).or(Some(object.object_root_id))
}

pub fn ensure_presented_transform_owner(scene: &mut Scene, start_id: NodeId) -> Option<NodeId> {
    let object = resolve_presented_object(scene, start_id)?;
    if !matches!(object.kind, PresentedObjectKind::Parametric)
        || object.attached_sculpt_id.is_none()
    {
        return current_transform_owner(scene, start_id);
    }

    if let Some(transform_id) = first_transform_wrapper(scene, object) {
        return Some(transform_id);
    }

    Some(scene.insert_transform_above(object.object_root_id))
}

pub fn normalize_attached_sculpt_transform_owners(scene: &mut Scene) -> usize {
    let mut node_ids: Vec<_> = scene.nodes.keys().copied().collect();
    node_ids.sort_unstable();

    let mut normalized = 0;
    let mut seen_hosts = HashSet::new();
    for node_id in node_ids {
        let Some(object) = resolve_presented_object(scene, node_id) else {
            continue;
        };
        if !seen_hosts.insert(object.host_id) {
            continue;
        }
        if !matches!(object.kind, PresentedObjectKind::Parametric)
            || object.attached_sculpt_id.is_none()
        {
            continue;
        }
        if first_transform_wrapper(scene, object).is_some() {
            continue;
        }
        scene.insert_transform_above(object.object_root_id);
        normalized += 1;
    }

    normalized
}

#[allow(dead_code)]
pub fn host_attached_sculpt_id(scene: &Scene, host_id: NodeId) -> Option<NodeId> {
    resolve_presented_object(scene, host_id).and_then(|presented| presented.attached_sculpt_id)
}

#[allow(dead_code)]
pub fn host_has_attached_sculpt(scene: &Scene, host_id: NodeId) -> bool {
    host_attached_sculpt_id(scene, host_id).is_some()
}

pub fn collect_presented_selection(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
) -> PresentedSelection {
    let mut ordered = Vec::new();
    let mut seen_hosts = HashSet::new();

    if let Some(primary_id) = selected {
        if let Some(primary) = resolve_presented_object(scene, primary_id) {
            if seen_hosts.insert(primary.host_id) {
                ordered.push(primary);
            }
        }
    }

    let mut extra_ids: Vec<_> = selected_set
        .iter()
        .copied()
        .filter(|node_id| Some(*node_id) != selected)
        .collect();
    extra_ids.sort_unstable();

    for node_id in extra_ids {
        if let Some(presented) = resolve_presented_object(scene, node_id) {
            if seen_hosts.insert(presented.host_id) {
                ordered.push(presented);
            }
        }
    }

    PresentedSelection {
        primary: ordered.first().copied(),
        ordered,
    }
}

pub fn collect_render_highlight_ids(
    scene: &Scene,
    selected: Option<NodeId>,
    selected_set: &HashSet<NodeId>,
) -> HashSet<NodeId> {
    collect_presented_selection(scene, selected, selected_set)
        .ordered
        .into_iter()
        .map(PresentedObjectRef::render_highlight_id)
        .collect()
}

pub fn presented_top_level_objects(scene: &Scene) -> Vec<PresentedObjectRef> {
    let mut objects = Vec::new();
    let mut seen_hosts = HashSet::new();

    for root_id in scene.top_level_nodes() {
        if let Some(presented) = resolve_presented_object(scene, root_id) {
            if seen_hosts.insert(presented.host_id) {
                objects.push(presented);
            }
        }
    }

    objects.sort_by_key(|presented| presented.object_root_id);
    objects
}

pub fn presented_children(scene: &Scene, object: PresentedObjectRef) -> Vec<PresentedObjectRef> {
    let Some(node) = scene.nodes.get(&object.host_id) else {
        return Vec::new();
    };

    let child_ids: Vec<NodeId> = match &node.data {
        NodeData::Operation { left, right, .. } => {
            let mut children = Vec::new();
            if let Some(left_id) = left {
                children.push(*left_id);
            }
            if let Some(right_id) = right {
                children.push(*right_id);
            }
            children
        }
        _ => Vec::new(),
    };

    let mut children = Vec::new();
    let mut seen_hosts = HashSet::new();
    for child_id in child_ids {
        if let Some(child) = resolve_presented_object(scene, child_id) {
            if seen_hosts.insert(child.host_id) {
                children.push(child);
            }
        }
    }
    children
}

pub fn collect_presented_wrapper_chain(scene: &Scene, object: PresentedObjectRef) -> Vec<NodeId> {
    let mut wrappers = Vec::new();
    let mut current = object.object_root_id;

    while current != object.host_id {
        wrappers.push(current);
        let Some(node) = scene.nodes.get(&current) else {
            break;
        };
        current = match &node.data {
            NodeData::Transform {
                input: Some(child_id),
                ..
            }
            | NodeData::Modifier {
                input: Some(child_id),
                ..
            }
            | NodeData::Sculpt {
                input: Some(child_id),
                ..
            } => *child_id,
            _ => break,
        };
    }

    wrappers
}

pub fn collect_presented_base_wrapper_chain(
    scene: &Scene,
    object: PresentedObjectRef,
) -> Vec<NodeId> {
    let Some(mut current) =
        attached_sculpt_base_root(scene, object).or(Some(object.object_root_id))
    else {
        return Vec::new();
    };

    let mut wrappers = Vec::new();
    while current != object.host_id {
        wrappers.push(current);
        let Some(node) = scene.nodes.get(&current) else {
            break;
        };
        current = match &node.data {
            NodeData::Transform {
                input: Some(child_id),
                ..
            }
            | NodeData::Modifier {
                input: Some(child_id),
                ..
            }
            | NodeData::Sculpt {
                input: Some(child_id),
                ..
            } => *child_id,
            _ => break,
        };
    }

    wrappers
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::graph::scene::{MaterialParams, ModifierKind, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;

    fn sculpt_grid() -> VoxelGrid {
        VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    #[test]
    fn resolve_presented_object_collapses_attached_sculpt_chain() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let transform_id = scene.insert_transform_above(primitive_id);
        let sculpt_id = scene.insert_sculpt_above(
            transform_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );
        let object = resolve_presented_object(&scene, sculpt_id).unwrap();

        assert_eq!(object.host_id, primitive_id);
        assert_eq!(object.object_root_id, sculpt_id);
        assert_eq!(object.attached_sculpt_id, Some(sculpt_id));
        assert_eq!(object.kind, PresentedObjectKind::Parametric);
    }

    #[test]
    fn resolve_presented_object_keeps_standalone_sculpt_as_voxel_host() {
        let mut scene = Scene::new();
        let sculpt_id = scene.add_node(
            "Voxel Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                material: MaterialParams::default(),
                layer_intensity: 1.0,
                voxel_grid: sculpt_grid(),
                desired_resolution: 8,
            },
        );

        let object = resolve_presented_object(&scene, sculpt_id).unwrap();
        assert_eq!(object.host_id, sculpt_id);
        assert_eq!(object.object_root_id, sculpt_id);
        assert_eq!(object.attached_sculpt_id, None);
        assert_eq!(object.kind, PresentedObjectKind::Voxel);
    }

    #[test]
    fn collect_presented_selection_deduplicates_raw_wrapper_selection() {
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
        let selected_set = HashSet::from([primitive_id, modifier_id, sculpt_id]);

        let selection = collect_presented_selection(&scene, Some(sculpt_id), &selected_set);
        assert_eq!(selection.ordered.len(), 1);
        assert_eq!(selection.primary.unwrap().host_id, primitive_id);
    }

    #[test]
    fn collect_render_highlight_ids_prefers_attached_sculpt_surface() {
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
        let selected_set = HashSet::from([primitive_id, modifier_id, sculpt_id]);

        let highlight_ids = collect_render_highlight_ids(&scene, Some(primitive_id), &selected_set);

        assert_eq!(highlight_ids, HashSet::from([sculpt_id]));
    }

    #[test]
    fn presented_children_use_host_rows_instead_of_wrapper_rows() {
        let mut scene = Scene::new();
        let left_id = scene.create_primitive(SdfPrimitive::Sphere);
        let right_id = scene.create_primitive(SdfPrimitive::Box);
        let right_wrapped = scene.insert_transform_above(right_id);
        let operation_id = scene.create_operation(
            crate::graph::scene::CsgOp::Union,
            Some(left_id),
            Some(right_wrapped),
        );

        let parent = resolve_presented_object(&scene, operation_id).unwrap();
        let children = presented_children(&scene, parent);

        assert_eq!(children.len(), 2);
        assert_eq!(children[0].host_id, left_id);
        assert_eq!(children[1].host_id, right_id);
    }

    #[test]
    fn current_transform_owner_prefers_attached_sculpt_object_transform() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let (transform_id, _) = scene.insert_sculpt_layer_above(
            primitive_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );

        assert_eq!(
            current_transform_owner(&scene, primitive_id),
            Some(transform_id)
        );
    }

    #[test]
    fn presented_wrap_target_prefers_base_chain_under_attached_sculpt() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let base_modifier_id = scene.insert_modifier_above(primitive_id, ModifierKind::Noise);
        let (object_transform_id, sculpt_id) = scene.insert_sculpt_layer_above(
            base_modifier_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );

        let object = resolve_presented_object(&scene, primitive_id).unwrap();

        assert_eq!(
            presented_wrap_target(&scene, primitive_id),
            Some(base_modifier_id)
        );
        assert_eq!(
            object_transform_wrapper(&scene, primitive_id),
            Some(object_transform_id)
        );
        assert_eq!(
            collect_presented_base_wrapper_chain(&scene, object),
            vec![base_modifier_id]
        );
        assert_eq!(object.attached_sculpt_id, Some(sculpt_id));
    }

    #[test]
    fn presented_wrap_target_keeps_operation_host_base_chain_separate_from_object_transform() {
        let mut scene = Scene::new();
        let left_id = scene.create_primitive(SdfPrimitive::Sphere);
        let right_id = scene.create_primitive(SdfPrimitive::Box);
        let operation_id = scene.create_operation(
            crate::graph::scene::CsgOp::Union,
            Some(left_id),
            Some(right_id),
        );
        let base_transform_id = scene.insert_transform_above(operation_id);
        let (object_transform_id, sculpt_id) = scene.insert_sculpt_layer_above(
            base_transform_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.2, 0.8, 0.2),
            sculpt_grid(),
        );

        let object = resolve_presented_object(&scene, operation_id).unwrap();

        assert_eq!(object.host_id, operation_id);
        assert_eq!(
            presented_wrap_target(&scene, operation_id),
            Some(base_transform_id)
        );
        assert_eq!(
            object_transform_wrapper(&scene, operation_id),
            Some(object_transform_id)
        );
        assert_eq!(
            collect_presented_base_wrapper_chain(&scene, object),
            vec![base_transform_id]
        );
        assert_eq!(object.attached_sculpt_id, Some(sculpt_id));
    }

    #[test]
    fn ensure_presented_transform_owner_wraps_legacy_attached_sculpt_root() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let sculpt_id = scene.insert_sculpt_above(
            primitive_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );
        let modifier_id = scene.insert_modifier_above(sculpt_id, ModifierKind::Noise);

        let transform_id = ensure_presented_transform_owner(&mut scene, primitive_id).unwrap();
        let object = resolve_presented_object(&scene, primitive_id).unwrap();

        assert_eq!(object.object_root_id, transform_id);
        let NodeData::Transform { input, .. } = &scene.nodes[&transform_id].data else {
            panic!("expected inserted transform wrapper");
        };
        assert_eq!(*input, Some(modifier_id));
        assert_eq!(
            current_transform_owner(&scene, primitive_id),
            Some(transform_id)
        );
    }

    #[test]
    fn current_transform_owner_leaves_non_sculpt_host_on_host() {
        let mut scene = Scene::new();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let _transform_id = scene.insert_transform_above(primitive_id);

        assert_eq!(
            current_transform_owner(&scene, primitive_id),
            Some(primitive_id)
        );
    }

    #[test]
    fn normalize_attached_sculpt_transform_owners_wraps_every_legacy_host_once() {
        let mut scene = Scene::new();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let left_sculpt = scene.insert_sculpt_above(
            left,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            sculpt_grid(),
        );
        let right_sculpt = scene.insert_sculpt_above(
            right,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.2, 0.8, 0.2),
            sculpt_grid(),
        );

        assert_eq!(normalize_attached_sculpt_transform_owners(&mut scene), 2);
        assert_ne!(current_transform_owner(&scene, left), Some(left));
        assert_ne!(current_transform_owner(&scene, right), Some(right));
        assert_eq!(normalize_attached_sculpt_transform_owners(&mut scene), 0);
        assert_eq!(
            resolve_presented_object(&scene, left_sculpt)
                .unwrap()
                .attached_sculpt_id,
            Some(left_sculpt)
        );
        assert_eq!(
            resolve_presented_object(&scene, right_sculpt)
                .unwrap()
                .attached_sculpt_id,
            Some(right_sculpt)
        );
    }
}
