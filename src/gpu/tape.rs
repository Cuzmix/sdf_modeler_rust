//! Bytecode "tape" representation of a scene SDF, used by the universal
//! fallback render/pick pipelines.
//!
//! When the user adds an SDF object, the structure-specific unrolled WGSL
//! pipeline must be recompiled (~hundreds of milliseconds on Windows / DX12).
//! To make adds feel instant, an always-resident *fallback* pipeline interprets
//! a tape built CPU-side from the scene tree. Topology lives in data, not
//! code, so structure changes update only the tape buffer — no shader
//! recompile, no main-thread stall, new geometry visible the same frame.
//!
//! See `docs/perf-techniques.md` §7 ("JIT-compile per-scene shader") for the
//! companion idea: the unrolled pipeline is the JIT-compiled fast path, and
//! the tape interpreter is the always-applicable slow path.
//!
//! ## Encoding
//!
//! The tape is a flat `Vec<u32>` of opcodes interpreted by the GPU as a
//! stack machine. Each opcode packs:
//! ```text
//!   bits 28..32: kind (TapeOp discriminant)
//!   bits  0..28: payload (node_idx into the nodes[] storage buffer)
//! ```
//!
//! The interpreter maintains a small fixed-depth distance stack:
//! - `EvalPrimitive(idx)` evaluates the primitive at `nodes[idx]` and pushes
//!   the resulting signed distance onto the stack.
//! - `Combine(idx)` pops the top two distances, combines them using the CSG
//!   operation at `nodes[idx]` (op kind in `type_op.x`, smoothing `k` in
//!   `type_op.y`), and pushes the result.
//! - `End` terminates execution; the top of the stack is the final scene SDF.
//!
//! ## Eligibility (the "F2 contract")
//!
//! `Tape::try_build` returns `None` if the scene contains any node the
//! current fallback shader cannot interpret. Such scenes fall back to the
//! existing sync-compile path (see `app/gpu_sync.rs`). As fallback shader
//! support grows, more node types become encodable.
//!
//! Supported in this implementation: primitives, CSG operations, transforms,
//! point modifiers (Twist, Bend, Taper, Elongate, Mirror, Repeat, FiniteRepeat,
//! RadialRepeat, Noise) and distance modifiers (Round, Onion, Offset). Sculpts
//! and composite-volume scenes are out of F2 scope by design — they fall
//! through to the existing sync-compile path so the user sees a frozen frame
//! during the unavoidable rebuild (see G-stale in `gpu_sync.rs`).
//!
//! ## Encoding rules in detail
//!
//! - **Primitives** emit a leaf segment: a per-leaf preamble of `ApplyInvXform`
//!   and `ApplyPointMod` opcodes (collected by walking up the leaf's parent
//!   chain through every Transform and point Modifier ancestor), then
//!   `EvalPrimitive`. The preamble is emitted outermost-first because the
//!   interpreter applies inverse transforms in tree-walk order (root → leaf).
//!
//! - **CSG operations** emit `Combine` once their two children's distances
//!   are on the stack — guaranteed by visible_topo_order's post-order traversal.
//!
//! - **Distance modifiers** (`Round`, `Onion`, `Offset`) emit `ApplyDistMod`
//!   in topo order. They modify the current top-of-stack — which is whatever
//!   their child subtree already produced (be it a single leaf or a fully
//!   combined Operation).
//!
//! - **Transforms** and **point modifiers** do NOT emit their own opcodes in
//!   topo order — their effect was already absorbed into every leaf they
//!   ancestored. The topo iteration silently passes over them.

// Module-level allow: a few of the helper accessors (`unpack`, `len`)
// are intended for tooling and tests rather than the runtime render
// path, and clippy's dead-code analysis flags them otherwise. Keep this
// pragma narrow — it is not a license to leave half-finished code here.
#![allow(dead_code)]

use crate::graph::scene::{NodeData, NodeId, Scene};
use std::collections::HashMap;

/// Maximum distance-stack depth the GPU interpreter is sized for.
///
/// One stack slot is consumed per pushed leaf and freed per combine. For a
/// strictly binary CSG tree, this depth equals the deepest path of nested
/// operations. In practice scenes rarely nest deeper than 8 or so; 16 gives
/// generous headroom without bloating the interpreter's WGSL register usage.
pub const MAX_TAPE_STACK_DEPTH: u32 = 16;

/// Maximum number of opcodes a single tape may contain.
///
/// Sized to comfortably hold a few thousand primitives plus their combines.
/// Tape buffer storage is allocated to this size up front so the worker
/// thread (when fallback compile lands) and the encoder never need to
/// reallocate between frames.
pub const MAX_TAPE_LEN: u32 = 8192;

/// Width (in bits) of the `node_idx` payload portion of an opcode word.
pub const TAPE_PAYLOAD_BITS: u32 = 28;

/// Mask covering the payload bits.
pub const TAPE_PAYLOAD_MASK: u32 = (1 << TAPE_PAYLOAD_BITS) - 1;

/// Maximum encodable `node_idx` (any larger and the tape encoder rejects
/// the scene).
pub const TAPE_MAX_NODE_IDX: u32 = TAPE_PAYLOAD_MASK;

/// Tape opcode kind. The discriminant value is what gets packed into the
/// high 4 bits of an opcode word; keep these in sync with the WGSL
/// constants in `fallback_render.wgsl`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TapeOp {
    /// Terminator. Marks the end of the executable region of the tape.
    End = 0,
    /// Evaluate the primitive at `nodes[idx]` using the interpreter's
    /// current `p_local`, push its signed distance, and reset `p_local`
    /// to the world-space ray point ready for the next leaf.
    EvalPrimitive = 1,
    /// Pop the top two distances, combine using the CSG op at `nodes[idx]`,
    /// push the result.
    Combine = 2,
    /// Set `p_local = inverse_transform(p_local, nodes[idx])` — applies
    /// the inverse of an ancestor `Transform` so the next primitive
    /// evaluates in the transform's local space.
    ApplyInvXform = 3,
    /// Set `p_local = inverse_point_modifier(p_local, nodes[idx])` —
    /// applies the inverse of a domain-deforming modifier (Twist, Bend,
    /// Taper, Elongate, Mirror, Repeat, FiniteRepeat, RadialRepeat, Noise).
    ApplyPointMod = 4,
    /// Pop the top distance, apply a distance-domain modifier (Round,
    /// Onion, Offset) at `nodes[idx]`, push the result. Stack net 0.
    ApplyDistMod = 5,
}

impl TapeOp {
    /// Pack an opcode + payload into a single 32-bit word.
    pub fn pack(self, payload: u32) -> u32 {
        debug_assert!(
            payload <= TAPE_MAX_NODE_IDX,
            "tape payload {payload} exceeds {TAPE_MAX_NODE_IDX}"
        );
        ((self as u32) << TAPE_PAYLOAD_BITS) | (payload & TAPE_PAYLOAD_MASK)
    }
}

/// Reasons the encoder may refuse to encode a scene. Each maps directly to
/// a node type or limit the current fallback shader can't handle; the
/// caller falls back to the existing sync-compile path on `Err`.
#[derive(Debug, PartialEq, Eq)]
pub enum TapeBuildError {
    /// Scene contains a `NodeData::Sculpt` node — out of F2 scope; sculpts
    /// always take the sync-compile path so their voxel binding layout
    /// can be rebuilt atomically with the pipeline.
    UnsupportedSculpt,
    /// Operation node has an unbound `left` or `right` child.
    OperationMissingChild,
    /// Scene exceeds `TAPE_MAX_NODE_IDX` reachable nodes.
    NodeIndexOverflow,
    /// Scene exceeds `MAX_TAPE_LEN` opcode words.
    TapeOverflow,
    /// Scene's combine nesting exceeds `MAX_TAPE_STACK_DEPTH`.
    StackOverflow,
}

/// Encoded tape, ready to be uploaded to the fallback shader's storage
/// buffer. The opcode at index 0 is executed first; `End` terminates.
#[derive(Debug, Clone)]
pub struct Tape {
    /// Opcode words, including a trailing `End` terminator.
    pub opcodes: Vec<u32>,
    /// Maximum stack depth this tape will reach at runtime. Always
    /// `<= MAX_TAPE_STACK_DEPTH` (else encoding would have failed).
    pub max_stack_depth: u32,
}

impl Tape {
    /// Build a tape from `scene` if every node it contains is supported by
    /// the current fallback shader. Returns `Err` with the offending kind
    /// on the first unsupported node — caller treats this as the
    /// eligibility gate that routes the structure change to the
    /// sync-compile path instead.
    pub fn try_build(scene: &Scene) -> Result<Self, TapeBuildError> {
        let topo_order = scene.visible_topo_order();
        let parent_map = scene.build_parent_map();
        let mut opcodes: Vec<u32> = Vec::with_capacity(topo_order.len() * 2 + 1);

        // Map each visible NodeId to the index it will occupy in the GPU
        // `nodes[]` storage buffer. `build_node_buffer` walks the same
        // `visible_topo_order`, so positional index in the topo-order
        // vector equals node_idx in the GPU buffer.
        let topo_index: HashMap<NodeId, u32> = topo_order
            .iter()
            .enumerate()
            .map(|(idx, node_id)| (*node_id, idx as u32))
            .collect();

        if topo_order.len() as u32 > TAPE_MAX_NODE_IDX {
            return Err(TapeBuildError::NodeIndexOverflow);
        }

        // Track depth as we encode so a single stack-overflow check covers
        // every possible scene shape. EvalPrimitive nets +1; Combine nets
        // -1; ApplyDistMod nets 0; ApplyInvXform/ApplyPointMod net 0
        // (they touch p_local, not the distance stack).
        let mut current_depth: i32 = 0;
        let mut max_depth: i32 = 0;

        for &node_id in &topo_order {
            let Some(node) = scene.nodes.get(&node_id) else {
                continue;
            };
            let &node_idx = topo_index
                .get(&node_id)
                .expect("topo_index built from topo_order");

            match &node.data {
                NodeData::Primitive { .. } => {
                    emit_leaf_preamble(node_id, scene, &parent_map, &topo_index, &mut opcodes)?;
                    push_op(&mut opcodes, TapeOp::EvalPrimitive, node_idx)?;
                    current_depth += 1;
                    if current_depth > max_depth {
                        max_depth = current_depth;
                    }
                    if current_depth as u32 > MAX_TAPE_STACK_DEPTH {
                        return Err(TapeBuildError::StackOverflow);
                    }
                }
                NodeData::Operation { left, right, .. } => {
                    if left.is_none() || right.is_none() {
                        return Err(TapeBuildError::OperationMissingChild);
                    }
                    push_op(&mut opcodes, TapeOp::Combine, node_idx)?;
                    current_depth -= 1;
                    if current_depth < 0 {
                        return Err(TapeBuildError::OperationMissingChild);
                    }
                }
                NodeData::Transform { .. } => {
                    // Transforms fold into descendant leaves' preambles —
                    // they don't emit their own opcode in topo order.
                }
                NodeData::Modifier { kind, .. } => {
                    if kind.is_point_modifier() {
                        // Point modifiers fold into descendant leaves'
                        // preambles, same as Transforms.
                    } else {
                        // Distance modifiers operate on top-of-stack and
                        // are emitted at the topo position they occupy —
                        // post-order guarantees the wrapped subtree's
                        // distance is already on the stack.
                        push_op(&mut opcodes, TapeOp::ApplyDistMod, node_idx)?;
                        // Stack net 0 — pops one, pushes one.
                    }
                }
                NodeData::Sculpt { .. } => return Err(TapeBuildError::UnsupportedSculpt),
                NodeData::Light { .. } => {
                    // Lights aren't part of the scene SDF — they live in
                    // the lighting pass. Skip; no opcode, no stack change.
                }
            }
        }

        push_op(&mut opcodes, TapeOp::End, 0)?;

        Ok(Tape {
            opcodes,
            max_stack_depth: max_depth.max(0) as u32,
        })
    }

    /// Number of opcode words in the tape, including the trailing `End`.
    pub fn len(&self) -> usize {
        self.opcodes.len()
    }

    /// True when the tape contains only the trailing `End` terminator —
    /// i.e. there's no scene SDF to evaluate.
    pub fn is_empty(&self) -> bool {
        self.opcodes.len() <= 1
    }
}

/// Walk up from `leaf_id` collecting Transforms and point Modifiers. Emit
/// `ApplyInvXform` and `ApplyPointMod` opcodes in outer-to-inner order so
/// the GPU interpreter can apply them to `p_local` in tree-walk order.
///
/// Distance modifiers are NOT collected here — they emit standalone
/// `ApplyDistMod` opcodes at their topo position, where the affected
/// subtree's distance is already on the interpreter's stack.
///
/// Operations are walked through (they don't contribute to the preamble),
/// because a Transform or point modifier sitting above an Operation still
/// transforms `p` for every leaf in that Operation's subtree.
fn emit_leaf_preamble(
    leaf_id: NodeId,
    scene: &Scene,
    parent_map: &HashMap<NodeId, NodeId>,
    topo_index: &HashMap<NodeId, u32>,
    opcodes: &mut Vec<u32>,
) -> Result<(), TapeBuildError> {
    // Collect (op, idx) pairs in outermost-first order. We walk up from
    // the leaf (innermost-first) and `insert(0, ..)` to reverse on the fly.
    let mut chain: Vec<(TapeOp, u32)> = Vec::new();
    let mut current = leaf_id;
    while let Some(&parent_id) = parent_map.get(&current) {
        let Some(parent_node) = scene.nodes.get(&parent_id) else {
            break;
        };
        let &parent_idx = topo_index
            .get(&parent_id)
            .expect("ancestor must be in visible topo order");
        match &parent_node.data {
            NodeData::Transform { .. } => {
                chain.insert(0, (TapeOp::ApplyInvXform, parent_idx));
            }
            NodeData::Modifier { kind, .. } => {
                if kind.is_point_modifier() {
                    chain.insert(0, (TapeOp::ApplyPointMod, parent_idx));
                }
                // Distance modifier: handled in the topo iteration; do
                // not fold into the leaf preamble.
            }
            NodeData::Operation { .. } => {
                // Operations don't contribute to the preamble themselves.
                // Walk through to pick up any further-out Transforms /
                // point modifiers — those still apply to this leaf.
            }
            NodeData::Sculpt { .. } => {
                // Sculpts wrap their input subtree but Phase 3 fallback
                // shader doesn't dispatch sculpt voxel sampling. Bail so
                // the caller routes to the sync-compile path.
                return Err(TapeBuildError::UnsupportedSculpt);
            }
            NodeData::Primitive { .. } | NodeData::Light { .. } => {
                // Unreachable in well-formed scenes — primitives and
                // lights are leaves, never parents. Stop walking.
                break;
            }
        }
        current = parent_id;
    }
    for (op, idx) in chain {
        push_op(opcodes, op, idx)?;
    }
    Ok(())
}

fn push_op(opcodes: &mut Vec<u32>, op: TapeOp, payload: u32) -> Result<(), TapeBuildError> {
    if opcodes.len() as u32 >= MAX_TAPE_LEN {
        return Err(TapeBuildError::TapeOverflow);
    }
    if payload > TAPE_MAX_NODE_IDX {
        return Err(TapeBuildError::NodeIndexOverflow);
    }
    opcodes.push(op.pack(payload));
    Ok(())
}

/// Unpack an opcode word into `(kind_discriminant, payload)`. Useful for
/// tests and the CPU-side debug inspector; the GPU does its own unpack
/// inline in WGSL.
pub fn unpack(word: u32) -> (u32, u32) {
    let kind = word >> TAPE_PAYLOAD_BITS;
    let payload = word & TAPE_PAYLOAD_MASK;
    (kind, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{CsgOp, NodeData, Scene, SdfPrimitive};
    use glam::Vec3;
    use std::collections::{HashMap, HashSet};

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
            structure_version: 0,
            data_version: 0,
        }
    }

    fn add_sphere(scene: &mut Scene, name: &str) -> NodeId {
        scene.add_node(
            name.to_string(),
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: Default::default(),
                voxel_grid: None,
            },
        )
    }

    fn add_box(scene: &mut Scene, name: &str) -> NodeId {
        scene.add_node(
            name.to_string(),
            NodeData::Primitive {
                kind: SdfPrimitive::Box,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: Default::default(),
                voxel_grid: None,
            },
        )
    }

    fn add_union(scene: &mut Scene, left: NodeId, right: NodeId, k: f32) -> NodeId {
        scene.add_node(
            "Union".to_string(),
            NodeData::Operation {
                op: if k > 0.0 {
                    CsgOp::SmoothUnion
                } else {
                    CsgOp::Union
                },
                smooth_k: k,
                steps: 0.0,
                color_blend: -1.0,
                left: Some(left),
                right: Some(right),
            },
        )
    }

    #[test]
    fn pack_round_trip_preserves_kind_and_payload() {
        let word = TapeOp::EvalPrimitive.pack(42);
        let (kind, payload) = unpack(word);
        assert_eq!(kind, TapeOp::EvalPrimitive as u32);
        assert_eq!(payload, 42);

        let word = TapeOp::Combine.pack(TAPE_MAX_NODE_IDX);
        let (kind, payload) = unpack(word);
        assert_eq!(kind, TapeOp::Combine as u32);
        assert_eq!(payload, TAPE_MAX_NODE_IDX);

        let word = TapeOp::End.pack(0);
        let (kind, payload) = unpack(word);
        assert_eq!(kind, TapeOp::End as u32);
        assert_eq!(payload, 0);
    }

    #[test]
    fn empty_scene_emits_only_terminator() {
        let scene = empty_scene();
        let tape = Tape::try_build(&scene).expect("empty scene is encodable");
        assert!(tape.is_empty());
        assert_eq!(tape.opcodes.len(), 1);
        let (kind, _) = unpack(tape.opcodes[0]);
        assert_eq!(kind, TapeOp::End as u32);
        assert_eq!(tape.max_stack_depth, 0);
    }

    #[test]
    fn single_primitive_emits_one_eval_then_end() {
        let mut scene = empty_scene();
        let _id = add_sphere(&mut scene, "Sphere");

        let tape = Tape::try_build(&scene).expect("single primitive is encodable");
        assert_eq!(tape.opcodes.len(), 2);

        let (kind0, payload0) = unpack(tape.opcodes[0]);
        assert_eq!(kind0, TapeOp::EvalPrimitive as u32);
        assert_eq!(payload0, 0); // first (and only) topo position

        let (kind1, _) = unpack(tape.opcodes[1]);
        assert_eq!(kind1, TapeOp::End as u32);

        assert_eq!(tape.max_stack_depth, 1);
    }

    #[test]
    fn two_primitives_with_union_emit_eval_eval_combine_end() {
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let box_id = add_box(&mut scene, "Box");
        let _union_id = add_union(&mut scene, sphere_id, box_id, 0.0);

        let tape = Tape::try_build(&scene).expect("union of two primitives is encodable");
        assert_eq!(tape.opcodes.len(), 4); // 2 evals + 1 combine + end

        let (k0, p0) = unpack(tape.opcodes[0]);
        let (k1, p1) = unpack(tape.opcodes[1]);
        let (k2, p2) = unpack(tape.opcodes[2]);
        let (k3, _) = unpack(tape.opcodes[3]);

        assert_eq!(k0, TapeOp::EvalPrimitive as u32);
        assert_eq!(k1, TapeOp::EvalPrimitive as u32);
        assert_eq!(k2, TapeOp::Combine as u32);
        assert_eq!(k3, TapeOp::End as u32);

        // The two leaves are the first two visible-topo-order positions
        // (children precede their operation parent in post-order). The
        // combine references the operation's topo position.
        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);

        // Two leaves on stack at peak; combine drops back to one.
        assert_eq!(tape.max_stack_depth, 2);
    }

    #[test]
    fn nested_combines_track_max_stack_depth_correctly() {
        // Tree: ((sphere ∪ box) ∪ sphere2)
        // Topo order: sphere, box, union1, sphere2, union2
        // Stack: 1, 2, 1 (combine), 2 (push sphere2), 1 (combine)
        // Peak = 2.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let box_id = add_box(&mut scene, "Box");
        let union1_id = add_union(&mut scene, sphere_id, box_id, 0.0);
        let sphere2_id = add_sphere(&mut scene, "Sphere2");
        let _union2_id = add_union(&mut scene, union1_id, sphere2_id, 0.0);

        let tape = Tape::try_build(&scene).expect("nested unions are encodable");
        assert_eq!(tape.max_stack_depth, 2);
    }

    #[test]
    fn deep_left_chain_grows_stack_until_first_combine() {
        // Pathological shape: chain of unions where the right child is
        // always a fresh primitive (left-leaning). Each new primitive
        // pushes before its combine pops. Peak depth = depth of leftmost
        // path before any combine fires.
        //
        // Tree: (((s1 ∪ s2) ∪ s3) ∪ s4)
        // Topo: s1, s2, u12, s3, u123, s4, u1234
        // Stack: 1, 2, 1, 2, 1, 2, 1 → peak = 2
        let mut scene = empty_scene();
        let s1 = add_sphere(&mut scene, "S1");
        let s2 = add_sphere(&mut scene, "S2");
        let u12 = add_union(&mut scene, s1, s2, 0.0);
        let s3 = add_sphere(&mut scene, "S3");
        let u123 = add_union(&mut scene, u12, s3, 0.0);
        let s4 = add_sphere(&mut scene, "S4");
        let _u1234 = add_union(&mut scene, u123, s4, 0.0);

        let tape = Tape::try_build(&scene).unwrap();
        assert_eq!(tape.max_stack_depth, 2);
    }

    #[test]
    fn balanced_binary_tree_stack_depth_equals_log2_leaves() {
        // 4 leaves balanced as ((a∪b) ∪ (c∪d)). Codegen visits children
        // post-order: a, b, ab, c, d, cd, abcd. Stack: 1, 2, 1, 2, 3, 2, 1.
        // Peak = 3.
        let mut scene = empty_scene();
        let a = add_sphere(&mut scene, "A");
        let b = add_sphere(&mut scene, "B");
        let ab = add_union(&mut scene, a, b, 0.0);
        let c = add_sphere(&mut scene, "C");
        let d = add_sphere(&mut scene, "D");
        let cd = add_union(&mut scene, c, d, 0.0);
        let _abcd = add_union(&mut scene, ab, cd, 0.0);

        let tape = Tape::try_build(&scene).unwrap();
        assert_eq!(tape.max_stack_depth, 3);
    }

    #[test]
    fn hidden_node_excluded_from_tape() {
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "VisibleSphere");
        let hidden_id = add_sphere(&mut scene, "HiddenSphere");
        scene.hidden_nodes.insert(hidden_id);

        let tape = Tape::try_build(&scene).unwrap();
        // Only the visible sphere should appear: 1 EvalPrimitive + 1 End.
        assert_eq!(tape.opcodes.len(), 2);
        let (k0, p0) = unpack(tape.opcodes[0]);
        assert_eq!(k0, TapeOp::EvalPrimitive as u32);
        // The visible sphere is the first (and only) entry of visible_topo_order.
        let topo = scene.visible_topo_order();
        assert_eq!(topo.len(), 1);
        assert_eq!(topo[0], sphere_id);
        assert_eq!(p0, 0);
    }

    #[test]
    fn lights_skipped_silently_no_tape_op_emitted() {
        // `Scene::create_light` pairs a Light with a parent Transform so
        // the light has a world-space pose. Phase 1a doesn't encode
        // Transforms yet (Phase 1b lifts that), so test the Light-skip
        // path in isolation by adding a parentless Light directly.
        let mut scene = empty_scene();
        let _sphere_id = add_sphere(&mut scene, "Sphere");
        scene.add_node(
            "Light".to_string(),
            NodeData::Light {
                light_type: crate::graph::scene::LightType::Point,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: crate::graph::scene::ProximityMode::Off,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );

        let tape =
            Tape::try_build(&scene).expect("a Light next to a primitive must not block encoding");
        // Sphere + (light contributes 0 ops) + End = 2.
        assert_eq!(tape.opcodes.len(), 2);
        let (k0, _) = unpack(tape.opcodes[0]);
        assert_eq!(k0, TapeOp::EvalPrimitive as u32);
    }

    #[test]
    fn single_transform_above_primitive_emits_inv_xform_then_eval() {
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let _xform_id = scene.add_node(
            "Transform".to_string(),
            NodeData::Transform {
                input: Some(sphere_id),
                translation: Vec3::new(1.0, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );

        let tape = Tape::try_build(&scene).expect("transform-wrapped sphere is encodable");
        // Topo order: [sphere, transform]. Sphere gets the transform in
        // its preamble; transform itself emits no opcode.
        // Tape: [ApplyInvXform(transform_idx=1), EvalPrimitive(sphere_idx=0), End].
        assert_eq!(tape.opcodes.len(), 3);

        let (k0, p0) = unpack(tape.opcodes[0]);
        let (k1, p1) = unpack(tape.opcodes[1]);
        let (k2, _) = unpack(tape.opcodes[2]);
        assert_eq!(k0, TapeOp::ApplyInvXform as u32);
        assert_eq!(p0, 1); // transform is at topo index 1
        assert_eq!(k1, TapeOp::EvalPrimitive as u32);
        assert_eq!(p1, 0); // sphere is at topo index 0
        assert_eq!(k2, TapeOp::End as u32);
        assert_eq!(tape.max_stack_depth, 1);
    }

    #[test]
    fn stacked_transforms_emit_outer_to_inner() {
        // Tree: T_outer wraps T_inner wraps Sphere.
        // Walking up from sphere: T_inner, T_outer.
        // Outer-to-inner emission: T_outer first, then T_inner.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let inner_id = scene.add_node(
            "T_inner".to_string(),
            NodeData::Transform {
                input: Some(sphere_id),
                translation: Vec3::new(1.0, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let outer_id = scene.add_node(
            "T_outer".to_string(),
            NodeData::Transform {
                input: Some(inner_id),
                translation: Vec3::new(0.0, 1.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );

        let tape = Tape::try_build(&scene).expect("stacked transforms are encodable");
        // Topo order: [sphere=0, inner=1, outer=2].
        // Tape: [ApplyInvXform(2), ApplyInvXform(1), EvalPrimitive(0), End]
        assert_eq!(tape.opcodes.len(), 4);

        let (k0, p0) = unpack(tape.opcodes[0]);
        let (k1, p1) = unpack(tape.opcodes[1]);
        let (k2, p2) = unpack(tape.opcodes[2]);
        let (k3, _) = unpack(tape.opcodes[3]);

        assert_eq!(k0, TapeOp::ApplyInvXform as u32);
        assert_eq!(k1, TapeOp::ApplyInvXform as u32);
        assert_eq!(k2, TapeOp::EvalPrimitive as u32);
        assert_eq!(k3, TapeOp::End as u32);

        // outer (idx=2) applied first, inner (idx=1) applied second.
        let topo = scene.visible_topo_order();
        let outer_topo = topo.iter().position(|id| *id == outer_id).unwrap() as u32;
        let inner_topo = topo.iter().position(|id| *id == inner_id).unwrap() as u32;
        assert_eq!(p0, outer_topo);
        assert_eq!(p1, inner_topo);
        assert_eq!(p2, 0); // sphere
    }

    #[test]
    fn transform_above_operation_appears_in_each_leaf_preamble() {
        // Tree: T1 wraps Op (Union of Sphere, Box).
        // Both leaves should have T1 in their preamble.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let box_id = add_box(&mut scene, "Box");
        let union_id = add_union(&mut scene, sphere_id, box_id, 0.0);
        let _xform_id = scene.add_node(
            "T1".to_string(),
            NodeData::Transform {
                input: Some(union_id),
                translation: Vec3::new(1.0, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );

        let tape = Tape::try_build(&scene).expect("transform above operation is encodable");
        // Expected sequence:
        //   ApplyInvXform(T1) EvalPrimitive(Sphere)
        //   ApplyInvXform(T1) EvalPrimitive(Box)
        //   Combine(Union)
        //   End
        // = 6 opcodes total.
        assert_eq!(tape.opcodes.len(), 6);

        let kinds: Vec<u32> = tape.opcodes.iter().map(|&w| unpack(w).0).collect();
        assert_eq!(
            kinds,
            vec![
                TapeOp::ApplyInvXform as u32,
                TapeOp::EvalPrimitive as u32,
                TapeOp::ApplyInvXform as u32,
                TapeOp::EvalPrimitive as u32,
                TapeOp::Combine as u32,
                TapeOp::End as u32,
            ]
        );
    }

    #[test]
    fn point_modifier_above_primitive_emits_apply_point_mod() {
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        scene.add_node(
            "Twist".to_string(),
            NodeData::Modifier {
                kind: crate::graph::scene::ModifierKind::Twist,
                input: Some(sphere_id),
                value: Vec3::new(1.0, 0.0, 0.0),
                extra: Vec3::ZERO,
            },
        );

        let tape = Tape::try_build(&scene).expect("point modifier is encodable");
        // Tape: [ApplyPointMod(twist_idx=1), EvalPrimitive(sphere_idx=0), End]
        assert_eq!(tape.opcodes.len(), 3);
        let kinds: Vec<u32> = tape.opcodes.iter().map(|&w| unpack(w).0).collect();
        assert_eq!(
            kinds,
            vec![
                TapeOp::ApplyPointMod as u32,
                TapeOp::EvalPrimitive as u32,
                TapeOp::End as u32,
            ]
        );
    }

    #[test]
    fn distance_modifier_above_primitive_emits_apply_dist_mod_after_eval() {
        // Round is a distance modifier — it operates on the output
        // distance, not on `p`. Topo order visits it after its child.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        scene.add_node(
            "Round".to_string(),
            NodeData::Modifier {
                kind: crate::graph::scene::ModifierKind::Round,
                input: Some(sphere_id),
                value: Vec3::new(0.1, 0.0, 0.0),
                extra: Vec3::ZERO,
            },
        );

        let tape = Tape::try_build(&scene).expect("distance modifier is encodable");
        // Tape: [EvalPrimitive(sphere_idx=0), ApplyDistMod(round_idx=1), End]
        assert_eq!(tape.opcodes.len(), 3);
        let kinds: Vec<u32> = tape.opcodes.iter().map(|&w| unpack(w).0).collect();
        assert_eq!(
            kinds,
            vec![
                TapeOp::EvalPrimitive as u32,
                TapeOp::ApplyDistMod as u32,
                TapeOp::End as u32,
            ]
        );
        // Stack depth: leaf pushes (1), dist mod is a no-op for depth.
        assert_eq!(tape.max_stack_depth, 1);
    }

    #[test]
    fn distance_modifier_above_operation_emits_post_combine() {
        // Tree: Round wraps (Sphere ∪ Box). Round must apply to the
        // combine's output, not per-leaf.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let box_id = add_box(&mut scene, "Box");
        let union_id = add_union(&mut scene, sphere_id, box_id, 0.0);
        scene.add_node(
            "Round".to_string(),
            NodeData::Modifier {
                kind: crate::graph::scene::ModifierKind::Round,
                input: Some(union_id),
                value: Vec3::new(0.1, 0.0, 0.0),
                extra: Vec3::ZERO,
            },
        );

        let tape = Tape::try_build(&scene).expect("dist modifier above operation is encodable");
        let kinds: Vec<u32> = tape.opcodes.iter().map(|&w| unpack(w).0).collect();
        assert_eq!(
            kinds,
            vec![
                TapeOp::EvalPrimitive as u32, // Sphere
                TapeOp::EvalPrimitive as u32, // Box
                TapeOp::Combine as u32,       // Union
                TapeOp::ApplyDistMod as u32,  // Round, after combine
                TapeOp::End as u32,
            ]
        );
    }

    #[test]
    fn mixed_transform_point_mod_dist_mod_chain_orders_correctly() {
        // Tree (bottom up): Sphere → Twist (point) → Transform → Round (dist).
        // Walk up from sphere: [Twist, Transform, Round]. Round is a dist
        // modifier, so it doesn't fold into the leaf preamble — it emits
        // its own opcode after EvalPrimitive in topo order.
        // Outer-to-inner point-chain emission: Transform, then Twist.
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        let twist_id = scene.add_node(
            "Twist".to_string(),
            NodeData::Modifier {
                kind: crate::graph::scene::ModifierKind::Twist,
                input: Some(sphere_id),
                value: Vec3::new(1.0, 0.0, 0.0),
                extra: Vec3::ZERO,
            },
        );
        let xform_id = scene.add_node(
            "T1".to_string(),
            NodeData::Transform {
                input: Some(twist_id),
                translation: Vec3::new(1.0, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        let _round_id = scene.add_node(
            "Round".to_string(),
            NodeData::Modifier {
                kind: crate::graph::scene::ModifierKind::Round,
                input: Some(xform_id),
                value: Vec3::new(0.1, 0.0, 0.0),
                extra: Vec3::ZERO,
            },
        );

        let tape = Tape::try_build(&scene).expect("mixed chain is encodable");
        let kinds: Vec<u32> = tape.opcodes.iter().map(|&w| unpack(w).0).collect();
        assert_eq!(
            kinds,
            vec![
                TapeOp::ApplyInvXform as u32, // T1 (outer)
                TapeOp::ApplyPointMod as u32, // Twist (inner)
                TapeOp::EvalPrimitive as u32, // Sphere
                TapeOp::ApplyDistMod as u32,  // Round (post-eval, in topo order)
                TapeOp::End as u32,
            ]
        );
    }

    #[test]
    fn light_with_transform_parent_skipped_cleanly() {
        // `Scene::create_light` makes a Light + parent Transform. The Light
        // contributes nothing to the SDF and the encoder must walk through
        // the Transform to the Light without emitting spurious opcodes
        // (since the Transform's only descendant subtree is the light).
        //
        // Today, visible_topo_order includes both nodes. The Light is
        // skipped. The Transform is also skipped (it only emits opcodes
        // when a descendant Primitive includes it in its preamble — there
        // are no primitive descendants here).
        let mut scene = empty_scene();
        let _sphere_id = add_sphere(&mut scene, "Sphere");
        let _ = scene.create_light(crate::graph::scene::LightType::Point);

        let tape = Tape::try_build(&scene).expect("scene with light + transform is encodable");
        // Only the sphere produces an opcode: [EvalPrimitive(sphere), End].
        assert_eq!(tape.opcodes.len(), 2);
        let (k0, _) = unpack(tape.opcodes[0]);
        assert_eq!(k0, TapeOp::EvalPrimitive as u32);
    }

    #[test]
    fn sculpt_node_still_unsupported() {
        let mut scene = empty_scene();
        scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                material: Default::default(),
                layer_intensity: 1.0,
                voxel_grid: crate::graph::voxel::VoxelGrid::new_displacement(
                    32,
                    Vec3::splat(-1.0),
                    Vec3::splat(1.0),
                ),
                desired_resolution: 32,
            },
        );
        assert_eq!(
            Tape::try_build(&scene).unwrap_err(),
            TapeBuildError::UnsupportedSculpt
        );
    }

    #[test]
    fn operation_with_unbound_child_rejected() {
        let mut scene = empty_scene();
        let sphere_id = add_sphere(&mut scene, "Sphere");
        scene.add_node(
            "DanglingUnion".to_string(),
            NodeData::Operation {
                op: CsgOp::Union,
                smooth_k: 0.0,
                steps: 0.0,
                color_blend: -1.0,
                left: Some(sphere_id),
                right: None, // unbound — malformed
            },
        );
        assert_eq!(
            Tape::try_build(&scene).unwrap_err(),
            TapeBuildError::OperationMissingChild
        );
    }

    #[test]
    fn pack_truncates_only_unused_high_bits() {
        // The opcode kind occupies the top 4 bits; payload occupies the
        // bottom 28 bits. The packing must not let payload bits leak into
        // the kind field.
        let word = TapeOp::EvalPrimitive.pack(TAPE_MAX_NODE_IDX);
        let (kind, payload) = unpack(word);
        assert_eq!(kind, TapeOp::EvalPrimitive as u32);
        assert_eq!(payload, TAPE_MAX_NODE_IDX);
    }
}
