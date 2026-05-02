// Universal fallback pick scene_sdf — interprets the tape buffer.
//
// Mirror of `fallback_render.wgsl` for the GPU pick compute pass. The pick
// pipeline rebuild (which happens alongside the render pipeline rebuild on
// every structure change) used to be the reason "click on just-added
// object" missed for ~1s after an add — the unrolled pick shader didn't
// know the new node's slot. With this fallback compiled once at startup,
// pick stays current the moment a new object appears via the render
// fallback. See also `fallback_render.wgsl` and `src/gpu/tape.rs`.
//
// Visual parity contract: the pick fallback's `scene_sdf` must produce
// the same hit/miss decision as the unrolled pick shader for the same
// scene. We achieve that by sharing the encoder (tape.rs) — same opcodes,
// same dispatch tables — so both shader paths fold the scene tree in
// identical post-order with identical CSG `k`/steps/color_blend.

// Render pipeline lays out 5 bind groups (camera/scene/voxel_tex/environment/tape)
// so tape lives at @group(4) there. The pick pipeline only has 4 bind groups
// (camera/scene/pick_io/tape), so for the fallback pick shader the tape
// binding lands at @group(3). The opcode encoding in `tape.rs` is independent
// of the binding slot.
@group(3) @binding(0) var<storage, read> tape: array<u32>;

// Opcode and field constants — keep in lock-step with src/gpu/tape.rs.
const TAPE_OP_END: u32 = 0u;
const TAPE_OP_EVAL_PRIM: u32 = 1u;
const TAPE_OP_COMBINE: u32 = 2u;
const TAPE_OP_APPLY_INV_XFORM: u32 = 3u;
const TAPE_OP_APPLY_POINT_MOD: u32 = 4u;
const TAPE_OP_APPLY_DIST_MOD: u32 = 5u;

const TAPE_PAYLOAD_MASK: u32 = 0x0FFFFFFFu;
const TAPE_OP_SHIFT: u32 = 28u;
const TAPE_MAX_ITER: u32 = 8192u;
const TAPE_STACK_DEPTH: u32 = 16u;

// Primitive kind IDs — match SdfPrimitive::gpu_type_id.
const PRIM_SPHERE: u32 = 0u;
const PRIM_BOX: u32 = 1u;
const PRIM_CYLINDER: u32 = 2u;
const PRIM_TORUS: u32 = 3u;
const PRIM_PLANE: u32 = 4u;
const PRIM_CONE: u32 = 5u;
const PRIM_CAPSULE: u32 = 6u;
const PRIM_ELLIPSOID: u32 = 7u;
const PRIM_HEX_PRISM: u32 = 8u;
const PRIM_PYRAMID: u32 = 9u;

// CSG op IDs — match CsgOp::gpu_op_id.
const OP_UNION: u32 = 10u;
const OP_SMOOTH_UNION: u32 = 11u;
const OP_SUBTRACT: u32 = 12u;
const OP_INTERSECT: u32 = 13u;
const OP_SMOOTH_SUBTRACT: u32 = 14u;
const OP_SMOOTH_INTERSECT: u32 = 15u;
const OP_CHAMFER_UNION: u32 = 16u;
const OP_CHAMFER_SUBTRACT: u32 = 17u;
const OP_CHAMFER_INTERSECT: u32 = 18u;
const OP_STAIRS_UNION: u32 = 19u;
const OP_STAIRS_SUBTRACT: u32 = 20u;
const OP_COLUMNS_UNION: u32 = 21u;
const OP_COLUMNS_SUBTRACT: u32 = 22u;

// Modifier IDs — match ModifierKind::gpu_type_id.
const MOD_TWIST: u32 = 30u;
const MOD_BEND: u32 = 31u;
const MOD_TAPER: u32 = 32u;
const MOD_ROUND: u32 = 33u;
const MOD_ONION: u32 = 34u;
const MOD_ELONGATE: u32 = 35u;
const MOD_MIRROR: u32 = 36u;
const MOD_REPEAT: u32 = 37u;
const MOD_FINITE_REPEAT: u32 = 38u;
const MOD_RADIAL_REPEAT: u32 = 39u;
const MOD_OFFSET: u32 = 40u;
const MOD_NOISE: u32 = 41u;

fn fbp_eval_primitive(p_local: vec3f, idx: u32) -> f32 {
    let lp = rotate_euler(p_local - nodes[idx].position.xyz, nodes[idx].rotation.xyz);
    let s = nodes[idx].scale.xyz;
    let prim = u32(nodes[idx].type_op.x);
    if prim == PRIM_SPHERE { return sdf_sphere(lp, s); }
    if prim == PRIM_BOX { return sdf_box(lp, s); }
    if prim == PRIM_CYLINDER { return sdf_cylinder(lp, s); }
    if prim == PRIM_TORUS { return sdf_torus(lp, s); }
    if prim == PRIM_PLANE { return sdf_plane(lp, s); }
    if prim == PRIM_CONE { return sdf_cone(lp, s); }
    if prim == PRIM_CAPSULE { return sdf_capsule(lp, s); }
    if prim == PRIM_ELLIPSOID { return sdf_ellipsoid(lp, s); }
    if prim == PRIM_HEX_PRISM { return sdf_hex_prism(lp, s); }
    if prim == PRIM_PYRAMID { return sdf_pyramid(lp, s); }
    return 1e10;
}

fn fbp_apply_inv_xform(p: vec3f, idx: u32) -> vec3f {
    let t = p - nodes[idx].position.xyz;
    let r = rotate_euler(t, nodes[idx].rotation.xyz);
    return r / nodes[idx].scale.xyz;
}

fn fbp_apply_point_mod(p: vec3f, idx: u32) -> vec3f {
    let kind = u32(nodes[idx].type_op.x);
    let pos = nodes[idx].position.xyz;
    let extra = nodes[idx].rotation.xyz;
    if kind == MOD_TWIST { return twist_point(p, pos.x); }
    if kind == MOD_BEND { return bend_point(p, pos.x); }
    if kind == MOD_TAPER { return taper_point(p, pos.x); }
    if kind == MOD_ELONGATE { return elongate_point(p, pos); }
    if kind == MOD_MIRROR { return mirror_point(p, pos); }
    if kind == MOD_REPEAT { return repeat_point(p, pos); }
    if kind == MOD_FINITE_REPEAT { return finite_repeat_point(p, pos, extra); }
    if kind == MOD_RADIAL_REPEAT { return radial_repeat_point(p, pos.x, pos.y); }
    if kind == MOD_NOISE { return p + fbm_noise(p, pos.x, pos.y, i32(pos.z)); }
    return p;
}

fn fbp_apply_dist_mod(top: vec4f, idx: u32) -> vec4f {
    let kind = u32(nodes[idx].type_op.x);
    let pp = nodes[idx].position.x;
    if kind == MOD_ROUND { return vec4f(top.x - pp, top.y, top.z, top.w); }
    if kind == MOD_ONION { return vec4f(abs(top.x) - pp, top.y, top.z, top.w); }
    if kind == MOD_OFFSET { return vec4f(top.x + pp, top.y, top.z, top.w); }
    return top;
}

fn fbp_combine(a: vec4f, b: vec4f, idx: u32) -> vec4f {
    let op_id = u32(nodes[idx].type_op.x);
    let k = nodes[idx].type_op.y;
    let steps = nodes[idx].type_op.z;
    let cb = nodes[idx].type_op.w;
    if op_id == OP_UNION { return op_union(a, b, k); }
    if op_id == OP_SMOOTH_UNION { return op_smooth_union(a, b, k, cb); }
    if op_id == OP_SUBTRACT { return op_subtract(a, b, k, cb); }
    if op_id == OP_INTERSECT { return op_intersect(a, b, k, cb); }
    if op_id == OP_SMOOTH_SUBTRACT { return op_subtract(a, b, k, cb); }
    if op_id == OP_SMOOTH_INTERSECT { return op_intersect(a, b, k, cb); }
    if op_id == OP_CHAMFER_UNION { return op_chamfer_union(a, b, k, cb); }
    if op_id == OP_CHAMFER_SUBTRACT { return op_chamfer_subtract(a, b, k, cb); }
    if op_id == OP_CHAMFER_INTERSECT { return op_chamfer_intersect(a, b, k, cb); }
    if op_id == OP_STAIRS_UNION { return op_stairs_union(a, b, k, steps, cb); }
    if op_id == OP_STAIRS_SUBTRACT { return op_stairs_subtract(a, b, k, steps, cb); }
    if op_id == OP_COLUMNS_UNION { return op_columns_union(a, b, k, steps, cb); }
    if op_id == OP_COLUMNS_SUBTRACT { return op_columns_subtract(a, b, k, steps, cb); }
    return op_union(a, b, 0.0);
}

fn scene_sdf(p_world: vec3f) -> vec4f {
    var stack: array<vec4f, 16>;
    var sp: u32 = 0u;
    var p_local: vec3f = p_world;

    let tape_len = arrayLength(&tape);

    var i: u32 = 0u;
    loop {
        if i >= tape_len { break; }
        if i >= TAPE_MAX_ITER { break; }

        let word = tape[i];
        let kind = word >> TAPE_OP_SHIFT;
        let idx = word & TAPE_PAYLOAD_MASK;

        if kind == TAPE_OP_END { break; }

        if kind == TAPE_OP_EVAL_PRIM {
            let d = fbp_eval_primitive(p_local, idx);
            if sp < TAPE_STACK_DEPTH {
                stack[sp] = vec4f(d, f32(idx), -1.0, 0.0);
                sp = sp + 1u;
            }
            p_local = p_world;
        } else if kind == TAPE_OP_COMBINE {
            if sp >= 2u {
                let a = stack[sp - 2u];
                let b = stack[sp - 1u];
                stack[sp - 2u] = fbp_combine(a, b, idx);
                sp = sp - 1u;
            }
        } else if kind == TAPE_OP_APPLY_INV_XFORM {
            p_local = fbp_apply_inv_xform(p_local, idx);
        } else if kind == TAPE_OP_APPLY_POINT_MOD {
            p_local = fbp_apply_point_mod(p_local, idx);
        } else if kind == TAPE_OP_APPLY_DIST_MOD {
            if sp > 0u {
                stack[sp - 1u] = fbp_apply_dist_mod(stack[sp - 1u], idx);
            }
        }

        i = i + 1u;
    }

    if sp == 0u {
        return vec4f(1e10, -1.0, -1.0, 0.0);
    }
    return stack[sp - 1u];
}
