use std::collections::{HashMap, HashSet};

use crate::graph::scene::{ModifierKind, NodeData, NodeId, Scene, SceneNode};
use crate::settings::RenderConfig;

use super::buffers::collect_sculpt_tex_info;
use super::shader_templates::{
    apply_march_placeholders, build_postlude, compute_prelude, format_f32, format_vec3,
    render_prelude, COMPOSITE_COMPUTE_ENTRY, PICK,
};

// ---------------------------------------------------------------------------
// Voxel texture declarations
// ---------------------------------------------------------------------------

/// Generate WGSL texture declarations and per-sculpt sampling functions.
/// Returns (wgsl_code, node_id→tex_idx map).
fn generate_voxel_texture_decls(scene: &Scene) -> (String, HashMap<NodeId, usize>) {
    let infos = collect_sculpt_tex_info(scene);
    if infos.is_empty() {
        return (String::new(), HashMap::new());
    }

    let mut lines = Vec::new();
    lines.push("@group(2) @binding(0) var voxel_sampler: sampler;".to_string());

    let mut tex_map = HashMap::new();
    for info in &infos {
        let i = info.tex_idx;
        let binding = i + 1;
        lines.push(format!(
            "@group(2) @binding({binding}) var voxel_tex_{i}: texture_3d<f32>;"
        ));
        tex_map.insert(info.node_id, i);
    }

    lines.push(String::new());

    // Per-sculpt sampling function (differential or total-SDF depending on has_input)
    for info in &infos {
        let i = info.tex_idx;
        if info.has_input {
            // Differential sculpt: displacement-only sampling (returns 0 outside grid)
            lines.push(format!("fn disp_voxel_tex_{i}(local_p: vec3f, node_idx: u32) -> f32 {{"));
            lines.push("    let bmin = nodes[node_idx].extra1.xyz;".to_string());
            lines.push("    let bmax = nodes[node_idx].extra2.xyz;".to_string());
            lines.push("    let norm = (local_p - bmin) / (bmax - bmin);".to_string());
            lines.push("    if any(norm < vec3f(0.0)) || any(norm > vec3f(1.0)) { return 0.0; }".to_string());
            lines.push(format!(
                "    return textureSampleLevel(voxel_tex_{i}, voxel_sampler, norm, 0.0).x;"
            ));
            lines.push("}".to_string());
        } else {
            // Standalone sculpt: total-SDF sampling (clamp to edge + box_dist)
            lines.push(format!("fn sdf_voxel_tex_{i}(local_p: vec3f, node_idx: u32) -> f32 {{"));
            lines.push("    let bmin = nodes[node_idx].extra1.xyz;".to_string());
            lines.push("    let bmax = nodes[node_idx].extra2.xyz;".to_string());
            lines.push("    let clamped = clamp(local_p, bmin, bmax);".to_string());
            lines.push("    let box_dist = length(local_p - clamped);".to_string());
            lines.push("    let uv = (clamped - bmin) / (bmax - bmin);".to_string());
            lines.push(format!(
                "    return textureSampleLevel(voxel_tex_{i}, voxel_sampler, uv, 0.0).x + box_dist;"
            ));
            lines.push("}".to_string());
        }
        lines.push(String::new());
    }

    (lines.join("\n"), tex_map)
}

// ---------------------------------------------------------------------------
// Shader generation (public API)
// ---------------------------------------------------------------------------

pub fn generate_shader(
    scene: &Scene,
    config: &RenderConfig,
) -> String {
    let (tex_decls, sculpt_tex_map) = generate_voxel_texture_decls(scene);
    let tex_map = if sculpt_tex_map.is_empty() { None } else { Some(&sculpt_tex_map) };
    let scene_sdf = generate_scene_sdf(scene, tex_map);
    let postlude = build_postlude(config);
    format!("{}\n{}\n{}\n{}", render_prelude(), tex_decls, scene_sdf, postlude)
}

pub fn generate_pick_shader(scene: &Scene, config: &RenderConfig) -> String {
    let scene_sdf = generate_scene_sdf(scene, None);
    let pick_postlude = apply_march_placeholders(PICK, config);
    format!("{}\n{}\n{}", compute_prelude(), scene_sdf, pick_postlude)
}

/// Generate the composite compute shader that pre-evaluates scene_sdf at every voxel
/// in a 3D grid and writes SDF + material ID to storage textures.
pub fn generate_composite_shader(scene: &Scene, _config: &RenderConfig) -> String {
    let (tex_decls, sculpt_tex_map) = generate_voxel_texture_decls(scene);
    let tex_map = if sculpt_tex_map.is_empty() { None } else { Some(&sculpt_tex_map) };
    let scene_sdf = generate_scene_sdf(scene, tex_map);
    format!("{}\n{}\n{}\n{}", compute_prelude(), tex_decls, scene_sdf, COMPOSITE_COMPUTE_ENTRY)
}

/// Generate the composite render shader that reads the pre-composited scene volume.
/// `scene_sdf()` becomes a single texture lookup regardless of scene complexity.
pub fn generate_composite_render_shader(
    config: &RenderConfig,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> String {
    let comp_scene_sdf = format!(
        r#"
@group(2) @binding(0) var comp_sampler: sampler;
@group(2) @binding(1) var comp_sdf_tex: texture_3d<f32>;
@group(2) @binding(2) var comp_mat_tex: texture_3d<u32>;
@group(2) @binding(3) var comp_normal_tex: texture_3d<f32>;

const COMP_BMIN: vec3f = vec3f({bmin});
const COMP_BMAX: vec3f = vec3f({bmax});

fn comp_to_uv(p: vec3f) -> vec3f {{
    return clamp((p - COMP_BMIN) / (COMP_BMAX - COMP_BMIN), vec3f(0.0), vec3f(1.0));
}}

fn scene_sdf(p: vec3f) -> vec4f {{
    let size = COMP_BMAX - COMP_BMIN;
    let norm = (p - COMP_BMIN) / size;

    // Outside bounds: return distance to AABB
    if any(norm < vec3f(-0.01)) || any(norm > vec3f(1.01)) {{
        return vec4f(length(max(p - COMP_BMAX, COMP_BMIN - p)), -1.0, -1.0, 0.0);
    }}

    let uv = clamp(norm, vec3f(0.0), vec3f(1.0));
    let d = textureSampleLevel(comp_sdf_tex, comp_sampler, uv, 0.0).x;
    let dims = textureDimensions(comp_mat_tex);
    let fc = clamp(uv * vec3f(dims), vec3f(0.0), vec3f(dims - vec3u(1u)));
    let ic = vec3u(fc);
    let mat_raw = textureLoad(comp_mat_tex, ic, 0).x;
    let mat_id = f32(mat_raw) - 1.0;
    return vec4f(d, mat_id, -1.0, 0.0);
}}
"#,
        bmin = format_vec3(bounds_min),
        bmax = format_vec3(bounds_max),
    );
    let mut postlude = build_postlude(config);

    // Replace calc_normal with precomputed normal texture lookup
    let old_calc_normal = r#"fn calc_normal(p: vec3f, t: f32) -> vec3f {
    // Tetrahedron technique: 4 SDF evals instead of 6.
    // Distance-adaptive epsilon: larger at distance (reduces aliasing), tighter up close (more detail).
    let e = clamp(0.001 * t, 0.0005, 0.05);
    let k = vec2f(1.0, -1.0);
    return normalize(
        k.xyy * scene_sdf(p + k.xyy * e).x +
        k.yyx * scene_sdf(p + k.yyx * e).x +
        k.yxy * scene_sdf(p + k.yxy * e).x +
        k.xxx * scene_sdf(p + k.xxx * e).x
    );
}"#;
    let new_calc_normal = r#"fn calc_normal(p: vec3f, t: f32) -> vec3f {
    // Precomputed normal from composite volume (single texture lookup).
    let uv = comp_to_uv(p);
    return normalize(textureSampleLevel(comp_normal_tex, comp_sampler, uv, 0.0).xyz);
}"#;
    if postlude.contains(old_calc_normal) {
        postlude = postlude.replace(old_calc_normal, new_calc_normal);
    } else {
        log::warn!("Composite render: calc_normal string replace FAILED — using analytical normals as fallback");
    }

    format!("{}\n{}\n{}", render_prelude(), comp_scene_sdf, postlude)
}

// ---------------------------------------------------------------------------
// Transform chain helpers
// ---------------------------------------------------------------------------

/// Chain entry: either a Transform or a point-modifying Modifier.
#[derive(Clone)]
enum ChainEntry {
    Transform,
    Modifier(ModifierKind),
}

/// Walk up from `node_id` through ancestors, collecting all Transform and
/// point-modifying Modifier ancestors. Returns chain from innermost to outermost.
fn get_transform_chain(
    node_id: NodeId,
    parent_map: &HashMap<NodeId, NodeId>,
    scene: &Scene,
    idx_map: &HashMap<NodeId, usize>,
) -> Vec<(usize, ChainEntry)> {
    let mut chain = Vec::new();
    let mut current = node_id;
    while let Some(&parent_id) = parent_map.get(&current) {
        if let Some(node) = scene.nodes.get(&parent_id) {
            if let Some(&idx) = idx_map.get(&parent_id) {
                match &node.data {
                    NodeData::Transform { .. } => {
                        chain.push((idx, ChainEntry::Transform));
                    }
                    NodeData::Modifier { kind, .. } if kind.is_point_modifier() => {
                        chain.push((idx, ChainEntry::Modifier(kind.clone())));
                    }
                    _ => {}
                }
            }
        }
        current = parent_id;
    }
    chain
}

/// Emit WGSL code for a transform chain. Returns the final point variable name.
/// Chain is innermost-first; we process outermost-first (reversed).
fn emit_transform_chain(
    lines: &mut Vec<String>,
    node_idx: usize,
    chain: &[(usize, ChainEntry)],
) -> String {
    if chain.is_empty() {
        return "p".to_string();
    }
    let mut current_var = "p".to_string();
    for (step, (idx, entry)) in chain.iter().rev().enumerate() {
        let new_var = format!("tp{}_{}", node_idx, step);
        match entry {
            ChainEntry::Transform => {
                // Inverse TRS: subtract T, inverse-rotate R, divide S
                let t_var = format!("tp{}_{}_t", node_idx, step);
                let r_var = format!("tp{}_{}_r", node_idx, step);
                lines.push(format!(
                    "    let {t_var} = {current_var} - nodes[{idx}].position.xyz;"
                ));
                lines.push(format!(
                    "    let {r_var} = rotate_euler({t_var}, nodes[{idx}].rotation.xyz);"
                ));
                lines.push(format!(
                    "    let {new_var} = {r_var} / nodes[{idx}].scale.xyz;"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Twist) => {
                lines.push(format!(
                    "    let {new_var} = twist_point({current_var}, nodes[{idx}].position.x);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Bend) => {
                lines.push(format!(
                    "    let {new_var} = bend_point({current_var}, nodes[{idx}].position.x);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Taper) => {
                lines.push(format!(
                    "    let {new_var} = taper_point({current_var}, nodes[{idx}].position.x);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Elongate) => {
                lines.push(format!(
                    "    let {new_var} = elongate_point({current_var}, nodes[{idx}].position.xyz);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Mirror) => {
                lines.push(format!(
                    "    let {new_var} = mirror_point({current_var}, nodes[{idx}].position.xyz);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Repeat) => {
                lines.push(format!(
                    "    let {new_var} = repeat_point({current_var}, nodes[{idx}].position.xyz);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::FiniteRepeat) => {
                lines.push(format!(
                    "    let {new_var} = finite_repeat_point({current_var}, nodes[{idx}].position.xyz, nodes[{idx}].rotation.xyz);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::RadialRepeat) => {
                lines.push(format!(
                    "    let {new_var} = radial_repeat_point({current_var}, nodes[{idx}].position.x, nodes[{idx}].position.y);"
                ));
            }
            ChainEntry::Modifier(ModifierKind::Noise) => {
                lines.push(format!(
                    "    let {new_var} = {current_var} + fbm_noise({current_var}, nodes[{idx}].position.x, nodes[{idx}].position.y, i32(nodes[{idx}].position.z));"
                ));
            }
            // Round/Onion/Offset are distance modifiers, not in chain
            ChainEntry::Modifier(ModifierKind::Round | ModifierKind::Onion | ModifierKind::Offset) => unreachable!(),
        }
        current_var = new_var;
    }
    current_var
}

// ---------------------------------------------------------------------------
// Node WGSL emission
// ---------------------------------------------------------------------------

/// Emit WGSL for a single node. Extracted so both cheap and expensive phases can use it.
/// `sculpt_tex_map`: if Some, sculpt nodes use `sdf_voxel_tex_N` (texture path);
///                    if None, they use `sdf_voxel_grid` (storage buffer path).
#[allow(clippy::too_many_arguments)]
fn emit_node_wgsl(
    lines: &mut Vec<String>,
    i: usize,
    node_id: NodeId,
    node: &SceneNode,
    parent_map: &HashMap<NodeId, NodeId>,
    scene: &Scene,
    idx_map: &HashMap<NodeId, usize>,
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) {
    match &node.data {
        NodeData::Primitive { kind, .. } => {
            let chain = get_transform_chain(node_id, parent_map, scene, idx_map);
            let point_var = emit_transform_chain(lines, i, &chain);
            lines.push(format!(
                "    let lp{i} = rotate_euler({point_var} - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
            ));
            let sdf_fn = kind.sdf_function_name();
            lines.push(format!(
                "    let n{i} = vec4f({sdf_fn}(lp{i}, nodes[{i}].scale.xyz), f32({i}), -1.0, 0.0);"
            ));
        }
        NodeData::Operation { op, left, right, .. } => {
            let li = left.and_then(|id| idx_map.get(&id).copied());
            let ri = right.and_then(|id| idx_map.get(&id).copied());
            match (li, ri) {
                (Some(li), Some(ri)) => {
                    let op_fn = op.wgsl_function_name();
                    lines.push(format!(
                        "    let n{i} = {op_fn}(n{li}, n{ri}, nodes[{i}].type_op.y);"
                    ));
                }
                (Some(ci), None) | (None, Some(ci)) => {
                    lines.push(format!("    let n{i} = n{ci};"));
                }
                (None, None) => {
                    lines.push(format!("    let n{i} = vec4f(1e10, -1.0, -1.0, 0.0);"));
                }
            }
        }
        NodeData::Sculpt { input, .. } => {
            let chain = get_transform_chain(node_id, parent_map, scene, idx_map);
            let point_var = emit_transform_chain(lines, i, &chain);
            lines.push(format!(
                "    let lp{i} = rotate_euler({point_var} - nodes[{i}].position.xyz, nodes[{i}].rotation.xyz);"
            ));

            let child_idx = input.and_then(|id| idx_map.get(&id).copied());

            if let Some(ci) = child_idx {
                // DIFFERENTIAL: analytical child SDF + displacement from grid * layer_intensity
                let disp_call = if let Some(tex_map) = sculpt_tex_map {
                    if let Some(&tex_idx) = tex_map.get(&node_id) {
                        format!("disp_voxel_tex_{tex_idx}(lp{i}, {i}u)")
                    } else {
                        format!("disp_voxel_grid(lp{i}, {i}u)")
                    }
                } else {
                    format!("disp_voxel_grid(lp{i}, {i}u)")
                };
                lines.push(format!(
                    "    let n{i} = vec4f(n{ci}.x + {disp_call} * nodes[{i}].position.w, f32({i}), -1.0, 0.0);"
                ));
            } else {
                // STANDALONE: total SDF from grid (unchanged behavior)
                let sdf_call = if let Some(tex_map) = sculpt_tex_map {
                    if let Some(&tex_idx) = tex_map.get(&node_id) {
                        format!("sdf_voxel_tex_{tex_idx}(lp{i}, {i}u)")
                    } else {
                        format!("sdf_voxel_grid(lp{i}, {i}u)")
                    }
                } else {
                    format!("sdf_voxel_grid(lp{i}, {i}u)")
                };
                lines.push(format!(
                    "    let n{i} = vec4f({sdf_call}, f32({i}), -1.0, 0.0);"
                ));
            }
        }
        NodeData::Transform { input, .. } => {
            let child_idx = input.and_then(|id| idx_map.get(&id).copied());
            if let Some(ci) = child_idx {
                // Always apply scale distance correction, propagate material blend info
                lines.push(format!(
                    "    let n{i} = vec4f(n{ci}.x * min(nodes[{i}].scale.x, min(nodes[{i}].scale.y, nodes[{i}].scale.z)), n{ci}.y, n{ci}.z, n{ci}.w);"
                ));
            } else {
                lines.push(format!("    let n{i} = vec4f(1e10, -1.0, -1.0, 0.0);"));
            }
        }
        NodeData::Modifier { kind, input, .. } => {
            let child_idx = input.and_then(|id| idx_map.get(&id).copied());
            if let Some(ci) = child_idx {
                match kind {
                    ModifierKind::Round => {
                        // Modify distance, propagate material blend info
                        lines.push(format!(
                            "    let n{i} = vec4f(n{ci}.x - nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
                        ));
                    }
                    ModifierKind::Onion => {
                        // Modify distance, propagate material blend info
                        lines.push(format!(
                            "    let n{i} = vec4f(abs(n{ci}.x) - nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
                        ));
                    }
                    ModifierKind::Offset => {
                        // Add offset to distance, propagate material blend info
                        lines.push(format!(
                            "    let n{i} = vec4f(n{ci}.x + nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
                        ));
                    }
                    // Point modifiers just pass through (transform chain handles the point)
                    _ => {
                        lines.push(format!("    let n{i} = n{ci};"));
                    }
                }
            } else {
                lines.push(format!("    let n{i} = vec4f(1e10, -1.0, -1.0, 0.0);"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scene SDF generation
// ---------------------------------------------------------------------------

fn generate_scene_sdf(
    scene: &Scene,
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) -> String {
    let order = scene.visible_topo_order();
    if order.is_empty() {
        return "fn scene_sdf(p: vec3f) -> vec4f {\n    return vec4f(1e10, -1.0, -1.0, 0.0);\n}"
            .to_string();
    }

    let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let parent_map = scene.build_parent_map();
    let tops = scene.top_level_nodes();

    // Classify top-level subtrees: cheap (no sculpt) vs expensive (has sculpt)
    let mut cheap_tops = Vec::new();
    let mut expensive_tops = Vec::new();
    for &top_id in &tops {
        if scene.subtree_has_sculpt(top_id) {
            expensive_tops.push(top_id);
        } else {
            cheap_tops.push(top_id);
        }
    }

    // No sculpts at all: use flat codegen (no bounding skip needed)
    if expensive_tops.is_empty() {
        return generate_scene_sdf_flat(scene, &order, &idx_map, &parent_map, &tops, sculpt_tex_map);
    }

    // Only one expensive subtree and no cheap tops: flat codegen (no skip benefit)
    if expensive_tops.len() == 1 && cheap_tops.is_empty() {
        return generate_scene_sdf_flat(scene, &order, &idx_map, &parent_map, &tops, sculpt_tex_map);
    }

    // Two-phase codegen with bounding skip for expensive subtrees
    let mut lines = Vec::new();
    lines.push("fn scene_sdf(p: vec3f) -> vec4f {".to_string());

    // Phase 1: Emit all cheap subtree nodes unconditionally
    let cheap_node_set: HashSet<NodeId> = cheap_tops.iter()
        .flat_map(|&id| scene.collect_subtree(id))
        .collect();

    for (i, &node_id) in order.iter().enumerate() {
        if !cheap_node_set.contains(&node_id) { continue; }
        let Some(node) = scene.nodes.get(&node_id) else { continue; };
        emit_node_wgsl(&mut lines, i, node_id, node, &parent_map, scene, &idx_map, sculpt_tex_map);
    }

    // Initialize result from cheap tops
    let cheap_indices: Vec<usize> = cheap_tops.iter()
        .filter_map(|id| idx_map.get(id).copied())
        .collect();
    if cheap_indices.is_empty() {
        lines.push("    var result = vec4f(1e10, -1.0, -1.0, 0.0);".to_string());
    } else {
        lines.push(format!("    var result = n{};", cheap_indices[0]));
        for &idx in &cheap_indices[1..] {
            lines.push(format!("    result = op_union(result, n{idx}, 0.0);"));
        }
    }

    // Phase 2: Emit expensive subtrees wrapped in bounding sphere check
    for &top_id in &expensive_tops {
        let (center, radius) = scene.compute_subtree_sphere(top_id, &parent_map);
        let subtree_nodes = scene.collect_subtree(top_id);

        lines.push(format!(
            "    {{ let _bd = length(p - vec3f({}, {}, {})) - {};",
            format_f32(center[0]), format_f32(center[1]), format_f32(center[2]),
            format_f32(radius),
        ));
        lines.push("    if _bd < result.x {".to_string());

        // Emit all nodes in this subtree (preserving topo order)
        for (i, &node_id) in order.iter().enumerate() {
            if !subtree_nodes.contains(&node_id) { continue; }
            let Some(node) = scene.nodes.get(&node_id) else { continue; };
            emit_node_wgsl(&mut lines, i, node_id, node, &parent_map, scene, &idx_map, sculpt_tex_map);
        }

        // Union this subtree's root with the result
        if let Some(&top_idx) = idx_map.get(&top_id) {
            lines.push(format!("        result = op_union(result, n{top_idx}, 0.0);"));
        }
        lines.push("    } }".to_string());
    }

    lines.push("    return result;".to_string());
    lines.push("}".to_string());

    lines.join("\n")
}

/// Original flat codegen (no expensive subtrees).
fn generate_scene_sdf_flat(
    scene: &Scene,
    order: &[NodeId],
    idx_map: &HashMap<NodeId, usize>,
    parent_map: &HashMap<NodeId, NodeId>,
    tops: &[NodeId],
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) -> String {
    let mut lines = Vec::new();
    lines.push("fn scene_sdf(p: vec3f) -> vec4f {".to_string());

    for (i, &node_id) in order.iter().enumerate() {
        let Some(node) = scene.nodes.get(&node_id) else { continue; };
        emit_node_wgsl(&mut lines, i, node_id, node, parent_map, scene, idx_map, sculpt_tex_map);
    }

    let top_indices: Vec<usize> = tops
        .iter()
        .filter_map(|id| idx_map.get(id).copied())
        .collect();
    match top_indices.len() {
        0 => lines.push("    return vec4f(1e10, -1.0, -1.0, 0.0);".to_string()),
        1 => lines.push(format!("    return n{};", top_indices[0])),
        _ => {
            lines.push(format!("    var result = n{};", top_indices[0]));
            for &idx in &top_indices[1..] {
                lines.push(format!("    result = op_union(result, n{idx}, 0.0);"));
            }
            lines.push("    return result;".to_string());
        }
    }
    lines.push("}".to_string());

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::scene::{CsgOp, ModifierKind, SdfPrimitive};
    use crate::graph::voxel::VoxelGrid;
    use glam::Vec3;
    use std::collections::HashSet;

    /// Create an empty scene (no default sphere) for predictable testing.
    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_scene_sdf — empty scene
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn empty_scene_returns_far_sentinel() {
        let scene = empty_scene();
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("fn scene_sdf(p: vec3f) -> vec4f"));
        assert!(wgsl.contains("1e10"));
        assert!(wgsl.contains("-1.0"));
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_scene_sdf — single primitive
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn single_sphere_generates_sdf_sphere_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_sphere"));
        assert!(wgsl.contains("fn scene_sdf"));
    }

    #[test]
    fn single_box_generates_sdf_box_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Box);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_box"));
    }

    #[test]
    fn single_cylinder_generates_sdf_cylinder_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Cylinder);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_cylinder"));
    }

    #[test]
    fn single_torus_generates_sdf_torus_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Torus);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_torus"));
    }

    #[test]
    fn single_plane_generates_sdf_plane_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Plane);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_plane"));
    }

    #[test]
    fn single_cone_generates_sdf_cone_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Cone);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_cone"));
    }

    #[test]
    fn single_capsule_generates_sdf_capsule_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Capsule);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_capsule"));
    }

    #[test]
    fn single_ellipsoid_generates_sdf_ellipsoid_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Ellipsoid);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_ellipsoid"));
    }

    #[test]
    fn single_hex_prism_generates_sdf_hex_prism_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::HexPrism);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_hex_prism"));
    }

    #[test]
    fn single_pyramid_generates_sdf_pyramid_call() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Pyramid);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_pyramid"));
    }

    #[test]
    fn single_primitive_uses_rotate_euler() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("rotate_euler"));
    }

    #[test]
    fn single_primitive_returns_directly() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        // Single top-level node → direct return, no `var result`
        assert!(wgsl.contains("return n0;"));
        assert!(!wgsl.contains("var result"));
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_scene_sdf — CSG operations
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn union_operation_generates_op_union() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        scene.create_operation(CsgOp::Union, Some(left), Some(right));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("op_union(n"));
    }

    #[test]
    fn smooth_union_generates_op_smooth_union() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        scene.create_operation(CsgOp::SmoothUnion, Some(left), Some(right));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("op_smooth_union(n"));
    }

    #[test]
    fn subtract_operation_generates_op_subtract() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        scene.create_operation(CsgOp::Subtract, Some(left), Some(right));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("op_subtract(n"));
    }

    #[test]
    fn intersect_operation_generates_op_intersect() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        scene.create_operation(CsgOp::Intersect, Some(left), Some(right));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("op_intersect(n"));
    }

    #[test]
    fn operation_with_one_child_passes_through() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let op = scene.create_operation(CsgOp::Union, Some(left), None);
        let wgsl = generate_scene_sdf(&scene, None);
        let op_idx = scene.visible_topo_order().iter().position(|&id| id == op).unwrap();
        let left_idx = scene.visible_topo_order().iter().position(|&id| id == left).unwrap();
        assert!(wgsl.contains(&format!("let n{op_idx} = n{left_idx};")));
    }

    #[test]
    fn operation_with_no_children_returns_far() {
        let mut scene = empty_scene();
        let op = scene.create_operation(CsgOp::Union, None, None);
        let wgsl = generate_scene_sdf(&scene, None);
        let op_idx = scene.visible_topo_order().iter().position(|&id| id == op).unwrap();
        assert!(wgsl.contains(&format!("let n{op_idx} = vec4f(1e10, -1.0, -1.0, 0.0);")));
    }

    #[test]
    fn operation_smooth_k_uses_type_op_y() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        scene.create_operation(CsgOp::SmoothUnion, Some(left), Some(right));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("type_op.y)"));
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_scene_sdf — multiple top-level nodes
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn two_top_level_primitives_union_with_var_result() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_primitive(SdfPrimitive::Box);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("var result"));
        assert!(wgsl.contains("op_union(result"));
        assert!(wgsl.contains("return result;"));
    }

    #[test]
    fn three_top_level_primitives_chain_unions() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_primitive(SdfPrimitive::Box);
        scene.create_primitive(SdfPrimitive::Cylinder);
        let wgsl = generate_scene_sdf(&scene, None);
        // Should have two op_union calls (one for 2nd, one for 3rd)
        let union_count = wgsl.matches("op_union(result").count();
        assert_eq!(union_count, 2);
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_scene_sdf — hidden nodes excluded
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn hidden_node_excluded_from_codegen() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_primitive(SdfPrimitive::Box);
        scene.toggle_visibility(sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(!wgsl.contains("sdf_sphere"));
        assert!(wgsl.contains("sdf_box"));
    }

    #[test]
    fn all_hidden_returns_far_sentinel() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.toggle_visibility(sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("1e10"));
        assert!(wgsl.contains("-1.0"));
        assert!(!wgsl.contains("sdf_sphere"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Transform nodes
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn transform_wrapping_primitive_generates_inverse_trs() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_transform(Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        // Transform chain should produce translated/rotated/scaled point
        assert!(wgsl.contains("position.xyz"));
        assert!(wgsl.contains("rotation.xyz"));
        assert!(wgsl.contains("scale.xyz"));
    }

    #[test]
    fn transform_with_no_child_returns_far() {
        let mut scene = empty_scene();
        let xform = scene.create_transform(None);
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let xform_idx = order.iter().position(|&id| id == xform).unwrap();
        assert!(wgsl.contains(&format!("let n{xform_idx} = vec4f(1e10, -1.0, -1.0, 0.0);")));
    }

    #[test]
    fn transform_applies_scale_distance_correction() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let xform = scene.create_transform(Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let xform_idx = order.iter().position(|&id| id == xform).unwrap();
        // Scale correction: min(scale.x, min(scale.y, scale.z))
        assert!(wgsl.contains(&format!("let n{xform_idx} = vec4f(n")));
        assert!(wgsl.contains("min(nodes["));
    }

    #[test]
    fn nested_transforms_chain_correctly() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let inner_xform = scene.create_transform(Some(sphere));
        scene.create_transform(Some(inner_xform));
        let wgsl = generate_scene_sdf(&scene, None);
        // Sphere should see two transform chain steps
        // tp<sphere_idx>_0 and tp<sphere_idx>_1
        assert!(wgsl.contains("tp0_0"));
        assert!(wgsl.contains("tp0_1"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Modifier nodes — distance modifiers (Round, Onion)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn round_modifier_subtracts_value() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier = scene.create_modifier(ModifierKind::Round, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let mod_idx = order.iter().position(|&id| id == modifier).unwrap();
        let sph_idx = order.iter().position(|&id| id == sphere).unwrap();
        // Round: n<mod> = vec2f(n<child>.x - nodes[<mod>].position.x, n<child>.y)
        assert!(wgsl.contains(&format!("let n{mod_idx} = vec4f(n{sph_idx}.x - nodes[{mod_idx}].position.x")));
    }

    #[test]
    fn onion_modifier_uses_abs() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier = scene.create_modifier(ModifierKind::Onion, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let mod_idx = order.iter().position(|&id| id == modifier).unwrap();
        let sph_idx = order.iter().position(|&id| id == sphere).unwrap();
        assert!(wgsl.contains(&format!("abs(n{sph_idx}.x) - nodes[{mod_idx}].position.x")));
    }

    #[test]
    fn offset_modifier_adds_value() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier = scene.create_modifier(ModifierKind::Offset, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let mod_idx = order.iter().position(|&id| id == modifier).unwrap();
        let sph_idx = order.iter().position(|&id| id == sphere).unwrap();
        assert!(wgsl.contains(&format!("let n{mod_idx} = vec4f(n{sph_idx}.x + nodes[{mod_idx}].position.x")));
    }

    #[test]
    fn modifier_with_no_child_returns_far() {
        let mut scene = empty_scene();
        let modifier = scene.create_modifier(ModifierKind::Round, None);
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let mod_idx = order.iter().position(|&id| id == modifier).unwrap();
        assert!(wgsl.contains(&format!("let n{mod_idx} = vec4f(1e10, -1.0, -1.0, 0.0);")));
    }

    // ═══════════════════════════════════════════════════════════════
    // Modifier nodes — point modifiers (in transform chain)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn twist_modifier_emits_twist_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Twist, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("twist_point("));
    }

    #[test]
    fn bend_modifier_emits_bend_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Bend, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("bend_point("));
    }

    #[test]
    fn taper_modifier_emits_taper_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Taper, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("taper_point("));
    }

    #[test]
    fn elongate_modifier_emits_elongate_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Elongate, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("elongate_point("));
    }

    #[test]
    fn mirror_modifier_emits_mirror_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Mirror, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("mirror_point("));
    }

    #[test]
    fn repeat_modifier_emits_repeat_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Repeat, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("repeat_point("));
    }

    #[test]
    fn finite_repeat_modifier_emits_finite_repeat_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::FiniteRepeat, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("finite_repeat_point("));
    }

    #[test]
    fn radial_repeat_modifier_emits_radial_repeat_point() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::RadialRepeat, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("radial_repeat_point("));
    }

    #[test]
    fn noise_modifier_emits_fbm_noise() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Noise, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("fbm_noise("));
    }

    #[test]
    fn point_modifier_passes_through_distance() {
        // Point modifiers don't modify the distance — they just pass n<child> through
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier = scene.create_modifier(ModifierKind::Twist, Some(sphere));
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let mod_idx = order.iter().position(|&id| id == modifier).unwrap();
        let sph_idx = order.iter().position(|&id| id == sphere).unwrap();
        assert!(wgsl.contains(&format!("let n{mod_idx} = n{sph_idx};")));
    }

    // ═══════════════════════════════════════════════════════════════
    // get_transform_chain
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn transform_chain_empty_for_top_level_primitive() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let parent_map = scene.build_parent_map();
        let order = scene.visible_topo_order();
        let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let chain = get_transform_chain(sphere, &parent_map, &scene, &idx_map);
        assert!(chain.is_empty());
    }

    #[test]
    fn transform_chain_includes_transform_ancestor() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_transform(Some(sphere));
        let parent_map = scene.build_parent_map();
        let order = scene.visible_topo_order();
        let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let chain = get_transform_chain(sphere, &parent_map, &scene, &idx_map);
        assert_eq!(chain.len(), 1);
        assert!(matches!(chain[0].1, ChainEntry::Transform));
    }

    #[test]
    fn transform_chain_includes_point_modifier_ancestor() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Twist, Some(sphere));
        let parent_map = scene.build_parent_map();
        let order = scene.visible_topo_order();
        let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let chain = get_transform_chain(sphere, &parent_map, &scene, &idx_map);
        assert_eq!(chain.len(), 1);
        assert!(matches!(chain[0].1, ChainEntry::Modifier(ModifierKind::Twist)));
    }

    #[test]
    fn transform_chain_skips_distance_modifier() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_modifier(ModifierKind::Round, Some(sphere));
        let parent_map = scene.build_parent_map();
        let order = scene.visible_topo_order();
        let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let chain = get_transform_chain(sphere, &parent_map, &scene, &idx_map);
        assert!(chain.is_empty());
    }

    #[test]
    fn transform_chain_innermost_first_order() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let twist = scene.create_modifier(ModifierKind::Twist, Some(sphere));
        scene.create_transform(Some(twist));
        let parent_map = scene.build_parent_map();
        let order = scene.visible_topo_order();
        let idx_map: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let chain = get_transform_chain(sphere, &parent_map, &scene, &idx_map);
        assert_eq!(chain.len(), 2);
        // First entry is innermost (Twist), second is outermost (Transform)
        assert!(matches!(chain[0].1, ChainEntry::Modifier(ModifierKind::Twist)));
        assert!(matches!(chain[1].1, ChainEntry::Transform));
    }

    // ═══════════════════════════════════════════════════════════════
    // emit_transform_chain
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn emit_transform_chain_empty_returns_p() {
        let mut lines = Vec::new();
        let result = emit_transform_chain(&mut lines, 0, &[]);
        assert_eq!(result, "p");
        assert!(lines.is_empty());
    }

    #[test]
    fn emit_transform_chain_single_transform_generates_three_lets() {
        let mut lines = Vec::new();
        let chain = vec![(1, ChainEntry::Transform)];
        let result = emit_transform_chain(&mut lines, 0, &chain);
        assert_eq!(result, "tp0_0");
        // Should generate translate, rotate, scale steps
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("position.xyz"));
        assert!(lines[1].contains("rotate_euler"));
        assert!(lines[2].contains("scale.xyz"));
    }

    #[test]
    fn emit_transform_chain_twist_generates_twist_point() {
        let mut lines = Vec::new();
        let chain = vec![(2, ChainEntry::Modifier(ModifierKind::Twist))];
        let result = emit_transform_chain(&mut lines, 0, &chain);
        assert_eq!(result, "tp0_0");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("twist_point(p"));
    }

    #[test]
    fn emit_transform_chain_two_entries_chains_vars() {
        let mut lines = Vec::new();
        let chain = vec![
            (2, ChainEntry::Modifier(ModifierKind::Mirror)),
            (3, ChainEntry::Transform),
        ];
        let result = emit_transform_chain(&mut lines, 5, &chain);
        assert_eq!(result, "tp5_1");
        // Outermost (Transform at idx 3) is processed first (chain reversed)
        assert!(lines[0].contains("nodes[3]"));
        // Inner (Mirror at idx 2) is processed second, using the output of the first step
        assert!(lines[3].contains("mirror_point(tp5_0"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Sculpt nodes
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn standalone_sculpt_uses_sdf_voxel_grid() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                color: Vec3::new(0.5, 0.5, 0.5),
                roughness: 0.5,
                metallic: 0.0,
                emissive: Vec3::ZERO,
                emissive_intensity: 0.0,
                fresnel: 0.04,
                layer_intensity: 1.0,
                voxel_grid: grid,
                desired_resolution: 8,
            },
        );
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let sculpt_idx = order.iter().position(|&id| id == sculpt_id).unwrap();
        // Standalone sculpt without tex_map uses sdf_voxel_grid
        assert!(wgsl.contains(&format!("sdf_voxel_grid(lp{sculpt_idx}")));
    }

    #[test]
    fn differential_sculpt_uses_disp_voxel_grid() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.create_sculpt(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );
        let wgsl = generate_scene_sdf(&scene, None);
        let order = scene.visible_topo_order();
        let sculpt_idx = order.iter().position(|&id| id == sculpt_id).unwrap();
        // Differential sculpt without tex_map uses disp_voxel_grid
        assert!(wgsl.contains(&format!("disp_voxel_grid(lp{sculpt_idx}")));
        // Should include layer_intensity via position.w
        assert!(wgsl.contains("position.w"));
    }

    #[test]
    fn sculpt_with_tex_map_uses_texture_path() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                color: Vec3::new(0.5, 0.5, 0.5),
                roughness: 0.5,
                metallic: 0.0,
                emissive: Vec3::ZERO,
                emissive_intensity: 0.0,
                fresnel: 0.04,
                layer_intensity: 1.0,
                voxel_grid: grid,
                desired_resolution: 8,
            },
        );
        let mut tex_map = HashMap::new();
        tex_map.insert(sculpt_id, 0usize);
        let wgsl = generate_scene_sdf(&scene, Some(&tex_map));
        assert!(wgsl.contains("sdf_voxel_tex_0("));
    }

    #[test]
    fn differential_sculpt_with_tex_map_uses_disp_texture_path() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.create_sculpt(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );
        let mut tex_map = HashMap::new();
        tex_map.insert(sculpt_id, 2usize);
        let wgsl = generate_scene_sdf(&scene, Some(&tex_map));
        assert!(wgsl.contains("disp_voxel_tex_2("));
    }

    // ═══════════════════════════════════════════════════════════════
    // generate_voxel_texture_decls
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn no_sculpt_nodes_empty_texture_decls() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let (decls, tex_map) = generate_voxel_texture_decls(&scene);
        assert!(decls.is_empty());
        assert!(tex_map.is_empty());
    }

    #[test]
    fn standalone_sculpt_generates_sdf_sampling_function() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                color: Vec3::new(0.5, 0.5, 0.5),
                roughness: 0.5,
                metallic: 0.0,
                emissive: Vec3::ZERO,
                emissive_intensity: 0.0,
                fresnel: 0.04,
                layer_intensity: 1.0,
                voxel_grid: grid,
                desired_resolution: 8,
            },
        );
        let (decls, tex_map) = generate_voxel_texture_decls(&scene);
        assert!(decls.contains("voxel_sampler"));
        assert!(decls.contains("voxel_tex_0"));
        assert!(decls.contains("fn sdf_voxel_tex_0("));
        assert!(decls.contains("box_dist"));
        assert_eq!(tex_map.len(), 1);
    }

    #[test]
    fn differential_sculpt_generates_disp_sampling_function() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.create_sculpt(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );
        let (decls, tex_map) = generate_voxel_texture_decls(&scene);
        assert!(decls.contains("fn disp_voxel_tex_0("));
        assert!(decls.contains("return 0.0"));
        assert!(tex_map.contains_key(&sculpt_id));
    }

    #[test]
    fn texture_decl_bindings_start_at_group_2() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                color: Vec3::new(0.5, 0.5, 0.5),
                roughness: 0.5,
                metallic: 0.0,
                emissive: Vec3::ZERO,
                emissive_intensity: 0.0,
                fresnel: 0.04,
                layer_intensity: 1.0,
                voxel_grid: grid,
                desired_resolution: 8,
            },
        );
        let (decls, _) = generate_voxel_texture_decls(&scene);
        assert!(decls.contains("@group(2) @binding(0) var voxel_sampler"));
        assert!(decls.contains("@group(2) @binding(1) var voxel_tex_0"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Two-phase codegen with bounding skip
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn expensive_subtree_wrapped_in_bounding_check() {
        let mut scene = empty_scene();
        // Cheap subtree (no sculpt)
        scene.create_primitive(SdfPrimitive::Sphere);
        // Expensive subtree (has sculpt)
        let box_prim = scene.create_primitive(SdfPrimitive::Box);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.create_sculpt(
            box_prim,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );
        let wgsl = generate_scene_sdf(&scene, None);
        // Two-phase: bounding sphere check wrapping the expensive subtree
        assert!(wgsl.contains("_bd"));
        assert!(wgsl.contains("if _bd < result.x"));
    }

    #[test]
    fn single_expensive_no_cheap_uses_flat_codegen() {
        let mut scene = empty_scene();
        // Only expensive subtree — no benefit from bounding skip, uses flat codegen
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.create_sculpt(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );
        let wgsl = generate_scene_sdf(&scene, None);
        // Should NOT have bounding skip — flat codegen
        assert!(!wgsl.contains("_bd"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Complex scenes
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn operation_with_transform_children() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let xform_sphere = scene.create_transform(Some(sphere));
        let box_prim = scene.create_primitive(SdfPrimitive::Box);
        let xform_box = scene.create_transform(Some(box_prim));
        scene.create_operation(CsgOp::Union, Some(xform_sphere), Some(xform_box));
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("sdf_sphere"));
        assert!(wgsl.contains("sdf_box"));
        assert!(wgsl.contains("op_union(n"));
    }

    #[test]
    fn modifier_chain_on_primitive() {
        // Round(Mirror(Sphere))
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let mirror = scene.create_modifier(ModifierKind::Mirror, Some(sphere));
        scene.create_modifier(ModifierKind::Round, Some(mirror));
        let wgsl = generate_scene_sdf(&scene, None);
        // Mirror is a point modifier → mirror_point in sphere's chain
        assert!(wgsl.contains("mirror_point("));
        // Round is a distance modifier → subtraction
        assert!(wgsl.contains("abs(") || wgsl.contains(".x -"));
    }

    // ═══════════════════════════════════════════════════════════════
    // structure_key stability via codegen
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn same_topology_generates_same_codegen() {
        let mut scene1 = empty_scene();
        scene1.create_primitive(SdfPrimitive::Sphere);
        let wgsl1 = generate_scene_sdf(&scene1, None);

        let mut scene2 = empty_scene();
        scene2.create_primitive(SdfPrimitive::Sphere);
        let wgsl2 = generate_scene_sdf(&scene2, None);

        assert_eq!(wgsl1, wgsl2);
    }

    #[test]
    fn different_topology_generates_different_codegen() {
        let mut scene1 = empty_scene();
        scene1.create_primitive(SdfPrimitive::Sphere);
        let wgsl1 = generate_scene_sdf(&scene1, None);

        let mut scene2 = empty_scene();
        scene2.create_primitive(SdfPrimitive::Box);
        let wgsl2 = generate_scene_sdf(&scene2, None);

        assert_ne!(wgsl1, wgsl2);
    }

    // ═══════════════════════════════════════════════════════════════
    // format_f32 and format_vec3 (from shader_templates)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn format_f32_integer_adds_decimal() {
        use super::super::shader_templates::format_f32;
        assert_eq!(format_f32(1.0), "1.0");
        assert_eq!(format_f32(0.0), "0.0");
        assert_eq!(format_f32(-5.0), "-5.0");
    }

    #[test]
    fn format_f32_decimal_unchanged() {
        use super::super::shader_templates::format_f32;
        assert_eq!(format_f32(1.5), "1.5");
        assert_eq!(format_f32(0.001), "0.001");
    }

    #[test]
    fn format_vec3_formats_all_components() {
        use super::super::shader_templates::format_vec3;
        let result = format_vec3([1.0, 2.5, -3.0]);
        assert_eq!(result, "1.0, 2.5, -3.0");
    }

    // ═══════════════════════════════════════════════════════════════
    // Primitive node uses local point variable
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn primitive_generates_local_point_lp() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("let lp0"));
    }

    #[test]
    fn primitive_material_id_is_node_index() {
        let mut scene = empty_scene();
        scene.create_primitive(SdfPrimitive::Sphere);
        let wgsl = generate_scene_sdf(&scene, None);
        // Material ID is f32(index)
        assert!(wgsl.contains("f32(0)"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Sculpt node with rotate_euler
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn sculpt_generates_rotate_euler_for_local_point() {
        let mut scene = empty_scene();
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.add_node(
            "Sculpt".to_string(),
            NodeData::Sculpt {
                input: None,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                color: Vec3::new(0.5, 0.5, 0.5),
                roughness: 0.5,
                metallic: 0.0,
                emissive: Vec3::ZERO,
                emissive_intensity: 0.0,
                fresnel: 0.04,
                layer_intensity: 1.0,
                voxel_grid: grid,
                desired_resolution: 8,
            },
        );
        let wgsl = generate_scene_sdf(&scene, None);
        assert!(wgsl.contains("rotate_euler("));
    }
}
