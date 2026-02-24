use std::collections::{HashMap, HashSet};

use crate::graph::scene::{NodeData, NodeId, Scene, SceneNode, TransformKind};
use crate::settings::RenderConfig;

use super::buffers::collect_sculpt_tex_info;
use super::shader_templates::{
    apply_march_placeholders, build_postlude, format_f32, format_vec3,
    COMPOSITE_COMPUTE_ENTRY, PICK_COMPUTE_POSTLUDE, SHADER_PRELUDE,
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

pub fn generate_shader(scene: &Scene, config: &RenderConfig) -> String {
    let (tex_decls, sculpt_tex_map) = generate_voxel_texture_decls(scene);
    let tex_map = if sculpt_tex_map.is_empty() { None } else { Some(&sculpt_tex_map) };
    let scene_sdf = generate_scene_sdf(scene, tex_map);
    let sel_sdf = generate_selected_sdf(scene, tex_map);
    let postlude = build_postlude(config);
    format!("{}\n{}\n{}\n{}\n{}", SHADER_PRELUDE, tex_decls, scene_sdf, sel_sdf, postlude)
}

pub fn generate_pick_shader(scene: &Scene, config: &RenderConfig) -> String {
    let scene_sdf = generate_scene_sdf(scene, None);
    let pick_postlude = apply_march_placeholders(PICK_COMPUTE_POSTLUDE, config);
    format!("{}\n{}\n{}", SHADER_PRELUDE, scene_sdf, pick_postlude)
}

/// Generate the composite compute shader that pre-evaluates scene_sdf at every voxel
/// in a 3D grid and writes SDF + material ID to storage textures.
pub fn generate_composite_shader(scene: &Scene, _config: &RenderConfig) -> String {
    let (tex_decls, sculpt_tex_map) = generate_voxel_texture_decls(scene);
    let tex_map = if sculpt_tex_map.is_empty() { None } else { Some(&sculpt_tex_map) };
    let scene_sdf = generate_scene_sdf(scene, tex_map);
    format!("{}\n{}\n{}\n{}", SHADER_PRELUDE, tex_decls, scene_sdf, COMPOSITE_COMPUTE_ENTRY)
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

fn scene_sdf(p: vec3f) -> vec2f {{
    let size = COMP_BMAX - COMP_BMIN;
    let norm = (p - COMP_BMIN) / size;

    // Outside bounds: return distance to AABB
    if any(norm < vec3f(-0.01)) || any(norm > vec3f(1.01)) {{
        return vec2f(length(max(p - COMP_BMAX, COMP_BMIN - p)), -1.0);
    }}

    let uv = clamp(norm, vec3f(0.0), vec3f(1.0));
    let d = textureSampleLevel(comp_sdf_tex, comp_sampler, uv, 0.0).x;
    let dims = textureDimensions(comp_mat_tex);
    let fc = clamp(uv * vec3f(dims), vec3f(0.0), vec3f(dims - vec3u(1u)));
    let ic = vec3u(fc);
    let mat_raw = textureLoad(comp_mat_tex, ic, 0).x;
    let mat_id = f32(mat_raw) - 1.0;
    return vec2f(d, mat_id);
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

    format!("{}\n{}\n{}", SHADER_PRELUDE, comp_scene_sdf, postlude)
}

// ---------------------------------------------------------------------------
// Transform chain helpers
// ---------------------------------------------------------------------------

/// Walk up from `node_id` through ancestors, collecting all Transform ancestors.
/// Returns chain from innermost (closest to leaf) to outermost.
fn get_transform_chain(
    node_id: NodeId,
    parent_map: &HashMap<NodeId, NodeId>,
    scene: &Scene,
    idx_map: &HashMap<NodeId, usize>,
) -> Vec<(usize, TransformKind)> {
    let mut chain = Vec::new();
    let mut current = node_id;
    while let Some(&parent_id) = parent_map.get(&current) {
        if let Some(node) = scene.nodes.get(&parent_id) {
            if let NodeData::Transform { kind, .. } = &node.data {
                if let Some(&idx) = idx_map.get(&parent_id) {
                    chain.push((idx, kind.clone()));
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
    chain: &[(usize, TransformKind)],
) -> String {
    if chain.is_empty() {
        return "p".to_string();
    }
    let mut current_var = "p".to_string();
    for (step, (transform_idx, kind)) in chain.iter().rev().enumerate() {
        let new_var = format!("tp{}_{}", node_idx, step);
        match kind {
            TransformKind::Translate => {
                lines.push(format!(
                    "    let {new_var} = {current_var} - nodes[{transform_idx}].position.xyz;"
                ));
            }
            TransformKind::Rotate => {
                lines.push(format!(
                    "    let {new_var} = rotate_euler({current_var}, nodes[{transform_idx}].rotation.xyz);"
                ));
            }
            TransformKind::Scale => {
                lines.push(format!(
                    "    let {new_var} = {current_var} / nodes[{transform_idx}].scale.xyz;"
                ));
            }
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
                "    let n{i} = vec2f({sdf_fn}(lp{i}, nodes[{i}].scale.xyz), f32({i}));"
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
                    lines.push(format!("    let n{i} = vec2f(1e10, -1.0);"));
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

            if child_idx.is_some() {
                // DIFFERENTIAL: analytical child SDF + displacement from grid
                let ci = child_idx.unwrap();
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
                    "    let n{i} = vec2f(n{ci}.x + {disp_call}, f32({i}));"
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
                    "    let n{i} = vec2f({sdf_call}, f32({i}));"
                ));
            }
        }
        NodeData::Transform { kind, input, .. } => {
            let child_idx = input.and_then(|id| idx_map.get(&id).copied());
            if let Some(ci) = child_idx {
                match kind {
                    TransformKind::Scale => {
                        lines.push(format!(
                            "    let n{i} = vec2f(n{ci}.x * min(nodes[{i}].scale.x, min(nodes[{i}].scale.y, nodes[{i}].scale.z)), n{ci}.y);"
                        ));
                    }
                    _ => {
                        lines.push(format!("    let n{i} = n{ci};"));
                    }
                }
            } else {
                lines.push(format!("    let n{i} = vec2f(1e10, -1.0);"));
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
    let order = scene.topo_order();
    if order.is_empty() {
        return "fn scene_sdf(p: vec3f) -> vec2f {\n    return vec2f(1e10, -1.0);\n}"
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
    lines.push("fn scene_sdf(p: vec3f) -> vec2f {".to_string());

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
        lines.push("    var result = vec2f(1e10, -1.0);".to_string());
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
    lines.push("fn scene_sdf(p: vec3f) -> vec2f {".to_string());

    for (i, &node_id) in order.iter().enumerate() {
        let Some(node) = scene.nodes.get(&node_id) else { continue; };
        emit_node_wgsl(&mut lines, i, node_id, node, parent_map, scene, idx_map, sculpt_tex_map);
    }

    let top_indices: Vec<usize> = tops
        .iter()
        .filter_map(|id| idx_map.get(id).copied())
        .collect();
    match top_indices.len() {
        0 => lines.push("    return vec2f(1e10, -1.0);".to_string()),
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

// ---------------------------------------------------------------------------
// Selected SDF generation
// ---------------------------------------------------------------------------

/// Generate `selected_sdf(p) -> f32`: evaluates all nodes, returns the selected
/// node's distance based on `camera.selected_idx`. Used for screen-space outline.
fn generate_selected_sdf(
    scene: &Scene,
    sculpt_tex_map: Option<&HashMap<NodeId, usize>>,
) -> String {
    let order = scene.topo_order();
    if order.is_empty() {
        return "fn selected_sdf(p: vec3f) -> f32 {\n    return 1e10;\n}".to_string();
    }

    let idx_map: HashMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let parent_map = scene.build_parent_map();

    let mut lines = Vec::new();
    lines.push("fn selected_sdf(p: vec3f) -> f32 {".to_string());
    lines.push("    if camera.selected_idx < 0.0 { return 1e10; }".to_string());
    lines.push("    let _sel_idx = i32(camera.selected_idx + 0.5);".to_string());

    for (i, &node_id) in order.iter().enumerate() {
        let Some(node) = scene.nodes.get(&node_id) else { continue; };
        emit_node_wgsl(
            &mut lines, i, node_id, node, &parent_map, scene, &idx_map, sculpt_tex_map,
        );
    }

    for i in 0..order.len() {
        lines.push(format!("    if _sel_idx == {i} {{ return n{i}.x; }}"));
    }
    lines.push("    return 1e10;".to_string());
    lines.push("}".to_string());

    lines.join("\n")
}

