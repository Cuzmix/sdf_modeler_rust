use std::collections::HashSet;
use std::fmt::Write;

use crate::core::AppCore;
use crate::gpu::buffers;
use crate::graph::scene::{NodeData, NodeId, Scene};
use crate::settings::Settings;

#[derive(Debug, Clone, Default)]
pub struct PanelReadModel {
    pub scene_tree_text: String,
    pub history_text: String,
    pub lights_text: String,
    pub render_settings_text: String,
    pub scene_stats_text: String,
    pub properties_text: String,
}

pub fn build_panel_read_model(core: &AppCore, settings: &Settings, scene_tree_filter: &str) -> PanelReadModel {
    PanelReadModel {
        scene_tree_text: build_scene_tree_text(&core.scene, core.selection.primary, scene_tree_filter),
        history_text: build_history_text(core),
        lights_text: build_lights_text(core),
        render_settings_text: build_render_settings_text(settings),
        scene_stats_text: build_scene_stats_text(core),
        properties_text: build_properties_text(core),
    }
}

fn build_scene_tree_text(scene: &Scene, selected_primary: Option<NodeId>, filter_text: &str) -> String {
    let top_level = scene.top_level_nodes();
    if top_level.is_empty() {
        return "Empty scene".to_string();
    }

    let mut output = String::new();
    let normalized_filter = filter_text.trim().to_ascii_lowercase();
    if !normalized_filter.is_empty() {
        let mut matching_ids: Vec<NodeId> = scene
            .nodes
            .iter()
            .filter_map(|(node_id, node)| {
                if node.name.to_ascii_lowercase().contains(&normalized_filter) {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();
        matching_ids.sort_unstable();

        if matching_ids.is_empty() {
            let _ = writeln!(output, "No matching nodes for '{}'", filter_text.trim());
            return output;
        }

        for node_id in matching_ids {
            if let Some(node) = scene.nodes.get(&node_id) {
                let selected_marker = if selected_primary == Some(node_id) { "*" } else { " " };
                let hidden_marker = if scene.is_hidden(node_id) { "H" } else { " " };
                let _ = writeln!(
                    output,
                    "{}{} [{}] {} (#{} )",
                    selected_marker,
                    hidden_marker,
                    node_kind_label(&node.data),
                    node.name,
                    node_id
                );
            }
        }
        return output;
    }

    let mut visited = HashSet::new();
    for node_id in top_level {
        append_scene_tree_node(scene, node_id, 0, selected_primary, &mut visited, &mut output);
    }
    output
}

fn append_scene_tree_node(
    scene: &Scene,
    node_id: NodeId,
    depth: usize,
    selected_primary: Option<NodeId>,
    visited: &mut HashSet<NodeId>,
    output: &mut String,
) {
    if !visited.insert(node_id) {
        let _ = writeln!(output, "{}[cycle] #{}", "  ".repeat(depth), node_id);
        return;
    }

    let Some(node) = scene.nodes.get(&node_id) else {
        let _ = writeln!(output, "{}[missing] #{}", "  ".repeat(depth), node_id);
        return;
    };

    let selected_marker = if selected_primary == Some(node_id) {
        "*"
    } else {
        " "
    };
    let hidden_marker = if scene.is_hidden(node_id) { "H" } else { " " };
    let _ = writeln!(
        output,
        "{}{}{} [{}] {} (#{} )",
        "  ".repeat(depth),
        selected_marker,
        hidden_marker,
        node_kind_label(&node.data),
        node.name,
        node_id
    );

    match &node.data {
        NodeData::Operation { left, right, .. } => {
            if let Some(left_id) = left {
                append_scene_tree_node(scene, *left_id, depth + 1, selected_primary, visited, output);
            }
            if let Some(right_id) = right {
                append_scene_tree_node(scene, *right_id, depth + 1, selected_primary, visited, output);
            }
        }
        NodeData::Sculpt { input, .. }
        | NodeData::Transform { input, .. }
        | NodeData::Modifier { input, .. } => {
            if let Some(input_id) = input {
                append_scene_tree_node(scene, *input_id, depth + 1, selected_primary, visited, output);
            }
        }
        NodeData::Primitive { .. } | NodeData::Light { .. } => {}
    }
}

fn build_history_text(core: &AppCore) -> String {
    let mut output = String::new();
    let undo_labels = core.history.undo_labels();
    let redo_labels = core.history.redo_labels();

    let _ = writeln!(
        output,
        "Undo: {} | Redo: {}",
        undo_labels.len(),
        redo_labels.len()
    );

    for label in redo_labels.iter().take(6) {
        let _ = writeln!(output, "redo: {}", label);
    }

    let _ = writeln!(output, "current: scene");

    for label in undo_labels.iter().rev().take(8) {
        let _ = writeln!(output, "undo: {}", label);
    }

    if undo_labels.is_empty() && redo_labels.is_empty() {
        let _ = writeln!(output, "No history yet");
    }

    output
}

fn build_lights_text(core: &AppCore) -> String {
    let (active_light_ids, total_light_count) =
        buffers::identify_active_lights(&core.scene, core.camera.eye());

    let mut output = String::new();
    let _ = writeln!(
        output,
        "Active lights: {}/{}",
        active_light_ids.len(),
        total_light_count
    );
    if let Some(solo_id) = core.soloed_light {
        let solo_name = core
            .scene
            .nodes
            .get(&solo_id)
            .map(|node| node.name.clone())
            .unwrap_or_else(|| format!("#{}", solo_id));
        let _ = writeln!(output, "Solo: {}", solo_name);
    } else {
        let _ = writeln!(output, "Solo: none");
    }

    let parent_map = core.scene.build_parent_map();
    let mut lights: Vec<NodeId> = core
        .scene
        .nodes
        .iter()
        .filter_map(|(node_id, node)| {
            if matches!(node.data, NodeData::Light { .. }) {
                Some(*node_id)
            } else {
                None
            }
        })
        .collect();
    lights.sort_unstable();

    if lights.is_empty() {
        let _ = writeln!(output, "No lights");
        return output;
    }

    for light_id in lights {
        let Some(node) = core.scene.nodes.get(&light_id) else {
            continue;
        };
        let NodeData::Light {
            light_type,
            intensity,
            range,
            ..
        } = &node.data
        else {
            continue;
        };

        let active_marker = if active_light_ids.contains(&light_id) {
            "on"
        } else {
            "off"
        };

        let transform_name = parent_map
            .get(&light_id)
            .and_then(|transform_id| core.scene.nodes.get(transform_id))
            .map(|transform_node| transform_node.name.as_str())
            .unwrap_or("none");

        let _ = writeln!(
            output,
            "[{active_marker}] {} {} I={:.2} R={:.1} T={}",
            light_type.label(),
            node.name,
            intensity,
            range,
            transform_name
        );
    }

    output
}

fn build_render_settings_text(settings: &Settings) -> String {
    let render = &settings.render;
    let mut output = String::new();

    let _ = writeln!(output, "Shading: {}", render.shading_mode.label());
    let _ = writeln!(
        output,
        "Scale rest/interact: {:.2}/{:.2}",
        render.rest_render_scale,
        render.interaction_render_scale
    );
    let _ = writeln!(output, "Shadows: {}", if render.shadows_enabled { "on" } else { "off" });
    let _ = writeln!(output, "AO: {}", if render.ao_enabled { "on" } else { "off" });
    let _ = writeln!(output, "Fog: {}", if render.fog_enabled { "on" } else { "off" });
    let _ = writeln!(output, "Bloom: {}", if render.bloom_enabled { "on" } else { "off" });

    output
}

fn build_scene_stats_text(core: &AppCore) -> String {
    let counts = core.scene.node_type_counts();
    let sdf_complexity = core.scene.sdf_eval_complexity();
    let voxel_memory_bytes = core.scene.voxel_memory_bytes();

    let mut output = String::new();
    let _ = writeln!(output, "Total: {}", counts.total);
    let _ = writeln!(output, "Visible: {}", counts.visible);
    let _ = writeln!(
        output,
        "Prim/Op/Xfm/Mod/Scl/Lgt: {}/{}/{}/{}/{}/{}",
        counts.primitives,
        counts.operations,
        counts.transforms,
        counts.modifiers,
        counts.sculpts,
        counts.lights
    );
    let _ = writeln!(output, "SDF complexity: {}", sdf_complexity);
    let _ = writeln!(output, "Voxel memory: {}", format_bytes(voxel_memory_bytes));

    output
}

fn build_properties_text(core: &AppCore) -> String {
    let Some(selected_primary) = core.selection.primary else {
        return "No selection".to_string();
    };

    let Some(node) = core.scene.nodes.get(&selected_primary) else {
        return "Selection missing".to_string();
    };

    let mut output = String::new();
    let _ = writeln!(output, "{} (#{} )", node.name, node.id);
    let _ = writeln!(output, "Type: {}", node_kind_label(&node.data));

    match &node.data {
        NodeData::Primitive {
            position,
            rotation,
            scale,
            roughness,
            metallic,
            ..
        } => {
            let _ = writeln!(output, "Pos: {}", format_vec3(*position));
            let _ = writeln!(output, "Rot: {}", format_vec3(*rotation));
            let _ = writeln!(output, "Scale: {}", format_vec3(*scale));
            let _ = writeln!(output, "Rgh/Met: {:.2}/{:.2}", roughness, metallic);
        }
        NodeData::Operation {
            smooth_k,
            left,
            right,
            ..
        } => {
            let _ = writeln!(output, "Smooth K: {:.3}", smooth_k);
            let _ = writeln!(output, "Left: {}", option_node_id(*left));
            let _ = writeln!(output, "Right: {}", option_node_id(*right));
        }
        NodeData::Transform {
            translation,
            rotation,
            scale,
            input,
        } => {
            let _ = writeln!(output, "Input: {}", option_node_id(*input));
            let _ = writeln!(output, "Pos: {}", format_vec3(*translation));
            let _ = writeln!(output, "Rot: {}", format_vec3(*rotation));
            let _ = writeln!(output, "Scale: {}", format_vec3(*scale));
        }
        NodeData::Modifier {
            kind,
            input,
            value,
            extra,
        } => {
            let _ = writeln!(output, "Kind: {}", kind.base_name());
            let _ = writeln!(output, "Input: {}", option_node_id(*input));
            let _ = writeln!(output, "Value: {}", format_vec3(*value));
            let _ = writeln!(output, "Extra: {}", format_vec3(*extra));
        }
        NodeData::Sculpt {
            input,
            position,
            rotation,
            voxel_grid,
            ..
        } => {
            let _ = writeln!(output, "Input: {}", option_node_id(*input));
            let _ = writeln!(output, "Pos: {}", format_vec3(*position));
            let _ = writeln!(output, "Rot: {}", format_vec3(*rotation));
            let _ = writeln!(output, "Resolution: {}", voxel_grid.resolution);
        }
        NodeData::Light {
            light_type,
            intensity,
            range,
            spot_angle,
            ..
        } => {
            let _ = writeln!(output, "Light: {}", light_type.label());
            let _ = writeln!(output, "Intensity: {:.2}", intensity);
            let _ = writeln!(output, "Range: {:.2}", range);
            let _ = writeln!(output, "Spot angle: {:.1}", spot_angle);
        }
    }

    output
}

fn node_kind_label(data: &NodeData) -> &'static str {
    match data {
        NodeData::Primitive { .. } => "Primitive",
        NodeData::Operation { .. } => "Operation",
        NodeData::Transform { .. } => "Transform",
        NodeData::Modifier { .. } => "Modifier",
        NodeData::Sculpt { .. } => "Sculpt",
        NodeData::Light { .. } => "Light",
    }
}

fn format_vec3(value: glam::Vec3) -> String {
    format!("{:.2}, {:.2}, {:.2}", value.x, value.y, value.z)
}

fn option_node_id(value: Option<NodeId>) -> String {
    value
        .map(|node_id| node_id.to_string())
        .unwrap_or_else(|| "None".to_string())
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{AppCoreInit, CoreAsyncState, CoreSelection};
    use crate::gpu::camera::Camera;
    use crate::graph::history::History;
    use crate::graph::scene::{Scene, SdfPrimitive};
    use crate::sculpt::{ActiveTool, SculptState};

    use super::{build_panel_read_model, PanelReadModel};

    fn build_test_core_with_selection() -> crate::core::AppCore {
        let mut scene = Scene::new();
        let selected_id = scene.create_primitive(SdfPrimitive::Box);
        let mut selection = CoreSelection::default();
        selection.select_single(selected_id);

        crate::core::AppCore::from_init(AppCoreInit {
            scene,
            history: History::new(),
            camera: Camera::default(),
            selection,
            active_tool: ActiveTool::Select,
            sculpt_state: SculptState::Inactive,
            async_state: CoreAsyncState::default(),
            soloed_light: None,
            show_debug: false,
            show_settings: false,
        })
    }

    #[test]
    fn panel_read_model_contains_expected_sections() {
        let core = build_test_core_with_selection();
        let settings = crate::settings::Settings::default();

        let model: PanelReadModel = build_panel_read_model(&core, &settings, "");

        assert!(model.scene_tree_text.contains("Primitive") || model.scene_tree_text.contains("Transform"));
        assert!(model.history_text.contains("Undo:"));
        assert!(model.lights_text.contains("Active lights:"));
        assert!(model.render_settings_text.contains("Shading:"));
        assert!(model.scene_stats_text.contains("Total:"));
        assert!(model.properties_text.contains("Type:"));
    }

    #[test]
    fn panel_read_model_no_selection_properties() {
        let core = crate::core::AppCore::from_init(AppCoreInit {
            scene: Scene::new(),
            history: History::new(),
            camera: Camera::default(),
            selection: CoreSelection::default(),
            active_tool: ActiveTool::Select,
            sculpt_state: SculptState::Inactive,
            async_state: CoreAsyncState::default(),
            soloed_light: None,
            show_debug: false,
            show_settings: false,
        });
        let settings = crate::settings::Settings::default();

        let model = build_panel_read_model(&core, &settings, "");
        assert_eq!(model.properties_text, "No selection");
    }
}





