use crate::bridge_state::{app_bridge, snapshot_json};
use sdf_modeler::{CsgOp, LightType, ModifierKind};

#[flutter_rust_bridge::frb(sync)]
pub fn ping() -> String {
    "pong".to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn bridge_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn scene_snapshot_json() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn new_scene() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.new_scene();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_scene() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_scene();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_recent_scene(path: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_recent_scene(&path);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn save_scene() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.save_scene();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn save_scene_as() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.save_scene_as();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn recover_autosave() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.recover_autosave();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn discard_recovery() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.discard_recovery();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_import_dialog() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_import_dialog();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_import_dialog() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_import_dialog();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_import_use_auto(use_auto: bool) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_import_use_auto(use_auto);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_import_resolution(resolution: u32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_import_resolution(resolution);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_import() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_import();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_import() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_import();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_sculpt_convert_dialog_for_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_sculpt_convert_dialog_for_selected();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_sculpt_convert_dialog() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_sculpt_convert_dialog();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_convert_mode(mode_id: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_convert_mode(&mode_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_convert_resolution(resolution: u32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_convert_resolution(resolution);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_sculpt_convert() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_sculpt_convert();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_export_resolution(resolution: u32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_export_resolution(resolution);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_adaptive_export(enabled: bool) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_adaptive_export(enabled);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_export() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_export();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_export() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_export();
    snapshot_json(&mut bridge)
}

pub fn render_preview_frame(width: u32, height: u32, time_seconds: f32) -> Vec<u8> {
    app_bridge()
        .lock()
        .expect("app bridge mutex")
        .render_viewport_frame(width, height, time_seconds)
        .pixels
}

#[flutter_rust_bridge::frb(sync)]
pub fn orbit_camera(delta_x: f32, delta_y: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.orbit_camera(delta_x, delta_y);
}

#[flutter_rust_bridge::frb(sync)]
pub fn pan_camera(delta_x: f32, delta_y: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.pan_camera(delta_x, delta_y);
}

#[flutter_rust_bridge::frb(sync)]
pub fn zoom_camera(delta: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.zoom_camera(delta);
}

#[flutter_rust_bridge::frb(sync)]
pub fn select_node(node_id: Option<u64>) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node(node_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn select_node_at_viewport(
    mouse_x: f32,
    mouse_y: f32,
    width: u32,
    height: u32,
    time_seconds: f32,
) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node_at_viewport(mouse_x, mouse_y, width, height, time_seconds);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_node_visibility(node_id: u64) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_visibility(node_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_node_lock(node_id: u64) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_lock(node_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn delete_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.delete_selected();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn duplicate_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.duplicate_selected();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn rename_node(node_id: u64, name: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.rename_node(node_id, &name);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_operation(operation_id: String) -> String {
    let operation = parse_operation_id(&operation_id)
        .unwrap_or_else(|| panic!("unknown operation id: {operation_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_operation(operation);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_transform() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_transform();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_modifier(modifier_id: String) -> String {
    let modifier = parse_modifier_id(&modifier_id)
        .unwrap_or_else(|| panic!("unknown modifier id: {modifier_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_modifier(modifier);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_light(light_id: String) -> String {
    let light_type =
        parse_light_id(&light_id).unwrap_or_else(|| panic!("unknown light id: {light_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_light(light_type);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_sculpt() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_sculpt();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn resume_sculpting_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.resume_sculpting_selected();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn stop_sculpting() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.stop_sculpting();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_mode(mode_id: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_mode(&mode_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_radius(radius: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_radius(radius);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_strength(strength: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_strength(strength);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_symmetry_axis(axis_id: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_symmetry_axis(&axis_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_sculpt_resolution(resolution: u32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_sculpt_resolution(resolution);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_primitive_parameter(parameter_key: String, value: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_primitive_parameter(&parameter_key, value);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_material_float(field_id: String, value: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_material_float(&field_id, value);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_material_color(
    field_id: String,
    red: f32,
    green: f32,
    blue: f32,
) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_material_color(&field_id, red, green, blue);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_position(x: f32, y: f32, z: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_position(x, y, z);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_rotation_degrees(
    x_degrees: f32,
    y_degrees: f32,
    z_degrees: f32,
) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_rotation_degrees(x_degrees, y_degrees, z_degrees);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_scale(x: f32, y: f32, z: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_scale(x, y, z);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_manipulator_mode(mode_id: String) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_manipulator_mode(&mode_id);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_manipulator_space() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_manipulator_space();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_manipulator_pivot_offset(x: f32, y: f32, z: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_manipulator_pivot_offset(x, y, z);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_manipulator_pivot() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_manipulator_pivot();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_translation(delta_x: f32, delta_y: f32, delta_z: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_translation(delta_x, delta_y, delta_z);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_rotation_degrees(
    delta_x_degrees: f32,
    delta_y_degrees: f32,
    delta_z_degrees: f32,
) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_rotation_degrees(delta_x_degrees, delta_y_degrees, delta_z_degrees);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_scale(delta_x: f32, delta_y: f32, delta_z: f32) -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_scale(delta_x, delta_y, delta_z);
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn undo() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.undo();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn redo() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.redo();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn focus_selected() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.focus_selected();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn frame_all() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.frame_all();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_front() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_front();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_top() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_top();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_right() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_right();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_back() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_back();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_left() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_left();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_bottom() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_bottom();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_orthographic() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_orthographic();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_sphere() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_sphere();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_box() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_box();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_cylinder() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_cylinder();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_torus() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_torus();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_scene() -> String {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_scene();
    snapshot_json(&mut bridge)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
    let _ = app_bridge();
}

fn parse_operation_id(operation_id: &str) -> Option<CsgOp> {
    Some(match operation_id {
        "union" => CsgOp::Union,
        "smooth_union" => CsgOp::SmoothUnion,
        "subtract" => CsgOp::Subtract,
        "intersect" => CsgOp::Intersect,
        "smooth_subtract" => CsgOp::SmoothSubtract,
        "smooth_intersect" => CsgOp::SmoothIntersect,
        "chamfer_union" => CsgOp::ChamferUnion,
        "chamfer_subtract" => CsgOp::ChamferSubtract,
        "chamfer_intersect" => CsgOp::ChamferIntersect,
        "stairs_union" => CsgOp::StairsUnion,
        "stairs_subtract" => CsgOp::StairsSubtract,
        "columns_union" => CsgOp::ColumnsUnion,
        "columns_subtract" => CsgOp::ColumnsSubtract,
        _ => return None,
    })
}

fn parse_modifier_id(modifier_id: &str) -> Option<ModifierKind> {
    Some(match modifier_id {
        "twist" => ModifierKind::Twist,
        "bend" => ModifierKind::Bend,
        "taper" => ModifierKind::Taper,
        "round" => ModifierKind::Round,
        "onion" => ModifierKind::Onion,
        "elongate" => ModifierKind::Elongate,
        "mirror" => ModifierKind::Mirror,
        "repeat" => ModifierKind::Repeat,
        "finite_repeat" => ModifierKind::FiniteRepeat,
        "radial_repeat" => ModifierKind::RadialRepeat,
        "offset" => ModifierKind::Offset,
        "noise" => ModifierKind::Noise,
        _ => return None,
    })
}

fn parse_light_id(light_id: &str) -> Option<LightType> {
    Some(match light_id {
        "point" => LightType::Point,
        "spot" => LightType::Spot,
        "directional" => LightType::Directional,
        "ambient" => LightType::Ambient,
        _ => return None,
    })
}
