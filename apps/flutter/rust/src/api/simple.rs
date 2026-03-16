use crate::bridge_state::app_bridge;
use crate::api::mirrors::{AppSceneSnapshot, AppVec3, AppWorkflowStatusSnapshot};
use sdf_modeler::app_bridge::{AppBridge as HostAppBridge, AppVec3 as HostAppVec3};
use sdf_modeler::{CsgOp, LightType, ModifierKind};

fn current_scene_snapshot(bridge: &mut HostAppBridge) -> AppSceneSnapshot {
    bridge.scene_snapshot().into()
}

fn current_workflow_status(bridge: &mut HostAppBridge) -> AppWorkflowStatusSnapshot {
    bridge.workflow_status_snapshot().into()
}

#[flutter_rust_bridge::frb(sync)]
pub fn ping() -> String {
    "pong".to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn bridge_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[flutter_rust_bridge::frb(sync)]
pub fn scene_snapshot() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn workflow_status() -> AppWorkflowStatusSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    current_workflow_status(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn new_scene() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.new_scene();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_scene() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_scene();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_recent_scene(path: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_recent_scene(&path);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn save_scene() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.save_scene();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn save_scene_as() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.save_scene_as();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn recover_autosave() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.recover_autosave();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn discard_recovery() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.discard_recovery();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_import_dialog() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_import_dialog();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_import_dialog() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_import_dialog();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_import_use_auto(use_auto: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_import_use_auto(use_auto);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_import_resolution(resolution: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_import_resolution(resolution);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_import() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_import();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_import() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_import();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn open_sculpt_convert_dialog_for_selected() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.open_sculpt_convert_dialog_for_selected();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_sculpt_convert_dialog() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_sculpt_convert_dialog();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_convert_mode(mode_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_convert_mode(&mode_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_convert_resolution(resolution: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_convert_resolution(resolution);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_sculpt_convert() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_sculpt_convert();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_export_resolution(resolution: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_export_resolution(resolution);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_adaptive_export(enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_adaptive_export(enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn start_export() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.start_export();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_export() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.cancel_export();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn apply_render_preset(preset_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.apply_render_preset(&preset_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_render_shading_mode(mode_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_render_shading_mode(&mode_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_render_toggle(field_id: String, enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_render_toggle(&field_id, enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_render_integer(field_id: String, value: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_render_integer(&field_id, value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_render_scalar(field_id: String, value: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_render_scalar(&field_id, value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_settings() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_settings();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn export_settings() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.export_settings();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn import_settings() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.import_settings();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_settings_toggle(field_id: String, enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_settings_toggle(&field_id, enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_settings_integer(field_id: String, value: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_settings_integer(&field_id, value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn save_camera_bookmark(slot_index: u8) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.save_camera_bookmark(slot_index);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn restore_camera_bookmark(slot_index: u8) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.restore_camera_bookmark(slot_index);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn clear_camera_bookmark(slot_index: u8) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.clear_camera_bookmark(slot_index);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_keymap() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_keymap();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn export_keymap() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.export_keymap();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn import_keymap() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.import_keymap();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn clear_keybinding(action_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.clear_keybinding(&action_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_keybinding(
    action_id: String,
    key_id: String,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_keybinding(&action_id, &key_id, ctrl, shift, alt);
    current_scene_snapshot(&mut bridge)
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
pub fn begin_interactive_edit() {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.begin_interactive_edit();
}

#[flutter_rust_bridge::frb(sync)]
pub fn select_node(node_id: Option<u64>) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node(node_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn select_node_at_viewport(
    mouse_x: f32,
    mouse_y: f32,
    width: u32,
    height: u32,
    time_seconds: f32,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.select_node_at_viewport(mouse_x, mouse_y, width, height, time_seconds);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_node_visibility(node_id: u64) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_visibility(node_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_node_lock(node_id: u64) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_node_lock(node_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn delete_selected() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.delete_selected();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn duplicate_selected() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.duplicate_selected();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn rename_node(node_id: u64, name: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.rename_node(node_id, &name);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_operation(operation_id: String) -> AppSceneSnapshot {
    let operation = parse_operation_id(&operation_id)
        .unwrap_or_else(|| panic!("unknown operation id: {operation_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_operation(operation);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_transform() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_transform();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_modifier(modifier_id: String) -> AppSceneSnapshot {
    let modifier = parse_modifier_id(&modifier_id)
        .unwrap_or_else(|| panic!("unknown modifier id: {modifier_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_modifier(modifier);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_light(light_id: String) -> AppSceneSnapshot {
    let light_type =
        parse_light_id(&light_id).unwrap_or_else(|| panic!("unknown light id: {light_id}"));
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_light(light_type);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn create_sculpt() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.create_sculpt();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn resume_sculpting_selected() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.resume_sculpting_selected();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn stop_sculpting() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.stop_sculpting();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_mode(mode_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_mode(&mode_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_radius(radius: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_radius(radius);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_brush_strength(strength: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_brush_strength(strength);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_sculpt_symmetry_axis(axis_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_sculpt_symmetry_axis(&axis_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_sculpt_resolution(resolution: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_sculpt_resolution(resolution);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_primitive_parameter(parameter_key: String, value: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_primitive_parameter(&parameter_key, value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_primitive_parameter(parameter_key: String, value: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_primitive_parameter(&parameter_key, value);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_material_float(field_id: String, value: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_material_float(&field_id, value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_material_float(field_id: String, value: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_material_float(&field_id, value);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_material_color(
    field_id: String,
    red: f32,
    green: f32,
    blue: f32,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_material_color(&field_id, red, green, blue);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_material_color(field_id: String, red: f32, green: f32, blue: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_material_color(&field_id, red, green, blue);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_position(x: f32, y: f32, z: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_position(x, y, z);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_transform_position(x: f32, y: f32, z: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_transform_position(x, y, z);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_rotation_degrees(
    x_degrees: f32,
    y_degrees: f32,
    z_degrees: f32,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_rotation_degrees(x_degrees, y_degrees, z_degrees);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_transform_rotation_degrees(
    x_degrees: f32,
    y_degrees: f32,
    z_degrees: f32,
) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_transform_rotation_degrees(x_degrees, y_degrees, z_degrees);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform_scale(x: f32, y: f32, z: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform_scale(x, y, z);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_transform_scale(x: f32, y: f32, z: f32) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_transform_scale(x, y, z);
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_transform(
    position: AppVec3,
    rotation_degrees: AppVec3,
    scale: Option<AppVec3>,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_transform(
        HostAppVec3::new(position.x, position.y, position.z),
        HostAppVec3::new(
            rotation_degrees.x,
            rotation_degrees.y,
            rotation_degrees.z,
        ),
        scale.map(|value| HostAppVec3::new(value.x, value.y, value.z)),
    );
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn preview_selected_transform(
    position: AppVec3,
    rotation_degrees: AppVec3,
    scale: Option<AppVec3>,
) {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.preview_selected_transform(
        HostAppVec3::new(position.x, position.y, position.z),
        HostAppVec3::new(
            rotation_degrees.x,
            rotation_degrees.y,
            rotation_degrees.z,
        ),
        scale.map(|value| HostAppVec3::new(value.x, value.y, value.z)),
    );
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_type(light_type_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_type(&light_type_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_color(red: f32, green: f32, blue: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_color(red, green, blue);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_intensity(intensity: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_intensity(intensity);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_range(range: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_range(range);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_spot_angle(angle_degrees: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_spot_angle(angle_degrees);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_cast_shadows(enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_cast_shadows(enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_shadow_softness(softness: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_shadow_softness(softness);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_shadow_color(red: f32, green: f32, blue: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_shadow_color(red, green, blue);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_volumetric(enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_volumetric(enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_volumetric_density(density: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_volumetric_density(density);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_cookie(cookie_node_id: u64) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_cookie(cookie_node_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn clear_selected_light_cookie() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.clear_selected_light_cookie();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_proximity_mode(mode_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_proximity_mode(&mode_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_proximity_range(range: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_proximity_range(range);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_array_pattern(pattern_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_array_pattern(&pattern_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_array_count(count: u32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_array_count(count);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_array_radius(radius: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_array_radius(radius);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_array_color_variation(value: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_array_color_variation(value);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_intensity_expression(expression: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_intensity_expression(&expression);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_selected_light_color_hue_expression(expression: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_selected_light_color_hue_expression(&expression);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_node_light_mask(node_id: u64, mask: u8) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_node_light_mask(node_id, mask);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_node_light_link_enabled(node_id: u64, light_id: u64, enabled: bool) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_node_light_link_enabled(node_id, light_id, enabled);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn set_manipulator_mode(mode_id: String) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.set_manipulator_mode(&mode_id);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_manipulator_space() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_manipulator_space();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_manipulator_pivot_offset(x: f32, y: f32, z: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_manipulator_pivot_offset(x, y, z);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_manipulator_pivot() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_manipulator_pivot();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_translation(delta_x: f32, delta_y: f32, delta_z: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_translation(delta_x, delta_y, delta_z);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_rotation_degrees(
    delta_x_degrees: f32,
    delta_y_degrees: f32,
    delta_z_degrees: f32,
) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_rotation_degrees(delta_x_degrees, delta_y_degrees, delta_z_degrees);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn nudge_selected_scale(delta_x: f32, delta_y: f32, delta_z: f32) -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.nudge_selected_scale(delta_x, delta_y, delta_z);
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn undo() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.undo();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn redo() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.redo();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn focus_selected() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.focus_selected();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn frame_all() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.frame_all();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_front() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_front();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_top() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_top();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_right() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_right();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_back() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_back();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_left() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_left();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn camera_bottom() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.camera_bottom();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn toggle_orthographic() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.toggle_orthographic();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_sphere() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_sphere();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_box() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_box();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_cylinder() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_cylinder();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn add_torus() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.add_torus();
    current_scene_snapshot(&mut bridge)
}

#[flutter_rust_bridge::frb(sync)]
pub fn reset_scene() -> AppSceneSnapshot {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");
    bridge.reset_scene();
    current_scene_snapshot(&mut bridge)
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
        "array" => LightType::Array,
        _ => return None,
    })
}
