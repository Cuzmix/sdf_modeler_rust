use std::mem::ManuallyDrop;
use std::ptr::null_mut;

use crate::api::mirrors::{
    AppCameraSnapshot,
    AppNodeSnapshot,
    AppVec3,
    AppViewportFeedbackSnapshot,
};
use crate::bridge_state::app_bridge;

const NATIVE_VIEWPORT_COMMAND_PICK: u32 = 1;
const NATIVE_VIEWPORT_COMMAND_HOVER: u32 = 1 << 1;
const NATIVE_VIEWPORT_COMMAND_CLEAR_HOVER: u32 = 1 << 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeViewportCommands {
    pub orbit_delta_x: f32,
    pub orbit_delta_y: f32,
    pub pan_delta_x: f32,
    pub pan_delta_y: f32,
    pub zoom_delta: f32,
    pub pick_normalized_x: f32,
    pub pick_normalized_y: f32,
    pub hover_normalized_x: f32,
    pub hover_normalized_y: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeViewportVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<AppVec3> for NativeViewportVec3 {
    fn from(value: AppVec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeViewportCameraSnapshot {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub distance: f32,
    pub fov_degrees: f32,
    pub orthographic: u8,
    pub target: NativeViewportVec3,
    pub eye: NativeViewportVec3,
}

impl From<AppCameraSnapshot> for NativeViewportCameraSnapshot {
    fn from(value: AppCameraSnapshot) -> Self {
        Self {
            yaw: value.yaw,
            pitch: value.pitch,
            roll: value.roll,
            distance: value.distance,
            fov_degrees: value.fov_degrees,
            orthographic: u8::from(value.orthographic),
            target: value.target.into(),
            eye: value.eye.into(),
        }
    }
}

#[repr(C)]
pub struct NativeViewportString {
    pub ptr: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl Default for NativeViewportString {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

impl NativeViewportString {
    fn from_string(value: String) -> Self {
        let mut bytes = ManuallyDrop::new(value.into_bytes());
        Self {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        }
    }
}

#[derive(Default)]
#[repr(C)]
pub struct NativeViewportNodeSnapshot {
    pub id: u64,
    pub visible: u8,
    pub locked: u8,
    pub name: NativeViewportString,
    pub kind_label: NativeViewportString,
}

impl From<AppNodeSnapshot> for NativeViewportNodeSnapshot {
    fn from(value: AppNodeSnapshot) -> Self {
        Self {
            id: value.id,
            visible: u8::from(value.visible),
            locked: u8::from(value.locked),
            name: NativeViewportString::from_string(value.name),
            kind_label: NativeViewportString::from_string(value.kind_label),
        }
    }
}

#[repr(C)]
pub struct NativeViewportFrame {
    pub pixels_ptr: *mut u8,
    pub pixels_len: usize,
    pub pixels_capacity: usize,
    pub camera: NativeViewportCameraSnapshot,
    pub selected_node_present: u8,
    pub selected_node: NativeViewportNodeSnapshot,
    pub hovered_node_present: u8,
    pub hovered_node: NativeViewportNodeSnapshot,
    pub camera_animating: u8,
}

fn free_native_string(value: NativeViewportString) {
    if !value.ptr.is_null() && value.capacity > 0 {
        unsafe {
            drop(Vec::from_raw_parts(value.ptr, value.len, value.capacity));
        }
    }
}

fn free_native_node_snapshot(value: NativeViewportNodeSnapshot) {
    free_native_string(value.name);
    free_native_string(value.kind_label);
}

#[unsafe(no_mangle)]
pub extern "C" fn sdf_modeler_native_process_viewport_frame(
    width: u32,
    height: u32,
    time_seconds: f32,
    commands: NativeViewportCommands,
) -> NativeViewportFrame {
    let mut bridge = app_bridge().lock().expect("app bridge mutex");

    if commands.orbit_delta_x != 0.0 || commands.orbit_delta_y != 0.0 {
        bridge.orbit_camera(commands.orbit_delta_x, commands.orbit_delta_y);
    }

    if commands.pan_delta_x != 0.0 || commands.pan_delta_y != 0.0 {
        bridge.pan_camera(commands.pan_delta_x, commands.pan_delta_y);
    }

    if commands.zoom_delta != 0.0 {
        bridge.zoom_camera(commands.zoom_delta);
    }

    if (commands.flags & NATIVE_VIEWPORT_COMMAND_CLEAR_HOVER) != 0 {
        bridge.clear_hovered_node();
    }

    if (commands.flags & NATIVE_VIEWPORT_COMMAND_HOVER) != 0 {
        let hover_mouse_x = commands.hover_normalized_x.clamp(0.0, 1.0)
            * (width.saturating_sub(1) as f32);
        let hover_mouse_y = commands.hover_normalized_y.clamp(0.0, 1.0)
            * (height.saturating_sub(1) as f32);
        bridge.hover_node_at_viewport(hover_mouse_x, hover_mouse_y, width, height, time_seconds);
    }

    if (commands.flags & NATIVE_VIEWPORT_COMMAND_PICK) != 0 {
        let pick_mouse_x = commands.pick_normalized_x.clamp(0.0, 1.0)
            * (width.saturating_sub(1) as f32);
        let pick_mouse_y = commands.pick_normalized_y.clamp(0.0, 1.0)
            * (height.saturating_sub(1) as f32);
        bridge.select_node_at_viewport(pick_mouse_x, pick_mouse_y, width, height, time_seconds);
    }

    let rendered_frame = bridge.render_viewport_frame(width, height, time_seconds);
    let AppViewportFeedbackSnapshot {
        camera,
        selected_node,
        hovered_node,
    } = bridge.viewport_feedback().into();
    let selected_node = selected_node.map(Into::into);
    let hovered_node = hovered_node.map(Into::into);
    let mut pixels = ManuallyDrop::new(rendered_frame.pixels);

    NativeViewportFrame {
        pixels_ptr: pixels.as_mut_ptr(),
        pixels_len: pixels.len(),
        pixels_capacity: pixels.capacity(),
        camera: camera.into(),
        selected_node_present: u8::from(selected_node.is_some()),
        selected_node: selected_node.unwrap_or_default(),
        hovered_node_present: u8::from(hovered_node.is_some()),
        hovered_node: hovered_node.unwrap_or_default(),
        camera_animating: u8::from(rendered_frame.camera_animating),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sdf_modeler_native_free_viewport_frame(frame: NativeViewportFrame) {
    if !frame.pixels_ptr.is_null() && frame.pixels_capacity > 0 {
        unsafe {
            drop(Vec::from_raw_parts(
                frame.pixels_ptr,
                frame.pixels_len,
                frame.pixels_capacity,
            ));
        }
    }

    free_native_node_snapshot(frame.selected_node);
    free_native_node_snapshot(frame.hovered_node);
}
