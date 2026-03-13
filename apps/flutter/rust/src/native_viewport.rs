use std::mem::ManuallyDrop;

use crate::bridge_state::{app_bridge, viewport_feedback_json};

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
pub struct NativeViewportFrame {
    pub pixels_ptr: *mut u8,
    pub pixels_len: usize,
    pub pixels_capacity: usize,
    pub feedback_ptr: *mut u8,
    pub feedback_len: usize,
    pub feedback_capacity: usize,
    pub camera_animating: u8,
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
    let feedback = viewport_feedback_json(&bridge).into_bytes();
    let mut pixels = ManuallyDrop::new(rendered_frame.pixels);
    let mut feedback = ManuallyDrop::new(feedback);

    NativeViewportFrame {
        pixels_ptr: pixels.as_mut_ptr(),
        pixels_len: pixels.len(),
        pixels_capacity: pixels.capacity(),
        feedback_ptr: feedback.as_mut_ptr(),
        feedback_len: feedback.len(),
        feedback_capacity: feedback.capacity(),
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

    if !frame.feedback_ptr.is_null() && frame.feedback_capacity > 0 {
        unsafe {
            drop(Vec::from_raw_parts(
                frame.feedback_ptr,
                frame.feedback_len,
                frame.feedback_capacity,
            ));
        }
    }
}