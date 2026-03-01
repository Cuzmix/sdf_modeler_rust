/// Flutter bridge API — thin FFI layer between Dart and the Rust core.
/// All public functions here are exposed to Dart via flutter_rust_bridge codegen.

#[cfg(feature = "flutter_ui")]
pub mod gpu_context;
#[cfg(feature = "flutter_ui")]
pub mod scene_handle;

#[cfg(feature = "flutter_ui")]
pub use gpu_context::GpuContext;
#[cfg(feature = "flutter_ui")]
pub use scene_handle::SceneHandle;

/// Simple hello function to verify the bridge is working end-to-end.
pub fn hello_from_rust() -> String {
    "Hello from SDF Modeler Rust core!".to_string()
}

/// Returns the wgpu backend name for the default adapter (useful for diagnostics).
pub fn gpu_backend_name() -> String {
    format!("wgpu {}", env!("CARGO_PKG_VERSION"))
}

// ---------------------------------------------------------------------------
// Phase 1: GPU context + scene + rendering
// ---------------------------------------------------------------------------

#[cfg(feature = "flutter_ui")]
pub fn create_gpu_context() -> Result<GpuContext, String> {
    GpuContext::new()
}

#[cfg(feature = "flutter_ui")]
pub fn create_scene() -> SceneHandle {
    SceneHandle::new()
}

/// Synchronize GPU pipelines and buffers with the current scene state.
/// Call this after any scene modification (add/remove nodes, change topology).
#[cfg(feature = "flutter_ui")]
pub fn sync_gpu(gpu: &mut GpuContext, scene: &mut SceneHandle) {
    use crate::gpu::{buffers, codegen};

    // Check if scene topology changed — rebuild shaders + pipelines if so
    let new_key = scene.scene.structure_key();
    if new_key != scene.current_structure_key {
        let shader_src = codegen::generate_shader(&scene.scene, &scene.config);
        let pick_shader_src = codegen::generate_pick_shader(&scene.scene, &scene.config);
        let sculpt_count = buffers::collect_sculpt_tex_info(&scene.scene).len();
        gpu.resources.rebuild_pipeline(
            &gpu.device, &shader_src, &pick_shader_src, sculpt_count,
        );
        scene.current_structure_key = new_key;
    }

    // Upload node + voxel data to GPU
    let (voxel_data, voxel_offsets) = buffers::build_voxel_buffer(&scene.scene);
    let node_data = buffers::build_node_buffer(&scene.scene, None, &voxel_offsets);
    let sculpt_infos = buffers::collect_sculpt_tex_info(&scene.scene);

    gpu.resources.update_scene_buffer(&gpu.device, &gpu.queue, &node_data);
    gpu.resources.update_voxel_buffer(&gpu.device, &gpu.queue, &voxel_data);

    // Upload voxel textures for sculpt nodes
    for info in &sculpt_infos {
        if let Some(node) = scene.scene.nodes.get(&info.node_id) {
            if let crate::graph::scene::NodeData::Sculpt { ref voxel_grid, .. } = node.data {
                gpu.resources.upload_voxel_texture(
                    &gpu.device, &gpu.queue, info.tex_idx,
                    voxel_grid.resolution, &voxel_grid.data,
                );
            }
        }
    }
}

/// Render a single frame and return RGBA pixel data.
/// Camera parameters are passed as primitives to avoid FFI struct complexity.
#[cfg(feature = "flutter_ui")]
pub fn render_frame(
    gpu: &GpuContext,
    scene: &SceneHandle,
    // Camera
    yaw: f32,
    pitch: f32,
    distance: f32,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    fov: f32,
    // Viewport
    width: u32,
    height: u32,
    // Render params
    quality_mode: f32,
    time: f32,
    grid_enabled: bool,
) -> Vec<u8> {
    use crate::gpu::camera::Camera;

    let camera = Camera {
        yaw,
        pitch,
        distance,
        target: glam::Vec3::new(target_x, target_y, target_z),
        fov,
    };

    let scene_bounds = scene.scene.compute_bounds();
    let uniform = camera.to_uniform(
        [0.0, 0.0, width as f32, height as f32],
        time,
        quality_mode,
        grid_enabled,
        scene_bounds,
        -1.0, // no selection highlight
    );

    gpu.resources.screenshot(&gpu.device, &gpu.queue, &uniform, width, height)
}
