//! Background compilation of the scene render + pick pipelines.
//!
//! Adding or removing a scene node bumps `Scene::structure_version`, which
//! means the generated WGSL scene SDF function is now stale and the render +
//! pick pipelines must be rebuilt. Rebuilding is expensive: `wgpu::Device`
//! has to parse the WGSL source, run `naga` validation, then hand the
//! validated module off to the graphics backend for final shader compilation.
//! For scenes with many nodes this can take tens of milliseconds — enough
//! to drop a frame whenever the user places a primitive.
//!
//! Doing this work on the main (UI / render) thread is what causes the
//! visible "stutter" when adding objects. This module moves compilation to a
//! short-lived worker thread. The currently active pipeline keeps rendering
//! while the worker builds the new one; as soon as the worker returns, the
//! main thread atomically swaps the old pipelines for the new ones.
//!
//! The worker is intentionally minimal: the main thread still owns the
//! scene, performs codegen, and manages any state that must change together
//! with the pipeline (voxel texture bind groups, composite volume
//! resources). Only the heavy `wgpu` calls run off-thread.
//!
//! If the scene's `structure_version` advances while a compile is in flight,
//! the main thread just waits for it to land. On apply it sets
//! `last_structure_version` to the compiled version; on the next frame the
//! natural `structure_version != last_structure_version` check sees the
//! scene has moved on and kicks off a fresh compile. Rendering the
//! slightly-stale pipeline for one or two frames is harmless — it draws the
//! previous scene topology against the current data buffer, which silently
//! ignores any extra nodes since the shader hard-codes node names.

use std::sync::{mpsc, Arc};
use std::thread;

use eframe::wgpu;

use crate::compat::Instant;

use crate::ui::viewport::ViewportResources;

/// Everything the worker thread needs to build a new render + pick pipeline.
///
/// Strings are owned (moved into the worker) so the main thread is free to
/// continue mutating the scene. `BindGroupLayout` is not `Clone` in wgpu
/// 22.1, so each BGL is stored once behind an `Arc` on
/// `ViewportResources` and cloned-as-`Arc` into the worker.
pub(crate) struct PipelineCompileInputs {
    pub device: Arc<wgpu::Device>,
    pub render_shader_src: String,
    pub pick_shader_src: String,
    pub target_format: wgpu::TextureFormat,
    pub camera_bgl: Arc<wgpu::BindGroupLayout>,
    pub scene_bgl: Arc<wgpu::BindGroupLayout>,
    pub voxel_tex_bgl: Arc<wgpu::BindGroupLayout>,
    pub environment_bgl: Arc<wgpu::BindGroupLayout>,
    pub pick_bgl: Arc<wgpu::BindGroupLayout>,
    /// The `structure_version` of the scene this compile targets.
    pub structure_version: u64,
    /// Sculpt count the shader was generated against. Used by the main thread
    /// to confirm the returned pipeline is still compatible with the live
    /// voxel texture bind group.
    pub sculpt_count: usize,
}

/// Compiled pipeline pair returned from the worker.
pub(crate) struct CompiledPipelines {
    pub render_pipeline: wgpu::RenderPipeline,
    pub pick_pipeline: wgpu::ComputePipeline,
    pub structure_version: u64,
    pub sculpt_count: usize,
    /// Wall-clock time spent inside the worker thread, for profiling.
    pub worker_wall_ms: f64,
}

/// Handle for an in-flight compile. The main thread polls the receiver each
/// frame to pick up the completed pipeline.
pub(crate) struct PipelineCompileHandle {
    pub receiver: mpsc::Receiver<CompiledPipelines>,
}

/// Spawn a worker thread that compiles the render + pick pipelines and sends
/// the result back via an mpsc channel. Returns a handle the main thread can
/// poll without blocking.
pub(crate) fn spawn_pipeline_compile(inputs: PipelineCompileInputs) -> PipelineCompileHandle {
    let (sender, receiver) = mpsc::channel();

    thread::Builder::new()
        .name("sdf-pipeline-compile".into())
        .spawn(move || {
            let work_start = Instant::now();
            let render_pipeline = ViewportResources::create_render_pipeline(
                &inputs.device,
                &inputs.render_shader_src,
                &inputs.camera_bgl,
                &inputs.scene_bgl,
                &inputs.voxel_tex_bgl,
                &inputs.environment_bgl,
                inputs.target_format,
            );
            let pick_pipeline = ViewportResources::create_pick_pipeline(
                &inputs.device,
                &inputs.pick_shader_src,
                &inputs.camera_bgl,
                &inputs.scene_bgl,
                &inputs.pick_bgl,
            );
            let worker_wall_ms = work_start.elapsed().as_secs_f64() * 1000.0;
            let _ = sender.send(CompiledPipelines {
                render_pipeline,
                pick_pipeline,
                structure_version: inputs.structure_version,
                sculpt_count: inputs.sculpt_count,
                worker_wall_ms,
            });
        })
        .expect("failed to spawn sdf-pipeline-compile worker thread");

    PipelineCompileHandle { receiver }
}
