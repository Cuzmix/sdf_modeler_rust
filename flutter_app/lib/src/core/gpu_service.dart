import '../rust/bridge.dart' as bridge;
import '../rust/bridge/gpu_context.dart';
import '../rust/bridge/scene_handle.dart';

/// Result of GPU initialisation — holds the two opaque Rust handles.
typedef GpuInitResult = ({GpuContext gpu, SceneHandle scene});

/// Initialises wgpu + default scene on a background thread (via FRB).
/// Throws [String] on failure.
Future<GpuInitResult> initGpu() async {
  final gpu = await bridge.createGpuContext();
  final scene = await bridge.createScene();
  await bridge.syncGpu(gpu: gpu, scene: scene);
  return (gpu: gpu, scene: scene);
}
