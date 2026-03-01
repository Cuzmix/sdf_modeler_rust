import 'dart:async';
import 'dart:ui' as ui;
import 'package:flutter/widgets.dart';
import '../../core/constants.dart';
import '../../models/camera_state.dart';
import '../../rust/bridge.dart' as bridge;
import '../../rust/bridge/gpu_context.dart';
import '../../rust/bridge/scene_handle.dart';

/// Handles the render loop: calls Rust via FFI, decodes pixels, manages
/// resolution scaling during interaction, and prevents overlapping renders.
class ViewportRenderer {
  bool _rendering = false;
  bool _interacting = false;
  Timer? _debounceTimer;

  bool get isRendering => _rendering;

  /// Begin an interaction (e.g. drag start) — lowers resolution for speed.
  void startInteraction() {
    _interacting = true;
    _debounceTimer?.cancel();
  }

  /// End an interaction (e.g. drag end) — debounces before full-quality render.
  void endInteraction(VoidCallback requestRender) {
    _debounceTimer?.cancel();
    _debounceTimer = Timer(
      const Duration(milliseconds: interactionDebounceMs),
      () {
        _interacting = false;
        requestRender();
      },
    );
  }

  /// Render a single frame and return a decoded [ui.Image], or null on error.
  /// Returns null if already rendering (caller should retry via dirty flag).
  Future<ui.Image?> renderFrame({
    required CameraState camera,
    required GpuContext gpu,
    required SceneHandle scene,
    required Size size,
    required double devicePixelRatio,
  }) async {
    if (_rendering) return null;
    _rendering = true;

    try {
      final scale = _interacting ? interactionScale : fullScale;
      final quality = _interacting ? interactionQuality : fullQuality;
      final w = (size.width * devicePixelRatio * scale)
          .round()
          .clamp(1, maxTextureSize);
      final h = (size.height * devicePixelRatio * scale)
          .round()
          .clamp(1, maxTextureSize);

      camera.clearDirty();

      final pixels = await bridge.renderFrame(
        gpu: gpu,
        scene: scene,
        yaw: camera.yaw,
        pitch: camera.pitch,
        distance: camera.distance,
        targetX: camera.targetX,
        targetY: camera.targetY,
        targetZ: camera.targetZ,
        fov: camera.fov,
        width: w,
        height: h,
        qualityMode: quality,
        time: 0.0,
        gridEnabled: true,
      );

      final completer = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        pixels,
        w,
        h,
        ui.PixelFormat.rgba8888,
        (img) => completer.complete(img),
      );
      return await completer.future;
    } catch (e) {
      debugPrint('Render error: $e');
      return null;
    } finally {
      _rendering = false;
    }
  }

  void dispose() {
    _debounceTimer?.cancel();
  }
}
