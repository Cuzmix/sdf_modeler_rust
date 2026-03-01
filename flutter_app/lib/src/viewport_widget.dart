import 'dart:async';
import 'dart:ui' as ui;
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import '../src/rust/bridge.dart' as bridge;
import '../src/rust/bridge/gpu_context.dart';
import '../src/rust/bridge/scene_handle.dart';
import 'camera_state.dart';

/// Displays a live SDF viewport rendered by Rust/wgpu.
/// Handles orbit (LMB drag), pan (RMB drag), and zoom (scroll).
class ViewportWidget extends StatefulWidget {
  final GpuContext gpu;
  final SceneHandle scene;

  const ViewportWidget({super.key, required this.gpu, required this.scene});

  @override
  State<ViewportWidget> createState() => _ViewportWidgetState();
}

class _ViewportWidgetState extends State<ViewportWidget> {
  final CameraState _camera = CameraState();
  ui.Image? _image;
  bool _rendering = false;
  bool _interacting = false;
  int _activeButton = 0; // tracks which mouse button is held
  Timer? _debounceTimer;

  @override
  void initState() {
    super.initState();
    _camera.addListener(_onCameraChanged);
    _requestRender();
  }

  @override
  void dispose() {
    _camera.removeListener(_onCameraChanged);
    _debounceTimer?.cancel();
    _image?.dispose();
    super.dispose();
  }

  void _onCameraChanged() {
    _requestRender();
  }

  void _requestRender() {
    if (_rendering) return;
    _rendering = true;
    _doRender();
  }

  Future<void> _doRender() async {
    if (!mounted) {
      _rendering = false;
      return;
    }

    final box = context.findRenderObject() as RenderBox?;
    if (box == null || !box.hasSize) {
      _rendering = false;
      // Retry next frame when layout is ready
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _rendering = false;
        _requestRender();
      });
      return;
    }

    final dpr = MediaQuery.of(context).devicePixelRatio;
    final scale = _interacting ? 0.5 : 1.0;
    final quality = _interacting ? 1.0 : 0.0;
    final w = (box.size.width * dpr * scale).round().clamp(1, 4096);
    final h = (box.size.height * dpr * scale).round().clamp(1, 4096);

    _camera.clearDirty();

    try {
      final pixels = await bridge.renderFrame(
        gpu: widget.gpu,
        scene: widget.scene,
        yaw: _camera.yaw,
        pitch: _camera.pitch,
        distance: _camera.distance,
        targetX: _camera.targetX,
        targetY: _camera.targetY,
        targetZ: _camera.targetZ,
        fov: _camera.fov,
        width: w,
        height: h,
        qualityMode: quality,
        time: 0.0,
        gridEnabled: true,
      );

      if (!mounted) {
        _rendering = false;
        return;
      }

      final completer = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        pixels,
        w,
        h,
        ui.PixelFormat.rgba8888,
        (img) => completer.complete(img),
      );
      final newImage = await completer.future;

      if (!mounted) {
        newImage.dispose();
        _rendering = false;
        return;
      }

      final old = _image;
      setState(() {
        _image = newImage;
      });
      old?.dispose();
    } catch (e) {
      debugPrint('Render error: $e');
    }

    _rendering = false;

    // If camera moved during render, re-render
    if (_camera.dirty) {
      _requestRender();
    }
  }

  void _onInteractionStart() {
    _interacting = true;
    _debounceTimer?.cancel();
  }

  void _onInteractionEnd() {
    _debounceTimer?.cancel();
    _debounceTimer = Timer(const Duration(milliseconds: 100), () {
      _interacting = false;
      _requestRender();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Listener(
      onPointerDown: (event) {
        _activeButton = event.buttons;
        _onInteractionStart();
      },
      onPointerMove: (event) {
        final dx = event.delta.dx;
        final dy = event.delta.dy;
        if (_activeButton & kSecondaryMouseButton != 0) {
          _camera.pan(dx, dy);
        } else {
          _camera.orbit(dx, dy);
        }
      },
      onPointerUp: (_) {
        _activeButton = 0;
        _onInteractionEnd();
      },
      onPointerSignal: (event) {
        if (event is PointerScrollEvent) {
          _onInteractionStart();
          _camera.zoom(-event.scrollDelta.dy);
          _onInteractionEnd();
        }
      },
      child: _image != null
          ? RawImage(
              image: _image,
              fit: BoxFit.contain,
              width: double.infinity,
              height: double.infinity,
            )
          : const Center(
              child: CircularProgressIndicator(),
            ),
    );
  }
}
