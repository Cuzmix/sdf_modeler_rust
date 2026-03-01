import 'dart:ui' as ui;
import 'package:flutter/widgets.dart';
import '../../models/camera_state.dart';
import '../../rust/bridge/gpu_context.dart';
import '../../rust/bridge/scene_handle.dart';
import 'viewport_display.dart';
import 'viewport_gesture_handler.dart';
import 'viewport_renderer.dart';

/// Orchestrates the SDF viewport: wires camera → renderer → display.
class ViewportWidget extends StatefulWidget {
  final GpuContext gpu;
  final SceneHandle scene;

  const ViewportWidget({super.key, required this.gpu, required this.scene});

  @override
  State<ViewportWidget> createState() => _ViewportWidgetState();
}

class _ViewportWidgetState extends State<ViewportWidget> {
  final CameraState _camera = CameraState();
  final ViewportRenderer _renderer = ViewportRenderer();
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _camera.addListener(_onCameraChanged);
    _requestRender();
  }

  @override
  void dispose() {
    _camera.removeListener(_onCameraChanged);
    _renderer.dispose();
    _image?.dispose();
    super.dispose();
  }

  void _onCameraChanged() => _requestRender();

  void _requestRender() async {
    if (!mounted) return;

    final box = context.findRenderObject() as RenderBox?;
    if (box == null || !box.hasSize) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _requestRender());
      return;
    }

    final dpr = MediaQuery.of(context).devicePixelRatio;
    final newImage = await _renderer.renderFrame(
      camera: _camera,
      gpu: widget.gpu,
      scene: widget.scene,
      size: box.size,
      devicePixelRatio: dpr,
    );

    if (!mounted) {
      newImage?.dispose();
      return;
    }

    if (newImage != null) {
      final old = _image;
      setState(() => _image = newImage);
      old?.dispose();
    }

    // Re-render if camera moved during this frame
    if (_camera.dirty) _requestRender();
  }

  @override
  Widget build(BuildContext context) {
    return ViewportGestureHandler(
      camera: _camera,
      onInteractionStart: () => _renderer.startInteraction(),
      onInteractionEnd: () => _renderer.endInteraction(_requestRender),
      child: ViewportDisplay(image: _image),
    );
  }
}
