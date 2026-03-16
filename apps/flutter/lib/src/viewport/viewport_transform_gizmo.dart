import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_gizmo_math.dart';

export 'package:sdf_modeler_flutter/src/viewport/viewport_gizmo_math.dart'
    show ViewportGizmoTransformData;

class ViewportTransformGizmoController extends ChangeNotifier {
  ViewportTransformGizmoController({
    required void Function() beginInteractiveEdit,
    required void Function(ViewportGizmoTransformData transform) previewTransform,
    required Future<void> Function(ViewportGizmoTransformData transform)
    commitTransform,
    required VoidCallback onViewportInteraction,
  }) : _beginInteractiveEdit = beginInteractiveEdit,
       _previewTransform = previewTransform,
       _commitTransform = commitTransform,
       _onViewportInteraction = onViewportInteraction;

  final void Function() _beginInteractiveEdit;
  final void Function(ViewportGizmoTransformData transform) _previewTransform;
  final Future<void> Function(ViewportGizmoTransformData transform)
  _commitTransform;
  final VoidCallback _onViewportInteraction;

  ViewportGizmoAxis? _hoveredAxis;
  ViewportGizmoDragSession? _dragSession;
  ViewportGizmoTransformData? _previewOverride;
  bool _interactivePreviewStarted = false;

  ViewportGizmoAxis? get hoveredAxis => _hoveredAxis;
  ViewportGizmoAxis? get activeAxis => _dragSession?.axis;

  ViewportGizmoTransformData? transformOverrideForSnapshot(
    AppSceneSnapshot? snapshot,
  ) {
    final selectedNodeId = snapshot?.selectedNode?.id;
    if (selectedNodeId == null || _previewOverride == null) {
      return null;
    }
    if (_dragSession != null && _dragSession!.selectedNodeId == selectedNodeId) {
      return _previewOverride;
    }
    return null;
  }

  bool handlePointerEvent(
    PointerEvent event,
    Size viewportSize, {
    required AppSceneSnapshot? snapshot,
    required TextureViewportFeedback? feedback,
    required bool enabled,
  }) {
    if (ShellGestureContract.isTouchLike(event.kind)) {
      return false;
    }

    if (!enabled) {
      _resetInteractionState();
      return false;
    }

    if (_dragSession != null) {
      return _handleActiveDrag(event);
    }

    final scene = buildViewportGizmoSceneState(
      snapshot,
      feedback,
      transformOverride: transformOverrideForSnapshot(snapshot),
    );
    if (scene == null) {
      _updateHoveredAxis(null);
      return false;
    }

    final geometry = computeViewportGizmoGeometry(scene, viewportSize);
    if (geometry == null) {
      _updateHoveredAxis(null);
      return false;
    }

    return switch (event) {
      PointerHoverEvent() => _handleHover(event, geometry),
      PointerMoveEvent() when event.buttons == 0 => _handleHover(event, geometry),
      PointerDownEvent() => _handlePointerDown(event, scene, geometry),
      PointerExitEvent() => _handlePointerExit(),
      _ => false,
    };
  }

  Offset? debugAxisHandlePosition(
    String axisId,
    Size viewportSize, {
    required AppSceneSnapshot? snapshot,
    required TextureViewportFeedback? feedback,
  }) {
    final axis = ViewportGizmoAxis.fromAxisId(axisId);
    if (axis == null) {
      return null;
    }

    final scene = buildViewportGizmoSceneState(
      snapshot,
      feedback,
      transformOverride: transformOverrideForSnapshot(snapshot),
    );
    if (scene == null) {
      return null;
    }

    final geometry = computeViewportGizmoGeometry(scene, viewportSize);
    if (geometry == null) {
      return null;
    }

    if (scene.tool.manipulatorModeId == 'rotate') {
      final segments = geometry.rotationRingSegments[axis];
      if (segments == null || segments.isEmpty) {
        return null;
      }
      final segment = segments.first;
      return Offset.lerp(segment.start, segment.end, 0.5);
    }

    return geometry.axisEndpoints[axis];
  }

  bool _handleHover(PointerEvent event, ViewportGizmoGeometry geometry) {
    final hoveredAxis = hitTestViewportGizmoAxis(
      event.localPosition,
      geometry,
      modeId: geometry.scene.tool.manipulatorModeId,
    );
    _updateHoveredAxis(hoveredAxis);
    return hoveredAxis != null;
  }

  bool _handlePointerDown(
    PointerDownEvent event,
    ViewportGizmoSceneState scene,
    ViewportGizmoGeometry geometry,
  ) {
    if ((event.buttons & kPrimaryMouseButton) == 0) {
      return false;
    }

    final hitAxis = hitTestViewportGizmoAxis(
      event.localPosition,
      geometry,
      modeId: scene.tool.manipulatorModeId,
    );
    if (hitAxis == null) {
      _updateHoveredAxis(null);
      return false;
    }

    _dragSession = beginViewportGizmoDrag(
      scene,
      geometry,
      hitAxis,
      event.localPosition,
    );
    _previewOverride = scene.transform;
    _interactivePreviewStarted = false;
    _updateHoveredAxis(hitAxis, notify: false);
    notifyListeners();
    return true;
  }

  bool _handlePointerExit() {
    _updateHoveredAxis(null);
    return false;
  }

  bool _handleActiveDrag(PointerEvent event) {
    final dragSession = _dragSession;
    if (dragSession == null) {
      return false;
    }

    switch (event) {
      case PointerMoveEvent():
        final totalDistance =
            (event.localPosition - dragSession.startPointerPosition).distance;
        if (!_interactivePreviewStarted) {
          final dragSlop = ShellGestureContract.dragStartSlopFor(event.kind);
          if (totalDistance < dragSlop) {
            return true;
          }
          _interactivePreviewStarted = true;
          _beginInteractiveEdit();
        }

        _onViewportInteraction();
        final nextTransform = dragSession.evaluateTransform(event.localPosition);
        _previewOverride = nextTransform;
        _updateHoveredAxis(dragSession.axis, notify: false);
        notifyListeners();
        _previewTransform(nextTransform);
        return true;
      case PointerUpEvent():
        final transformToCommit = _previewOverride;
        final shouldCommit = _interactivePreviewStarted && transformToCommit != null;
        _resetInteractionState();
        if (shouldCommit) {
          _onViewportInteraction();
          unawaited(_commitTransform(transformToCommit));
        }
        return true;
      case PointerCancelEvent():
        _resetInteractionState();
        return true;
      case PointerExitEvent():
        return true;
      default:
        return true;
    }
  }

  void _updateHoveredAxis(
    ViewportGizmoAxis? axis, {
    bool notify = true,
  }) {
    if (_hoveredAxis == axis) {
      return;
    }
    _hoveredAxis = axis;
    if (notify) {
      notifyListeners();
    }
  }

  void _resetInteractionState() {
    if (_hoveredAxis == null &&
        _dragSession == null &&
        _previewOverride == null &&
        !_interactivePreviewStarted) {
      return;
    }

    _hoveredAxis = null;
    _dragSession = null;
    _previewOverride = null;
    _interactivePreviewStarted = false;
    notifyListeners();
  }
}

class ViewportTransformGizmoOverlay extends StatelessWidget {
  const ViewportTransformGizmoOverlay({
    super.key,
    required this.controller,
    required this.snapshot,
    required this.feedback,
    required this.enabled,
  });

  final ViewportTransformGizmoController controller;
  final AppSceneSnapshot? snapshot;
  final TextureViewportFeedback? feedback;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final viewportSize = Size(constraints.maxWidth, constraints.maxHeight);
        return AnimatedBuilder(
          animation: controller,
          builder: (context, child) {
            final scene = buildViewportGizmoSceneState(
              snapshot,
              feedback,
              transformOverride: controller.transformOverrideForSnapshot(snapshot),
            );
            if (scene == null) {
              return const SizedBox.expand();
            }

            final geometry = computeViewportGizmoGeometry(scene, viewportSize);
            if (geometry == null) {
              return const SizedBox.expand();
            }

            return CustomPaint(
              painter: _ViewportTransformGizmoPainter(
                geometry: geometry,
                hoveredAxis: controller.hoveredAxis,
                activeAxis: controller.activeAxis,
                enabled: enabled,
              ),
              child: const SizedBox.expand(),
            );
          },
        );
      },
    );
  }
}

class _ViewportTransformGizmoPainter extends CustomPainter {
  const _ViewportTransformGizmoPainter({
    required this.geometry,
    required this.hoveredAxis,
    required this.activeAxis,
    required this.enabled,
  });

  static const double _axisStrokeWidth = 2.5;
  static const double _arrowSize = 8.0;
  static const double _scaleBoxSize = 6.0;

  final ViewportGizmoGeometry geometry;
  final ViewportGizmoAxis? hoveredAxis;
  final ViewportGizmoAxis? activeAxis;
  final bool enabled;

  @override
  void paint(Canvas canvas, Size size) {
    switch (geometry.scene.tool.manipulatorModeId) {
      case 'rotate':
        _paintRotateGizmo(canvas);
      case 'scale':
        _paintScaleGizmo(canvas);
      default:
        _paintTranslateGizmo(canvas);
    }

    if (geometry.scene.tool.canResetPivot && geometry.nodeOriginScreen != null) {
      final pivotPaint = Paint()
        ..color = _applyEnabledOpacity(const Color(0xFFFFC850))
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
      final pivotLinePaint = Paint()
        ..color = _applyEnabledOpacity(const Color(0x66FFC850))
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;

      canvas.drawLine(
        geometry.nodeOriginScreen!,
        geometry.originScreen,
        pivotLinePaint,
      );
      canvas.drawCircle(
        geometry.nodeOriginScreen!,
        4.0,
        pivotPaint,
      );
    }
  }

  @override
  bool shouldRepaint(_ViewportTransformGizmoPainter oldDelegate) {
    return oldDelegate.geometry != geometry ||
        oldDelegate.hoveredAxis != hoveredAxis ||
        oldDelegate.activeAxis != activeAxis ||
        oldDelegate.enabled != enabled;
  }

  void _paintTranslateGizmo(Canvas canvas) {
    for (final axis in ViewportGizmoAxis.values) {
      final endpoint = geometry.axisEndpoints[axis];
      final direction = geometry.screenAxisDirections[axis];
      if (endpoint == null || direction == null) {
        continue;
      }

      final strokePaint = Paint()
        ..color = _axisColor(axis)
        ..style = PaintingStyle.stroke
        ..strokeWidth = _axisStrokeWidth;
      canvas.drawLine(geometry.originScreen, endpoint, strokePaint);

      final path = Path()
        ..moveTo(endpoint.dx, endpoint.dy)
        ..lineTo(
          endpoint.dx - direction.dx * _arrowSize + direction.dy * (_arrowSize * 0.5),
          endpoint.dy - direction.dy * _arrowSize - direction.dx * (_arrowSize * 0.5),
        )
        ..lineTo(
          endpoint.dx - direction.dx * _arrowSize - direction.dy * (_arrowSize * 0.5),
          endpoint.dy - direction.dy * _arrowSize + direction.dx * (_arrowSize * 0.5),
        )
        ..close();

      final fillPaint = Paint()
        ..color = _axisColor(axis)
        ..style = PaintingStyle.fill;
      canvas.drawPath(path, fillPaint);
    }
  }

  void _paintScaleGizmo(Canvas canvas) {
    for (final axis in ViewportGizmoAxis.values) {
      final endpoint = geometry.axisEndpoints[axis];
      if (endpoint == null) {
        continue;
      }

      final strokePaint = Paint()
        ..color = _axisColor(axis)
        ..style = PaintingStyle.stroke
        ..strokeWidth = _axisStrokeWidth;
      canvas.drawLine(geometry.originScreen, endpoint, strokePaint);

      final fillPaint = Paint()
        ..color = _axisColor(axis)
        ..style = PaintingStyle.fill;
      canvas.drawRect(
        Rect.fromCenter(
          center: endpoint,
          width: _scaleBoxSize,
          height: _scaleBoxSize,
        ),
        fillPaint,
      );
    }
  }

  void _paintRotateGizmo(Canvas canvas) {
    for (final axis in ViewportGizmoAxis.values) {
      final segments = geometry.rotationRingSegments[axis];
      if (segments == null || segments.isEmpty) {
        continue;
      }

      final ringPaint = Paint()
        ..color = _axisColor(axis)
        ..style = PaintingStyle.stroke
        ..strokeWidth = _axisStrokeWidth;

      for (final segment in segments) {
        canvas.drawLine(segment.start, segment.end, ringPaint);
      }
    }
  }

  Color _axisColor(ViewportGizmoAxis axis) {
    final highlighted = activeAxis == axis || hoveredAxis == axis;
    return _applyEnabledOpacity(
      highlighted ? axis.highlightColor : axis.baseColor,
    );
  }

  Color _applyEnabledOpacity(Color color) {
    if (enabled) {
      return color;
    }
    final alpha = (color.a * 255.0 * 0.45).round().clamp(0, 255);
    return color.withAlpha(alpha);
  }
}
