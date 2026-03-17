import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';

const double _iconHitRadius = 16.0;
const double _iconSizeMin = 12.0;
const double _iconSizeMax = 48.0;
const double _fadeStartDistance = 50.0;
const int _circleSegments = 48;
const double _wireframeStrokeWidth = 1.5;

BigInt? hitTestViewportLightBillboardTransformId(
  Offset pointerPosition,
  Size viewportSize, {
  required AppSceneSnapshot? snapshot,
  required TextureViewportFeedback? feedback,
}) {
  final geometry = _computeViewportLightGeometry(
    snapshot,
    feedback,
    viewportSize,
  );
  if (geometry == null) {
    return null;
  }
  return _hitTestBillboard(pointerPosition, geometry)?.light.transformNodeId;
}

Offset? debugViewportLightBillboardPosition(
  Size viewportSize, {
  required AppSceneSnapshot? snapshot,
  required TextureViewportFeedback? feedback,
  BigInt? nodeId,
}) {
  final geometry = _computeViewportLightGeometry(
    snapshot,
    feedback,
    viewportSize,
  );
  if (geometry == null || geometry.billboards.isEmpty) {
    return null;
  }

  if (nodeId == null) {
    return geometry.billboards.first.screenPosition;
  }

  for (final billboard in geometry.billboards) {
    if (billboard.light.lightNodeId == nodeId ||
        billboard.light.transformNodeId == nodeId) {
      return billboard.screenPosition;
    }
  }

  return null;
}

class ViewportLightGizmoOverlay extends StatelessWidget {
  const ViewportLightGizmoOverlay({
    super.key,
    required this.snapshot,
    required this.feedback,
  });

  final AppSceneSnapshot? snapshot;
  final TextureViewportFeedback? feedback;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final geometry = _computeViewportLightGeometry(
          snapshot,
          feedback,
          Size(constraints.maxWidth, constraints.maxHeight),
        );
        if (geometry == null) {
          return const SizedBox.expand();
        }

        return CustomPaint(
          painter: _ViewportLightGizmoPainter(
            geometry: geometry,
            shellPalette: context.shellPalette,
          ),
          child: const SizedBox.expand(),
        );
      },
    );
  }
}

class _ViewportLightGizmoSceneState {
  const _ViewportLightGizmoSceneState({
    required this.camera,
    required this.selectedNodeId,
    required this.viewportLights,
  });

  final AppCameraSnapshot camera;
  final BigInt? selectedNodeId;
  final List<AppViewportLightSnapshot> viewportLights;
}

class _ViewportLightBillboardGeometry {
  const _ViewportLightBillboardGeometry({
    required this.light,
    required this.screenPosition,
    required this.distanceToCamera,
    required this.iconSize,
    required this.iconAlpha,
    required this.selected,
  });

  final AppViewportLightSnapshot light;
  final Offset screenPosition;
  final double distanceToCamera;
  final double iconSize;
  final double iconAlpha;
  final bool selected;
}

class _ViewportLightGizmoGeometry {
  const _ViewportLightGizmoGeometry({
    required this.scene,
    required this.viewportSize,
    required this.billboards,
  });

  final _ViewportLightGizmoSceneState scene;
  final Size viewportSize;
  final List<_ViewportLightBillboardGeometry> billboards;
}

class _ViewportLightGizmoPainter extends CustomPainter {
  const _ViewportLightGizmoPainter({
    required this.geometry,
    required this.shellPalette,
  });

  final _ViewportLightGizmoGeometry geometry;
  final ShellPalette shellPalette;

  @override
  void paint(Canvas canvas, Size size) {
    for (final billboard in geometry.billboards) {
      final drawColor = _resolvedBillboardColor(billboard);
      if (billboard.selected) {
        final ringPaint = Paint()
          ..color = shellPalette.selectionBorder.withValues(alpha: 0.95)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.6;
        canvas.drawCircle(
          billboard.screenPosition,
          billboard.iconSize * 0.9,
          ringPaint,
        );
      }

      _paintBillboardIcon(canvas, billboard, drawColor);

      if (billboard.light.intensity < 0.0) {
        _paintNegativeLightIndicator(canvas, billboard);
      }
      if (!billboard.light.active) {
        _paintInactiveLightIndicator(canvas, billboard);
      }
      if (billboard.selected) {
        _paintSelectedWireframe(canvas, billboard);
      }
    }
  }

  @override
  bool shouldRepaint(_ViewportLightGizmoPainter oldDelegate) {
    return oldDelegate.geometry != geometry ||
        oldDelegate.shellPalette != shellPalette;
  }

  void _paintBillboardIcon(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
    Color drawColor,
  ) {
    switch (billboard.light.lightTypeId) {
      case 'spot':
        _paintSpotLightIcon(canvas, billboard, drawColor);
        return;
      case 'directional':
        _paintDirectionalLightIcon(canvas, billboard, drawColor);
        return;
      case 'ambient':
        _paintAmbientLightIcon(canvas, billboard, drawColor);
        return;
      case 'array':
        _paintArrayLightIcon(canvas, billboard, drawColor);
        return;
      default:
        _paintPointLightIcon(
          canvas,
          billboard.screenPosition,
          billboard.iconSize,
          drawColor,
        );
        return;
    }
  }

  void _paintPointLightIcon(
    Canvas canvas,
    Offset center,
    double iconSize,
    Color color,
  ) {
    final coreRadius = iconSize * 0.3;
    final rayInner = iconSize * 0.4;
    final rayOuter = iconSize * 0.7;
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    canvas.drawCircle(center, coreRadius, fillPaint);

    for (var index = 0; index < 8; index += 1) {
      final angle = index / 8.0 * math.pi * 2.0;
      final direction = Offset(math.cos(angle), math.sin(angle));
      canvas.drawLine(
        center + direction * rayInner,
        center + direction * rayOuter,
        strokePaint,
      );
    }
  }

  void _paintSpotLightIcon(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
    Color color,
  ) {
    final half = billboard.iconSize * 0.5;
    final center = billboard.screenPosition;
    final path = Path()
      ..moveTo(center.dx - half * 0.4, center.dy - half * 0.5)
      ..lineTo(center.dx + half * 0.4, center.dy - half * 0.5)
      ..lineTo(center.dx, center.dy + half * 0.7)
      ..close();
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawPath(path, fillPaint);
    canvas.drawCircle(
      center + Offset(0.0, -half * 0.3),
      billboard.iconSize * 0.15,
      fillPaint,
    );
  }

  void _paintDirectionalLightIcon(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
    Color color,
  ) {
    final center = billboard.screenPosition;
    final coreRadius = billboard.iconSize * 0.25;
    final rayInner = billboard.iconSize * 0.35;
    final rayOuter = billboard.iconSize * 0.65;
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    canvas.drawCircle(center, coreRadius, fillPaint);
    canvas.drawCircle(center, coreRadius, strokePaint);

    for (var index = 0; index < 12; index += 1) {
      final angle = index / 12.0 * math.pi * 2.0;
      final direction = Offset(math.cos(angle), math.sin(angle));
      canvas.drawLine(
        center + direction * rayInner,
        center + direction * rayOuter,
        strokePaint,
      );
    }
  }

  void _paintAmbientLightIcon(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
    Color color,
  ) {
    final center = billboard.screenPosition;
    final ringPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, billboard.iconSize * 0.2, ringPaint);
    canvas.drawCircle(center, billboard.iconSize * 0.4, ringPaint);
    canvas.drawCircle(center, billboard.iconSize * 0.6, ringPaint);
    canvas.drawCircle(center, billboard.iconSize * 0.08, fillPaint);
  }

  void _paintArrayLightIcon(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
    Color color,
  ) {
    final center = billboard.screenPosition;
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    final dotRadius = billboard.iconSize * 0.15;
    final spacing = billboard.iconSize * 0.28;

    canvas.drawCircle(center + Offset(0.0, -spacing), dotRadius, fillPaint);
    canvas.drawCircle(center, dotRadius, fillPaint);
    canvas.drawCircle(center + Offset(0.0, spacing), dotRadius, fillPaint);

    final dashRadius = billboard.iconSize * 0.55;
    for (var index = 0; index < 12; index += 1) {
      final startAngle = index / 12.0 * math.pi * 2.0;
      final endAngle = (index + 0.5) / 12.0 * math.pi * 2.0;
      final start = center +
          Offset(math.cos(startAngle), math.sin(startAngle)) * dashRadius;
      final end =
          center + Offset(math.cos(endAngle), math.sin(endAngle)) * dashRadius;
      canvas.drawLine(start, end, strokePaint);
    }

    final instanceSize = billboard.iconSize * 0.6;
    for (var index = 0; index < billboard.light.arrayPositions.length; index += 1) {
      final instancePosition = _projectWorldToScreen(
        _Vec3.fromAppVec3(billboard.light.arrayPositions[index]),
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (instancePosition == null ||
          !_viewportContains(geometry.viewportSize, instancePosition)) {
        continue;
      }
      final instanceColor = _resolvedArrayInstanceColor(
        billboard,
        index,
        fallback: color,
      );
      _paintPointLightIcon(canvas, instancePosition, instanceSize, instanceColor);
    }
  }

  void _paintNegativeLightIndicator(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
  ) {
    final minusHalf = billboard.iconSize * 0.35;
    final strokePaint = Paint()
      ..color = shellPalette.dangerAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawLine(
      billboard.screenPosition + Offset(-minusHalf, 0.0),
      billboard.screenPosition + Offset(minusHalf, 0.0),
      strokePaint,
    );
  }

  void _paintInactiveLightIndicator(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
  ) {
    final size = billboard.iconSize * 0.3;
    final strokePaint = Paint()
      ..color = shellPalette.dangerAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawLine(
      billboard.screenPosition + Offset(-size, -size),
      billboard.screenPosition + Offset(size, size),
      strokePaint,
    );
    canvas.drawLine(
      billboard.screenPosition + Offset(-size, size),
      billboard.screenPosition + Offset(size, -size),
      strokePaint,
    );
  }

  void _paintSelectedWireframe(
    Canvas canvas,
    _ViewportLightBillboardGeometry billboard,
  ) {
    final selectionColor = shellPalette.warningAccent;
    switch (billboard.light.lightTypeId) {
      case 'point':
        _paintPointLightWireframe(canvas, billboard.light, selectionColor);
        return;
      case 'spot':
        _paintSpotLightWireframe(canvas, billboard.light, selectionColor);
        return;
      case 'directional':
        _paintDirectionalLightWireframe(canvas, billboard.light, selectionColor);
        return;
      case 'ambient':
        final strokePaint = Paint()
          ..color = selectionColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = _wireframeStrokeWidth;
        canvas.drawCircle(
          billboard.screenPosition,
          billboard.iconSize * 0.8,
          strokePaint,
        );
        return;
      case 'array':
        _paintArrayWireframe(canvas, billboard.light, selectionColor);
        return;
    }
  }

  void _paintPointLightWireframe(
    Canvas canvas,
    AppViewportLightSnapshot light,
    Color color,
  ) {
    final strokePaint = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _wireframeStrokeWidth;
    _paintWireframeCircle(
      canvas,
      center: light.worldPosition,
      axisDirection: const AppVec3(x: 1.0, y: 0.0, z: 0.0),
      radius: light.range,
      paint: strokePaint,
    );
    _paintWireframeCircle(
      canvas,
      center: light.worldPosition,
      axisDirection: const AppVec3(x: 0.0, y: 1.0, z: 0.0),
      radius: light.range,
      paint: strokePaint,
    );
    _paintWireframeCircle(
      canvas,
      center: light.worldPosition,
      axisDirection: const AppVec3(x: 0.0, y: 0.0, z: 1.0),
      radius: light.range,
      paint: strokePaint,
    );
  }

  void _paintSpotLightWireframe(
    Canvas canvas,
    AppViewportLightSnapshot light,
    Color color,
  ) {
    final direction = _Vec3.fromAppVec3(light.direction).normalized();
    final center = _Vec3.fromAppVec3(light.worldPosition);
    final baseCenter = center + direction * light.range;
    final outerHalfAngle = light.spotAngle * 0.5 * math.pi / 180.0;
    final outerRadius = light.range * math.tan(outerHalfAngle);
    final innerHalfAngle = light.spotAngle * 0.5 * 0.8 * math.pi / 180.0;
    final innerRadius = light.range * math.tan(innerHalfAngle);
    final outerPaint = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _wireframeStrokeWidth;
    final innerPaint = Paint()
      ..color = color.withValues(alpha: 0.25)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _wireframeStrokeWidth * 0.7;

    _paintWireframeCircle(
      canvas,
      center: baseCenter.toAppVec3(),
      axisDirection: direction.toAppVec3(),
      radius: outerRadius,
      paint: outerPaint,
    );
    _paintWireframeCircle(
      canvas,
      center: baseCenter.toAppVec3(),
      axisDirection: direction.toAppVec3(),
      radius: innerRadius,
      paint: innerPaint,
    );

    final basis = _orthonormalBasis(direction);
    final tipScreen = _projectWorldToScreen(center, geometry.scene.camera, geometry.viewportSize);
    if (tipScreen == null) {
      return;
    }

    for (var index = 0; index < 8; index += 1) {
      final angle = index / 8.0 * math.pi * 2.0;
      final edgeDirection =
          basis.$1 * math.cos(angle) + basis.$2 * math.sin(angle);
      final outerEdge = baseCenter + edgeDirection * outerRadius;
      final outerEdgeScreen = _projectWorldToScreen(
        outerEdge,
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (outerEdgeScreen != null) {
        canvas.drawLine(tipScreen, outerEdgeScreen, outerPaint);
      }

      if (index.isOdd) {
        continue;
      }
      final innerEdge = baseCenter + edgeDirection * innerRadius;
      final innerEdgeScreen = _projectWorldToScreen(
        innerEdge,
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (innerEdgeScreen != null) {
        canvas.drawLine(tipScreen, innerEdgeScreen, innerPaint);
      }
    }
  }

  void _paintDirectionalLightWireframe(
    Canvas canvas,
    AppViewportLightSnapshot light,
    Color color,
  ) {
    final direction = _Vec3.fromAppVec3(light.direction).normalized();
    final center = _Vec3.fromAppVec3(light.worldPosition);
    final basis = _orthonormalBasis(direction);
    final offsets = <_Vec3>[
      const _Vec3.zero(),
      basis.$1 * 0.8,
      basis.$1 * -0.8,
      basis.$2 * 0.8,
      basis.$2 * -0.8,
    ];
    final linePaint = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _wireframeStrokeWidth;
    final arrowPaint = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.fill;

    for (final offset in offsets) {
      final start = center + offset;
      final end = start + direction * 5.0;
      final startScreen = _projectWorldToScreen(
        start,
        geometry.scene.camera,
        geometry.viewportSize,
      );
      final endScreen = _projectWorldToScreen(
        end,
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (startScreen == null || endScreen == null) {
        continue;
      }

      canvas.drawLine(startScreen, endScreen, linePaint);
      final direction2d = endScreen - startScreen;
      final directionLength = direction2d.distance;
      if (directionLength <= 5.0) {
        continue;
      }

      final normalized = direction2d / directionLength;
      final perpendicular = Offset(-normalized.dy, normalized.dx);
      final path = Path()
        ..moveTo(endScreen.dx, endScreen.dy)
        ..lineTo(
          endScreen.dx - normalized.dx * 5.0 + perpendicular.dx * 2.0,
          endScreen.dy - normalized.dy * 5.0 + perpendicular.dy * 2.0,
        )
        ..lineTo(
          endScreen.dx - normalized.dx * 5.0 - perpendicular.dx * 2.0,
          endScreen.dy - normalized.dy * 5.0 - perpendicular.dy * 2.0,
        )
        ..close();
      canvas.drawPath(path, arrowPaint);
    }
  }

  void _paintArrayWireframe(
    Canvas canvas,
    AppViewportLightSnapshot light,
    Color color,
  ) {
    if (light.arrayPositions.length < 2) {
      return;
    }

    final strokePaint = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = _wireframeStrokeWidth;

    for (var index = 0; index < light.arrayPositions.length; index += 1) {
      final nextIndex = (index + 1) % light.arrayPositions.length;
      final start = _projectWorldToScreen(
        _Vec3.fromAppVec3(light.arrayPositions[index]),
        geometry.scene.camera,
        geometry.viewportSize,
      );
      final end = _projectWorldToScreen(
        _Vec3.fromAppVec3(light.arrayPositions[nextIndex]),
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (start == null || end == null) {
        continue;
      }
      _paintDashedLine(
        canvas,
        start,
        end,
        strokePaint,
        dashLength: 6.0,
        gapLength: 4.0,
      );
    }
  }

  void _paintWireframeCircle(
    Canvas canvas, {
    required AppVec3 center,
    required AppVec3 axisDirection,
    required double radius,
    required Paint paint,
  }) {
    if (radius <= 0.0) {
      return;
    }

    final basis = _orthonormalBasis(_Vec3.fromAppVec3(axisDirection).normalized());
    final worldCenter = _Vec3.fromAppVec3(center);
    Offset? previousPoint;
    for (var index = 0; index <= _circleSegments; index += 1) {
      final angle = index / _circleSegments * math.pi * 2.0;
      final worldPoint =
          worldCenter +
          (basis.$1 * math.cos(angle) + basis.$2 * math.sin(angle)) * radius;
      final screenPoint = _projectWorldToScreen(
        worldPoint,
        geometry.scene.camera,
        geometry.viewportSize,
      );
      if (previousPoint != null && screenPoint != null) {
        canvas.drawLine(previousPoint, screenPoint, paint);
      }
      previousPoint = screenPoint;
    }
  }

  void _paintDashedLine(
    Canvas canvas,
    Offset start,
    Offset end,
    Paint paint, {
    required double dashLength,
    required double gapLength,
  }) {
    final direction = end - start;
    final lineLength = direction.distance;
    if (lineLength < 1.0) {
      return;
    }

    final normalized = direction / lineLength;
    var distance = 0.0;
    while (distance < lineLength) {
      final segmentEnd = math.min(distance + dashLength, lineLength);
      canvas.drawLine(
        start + normalized * distance,
        start + normalized * segmentEnd,
        paint,
      );
      distance = segmentEnd + gapLength;
    }
  }

  Color _resolvedBillboardColor(_ViewportLightBillboardGeometry billboard) {
    if (!billboard.light.active) {
      return Color.fromRGBO(120, 120, 120, billboard.iconAlpha * 0.5);
    }

    final colorMagnitude = _Vec3.fromAppVec3(billboard.light.color).length;
    if (colorMagnitude < 0.3) {
      return Color.fromRGBO(200, 200, 200, billboard.iconAlpha);
    }

    return _lightColor(billboard.light.color, billboard.iconAlpha);
  }

  Color _resolvedArrayInstanceColor(
    _ViewportLightBillboardGeometry billboard,
    int index, {
    required Color fallback,
  }) {
    if (index >= billboard.light.arrayColors.length) {
      return fallback;
    }
    return _lightColor(billboard.light.arrayColors[index], billboard.iconAlpha);
  }

  Color _lightColor(AppVec3 color, double alpha) {
    return Color.fromRGBO(
      (color.x.clamp(0.0, 1.0) * 255.0).round(),
      (color.y.clamp(0.0, 1.0) * 255.0).round(),
      (color.z.clamp(0.0, 1.0) * 255.0).round(),
      alpha.clamp(0.0, 1.0),
    );
  }
}

_ViewportLightGizmoGeometry? _computeViewportLightGeometry(
  AppSceneSnapshot? snapshot,
  TextureViewportFeedback? feedback,
  Size viewportSize,
) {
  final scene = _buildViewportLightScene(snapshot, feedback);
  if (scene == null ||
      viewportSize.width <= 0 ||
      viewportSize.height <= 0) {
    return null;
  }

  final cameraEye = _Vec3.fromAppVec3(scene.camera.eye);
  final billboards = <_ViewportLightBillboardGeometry>[];
  for (final light in scene.viewportLights) {
    final worldPosition = _Vec3.fromAppVec3(light.worldPosition);
    final screenPosition =
        _projectWorldToScreen(worldPosition, scene.camera, viewportSize);
    if (screenPosition == null || !_viewportContains(viewportSize, screenPosition)) {
      continue;
    }

    final distance = (worldPosition - cameraEye).length;
    billboards.add(
      _ViewportLightBillboardGeometry(
        light: light,
        screenPosition: screenPosition,
        distanceToCamera: distance,
        iconSize: _iconSize(distance),
        iconAlpha: _iconAlpha(distance),
        selected:
            scene.selectedNodeId == light.lightNodeId ||
            scene.selectedNodeId == light.transformNodeId,
      ),
    );
  }

  if (billboards.isEmpty) {
    return null;
  }

  return _ViewportLightGizmoGeometry(
    scene: scene,
    viewportSize: viewportSize,
    billboards: billboards,
  );
}

_ViewportLightGizmoSceneState? _buildViewportLightScene(
  AppSceneSnapshot? snapshot,
  TextureViewportFeedback? feedback,
) {
  if (snapshot == null ||
      !snapshot.settings.showLightGizmos ||
      snapshot.viewportLights.isEmpty) {
    return null;
  }

  return _ViewportLightGizmoSceneState(
    camera: feedback?.camera ?? snapshot.camera,
    selectedNodeId: snapshot.selectedNode?.id,
    viewportLights: snapshot.viewportLights,
  );
}

_ViewportLightBillboardGeometry? _hitTestBillboard(
  Offset pointerPosition,
  _ViewportLightGizmoGeometry geometry,
) {
  _ViewportLightBillboardGeometry? closest;
  double? closestDistance;

  for (final billboard in geometry.billboards) {
    final distance = (pointerPosition - billboard.screenPosition).distance;
    if (distance > math.max(_iconHitRadius, billboard.iconSize * 0.5)) {
      continue;
    }
    if (closestDistance == null || distance < closestDistance) {
      closestDistance = distance;
      closest = billboard;
    }
  }

  return closest;
}

double _iconSize(double distance) {
  return (30.0 / (distance * 0.15 + 1.0)).clamp(_iconSizeMin, _iconSizeMax);
}

double _iconAlpha(double distance) {
  if (distance > _fadeStartDistance) {
    return ((60.0 - distance) / 10.0).clamp(0.1, 1.0);
  }
  return 1.0;
}

bool _viewportContains(Size viewportSize, Offset position) {
  return position.dx >= 0.0 &&
      position.dy >= 0.0 &&
      position.dx <= viewportSize.width &&
      position.dy <= viewportSize.height;
}

(_Vec3, _Vec3) _orthonormalBasis(_Vec3 axisDirection) {
  final up = axisDirection.y.abs() > 0.99
      ? const _Vec3(1.0, 0.0, 0.0)
      : const _Vec3(0.0, 1.0, 0.0);
  final tangent = axisDirection.cross(up).normalized();
  final bitangent = axisDirection.cross(tangent).normalized();
  return (tangent, bitangent);
}

Offset? _projectWorldToScreen(
  _Vec3 worldPoint,
  AppCameraSnapshot camera,
  Size viewportSize,
) {
  final eye = _Vec3.fromAppVec3(camera.eye);
  final target = _Vec3.fromAppVec3(camera.target);
  final forward = (target - eye).normalized();
  final up = math.cos(camera.pitch) >= 0.0
      ? const _Vec3(0.0, 1.0, 0.0)
      : const _Vec3(0.0, -1.0, 0.0);
  final right = forward.cross(up).normalized();
  final trueUp = right.cross(forward).normalized();
  final localPoint = worldPoint - eye;

  var cameraX = localPoint.dot(right);
  var cameraY = localPoint.dot(trueUp);
  final cameraZ = -localPoint.dot(forward);
  final depth = -cameraZ;
  if (depth <= 1e-4) {
    return null;
  }

  if (camera.roll != 0.0) {
    final cosine = math.cos(camera.roll);
    final sine = math.sin(camera.roll);
    final rotatedX = cosine * cameraX - sine * cameraY;
    final rotatedY = sine * cameraX + cosine * cameraY;
    cameraX = rotatedX;
    cameraY = rotatedY;
  }

  final aspect =
      viewportSize.width / viewportSize.height.clamp(1.0, double.infinity);
  late final double ndcX;
  late final double ndcY;

  if (camera.orthographic) {
    final halfHeight =
        camera.distance * math.tan(camera.fovDegrees * math.pi / 180.0 * 0.5);
    final halfWidth = halfHeight * aspect;
    if (halfWidth.abs() < 1e-6 || halfHeight.abs() < 1e-6) {
      return null;
    }
    ndcX = cameraX / halfWidth;
    ndcY = cameraY / halfHeight;
  } else {
    final tanHalfFov = math.tan(camera.fovDegrees * math.pi / 180.0 * 0.5);
    if (tanHalfFov.abs() < 1e-6) {
      return null;
    }
    ndcX = cameraX / (depth * tanHalfFov * aspect);
    ndcY = cameraY / (depth * tanHalfFov);
  }

  return Offset(
    (ndcX * 0.5 + 0.5) * viewportSize.width,
    (-ndcY * 0.5 + 0.5) * viewportSize.height,
  );
}

class _Vec3 {
  const _Vec3(this.x, this.y, this.z);

  const _Vec3.zero() : this(0.0, 0.0, 0.0);

  final double x;
  final double y;
  final double z;

  factory _Vec3.fromAppVec3(AppVec3 value) {
    return _Vec3(value.x, value.y, value.z);
  }

  AppVec3 toAppVec3() => AppVec3(x: x, y: y, z: z);

  _Vec3 operator +(_Vec3 other) => _Vec3(x + other.x, y + other.y, z + other.z);

  _Vec3 operator -(_Vec3 other) => _Vec3(x - other.x, y - other.y, z - other.z);

  _Vec3 operator *(double factor) => _Vec3(x * factor, y * factor, z * factor);

  double get length => math.sqrt(lengthSquared);

  double get lengthSquared => x * x + y * y + z * z;

  double dot(_Vec3 other) => x * other.x + y * other.y + z * other.z;

  _Vec3 cross(_Vec3 other) => _Vec3(
    y * other.z - z * other.y,
    z * other.x - x * other.z,
    x * other.y - y * other.x,
  );

  _Vec3 normalized() {
    final currentLength = length;
    if (currentLength <= 1e-6) {
      return const _Vec3.zero();
    }
    return _Vec3(x / currentLength, y / currentLength, z / currentLength);
  }
}
