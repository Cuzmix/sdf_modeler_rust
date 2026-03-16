import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';

const double viewportGizmoAxisLength = 1.2;
const double viewportGizmoHitThreshold = 12.0;
const double viewportGizmoRingRadius = 1.0;
const int viewportGizmoRingSegments = 48;
const double viewportGizmoRingHitThreshold = 14.0;
const double viewportGizmoTranslateSensitivity = 0.003;
const double viewportGizmoScaleSensitivity = 0.005;
const double viewportGizmoRotateSensitivity = 0.01;
const double viewportGizmoScaleMin = 0.01;
const double viewportGizmoScaleMax = 100.0;

enum ViewportGizmoAxis {
  x(
    axisId: 'x',
    baseColor: Color(0xFFDC3C3C),
    highlightColor: Color(0xFFFF7878),
    unitVector: AppVec3(x: 1, y: 0, z: 0),
  ),
  y(
    axisId: 'y',
    baseColor: Color(0xFF3CC83C),
    highlightColor: Color(0xFF78FF78),
    unitVector: AppVec3(x: 0, y: 1, z: 0),
  ),
  z(
    axisId: 'z',
    baseColor: Color(0xFF3C64F0),
    highlightColor: Color(0xFF78A0FF),
    unitVector: AppVec3(x: 0, y: 0, z: 1),
  );

  const ViewportGizmoAxis({
    required this.axisId,
    required this.baseColor,
    required this.highlightColor,
    required this.unitVector,
  });

  final String axisId;
  final Color baseColor;
  final Color highlightColor;
  final AppVec3 unitVector;

  static ViewportGizmoAxis? fromAxisId(String axisId) {
    for (final axis in ViewportGizmoAxis.values) {
      if (axis.axisId == axisId) {
        return axis;
      }
    }
    return null;
  }
}

class ViewportGizmoTransformData {
  const ViewportGizmoTransformData({
    required this.position,
    required this.rotationDegrees,
    this.scale,
  });

  final AppVec3 position;
  final AppVec3 rotationDegrees;
  final AppVec3? scale;

  bool get hasScale => scale != null;

  ViewportGizmoTransformData copyWith({
    AppVec3? position,
    AppVec3? rotationDegrees,
    AppVec3? scale,
    bool clearScale = false,
  }) {
    return ViewportGizmoTransformData(
      position: position ?? this.position,
      rotationDegrees: rotationDegrees ?? this.rotationDegrees,
      scale: clearScale ? null : (scale ?? this.scale),
    );
  }

  static ViewportGizmoTransformData? fromSceneSnapshot(
    AppSceneSnapshot? snapshot,
  ) {
    final transform = snapshot?.selectedNodeProperties?.transform;
    if (transform == null) {
      return null;
    }
    return ViewportGizmoTransformData(
      position: transform.position,
      rotationDegrees: transform.rotationDegrees,
      scale: transform.scale,
    );
  }
}

class ViewportGizmoSceneState {
  const ViewportGizmoSceneState({
    required this.selectedNodeId,
    required this.camera,
    required this.tool,
    required this.transform,
  });

  final BigInt selectedNodeId;
  final AppCameraSnapshot camera;
  final AppToolSnapshot tool;
  final ViewportGizmoTransformData transform;

  bool get showsScaleGizmo =>
      tool.manipulatorModeId != 'scale' || transform.hasScale;
}

class ViewportGizmoScreenSegment {
  const ViewportGizmoScreenSegment({
    required this.start,
    required this.end,
  });

  final Offset start;
  final Offset end;
}

class ViewportGizmoGeometry {
  const ViewportGizmoGeometry({
    required this.scene,
    required this.originScreen,
    required this.nodeOriginScreen,
    required this.gizmoCenterWorld,
    required this.viewDirection,
    required this.axisDirections,
    required this.axisEndpoints,
    required this.screenAxisDirections,
    required this.rotationRingSegments,
  });

  final ViewportGizmoSceneState scene;
  final Offset originScreen;
  final Offset? nodeOriginScreen;
  final AppVec3 gizmoCenterWorld;
  final AppVec3 viewDirection;
  final Map<ViewportGizmoAxis, AppVec3> axisDirections;
  final Map<ViewportGizmoAxis, Offset> axisEndpoints;
  final Map<ViewportGizmoAxis, Offset> screenAxisDirections;
  final Map<ViewportGizmoAxis, List<ViewportGizmoScreenSegment>>
  rotationRingSegments;
}

class ViewportGizmoDragSession {
  ViewportGizmoDragSession({
    required this.selectedNodeId,
    required this.modeId,
    required this.spaceId,
    required this.axis,
    required this.startPointerPosition,
    required this.originScreen,
    required this.screenAxisDirection,
    required this.axisDirection,
    required this.viewDirection,
    required this.cameraDistance,
    required this.startTransform,
    required this.pivotOffset,
  }) : _startRotationRadians = _degreesToRadians(startTransform.rotationDegrees),
       _startPosition = _Vec3.fromAppVec3(startTransform.position),
       _startScale = startTransform.scale == null
           ? null
           : _Vec3.fromAppVec3(startTransform.scale!),
       _pivotWorld =
           _Vec3.fromAppVec3(startTransform.position) +
           _inverseRotateEuler(
             _Vec3.fromAppVec3(pivotOffset),
             _degreesToRadians(startTransform.rotationDegrees),
           );

  final BigInt selectedNodeId;
  final String modeId;
  final String spaceId;
  final ViewportGizmoAxis axis;
  final Offset startPointerPosition;
  final Offset originScreen;
  final Offset screenAxisDirection;
  final AppVec3 axisDirection;
  final AppVec3 viewDirection;
  final double cameraDistance;
  final ViewportGizmoTransformData startTransform;
  final AppVec3 pivotOffset;

  final _Vec3 _startRotationRadians;
  final _Vec3 _startPosition;
  final _Vec3? _startScale;
  final _Vec3 _pivotWorld;

  ViewportGizmoTransformData evaluateTransform(Offset pointerPosition) {
    return switch (modeId) {
      'rotate' => _evaluateRotate(pointerPosition),
      'scale' => _evaluateScale(pointerPosition),
      _ => _evaluateTranslate(pointerPosition),
    };
  }

  ViewportGizmoTransformData _evaluateTranslate(Offset pointerPosition) {
    final projectedDrag =
        (pointerPosition - startPointerPosition).dx * screenAxisDirection.dx +
        (pointerPosition - startPointerPosition).dy * screenAxisDirection.dy;
    final axisDirection3 = _Vec3.fromAppVec3(axisDirection);
    final worldDelta =
        axisDirection3 * projectedDrag * cameraDistance * viewportGizmoTranslateSensitivity;
    final nextPosition = (_startPosition + worldDelta).toAppVec3();
    return startTransform.copyWith(position: nextPosition);
  }

  ViewportGizmoTransformData _evaluateScale(Offset pointerPosition) {
    final startScale = _startScale;
    if (startScale == null) {
      return startTransform;
    }

    final projectedDrag =
        (pointerPosition - startPointerPosition).dx * screenAxisDirection.dx +
        (pointerPosition - startPointerPosition).dy * screenAxisDirection.dy;
    final rawFactor = 1.0 + projectedDrag * viewportGizmoScaleSensitivity;

    final nextScale = _Vec3(
      axis == ViewportGizmoAxis.x
          ? (startScale.x * rawFactor).clamp(
              viewportGizmoScaleMin,
              viewportGizmoScaleMax,
            )
          : startScale.x,
      axis == ViewportGizmoAxis.y
          ? (startScale.y * rawFactor).clamp(
              viewportGizmoScaleMin,
              viewportGizmoScaleMax,
            )
          : startScale.y,
      axis == ViewportGizmoAxis.z
          ? (startScale.z * rawFactor).clamp(
              viewportGizmoScaleMin,
              viewportGizmoScaleMax,
            )
          : startScale.z,
    );

    _Vec3 nextPosition = _startPosition;
    if (_pivotOffsetHasMagnitude) {
      final scaleFactor = _Vec3(
        axis == ViewportGizmoAxis.x && startScale.x.abs() > 1e-6
            ? nextScale.x / startScale.x
            : 1.0,
        axis == ViewportGizmoAxis.y && startScale.y.abs() > 1e-6
            ? nextScale.y / startScale.y
            : 1.0,
        axis == ViewportGizmoAxis.z && startScale.z.abs() > 1e-6
            ? nextScale.z / startScale.z
            : 1.0,
      );
      final offsetFromPivot = _startPosition - _pivotWorld;
      nextPosition = _pivotWorld + offsetFromPivot.scale(scaleFactor);
    }

    return startTransform.copyWith(
      position: nextPosition.toAppVec3(),
      scale: nextScale.toAppVec3(),
    );
  }

  ViewportGizmoTransformData _evaluateRotate(Offset pointerPosition) {
    final radiusVector = pointerPosition - originScreen;
    final radiusLength = radiusVector.distance;
    if (radiusLength < 1.0) {
      return startTransform;
    }

    final tangentDirection = Offset(
      -radiusVector.dy / radiusLength,
      radiusVector.dx / radiusLength,
    );
    final totalDrag = pointerPosition - startPointerPosition;
    final projectedDrag =
        totalDrag.dx * tangentDirection.dx + totalDrag.dy * tangentDirection.dy;
    final axisDirection3 = _Vec3.fromAppVec3(axisDirection).normalized();
    final viewDirection3 = _Vec3.fromAppVec3(viewDirection);
    final angleSign = axisDirection3.dot(viewDirection3) >= 0.0 ? 1.0 : -1.0;
    final angleDelta =
        projectedDrag * viewportGizmoRotateSensitivity * angleSign;
    final deltaRotation = _Quat.fromAxisAngle(axisDirection3, angleDelta);
    final nextRotationRadians = _applyRotationDelta(
      startRotationRadians: _startRotationRadians,
      deltaRotation: deltaRotation,
      spaceId: spaceId,
    );

    _Vec3 nextPosition = _startPosition;
    if (_pivotOffsetHasMagnitude) {
      final offsetFromPivot = _startPosition - _pivotWorld;
      nextPosition = _pivotWorld + deltaRotation.rotate(offsetFromPivot);
    }

    return startTransform.copyWith(
      position: nextPosition.toAppVec3(),
      rotationDegrees: _radiansToDegrees(nextRotationRadians),
    );
  }

  bool get _pivotOffsetHasMagnitude =>
      _Vec3.fromAppVec3(pivotOffset).lengthSquared > 1e-6;
}

ViewportGizmoSceneState? buildViewportGizmoSceneState(
  AppSceneSnapshot? snapshot,
  TextureViewportFeedback? feedback, {
  ViewportGizmoTransformData? transformOverride,
  AppToolSnapshot? toolOverride,
}) {
  final selectedNode = snapshot?.selectedNode;
  final tool = toolOverride ?? snapshot?.tool;
  final transform =
      transformOverride ?? ViewportGizmoTransformData.fromSceneSnapshot(snapshot);
  if (selectedNode == null || tool == null || transform == null) {
    return null;
  }
  if (!tool.manipulatorVisible) {
    return null;
  }

  return ViewportGizmoSceneState(
    selectedNodeId: selectedNode.id,
    camera: feedback?.camera ?? snapshot!.camera,
    tool: tool,
    transform: transform,
  );
}

ViewportGizmoGeometry? computeViewportGizmoGeometry(
  ViewportGizmoSceneState scene,
  Size viewportSize,
) {
  if (viewportSize.width <= 0 || viewportSize.height <= 0) {
    return null;
  }
  if (!scene.showsScaleGizmo) {
    return null;
  }

  final transform = scene.transform;
  final rotationRadians = _degreesToRadians(transform.rotationDegrees);
  final pivotOffset = _Vec3.fromAppVec3(scene.tool.pivotOffset);
  final gizmoCenter =
      _Vec3.fromAppVec3(transform.position) + _rotateEuler(pivotOffset, rotationRadians);
  final originScreen =
      _projectWorldToScreen(gizmoCenter, scene.camera, viewportSize);
  if (originScreen == null) {
    return null;
  }

  final axisDirections = <ViewportGizmoAxis, AppVec3>{};
  final axisEndpoints = <ViewportGizmoAxis, Offset>{};
  final screenAxisDirections = <ViewportGizmoAxis, Offset>{};
  final rotationRingSegments =
      <ViewportGizmoAxis, List<ViewportGizmoScreenSegment>>{};

  for (final axis in ViewportGizmoAxis.values) {
    final axisDirection = _computeAxisDirection(axis, scene.tool, rotationRadians);
    axisDirections[axis] = axisDirection.toAppVec3();

    final endpointWorld = gizmoCenter + axisDirection * viewportGizmoAxisLength;
    final endpointScreen =
        _projectWorldToScreen(endpointWorld, scene.camera, viewportSize) ??
        originScreen;
    axisEndpoints[axis] = endpointScreen;

    final screenDirection = endpointScreen - originScreen;
    final screenLength = screenDirection.distance;
    screenAxisDirections[axis] = screenLength > 0.1
        ? screenDirection / screenLength
        : Offset.zero;

    rotationRingSegments[axis] = _buildRotationRingSegments(
      center: gizmoCenter,
      axisDirection: axisDirection,
      camera: scene.camera,
      viewportSize: viewportSize,
    );
  }

  return ViewportGizmoGeometry(
    scene: scene,
    originScreen: originScreen,
    nodeOriginScreen: _projectWorldToScreen(
      _Vec3.fromAppVec3(transform.position),
      scene.camera,
      viewportSize,
    ),
    gizmoCenterWorld: gizmoCenter.toAppVec3(),
    viewDirection:
        (_Vec3.fromAppVec3(scene.camera.eye) - gizmoCenter).toAppVec3(),
    axisDirections: axisDirections,
    axisEndpoints: axisEndpoints,
    screenAxisDirections: screenAxisDirections,
    rotationRingSegments: rotationRingSegments,
  );
}

ViewportGizmoAxis? hitTestViewportGizmoAxis(
  Offset pointerPosition,
  ViewportGizmoGeometry geometry, {
  required String modeId,
}) {
  if (modeId == 'rotate') {
    return _hitTestRotationRings(pointerPosition, geometry);
  }
  return _hitTestLinearAxes(pointerPosition, geometry);
}

ViewportGizmoDragSession beginViewportGizmoDrag(
  ViewportGizmoSceneState scene,
  ViewportGizmoGeometry geometry,
  ViewportGizmoAxis axis,
  Offset pointerPosition,
) {
  return ViewportGizmoDragSession(
    selectedNodeId: scene.selectedNodeId,
    modeId: scene.tool.manipulatorModeId,
    spaceId: scene.tool.manipulatorSpaceId,
    axis: axis,
    startPointerPosition: pointerPosition,
    originScreen: geometry.originScreen,
    screenAxisDirection: geometry.screenAxisDirections[axis] ?? Offset.zero,
    axisDirection: geometry.axisDirections[axis] ?? axis.unitVector,
    viewDirection: geometry.viewDirection,
    cameraDistance: scene.camera.distance,
    startTransform: scene.transform,
    pivotOffset: scene.tool.pivotOffset,
  );
}

ViewportGizmoAxis? _hitTestLinearAxes(
  Offset pointerPosition,
  ViewportGizmoGeometry geometry,
) {
  ViewportGizmoAxis? closestAxis;
  double? closestDistance;

  for (final axis in ViewportGizmoAxis.values) {
    final endpoint = geometry.axisEndpoints[axis];
    if (endpoint == null) {
      continue;
    }

    final distance = _pointToSegmentDistance(
      pointerPosition,
      geometry.originScreen,
      endpoint,
    );
    if (distance > viewportGizmoHitThreshold) {
      continue;
    }
    if (closestDistance == null || distance < closestDistance) {
      closestDistance = distance;
      closestAxis = axis;
    }
  }

  return closestAxis;
}

ViewportGizmoAxis? _hitTestRotationRings(
  Offset pointerPosition,
  ViewportGizmoGeometry geometry,
) {
  ViewportGizmoAxis? closestAxis;
  double? closestDistance;

  for (final axis in ViewportGizmoAxis.values) {
    final segments = geometry.rotationRingSegments[axis];
    if (segments == null || segments.isEmpty) {
      continue;
    }

    for (final segment in segments) {
      final distance = _pointToSegmentDistance(
        pointerPosition,
        segment.start,
        segment.end,
      );
      if (distance > viewportGizmoRingHitThreshold) {
        continue;
      }
      if (closestDistance == null || distance < closestDistance) {
        closestDistance = distance;
        closestAxis = axis;
      }
    }
  }

  return closestAxis;
}

List<ViewportGizmoScreenSegment> _buildRotationRingSegments({
  required _Vec3 center,
  required _Vec3 axisDirection,
  required AppCameraSnapshot camera,
  required Size viewportSize,
}) {
  final segments = <ViewportGizmoScreenSegment>[];
  final basis = _ringBasis(axisDirection);
  Offset? previousPoint;

  for (var index = 0; index <= viewportGizmoRingSegments; index += 1) {
    final angle = 2.0 * math.pi * index / viewportGizmoRingSegments;
    final ringWorldPoint =
        center +
        (basis.$1 * math.cos(angle) + basis.$2 * math.sin(angle)) *
            viewportGizmoRingRadius;
    final ringScreenPoint =
        _projectWorldToScreen(ringWorldPoint, camera, viewportSize);

    if (previousPoint != null && ringScreenPoint != null) {
      segments.add(
        ViewportGizmoScreenSegment(
          start: previousPoint,
          end: ringScreenPoint,
        ),
      );
    }
    previousPoint = ringScreenPoint;
  }

  return segments;
}

(_Vec3, _Vec3) _ringBasis(_Vec3 axisDirection) {
  final up = axisDirection.y.abs() > 0.99
      ? const _Vec3(1.0, 0.0, 0.0)
      : const _Vec3(0.0, 1.0, 0.0);
  final tangent = axisDirection.cross(up).normalized();
  final bitangent = axisDirection.cross(tangent).normalized();
  return (tangent, bitangent);
}

_Vec3 _computeAxisDirection(
  ViewportGizmoAxis axis,
  AppToolSnapshot tool,
  _Vec3 rotationRadians,
) {
  if (tool.manipulatorSpaceId == 'world') {
    return _Vec3.fromAppVec3(axis.unitVector);
  }
  return _inverseRotateEuler(_Vec3.fromAppVec3(axis.unitVector), rotationRadians);
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

  final aspect = viewportSize.width / viewportSize.height.clamp(1.0, double.infinity);
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
    final tanHalfFov =
        math.tan(camera.fovDegrees * math.pi / 180.0 * 0.5);
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

double _pointToSegmentDistance(Offset point, Offset start, Offset end) {
  final segment = end - start;
  final fromStart = point - start;
  final lengthSquared = segment.dx * segment.dx + segment.dy * segment.dy;
  if (lengthSquared < 1e-6) {
    return (point - start).distance;
  }

  final projected = ((fromStart.dx * segment.dx) + (fromStart.dy * segment.dy)) /
      lengthSquared;
  final t = projected.clamp(0.0, 1.0);
  final closest = Offset(
    start.dx + segment.dx * t,
    start.dy + segment.dy * t,
  );
  return (point - closest).distance;
}

_Vec3 _degreesToRadians(AppVec3 degrees) {
  return _Vec3(
    degrees.x * math.pi / 180.0,
    degrees.y * math.pi / 180.0,
    degrees.z * math.pi / 180.0,
  );
}

AppVec3 _radiansToDegrees(_Vec3 radians) {
  return AppVec3(
    x: radians.x * 180.0 / math.pi,
    y: radians.y * 180.0 / math.pi,
    z: radians.z * 180.0 / math.pi,
  );
}

_Vec3 _rotateEuler(_Vec3 point, _Vec3 rotation) {
  var rotated = point;
  final sinX = math.sin(rotation.x);
  final cosX = math.cos(rotation.x);
  rotated = _Vec3(
    rotated.x,
    cosX * rotated.y - sinX * rotated.z,
    sinX * rotated.y + cosX * rotated.z,
  );
  final sinY = math.sin(rotation.y);
  final cosY = math.cos(rotation.y);
  rotated = _Vec3(
    cosY * rotated.x + sinY * rotated.z,
    rotated.y,
    -sinY * rotated.x + cosY * rotated.z,
  );
  final sinZ = math.sin(rotation.z);
  final cosZ = math.cos(rotation.z);
  return _Vec3(
    cosZ * rotated.x - sinZ * rotated.y,
    sinZ * rotated.x + cosZ * rotated.y,
    rotated.z,
  );
}

_Vec3 _inverseRotateEuler(_Vec3 point, _Vec3 rotation) {
  var rotated = point;
  final sinZ = math.sin(rotation.z);
  final cosZ = math.cos(rotation.z);
  rotated = _Vec3(
    cosZ * rotated.x + sinZ * rotated.y,
    -sinZ * rotated.x + cosZ * rotated.y,
    rotated.z,
  );
  final sinY = math.sin(rotation.y);
  final cosY = math.cos(rotation.y);
  rotated = _Vec3(
    cosY * rotated.x - sinY * rotated.z,
    rotated.y,
    sinY * rotated.x + cosY * rotated.z,
  );
  final sinX = math.sin(rotation.x);
  final cosX = math.cos(rotation.x);
  return _Vec3(
    rotated.x,
    cosX * rotated.y + sinX * rotated.z,
    -sinX * rotated.y + cosX * rotated.z,
  );
}

_Vec3 _applyRotationDelta({
  required _Vec3 startRotationRadians,
  required _Quat deltaRotation,
  required String spaceId,
}) {
  final currentRotation = _Quat.fromEulerZyx(startRotationRadians);
  final nextRotation = switch (spaceId) {
    'world' => deltaRotation * currentRotation,
    _ => currentRotation * deltaRotation,
  };
  return _quatToEulerStable(nextRotation, startRotationRadians);
}

_Vec3 _quatToEulerStable(_Quat quaternion, _Vec3 previousRotation) {
  double wrapNear(double angle, double reference) {
    var wrappedAngle = angle;
    while (wrappedAngle - reference > math.pi) {
      wrappedAngle -= math.pi * 2.0;
    }
    while (wrappedAngle - reference < -math.pi) {
      wrappedAngle += math.pi * 2.0;
    }
    return wrappedAngle;
  }

  _Vec3 normalizeNear(_Vec3 value, _Vec3 previous) {
    return _Vec3(
      wrapNear(value.x, previous.x),
      wrapNear(value.y, previous.y),
      wrapNear(value.z, previous.z),
    );
  }

  final euler = quaternion.toEulerZyx();
  final optionA = normalizeNear(euler, previousRotation);
  final optionB = normalizeNear(
    _Vec3(
      euler.x + math.pi,
      math.pi - euler.y,
      euler.z + math.pi,
    ),
    previousRotation,
  );

  return (optionA - previousRotation).lengthSquared <=
          (optionB - previousRotation).lengthSquared
      ? optionA
      : optionB;
}

class _Vec3 {
  const _Vec3(this.x, this.y, this.z);

  const _Vec3.zero() : this(0.0, 0.0, 0.0);

  final double x;
  final double y;
  final double z;

  double get length => math.sqrt(lengthSquared);
  double get lengthSquared => x * x + y * y + z * z;

  _Vec3 normalized() {
    final valueLength = length;
    if (valueLength <= 1e-6) {
      return const _Vec3.zero();
    }
    return this / valueLength;
  }

  _Vec3 scale(_Vec3 factor) {
    return _Vec3(x * factor.x, y * factor.y, z * factor.z);
  }

  double dot(_Vec3 other) => x * other.x + y * other.y + z * other.z;

  _Vec3 cross(_Vec3 other) {
    return _Vec3(
      y * other.z - z * other.y,
      z * other.x - x * other.z,
      x * other.y - y * other.x,
    );
  }

  AppVec3 toAppVec3() => AppVec3(x: x, y: y, z: z);

  static _Vec3 fromAppVec3(AppVec3 value) => _Vec3(value.x, value.y, value.z);

  _Vec3 operator +(_Vec3 other) => _Vec3(x + other.x, y + other.y, z + other.z);
  _Vec3 operator -(_Vec3 other) => _Vec3(x - other.x, y - other.y, z - other.z);
  _Vec3 operator -() => _Vec3(-x, -y, -z);
  _Vec3 operator *(double scalar) => _Vec3(x * scalar, y * scalar, z * scalar);
  _Vec3 operator /(double scalar) => _Vec3(x / scalar, y / scalar, z / scalar);
}

class _Quat {
  const _Quat(this.x, this.y, this.z, this.w);

  final double x;
  final double y;
  final double z;
  final double w;

  factory _Quat.fromAxisAngle(_Vec3 axis, double angle) {
    final halfAngle = angle * 0.5;
    final sine = math.sin(halfAngle);
    final normalizedAxis = axis.normalized();
    return _Quat(
      normalizedAxis.x * sine,
      normalizedAxis.y * sine,
      normalizedAxis.z * sine,
      math.cos(halfAngle),
    );
  }

  factory _Quat.fromEulerZyx(_Vec3 rotationRadians) {
    final halfX = rotationRadians.x * 0.5;
    final halfY = rotationRadians.y * 0.5;
    final halfZ = rotationRadians.z * 0.5;

    final sinX = math.sin(halfX);
    final cosX = math.cos(halfX);
    final sinY = math.sin(halfY);
    final cosY = math.cos(halfY);
    final sinZ = math.sin(halfZ);
    final cosZ = math.cos(halfZ);

    return _Quat(
      sinX * cosY * cosZ - cosX * sinY * sinZ,
      cosX * sinY * cosZ + sinX * cosY * sinZ,
      cosX * cosY * sinZ - sinX * sinY * cosZ,
      cosX * cosY * cosZ + sinX * sinY * sinZ,
    );
  }

  _Quat operator *(_Quat other) {
    return _Quat(
      w * other.x + x * other.w + y * other.z - z * other.y,
      w * other.y - x * other.z + y * other.w + z * other.x,
      w * other.z + x * other.y - y * other.x + z * other.w,
      w * other.w - x * other.x - y * other.y - z * other.z,
    );
  }

  _Vec3 rotate(_Vec3 value) {
    final qVector = _Vec3(x, y, z);
    final uv = qVector.cross(value);
    final uuv = qVector.cross(uv);
    return value + uv * (2.0 * w) + uuv * 2.0;
  }

  _Vec3 toEulerZyx() {
    final sinXCosY = 2.0 * (w * x + y * z);
    final cosXCosY = 1.0 - 2.0 * (x * x + y * y);
    final rotationX = math.atan2(sinXCosY, cosXCosY);

    final sinY = 2.0 * (w * y - z * x);
    final rotationY = sinY.abs() >= 1.0
        ? math.pi / 2.0 * sinY.sign
        : math.asin(sinY);

    final sinZCosY = 2.0 * (w * z + x * y);
    final cosZCosY = 1.0 - 2.0 * (y * y + z * z);
    final rotationZ = math.atan2(sinZCosY, cosZCosY);

    return _Vec3(rotationX, rotationY, rotationZ);
  }
}
