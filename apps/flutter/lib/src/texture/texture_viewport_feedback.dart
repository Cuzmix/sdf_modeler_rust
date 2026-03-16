import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';

class TextureViewportFeedback {
  const TextureViewportFeedback({
    required this.camera,
    required this.selectedNode,
    required this.hoveredNode,
  });

  final AppCameraSnapshot camera;
  final AppNodeSnapshot? selectedNode;
  final AppNodeSnapshot? hoveredNode;

  factory TextureViewportFeedback.fromDynamic(Object? rawFeedback) {
    if (rawFeedback is! Map) {
      throw FormatException('Viewport feedback must be a map.');
    }

    final feedbackMap = rawFeedback.cast<Object?, Object?>();
    return TextureViewportFeedback(
      camera: _readCamera(_readMap(feedbackMap, 'camera')),
      selectedNode: _readNode(feedbackMap['selected_node']),
      hoveredNode: _readNode(feedbackMap['hovered_node']),
    );
  }

  factory TextureViewportFeedback.fromSceneSnapshot(AppSceneSnapshot snapshot) {
    return TextureViewportFeedback(
      camera: snapshot.camera,
      selectedNode: snapshot.selectedNode,
      hoveredNode: null,
    );
  }

  static Map<Object?, Object?> _readMap(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is Map) {
      return value.cast<Object?, Object?>();
    }
    throw FormatException('Viewport feedback key "$key" must be a map.');
  }

  static AppNodeSnapshot? _readNode(Object? rawNode) {
    if (rawNode == null) {
      return null;
    }

    if (rawNode is! Map) {
      throw FormatException('Viewport node feedback must be a map.');
    }

    final json = rawNode.cast<Object?, Object?>();
    return AppNodeSnapshot(
      id: _readBigInt(json, 'id'),
      name: _readString(json, 'name'),
      kindLabel: _readString(json, 'kind_label'),
      visible: _readBool(json, 'visible'),
      locked: _readBool(json, 'locked'),
    );
  }

  static AppCameraSnapshot _readCamera(Map<Object?, Object?> json) {
    return AppCameraSnapshot(
      yaw: _readDouble(json, 'yaw'),
      pitch: _readDouble(json, 'pitch'),
      roll: _readDouble(json, 'roll'),
      distance: _readDouble(json, 'distance'),
      fovDegrees: _readDouble(json, 'fov_degrees'),
      orthographic: _readBool(json, 'orthographic'),
      target: _readVec3(_readMap(json, 'target')),
      eye: _readVec3(_readMap(json, 'eye')),
    );
  }

  static AppVec3 _readVec3(Map<Object?, Object?> json) {
    return AppVec3(
      x: _readDouble(json, 'x'),
      y: _readDouble(json, 'y'),
      z: _readDouble(json, 'z'),
    );
  }

  static double _readDouble(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is num) {
      return value.toDouble();
    }
    throw FormatException('Viewport feedback key "$key" must be numeric.');
  }

  static String _readString(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is String) {
      return value;
    }
    throw FormatException('Viewport feedback key "$key" must be a string.');
  }

  static bool _readBool(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is bool) {
      return value;
    }
    throw FormatException('Viewport feedback key "$key" must be boolean.');
  }

  static BigInt _readBigInt(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is int) {
      return BigInt.from(value);
    }
    if (value is double) {
      return BigInt.from(value.round());
    }
    throw FormatException('Viewport feedback key "$key" must be numeric.');
  }
}
