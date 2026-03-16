import 'dart:convert';

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

  factory TextureViewportFeedback.fromJson(Map<String, dynamic> json) {
    return TextureViewportFeedback(
      camera: _readCamera(json['camera'] as Map<String, dynamic>),
      selectedNode: _readNode(json['selected_node']),
      hoveredNode: _readNode(json['hovered_node']),
    );
  }

  factory TextureViewportFeedback.fromJsonString(String rawJson) {
    final decoded = jsonDecode(rawJson);
    if (decoded is! Map) {
      throw FormatException('Viewport feedback must decode to a map.');
    }

    return TextureViewportFeedback.fromJson(decoded.cast<String, dynamic>());
  }

  factory TextureViewportFeedback.fromSceneSnapshot(AppSceneSnapshot snapshot) {
    return TextureViewportFeedback(
      camera: snapshot.camera,
      selectedNode: snapshot.selectedNode,
      hoveredNode: null,
    );
  }

  static AppNodeSnapshot? _readNode(Object? rawNode) {
    if (rawNode == null) {
      return null;
    }

    final json = rawNode as Map<String, dynamic>;
    return AppNodeSnapshot(
      id: BigInt.from((json['id'] as num).toInt()),
      name: json['name'] as String,
      kindLabel: json['kind_label'] as String,
      visible: json['visible'] as bool,
      locked: json['locked'] as bool,
    );
  }

  static AppCameraSnapshot _readCamera(Map<String, dynamic> json) {
    return AppCameraSnapshot(
      yaw: (json['yaw'] as num).toDouble(),
      pitch: (json['pitch'] as num).toDouble(),
      roll: (json['roll'] as num).toDouble(),
      distance: (json['distance'] as num).toDouble(),
      fovDegrees: (json['fov_degrees'] as num).toDouble(),
      orthographic: json['orthographic'] as bool,
      target: _readVec3(json['target'] as Map<String, dynamic>),
      eye: _readVec3(json['eye'] as Map<String, dynamic>),
    );
  }

  static AppVec3 _readVec3(Map<String, dynamic> json) {
    return AppVec3(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      z: (json['z'] as num).toDouble(),
    );
  }
}
