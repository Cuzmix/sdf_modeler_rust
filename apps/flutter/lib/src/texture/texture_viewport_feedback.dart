import 'dart:convert';

import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';

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
      camera: AppCameraSnapshot.fromJson(json['camera'] as Map<String, dynamic>),
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

    return AppNodeSnapshot.fromJson(rawNode as Map<String, dynamic>);
  }
}