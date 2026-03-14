import 'texture_viewport_feedback.dart';

class TextureViewportEvent {
  const TextureViewportEvent({
    required this.textureId,
    required this.frameWidth,
    required this.frameHeight,
    required this.frameTimeMs,
    required this.frameCount,
    required this.droppedFrameCount,
    required this.interactionPhase,
    required this.sceneStateChanged,
    required this.hostError,
    required this.feedback,
  });

  final int textureId;
  final int frameWidth;
  final int frameHeight;
  final double frameTimeMs;
  final int frameCount;
  final int droppedFrameCount;
  final String interactionPhase;
  final bool sceneStateChanged;
  final String? hostError;
  final TextureViewportFeedback? feedback;

  bool get interactionActive => interactionPhase != 'idle';

  factory TextureViewportEvent.fromDynamic(dynamic rawEvent) {
    if (rawEvent is! Map) {
      throw FormatException('Texture viewport event must be a map.');
    }

    final eventMap = rawEvent.cast<Object?, Object?>();
    return TextureViewportEvent(
      textureId: _readInt(eventMap, 'textureId'),
      frameWidth: _readInt(eventMap, 'frameWidth'),
      frameHeight: _readInt(eventMap, 'frameHeight'),
      frameTimeMs: _readDouble(eventMap, 'frameTimeMs'),
      frameCount: _readInt(eventMap, 'frameCount'),
      droppedFrameCount: _readInt(eventMap, 'droppedFrameCount'),
      interactionPhase: _readInteractionPhase(eventMap),
      sceneStateChanged: _readBool(eventMap, 'sceneStateChanged'),
      hostError: _readOptionalString(eventMap, 'hostError'),
      feedback: _readFeedback(eventMap, 'feedbackJson'),
    );
  }

  static int _readInt(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is int) {
      return value;
    }
    if (value is double) {
      return value.round();
    }
    throw FormatException('Texture viewport event key "$key" must be numeric.');
  }

  static double _readDouble(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is num) {
      return value.toDouble();
    }
    throw FormatException('Texture viewport event key "$key" must be numeric.');
  }

  static bool _readBool(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value is bool) {
      return value;
    }
    throw FormatException('Texture viewport event key "$key" must be boolean.');
  }

  static String _readInteractionPhase(Map<Object?, Object?> map) {
    final phaseValue = map['interactionPhase'];
    if (phaseValue is String && phaseValue.isNotEmpty) {
      return phaseValue;
    }

    final interactionActiveValue = map['interactionActive'];
    if (interactionActiveValue is bool) {
      return interactionActiveValue ? 'interacting' : 'idle';
    }

    throw FormatException(
      'Texture viewport event must contain interactionPhase or interactionActive.',
    );
  }

  static String? _readOptionalString(Map<Object?, Object?> map, String key) {
    final value = map[key];
    if (value == null) {
      return null;
    }
    if (value is String) {
      return value.isEmpty ? null : value;
    }
    throw FormatException('Texture viewport event key "$key" must be a string.');
  }

  static TextureViewportFeedback? _readFeedback(
    Map<Object?, Object?> map,
    String key,
  ) {
    final value = map[key];
    if (value == null) {
      return null;
    }

    if (value is! String) {
      throw FormatException(
        'Texture viewport event key "$key" must be a JSON string.',
      );
    }

    if (value.isEmpty) {
      return null;
    }

    return TextureViewportFeedback.fromJsonString(value);
  }
}
