import 'package:flutter/services.dart';

import 'texture_viewport_event.dart';

class TextureBridge {
  TextureBridge._();

  static final TextureBridge instance = TextureBridge._();

  static const MethodChannel _channel = MethodChannel('sdf_modeler/texture');
  static const EventChannel _eventsChannel = EventChannel(
    'sdf_modeler/texture_events',
  );

  Stream<TextureViewportEvent> get events {
    return _eventsChannel.receiveBroadcastStream().map(
      TextureViewportEvent.fromDynamic,
    );
  }

  Future<int> createTexture({required int width, required int height}) async {
    final textureId = await _channel.invokeMethod<int>('createTexture', {
      'width': width,
      'height': height,
    });

    if (textureId == null) {
      throw StateError('Native texture bridge returned null texture ID.');
    }

    return textureId;
  }

  Future<void> updateTexture({
    required int textureId,
    required int width,
    required int height,
    required Uint8List pixels,
  }) {
    return _channel.invokeMethod<void>('updateTexture', {
      'textureId': textureId,
      'width': width,
      'height': height,
      'pixels': pixels,
    });
  }

  Future<void> setTextureSize({
    required int textureId,
    required int width,
    required int height,
  }) {
    return _channel.invokeMethod<void>('setTextureSize', {
      'textureId': textureId,
      'width': width,
      'height': height,
    });
  }

  Future<void> requestFrame({required int textureId}) {
    return _channel.invokeMethod<void>('requestFrame', {
      'textureId': textureId,
    });
  }

  Future<void> orbitCamera({
    required int textureId,
    required double deltaX,
    required double deltaY,
  }) {
    return _channel.invokeMethod<void>('orbitCamera', {
      'textureId': textureId,
      'deltaX': deltaX,
      'deltaY': deltaY,
    });
  }

  Future<void> panCamera({
    required int textureId,
    required double deltaX,
    required double deltaY,
  }) {
    return _channel.invokeMethod<void>('panCamera', {
      'textureId': textureId,
      'deltaX': deltaX,
      'deltaY': deltaY,
    });
  }

  Future<void> zoomCamera({required int textureId, required double delta}) {
    return _channel.invokeMethod<void>('zoomCamera', {
      'textureId': textureId,
      'delta': delta,
    });
  }

  Future<void> pickNode({
    required int textureId,
    required double normalizedX,
    required double normalizedY,
  }) {
    return _channel.invokeMethod<void>('pickNode', {
      'textureId': textureId,
      'normalizedX': normalizedX,
      'normalizedY': normalizedY,
    });
  }

  Future<void> hoverNode({
    required int textureId,
    required double normalizedX,
    required double normalizedY,
  }) {
    return _channel.invokeMethod<void>('hoverNode', {
      'textureId': textureId,
      'normalizedX': normalizedX,
      'normalizedY': normalizedY,
    });
  }

  Future<void> clearHover({required int textureId}) {
    return _channel.invokeMethod<void>('clearHover', {
      'textureId': textureId,
    });
  }

  Future<void> disposeTexture(int textureId) {
    return _channel.invokeMethod<void>('disposeTexture', {
      'textureId': textureId,
    });
  }
}