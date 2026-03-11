import 'package:flutter/services.dart';

class TextureBridge {
  TextureBridge._();

  static final TextureBridge instance = TextureBridge._();

  static const MethodChannel _channel = MethodChannel('sdf_modeler/texture');

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

  Future<void> disposeTexture(int textureId) {
    return _channel.invokeMethod<void>('disposeTexture', {
      'textureId': textureId,
    });
  }
}

