import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_event.dart';

void main() {
  test('decodes viewport events from standard channel maps', () {
    final event = TextureViewportEvent.fromDynamic(<String, Object?>{
      'textureId': 7,
      'frameWidth': 960,
      'frameHeight': 540,
      'frameTimeMs': 12.5,
      'frameCount': 42,
      'droppedFrameCount': 3,
      'interactionPhase': 'interacting',
      'sceneStateChanged': true,
      'feedbackJson':
          '{"camera":{"yaw":0.5,"pitch":0.25,"roll":0.0,"distance":5.5,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":1.0,"y":2.0,"z":3.0}},"selected_node":{"id":9,"name":"Sphere 2","kind_label":"Sphere","visible":true,"locked":false},"hovered_node":{"id":10,"name":"Box 1","kind_label":"Box","visible":true,"locked":false}}',
    });

    expect(event.textureId, 7);
    expect(event.frameWidth, 960);
    expect(event.frameHeight, 540);
    expect(event.frameTimeMs, 12.5);
    expect(event.frameCount, 42);
    expect(event.droppedFrameCount, 3);
    expect(event.interactionPhase, 'interacting');
    expect(event.interactionActive, isTrue);
    expect(event.sceneStateChanged, isTrue);
    expect(event.feedback, isNotNull);
    expect(event.feedback!.camera.distance, 5.5);
    expect(event.feedback!.selectedNode!.name, 'Sphere 2');
    expect(event.feedback!.hoveredNode!.name, 'Box 1');
  });

  test('falls back to legacy interactionActive payloads', () {
    final event = TextureViewportEvent.fromDynamic(<String, Object?>{
      'textureId': 7,
      'frameWidth': 640,
      'frameHeight': 360,
      'frameTimeMs': 16.0,
      'frameCount': 1,
      'droppedFrameCount': 0,
      'interactionActive': false,
      'sceneStateChanged': false,
      'feedbackJson': '',
    });

    expect(event.interactionPhase, 'idle');
    expect(event.interactionActive, isFalse);
    expect(event.feedback, isNull);
  });
}