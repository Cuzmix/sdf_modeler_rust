import 'package:flutter/gestures.dart';
import 'package:flutter/widgets.dart';
import '../../models/camera_state.dart';

/// Wraps a child widget with tablet-first gesture handling.
///
/// Touch (tablet / mobile):
///   - 1-finger drag → orbit
///   - 2-finger pinch → zoom
///   - 2-finger pan → pan
///
/// Mouse (desktop fallback):
///   - LMB drag → orbit
///   - RMB drag → pan
///   - Scroll wheel → zoom
class ViewportGestureHandler extends StatefulWidget {
  final CameraState camera;
  final Widget child;
  final VoidCallback onInteractionStart;
  final VoidCallback onInteractionEnd;

  const ViewportGestureHandler({
    super.key,
    required this.camera,
    required this.child,
    required this.onInteractionStart,
    required this.onInteractionEnd,
  });

  @override
  State<ViewportGestureHandler> createState() =>
      _ViewportGestureHandlerState();
}

class _ViewportGestureHandlerState extends State<ViewportGestureHandler> {
  int _activeButton = 0;
  int _pointerCount = 0;
  double _previousScale = 1.0;

  @override
  Widget build(BuildContext context) {
    return Listener(
      // Mouse button tracking (desktop)
      onPointerDown: (event) {
        _pointerCount++;
        _activeButton = event.buttons;
        widget.onInteractionStart();
      },
      onPointerMove: (event) {
        // Only handle mouse moves (touch goes through GestureDetector)
        if (event.kind == PointerDeviceKind.mouse) {
          final dx = event.delta.dx;
          final dy = event.delta.dy;
          if (_activeButton & kSecondaryMouseButton != 0) {
            widget.camera.pan(dx, dy);
          } else {
            widget.camera.orbit(dx, dy);
          }
        }
      },
      onPointerUp: (_) {
        _pointerCount--;
        if (_pointerCount <= 0) {
          _pointerCount = 0;
          _activeButton = 0;
          widget.onInteractionEnd();
        }
      },
      onPointerCancel: (_) {
        _pointerCount--;
        if (_pointerCount <= 0) {
          _pointerCount = 0;
          _activeButton = 0;
          widget.onInteractionEnd();
        }
      },
      onPointerSignal: (event) {
        if (event is PointerScrollEvent) {
          widget.onInteractionStart();
          widget.camera.zoom(-event.scrollDelta.dy);
          widget.onInteractionEnd();
        }
      },
      child: GestureDetector(
        // Touch gesture handling (tablet-first)
        onScaleStart: (_) {
          _previousScale = 1.0;
          widget.onInteractionStart();
        },
        onScaleUpdate: (details) {
          if (details.pointerCount == 1) {
            // Single finger → orbit
            widget.camera.orbit(
              details.focalPointDelta.dx,
              details.focalPointDelta.dy,
            );
          } else if (details.pointerCount >= 2) {
            // Two fingers → zoom + pan
            final scaleDelta = details.scale / _previousScale;
            if (scaleDelta != 1.0) {
              // Convert scale ratio to zoom delta
              widget.camera.zoom((scaleDelta - 1.0) * 500.0);
            }
            _previousScale = details.scale;
            widget.camera.pan(
              details.focalPointDelta.dx,
              details.focalPointDelta.dy,
            );
          }
        },
        onScaleEnd: (_) {
          widget.onInteractionEnd();
        },
        child: widget.child,
      ),
    );
  }
}
