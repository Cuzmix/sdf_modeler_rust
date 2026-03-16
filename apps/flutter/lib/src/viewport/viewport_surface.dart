import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

const double _viewportAspectRatio = 16.0 / 9.0;

typedef ViewportSizeChanged = void Function(
  Size logicalViewportSize,
  double devicePixelRatio,
);
typedef ViewportDragCallback = void Function(Offset delta);
typedef ViewportTapCallback = void Function(
  Offset localPosition,
  Size logicalViewportSize,
);
typedef ViewportHoverCallback = void Function(
  Offset localPosition,
  Size logicalViewportSize,
);
typedef ViewportScrollCallback = void Function(double deltaY);
typedef ViewportPointerInterceptCallback = bool Function(
  PointerEvent event,
  Size logicalViewportSize,
);

class ViewportSurface extends StatefulWidget {
  const ViewportSurface({
    super.key,
    required this.textureId,
    required this.onViewportSizeChanged,
    required this.onOrbitDrag,
    required this.onPanDrag,
    required this.onPrimaryTap,
    required this.onHover,
    required this.onHoverExit,
    required this.onScroll,
    required this.onInteractionEnd,
    this.onPointerIntercept,
    this.overlay,
    this.controlsOverlay,
  });

  final int? textureId;
  final ViewportSizeChanged onViewportSizeChanged;
  final ViewportDragCallback onOrbitDrag;
  final ViewportDragCallback onPanDrag;
  final ViewportTapCallback onPrimaryTap;
  final ViewportHoverCallback onHover;
  final VoidCallback onHoverExit;
  final ViewportScrollCallback onScroll;
  final VoidCallback onInteractionEnd;
  final ViewportPointerInterceptCallback? onPointerIntercept;
  final Widget? overlay;
  final Widget? controlsOverlay;

  @override
  State<ViewportSurface> createState() => _ViewportSurfaceState();
}

class _ViewportSurfaceState extends State<ViewportSurface> {
  Offset? _pressLocalPosition;
  Offset? _lastHoverLocalPosition;
  Offset? _lastDragLocalPosition;
  PointerDeviceKind? _pressPointerKind;
  int _pressButtons = 0;
  bool _dragGestureStarted = false;
  bool _tapCanceled = false;
  Size _currentViewportSize = const Size(1, 1);
  final Map<int, Offset> _touchPositions = <int, Offset>{};
  final Map<int, Offset> _touchStartPositions = <int, Offset>{};
  Offset? _lastTouchCentroid;
  double? _lastTouchSpan;
  bool _touchGestureStarted = false;
  bool _touchTapCanceled = false;

  void _handlePointerDown(PointerDownEvent event) {
    if (_interceptPointerEvent(event)) {
      _resetMouseState();
      return;
    }
    if (ShellGestureContract.isTouchLike(event.kind)) {
      _handleTouchPointerDown(event);
      return;
    }

    _pressLocalPosition = event.localPosition;
    _lastHoverLocalPosition = event.localPosition;
    _lastDragLocalPosition = event.localPosition;
    _pressPointerKind = event.kind;
    _pressButtons = event.buttons;
    _dragGestureStarted = false;
    _tapCanceled = false;
  }

  void _handlePointerMove(PointerMoveEvent event) {
    if (_interceptPointerEvent(event)) {
      return;
    }
    if (ShellGestureContract.isTouchLike(event.kind)) {
      _handleTouchPointerMove(event);
      return;
    }

    final pressLocalPosition = _pressLocalPosition;
    final pointerKind = _pressPointerKind ?? event.kind;

    if (pressLocalPosition != null &&
        !_tapCanceled &&
        (event.localPosition - pressLocalPosition).distance >
            ShellGestureContract.tapSlopFor(pointerKind)) {
      _tapCanceled = true;
    }

    final isOrbitDrag = (event.buttons & kPrimaryMouseButton) != 0;
    final isPanDrag =
        (event.buttons & kSecondaryMouseButton) != 0 ||
        (event.buttons & kMiddleMouseButton) != 0;
    if (!isOrbitDrag && !isPanDrag) {
      return;
    }

    if (pressLocalPosition == null) {
      return;
    }

    if (!_dragGestureStarted) {
      final dragDistance = (event.localPosition - pressLocalPosition).distance;
      if (dragDistance < ShellGestureContract.dragStartSlopFor(pointerKind)) {
        return;
      }

      _dragGestureStarted = true;
      _tapCanceled = true;
      _lastDragLocalPosition = event.localPosition;
      if (isOrbitDrag) {
        widget.onOrbitDrag(event.localDelta);
        return;
      }

      widget.onPanDrag(event.localDelta);
      return;
    }

    final previousDragPosition = _lastDragLocalPosition ?? event.localPosition;
    final dragDelta = event.localPosition - previousDragPosition;
    if (dragDelta.distance == 0.0) {
      return;
    }

    _lastDragLocalPosition = event.localPosition;

    if (isOrbitDrag) {
      widget.onOrbitDrag(dragDelta);
      return;
    }

    widget.onPanDrag(dragDelta);
  }

  void _handlePointerUp(PointerUpEvent event) {
    if (_interceptPointerEvent(event)) {
      _resetMouseState();
      widget.onInteractionEnd();
      return;
    }
    if (ShellGestureContract.isTouchLike(event.kind)) {
      _handleTouchPointerUp(event);
      return;
    }

    final pressedButtons = _pressButtons;
    final shouldSelect =
        !_tapCanceled && (pressedButtons & kPrimaryMouseButton) != 0;

    _pressLocalPosition = null;
    _lastDragLocalPosition = null;
    _pressPointerKind = null;
    _pressButtons = 0;
    _dragGestureStarted = false;
    _tapCanceled = false;
    _lastHoverLocalPosition = null;

    if (shouldSelect) {
      widget.onPrimaryTap(event.localPosition, _currentViewportSize);
    }

    widget.onInteractionEnd();
  }

  void _handlePointerCancel(PointerCancelEvent event) {
    if (_interceptPointerEvent(event)) {
      _resetMouseState();
      widget.onInteractionEnd();
      return;
    }
    if (ShellGestureContract.isTouchLike(event.kind)) {
      _handleTouchPointerCancel(event);
      return;
    }

    _resetMouseState();
    widget.onInteractionEnd();
  }

  void _handlePointerSignal(PointerSignalEvent event) {
    if (_interceptPointerEvent(event)) {
      widget.onInteractionEnd();
      return;
    }
    if (event is! PointerScrollEvent) {
      return;
    }

    widget.onScroll(-event.scrollDelta.dy);
    widget.onInteractionEnd();
  }

  void _handleMouseHover(PointerHoverEvent event) {
    if (_interceptPointerEvent(event)) {
      return;
    }
    if (_pressButtons != 0) {
      return;
    }

    final previousHoverPosition = _lastHoverLocalPosition;
    if (previousHoverPosition != null &&
        (event.localPosition - previousHoverPosition).distance <
            ShellGestureContract.hoverUpdateSlop) {
      return;
    }

    _lastHoverLocalPosition = event.localPosition;
    widget.onHover(event.localPosition, _currentViewportSize);
  }

  void _handleMouseExit(PointerExitEvent event) {
    if (_interceptPointerEvent(event)) {
      return;
    }
    _lastHoverLocalPosition = null;
    if (_pressButtons != 0) {
      return;
    }

    widget.onHoverExit();
  }

  bool _interceptPointerEvent(PointerEvent event) {
    final intercept = widget.onPointerIntercept;
    if (intercept == null) {
      return false;
    }
    return intercept(event, _currentViewportSize);
  }

  void _resetMouseState() {
    _pressLocalPosition = null;
    _lastDragLocalPosition = null;
    _pressPointerKind = null;
    _pressButtons = 0;
    _dragGestureStarted = false;
    _tapCanceled = false;
    _lastHoverLocalPosition = null;
  }

  void _handleTouchPointerDown(PointerDownEvent event) {
    _touchPositions[event.pointer] = event.localPosition;
    _touchStartPositions[event.pointer] = event.localPosition;

    if (_touchPositions.length == 1) {
      _touchGestureStarted = false;
      _touchTapCanceled = false;
    } else {
      _touchTapCanceled = true;
    }

    _lastTouchCentroid = _touchCentroid();
    _lastTouchSpan = _touchSpan();
  }

  void _handleTouchPointerMove(PointerMoveEvent event) {
    if (!_touchPositions.containsKey(event.pointer)) {
      return;
    }

    _touchPositions[event.pointer] = event.localPosition;
    final pointerCount = _touchPositions.length;
    if (pointerCount == 0) {
      return;
    }

    if (pointerCount == 1) {
      final currentPosition = event.localPosition;
      final startPosition =
          _touchStartPositions[event.pointer] ?? currentPosition;
      final dragDistance = (currentPosition - startPosition).distance;
      if (!_touchGestureStarted) {
        if (dragDistance < ShellGestureContract.dragStartSlopFor(event.kind)) {
          return;
        }

        _touchGestureStarted = true;
        _touchTapCanceled = true;
        _lastTouchCentroid = currentPosition;
      }

      final previousPosition = _lastTouchCentroid ?? currentPosition;
      final delta = currentPosition - previousPosition;
      _lastTouchCentroid = currentPosition;
      if (delta.distance == 0.0) {
        return;
      }

      widget.onOrbitDrag(delta);
      return;
    }

    final centroid = _touchCentroid();
    final span = _touchSpan();
    if (centroid == null || span == null) {
      return;
    }

    if (!_touchGestureStarted) {
      final maxTouchMovement = _touchStartPositions.entries.fold<double>(
        0.0,
        (maxDistance, entry) {
          final currentPosition = _touchPositions[entry.key];
          if (currentPosition == null) {
            return maxDistance;
          }

          final distance = (currentPosition - entry.value).distance;
          return distance > maxDistance ? distance : maxDistance;
        },
      );
      final spanDelta = (span - (_lastTouchSpan ?? span)).abs();
      final gestureSlop = ShellGestureContract.dragStartSlopFor(event.kind);
      if (maxTouchMovement < gestureSlop && spanDelta < gestureSlop) {
        return;
      }

      _touchGestureStarted = true;
      _touchTapCanceled = true;
    }

    final previousCentroid = _lastTouchCentroid ?? centroid;
    final previousSpan = _lastTouchSpan ?? span;
    final panDelta = centroid - previousCentroid;
    final zoomDelta = span - previousSpan;
    _lastTouchCentroid = centroid;
    _lastTouchSpan = span;

    if (panDelta.distance != 0.0) {
      widget.onPanDrag(panDelta);
    }
    if (zoomDelta != 0.0) {
      widget.onScroll(zoomDelta);
    }
  }

  void _handleTouchPointerUp(PointerUpEvent event) {
    final shouldSelect =
        !_touchTapCanceled &&
        !_touchGestureStarted &&
        _touchPositions.length == 1 &&
        _touchPositions.containsKey(event.pointer);

    _touchPositions.remove(event.pointer);
    _touchStartPositions.remove(event.pointer);

    if (shouldSelect) {
      widget.onPrimaryTap(event.localPosition, _currentViewportSize);
    }

    if (_touchPositions.isEmpty) {
      _resetTouchState();
      widget.onInteractionEnd();
      return;
    }

    _reseedTouchGesture();
  }

  void _handleTouchPointerCancel(PointerCancelEvent event) {
    _touchPositions.remove(event.pointer);
    _touchStartPositions.remove(event.pointer);
    if (_touchPositions.isEmpty) {
      _resetTouchState();
      widget.onInteractionEnd();
      return;
    }

    _reseedTouchGesture();
  }

  void _reseedTouchGesture() {
    final currentPositions = Map<int, Offset>.from(_touchPositions);
    _touchStartPositions
      ..clear()
      ..addAll(currentPositions);
    _touchGestureStarted = false;
    _touchTapCanceled = true;
    _lastTouchCentroid = _touchCentroid();
    _lastTouchSpan = _touchSpan();
  }

  void _resetTouchState() {
    _touchPositions.clear();
    _touchStartPositions.clear();
    _lastTouchCentroid = null;
    _lastTouchSpan = null;
    _touchGestureStarted = false;
    _touchTapCanceled = false;
  }

  Offset? _touchCentroid() {
    if (_touchPositions.isEmpty) {
      return null;
    }

    var sumX = 0.0;
    var sumY = 0.0;
    for (final position in _touchPositions.values) {
      sumX += position.dx;
      sumY += position.dy;
    }

    return Offset(sumX / _touchPositions.length, sumY / _touchPositions.length);
  }

  double? _touchSpan() {
    if (_touchPositions.length < 2) {
      return null;
    }

    final positions = _touchPositions.values.take(2).toList(growable: false);
    return (positions[0] - positions[1]).distance;
  }

  @override
  Widget build(BuildContext context) {
    final shellPalette = context.shellPalette;

    return LayoutBuilder(
      builder: (context, constraints) {
        final viewportSize = _containedViewportSize(
          Size(constraints.maxWidth, constraints.maxHeight),
        );
        _currentViewportSize = viewportSize;
        final devicePixelRatio = MediaQuery.devicePixelRatioOf(context);

        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (context.mounted) {
            widget.onViewportSizeChanged(viewportSize, devicePixelRatio);
          }
        });

        return DecoratedBox(
          decoration: ShellSurfaceStyles.viewportFrame(context),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
            child: Center(
              child: SizedBox(
                width: viewportSize.width,
                height: viewportSize.height,
                child: MouseRegion(
                  cursor: SystemMouseCursors.precise,
                  onHover: _handleMouseHover,
                  onExit: _handleMouseExit,
                  child: Listener(
                    behavior: HitTestBehavior.opaque,
                    onPointerDown: _handlePointerDown,
                    onPointerMove: _handlePointerMove,
                    onPointerUp: _handlePointerUp,
                    onPointerCancel: _handlePointerCancel,
                    onPointerSignal: _handlePointerSignal,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        widget.textureId == null
                            ? Center(
                                child: Text(
                                  'Preparing real viewport...',
                                  style: Theme.of(context).textTheme.bodyMedium
                                      ?.copyWith(
                                        color: shellPalette.overlayMutedText,
                                      ),
                                ),
                              )
                            : Texture(textureId: widget.textureId!),
                        if (widget.overlay != null)
                          IgnorePointer(child: widget.overlay!),
                        if (widget.controlsOverlay != null)
                          widget.controlsOverlay!,
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Size _containedViewportSize(Size availableSize) {
    if (availableSize.width <= 0 || availableSize.height <= 0) {
      return const Size(1, 1);
    }

    var viewportWidth = availableSize.width;
    var viewportHeight = viewportWidth / _viewportAspectRatio;

    if (viewportHeight > availableSize.height) {
      viewportHeight = availableSize.height;
      viewportWidth = viewportHeight * _viewportAspectRatio;
    }

    return Size(viewportWidth, viewportHeight);
  }
}
