import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

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
    this.overlay,
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
  final Widget? overlay;

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

  void _handlePointerDown(PointerDownEvent event) {
    _pressLocalPosition = event.localPosition;
    _lastHoverLocalPosition = event.localPosition;
    _lastDragLocalPosition = event.localPosition;
    _pressPointerKind = event.kind;
    _pressButtons = event.buttons;
    _dragGestureStarted = false;
    _tapCanceled = false;
  }

  void _handlePointerMove(PointerMoveEvent event) {
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
    _pressLocalPosition = null;
    _lastDragLocalPosition = null;
    _pressPointerKind = null;
    _pressButtons = 0;
    _dragGestureStarted = false;
    _tapCanceled = false;
    _lastHoverLocalPosition = null;
    widget.onInteractionEnd();
  }

  void _handlePointerSignal(PointerSignalEvent event) {
    if (event is! PointerScrollEvent) {
      return;
    }

    widget.onScroll(-event.scrollDelta.dy);
    widget.onInteractionEnd();
  }

  void _handleMouseHover(PointerHoverEvent event) {
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
    _lastHoverLocalPosition = null;
    if (_pressButtons != 0) {
      return;
    }

    widget.onHoverExit();
  }

  @override
  Widget build(BuildContext context) {
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
          decoration: BoxDecoration(
            color: Colors.black,
            borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
            border: Border.all(color: Colors.white24),
          ),
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
                            ? const Center(
                                child: Text(
                                  'Preparing real viewport...',
                                  style: TextStyle(color: Colors.white70),
                                ),
                              )
                            : Texture(textureId: widget.textureId!),
                        if (widget.overlay != null)
                          IgnorePointer(child: widget.overlay!),
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
