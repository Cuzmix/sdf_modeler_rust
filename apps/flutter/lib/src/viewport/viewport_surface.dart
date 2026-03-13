import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

const double _viewportAspectRatio = 16.0 / 9.0;
const double _tapSlop = 6.0;

typedef ViewportSizeChanged = void Function(
  Size logicalViewportSize,
  double devicePixelRatio,
);
typedef ViewportDragCallback = void Function(Offset delta);
typedef ViewportTapCallback = void Function(
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
    required this.onScroll,
    required this.onInteractionEnd,
  });

  final int? textureId;
  final ViewportSizeChanged onViewportSizeChanged;
  final ViewportDragCallback onOrbitDrag;
  final ViewportDragCallback onPanDrag;
  final ViewportTapCallback onPrimaryTap;
  final ViewportScrollCallback onScroll;
  final VoidCallback onInteractionEnd;

  @override
  State<ViewportSurface> createState() => _ViewportSurfaceState();
}

class _ViewportSurfaceState extends State<ViewportSurface> {
  Offset? _pressLocalPosition;
  int _pressButtons = 0;
  bool _tapCanceled = false;
  Size _currentViewportSize = const Size(1, 1);

  void _handlePointerDown(PointerDownEvent event) {
    _pressLocalPosition = event.localPosition;
    _pressButtons = event.buttons;
    _tapCanceled = false;
  }

  void _handlePointerMove(PointerMoveEvent event) {
    if (_pressLocalPosition != null &&
        !_tapCanceled &&
        (event.localPosition - _pressLocalPosition!).distance > _tapSlop) {
      _tapCanceled = true;
    }

    if ((event.buttons & kPrimaryMouseButton) != 0) {
      widget.onOrbitDrag(event.localDelta);
      return;
    }

    if ((event.buttons & kSecondaryMouseButton) != 0 ||
        (event.buttons & kMiddleMouseButton) != 0) {
      widget.onPanDrag(event.localDelta);
    }
  }

  void _handlePointerUp(PointerUpEvent event) {
    final pressedButtons = _pressButtons;
    final shouldSelect =
        !_tapCanceled && (pressedButtons & kPrimaryMouseButton) != 0;

    _pressLocalPosition = null;
    _pressButtons = 0;
    _tapCanceled = false;

    if (shouldSelect) {
      widget.onPrimaryTap(event.localPosition, _currentViewportSize);
    }

    widget.onInteractionEnd();
  }

  void _handlePointerCancel(PointerCancelEvent event) {
    _pressLocalPosition = null;
    _pressButtons = 0;
    _tapCanceled = false;
    widget.onInteractionEnd();
  }

  void _handlePointerSignal(PointerSignalEvent event) {
    if (event is! PointerScrollEvent) {
      return;
    }

    widget.onScroll(-event.scrollDelta.dy);
    widget.onInteractionEnd();
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
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white24),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Center(
              child: SizedBox(
                width: viewportSize.width,
                height: viewportSize.height,
                child: Listener(
                  behavior: HitTestBehavior.opaque,
                  onPointerDown: _handlePointerDown,
                  onPointerMove: _handlePointerMove,
                  onPointerUp: _handlePointerUp,
                  onPointerCancel: _handlePointerCancel,
                  onPointerSignal: _handlePointerSignal,
                  child: MouseRegion(
                    cursor: SystemMouseCursors.precise,
                    child: widget.textureId == null
                        ? const Center(
                            child: Text(
                              'Preparing real viewport...',
                              style: TextStyle(color: Colors.white70),
                            ),
                          )
                        : Texture(textureId: widget.textureId!),
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