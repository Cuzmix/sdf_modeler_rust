import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

enum ShellSizeClass { tablet, desktop }

abstract final class ShellTokens {
  static const Color themeSeedColor = Colors.teal;

  static const double minimumTouchTarget = 52.0;
  static const double shellPaddingTablet = 16.0;
  static const double shellPaddingDesktop = 24.0;
  static const double panelGapTablet = 16.0;
  static const double panelGapDesktop = 20.0;
  static const double panelPadding = 20.0;
  static const double sectionGap = 20.0;
  static const double controlGap = 12.0;
  static const double compactGap = 6.0;
  static const double surfaceRadius = 16.0;
  static const double sceneTreeIndent = 20.0;
  static const double sceneTreeNodeGap = 6.0;
  static const double sceneTreeTileHorizontalPadding = 14.0;
  static const double sceneTreeTileVerticalPadding = 12.0;
  static const double overlayPadding = 12.0;
  static const double overlayChipHorizontalPadding = 12.0;
  static const double overlayChipVerticalPadding = 8.0;
  static const double sheetHandleWidth = 56.0;
  static const double sheetHandleHeight = 6.0;
  static const double modalPanelMaxWidth = 760.0;
  static const double modalPanelHeightFactor = 0.78;
}

abstract final class ShellPanelPlacement {
  static const double tabletSheetMinSize = 0.24;
  static const double tabletSheetInitialSize = 0.36;
  static const double tabletSheetMaxSize = 0.88;
}

@immutable
class ShellLayout {
  const ShellLayout._({
    required this.sizeClass,
    required this.screenPadding,
    required this.panelGap,
    required this.inspectorPanelExtent,
  });

  static const double desktopBreakpoint = 1100.0;

  static const ShellLayout tablet = ShellLayout._(
    sizeClass: ShellSizeClass.tablet,
    screenPadding: ShellTokens.shellPaddingTablet,
    panelGap: ShellTokens.panelGapTablet,
    inspectorPanelExtent: 420.0,
  );

  static const ShellLayout desktop = ShellLayout._(
    sizeClass: ShellSizeClass.desktop,
    screenPadding: ShellTokens.shellPaddingDesktop,
    panelGap: ShellTokens.panelGapDesktop,
    inspectorPanelExtent: 360.0,
  );

  final ShellSizeClass sizeClass;
  final double screenPadding;
  final double panelGap;
  final double inspectorPanelExtent;

  bool get useSidePanel => sizeClass == ShellSizeClass.desktop;

  static ShellLayout forWidth(double maxWidth) {
    return maxWidth >= desktopBreakpoint ? desktop : tablet;
  }
}

abstract final class ShellGestureContract {
  static const Duration viewportInteractionCooldown = Duration(
    milliseconds: 180,
  );
  static const double mouseTapSlop = 6.0;
  static const double touchTapSlop = 18.0;
  static const double mouseDragStartSlop = 6.0;
  static const double touchDragStartSlop = 18.0;
  static const double hoverUpdateSlop = 6.0;

  static bool isTouchLike(PointerDeviceKind kind) {
    return kind == PointerDeviceKind.touch ||
        kind == PointerDeviceKind.stylus ||
        kind == PointerDeviceKind.invertedStylus ||
        kind == PointerDeviceKind.unknown;
  }

  static double tapSlopFor(PointerDeviceKind kind) {
    return isTouchLike(kind) ? touchTapSlop : mouseTapSlop;
  }

  static double dragStartSlopFor(PointerDeviceKind kind) {
    return isTouchLike(kind) ? touchDragStartSlop : mouseDragStartSlop;
  }
}
