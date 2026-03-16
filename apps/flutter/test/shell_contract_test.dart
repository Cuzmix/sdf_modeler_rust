import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

void main() {
  test('tablet layout stays primary until the desktop breakpoint', () {
    expect(ShellLayout.forWidth(900).sizeClass, ShellSizeClass.tablet);
    expect(
      ShellLayout.forWidth(ShellLayout.desktopBreakpoint - 1).useSidePanel,
      isFalse,
    );
    expect(
      ShellLayout.forWidth(ShellLayout.desktopBreakpoint).sizeClass,
      ShellSizeClass.desktop,
    );
  });

  test('touch-like pointers use larger gesture thresholds than mouse', () {
    expect(
      ShellGestureContract.tapSlopFor(PointerDeviceKind.touch),
      greaterThan(ShellGestureContract.tapSlopFor(PointerDeviceKind.mouse)),
    );
    expect(
      ShellGestureContract.dragStartSlopFor(PointerDeviceKind.stylus),
      ShellGestureContract.dragStartSlopFor(PointerDeviceKind.touch),
    );
    expect(
      ShellGestureContract.dragStartSlopFor(PointerDeviceKind.mouse),
      lessThan(
        ShellGestureContract.dragStartSlopFor(PointerDeviceKind.touch),
      ),
    );
  });

  test('shell theme exposes the customization palette and touch-first controls', () {
    final theme = buildTouchFirstShellTheme();
    final shellPalette = theme.extension<ShellPalette>();

    expect(shellPalette, isNotNull);
    expect(theme.scaffoldBackgroundColor, ShellPalette.dusk.canvasBase);
    expect(theme.inputDecorationTheme.filled, isTrue);
    expect(
      theme.filledButtonTheme.style?.minimumSize?.resolve(const <WidgetState>{}),
      const Size(0, ShellTokens.minimumTouchTarget),
    );
    expect(theme.textTheme.titleLarge?.fontFamily, 'Bahnschrift');
  });
}
