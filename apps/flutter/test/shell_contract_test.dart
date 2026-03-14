import 'package:flutter/gestures.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

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
}
