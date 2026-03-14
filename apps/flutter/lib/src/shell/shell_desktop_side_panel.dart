import 'package:flutter/material.dart';

class ShellDesktopSidePanel extends StatelessWidget {
  const ShellDesktopSidePanel({
    super.key,
    required this.width,
    required this.child,
  });

  final double width;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      child: child,
    );
  }
}
