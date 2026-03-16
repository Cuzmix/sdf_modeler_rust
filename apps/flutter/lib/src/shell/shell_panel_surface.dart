import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

class ShellPanelSurface extends StatelessWidget {
  const ShellPanelSurface({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(ShellTokens.panelPadding),
  });

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: ShellSurfaceStyles.panel(context),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
        child: Material(
          color: Colors.transparent,
          child: Padding(
            padding: padding,
            child: child,
          ),
        ),
      ),
    );
  }
}
