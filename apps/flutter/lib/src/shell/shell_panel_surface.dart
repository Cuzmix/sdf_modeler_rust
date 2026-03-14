import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

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
    return Card(
      child: Padding(
        padding: padding,
        child: child,
      ),
    );
  }
}
