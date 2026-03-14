import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class ShellCommandStrip extends StatelessWidget {
  const ShellCommandStrip({
    super.key,
    required this.children,
  });

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    if (children.isEmpty) {
      return const SizedBox.shrink();
    }

    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Padding(
          padding: const EdgeInsets.all(ShellTokens.controlGap),
          child: Row(
            children: _withSpacing(children),
          ),
        ),
      ),
    );
  }

  List<Widget> _withSpacing(List<Widget> widgets) {
    final spacedWidgets = <Widget>[];
    for (var index = 0; index < widgets.length; index += 1) {
      if (index > 0) {
        spacedWidgets.add(const SizedBox(width: ShellTokens.controlGap));
      }
      spacedWidgets.add(widgets[index]);
    }
    return spacedWidgets;
  }
}
