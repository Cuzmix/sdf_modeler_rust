import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class ShellContextShelf extends StatelessWidget {
  const ShellContextShelf({
    super.key,
    required this.workspace,
    required this.selectionContext,
    required this.enabled,
    required this.onExecuteCommand,
  });

  final AppWorkspaceSnapshot workspace;
  final AppSelectionContextSnapshot selectionContext;
  final bool enabled;
  final ValueChanged<String> onExecuteCommand;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ShellPanelSurface(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.compactGap,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                selectionContext.headline,
                style: theme.textTheme.titleMedium,
              ),
              Chip(
                label: Text(workspace.label),
                avatar: const Icon(Icons.dashboard_customize_outlined, size: 18),
              ),
              Chip(
                label: Text(selectionContext.workflowStatusLabel),
                avatar: const Icon(Icons.layers_outlined, size: 18),
              ),
              if (selectionContext.selectionCount > 1)
                Chip(
                  label: Text('${selectionContext.selectionCount} selected'),
                  avatar: const Icon(Icons.select_all_outlined, size: 18),
                ),
            ],
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            selectionContext.detail,
            style: theme.textTheme.bodySmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.controlGap,
            children: selectionContext.quickActions.map((action) {
              final buttonChild = Text(action.label);
              if (action.prominent) {
                return FilledButton(
                  key: ValueKey('context-action-${action.id}'),
                  onPressed: enabled && action.enabled
                      ? () => onExecuteCommand(action.id)
                      : null,
                  child: buttonChild,
                );
              }

              return OutlinedButton(
                key: ValueKey('context-action-${action.id}'),
                onPressed: enabled && action.enabled
                    ? () => onExecuteCommand(action.id)
                    : null,
                child: buttonChild,
              );
            }).toList(growable: false),
          ),
        ],
      ),
    );
  }
}
