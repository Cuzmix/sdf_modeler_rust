import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class ShellToolRail extends StatelessWidget {
  const ShellToolRail({
    super.key,
    required this.currentWorkspaceId,
    required this.enabled,
    required this.onSelectWorkspace,
    required this.onExecuteCommand,
    required this.onOpenCommandSearch,
  });

  final String currentWorkspaceId;
  final bool enabled;
  final ValueChanged<String> onSelectWorkspace;
  final ValueChanged<String> onExecuteCommand;
  final VoidCallback onOpenCommandSearch;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 104,
      child: ShellPanelSurface(
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 10),
        child: SingleChildScrollView(
          key: const ValueKey('tool-rail-scrollable'),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _ToolRailButton(
                tooltip: 'Blockout workspace',
                label: 'Blockout',
                icon: Icons.category_outlined,
                selected: currentWorkspaceId == 'blockout',
                onPressed: enabled ? () => onSelectWorkspace('blockout') : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Sculpt workspace',
                label: 'Sculpt',
                icon: Icons.brush_outlined,
                selected: currentWorkspaceId == 'sculpt',
                onPressed: enabled ? () => onSelectWorkspace('sculpt') : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Lookdev workspace',
                label: 'Lookdev',
                icon: Icons.light_mode_outlined,
                selected: currentWorkspaceId == 'lookdev',
                onPressed: enabled ? () => onSelectWorkspace('lookdev') : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Review workspace',
                label: 'Review',
                icon: Icons.fact_check_outlined,
                selected: currentWorkspaceId == 'review',
                onPressed: enabled ? () => onSelectWorkspace('review') : null,
              ),
              const Padding(
                padding: EdgeInsets.symmetric(vertical: ShellTokens.controlGap),
                child: Divider(),
              ),
              _ToolRailButton(
                tooltip: 'Add sphere',
                label: 'Sphere',
                icon: Icons.circle_outlined,
                onPressed: enabled ? () => onExecuteCommand('add_sphere') : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Add box',
                label: 'Box',
                icon: Icons.crop_square_outlined,
                onPressed: enabled ? () => onExecuteCommand('add_box') : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Frame all',
                label: 'Frame',
                icon: Icons.center_focus_strong_outlined,
                onPressed: enabled ? () => onExecuteCommand('frame_all') : null,
              ),
              const SizedBox(height: ShellTokens.sectionGap),
              _ToolRailButton(
                tooltip: 'Open command search',
                label: 'Search',
                icon: Icons.search,
                onPressed: enabled ? onOpenCommandSearch : null,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ToolRailButton extends StatelessWidget {
  const _ToolRailButton({
    required this.tooltip,
    required this.label,
    required this.icon,
    this.selected = false,
    this.onPressed,
  });

  final String tooltip;
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: SizedBox(
        width: double.infinity,
        child: TextButton.icon(
          onPressed: onPressed,
          icon: Icon(icon),
          label: Text(label),
          style: TextButton.styleFrom(
            backgroundColor: selected
                ? Theme.of(context).colorScheme.primaryContainer
                : Colors.transparent,
            foregroundColor: selected
                ? Theme.of(context).colorScheme.onPrimaryContainer
                : Theme.of(context).colorScheme.onSurface,
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 14),
          ),
        ),
      ),
    );
  }
}
