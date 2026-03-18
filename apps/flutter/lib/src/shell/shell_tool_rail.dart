import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class ShellToolRail extends StatelessWidget {
  const ShellToolRail({
    super.key,
    required this.currentWorkspaceId,
    required this.enabled,
    required this.onSelectWorkspace,
    required this.sceneDrawerOpen,
    required this.onToggleSceneDrawer,
    required this.onToggleLeadingEdgeSide,
    required this.leadingEdgeSide,
  });

  final String currentWorkspaceId;
  final bool enabled;
  final ValueChanged<String> onSelectWorkspace;
  final bool sceneDrawerOpen;
  final VoidCallback onToggleSceneDrawer;
  final VoidCallback onToggleLeadingEdgeSide;
  final String leadingEdgeSide;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      child: ShellPanelSurface(
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 8),
        child: SingleChildScrollView(
          key: const ValueKey('tool-rail-scrollable'),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _ToolRailButton(
                tooltip: sceneDrawerOpen ? 'Hide scene drawer' : 'Open scene drawer',
                label: 'Scene',
                icon: sceneDrawerOpen ? Icons.inventory_2 : Icons.inventory_2_outlined,
                selected: sceneDrawerOpen,
                onPressed: enabled ? onToggleSceneDrawer : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ToolRailButton(
                tooltip: 'Blockout workspace',
                label: 'Block',
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
                tooltip: leadingEdgeSide == 'left'
                    ? 'Mirror shell to the right edge'
                    : 'Mirror shell to the left edge',
                label: leadingEdgeSide == 'left' ? 'Right' : 'Left',
                icon: Icons.swap_horiz,
                onPressed: enabled ? onToggleLeadingEdgeSide : null,
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
        child: TextButton(
          onPressed: onPressed,
          style: TextButton.styleFrom(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
            ),
            backgroundColor: selected
                ? Theme.of(context).colorScheme.primaryContainer
                : Colors.transparent,
            foregroundColor: selected
                ? Theme.of(context).colorScheme.onPrimaryContainer
                : Theme.of(context).colorScheme.onSurface,
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 10),
          ),
          child: Column(
            children: [
              Icon(icon),
              const SizedBox(height: 6),
              Text(
                label,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelSmall,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
