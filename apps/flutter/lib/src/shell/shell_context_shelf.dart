import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class ShellContextShelf extends StatelessWidget {
  const ShellContextShelf({
    super.key,
    required this.workspace,
    required this.selectionContext,
    required this.shellPreferences,
    required this.commands,
    required this.sculpt,
    required this.enabled,
    required this.sceneDrawerOpen,
    required this.propertiesOpen,
    required this.onExecuteCommand,
    required this.onToggleFavoriteCommand,
    required this.onToggleSceneDrawer,
    required this.onToggleProperties,
    required this.onOpenCommandSearch,
    required this.onOpenQuickWheel,
    required this.onEditFavorites,
    required this.onSetSculptBrushMode,
    required this.onSetSculptBrushRadius,
    required this.onSetSculptBrushStrength,
    required this.onSetSculptSymmetryAxis,
  });

  final AppWorkspaceSnapshot workspace;
  final AppSelectionContextSnapshot selectionContext;
  final AppShellPreferencesSnapshot shellPreferences;
  final List<AppCommandSnapshot> commands;
  final AppSculptSnapshot sculpt;
  final bool enabled;
  final bool sceneDrawerOpen;
  final bool propertiesOpen;
  final ValueChanged<String> onExecuteCommand;
  final ValueChanged<String> onToggleFavoriteCommand;
  final VoidCallback onToggleSceneDrawer;
  final VoidCallback onToggleProperties;
  final VoidCallback onOpenCommandSearch;
  final VoidCallback onOpenQuickWheel;
  final VoidCallback onEditFavorites;
  final ValueChanged<String> onSetSculptBrushMode;
  final ValueChanged<double> onSetSculptBrushRadius;
  final ValueChanged<double> onSetSculptBrushStrength;
  final ValueChanged<String> onSetSculptSymmetryAxis;

  @override
  Widget build(BuildContext context) {
    final favoriteCommandIds =
        shellPreferences.favoriteCommandIdsByWorkspace[workspace.id] ??
            const <String>[];
    AppCommandSnapshot? findCommand(String commandId) {
      for (final command in commands) {
        if (command.id == commandId) {
          return command;
        }
      }
      return null;
    }

    final favoriteCommands = favoriteCommandIds
        .map(findCommand)
        .whereType<AppCommandSnapshot>()
        .toList(growable: false);

    return ShellPanelSurface(
      padding: const EdgeInsets.all(14),
      child: Column(
        key: const ValueKey('shell-bottom-dock'),
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.compactGap,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              _DockToggleButton(
                icon: sceneDrawerOpen
                    ? Icons.inventory_2
                    : Icons.inventory_2_outlined,
                label: 'Scene',
                selected: sceneDrawerOpen,
                onPressed: enabled ? onToggleSceneDrawer : null,
              ),
              _DockToggleButton(
                icon: propertiesOpen ? Icons.tune : Icons.tune_outlined,
                label: 'Props',
                selected: propertiesOpen,
                onPressed: enabled ? onToggleProperties : null,
              ),
              Chip(
                label: Text(workspace.label),
                avatar: const Icon(Icons.dashboard_customize_outlined, size: 18),
              ),
              Chip(
                label: Text(selectionContext.workflowStatusLabel),
                avatar: const Icon(Icons.layers_outlined, size: 18),
              ),
              if (selectionContext.selectionCount > 0)
                Chip(
                  label: Text('${selectionContext.selectionCount} selected'),
                  avatar: const Icon(Icons.select_all_outlined, size: 18),
                ),
              Text(
                selectionContext.headline,
                style: Theme.of(context).textTheme.titleSmall,
              ),
            ],
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            selectionContext.detail,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          if (workspace.id == 'sculpt')
            _SculptDock(
              sculpt: sculpt,
              favorites: favoriteCommands,
              enabled: enabled,
              onExecuteCommand: onExecuteCommand,
              onToggleFavoriteCommand: onToggleFavoriteCommand,
              onSetBrushMode: onSetSculptBrushMode,
              onSetBrushRadius: onSetSculptBrushRadius,
              onSetBrushStrength: onSetSculptBrushStrength,
              onSetSymmetryAxis: onSetSculptSymmetryAxis,
            )
          else
            _StandardDock(
              workspaceId: workspace.id,
              quickActions: selectionContext.quickActions,
              favoriteCommands: favoriteCommands,
              enabled: enabled,
              onExecuteCommand: onExecuteCommand,
              onToggleFavoriteCommand: onToggleFavoriteCommand,
            ),
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.compactGap,
            children: [
              FilledButton.tonalIcon(
                onPressed: enabled ? onOpenQuickWheel : null,
                icon: const Icon(Icons.control_camera_outlined),
                label: const Text('Quick Wheel'),
              ),
              OutlinedButton.icon(
                onPressed: enabled ? onOpenCommandSearch : null,
                icon: const Icon(Icons.search),
                label: const Text('Search'),
              ),
              OutlinedButton.icon(
                onPressed: enabled ? onEditFavorites : null,
                icon: const Icon(Icons.star_outline),
                label: const Text('Edit Favorites'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StandardDock extends StatelessWidget {
  const _StandardDock({
    required this.workspaceId,
    required this.quickActions,
    required this.favoriteCommands,
    required this.enabled,
    required this.onExecuteCommand,
    required this.onToggleFavoriteCommand,
  });

  final String workspaceId;
  final List<AppQuickActionSnapshot> quickActions;
  final List<AppCommandSnapshot> favoriteCommands;
  final bool enabled;
  final ValueChanged<String> onExecuteCommand;
  final ValueChanged<String> onToggleFavoriteCommand;

  @override
  Widget build(BuildContext context) {
    final createCommandIds = switch (workspaceId) {
      'blockout' => const <String>[
          'add_sphere',
          'add_box',
          'add_cylinder',
          'add_torus',
        ],
      'lookdev' => const <String>['create_light_point'],
      'review' => const <String>['frame_all', 'toggle_projection'],
      _ => const <String>[],
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (createCommandIds.isNotEmpty) ...[
          Text('Create', style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: ShellTokens.compactGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.compactGap,
            children: [
              for (final commandId in createCommandIds)
                _ActionButton(
                  commandId: commandId,
                  label: _commandLabel(commandId),
                  highlighted: true,
                  enabled: enabled,
                  favorite: favoriteCommands.any((command) => command.id == commandId),
                  onExecute: onExecuteCommand,
                  onToggleFavorite: onToggleFavoriteCommand,
                ),
            ],
          ),
          const SizedBox(height: ShellTokens.controlGap),
        ],
        if (quickActions.isNotEmpty) ...[
          Text('Selection', style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: ShellTokens.compactGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.compactGap,
            children: [
              for (final action in quickActions)
                _ActionButton(
                  commandId: action.id,
                  label: action.label,
                  highlighted: action.prominent,
                  enabled: enabled && action.enabled,
                  favorite: favoriteCommands.any((command) => command.id == action.id),
                  onExecute: onExecuteCommand,
                  onToggleFavorite: onToggleFavoriteCommand,
                ),
            ],
          ),
          const SizedBox(height: ShellTokens.controlGap),
        ],
        Text('Favorites', style: Theme.of(context).textTheme.labelLarge),
        const SizedBox(height: ShellTokens.compactGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.compactGap,
          children: favoriteCommands
              .map(
                (command) => _ActionButton(
                  commandId: command.id,
                  label: command.label,
                  highlighted: false,
                  enabled: enabled && command.enabled,
                  favorite: true,
                  onExecute: onExecuteCommand,
                  onToggleFavorite: onToggleFavoriteCommand,
                ),
              )
              .toList(growable: false),
        ),
      ],
    );
  }

  static String _commandLabel(String commandId) {
    return switch (commandId) {
      'add_sphere' => 'Sphere',
      'add_box' => 'Box',
      'add_cylinder' => 'Cylinder',
      'add_torus' => 'Torus',
      'create_light_point' => 'Point Light',
      'frame_all' => 'Frame All',
      'toggle_projection' => 'Projection',
      _ => commandId,
    };
  }
}

class _SculptDock extends StatelessWidget {
  const _SculptDock({
    required this.sculpt,
    required this.favorites,
    required this.enabled,
    required this.onExecuteCommand,
    required this.onToggleFavoriteCommand,
    required this.onSetBrushMode,
    required this.onSetBrushRadius,
    required this.onSetBrushStrength,
    required this.onSetSymmetryAxis,
  });

  final AppSculptSnapshot sculpt;
  final List<AppCommandSnapshot> favorites;
  final bool enabled;
  final ValueChanged<String> onExecuteCommand;
  final ValueChanged<String> onToggleFavoriteCommand;
  final ValueChanged<String> onSetBrushMode;
  final ValueChanged<double> onSetBrushRadius;
  final ValueChanged<double> onSetBrushStrength;
  final ValueChanged<String> onSetSymmetryAxis;

  @override
  Widget build(BuildContext context) {
    final session = sculpt.session;
    if (session == null) {
      return _StandardDock(
        workspaceId: 'sculpt',
        quickActions: const <AppQuickActionSnapshot>[],
        favoriteCommands: favorites,
        enabled: enabled,
        onExecuteCommand: onExecuteCommand,
        onToggleFavoriteCommand: onToggleFavoriteCommand,
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.compactGap,
          children: [
            for (final mode in const [
              ('add', 'Add'),
              ('carve', 'Carve'),
              ('smooth', 'Smooth'),
              ('flatten', 'Flatten'),
              ('inflate', 'Inflate'),
              ('grab', 'Grab'),
            ])
              ChoiceChip(
                label: Text(mode.$2),
                selected: session.brushModeId == mode.$1,
                onSelected: enabled ? (_) => onSetBrushMode(mode.$1) : null,
              ),
          ],
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Row(
          children: [
            Expanded(
              child: _BrushSlider(
                label: 'Radius',
                value: session.brushRadius,
                min: 0.05,
                max: 2.0,
                onChanged: enabled ? onSetBrushRadius : null,
              ),
            ),
            const SizedBox(width: ShellTokens.controlGap),
            Expanded(
              child: _BrushSlider(
                label: 'Strength',
                value: session.brushStrength,
                min: 0.01,
                max: session.brushModeId == 'grab' ? 3.0 : 1.0,
                onChanged: enabled ? onSetBrushStrength : null,
              ),
            ),
          ],
        ),
        const SizedBox(height: ShellTokens.compactGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.compactGap,
          children: [
            for (final axis in const [
              ('off', 'Sym Off'),
              ('x', 'X'),
              ('y', 'Y'),
              ('z', 'Z'),
            ])
              ChoiceChip(
                label: Text(axis.$2),
                selected: session.symmetryAxisId == axis.$1,
                onSelected: enabled ? (_) => onSetSymmetryAxis(axis.$1) : null,
              ),
            for (final command in favorites)
              _ActionButton(
                commandId: command.id,
                label: command.label,
                highlighted: false,
                enabled: enabled && command.enabled,
                favorite: true,
                onExecute: onExecuteCommand,
                onToggleFavorite: onToggleFavoriteCommand,
              ),
          ],
        ),
      ],
    );
  }
}

class _BrushSlider extends StatelessWidget {
  const _BrushSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    this.onChanged,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label ${value.toStringAsFixed(2)}'),
        Slider(
          value: value.clamp(min, max),
          min: min,
          max: max,
          onChanged: onChanged,
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  const _ActionButton({
    required this.commandId,
    required this.label,
    required this.highlighted,
    required this.enabled,
    required this.favorite,
    required this.onExecute,
    required this.onToggleFavorite,
  });

  final String commandId;
  final String label;
  final bool highlighted;
  final bool enabled;
  final bool favorite;
  final ValueChanged<String> onExecute;
  final ValueChanged<String> onToggleFavorite;

  @override
  Widget build(BuildContext context) {
    final button = highlighted
        ? FilledButton(
            key: ValueKey('context-action-$commandId'),
            onPressed: enabled ? () => onExecute(commandId) : null,
            child: Text(label),
          )
        : OutlinedButton(
            key: ValueKey('context-action-$commandId'),
            onPressed: enabled ? () => onExecute(commandId) : null,
            child: Text(label),
          );

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        button,
        IconButton(
          tooltip: favorite ? 'Remove favorite' : 'Pin favorite',
          onPressed: () => onToggleFavorite(commandId),
          icon: Icon(favorite ? Icons.star : Icons.star_border),
        ),
      ],
    );
  }
}

class _DockToggleButton extends StatelessWidget {
  const _DockToggleButton({
    required this.icon,
    required this.label,
    required this.selected,
    this.onPressed,
  });

  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon),
      label: Text(label),
      style: OutlinedButton.styleFrom(
        backgroundColor: selected
            ? Theme.of(context).colorScheme.primaryContainer
            : null,
      ),
    );
  }
}
