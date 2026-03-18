import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class ShellFavoriteCommandsEditor extends StatefulWidget {
  const ShellFavoriteCommandsEditor({
    super.key,
    required this.workspace,
    required this.commands,
    required this.favoriteCommandIds,
    required this.onApply,
  });

  final AppWorkspaceSnapshot workspace;
  final List<AppCommandSnapshot> commands;
  final List<String> favoriteCommandIds;
  final ValueChanged<List<String>> onApply;

  @override
  State<ShellFavoriteCommandsEditor> createState() =>
      _ShellFavoriteCommandsEditorState();
}

class _ShellFavoriteCommandsEditorState
    extends State<ShellFavoriteCommandsEditor> {
  late final List<String> _favoriteCommandIds = List<String>.from(
    widget.favoriteCommandIds,
  );

  List<AppCommandSnapshot> get _workspaceCommands => widget.commands
      .where((command) => command.workspaceIds.contains(widget.workspace.id))
      .toList(growable: false);

  @override
  Widget build(BuildContext context) {
    final workspaceCommands = _workspaceCommands;
    AppCommandSnapshot? findCommand(String commandId) {
      for (final command in workspaceCommands) {
        if (command.id == commandId) {
          return command;
        }
      }
      return null;
    }

    final favoriteCommands = _favoriteCommandIds
        .map(findCommand)
        .whereType<AppCommandSnapshot>()
        .toList(growable: false);
    final availableCommands = workspaceCommands
        .where((command) => !_favoriteCommandIds.contains(command.id))
        .toList(growable: false);

    return SizedBox(
      width: 760,
      height: 620,
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '${widget.workspace.label} Favorites',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              'Pin, remove, and reorder the actions that stay in the bottom dock and quick wheel.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Text('Pinned', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: ShellTokens.compactGap),
            Expanded(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerLowest,
                  borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
                  border: Border.all(
                    color: Theme.of(context).colorScheme.outlineVariant,
                  ),
                ),
                child: favoriteCommands.isEmpty
                    ? const Center(
                        child: Text('No favorites pinned for this workspace.'),
                      )
                    : ReorderableListView.builder(
                        padding: const EdgeInsets.all(12),
                        itemCount: favoriteCommands.length,
                        onReorder: _reorderFavorite,
                        itemBuilder: (context, index) {
                          final command = favoriteCommands[index];
                          return Card(
                            key: ValueKey('favorite-editor-favorite-${command.id}'),
                            child: ListTile(
                              title: Text(command.label),
                              subtitle: Text(command.category),
                              leading: const Icon(Icons.drag_indicator),
                              trailing: IconButton(
                                tooltip: 'Remove favorite',
                                onPressed: () => setState(() {
                                  _favoriteCommandIds.remove(command.id);
                                }),
                                icon: const Icon(Icons.star),
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Text('Available', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: ShellTokens.compactGap),
            Expanded(
              child: ListView.separated(
                itemCount: availableCommands.length,
                separatorBuilder: (context, index) =>
                    const SizedBox(height: ShellTokens.compactGap),
                itemBuilder: (context, index) {
                  final command = availableCommands[index];
                  return ListTile(
                    key: ValueKey('favorite-editor-available-${command.id}'),
                    tileColor: Theme.of(context).colorScheme.surfaceContainerLow,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(
                        ShellTokens.surfaceRadius,
                      ),
                    ),
                    title: Text(command.label),
                    subtitle: Text(command.category),
                    trailing: IconButton(
                      tooltip: 'Pin favorite',
                      onPressed: () => setState(() {
                        _favoriteCommandIds.add(command.id);
                      }),
                      icon: const Icon(Icons.star_border),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: ShellTokens.controlGap),
                FilledButton(
                  key: const ValueKey('favorite-editor-save'),
                  onPressed: () {
                    widget.onApply(_favoriteCommandIds);
                    Navigator.of(context).pop();
                  },
                  child: const Text('Apply'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _reorderFavorite(int oldIndex, int newIndex) {
    setState(() {
      if (oldIndex < newIndex) {
        newIndex -= 1;
      }
      final commandId = _favoriteCommandIds.removeAt(oldIndex);
      _favoriteCommandIds.insert(newIndex, commandId);
    });
  }
}
