import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class ShellCommandSearchPanel extends StatefulWidget {
  const ShellCommandSearchPanel({
    super.key,
    required this.commands,
    required this.onExecuteCommand,
  });

  final List<AppCommandSnapshot> commands;
  final ValueChanged<String> onExecuteCommand;

  @override
  State<ShellCommandSearchPanel> createState() => _ShellCommandSearchPanelState();
}

class _ShellCommandSearchPanelState extends State<ShellCommandSearchPanel> {
  final TextEditingController _queryController = TextEditingController();

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final normalizedQuery = _queryController.text.trim().toLowerCase();
    final filteredCommands = widget.commands.where((command) {
      if (normalizedQuery.isEmpty) {
        return true;
      }
      return command.label.toLowerCase().contains(normalizedQuery) ||
          command.category.toLowerCase().contains(normalizedQuery) ||
          command.id.toLowerCase().contains(normalizedQuery);
    }).toList(growable: false);

    return SizedBox(
      width: 720,
      height: 520,
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Command Search',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              'Search the backend-owned command registry and run the next action directly.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            TextField(
              key: const ValueKey('command-search-field'),
              controller: _queryController,
              autofocus: true,
              decoration: const InputDecoration(
                hintText: 'Search commands, workspaces, or actions',
                prefixIcon: Icon(Icons.search),
              ),
              onChanged: (_) => setState(() {}),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Expanded(
              child: filteredCommands.isEmpty
                  ? const Center(
                      child: Text('No commands match the current query.'),
                    )
                  : ListView.separated(
                      itemCount: filteredCommands.length,
                      separatorBuilder: (context, index) =>
                          const SizedBox(height: ShellTokens.compactGap),
                      itemBuilder: (context, index) {
                        final command = filteredCommands[index];
                        return ListTile(
                          key: ValueKey('command-search-item-${command.id}'),
                          tileColor: Theme.of(context).colorScheme.surfaceContainerLow,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(
                              ShellTokens.surfaceRadius,
                            ),
                          ),
                          enabled: command.enabled,
                          title: Text(command.label),
                          subtitle: Text(command.category),
                          trailing: command.shortcutLabel == null
                              ? null
                              : Text(command.shortcutLabel!),
                          onTap: command.enabled
                              ? () {
                                  widget.onExecuteCommand(command.id);
                                  Navigator.of(context).pop();
                                }
                              : null,
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
