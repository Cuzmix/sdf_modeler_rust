import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

class ShellWorkspaceBar extends StatelessWidget {
  const ShellWorkspaceBar({
    super.key,
    required this.workspace,
    required this.selectionContext,
    required this.document,
    required this.history,
    required this.enabled,
    required this.onSelectWorkspace,
    required this.onOpenCommandSearch,
    required this.onUndo,
    required this.onRedo,
  });

  final AppWorkspaceSnapshot workspace;
  final AppSelectionContextSnapshot selectionContext;
  final AppDocumentSnapshot? document;
  final AppHistorySnapshot? history;
  final bool enabled;
  final ValueChanged<String> onSelectWorkspace;
  final VoidCallback onOpenCommandSearch;
  final VoidCallback onUndo;
  final VoidCallback onRedo;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final currentFileName = document?.currentFileName ?? 'Untitled Scene';
    final saveStatus = document?.hasUnsavedChanges ?? false
        ? 'Unsaved changes'
        : 'All changes saved';

    return DecoratedBox(
      decoration: ShellSurfaceStyles.commandStrip(context),
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: ShellTokens.panelPadding,
          vertical: 14,
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('SDF Modeler', style: theme.textTheme.titleLarge),
                  const SizedBox(height: 4),
                  Text(
                    '$currentFileName  •  $saveStatus',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
            Expanded(
              flex: 2,
              child: Align(
                alignment: Alignment.center,
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: SegmentedButton<String>(
                    segments: const [
                      ButtonSegment<String>(
                        value: 'blockout',
                        icon: Icon(Icons.category_outlined),
                        label: Text('Blockout'),
                      ),
                      ButtonSegment<String>(
                        value: 'sculpt',
                        icon: Icon(Icons.brush_outlined),
                        label: Text('Sculpt'),
                      ),
                      ButtonSegment<String>(
                        value: 'lookdev',
                        icon: Icon(Icons.light_mode_outlined),
                        label: Text('Lookdev'),
                      ),
                      ButtonSegment<String>(
                        value: 'review',
                        icon: Icon(Icons.fact_check_outlined),
                        label: Text('Review'),
                      ),
                    ],
                    selected: <String>{workspace.id},
                    onSelectionChanged: enabled
                        ? (selection) {
                            final nextWorkspace = selection.isEmpty
                                ? null
                                : selection.first;
                            if (nextWorkspace != null) {
                              onSelectWorkspace(nextWorkspace);
                            }
                          }
                        : null,
                  ),
                ),
              ),
            ),
            Expanded(
              child: Align(
                alignment: Alignment.centerRight,
                child: Wrap(
                  crossAxisAlignment: WrapCrossAlignment.center,
                  spacing: ShellTokens.controlGap,
                  runSpacing: ShellTokens.compactGap,
                  children: [
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 260),
                      child: Text(
                        selectionContext.headline,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.titleSmall,
                        textAlign: TextAlign.right,
                      ),
                    ),
                    Tooltip(
                      message: 'Open command search',
                      child: FilledButton.tonalIcon(
                        key: const ValueKey('shell-command-search-button'),
                        onPressed: enabled ? onOpenCommandSearch : null,
                        icon: const Icon(Icons.search),
                        label: const Text('Commands'),
                      ),
                    ),
                    IconButton(
                      tooltip: 'Undo',
                      onPressed: enabled && (history?.canUndo ?? false)
                          ? onUndo
                          : null,
                      icon: const Icon(Icons.undo),
                    ),
                    IconButton(
                      tooltip: 'Redo',
                      onPressed: enabled && (history?.canRedo ?? false)
                          ? onRedo
                          : null,
                      icon: const Icon(Icons.redo),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
