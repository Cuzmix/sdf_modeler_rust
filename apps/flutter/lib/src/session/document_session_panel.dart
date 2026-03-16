import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class DocumentSessionPanel extends StatelessWidget {
  const DocumentSessionPanel({
    super.key,
    required this.document,
    required this.enabled,
    required this.onNewScene,
    required this.onOpenScene,
    required this.onSaveScene,
    required this.onSaveSceneAs,
    required this.onOpenRecentScene,
    required this.onRecoverAutosave,
    required this.onDiscardRecovery,
  });

  final AppDocumentSnapshot? document;
  final bool enabled;
  final VoidCallback onNewScene;
  final VoidCallback onOpenScene;
  final VoidCallback onSaveScene;
  final VoidCallback onSaveSceneAs;
  final ValueChanged<String> onOpenRecentScene;
  final VoidCallback onRecoverAutosave;
  final VoidCallback onDiscardRecovery;

  @override
  Widget build(BuildContext context) {
    final documentSnapshot = document;
    final recentFiles = documentSnapshot?.recentFiles ?? const <String>[];
    final currentFileLabel =
        documentSnapshot?.currentFileName ?? 'Unsaved scene';
    final dirtyLabel = documentSnapshot?.hasUnsavedChanges ?? false
        ? 'Unsaved changes'
        : 'All changes saved';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Document',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Text(currentFileLabel, style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: ShellTokens.compactGap),
        Text(
          dirtyLabel,
          style: Theme.of(context).textTheme.bodySmall,
        ),
        if ((documentSnapshot?.currentFilePath ?? '').isNotEmpty) ...[
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            documentSnapshot!.currentFilePath!,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
        if (documentSnapshot?.recoveryAvailable ?? false) ...[
          const SizedBox(height: ShellTokens.sectionGap),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(ShellTokens.controlGap),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Recovery Available',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Text(
                    documentSnapshot?.recoverySummary ??
                        'Recovered work is available.',
                  ),
                  const SizedBox(height: ShellTokens.controlGap),
                  Wrap(
                    spacing: ShellTokens.controlGap,
                    runSpacing: ShellTokens.controlGap,
                    children: [
                      FilledButton(
                        key: const ValueKey('document-recover-command'),
                        onPressed: enabled ? onRecoverAutosave : null,
                        child: const Text('Recover'),
                      ),
                      OutlinedButton(
                        key: const ValueKey('document-discard-recovery-command'),
                        onPressed: enabled ? onDiscardRecovery : null,
                        child: const Text('Discard Recovery'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
        const SizedBox(height: ShellTokens.controlGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: [
            FilledButton(
              key: const ValueKey('document-new-command'),
              onPressed: enabled ? onNewScene : null,
              child: const Text('New Scene'),
            ),
            OutlinedButton(
              key: const ValueKey('document-open-command'),
              onPressed: enabled ? onOpenScene : null,
              child: const Text('Open'),
            ),
            OutlinedButton(
              key: const ValueKey('document-save-command'),
              onPressed: enabled ? onSaveScene : null,
              child: const Text('Save'),
            ),
            OutlinedButton(
              key: const ValueKey('document-save-as-command'),
              onPressed: enabled ? onSaveSceneAs : null,
              child: const Text('Save As'),
            ),
          ],
        ),
        if (recentFiles.isNotEmpty) ...[
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Recent Files',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.controlGap,
            children: [
              for (final recentFile in recentFiles)
                ActionChip(
                  key: ValueKey('document-recent-$recentFile'),
                  label: Text(_displayNameForPath(recentFile)),
                  onPressed: enabled
                      ? () => onOpenRecentScene(recentFile)
                      : null,
                ),
            ],
          ),
        ],
      ],
    );
  }

  static String _displayNameForPath(String path) {
    final normalizedPath = path.replaceAll('\\', '/');
    final segments = normalizedPath.split('/');
    return segments.isEmpty ? path : segments.last;
  }
}
