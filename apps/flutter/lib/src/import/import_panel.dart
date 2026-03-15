import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class ImportPanel extends StatefulWidget {
  const ImportPanel({
    super.key,
    required this.importSnapshot,
    required this.enabled,
    required this.onOpenImportDialog,
    required this.onCancelImportDialog,
    required this.onSetUseAuto,
    required this.onSetResolution,
    required this.onStartImport,
    required this.onCancelImport,
  });

  final AppImportSnapshot? importSnapshot;
  final bool enabled;
  final VoidCallback onOpenImportDialog;
  final VoidCallback onCancelImportDialog;
  final ValueChanged<bool> onSetUseAuto;
  final ValueChanged<int> onSetResolution;
  final VoidCallback onStartImport;
  final VoidCallback onCancelImport;

  @override
  State<ImportPanel> createState() => _ImportPanelState();
}

class _ImportPanelState extends State<ImportPanel> {
  late final TextEditingController _resolutionController;

  @override
  void initState() {
    super.initState();
    _resolutionController = TextEditingController(
      text: widget.importSnapshot?.dialog?.resolution.toString() ?? '',
    );
  }

  @override
  void didUpdateWidget(covariant ImportPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    final nextText = widget.importSnapshot?.dialog?.resolution.toString() ?? '';
    if (_resolutionController.text != nextText) {
      _resolutionController.text = nextText;
    }
  }

  @override
  void dispose() {
    _resolutionController.dispose();
    super.dispose();
  }

  void _commitResolution() {
    final dialog = widget.importSnapshot?.dialog;
    if (dialog == null) {
      return;
    }
    final parsed = int.tryParse(_resolutionController.text) ?? dialog.resolution;
    final clamped = parsed.clamp(dialog.minResolution, dialog.maxResolution);
    _resolutionController.text = clamped.toString();
    widget.onSetResolution(clamped);
  }

  @override
  Widget build(BuildContext context) {
    final importSnapshot = widget.importSnapshot;
    if (importSnapshot == null) {
      return const Text('Import state is still loading.');
    }

    final dialog = importSnapshot.dialog;
    final status = importSnapshot.status;
    final isImporting = status.isInProgress;
    final dialogOpen = dialog != null;
    final controlsEnabled = widget.enabled && !isImporting;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (status.message != null) ...[
          _MessageCard(message: status.message!, isError: status.isError),
          const SizedBox(height: ShellTokens.controlGap),
        ],
        if (isImporting) ...[
          Text(
            'Importing ${status.filename ?? 'mesh'}',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          LinearProgressIndicator(
            key: const ValueKey('import-progress-indicator'),
            value: status.total > 0 ? status.progress / status.total : null,
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(status.phaseLabel ?? 'Voxelizing mesh...'),
          const SizedBox(height: ShellTokens.controlGap),
          OutlinedButton(
            key: const ValueKey('import-cancel-command'),
            onPressed: widget.enabled ? widget.onCancelImport : null,
            child: const Text('Cancel Import'),
          ),
        ] else if (!dialogOpen) ...[
          FilledButton.icon(
            key: const ValueKey('open-import-dialog-command'),
            onPressed: widget.enabled ? widget.onOpenImportDialog : null,
            icon: const Icon(Icons.upload_file_outlined),
            label: const Text('Import Mesh'),
          ),
        ] else ...[
          Text(dialog.filename, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: ShellTokens.controlGap),
          Text('Vertices: ${dialog.vertexCount}'),
          Text('Triangles: ${dialog.triangleCount}'),
          Text(
            'Bounds: ${dialog.boundsSize.x.toStringAsFixed(2)} x ${dialog.boundsSize.y.toStringAsFixed(2)} x ${dialog.boundsSize.z.toStringAsFixed(2)}',
          ),
          const SizedBox(height: ShellTokens.controlGap),
          IgnorePointer(
            ignoring: !controlsEnabled,
            child: RadioGroup<bool>(
              groupValue: dialog.useAuto,
              onChanged: (value) {
                if (value != null) {
                  widget.onSetUseAuto(value);
                }
              },
              child: Column(
                children: [
                  RadioListTile<bool>(
                    contentPadding: EdgeInsets.zero,
                    value: true,
                    title: Text('Auto (${dialog.autoResolution}^3)'),
                  ),
                  RadioListTile<bool>(
                    contentPadding: EdgeInsets.zero,
                    value: false,
                    title: const Text('Manual'),
                  ),
                ],
              ),
            ),
          ),
          if (!dialog.useAuto) ...[
            Row(
              children: [
                Expanded(
                  child: TextField(
                    key: const ValueKey('import-resolution-field'),
                    controller: _resolutionController,
                    enabled: controlsEnabled,
                    keyboardType: TextInputType.number,
                    inputFormatters: <TextInputFormatter>[
                      FilteringTextInputFormatter.digitsOnly,
                    ],
                    decoration: const InputDecoration(
                      labelText: 'Resolution',
                      suffixText: '^3',
                    ),
                    onSubmitted: (_) => _commitResolution(),
                  ),
                ),
                const SizedBox(width: ShellTokens.controlGap),
                FilledButton.tonal(
                  key: const ValueKey('import-apply-resolution'),
                  onPressed: controlsEnabled ? _commitResolution : null,
                  child: const Text('Apply'),
                ),
              ],
            ),
          ],
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.controlGap,
            children: [
              FilledButton(
                key: const ValueKey('start-import-command'),
                onPressed: controlsEnabled ? widget.onStartImport : null,
                child: const Text('Import as Sculpt'),
              ),
              OutlinedButton(
                key: const ValueKey('close-import-dialog-command'),
                onPressed: controlsEnabled ? widget.onCancelImportDialog : null,
                child: const Text('Cancel'),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

class _MessageCard extends StatelessWidget {
  const _MessageCard({required this.message, required this.isError});

  final String message;
  final bool isError;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: isError
            ? Theme.of(context).colorScheme.errorContainer
            : Theme.of(context).colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
      ),
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.controlGap),
        child: Text(message),
      ),
    );
  }
}
