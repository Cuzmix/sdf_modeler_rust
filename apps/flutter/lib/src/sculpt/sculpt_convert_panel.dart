import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class SculptConvertPanel extends StatefulWidget {
  const SculptConvertPanel({
    super.key,
    required this.selectedNode,
    required this.sculptConvertSnapshot,
    required this.enabled,
    required this.onOpenDialog,
    required this.onCancelDialog,
    required this.onSetMode,
    required this.onSetResolution,
    required this.onStartConvert,
  });

  final AppNodeSnapshot? selectedNode;
  final AppSculptConvertSnapshot? sculptConvertSnapshot;
  final bool enabled;
  final VoidCallback onOpenDialog;
  final VoidCallback onCancelDialog;
  final ValueChanged<String> onSetMode;
  final ValueChanged<int> onSetResolution;
  final VoidCallback onStartConvert;

  @override
  State<SculptConvertPanel> createState() => _SculptConvertPanelState();
}

class _SculptConvertPanelState extends State<SculptConvertPanel> {
  late final TextEditingController _resolutionController;

  @override
  void initState() {
    super.initState();
    _resolutionController = TextEditingController(
      text: widget.sculptConvertSnapshot?.dialog?.resolution.toString() ?? '',
    );
  }

  @override
  void didUpdateWidget(covariant SculptConvertPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    final nextText =
        widget.sculptConvertSnapshot?.dialog?.resolution.toString() ?? '';
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
    final dialog = widget.sculptConvertSnapshot?.dialog;
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
    final sculptConvertSnapshot = widget.sculptConvertSnapshot;
    if (sculptConvertSnapshot == null) {
      return const Text('Sculpt convert state is still loading.');
    }

    final dialog = sculptConvertSnapshot.dialog;
    final status = sculptConvertSnapshot.status;
    final isConverting = status.isInProgress;
    final dialogOpen = dialog != null;
    final controlsEnabled = widget.enabled && !isConverting;
    final selectedNode = widget.selectedNode;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (status.message != null) ...[
          _SculptMessageCard(message: status.message!, isError: status.isError),
          const SizedBox(height: ShellTokens.controlGap),
        ],
        if (isConverting) ...[
          Text(
            'Converting ${status.targetName ?? 'selection'}',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          LinearProgressIndicator(
            key: const ValueKey('sculpt-convert-progress-indicator'),
            value: status.total > 0 ? status.progress / status.total : null,
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(status.phaseLabel ?? 'Preparing sculpt volume...'),
        ] else if (!dialogOpen) ...[
          FilledButton.icon(
            key: const ValueKey('open-sculpt-convert-dialog-command'),
            onPressed: controlsEnabled && selectedNode != null
                ? widget.onOpenDialog
                : null,
            icon: const Icon(Icons.draw_outlined),
            label: const Text('Convert to Sculpt'),
          ),
        ] else ...[
          Text(
            'Target: ${dialog.targetName}',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          IgnorePointer(
            ignoring: !controlsEnabled,
            child: RadioGroup<String>(
              groupValue: dialog.modeId,
              onChanged: (value) {
                if (value != null) {
                  widget.onSetMode(value);
                }
              },
              child: Column(
                children: const [
                  RadioListTile<String>(
                    contentPadding: EdgeInsets.zero,
                    value: 'active_node',
                    title: Text('Bake active node'),
                  ),
                  RadioListTile<String>(
                    contentPadding: EdgeInsets.zero,
                    value: 'whole_scene',
                    title: Text('Bake whole scene'),
                  ),
                  RadioListTile<String>(
                    contentPadding: EdgeInsets.zero,
                    value: 'whole_scene_flatten',
                    title: Text('Bake whole scene + flatten'),
                  ),
                ],
              ),
            ),
          ),
          Row(
            children: [
              Expanded(
                child: TextField(
                  key: const ValueKey('sculpt-convert-resolution-field'),
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
                key: const ValueKey('sculpt-convert-apply-resolution'),
                onPressed: controlsEnabled ? _commitResolution : null,
                child: const Text('Apply'),
              ),
            ],
          ),
          if (dialog.modeId == 'whole_scene_flatten') ...[
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              'Flatten replaces the original nodes with one sculpt.',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: Theme.of(context).colorScheme.error),
            ),
          ],
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.controlGap,
            children: [
              FilledButton(
                key: const ValueKey('start-sculpt-convert-command'),
                onPressed: controlsEnabled ? widget.onStartConvert : null,
                child: const Text('Convert'),
              ),
              OutlinedButton(
                key: const ValueKey('close-sculpt-convert-dialog-command'),
                onPressed: controlsEnabled ? widget.onCancelDialog : null,
                child: const Text('Cancel'),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

class _SculptMessageCard extends StatelessWidget {
  const _SculptMessageCard({required this.message, required this.isError});

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
