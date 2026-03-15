import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class SculptSessionPanel extends StatefulWidget {
  const SculptSessionPanel({
    super.key,
    required this.selectedNode,
    required this.sculptSnapshot,
    required this.enabled,
    required this.onCreateSculpt,
    required this.onResumeSelected,
    required this.onStopSculpting,
    required this.onSetBrushMode,
    required this.onSetBrushRadius,
    required this.onSetBrushStrength,
    required this.onSetSymmetryAxis,
    required this.onSetResolution,
  });

  final AppNodeSnapshot? selectedNode;
  final AppSculptSnapshot? sculptSnapshot;
  final bool enabled;
  final VoidCallback onCreateSculpt;
  final VoidCallback onResumeSelected;
  final VoidCallback onStopSculpting;
  final ValueChanged<String> onSetBrushMode;
  final ValueChanged<double> onSetBrushRadius;
  final ValueChanged<double> onSetBrushStrength;
  final ValueChanged<String> onSetSymmetryAxis;
  final ValueChanged<int> onSetResolution;

  @override
  State<SculptSessionPanel> createState() => _SculptSessionPanelState();
}

class _SculptSessionPanelState extends State<SculptSessionPanel> {
  late final TextEditingController _resolutionController;

  @override
  void initState() {
    super.initState();
    _resolutionController = TextEditingController(
      text: widget.sculptSnapshot?.selected?.desiredResolution.toString() ?? '',
    );
  }

  @override
  void didUpdateWidget(covariant SculptSessionPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    final nextText =
        widget.sculptSnapshot?.selected?.desiredResolution.toString() ?? '';
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
    final sculpt = widget.sculptSnapshot?.selected;
    final snapshot = widget.sculptSnapshot;
    if (sculpt == null || snapshot == null) {
      return;
    }

    final parsedValue =
        int.tryParse(_resolutionController.text) ?? sculpt.desiredResolution;
    final clampedValue = parsedValue.clamp(16, snapshot.maxResolution);
    _resolutionController.text = clampedValue.toString();
    widget.onSetResolution(clampedValue);
  }

  @override
  Widget build(BuildContext context) {
    final sculptSnapshot = widget.sculptSnapshot;
    if (sculptSnapshot == null) {
      return const Text('Sculpt controls are still loading.');
    }

    final selectedSculpt = sculptSnapshot.selected;
    final session = sculptSnapshot.session;
    final selectedNode = widget.selectedNode;
    final controlsEnabled = widget.enabled;
    final hasSelectedSculpt = selectedSculpt != null;
    final hasSession = session != null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (selectedSculpt == null) ...[
          Text(
            selectedNode == null
                ? 'Select a node to start sculpting.'
                : '${selectedNode.name} is not a sculpt node yet.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          FilledButton.icon(
            key: const ValueKey('create-sculpt-workflow-command'),
            onPressed: controlsEnabled && selectedNode != null
                ? widget.onCreateSculpt
                : null,
            icon: const Icon(Icons.draw_outlined),
            label: const Text('Create Sculpt From Selection'),
          ),
        ] else ...[
          DecoratedBox(
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.secondaryContainer,
              borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
            ),
            child: Padding(
              padding: const EdgeInsets.all(ShellTokens.controlGap),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Selected Sculpt: ${selectedSculpt.nodeName}',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Text(
                    'Current ${selectedSculpt.currentResolution}^3, target ${selectedSculpt.desiredResolution}^3',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  if (hasSession) ...[
                    const SizedBox(height: ShellTokens.compactGap),
                    Text(
                      'Active session: ${session.nodeName}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: ShellTokens.controlGap),
          Wrap(
            spacing: ShellTokens.controlGap,
            runSpacing: ShellTokens.controlGap,
            children: [
              if (sculptSnapshot.canResumeSelected)
                FilledButton.icon(
                  key: const ValueKey('resume-sculpt-command'),
                  onPressed: controlsEnabled ? widget.onResumeSelected : null,
                  icon: const Icon(Icons.play_arrow_outlined),
                  label: const Text('Resume Sculpting'),
                ),
              if (sculptSnapshot.canStop)
                OutlinedButton.icon(
                  key: const ValueKey('stop-sculpt-command'),
                  onPressed: controlsEnabled ? widget.onStopSculpting : null,
                  icon: const Icon(Icons.pause_circle_outline),
                  label: const Text('Stop Sculpting'),
                ),
            ],
          ),
          const SizedBox(height: ShellTokens.controlGap),
          Row(
            children: [
              Expanded(
                child: TextField(
                  key: const ValueKey('selected-sculpt-resolution-field'),
                  controller: _resolutionController,
                  enabled: controlsEnabled && hasSelectedSculpt,
                  keyboardType: TextInputType.number,
                  inputFormatters: <TextInputFormatter>[
                    FilteringTextInputFormatter.digitsOnly,
                  ],
                  decoration: const InputDecoration(
                    labelText: 'Target Resolution',
                    suffixText: '^3',
                  ),
                  onSubmitted: (_) => _commitResolution(),
                ),
              ),
              const SizedBox(width: ShellTokens.controlGap),
              FilledButton.tonal(
                key: const ValueKey('selected-sculpt-apply-resolution'),
                onPressed: controlsEnabled && hasSelectedSculpt
                    ? _commitResolution
                    : null,
                child: const Text('Apply'),
              ),
            ],
          ),
          const SizedBox(height: ShellTokens.controlGap),
          if (!hasSession)
            Text(
              'Resume the selected sculpt session to edit brush controls.',
              style: Theme.of(context).textTheme.bodySmall,
            )
          else ...[
            Text(
              'Brush Mode',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Wrap(
              spacing: ShellTokens.controlGap,
              runSpacing: ShellTokens.controlGap,
              children: [
                for (final brush in _brushModes)
                  ChoiceChip(
                    key: ValueKey('sculpt-brush-mode-${brush.id}'),
                    label: Text(brush.label),
                    selected: session.brushModeId == brush.id,
                    onSelected: controlsEnabled
                        ? (_) => widget.onSetBrushMode(brush.id)
                        : null,
                  ),
              ],
            ),
            const SizedBox(height: ShellTokens.controlGap),
            _StepperRow(
              label: 'Brush Radius',
              value: session.brushRadius,
              fractionDigits: 2,
              onDecrease: controlsEnabled
                  ? () => widget.onSetBrushRadius(
                      (session.brushRadius - 0.05).clamp(0.05, 2.0),
                    )
                  : null,
              onIncrease: controlsEnabled
                  ? () => widget.onSetBrushRadius(
                      (session.brushRadius + 0.05).clamp(0.05, 2.0),
                    )
                  : null,
              decreaseKey: const ValueKey('sculpt-radius-decrease'),
              increaseKey: const ValueKey('sculpt-radius-increase'),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            _StepperRow(
              label: 'Brush Strength',
              value: session.brushStrength,
              fractionDigits: 2,
              onDecrease: controlsEnabled
                  ? () => widget.onSetBrushStrength(
                      _nextStrengthValue(session, -1),
                    )
                  : null,
              onIncrease: controlsEnabled
                  ? () => widget.onSetBrushStrength(
                      _nextStrengthValue(session, 1),
                    )
                  : null,
              decreaseKey: const ValueKey('sculpt-strength-decrease'),
              increaseKey: const ValueKey('sculpt-strength-increase'),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Text(
              'Symmetry',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Wrap(
              spacing: ShellTokens.controlGap,
              runSpacing: ShellTokens.controlGap,
              children: [
                for (final axis in _symmetryAxes)
                  ChoiceChip(
                    key: ValueKey('sculpt-symmetry-${axis.id}'),
                    label: Text(axis.label),
                    selected: session.symmetryAxisId == axis.id,
                    onSelected: controlsEnabled
                        ? (_) => widget.onSetSymmetryAxis(axis.id)
                        : null,
                  ),
              ],
            ),
          ],
        ],
      ],
    );
  }

  double _nextStrengthValue(AppSculptSessionSnapshot session, int direction) {
    final step = session.brushModeId == 'grab' ? 0.25 : 0.05;
    final next = session.brushStrength + (step * direction);
    final max = session.brushModeId == 'grab' ? 3.0 : 0.5;
    final min = session.brushModeId == 'grab' ? 0.1 : 0.01;
    return next.clamp(min, max);
  }
}

class _StepperRow extends StatelessWidget {
  const _StepperRow({
    required this.label,
    required this.value,
    required this.fractionDigits,
    required this.onDecrease,
    required this.onIncrease,
    required this.decreaseKey,
    required this.increaseKey,
  });

  final String label;
  final double value;
  final int fractionDigits;
  final VoidCallback? onDecrease;
  final VoidCallback? onIncrease;
  final Key decreaseKey;
  final Key increaseKey;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            '$label: ${value.toStringAsFixed(fractionDigits)}',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
        IconButton.filledTonal(
          key: decreaseKey,
          onPressed: onDecrease,
          icon: const Icon(Icons.remove),
        ),
        const SizedBox(width: ShellTokens.compactGap),
        IconButton.filledTonal(
          key: increaseKey,
          onPressed: onIncrease,
          icon: const Icon(Icons.add),
        ),
      ],
    );
  }
}

class _OptionChipData {
  const _OptionChipData({required this.id, required this.label});

  final String id;
  final String label;
}

const List<_OptionChipData> _brushModes = <_OptionChipData>[
  _OptionChipData(id: 'add', label: 'Add'),
  _OptionChipData(id: 'carve', label: 'Carve'),
  _OptionChipData(id: 'smooth', label: 'Smooth'),
  _OptionChipData(id: 'flatten', label: 'Flatten'),
  _OptionChipData(id: 'inflate', label: 'Inflate'),
  _OptionChipData(id: 'grab', label: 'Grab'),
];

const List<_OptionChipData> _symmetryAxes = <_OptionChipData>[
  _OptionChipData(id: 'off', label: 'Off'),
  _OptionChipData(id: 'x', label: 'X'),
  _OptionChipData(id: 'y', label: 'Y'),
  _OptionChipData(id: 'z', label: 'Z'),
];
