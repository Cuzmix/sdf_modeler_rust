import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class ExportPanel extends StatefulWidget {
  const ExportPanel({
    super.key,
    required this.export,
    required this.enabled,
    required this.onSetResolution,
    required this.onSetAdaptive,
    required this.onStartExport,
    required this.onCancelExport,
  });

  final AppExportSnapshot? export;
  final bool enabled;
  final ValueChanged<int> onSetResolution;
  final ValueChanged<bool> onSetAdaptive;
  final VoidCallback onStartExport;
  final VoidCallback onCancelExport;

  @override
  State<ExportPanel> createState() => _ExportPanelState();
}

class _ExportPanelState extends State<ExportPanel> {
  late final TextEditingController _resolutionController;

  @override
  void initState() {
    super.initState();
    _resolutionController = TextEditingController(
      text: widget.export?.resolution.toString() ?? '',
    );
  }

  @override
  void didUpdateWidget(covariant ExportPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    final nextResolution = widget.export?.resolution.toString() ?? '';
    if (_resolutionController.text != nextResolution) {
      _resolutionController.text = nextResolution;
    }
  }

  @override
  void dispose() {
    _resolutionController.dispose();
    super.dispose();
  }

  void _commitResolution() {
    final export = widget.export;
    if (export == null) {
      return;
    }

    final parsedValue = int.tryParse(_resolutionController.text);
    final clampedValue = (parsedValue ?? export.resolution).clamp(
      export.minResolution,
      export.maxResolution,
    );
    _resolutionController.text = clampedValue.toString();
    widget.onSetResolution(clampedValue);
  }

  @override
  Widget build(BuildContext context) {
    final export = widget.export;
    if (export == null) {
      return const Text('Export settings are still loading.');
    }

    final status = export.status;
    final isRunning = status.isInProgress;
    final controlsEnabled = widget.enabled && !isRunning;
    final progressValue = status.total > 0 ? status.progress / status.total : null;
    final estimate = _ExportEstimate.fromResolution(export.resolution);
    final messageColor = status.isError
        ? Theme.of(context).colorScheme.errorContainer
        : Theme.of(context).colorScheme.secondaryContainer;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'OBJ, STL, PLY, GLB, and USDA are supported.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: [
            for (final preset in export.presets)
              ActionChip(
                key: ValueKey('export-preset-${preset.name}'),
                label: Text('${preset.name} ${preset.resolution}^3'),
                onPressed: controlsEnabled
                    ? () {
                        _resolutionController.text = preset.resolution.toString();
                        widget.onSetResolution(preset.resolution);
                      }
                    : null,
              ),
          ],
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Row(
          children: [
            Expanded(
              child: TextField(
                key: const ValueKey('export-resolution-field'),
                controller: _resolutionController,
                enabled: controlsEnabled,
                keyboardType: TextInputType.number,
                inputFormatters: <TextInputFormatter>[
                  FilteringTextInputFormatter.digitsOnly,
                ],
                decoration: const InputDecoration(
                  labelText: 'Resolution',
                  helperText: 'Voxel grid edge length',
                  suffixText: '^3',
                ),
                onSubmitted: (_) => _commitResolution(),
              ),
            ),
            const SizedBox(width: ShellTokens.controlGap),
            FilledButton.tonal(
              key: const ValueKey('export-apply-resolution'),
              onPressed: controlsEnabled ? _commitResolution : null,
              child: const Text('Apply'),
            ),
          ],
        ),
        const SizedBox(height: ShellTokens.compactGap),
        Text(
          '${estimate.voxelLabel} voxels (${estimate.memoryMb.toStringAsFixed(1)} MB)',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: ShellTokens.compactGap),
        Text(
          '~${estimate.triangleLabel} triangles, ~${estimate.vertexLabel} vertices',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        if (export.resolution > 512) ...[
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            'Warning: export may take a long time or run out of memory.',
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: Theme.of(context).colorScheme.error),
          ),
        ] else if (export.resolution > 256) ...[
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            'High resolution exports take longer.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
        const SizedBox(height: ShellTokens.controlGap),
        SwitchListTile.adaptive(
          key: const ValueKey('export-adaptive-toggle'),
          contentPadding: EdgeInsets.zero,
          value: export.adaptive,
          onChanged: controlsEnabled ? widget.onSetAdaptive : null,
          title: const Text('Adaptive Sampling'),
          subtitle: const Text('Skip empty regions during marching cubes.'),
        ),
        if (status.message != null) ...[
          const SizedBox(height: ShellTokens.controlGap),
          DecoratedBox(
            decoration: BoxDecoration(
              color: messageColor,
              borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
            ),
            child: Padding(
              padding: const EdgeInsets.all(ShellTokens.controlGap),
              child: Text(status.message!),
            ),
          ),
        ],
        const SizedBox(height: ShellTokens.controlGap),
        if (isRunning) ...[
          Text(
            'Exporting ${status.targetFileName ?? 'mesh'}',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          if ((status.formatLabel ?? '').isNotEmpty) ...[
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              '${status.formatLabel} at ${status.resolution}^3',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
          const SizedBox(height: ShellTokens.controlGap),
          LinearProgressIndicator(
            key: const ValueKey('export-progress-indicator'),
            value: progressValue,
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(status.phaseLabel ?? 'Preparing export...'),
          const SizedBox(height: ShellTokens.controlGap),
          OutlinedButton(
            key: const ValueKey('export-cancel-command'),
            onPressed: widget.enabled ? widget.onCancelExport : null,
            child: const Text('Cancel Export'),
          ),
        ] else
          FilledButton.icon(
            key: const ValueKey('export-start-command'),
            onPressed: widget.enabled ? widget.onStartExport : null,
            icon: const Icon(Icons.file_upload_outlined),
            label: const Text('Export Mesh'),
          ),
      ],
    );
  }
}

class _ExportEstimate {
  const _ExportEstimate({
    required this.voxelLabel,
    required this.triangleLabel,
    required this.vertexLabel,
    required this.memoryMb,
  });

  final String voxelLabel;
  final String triangleLabel;
  final String vertexLabel;
  final double memoryMb;

  factory _ExportEstimate.fromResolution(int resolution) {
    final voxels = resolution * resolution * resolution;
    final estimatedTriangles = 12 * resolution * resolution;
    final estimatedVertices = estimatedTriangles * 3 ~/ 2;
    final memoryMb = (voxels * 4.0) / (1024.0 * 1024.0);

    return _ExportEstimate(
      voxelLabel: _formatLargeCount(voxels),
      triangleLabel: _formatLargeCount(estimatedTriangles),
      vertexLabel: _formatLargeCount(estimatedVertices),
      memoryMb: memoryMb,
    );
  }

  static String _formatLargeCount(int value) {
    if (value >= 1000000) {
      return '${(value / 1000000).toStringAsFixed(1)}M';
    }
    if (value >= 1000) {
      return '${(value / 1000).round()}K';
    }
    return value.toString();
  }
}
