import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

class ViewportToolOverlay extends StatelessWidget {
  const ViewportToolOverlay({
    super.key,
    required this.tool,
    required this.hasSelection,
    required this.enabled,
    required this.onSetManipulatorMode,
    required this.onToggleManipulatorSpace,
    required this.onResetManipulatorPivot,
    required this.onNudgeManipulatorAxis,
    required this.onNudgeManipulatorPivot,
  });

  final AppToolSnapshot? tool;
  final bool hasSelection;
  final bool enabled;
  final ValueChanged<String> onSetManipulatorMode;
  final VoidCallback onToggleManipulatorSpace;
  final VoidCallback onResetManipulatorPivot;
  final void Function(String modeId, String axisId, double direction)
  onNudgeManipulatorAxis;
  final void Function(String axisId, double direction) onNudgeManipulatorPivot;

  @override
  Widget build(BuildContext context) {
    final currentTool = tool;
    if (currentTool == null || !hasSelection || !currentTool.manipulatorVisible) {
      return const SizedBox.shrink();
    }

    final shellPalette = context.shellPalette;

    return Padding(
      padding: const EdgeInsets.all(ShellTokens.overlayPadding),
      child: Align(
        alignment: Alignment.bottomLeft,
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 360),
          child: DecoratedBox(
            decoration: ShellSurfaceStyles.overlayPanel(
              context,
              accentColor: shellPalette.infoAccent,
            ),
            child: Padding(
              padding: const EdgeInsets.all(ShellTokens.overlayChipHorizontalPadding),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Viewport Tools',
                    style: Theme.of(context).textTheme.labelLarge?.copyWith(
                      color: shellPalette.overlayText,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Text(
                    '${currentTool.manipulatorModeLabel} · ${currentTool.manipulatorSpaceLabel} · Pivot ${_formatVec3(currentTool.pivotOffset)}',
                    style: Theme.of(
                      context,
                    ).textTheme.bodySmall?.copyWith(
                      color: shellPalette.overlayMutedText,
                    ),
                  ),
                  const SizedBox(height: ShellTokens.controlGap),
                  Wrap(
                    spacing: ShellTokens.compactGap,
                    runSpacing: ShellTokens.compactGap,
                    children: [
                      _ModeButton(
                        key: const ValueKey('viewport-tool-mode-translate'),
                        label: 'Move',
                        selected:
                            currentTool.manipulatorModeId == 'translate',
                        enabled: enabled,
                        onPressed: () => onSetManipulatorMode('translate'),
                      ),
                      _ModeButton(
                        key: const ValueKey('viewport-tool-mode-rotate'),
                        label: 'Rotate',
                        selected: currentTool.manipulatorModeId == 'rotate',
                        enabled: enabled,
                        onPressed: () => onSetManipulatorMode('rotate'),
                      ),
                      _ModeButton(
                        key: const ValueKey('viewport-tool-mode-scale'),
                        label: 'Scale',
                        selected: currentTool.manipulatorModeId == 'scale',
                        enabled: enabled,
                        onPressed: () => onSetManipulatorMode('scale'),
                      ),
                    ],
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Wrap(
                    spacing: ShellTokens.compactGap,
                    runSpacing: ShellTokens.compactGap,
                    children: [
                      OutlinedButton(
                        key: const ValueKey('viewport-tool-space-toggle'),
                        onPressed: enabled ? onToggleManipulatorSpace : null,
                        child: Text(currentTool.manipulatorSpaceLabel),
                      ),
                      OutlinedButton(
                        key: const ValueKey('viewport-tool-pivot-reset'),
                        onPressed: enabled && currentTool.canResetPivot
                            ? onResetManipulatorPivot
                            : null,
                        child: const Text('Reset Pivot'),
                      ),
                    ],
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Text(
                    '${currentTool.manipulatorModeLabel} Nudges',
                    style: Theme.of(context).textTheme.labelMedium?.copyWith(
                      color: shellPalette.overlayText,
                    ),
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Wrap(
                    spacing: ShellTokens.compactGap,
                    runSpacing: ShellTokens.compactGap,
                    children: [
                      for (final axis in const ['x', 'y', 'z']) ...[
                        _AxisButton(
                          key: ValueKey('viewport-tool-nudge-$axis-negative'),
                          label: '${axis.toUpperCase()}-',
                          enabled: enabled,
                          onPressed: () => onNudgeManipulatorAxis(
                            currentTool.manipulatorModeId,
                            axis,
                            -1.0,
                          ),
                        ),
                        _AxisButton(
                          key: ValueKey('viewport-tool-nudge-$axis-positive'),
                          label: '${axis.toUpperCase()}+',
                          enabled: enabled,
                          onPressed: () => onNudgeManipulatorAxis(
                            currentTool.manipulatorModeId,
                            axis,
                            1.0,
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Text(
                    'Pivot Nudges',
                    style: Theme.of(context).textTheme.labelMedium?.copyWith(
                      color: shellPalette.overlayText,
                    ),
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Wrap(
                    spacing: ShellTokens.compactGap,
                    runSpacing: ShellTokens.compactGap,
                    children: [
                      for (final axis in const ['x', 'y', 'z'])
                        _AxisButton(
                          key: ValueKey('viewport-tool-pivot-$axis-positive'),
                          label: 'Pivot ${axis.toUpperCase()}+',
                          enabled: enabled,
                          onPressed: () => onNudgeManipulatorPivot(axis, 1.0),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  static String _formatVec3(AppVec3 value) {
    return [
      value.x.toStringAsFixed(2),
      value.y.toStringAsFixed(2),
      value.z.toStringAsFixed(2),
    ].join(', ');
  }
}

class _ModeButton extends StatelessWidget {
  const _ModeButton({
    super.key,
    required this.label,
    required this.selected,
    required this.enabled,
    required this.onPressed,
  });

  final String label;
  final bool selected;
  final bool enabled;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    if (selected) {
      return FilledButton(
        onPressed: enabled ? onPressed : null,
        child: Text(label),
      );
    }

    return OutlinedButton(
      onPressed: enabled ? onPressed : null,
      child: Text(label),
    );
  }
}

class _AxisButton extends StatelessWidget {
  const _AxisButton({
    super.key,
    required this.label,
    required this.enabled,
    required this.onPressed,
  });

  final String label;
  final bool enabled;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton(
      onPressed: enabled ? onPressed : null,
      child: Text(label),
    );
  }
}
