import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

class ShellQuickWheelAction {
  const ShellQuickWheelAction({
    required this.id,
    required this.label,
    required this.enabled,
    required this.favorite,
    this.shortcutLabel,
  });

  final String id;
  final String label;
  final bool enabled;
  final bool favorite;
  final String? shortcutLabel;
}

class ShellQuickWheel extends StatelessWidget {
  const ShellQuickWheel({
    super.key,
    required this.anchor,
    required this.actions,
    required this.onDismiss,
    required this.onExecute,
    required this.onToggleFavorite,
  });

  final Offset anchor;
  final List<ShellQuickWheelAction> actions;
  final VoidCallback onDismiss;
  final ValueChanged<String> onExecute;
  final ValueChanged<String> onToggleFavorite;

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      key: const ValueKey('shell-quick-wheel'),
      child: GestureDetector(
        onTap: onDismiss,
        behavior: HitTestBehavior.opaque,
        child: ColoredBox(
          color: Colors.black.withValues(alpha: 0.2),
          child: LayoutBuilder(
            builder: (context, constraints) {
              final displayedActions = actions.take(8).toList(growable: false);
              final center = Offset(
                anchor.dx.clamp(180.0, constraints.maxWidth - 180.0),
                anchor.dy.clamp(180.0, constraints.maxHeight - 180.0),
              );

              return Stack(
                children: [
                  for (var index = 0; index < displayedActions.length; index++)
                    _positionedAction(
                      context,
                      action: displayedActions[index],
                      index: index,
                      total: displayedActions.length,
                      center: center,
                    ),
                  Positioned(
                    left: center.dx - 70,
                    top: center.dy - 70,
                    child: GestureDetector(
                      onTap: () {},
                      child: DecoratedBox(
                        decoration: ShellSurfaceStyles.overlayPanel(
                          context,
                          accentColor: context.shellPalette.infoAccent,
                          pill: true,
                        ),
                        child: SizedBox(
                          width: 140,
                          height: 140,
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                'Quick Wheel',
                                style: Theme.of(context).textTheme.titleSmall
                                    ?.copyWith(
                                      color: context.shellPalette.overlayText,
                                    ),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                'Long press\nfor actions',
                                textAlign: TextAlign.center,
                                style: Theme.of(context).textTheme.bodySmall
                                    ?.copyWith(
                                      color:
                                          context.shellPalette.overlayMutedText,
                                    ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _positionedAction(
    BuildContext context, {
    required ShellQuickWheelAction action,
    required int index,
    required int total,
    required Offset center,
  }) {
    final angleStep = (math.pi * 2) / math.max(total, 1);
    final angle = (-math.pi / 2) + (angleStep * index);
    const radius = 138.0;
    final actionCenter = Offset(
      center.dx + (math.cos(angle) * radius),
      center.dy + (math.sin(angle) * radius),
    );

    return Positioned(
      left: actionCenter.dx - 76,
      top: actionCenter.dy - 34,
      child: GestureDetector(
        onTap: () {},
        child: SizedBox(
          width: 152,
          child: DecoratedBox(
            decoration: ShellSurfaceStyles.overlayPanel(
              context,
              accentColor: action.favorite
                  ? context.shellPalette.successAccent
                  : context.shellPalette.infoAccent,
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    action.label,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: context.shellPalette.overlayText,
                        ),
                  ),
                  if (action.shortcutLabel != null) ...[
                    const SizedBox(height: 2),
                    Text(
                      action.shortcutLabel!,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: context.shellPalette.overlayMutedText,
                          ),
                    ),
                  ],
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton.filledTonal(
                        tooltip: action.favorite
                            ? 'Remove from favorites'
                            : 'Pin to favorites',
                        onPressed: () => onToggleFavorite(action.id),
                        icon: Icon(
                          action.favorite ? Icons.star : Icons.star_border,
                        ),
                      ),
                      const SizedBox(width: 8),
                      FilledButton(
                        onPressed: action.enabled
                            ? () {
                                onExecute(action.id);
                                onDismiss();
                              }
                            : null,
                        child: const Text('Run'),
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
}
