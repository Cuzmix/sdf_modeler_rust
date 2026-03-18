import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class ShellUtilityStrip extends StatelessWidget {
  const ShellUtilityStrip({
    super.key,
    required this.renderSettings,
    required this.selectionContext,
    required this.orthographic,
    required this.propertiesOpen,
    required this.propertiesPinned,
    required this.canPinProperties,
    required this.enabled,
    required this.adaptiveInteractionResolutionEnabled,
    required this.onOpenProperties,
    required this.onTogglePropertiesPin,
    required this.onFrameAll,
    required this.onFocusSelected,
    required this.onToggleProjection,
    required this.onSetRenderShadingMode,
    required this.onToggleAdaptiveInteractionResolution,
  });

  final AppRenderSettingsSnapshot renderSettings;
  final AppSelectionContextSnapshot selectionContext;
  final bool orthographic;
  final bool propertiesOpen;
  final bool propertiesPinned;
  final bool canPinProperties;
  final bool enabled;
  final bool adaptiveInteractionResolutionEnabled;
  final VoidCallback onOpenProperties;
  final VoidCallback onTogglePropertiesPin;
  final VoidCallback onFrameAll;
  final VoidCallback onFocusSelected;
  final VoidCallback onToggleProjection;
  final ValueChanged<String> onSetRenderShadingMode;
  final ValueChanged<bool> onToggleAdaptiveInteractionResolution;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 88,
      child: ShellPanelSurface(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _UtilityButton(
              tooltip: propertiesOpen ? 'Hide properties' : 'Open properties',
              icon: propertiesOpen
                  ? Icons.tune
                  : Icons.tune_outlined,
              selected: propertiesOpen || propertiesPinned,
              onPressed: enabled ? onOpenProperties : null,
            ),
            if (canPinProperties) ...[
              const SizedBox(height: ShellTokens.compactGap),
              _UtilityButton(
                tooltip: propertiesPinned
                    ? 'Unpin properties drawer'
                    : 'Pin properties drawer',
                icon: propertiesPinned
                    ? Icons.push_pin
                    : Icons.push_pin_outlined,
                selected: propertiesPinned,
                onPressed: enabled ? onTogglePropertiesPin : null,
              ),
            ],
            const SizedBox(height: ShellTokens.controlGap),
            _UtilityButton(
              tooltip: 'Frame all',
              icon: Icons.center_focus_strong_outlined,
              onPressed: enabled ? onFrameAll : null,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            _UtilityButton(
              tooltip: 'Focus selected',
              icon: Icons.filter_center_focus_outlined,
              onPressed: enabled && selectionContext.selectionCount > 0
                  ? onFocusSelected
                  : null,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            _UtilityButton(
              tooltip: orthographic
                  ? 'Switch to perspective'
                  : 'Switch to orthographic',
              icon: orthographic
                  ? Icons.threed_rotation_outlined
                  : Icons.crop_free_outlined,
              onPressed: enabled ? onToggleProjection : null,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            PopupMenuButton<String>(
              key: const ValueKey('shell-shading-menu'),
              enabled: enabled,
              tooltip: 'Viewport shading',
              onSelected: onSetRenderShadingMode,
              itemBuilder: (context) {
                return [
                  for (final mode in renderSettings.shadingModes)
                    PopupMenuItem<String>(
                      value: mode.id,
                      child: Text(mode.label),
                    ),
                ];
              },
              child: const _UtilityButton(
                tooltip: 'Viewport shading',
                icon: Icons.visibility_outlined,
              ),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            DecoratedBox(
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerLow,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: Theme.of(context).colorScheme.outlineVariant,
                ),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 8,
                  vertical: 10,
                ),
                child: Column(
                  children: [
                    RotatedBox(
                      quarterTurns: 3,
                      child: Switch.adaptive(
                        value: adaptiveInteractionResolutionEnabled,
                        onChanged: enabled
                            ? onToggleAdaptiveInteractionResolution
                            : null,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      renderSettings.shadingModeLabel,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.labelSmall,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      selectionContext.selectionCount > 0
                          ? '${selectionContext.selectionCount} sel'
                          : 'Canvas',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodySmall,
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

class _UtilityButton extends StatelessWidget {
  const _UtilityButton({
    required this.tooltip,
    required this.icon,
    this.selected = false,
    this.onPressed,
  });

  final String tooltip;
  final IconData icon;
  final bool selected;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Tooltip(
      message: tooltip,
      child: InkResponse(
        onTap: onPressed,
        radius: 28,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: selected
                ? colorScheme.primaryContainer
                : colorScheme.surfaceContainerLow,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: selected
                  ? colorScheme.primary
                  : colorScheme.outlineVariant,
            ),
          ),
          child: Icon(
            icon,
            color: selected
                ? colorScheme.onPrimaryContainer
                : colorScheme.onSurface,
          ),
        ),
      ),
    );
  }
}
