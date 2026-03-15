import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class LightInspectorPanel extends StatefulWidget {
  const LightInspectorPanel({
    super.key,
    required this.properties,
    required this.lightLinking,
    required this.enabled,
    required this.onSetLightType,
    required this.onSetLightColor,
    required this.onSetLightIntensity,
    required this.onSetLightRange,
    required this.onSetLightSpotAngle,
    required this.onSetLightCastShadows,
    required this.onSetLightShadowSoftness,
    required this.onSetLightShadowColor,
    required this.onSetLightVolumetric,
    required this.onSetLightVolumetricDensity,
    required this.onSetLightCookie,
    required this.onClearLightCookie,
    required this.onSetLightProximityMode,
    required this.onSetLightProximityRange,
    required this.onSetLightArrayPattern,
    required this.onSetLightArrayCount,
    required this.onSetLightArrayRadius,
    required this.onSetLightArrayColorVariation,
    required this.onSetLightIntensityExpression,
    required this.onSetLightColorHueExpression,
    required this.onSetNodeLightMask,
    required this.onSetNodeLightLinkEnabled,
  });

  final AppSelectedNodePropertiesSnapshot? properties;
  final AppLightLinkingSnapshot? lightLinking;
  final bool enabled;
  final ValueChanged<String> onSetLightType;
  final ValueChanged<AppVec3> onSetLightColor;
  final ValueChanged<double> onSetLightIntensity;
  final ValueChanged<double> onSetLightRange;
  final ValueChanged<double> onSetLightSpotAngle;
  final ValueChanged<bool> onSetLightCastShadows;
  final ValueChanged<double> onSetLightShadowSoftness;
  final ValueChanged<AppVec3> onSetLightShadowColor;
  final ValueChanged<bool> onSetLightVolumetric;
  final ValueChanged<double> onSetLightVolumetricDensity;
  final ValueChanged<int> onSetLightCookie;
  final VoidCallback onClearLightCookie;
  final ValueChanged<String> onSetLightProximityMode;
  final ValueChanged<double> onSetLightProximityRange;
  final ValueChanged<String> onSetLightArrayPattern;
  final ValueChanged<int> onSetLightArrayCount;
  final ValueChanged<double> onSetLightArrayRadius;
  final ValueChanged<double> onSetLightArrayColorVariation;
  final ValueChanged<String> onSetLightIntensityExpression;
  final ValueChanged<String> onSetLightColorHueExpression;
  final void Function(int nodeId, int mask) onSetNodeLightMask;
  final void Function(int nodeId, int lightId, bool enabled) onSetNodeLightLinkEnabled;

  @override
  State<LightInspectorPanel> createState() => _LightInspectorPanelState();
}

class _LightInspectorPanelState extends State<LightInspectorPanel> {
  late final TextEditingController _intensityExpressionController;
  late final TextEditingController _colorHueExpressionController;

  AppLightPropertiesSnapshot? get _light => widget.properties?.light;

  @override
  void initState() {
    super.initState();
    _intensityExpressionController = TextEditingController(
      text: _light?.intensityExpression ?? '',
    );
    _colorHueExpressionController = TextEditingController(
      text: _light?.colorHueExpression ?? '',
    );
  }

  @override
  void didUpdateWidget(covariant LightInspectorPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_light != oldWidget.properties?.light) {
      final nextIntensityExpression = _light?.intensityExpression ?? '';
      final nextColorHueExpression = _light?.colorHueExpression ?? '';
      if (_intensityExpressionController.text != nextIntensityExpression) {
        _intensityExpressionController.text = nextIntensityExpression;
      }
      if (_colorHueExpressionController.text != nextColorHueExpression) {
        _colorHueExpressionController.text = nextColorHueExpression;
      }
    }
  }

  @override
  void dispose() {
    _intensityExpressionController.dispose();
    _colorHueExpressionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final selectedProperties = widget.properties;
    final lightLinking = widget.lightLinking;
    final light = _light;

    if (selectedProperties == null && lightLinking == null) {
      return const Text('Light controls are still loading.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (light == null)
          Text(
            'Select a light node or its parent transform to inspect backend-owned light controls.',
            style: Theme.of(context).textTheme.bodyMedium,
          )
        else ...[
          _LightDetailsCard(
            properties: selectedProperties!,
            light: light,
            enabled: widget.enabled,
            intensityExpressionController: _intensityExpressionController,
            colorHueExpressionController: _colorHueExpressionController,
            onSetLightType: widget.onSetLightType,
            onSetLightColor: widget.onSetLightColor,
            onSetLightIntensity: widget.onSetLightIntensity,
            onSetLightRange: widget.onSetLightRange,
            onSetLightSpotAngle: widget.onSetLightSpotAngle,
            onSetLightCastShadows: widget.onSetLightCastShadows,
            onSetLightShadowSoftness: widget.onSetLightShadowSoftness,
            onSetLightShadowColor: widget.onSetLightShadowColor,
            onSetLightVolumetric: widget.onSetLightVolumetric,
            onSetLightVolumetricDensity: widget.onSetLightVolumetricDensity,
            onSetLightCookie: widget.onSetLightCookie,
            onClearLightCookie: widget.onClearLightCookie,
            onSetLightProximityMode: widget.onSetLightProximityMode,
            onSetLightProximityRange: widget.onSetLightProximityRange,
            onSetLightArrayPattern: widget.onSetLightArrayPattern,
            onSetLightArrayCount: widget.onSetLightArrayCount,
            onSetLightArrayRadius: widget.onSetLightArrayRadius,
            onSetLightArrayColorVariation: widget.onSetLightArrayColorVariation,
            onSetLightIntensityExpression: widget.onSetLightIntensityExpression,
            onSetLightColorHueExpression: widget.onSetLightColorHueExpression,
          ),
          const SizedBox(height: ShellTokens.controlGap),
        ],
        _LightLinkingCard(
          lightLinking: lightLinking,
          enabled: widget.enabled,
          onSetNodeLightMask: widget.onSetNodeLightMask,
          onSetNodeLightLinkEnabled: widget.onSetNodeLightLinkEnabled,
        ),
      ],
    );
  }
}

class _LightDetailsCard extends StatelessWidget {
  const _LightDetailsCard({
    required this.properties,
    required this.light,
    required this.enabled,
    required this.intensityExpressionController,
    required this.colorHueExpressionController,
    required this.onSetLightType,
    required this.onSetLightColor,
    required this.onSetLightIntensity,
    required this.onSetLightRange,
    required this.onSetLightSpotAngle,
    required this.onSetLightCastShadows,
    required this.onSetLightShadowSoftness,
    required this.onSetLightShadowColor,
    required this.onSetLightVolumetric,
    required this.onSetLightVolumetricDensity,
    required this.onSetLightCookie,
    required this.onClearLightCookie,
    required this.onSetLightProximityMode,
    required this.onSetLightProximityRange,
    required this.onSetLightArrayPattern,
    required this.onSetLightArrayCount,
    required this.onSetLightArrayRadius,
    required this.onSetLightArrayColorVariation,
    required this.onSetLightIntensityExpression,
    required this.onSetLightColorHueExpression,
  });

  final AppSelectedNodePropertiesSnapshot properties;
  final AppLightPropertiesSnapshot light;
  final bool enabled;
  final TextEditingController intensityExpressionController;
  final TextEditingController colorHueExpressionController;
  final ValueChanged<String> onSetLightType;
  final ValueChanged<AppVec3> onSetLightColor;
  final ValueChanged<double> onSetLightIntensity;
  final ValueChanged<double> onSetLightRange;
  final ValueChanged<double> onSetLightSpotAngle;
  final ValueChanged<bool> onSetLightCastShadows;
  final ValueChanged<double> onSetLightShadowSoftness;
  final ValueChanged<AppVec3> onSetLightShadowColor;
  final ValueChanged<bool> onSetLightVolumetric;
  final ValueChanged<double> onSetLightVolumetricDensity;
  final ValueChanged<int> onSetLightCookie;
  final VoidCallback onClearLightCookie;
  final ValueChanged<String> onSetLightProximityMode;
  final ValueChanged<double> onSetLightProximityRange;
  final ValueChanged<String> onSetLightArrayPattern;
  final ValueChanged<int> onSetLightArrayCount;
  final ValueChanged<double> onSetLightArrayRadius;
  final ValueChanged<double> onSetLightArrayColorVariation;
  final ValueChanged<String> onSetLightIntensityExpression;
  final ValueChanged<String> onSetLightColorHueExpression;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(properties.name, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              'Editing ${light.lightTypeLabel.toLowerCase()} light state through backend-owned commands.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            Text('Light Type', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: ShellTokens.compactGap),
            Wrap(
              spacing: ShellTokens.controlGap,
              runSpacing: ShellTokens.controlGap,
              children: [
                for (final option in _lightTypeOptions)
                  ChoiceChip(
                    key: ValueKey('selected-light-type-${option.id}'),
                    label: Text(option.label),
                    selected: light.lightTypeId == option.id,
                    onSelected: enabled ? (_) => onSetLightType(option.id) : null,
                  ),
              ],
            ),
            const SizedBox(height: ShellTokens.controlGap),
            _VectorStepperGroup(
              label: 'Light Color',
              keyPrefix: 'selected-light-color',
              value: light.color,
              enabled: enabled,
              step: 0.05,
              onChanged: onSetLightColor,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            _StepperRow(
              label: 'Intensity',
              value: light.intensity,
              fractionDigits: 2,
              decreaseKey: const ValueKey('selected-light-intensity-decrease'),
              increaseKey: const ValueKey('selected-light-intensity-increase'),
              onDecrease: enabled
                  ? () => onSetLightIntensity((light.intensity - 0.25).clamp(-10.0, 10.0))
                  : null,
              onIncrease: enabled
                  ? () => onSetLightIntensity((light.intensity + 0.25).clamp(-10.0, 10.0))
                  : null,
            ),
            if (light.supportsRange) ...[
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Range',
                value: light.range,
                fractionDigits: 2,
                decreaseKey: const ValueKey('selected-light-range-decrease'),
                increaseKey: const ValueKey('selected-light-range-increase'),
                onDecrease: enabled
                    ? () => onSetLightRange((light.range - 0.5).clamp(0.1, 50.0))
                    : null,
                onIncrease: enabled
                    ? () => onSetLightRange((light.range + 0.5).clamp(0.1, 50.0))
                    : null,
              ),
            ],
            if (light.supportsSpotAngle) ...[
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Spot Angle',
                value: light.spotAngle,
                fractionDigits: 1,
                decreaseKey: const ValueKey('selected-light-spot-angle-decrease'),
                increaseKey: const ValueKey('selected-light-spot-angle-increase'),
                onDecrease: enabled
                    ? () => onSetLightSpotAngle((light.spotAngle - 1.0).clamp(1.0, 179.0))
                    : null,
                onIncrease: enabled
                    ? () => onSetLightSpotAngle((light.spotAngle + 1.0).clamp(1.0, 179.0))
                    : null,
              ),
            ],
            if (light.supportsShadows) ...[
              const SizedBox(height: ShellTokens.controlGap),
              SwitchListTile.adaptive(
                key: const ValueKey('selected-light-cast-shadows-toggle'),
                contentPadding: EdgeInsets.zero,
                value: light.castShadows,
                onChanged: enabled ? onSetLightCastShadows : null,
                title: const Text('Cast Shadows'),
              ),
              _StepperRow(
                label: 'Shadow Softness',
                value: light.shadowSoftness,
                fractionDigits: 1,
                decreaseKey: const ValueKey('selected-light-shadow-softness-decrease'),
                increaseKey: const ValueKey('selected-light-shadow-softness-increase'),
                onDecrease: enabled
                    ? () => onSetLightShadowSoftness(
                        (light.shadowSoftness - 1.0).clamp(1.0, 64.0),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetLightShadowSoftness(
                        (light.shadowSoftness + 1.0).clamp(1.0, 64.0),
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _VectorStepperGroup(
                label: 'Shadow Color',
                keyPrefix: 'selected-light-shadow-color',
                value: light.shadowColor,
                enabled: enabled,
                step: 0.05,
                onChanged: onSetLightShadowColor,
              ),
            ],
            if (light.supportsVolumetric) ...[
              const SizedBox(height: ShellTokens.controlGap),
              SwitchListTile.adaptive(
                key: const ValueKey('selected-light-volumetric-toggle'),
                contentPadding: EdgeInsets.zero,
                value: light.volumetric,
                onChanged: enabled ? onSetLightVolumetric : null,
                title: const Text('Volumetric'),
              ),
              _StepperRow(
                label: 'Volumetric Density',
                value: light.volumetricDensity,
                fractionDigits: 2,
                decreaseKey: const ValueKey('selected-light-volumetric-density-decrease'),
                increaseKey: const ValueKey('selected-light-volumetric-density-increase'),
                onDecrease: enabled
                    ? () => onSetLightVolumetricDensity(
                        (light.volumetricDensity - 0.05).clamp(0.01, 1.0),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetLightVolumetricDensity(
                        (light.volumetricDensity + 0.05).clamp(0.01, 1.0),
                      )
                    : null,
              ),
            ],
            if (light.supportsCookie) ...[
              const SizedBox(height: ShellTokens.controlGap),
              DropdownButtonFormField<int?>(
                key: const ValueKey('selected-light-cookie-dropdown'),
                initialValue: light.cookieNodeId,
                isExpanded: true,
                decoration: const InputDecoration(labelText: 'Cookie Source'),
                items: [
                  const DropdownMenuItem<int?>(
                    value: null,
                    child: Text('None', overflow: TextOverflow.ellipsis),
                  ),
                  for (final candidate in light.cookieCandidates)
                    DropdownMenuItem<int?>(
                      value: candidate.nodeId,
                      child: Text(
                        '${candidate.name} (${candidate.kindLabel})',
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
                onChanged: enabled
                    ? (value) {
                        if (value == null) {
                          onClearLightCookie();
                        } else {
                          onSetLightCookie(value);
                        }
                      }
                    : null,
              ),
            ],
            if (light.supportsProximity) ...[
              const SizedBox(height: ShellTokens.controlGap),
              Wrap(
                spacing: ShellTokens.controlGap,
                runSpacing: ShellTokens.controlGap,
                children: [
                  for (final option in _proximityOptions)
                    ChoiceChip(
                      key: ValueKey('selected-light-proximity-${option.id}'),
                      label: Text(option.label),
                      selected: light.proximityModeId == option.id,
                      onSelected: enabled
                          ? (_) => onSetLightProximityMode(option.id)
                          : null,
                    ),
                ],
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Proximity Range',
                value: light.proximityRange,
                fractionDigits: 2,
                decreaseKey: const ValueKey('selected-light-proximity-range-decrease'),
                increaseKey: const ValueKey('selected-light-proximity-range-increase'),
                onDecrease: enabled
                    ? () => onSetLightProximityRange(
                        (light.proximityRange - 0.25).clamp(0.1, 10.0),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetLightProximityRange(
                        (light.proximityRange + 0.25).clamp(0.1, 10.0),
                      )
                    : null,
              ),
            ],
            if (light.supportsArray) ...[
              const SizedBox(height: ShellTokens.controlGap),
              Wrap(
                spacing: ShellTokens.controlGap,
                runSpacing: ShellTokens.controlGap,
                children: [
                  for (final option in _arrayPatternOptions)
                    ChoiceChip(
                      key: ValueKey('selected-light-array-pattern-${option.id}'),
                      label: Text(option.label),
                      selected: light.arrayPatternId == option.id,
                      onSelected: enabled
                          ? (_) => onSetLightArrayPattern(option.id)
                          : null,
                    ),
                ],
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Array Count',
                value: (light.arrayCount ?? 2).toDouble(),
                fractionDigits: 0,
                decreaseKey: const ValueKey('selected-light-array-count-decrease'),
                increaseKey: const ValueKey('selected-light-array-count-increase'),
                onDecrease: enabled
                    ? () => onSetLightArrayCount(((light.arrayCount ?? 2) - 1).clamp(2, 32))
                    : null,
                onIncrease: enabled
                    ? () => onSetLightArrayCount(((light.arrayCount ?? 2) + 1).clamp(2, 32))
                    : null,
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Array Radius',
                value: light.arrayRadius ?? 1.0,
                fractionDigits: 2,
                decreaseKey: const ValueKey('selected-light-array-radius-decrease'),
                increaseKey: const ValueKey('selected-light-array-radius-increase'),
                onDecrease: enabled
                    ? () => onSetLightArrayRadius(
                        ((light.arrayRadius ?? 1.0) - 0.25).clamp(0.1, 20.0),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetLightArrayRadius(
                        ((light.arrayRadius ?? 1.0) + 0.25).clamp(0.1, 20.0),
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _StepperRow(
                label: 'Color Variation',
                value: light.arrayColorVariation ?? 0.0,
                fractionDigits: 2,
                decreaseKey: const ValueKey('selected-light-array-color-variation-decrease'),
                increaseKey: const ValueKey('selected-light-array-color-variation-increase'),
                onDecrease: enabled
                    ? () => onSetLightArrayColorVariation(
                        ((light.arrayColorVariation ?? 0.0) - 0.05).clamp(0.0, 1.0),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetLightArrayColorVariation(
                        ((light.arrayColorVariation ?? 0.0) + 0.05).clamp(0.0, 1.0),
                      )
                    : null,
              ),
            ],
            if (light.supportsExpressions) ...[
              const SizedBox(height: ShellTokens.controlGap),
              _ExpressionField(
                keyPrefix: 'selected-light-intensity-expression',
                label: 'Intensity Expression',
                controller: intensityExpressionController,
                enabled: enabled,
                errorText: light.intensityExpressionError,
                onApply: () => onSetLightIntensityExpression(
                  intensityExpressionController.text,
                ),
              ),
              const SizedBox(height: ShellTokens.controlGap),
              _ExpressionField(
                keyPrefix: 'selected-light-color-hue-expression',
                label: 'Color Hue Expression',
                controller: colorHueExpressionController,
                enabled: enabled,
                errorText: light.colorHueExpressionError,
                onApply: () => onSetLightColorHueExpression(
                  colorHueExpressionController.text,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _LightLinkingCard extends StatelessWidget {
  const _LightLinkingCard({
    required this.lightLinking,
    required this.enabled,
    required this.onSetNodeLightMask,
    required this.onSetNodeLightLinkEnabled,
  });

  final AppLightLinkingSnapshot? lightLinking;
  final bool enabled;
  final void Function(int nodeId, int mask) onSetNodeLightMask;
  final void Function(int nodeId, int lightId, bool enabled) onSetNodeLightLinkEnabled;

  @override
  Widget build(BuildContext context) {
    if (lightLinking == null) {
      return const Text('Light linking is still loading.');
    }
    if (lightLinking!.lights.isEmpty) {
      return const Text('No visible lights are available for backend-owned linking.');
    }
    if (lightLinking!.geometryNodes.isEmpty) {
      return const Text('No geometry nodes are available for backend-owned light linking.');
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Light Linking', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              '${lightLinking!.totalVisibleLightCount} visible lights, ${lightLinking!.lights.length} listed here (limit ${lightLinking!.maxLightCount}).',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            for (final geometryNode in lightLinking!.geometryNodes) ...[
              DecoratedBox(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(ShellTokens.controlGap),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            geometryNode.nodeName,
                            style: Theme.of(context).textTheme.titleSmall,
                          ),
                          Text(
                            geometryNode.kindLabel,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                      const SizedBox(height: ShellTokens.controlGap),
                      Wrap(
                        spacing: ShellTokens.controlGap,
                        runSpacing: ShellTokens.controlGap,
                        children: [
                          for (final light in lightLinking!.lights)
                            FilterChip(
                              key: ValueKey(
                                'light-link-node-${geometryNode.nodeId}-light-${light.lightNodeId}',
                              ),
                              selected:
                                  (geometryNode.lightMask & (1 << light.maskBit)) != 0,
                              onSelected: enabled
                                  ? (selected) => onSetNodeLightLinkEnabled(
                                      geometryNode.nodeId,
                                      light.lightNodeId,
                                      selected,
                                    )
                                  : null,
                              avatar: _LightColorDot(color: light.color),
                              label: Text(
                                light.active
                                    ? light.lightName
                                    : '${light.lightName} (inactive)',
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              if (geometryNode != lightLinking!.geometryNodes.last)
                const SizedBox(height: ShellTokens.controlGap),
            ],
          ],
        ),
      ),
    );
  }
}

class _StepperRow extends StatelessWidget {
  const _StepperRow({
    required this.label,
    required this.value,
    required this.fractionDigits,
    required this.decreaseKey,
    required this.increaseKey,
    required this.onDecrease,
    required this.onIncrease,
  });

  final String label;
  final double value;
  final int fractionDigits;
  final Key decreaseKey;
  final Key increaseKey;
  final VoidCallback? onDecrease;
  final VoidCallback? onIncrease;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '$label: ${value.toStringAsFixed(fractionDigits)}',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: ShellTokens.compactGap),
        Wrap(
          spacing: ShellTokens.compactGap,
          runSpacing: ShellTokens.compactGap,
          children: [
            IconButton.filledTonal(
              key: decreaseKey,
              onPressed: onDecrease,
              icon: const Icon(Icons.remove),
            ),
            IconButton.filledTonal(
              key: increaseKey,
              onPressed: onIncrease,
              icon: const Icon(Icons.add),
            ),
          ],
        ),
      ],
    );
  }
}

class _VectorStepperGroup extends StatelessWidget {
  const _VectorStepperGroup({
    required this.label,
    required this.keyPrefix,
    required this.value,
    required this.enabled,
    required this.step,
    required this.onChanged,
  });

  final String label;
  final String keyPrefix;
  final AppVec3 value;
  final bool enabled;
  final double step;
  final ValueChanged<AppVec3> onChanged;

  @override
  Widget build(BuildContext context) {
    final rows = <(String label, String id, double value)>[
      ('R', 'red', value.x),
      ('G', 'green', value.y),
      ('B', 'blue', value.z),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '$label: ${value.x.toStringAsFixed(2)}, ${value.y.toStringAsFixed(2)}, ${value.z.toStringAsFixed(2)}',
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: ShellTokens.compactGap),
        for (final row in rows) ...[
          _StepperRow(
            label: row.$1,
            value: row.$3,
            fractionDigits: 2,
            decreaseKey: ValueKey('$keyPrefix-${row.$2}-decrease'),
            increaseKey: ValueKey('$keyPrefix-${row.$2}-increase'),
            onDecrease: enabled
                ? () => onChanged(_updatedColor(value, row.$2, row.$3 - step))
                : null,
            onIncrease: enabled
                ? () => onChanged(_updatedColor(value, row.$2, row.$3 + step))
                : null,
          ),
          if (row != rows.last) const SizedBox(height: ShellTokens.compactGap),
        ],
      ],
    );
  }

  AppVec3 _updatedColor(AppVec3 current, String component, double nextValue) {
    final clampedValue = nextValue.clamp(0.0, 1.0);
    return switch (component) {
      'red' => AppVec3(x: clampedValue, y: current.y, z: current.z),
      'green' => AppVec3(x: current.x, y: clampedValue, z: current.z),
      _ => AppVec3(x: current.x, y: current.y, z: clampedValue),
    };
  }
}

class _ExpressionField extends StatelessWidget {
  const _ExpressionField({
    required this.keyPrefix,
    required this.label,
    required this.controller,
    required this.enabled,
    required this.errorText,
    required this.onApply,
  });

  final String keyPrefix;
  final String label;
  final TextEditingController controller;
  final bool enabled;
  final String? errorText;
  final VoidCallback onApply;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          key: ValueKey('$keyPrefix-field'),
          controller: controller,
          enabled: enabled,
          decoration: InputDecoration(labelText: label, errorText: errorText),
          onSubmitted: (_) => onApply(),
        ),
        const SizedBox(height: ShellTokens.compactGap),
        FilledButton.tonal(
          key: ValueKey('$keyPrefix-apply'),
          onPressed: enabled ? onApply : null,
          child: const Text('Apply Expression'),
        ),
      ],
    );
  }
}

class _LightColorDot extends StatelessWidget {
  const _LightColorDot({required this.color});

  final AppVec3 color;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(
        color: Color.fromRGBO(
          (color.x.clamp(0.0, 1.0) * 255).round(),
          (color.y.clamp(0.0, 1.0) * 255).round(),
          (color.z.clamp(0.0, 1.0) * 255).round(),
          1,
        ),
        shape: BoxShape.circle,
      ),
    );
  }
}

class _OptionChipData {
  const _OptionChipData({required this.id, required this.label});

  final String id;
  final String label;
}

const List<_OptionChipData> _lightTypeOptions = <_OptionChipData>[
  _OptionChipData(id: 'point', label: 'Point'),
  _OptionChipData(id: 'spot', label: 'Spot'),
  _OptionChipData(id: 'directional', label: 'Directional'),
  _OptionChipData(id: 'ambient', label: 'Ambient'),
  _OptionChipData(id: 'array', label: 'Array'),
];

const List<_OptionChipData> _proximityOptions = <_OptionChipData>[
  _OptionChipData(id: 'off', label: 'Off'),
  _OptionChipData(id: 'brighten', label: 'Brighten'),
  _OptionChipData(id: 'dim', label: 'Dim'),
];

const List<_OptionChipData> _arrayPatternOptions = <_OptionChipData>[
  _OptionChipData(id: 'ring', label: 'Ring'),
  _OptionChipData(id: 'line', label: 'Line'),
  _OptionChipData(id: 'grid', label: 'Grid'),
  _OptionChipData(id: 'spiral', label: 'Spiral'),
];
