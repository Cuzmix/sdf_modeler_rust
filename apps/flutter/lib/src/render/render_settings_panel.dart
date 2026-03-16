import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class RenderSettingsPanel extends StatelessWidget {
  const RenderSettingsPanel({
    super.key,
    required this.renderSettings,
    required this.enabled,
    required this.onApplyPreset,
    required this.onSetShadingMode,
    required this.onSetToggle,
    required this.onSetInteger,
    required this.onSetScalar,
  });

  final AppRenderSettingsSnapshot? renderSettings;
  final bool enabled;
  final ValueChanged<String> onApplyPreset;
  final ValueChanged<String> onSetShadingMode;
  final void Function(String fieldId, bool enabled) onSetToggle;
  final void Function(String fieldId, int value) onSetInteger;
  final void Function(String fieldId, double value) onSetScalar;

  @override
  Widget build(BuildContext context) {
    final renderSettings = this.renderSettings;
    if (renderSettings == null) {
      return const Text('Render settings are still loading.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Renderer quality, shading, and post effects stay on the Rust command path.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: const [
            _RenderPresetChip(presetId: 'fast', label: 'Fast'),
            _RenderPresetChip(presetId: 'balanced', label: 'Balanced'),
            _RenderPresetChip(presetId: 'quality', label: 'Quality'),
          ].map((chip) {
            return ActionChip(
              key: ValueKey('render-preset-${chip.presetId}'),
              label: Text(chip.label),
              onPressed: enabled ? () => onApplyPreset(chip.presetId) : null,
            );
          }).toList(growable: false),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Shading',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Wrap(
                spacing: ShellTokens.controlGap,
                runSpacing: ShellTokens.controlGap,
                children: [
                  for (final mode in renderSettings.shadingModes)
                    ChoiceChip(
                      key: ValueKey('render-shading-${mode.id}'),
                      label: Text(mode.label),
                      selected: renderSettings.shadingModeId == mode.id,
                      onSelected: enabled
                          ? (_) => onSetShadingMode(mode.id)
                          : null,
                    ),
                ],
              ),
              const SizedBox(height: ShellTokens.compactGap),
              SwitchListTile.adaptive(
                key: const ValueKey('render-grid-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.showGrid,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('show_grid', nextValue)
                    : null,
                title: const Text('Show Grid'),
              ),
            ],
          ),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Quality',
          child: Column(
            children: [
              _StepperRow(
                label: 'March Steps',
                valueLabel: renderSettings.marchMaxSteps.toString(),
                decreaseKey: const ValueKey('render-march-steps-decrease'),
                increaseKey: const ValueKey('render-march-steps-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'march_max_steps',
                        (renderSettings.marchMaxSteps - 32).clamp(32, 512),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'march_max_steps',
                        (renderSettings.marchMaxSteps + 32).clamp(32, 512),
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'Interaction Scale',
                valueLabel:
                    '${(renderSettings.interactionRenderScale * 100).round()}%',
                decreaseKey: const ValueKey('render-interaction-scale-decrease'),
                increaseKey: const ValueKey('render-interaction-scale-increase'),
                onDecrease: enabled
                    ? () => onSetScalar(
                        'interaction_render_scale',
                        renderSettings.interactionRenderScale - 0.05,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar(
                        'interaction_render_scale',
                        renderSettings.interactionRenderScale + 0.05,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'Rest Scale',
                valueLabel: '${(renderSettings.restRenderScale * 100).round()}%',
                decreaseKey: const ValueKey('render-rest-scale-decrease'),
                increaseKey: const ValueKey('render-rest-scale-increase'),
                onDecrease: enabled
                    ? () => onSetScalar(
                        'rest_render_scale',
                        renderSettings.restRenderScale - 0.05,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar(
                        'rest_render_scale',
                        renderSettings.restRenderScale + 0.05,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              SwitchListTile.adaptive(
                key: const ValueKey('render-sculpt-fast-mode-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.sculptFastMode,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('sculpt_fast_mode', nextValue)
                    : null,
                title: const Text('Sculpt Fast Mode'),
              ),
              SwitchListTile.adaptive(
                key: const ValueKey('render-auto-reduce-steps-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.autoReduceSteps,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('auto_reduce_steps', nextValue)
                    : null,
                title: const Text('Auto Reduce Steps'),
              ),
            ],
          ),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Shadows And AO',
          child: Column(
            children: [
              SwitchListTile.adaptive(
                key: const ValueKey('render-shadow-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.shadowsEnabled,
                onChanged: enabled
                    ? (nextValue) =>
                        onSetToggle('shadows_enabled', nextValue)
                    : null,
                title: const Text('Shadows'),
              ),
              _StepperRow(
                label: 'Shadow Steps',
                valueLabel: renderSettings.shadowSteps.toString(),
                decreaseKey: const ValueKey('render-shadow-steps-decrease'),
                increaseKey: const ValueKey('render-shadow-steps-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'shadow_steps',
                        (renderSettings.shadowSteps - 8).clamp(8, 128),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'shadow_steps',
                        (renderSettings.shadowSteps + 8).clamp(8, 128),
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              SwitchListTile.adaptive(
                key: const ValueKey('render-ao-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.aoEnabled,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('ao_enabled', nextValue)
                    : null,
                title: const Text('Ambient Occlusion'),
              ),
              _StepperRow(
                label: 'AO Samples',
                valueLabel: renderSettings.aoSamples.toString(),
                decreaseKey: const ValueKey('render-ao-samples-decrease'),
                increaseKey: const ValueKey('render-ao-samples-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'ao_samples',
                        (renderSettings.aoSamples - 1).clamp(1, 16),
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'ao_samples',
                        (renderSettings.aoSamples + 1).clamp(1, 16),
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'AO Intensity',
                valueLabel: renderSettings.aoIntensity.toStringAsFixed(2),
                decreaseKey: const ValueKey('render-ao-intensity-decrease'),
                increaseKey: const ValueKey('render-ao-intensity-increase'),
                onDecrease: enabled
                    ? () => onSetScalar(
                        'ao_intensity',
                        renderSettings.aoIntensity - 0.5,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar(
                        'ao_intensity',
                        renderSettings.aoIntensity + 0.5,
                      )
                    : null,
              ),
            ],
          ),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Post Processing',
          child: Column(
            children: [
              SwitchListTile.adaptive(
                key: const ValueKey('render-fog-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.fogEnabled,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('fog_enabled', nextValue)
                    : null,
                title: const Text('Fog'),
              ),
              _StepperRow(
                label: 'Fog Density',
                valueLabel: renderSettings.fogDensity.toStringAsFixed(3),
                decreaseKey: const ValueKey('render-fog-density-decrease'),
                increaseKey: const ValueKey('render-fog-density-increase'),
                onDecrease: enabled
                    ? () => onSetScalar(
                        'fog_density',
                        renderSettings.fogDensity - 0.01,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar(
                        'fog_density',
                        renderSettings.fogDensity + 0.01,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              SwitchListTile.adaptive(
                key: const ValueKey('render-bloom-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.bloomEnabled,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('bloom_enabled', nextValue)
                    : null,
                title: const Text('Bloom'),
              ),
              _StepperRow(
                label: 'Bloom Intensity',
                valueLabel: renderSettings.bloomIntensity.toStringAsFixed(2),
                decreaseKey: const ValueKey('render-bloom-intensity-decrease'),
                increaseKey: const ValueKey('render-bloom-intensity-increase'),
                onDecrease: enabled
                    ? () => onSetScalar(
                        'bloom_intensity',
                        renderSettings.bloomIntensity - 0.1,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar(
                        'bloom_intensity',
                        renderSettings.bloomIntensity + 0.1,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'Gamma',
                valueLabel: renderSettings.gamma.toStringAsFixed(2),
                decreaseKey: const ValueKey('render-gamma-decrease'),
                increaseKey: const ValueKey('render-gamma-increase'),
                onDecrease: enabled
                    ? () => onSetScalar('gamma', renderSettings.gamma - 0.1)
                    : null,
                onIncrease: enabled
                    ? () => onSetScalar('gamma', renderSettings.gamma + 0.1)
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              SwitchListTile.adaptive(
                key: const ValueKey('render-tonemapping-aces-toggle'),
                contentPadding: EdgeInsets.zero,
                value: renderSettings.tonemappingAces,
                onChanged: enabled
                    ? (nextValue) =>
                        onSetToggle('tonemapping_aces', nextValue)
                    : null,
                title: const Text('ACES Tonemapping'),
              ),
            ],
          ),
        ),
        if (renderSettings.shadingModeId == 'cross_section') ...[
          const SizedBox(height: ShellTokens.controlGap),
          _SectionCard(
            title: 'Cross Section',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: ShellTokens.controlGap,
                  runSpacing: ShellTokens.controlGap,
                  children: [
                    for (final axis in _crossSectionAxes)
                      ChoiceChip(
                        key: ValueKey('render-cross-section-axis-${axis.id}'),
                        label: Text(axis.label),
                        selected: renderSettings.crossSectionAxis == axis.value,
                        onSelected: enabled
                            ? (_) => onSetInteger(
                                'cross_section_axis',
                                axis.value,
                              )
                            : null,
                      ),
                  ],
                ),
                const SizedBox(height: ShellTokens.compactGap),
                _StepperRow(
                  label: 'Plane Position',
                  valueLabel:
                      renderSettings.crossSectionPosition.toStringAsFixed(2),
                  decreaseKey: const ValueKey(
                    'render-cross-section-position-decrease',
                  ),
                  increaseKey: const ValueKey(
                    'render-cross-section-position-increase',
                  ),
                  onDecrease: enabled
                      ? () => onSetScalar(
                          'cross_section_position',
                          renderSettings.crossSectionPosition - 0.25,
                        )
                      : null,
                  onIncrease: enabled
                      ? () => onSetScalar(
                          'cross_section_position',
                          renderSettings.crossSectionPosition + 0.25,
                        )
                      : null,
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }
}

class _RenderPresetChip {
  const _RenderPresetChip({required this.presetId, required this.label});

  final String presetId;
  final String label;
}

class _CrossSectionAxisOption {
  const _CrossSectionAxisOption({
    required this.id,
    required this.label,
    required this.value,
  });

  final String id;
  final String label;
  final int value;
}

const List<_CrossSectionAxisOption> _crossSectionAxes =
    <_CrossSectionAxisOption>[
      _CrossSectionAxisOption(id: 'x', label: 'X', value: 0),
      _CrossSectionAxisOption(id: 'y', label: 'Y', value: 1),
      _CrossSectionAxisOption(id: 'z', label: 'Z', value: 2),
    ];

class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.title, required this.child});

  final String title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: ShellTokens.controlGap),
            child,
          ],
        ),
      ),
    );
  }
}

class _StepperRow extends StatelessWidget {
  const _StepperRow({
    required this.label,
    required this.valueLabel,
    required this.decreaseKey,
    required this.increaseKey,
    required this.onDecrease,
    required this.onIncrease,
  });

  final String label;
  final String valueLabel;
  final Key decreaseKey;
  final Key increaseKey;
  final VoidCallback? onDecrease;
  final VoidCallback? onIncrease;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: Text(label)),
        IconButton(
          key: decreaseKey,
          onPressed: onDecrease,
          icon: const Icon(Icons.remove_circle_outline),
        ),
        Text(
          valueLabel,
          style: Theme.of(context).textTheme.titleSmall,
        ),
        IconButton(
          key: increaseKey,
          onPressed: onIncrease,
          icon: const Icon(Icons.add_circle_outline),
        ),
      ],
    );
  }
}
