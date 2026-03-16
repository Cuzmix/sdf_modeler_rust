import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

class SettingsPanel extends StatelessWidget {
  const SettingsPanel({
    super.key,
    required this.settings,
    required this.enabled,
    required this.onResetSettings,
    required this.onExportSettings,
    required this.onImportSettings,
    required this.onSetToggle,
    required this.onSetInteger,
    required this.onSaveCameraBookmark,
    required this.onRestoreCameraBookmark,
    required this.onClearCameraBookmark,
    required this.onResetKeymap,
    required this.onExportKeymap,
    required this.onImportKeymap,
    required this.onClearKeybinding,
    required this.onSetKeybinding,
  });

  final AppSettingsSnapshot? settings;
  final bool enabled;
  final VoidCallback onResetSettings;
  final VoidCallback onExportSettings;
  final VoidCallback onImportSettings;
  final void Function(String fieldId, bool enabled) onSetToggle;
  final void Function(String fieldId, int value) onSetInteger;
  final ValueChanged<int> onSaveCameraBookmark;
  final ValueChanged<int> onRestoreCameraBookmark;
  final ValueChanged<int> onClearCameraBookmark;
  final VoidCallback onResetKeymap;
  final VoidCallback onExportKeymap;
  final VoidCallback onImportKeymap;
  final ValueChanged<String> onClearKeybinding;
  final void Function(
    String actionId,
    String keyId,
    bool ctrl,
    bool shift,
    bool alt,
  )
  onSetKeybinding;

  @override
  Widget build(BuildContext context) {
    final settings = this.settings;
    if (settings == null) {
      return const Text('Application settings are still loading.');
    }

    final keybindingsByCategory = <String, List<AppKeybindingSnapshot>>{};
    for (final binding in settings.keybindings) {
      keybindingsByCategory.putIfAbsent(binding.category, () => <AppKeybindingSnapshot>[]).add(binding);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Persisted host preferences, camera bookmarks, and shortcut editing stay on the Rust settings path.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: [
            FilledButton.tonal(
              key: const ValueKey('settings-reset-command'),
              onPressed: enabled ? onResetSettings : null,
              child: const Text('Reset Settings'),
            ),
            OutlinedButton(
              key: const ValueKey('settings-export-command'),
              onPressed: enabled ? onExportSettings : null,
              child: const Text('Export Settings'),
            ),
            OutlinedButton(
              key: const ValueKey('settings-import-command'),
              onPressed: enabled ? onImportSettings : null,
              child: const Text('Import Settings'),
            ),
          ],
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Viewport And Autosave',
          child: Column(
            children: [
              SwitchListTile.adaptive(
                key: const ValueKey('settings-fps-overlay-toggle'),
                contentPadding: EdgeInsets.zero,
                value: settings.showFpsOverlay,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('show_fps_overlay', nextValue)
                    : null,
                title: const Text('Show FPS Overlay'),
              ),
              SwitchListTile.adaptive(
                key: const ValueKey('settings-node-labels-toggle'),
                contentPadding: EdgeInsets.zero,
                value: settings.showNodeLabels,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('show_node_labels', nextValue)
                    : null,
                title: const Text('Show Node Labels'),
              ),
              SwitchListTile.adaptive(
                key: const ValueKey('settings-bounding-box-toggle'),
                contentPadding: EdgeInsets.zero,
                value: settings.showBoundingBox,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('show_bounding_box', nextValue)
                    : null,
                title: const Text('Show Bounding Box'),
              ),
              SwitchListTile.adaptive(
                key: const ValueKey('settings-light-gizmos-toggle'),
                contentPadding: EdgeInsets.zero,
                value: settings.showLightGizmos,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('show_light_gizmos', nextValue)
                    : null,
                title: const Text('Show Light Gizmos'),
              ),
              SwitchListTile.adaptive(
                key: const ValueKey('settings-auto-save-toggle'),
                contentPadding: EdgeInsets.zero,
                value: settings.autoSaveEnabled,
                onChanged: enabled
                    ? (nextValue) => onSetToggle('auto_save_enabled', nextValue)
                    : null,
                title: const Text('Enable Auto-save'),
              ),
              _StepperRow(
                label: 'Auto-save Interval',
                valueLabel: '${settings.autoSaveIntervalSecs}s',
                decreaseKey: const ValueKey('settings-auto-save-interval-decrease'),
                increaseKey: const ValueKey('settings-auto-save-interval-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'auto_save_interval_secs',
                        settings.autoSaveIntervalSecs - 30,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'auto_save_interval_secs',
                        settings.autoSaveIntervalSecs + 30,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'Max Export Resolution',
                valueLabel: '${settings.maxExportResolution}^3',
                decreaseKey: const ValueKey('settings-max-export-resolution-decrease'),
                increaseKey: const ValueKey('settings-max-export-resolution-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'max_export_resolution',
                        settings.maxExportResolution - 64,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'max_export_resolution',
                        settings.maxExportResolution + 64,
                      )
                    : null,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _StepperRow(
                label: 'Max Sculpt Resolution',
                valueLabel: '${settings.maxSculptResolution}^3',
                decreaseKey: const ValueKey('settings-max-sculpt-resolution-decrease'),
                increaseKey: const ValueKey('settings-max-sculpt-resolution-increase'),
                onDecrease: enabled
                    ? () => onSetInteger(
                        'max_sculpt_resolution',
                        settings.maxSculptResolution - 16,
                      )
                    : null,
                onIncrease: enabled
                    ? () => onSetInteger(
                        'max_sculpt_resolution',
                        settings.maxSculptResolution + 16,
                      )
                    : null,
              ),
            ],
          ),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Camera Bookmarks',
          child: Column(
            children: [
              for (final bookmark in settings.cameraBookmarks) ...[
                _BookmarkRow(
                  bookmark: bookmark,
                  enabled: enabled,
                  onSave: () => onSaveCameraBookmark(bookmark.slotIndex),
                  onRestore: bookmark.saved
                      ? () => onRestoreCameraBookmark(bookmark.slotIndex)
                      : null,
                  onClear: bookmark.saved
                      ? () => onClearCameraBookmark(bookmark.slotIndex)
                      : null,
                ),
                if (bookmark != settings.cameraBookmarks.last)
                  const SizedBox(height: ShellTokens.compactGap),
              ],
            ],
          ),
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SectionCard(
          title: 'Keybindings',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Wrap(
                spacing: ShellTokens.controlGap,
                runSpacing: ShellTokens.controlGap,
                children: [
                  FilledButton.tonal(
                    key: const ValueKey('keymap-reset-command'),
                    onPressed: enabled ? onResetKeymap : null,
                    child: const Text('Reset Keymap'),
                  ),
                  OutlinedButton(
                    key: const ValueKey('keymap-export-command'),
                    onPressed: enabled ? onExportKeymap : null,
                    child: const Text('Export Keymap'),
                  ),
                  OutlinedButton(
                    key: const ValueKey('keymap-import-command'),
                    onPressed: enabled ? onImportKeymap : null,
                    child: const Text('Import Keymap'),
                  ),
                ],
              ),
              const SizedBox(height: ShellTokens.controlGap),
              for (final category in keybindingsByCategory.keys) ...[
                ExpansionTile(
                  key: ValueKey('keymap-category-$category'),
                  title: Text(category),
                  initiallyExpanded: category == 'General',
                  childrenPadding: EdgeInsets.zero,
                  children: keybindingsByCategory[category]!
                      .map(
                        (binding) => _KeybindingRow(
                          binding: binding,
                          enabled: enabled,
                          onEdit: enabled
                              ? () => _showKeybindingDialog(
                                  context: context,
                                  settings: settings,
                                  binding: binding,
                                )
                              : null,
                          onClear: enabled && binding.binding != null
                              ? () => onClearKeybinding(binding.actionId)
                              : null,
                        ),
                      )
                      .toList(growable: false),
                ),
                if (category != keybindingsByCategory.keys.last)
                  const SizedBox(height: ShellTokens.compactGap),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Future<void> _showKeybindingDialog({
    required BuildContext context,
    required AppSettingsSnapshot settings,
    required AppKeybindingSnapshot binding,
  }) async {
    if (settings.keyOptions.isEmpty) {
      return;
    }

    var selectedKeyId = binding.binding?.keyId ?? settings.keyOptions.first.id;
    var ctrl = binding.binding?.ctrl ?? false;
    var shift = binding.binding?.shift ?? false;
    var alt = binding.binding?.alt ?? false;

    final result = await showDialog<_KeybindingDraft>(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: Text('Edit ${binding.actionLabel}'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    DropdownButtonFormField<String>(
                      key: const ValueKey('keybinding-key-dropdown'),
                      initialValue: selectedKeyId,
                      decoration: const InputDecoration(labelText: 'Key'),
                      items: settings.keyOptions
                          .map(
                            (option) => DropdownMenuItem<String>(
                              value: option.id,
                              child: Text(option.label, key: ValueKey('keybinding-key-option-${option.id}')),
                            ),
                          )
                          .toList(growable: false),
                      onChanged: (nextValue) {
                        if (nextValue == null) {
                          return;
                        }
                        setState(() {
                          selectedKeyId = nextValue;
                        });
                      },
                    ),
                    const SizedBox(height: ShellTokens.controlGap),
                    CheckboxListTile(
                      key: const ValueKey('keybinding-ctrl-toggle'),
                      contentPadding: EdgeInsets.zero,
                      value: ctrl,
                      title: const Text('Ctrl'),
                      onChanged: (nextValue) {
                        setState(() {
                          ctrl = nextValue ?? false;
                        });
                      },
                    ),
                    CheckboxListTile(
                      key: const ValueKey('keybinding-shift-toggle'),
                      contentPadding: EdgeInsets.zero,
                      value: shift,
                      title: const Text('Shift'),
                      onChanged: (nextValue) {
                        setState(() {
                          shift = nextValue ?? false;
                        });
                      },
                    ),
                    CheckboxListTile(
                      key: const ValueKey('keybinding-alt-toggle'),
                      contentPadding: EdgeInsets.zero,
                      value: alt,
                      title: const Text('Alt'),
                      onChanged: (nextValue) {
                        setState(() {
                          alt = nextValue ?? false;
                        });
                      },
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  key: const ValueKey('keybinding-cancel-command'),
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  key: const ValueKey('keybinding-save-command'),
                  onPressed: () => Navigator.of(context).pop(
                    _KeybindingDraft(
                      keyId: selectedKeyId,
                      ctrl: ctrl,
                      shift: shift,
                      alt: alt,
                    ),
                  ),
                  child: const Text('Save'),
                ),
              ],
            );
          },
        );
      },
    );

    if (result == null) {
      return;
    }

    onSetKeybinding(
      binding.actionId,
      result.keyId,
      result.ctrl,
      result.shift,
      result.alt,
    );
  }
}

class _BookmarkRow extends StatelessWidget {
  const _BookmarkRow({
    required this.bookmark,
    required this.enabled,
    required this.onSave,
    required this.onRestore,
    required this.onClear,
  });

  final AppCameraBookmarkSnapshot bookmark;
  final bool enabled;
  final VoidCallback onSave;
  final VoidCallback? onRestore;
  final VoidCallback? onClear;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Slot ${bookmark.slotIndex + 1}',
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: ShellTokens.compactGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: [
            FilledButton.tonal(
              key: ValueKey('settings-bookmark-save-${bookmark.slotIndex}'),
              onPressed: enabled ? onSave : null,
              child: const Text('Save'),
            ),
            OutlinedButton(
              key: ValueKey('settings-bookmark-restore-${bookmark.slotIndex}'),
              onPressed: enabled ? onRestore : null,
              child: const Text('Restore'),
            ),
            OutlinedButton(
              key: ValueKey('settings-bookmark-clear-${bookmark.slotIndex}'),
              onPressed: enabled ? onClear : null,
              child: const Text('Clear'),
            ),
          ],
        ),
      ],
    );
  }
}

class _KeybindingRow extends StatelessWidget {
  const _KeybindingRow({
    required this.binding,
    required this.enabled,
    required this.onEdit,
    required this.onClear,
  });

  final AppKeybindingSnapshot binding;
  final bool enabled;
  final VoidCallback? onEdit;
  final VoidCallback? onClear;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      title: Text(binding.actionLabel),
      subtitle: Text(binding.binding?.shortcutLabel ?? 'Unbound'),
      trailing: Wrap(
        spacing: ShellTokens.compactGap,
        children: [
          OutlinedButton(
            key: ValueKey('keybinding-edit-${binding.actionId}'),
            onPressed: enabled ? onEdit : null,
            child: const Text('Edit'),
          ),
          OutlinedButton(
            key: ValueKey('keybinding-clear-${binding.actionId}'),
            onPressed: enabled ? onClear : null,
            child: const Text('Clear'),
          ),
        ],
      ),
    );
  }
}

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
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label),
              Text(valueLabel, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
        IconButton(
          key: decreaseKey,
          tooltip: 'Decrease $label',
          onPressed: onDecrease,
          icon: const Icon(Icons.remove_circle_outline),
        ),
        IconButton(
          key: increaseKey,
          tooltip: 'Increase $label',
          onPressed: onIncrease,
          icon: const Icon(Icons.add_circle_outline),
        ),
      ],
    );
  }
}

class _KeybindingDraft {
  const _KeybindingDraft({
    required this.keyId,
    required this.ctrl,
    required this.shift,
    required this.alt,
  });

  final String keyId;
  final bool ctrl;
  final bool shift;
  final bool alt;
}
