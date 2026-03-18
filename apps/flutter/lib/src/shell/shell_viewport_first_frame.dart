import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_context_shelf.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_favorite_commands_editor.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_quick_wheel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_tool_rail.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_utility_strip.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_workspace_bar.dart';

enum _ShellSurface { scene, properties }

class ShellViewportFirstFrame extends StatefulWidget {
  const ShellViewportFirstFrame({
    super.key,
    required this.shellLayout,
    required this.viewport,
    required this.workspace,
    required this.selectionContext,
    required this.document,
    required this.history,
    required this.settings,
    required this.renderSettings,
    required this.camera,
    required this.sculpt,
    required this.commands,
    required this.sceneDrawer,
    required this.propertiesPanel,
    required this.quickWheelRequestListenable,
    required this.enabled,
    required this.adaptiveInteractionResolutionEnabled,
    required this.onSelectWorkspace,
    required this.onExecuteCommand,
    required this.onOpenCommandSearch,
    required this.onUpdateShellPreferences,
    required this.onToggleAdaptiveInteractionResolution,
    required this.onSetRenderShadingMode,
    required this.onFrameAll,
    required this.onFocusSelected,
    required this.onToggleProjection,
    required this.onSetSculptBrushMode,
    required this.onSetSculptBrushRadius,
    required this.onSetSculptBrushStrength,
    required this.onSetSculptSymmetryAxis,
  });

  final ShellLayout shellLayout;
  final Widget viewport;
  final AppWorkspaceSnapshot workspace;
  final AppSelectionContextSnapshot selectionContext;
  final AppDocumentSnapshot? document;
  final AppHistorySnapshot? history;
  final AppSettingsSnapshot settings;
  final AppRenderSettingsSnapshot renderSettings;
  final AppCameraSnapshot camera;
  final AppSculptSnapshot sculpt;
  final List<AppCommandSnapshot> commands;
  final Widget sceneDrawer;
  final Widget propertiesPanel;
  final ValueListenable<int> quickWheelRequestListenable;
  final bool enabled;
  final bool adaptiveInteractionResolutionEnabled;
  final ValueChanged<String> onSelectWorkspace;
  final ValueChanged<String> onExecuteCommand;
  final VoidCallback onOpenCommandSearch;
  final ValueChanged<AppShellPreferencesUpdate> onUpdateShellPreferences;
  final ValueChanged<bool> onToggleAdaptiveInteractionResolution;
  final ValueChanged<String> onSetRenderShadingMode;
  final VoidCallback onFrameAll;
  final VoidCallback onFocusSelected;
  final VoidCallback onToggleProjection;
  final ValueChanged<String> onSetSculptBrushMode;
  final ValueChanged<double> onSetSculptBrushRadius;
  final ValueChanged<double> onSetSculptBrushStrength;
  final ValueChanged<String> onSetSculptSymmetryAxis;

  @override
  State<ShellViewportFirstFrame> createState() => _ShellViewportFirstFrameState();
}

class _ShellViewportFirstFrameState extends State<ShellViewportFirstFrame> {
  _ShellSurface? _transientSurface;
  _ShellSurface _lastPinnedSurface = _ShellSurface.scene;
  Offset? _quickWheelAnchor;
  double _lastMaxWidth = 0;
  int _lastQuickWheelRequest = 0;

  AppShellPreferencesSnapshot get _preferences => widget.settings.shellPreferences;

  bool get _leadingOnLeft => _preferences.leadingEdgeSide != 'right';

  @override
  void initState() {
    super.initState();
    _lastQuickWheelRequest = widget.quickWheelRequestListenable.value;
    widget.quickWheelRequestListenable.addListener(_handleQuickWheelRequest);
  }

  @override
  void didUpdateWidget(covariant ShellViewportFirstFrame oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.quickWheelRequestListenable !=
        widget.quickWheelRequestListenable) {
      oldWidget.quickWheelRequestListenable.removeListener(
        _handleQuickWheelRequest,
      );
      _lastQuickWheelRequest = widget.quickWheelRequestListenable.value;
      widget.quickWheelRequestListenable.addListener(_handleQuickWheelRequest);
    }
  }

  @override
  void dispose() {
    widget.quickWheelRequestListenable.removeListener(_handleQuickWheelRequest);
    super.dispose();
  }

  void _handleQuickWheelRequest() {
    final nextRequest = widget.quickWheelRequestListenable.value;
    if (nextRequest == _lastQuickWheelRequest) {
      return;
    }
    _lastQuickWheelRequest = nextRequest;
    _openQuickWheel();
  }

  void _openQuickWheel([Offset? anchor]) {
    if (!widget.enabled) {
      return;
    }
    if (!_preferences.quickWheelHintDismissed) {
      _dismissQuickWheelHint();
    }
    setState(() {
      _quickWheelAnchor = anchor ??
          Offset(
            _lastMaxWidth / 2,
            widget.shellLayout.useSidePanel ? 320 : 280,
          );
    });
  }

  void _closeQuickWheel() {
    setState(() {
      _quickWheelAnchor = null;
    });
  }

  void _dismissQuickWheelHint() {
    widget.onUpdateShellPreferences(
      const AppShellPreferencesUpdate(quickWheelHintDismissed: true),
    );
  }

  void _toggleLeadingEdgeSide() {
    widget.onUpdateShellPreferences(
      AppShellPreferencesUpdate(
        leadingEdgeSide: _leadingOnLeft ? 'right' : 'left',
      ),
    );
  }

  void _closeTransientSurfaces() {
    setState(() {
      _transientSurface = null;
    });
  }

  _EffectivePins _effectivePins(_ShellViewportMetrics metrics) {
    if (!widget.shellLayout.useSidePanel) {
      return const _EffectivePins(scenePinned: false, propertiesPinned: false);
    }

    var scenePinned = _preferences.desktopScenePinned;
    var propertiesPinned = _preferences.desktopPropertiesPinned;
    if (scenePinned &&
        propertiesPinned &&
        !metrics.pinnedViewportVisibleEnough(
          scenePinned: true,
          propertiesPinned: true,
        )) {
      if (_lastPinnedSurface == _ShellSurface.scene) {
        propertiesPinned = false;
      } else {
        scenePinned = false;
      }
    }

    return _EffectivePins(
      scenePinned: scenePinned,
      propertiesPinned: propertiesPinned,
    );
  }

  void _toggleSceneDrawer(_ShellViewportMetrics metrics) {
    final scenePinned = _effectivePins(metrics).scenePinned;
    if (scenePinned) {
      widget.onUpdateShellPreferences(
        const AppShellPreferencesUpdate(desktopScenePinned: false),
      );
      return;
    }
    setState(() {
      _transientSurface = _transientSurface == _ShellSurface.scene
          ? null
          : _ShellSurface.scene;
    });
  }

  void _toggleProperties(_ShellViewportMetrics metrics) {
    final propertiesPinned = _effectivePins(metrics).propertiesPinned;
    if (propertiesPinned) {
      widget.onUpdateShellPreferences(
        const AppShellPreferencesUpdate(desktopPropertiesPinned: false),
      );
      return;
    }
    setState(() {
      _transientSurface = _transientSurface == _ShellSurface.properties
          ? null
          : _ShellSurface.properties;
    });
  }

  void _togglePinnedSurface(_ShellSurface surface, _ShellViewportMetrics metrics) {
    if (!widget.shellLayout.useSidePanel) {
      return;
    }

    var scenePinned = _preferences.desktopScenePinned;
    var propertiesPinned = _preferences.desktopPropertiesPinned;
    final enablePin = surface == _ShellSurface.scene
        ? !scenePinned
        : !propertiesPinned;

    if (surface == _ShellSurface.scene) {
      scenePinned = enablePin;
    } else {
      propertiesPinned = enablePin;
    }

    if (enablePin &&
        scenePinned &&
        propertiesPinned &&
        !metrics.pinnedViewportVisibleEnough(
          scenePinned: scenePinned,
          propertiesPinned: propertiesPinned,
        )) {
      if (surface == _ShellSurface.scene) {
        propertiesPinned = false;
      } else {
        scenePinned = false;
      }
    }

    widget.onUpdateShellPreferences(
      AppShellPreferencesUpdate(
        desktopScenePinned: scenePinned,
        desktopPropertiesPinned: propertiesPinned,
      ),
    );
    setState(() {
      _lastPinnedSurface = surface;
      _transientSurface = enablePin ? surface : null;
    });
  }

  List<String> _favoriteCommandIdsForWorkspace() {
    return List<String>.from(
      _preferences.favoriteCommandIdsByWorkspace[widget.workspace.id] ??
          const <String>[],
    );
  }

  void _toggleFavoriteCommand(String commandId) {
    final nextFavoriteIds = _favoriteCommandIdsForWorkspace();
    if (nextFavoriteIds.contains(commandId)) {
      nextFavoriteIds.remove(commandId);
    } else {
      nextFavoriteIds.add(commandId);
    }
    _updateFavoriteCommands(nextFavoriteIds);
  }

  void _updateFavoriteCommands(List<String> favoriteCommandIds) {
    final favoriteCommandIdsByWorkspace = _preferences.favoriteCommandIdsByWorkspace
        .map(
          (workspaceId, commandIds) => MapEntry(
            workspaceId,
            List<String>.from(commandIds),
          ),
        );
    favoriteCommandIdsByWorkspace[widget.workspace.id] = favoriteCommandIds;

    widget.onUpdateShellPreferences(
      AppShellPreferencesUpdate(
        favoriteCommandIdsByWorkspace: favoriteCommandIdsByWorkspace,
      ),
    );
  }

  Future<void> _editFavorites() async {
    await showDialog<void>(
      context: context,
      builder: (context) {
        return Dialog(
          child: ShellFavoriteCommandsEditor(
            workspace: widget.workspace,
            commands: widget.commands,
            favoriteCommandIds: _favoriteCommandIdsForWorkspace(),
            onApply: _updateFavoriteCommands,
          ),
        );
      },
    );
  }

  List<ShellQuickWheelAction> _quickWheelActions() {
    final favoriteCommandIds = _favoriteCommandIdsForWorkspace();
    final seenCommandIds = <String>{};
    final actions = <ShellQuickWheelAction>[
      for (final action in widget.selectionContext.quickActions)
        if (seenCommandIds.add(action.id))
          ShellQuickWheelAction(
            id: action.id,
            label: action.label,
            enabled: widget.enabled && action.enabled,
            favorite: favoriteCommandIds.contains(action.id),
            shortcutLabel: action.shortcutLabel,
          ),
      for (final commandId in favoriteCommandIds)
        for (final command in widget.commands.where((item) => item.id == commandId))
          if (seenCommandIds.add(command.id))
            ShellQuickWheelAction(
              id: command.id,
              label: command.label,
              enabled: widget.enabled && command.enabled,
              favorite: true,
              shortcutLabel: command.shortcutLabel,
            ),
    ];

    return actions;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        _lastMaxWidth = constraints.maxWidth;
        final metrics = _ShellViewportMetrics.fromConstraints(
          shellLayout: widget.shellLayout,
          constraints: constraints,
          sculptActive: widget.sculpt.session != null,
        );
        final pins = _effectivePins(metrics);
        final sceneVisible =
            pins.scenePinned || _transientSurface == _ShellSurface.scene;
        final propertiesVisible =
            pins.propertiesPinned || _transientSurface == _ShellSurface.properties;
        final showTransientBarrier =
            _transientSurface != null && !(pins.scenePinned && pins.propertiesPinned);

        return Stack(
          fit: StackFit.expand,
          children: [
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.translucent,
                onLongPressStart: (details) =>
                    _openQuickWheel(details.localPosition),
                onSecondaryTapDown: (details) =>
                    _openQuickWheel(details.localPosition),
                child: widget.viewport,
              ),
            ),
            if (showTransientBarrier)
              Positioned.fill(
                child: GestureDetector(
                  onTap: _closeTransientSurfaces,
                  behavior: HitTestBehavior.opaque,
                  child: const SizedBox.expand(),
                ),
              ),
            _buildTopBar(metrics),
            _buildToolRail(metrics, sceneVisible),
            _buildUtilityStrip(metrics, propertiesVisible),
            _buildSceneSurface(metrics, visible: sceneVisible, pinned: pins.scenePinned),
            _buildPropertiesSurface(
              metrics,
              visible: propertiesVisible,
              pinned: pins.propertiesPinned,
            ),
            if (!sceneVisible) _buildSceneHandle(metrics),
            _buildBottomDock(
              metrics,
              sceneVisible: sceneVisible,
              propertiesVisible: propertiesVisible,
            ),
            if (!_preferences.quickWheelHintDismissed)
              _buildQuickWheelHint(metrics),
            if (_quickWheelAnchor != null)
              ShellQuickWheel(
                anchor: _quickWheelAnchor!,
                actions: _quickWheelActions(),
                onDismiss: _closeQuickWheel,
                onExecute: widget.onExecuteCommand,
                onToggleFavorite: _toggleFavoriteCommand,
              ),
          ],
        );
      },
    );
  }

  Widget _buildTopBar(_ShellViewportMetrics metrics) {
    return Align(
      key: const ValueKey('shell-top-bar'),
      alignment: Alignment.topCenter,
      child: Padding(
        padding: EdgeInsets.only(
          left: metrics.edgeInset,
          top: metrics.topInset,
          right: metrics.edgeInset,
        ),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: metrics.topBarMaxWidth),
          child: ShellWorkspaceBar(
            workspace: widget.workspace,
            selectionContext: widget.selectionContext,
            document: widget.document,
            history: widget.history,
            enabled: widget.enabled,
            onSelectWorkspace: widget.onSelectWorkspace,
            onOpenCommandSearch: widget.onOpenCommandSearch,
            onUndo: () => widget.onExecuteCommand('undo'),
            onRedo: () => widget.onExecuteCommand('redo'),
          ),
        ),
      ),
    );
  }

  Widget _buildToolRail(_ShellViewportMetrics metrics, bool sceneVisible) {
    return Positioned(
      key: const ValueKey('shell-leading-rail'),
      left: _leadingOnLeft ? metrics.edgeInset : null,
      right: _leadingOnLeft ? null : metrics.edgeInset,
      top: metrics.sidePanelTop,
      bottom: metrics.sidePanelBottom,
      child: ShellToolRail(
        currentWorkspaceId: widget.workspace.id,
        enabled: widget.enabled,
        sceneDrawerOpen: sceneVisible,
        leadingEdgeSide: _preferences.leadingEdgeSide,
        onSelectWorkspace: widget.onSelectWorkspace,
        onToggleSceneDrawer: () => _toggleSceneDrawer(metrics),
        onToggleLeadingEdgeSide: _toggleLeadingEdgeSide,
      ),
    );
  }

  Widget _buildUtilityStrip(
    _ShellViewportMetrics metrics,
    bool propertiesVisible,
  ) {
    return Positioned(
      key: const ValueKey('shell-trailing-utility-strip'),
      left: _leadingOnLeft ? null : metrics.edgeInset,
      right: _leadingOnLeft ? metrics.edgeInset : null,
      top: metrics.sidePanelTop,
      bottom: metrics.sidePanelBottom,
      child: ShellUtilityStrip(
        renderSettings: widget.renderSettings,
        selectionContext: widget.selectionContext,
        orthographic: widget.camera.orthographic,
        propertiesOpen: propertiesVisible,
        propertiesPinned: _effectivePins(metrics).propertiesPinned,
        canPinProperties: widget.shellLayout.useSidePanel,
        enabled: widget.enabled,
        adaptiveInteractionResolutionEnabled:
            widget.adaptiveInteractionResolutionEnabled,
        onOpenProperties: () => _toggleProperties(metrics),
        onTogglePropertiesPin: () =>
            _togglePinnedSurface(_ShellSurface.properties, metrics),
        onFrameAll: widget.onFrameAll,
        onFocusSelected: widget.onFocusSelected,
        onToggleProjection: widget.onToggleProjection,
        onSetRenderShadingMode: widget.onSetRenderShadingMode,
        onToggleAdaptiveInteractionResolution:
            widget.onToggleAdaptiveInteractionResolution,
      ),
    );
  }

  Widget _buildSceneSurface(
    _ShellViewportMetrics metrics, {
    required bool visible,
    required bool pinned,
  }) {
    final drawerChild = switch (_preferences.preferredDrawerTab) {
      'layers' => const _ScenePlaceholder(
          title: 'Layers',
          detail:
              'Layer stacks are reserved in Batch 1 so the information architecture stays stable before authoring lands.',
        ),
      'sets' => const _ScenePlaceholder(
          title: 'Sets',
          detail:
              'Selection sets are intentionally deferred, but the drawer tab is reserved now.',
        ),
      _ => widget.sceneDrawer,
    };

    return Positioned(
      key: const ValueKey('shell-scene-drawer'),
      left: _leadingOnLeft
          ? metrics.edgeInset + metrics.railWidth + metrics.panelGap
          : null,
      right: _leadingOnLeft
          ? null
          : metrics.edgeInset + metrics.railWidth + metrics.panelGap,
      top: metrics.sidePanelTop,
      bottom: metrics.sidePanelBottom,
      width: metrics.sceneDrawerWidth,
      child: IgnorePointer(
        ignoring: !visible,
        child: AnimatedSlide(
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOutCubic,
          offset: visible
              ? Offset.zero
              : (_leadingOnLeft
                    ? const Offset(-1.08, 0)
                    : const Offset(1.08, 0)),
          child: AnimatedOpacity(
            duration: const Duration(milliseconds: 180),
            opacity: visible ? 1 : 0,
            child: ShellPanelSurface(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        'Scene',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const Spacer(),
                      if (widget.shellLayout.useSidePanel)
                        IconButton(
                          key: const ValueKey('shell-scene-pin-toggle'),
                          tooltip: pinned
                              ? 'Unpin scene drawer'
                              : 'Pin scene drawer',
                          onPressed: () =>
                              _togglePinnedSurface(_ShellSurface.scene, metrics),
                          icon: Icon(
                            pinned ? Icons.push_pin : Icons.push_pin_outlined,
                          ),
                        ),
                      IconButton(
                        tooltip: 'Close scene drawer',
                        onPressed: _closeTransientSurfaces,
                        icon: const Icon(Icons.close),
                      ),
                    ],
                  ),
                  const SizedBox(height: ShellTokens.compactGap),
                  Wrap(
                    spacing: ShellTokens.compactGap,
                    children: [
                      for (final tab in const [
                        ('scene', 'Scene'),
                        ('layers', 'Layers'),
                        ('sets', 'Sets'),
                      ])
                        ChoiceChip(
                          label: Text(tab.$2),
                          selected: _preferences.preferredDrawerTab == tab.$1,
                          onSelected: (_) {
                            widget.onUpdateShellPreferences(
                              AppShellPreferencesUpdate(
                                preferredDrawerTab: tab.$1,
                              ),
                            );
                          },
                        ),
                    ],
                  ),
                  const SizedBox(height: ShellTokens.controlGap),
                  Expanded(child: drawerChild),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPropertiesSurface(
    _ShellViewportMetrics metrics, {
    required bool visible,
    required bool pinned,
  }) {
    return Positioned(
      key: const ValueKey('shell-properties-drawer'),
      left: _leadingOnLeft
          ? null
          : metrics.edgeInset + metrics.utilityWidth + metrics.panelGap,
      right: _leadingOnLeft
          ? metrics.edgeInset + metrics.utilityWidth + metrics.panelGap
          : null,
      top: metrics.sidePanelTop,
      bottom: metrics.sidePanelBottom,
      width: metrics.propertiesDrawerWidth,
      child: IgnorePointer(
        ignoring: !visible,
        child: AnimatedSlide(
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOutCubic,
          offset: visible
              ? Offset.zero
              : (_leadingOnLeft
                    ? const Offset(1.08, 0)
                    : const Offset(-1.08, 0)),
          child: AnimatedOpacity(
            duration: const Duration(milliseconds: 180),
            opacity: visible ? 1 : 0,
            child: Stack(
              children: [
                Positioned.fill(child: widget.propertiesPanel),
                Positioned(
                  top: 12,
                  left: 12,
                  right: 12,
                  child: Row(
                    children: [
                      DecoratedBox(
                        decoration: ShellSurfaceStyles.overlayPanel(
                          context,
                          accentColor: context.shellPalette.infoAccent,
                          pill: true,
                        ),
                        child: const Padding(
                          padding: EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                          child: Text('Properties'),
                        ),
                      ),
                      const Spacer(),
                      if (widget.shellLayout.useSidePanel)
                        IconButton.filledTonal(
                          key: const ValueKey('shell-properties-pin-toggle'),
                          tooltip: pinned
                              ? 'Unpin properties drawer'
                              : 'Pin properties drawer',
                          onPressed: () => _togglePinnedSurface(
                            _ShellSurface.properties,
                            metrics,
                          ),
                          icon: Icon(
                            pinned ? Icons.push_pin : Icons.push_pin_outlined,
                          ),
                        ),
                      const SizedBox(width: 6),
                      IconButton.filledTonal(
                        tooltip: 'Close properties drawer',
                        onPressed: _closeTransientSurfaces,
                        icon: const Icon(Icons.close),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSceneHandle(_ShellViewportMetrics metrics) {
    return Positioned(
      key: const ValueKey('shell-scene-edge-handle'),
      left: _leadingOnLeft ? metrics.edgeInset + metrics.railWidth + 4 : null,
      right: _leadingOnLeft ? null : metrics.edgeInset + metrics.railWidth + 4,
      top: metrics.sidePanelTop + 120,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => _toggleSceneDrawer(metrics),
          borderRadius: BorderRadius.circular(999),
          child: DecoratedBox(
            decoration: ShellSurfaceStyles.overlayPanel(
              context,
              accentColor: context.shellPalette.infoAccent,
              pill: true,
            ),
            child: const Padding(
              padding: EdgeInsets.symmetric(horizontal: 10, vertical: 18),
              child: Icon(Icons.chevron_right),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomDock(
    _ShellViewportMetrics metrics, {
    required bool sceneVisible,
    required bool propertiesVisible,
  }) {
    return Align(
      key: const ValueKey('shell-bottom-dock-container'),
      alignment: Alignment.bottomCenter,
      child: Padding(
        padding: EdgeInsets.only(
          left: metrics.edgeInset,
          right: metrics.edgeInset,
          bottom: metrics.bottomInset,
        ),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: metrics.bottomDockMaxWidth),
          child: ShellContextShelf(
            workspace: widget.workspace,
            selectionContext: widget.selectionContext,
            shellPreferences: _preferences,
            commands: widget.commands,
            sculpt: widget.sculpt,
            enabled: widget.enabled,
            sceneDrawerOpen: sceneVisible,
            propertiesOpen: propertiesVisible,
            onExecuteCommand: widget.onExecuteCommand,
            onToggleFavoriteCommand: _toggleFavoriteCommand,
            onToggleSceneDrawer: () => _toggleSceneDrawer(metrics),
            onToggleProperties: () => _toggleProperties(metrics),
            onOpenCommandSearch: widget.onOpenCommandSearch,
            onOpenQuickWheel: _openQuickWheel,
            onEditFavorites: _editFavorites,
            onSetSculptBrushMode: widget.onSetSculptBrushMode,
            onSetSculptBrushRadius: widget.onSetSculptBrushRadius,
            onSetSculptBrushStrength: widget.onSetSculptBrushStrength,
            onSetSculptSymmetryAxis: widget.onSetSculptSymmetryAxis,
          ),
        ),
      ),
    );
  }

  Widget _buildQuickWheelHint(_ShellViewportMetrics metrics) {
    return Positioned(
      key: const ValueKey('shell-quick-wheel-hint'),
      left: _leadingOnLeft ? metrics.edgeInset + metrics.railWidth + 16 : null,
      right: _leadingOnLeft ? null : metrics.edgeInset + metrics.railWidth + 16,
      bottom: metrics.bottomInset + metrics.quickWheelHintBottomOffset,
      child: DecoratedBox(
        decoration: ShellSurfaceStyles.overlayPanel(
          context,
          accentColor: context.shellPalette.warningAccent,
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.touch_app_outlined),
              const SizedBox(width: 10),
              Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Quick wheel',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          color: context.shellPalette.overlayText,
                        ),
                  ),
                  Text(
                    'Long press the canvas or press Q.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: context.shellPalette.overlayMutedText,
                        ),
                  ),
                ],
              ),
              const SizedBox(width: 10),
              IconButton(
                tooltip: 'Dismiss hint',
                onPressed: _dismissQuickWheelHint,
                icon: const Icon(Icons.close),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ScenePlaceholder extends StatelessWidget {
  const _ScenePlaceholder({required this.title, required this.detail});

  final String title;
  final String detail;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(
                detail,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _EffectivePins {
  const _EffectivePins({
    required this.scenePinned,
    required this.propertiesPinned,
  });

  final bool scenePinned;
  final bool propertiesPinned;
}

class _ShellViewportMetrics {
  const _ShellViewportMetrics({
    required this.edgeInset,
    required this.topInset,
    required this.bottomInset,
    required this.topBarMaxWidth,
    required this.bottomDockMaxWidth,
    required this.railWidth,
    required this.utilityWidth,
    required this.sceneDrawerWidth,
    required this.propertiesDrawerWidth,
    required this.sidePanelTop,
    required this.sidePanelBottom,
    required this.panelGap,
    required this.totalWidth,
    required this.quickWheelHintBottomOffset,
  });

  final double edgeInset;
  final double topInset;
  final double bottomInset;
  final double topBarMaxWidth;
  final double bottomDockMaxWidth;
  final double railWidth;
  final double utilityWidth;
  final double sceneDrawerWidth;
  final double propertiesDrawerWidth;
  final double sidePanelTop;
  final double sidePanelBottom;
  final double panelGap;
  final double totalWidth;
  final double quickWheelHintBottomOffset;

  factory _ShellViewportMetrics.fromConstraints({
    required ShellLayout shellLayout,
    required BoxConstraints constraints,
    required bool sculptActive,
  }) {
    final isDesktop = shellLayout.useSidePanel;
    final edgeInset = isDesktop ? 18.0 : 12.0;
    final topInset = isDesktop ? 14.0 : 10.0;
    final bottomInset = isDesktop ? 18.0 : 12.0;
    final railWidth = 80.0;
    final utilityWidth = 88.0;
    final sceneDrawerWidth = isDesktop
        ? 320.0
        : math.min(320.0, constraints.maxWidth * 0.72);
    final propertiesDrawerWidth = isDesktop
        ? 360.0
        : math.min(340.0, constraints.maxWidth * 0.78);
    final topBarMaxWidth = math.min(
      constraints.maxWidth - (edgeInset * 2),
      isDesktop ? 1180.0 : 980.0,
    );
    final bottomDockMaxWidth = math.min(
      constraints.maxWidth - (edgeInset * 2),
      isDesktop ? 1160.0 : 980.0,
    );
    final sidePanelTop = topInset + 82.0;
    final sidePanelBottom = bottomInset + (sculptActive ? 188.0 : 164.0);

    return _ShellViewportMetrics(
      edgeInset: edgeInset,
      topInset: topInset,
      bottomInset: bottomInset,
      topBarMaxWidth: topBarMaxWidth,
      bottomDockMaxWidth: bottomDockMaxWidth,
      railWidth: railWidth,
      utilityWidth: utilityWidth,
      sceneDrawerWidth: sceneDrawerWidth,
      propertiesDrawerWidth: propertiesDrawerWidth,
      sidePanelTop: sidePanelTop,
      sidePanelBottom: sidePanelBottom,
      panelGap: shellLayout.panelGap,
      totalWidth: constraints.maxWidth,
      quickWheelHintBottomOffset: sculptActive ? 196.0 : 168.0,
    );
  }

  bool pinnedViewportVisibleEnough({
    required bool scenePinned,
    required bool propertiesPinned,
  }) {
    var occupiedWidth = railWidth + utilityWidth + (panelGap * 2);
    if (scenePinned) {
      occupiedWidth += sceneDrawerWidth + panelGap;
    }
    if (propertiesPinned) {
      occupiedWidth += propertiesDrawerWidth + panelGap;
    }
    final visibleViewportWidth = totalWidth - (edgeInset * 2) - occupiedWidth;
    return visibleViewportWidth / totalWidth >= 0.55;
  }
}
