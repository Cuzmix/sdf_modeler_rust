import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_tree_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_command_strip.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_desktop_side_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_modal_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_stacked_panes.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_bridge.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_event.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

enum _BridgeModalPanel { commands }

class _CreateNodeOption {
  const _CreateNodeOption({required this.id, required this.label});

  final String id;
  final String label;
}

const List<_CreateNodeOption> _operationOptions = <_CreateNodeOption>[
  _CreateNodeOption(id: 'union', label: 'Union'),
  _CreateNodeOption(id: 'smooth_union', label: 'Smooth Union'),
  _CreateNodeOption(id: 'subtract', label: 'Subtract'),
  _CreateNodeOption(id: 'intersect', label: 'Intersect'),
  _CreateNodeOption(id: 'smooth_subtract', label: 'Smooth Subtract'),
  _CreateNodeOption(id: 'smooth_intersect', label: 'Smooth Intersect'),
  _CreateNodeOption(id: 'chamfer_union', label: 'Chamfer Union'),
  _CreateNodeOption(id: 'chamfer_subtract', label: 'Chamfer Subtract'),
  _CreateNodeOption(id: 'chamfer_intersect', label: 'Chamfer Intersect'),
  _CreateNodeOption(id: 'stairs_union', label: 'Stairs Union'),
  _CreateNodeOption(id: 'stairs_subtract', label: 'Stairs Subtract'),
  _CreateNodeOption(id: 'columns_union', label: 'Columns Union'),
  _CreateNodeOption(id: 'columns_subtract', label: 'Columns Subtract'),
];

const List<_CreateNodeOption> _modifierOptions = <_CreateNodeOption>[
  _CreateNodeOption(id: 'twist', label: 'Twist'),
  _CreateNodeOption(id: 'bend', label: 'Bend'),
  _CreateNodeOption(id: 'taper', label: 'Taper'),
  _CreateNodeOption(id: 'round', label: 'Round'),
  _CreateNodeOption(id: 'onion', label: 'Onion'),
  _CreateNodeOption(id: 'elongate', label: 'Elongate'),
  _CreateNodeOption(id: 'mirror', label: 'Mirror'),
  _CreateNodeOption(id: 'repeat', label: 'Repeat'),
  _CreateNodeOption(id: 'finite_repeat', label: 'Finite Repeat'),
  _CreateNodeOption(id: 'radial_repeat', label: 'Radial Repeat'),
  _CreateNodeOption(id: 'offset', label: 'Offset'),
  _CreateNodeOption(id: 'noise', label: 'Noise'),
];

class SdfModelerApp extends StatelessWidget {
  const SdfModelerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SDF Modeler Flutter',
      theme: buildTouchFirstShellTheme(),
      home: const BridgeStatusPage(),
    );
  }
}

class BridgeStatusPage extends StatefulWidget {
  const BridgeStatusPage({super.key});

  @override
  State<BridgeStatusPage> createState() => _BridgeStatusPageState();
}

class _BridgeStatusPageState extends State<BridgeStatusPage> {
  static const int _defaultFrameWidth = 640;
  static const int _defaultFrameHeight = 360;
  static const int _minimumFrameWidth = 320;
  static const int _minimumFrameHeight = 180;
  static const int _maximumFrameDimension = 4096;
  static const int _frameDimensionAlignment = 8;
  static const double _steadyStateRenderScale = 1.0;
  static const double _interactionRenderScaleCap = 0.65;
  static const double _frameRateSmoothingFactor = 0.18;

  String _statusLine = 'Checking Rust bridge...';
  String _versionLine = '';
  String _previewLine = 'Initializing viewport texture...';

  AppSceneSnapshot? _sceneSnapshot;
  TextureViewportFeedback? _viewportFeedback;
  int? _textureId;
  StreamSubscription<TextureViewportEvent>? _textureEventSubscription;
  Timer? _interactionCooldownTimer;
  bool _commandInFlight = false;
  bool _adaptiveInteractionResolutionEnabled = false;
  bool _viewportInteractionActive = false;
  int _frameWidth = _defaultFrameWidth;
  int _frameHeight = _defaultFrameHeight;
  int _nativeFrameWidth = 0;
  int _nativeFrameHeight = 0;
  int _nativeFrameCount = 0;
  int _droppedFrameCount = 0;
  double? _lastNativeFrameTimeMs;
  double? _smoothedFramesPerSecond;
  String _interactionPhase = 'idle';
  Size _lastLogicalViewportSize = Size.zero;
  double _lastDevicePixelRatio = 1.0;
  _BridgeModalPanel? _activeModalPanel;

  @override
  void initState() {
    super.initState();
    _textureEventSubscription = TextureBridge.instance.events.listen(
      _handleTextureEvent,
      onError: _handleTextureEventError,
    );
    _initializeBridge();
  }

  Future<void> _initializeBridge() async {
    try {
      final pingValue = ping();
      final versionValue = bridgeVersion();
      final createdTextureId = await TextureBridge.instance.createTexture(
        width: _frameWidth,
        height: _frameHeight,
      );
      final snapshot = _decodeSnapshot(sceneSnapshotJson());
      final viewportFeedback = TextureViewportFeedback.fromSceneSnapshot(snapshot);

      if (!mounted) {
        await TextureBridge.instance.disposeTexture(createdTextureId);
        return;
      }

      setState(() {
        _statusLine = 'Rust ping: $pingValue';
        _versionLine = 'Bridge crate version: $versionValue';
        _textureId = createdTextureId;
        _sceneSnapshot = snapshot;
        _viewportFeedback = viewportFeedback;
        _previewLine = _buildPreviewLine();
      });

      _requestNativeFrame(textureId: createdTextureId);
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _statusLine = 'Rust bridge error: $error';
        _versionLine = '';
        _previewLine = 'Viewport stopped because initialization failed.';
      });
    }
  }

  AppSceneSnapshot _decodeSnapshot(String rawJson) {
    final decoded = jsonDecode(rawJson) as Map<String, dynamic>;
    return AppSceneSnapshot.fromJson(decoded);
  }

  double get _effectiveRenderScale {
    if (_adaptiveInteractionResolutionEnabled && _viewportInteractionActive) {
      return math.min(_steadyStateRenderScale, _interactionRenderScaleCap);
    }

    return _steadyStateRenderScale;
  }

  String _buildPreviewLine() {
    final scalePercent = (_effectiveRenderScale * 100).round();
    final interactionLabel = _viewportInteractionActive ? ' interactive' : '';
    final base =
        'Viewport render target: ${_frameWidth}x$_frameHeight at $scalePercent% scale$interactionLabel';
    if (_nativeFrameCount == 0 || _lastNativeFrameTimeMs == null) {
      return '$base (native host render loop, phase $_interactionPhase).';
    }

    final nativeSize = _nativeFrameWidth > 0 && _nativeFrameHeight > 0
        ? '${_nativeFrameWidth}x$_nativeFrameHeight'
        : '${_frameWidth}x$_frameHeight';
    final fpsSegment = _smoothedFramesPerSecond != null
        ? '${_smoothedFramesPerSecond!.toStringAsFixed(1)} FPS, '
        : '';
    return '$base (native $nativeSize, $fpsSegment${_lastNativeFrameTimeMs!.toStringAsFixed(1)} ms, $_nativeFrameCount frames, dropped $_droppedFrameCount, phase $_interactionPhase).';
  }

  void _handleTextureEvent(TextureViewportEvent event) {
    final activeTextureId = _textureId;
    if (activeTextureId == null || event.textureId != activeTextureId) {
      return;
    }

    if (!mounted) {
      return;
    }

    final instantaneousFramesPerSecond = _framesPerSecondFromFrameTime(
      event.frameTimeMs,
    );

    setState(() {
      _nativeFrameWidth = event.frameWidth;
      _nativeFrameHeight = event.frameHeight;
      _nativeFrameCount = event.frameCount;
      _droppedFrameCount = event.droppedFrameCount;
      _lastNativeFrameTimeMs = event.frameTimeMs;
      _smoothedFramesPerSecond = _nextSmoothedFramesPerSecond(
        _smoothedFramesPerSecond,
        instantaneousFramesPerSecond,
      );
      _interactionPhase = event.interactionPhase;
      if (event.feedback != null) {
        _viewportFeedback = event.feedback;
      }
      _previewLine = _buildPreviewLine();
    });
  }

  double? _framesPerSecondFromFrameTime(double frameTimeMs) {
    if (frameTimeMs <= 0.0) {
      return null;
    }

    return 1000.0 / frameTimeMs;
  }

  double? _nextSmoothedFramesPerSecond(
    double? currentFramesPerSecond,
    double? nextFramesPerSecond,
  ) {
    if (nextFramesPerSecond == null) {
      return currentFramesPerSecond;
    }
    if (currentFramesPerSecond == null) {
      return nextFramesPerSecond;
    }

    return (currentFramesPerSecond * (1.0 - _frameRateSmoothingFactor)) +
        (nextFramesPerSecond * _frameRateSmoothingFactor);
  }

  void _handleTextureEventError(Object error) {
    if (!mounted) {
      return;
    }

    setState(() {
      _previewLine = 'Viewport event error: $error';
    });
  }

  void _refreshBridgeStatus() {
    try {
      final pingValue = ping();
      final versionValue = bridgeVersion();
      final snapshot = _decodeSnapshot(sceneSnapshotJson());
      setState(() {
        _statusLine = 'Rust ping: $pingValue';
        _versionLine = 'Bridge crate version: $versionValue';
        _sceneSnapshot = snapshot;
        _viewportFeedback = TextureViewportFeedback.fromSceneSnapshot(snapshot);
      });
    } catch (error) {
      setState(() {
        _statusLine = 'Rust bridge error: $error';
        _versionLine = '';
      });
    }
  }

  void _beginViewportInteraction() {
    _interactionCooldownTimer?.cancel();

    if (!_viewportInteractionActive) {
      _viewportInteractionActive = true;
      final renderTargetChanged = _updateRenderTargetSize();
      if (!renderTargetChanged && mounted) {
        setState(() {
          _previewLine = _buildPreviewLine();
        });
      }
    }

    _interactionCooldownTimer = Timer(
      ShellGestureContract.viewportInteractionCooldown,
      _endViewportInteraction,
    );
  }

  void _endViewportInteraction() {
    _interactionCooldownTimer?.cancel();
    _interactionCooldownTimer = null;

    if (!_viewportInteractionActive) {
      return;
    }

    _viewportInteractionActive = false;
    final renderTargetChanged = _updateRenderTargetSize();
    if (!renderTargetChanged && mounted) {
      setState(() {
        _previewLine = _buildPreviewLine();
      });
    }
  }

  void _toggleAdaptiveInteractionResolution(bool enabled) {
    if (_adaptiveInteractionResolutionEnabled == enabled) {
      return;
    }

    setState(() {
      _adaptiveInteractionResolutionEnabled = enabled;
    });

    final renderTargetChanged = _updateRenderTargetSize();
    if (!renderTargetChanged && mounted) {
      setState(() {
        _previewLine = _buildPreviewLine();
      });
    }
  }

  void _dispatchTextureCommand({
    required Future<void> Function(int textureId) command,
    required String errorPrefix,
    int? textureId,
  }) {
    final activeTextureId = textureId ?? _textureId;
    if (activeTextureId == null) {
      return;
    }

    unawaited(_performTextureCommand(activeTextureId, command, errorPrefix));
  }

  Future<void> _performTextureCommand(
    int textureId,
    Future<void> Function(int textureId) command,
    String errorPrefix,
  ) async {
    try {
      await command(textureId);
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = '$errorPrefix: $error';
      });
    }
  }

  void _requestNativeFrame({int? textureId}) {
    _dispatchTextureCommand(
      textureId: textureId,
      errorPrefix: 'Viewport request error',
      command: (activeTextureId) =>
          TextureBridge.instance.requestFrame(textureId: activeTextureId),
    );
  }

  void _syncNativeViewportSize() {
    final activeTextureId = _textureId;
    if (activeTextureId == null) {
      return;
    }

    final targetWidth = _frameWidth;
    final targetHeight = _frameHeight;
    _dispatchTextureCommand(
      textureId: activeTextureId,
      errorPrefix: 'Viewport resize error',
      command: (textureId) async {
        await TextureBridge.instance.setTextureSize(
          textureId: textureId,
          width: targetWidth,
          height: targetHeight,
        );
        await TextureBridge.instance.requestFrame(textureId: textureId);
      },
    );
  }

  void _handleViewportOrbitDrag(Offset delta) {
    _beginViewportInteraction();
    _dispatchTextureCommand(
      errorPrefix: 'Viewport orbit error',
      command: (textureId) => TextureBridge.instance.orbitCamera(
        textureId: textureId,
        deltaX: delta.dx,
        deltaY: delta.dy,
      ),
    );
  }

  void _handleViewportPanDrag(Offset delta) {
    _beginViewportInteraction();
    _dispatchTextureCommand(
      errorPrefix: 'Viewport pan error',
      command: (textureId) => TextureBridge.instance.panCamera(
        textureId: textureId,
        deltaX: delta.dx,
        deltaY: delta.dy,
      ),
    );
  }

  void _handleViewportScroll(double deltaY) {
    _beginViewportInteraction();
    _dispatchTextureCommand(
      errorPrefix: 'Viewport zoom error',
      command: (textureId) => TextureBridge.instance.zoomCamera(
        textureId: textureId,
        delta: deltaY,
      ),
    );
  }

  void _handleViewportPrimaryTap(
    Offset localPosition,
    Size logicalViewportSize,
  ) {
    final normalizedX = _normalizeViewportCoordinate(
      localPosition.dx,
      logicalViewportSize.width,
    );
    final normalizedY = _normalizeViewportCoordinate(
      localPosition.dy,
      logicalViewportSize.height,
    );

    _dispatchTextureCommand(
      errorPrefix: 'Viewport pick error',
      command: (textureId) => TextureBridge.instance.pickNode(
        textureId: textureId,
        normalizedX: normalizedX,
        normalizedY: normalizedY,
      ),
    );
  }

  void _handleViewportHover(
    Offset localPosition,
    Size logicalViewportSize,
  ) {
    final normalizedX = _normalizeViewportCoordinate(
      localPosition.dx,
      logicalViewportSize.width,
    );
    final normalizedY = _normalizeViewportCoordinate(
      localPosition.dy,
      logicalViewportSize.height,
    );

    _dispatchTextureCommand(
      errorPrefix: 'Viewport hover error',
      command: (textureId) => TextureBridge.instance.hoverNode(
        textureId: textureId,
        normalizedX: normalizedX,
        normalizedY: normalizedY,
      ),
    );
  }

  void _handleViewportHoverExit() {
    _dispatchTextureCommand(
      errorPrefix: 'Viewport hover clear error',
      command: (textureId) => TextureBridge.instance.clearHover(
        textureId: textureId,
      ),
    );
  }

  void _handleViewportInteractionEnd() {}

  void _openCommandPanel() {
    if (_activeModalPanel == _BridgeModalPanel.commands) {
      return;
    }

    setState(() {
      _activeModalPanel = _BridgeModalPanel.commands;
    });
  }

  void _closeCommandPanel() {
    if (_activeModalPanel == null) {
      return;
    }

    setState(() {
      _activeModalPanel = null;
    });
  }

  void _runModalSceneCommand(String Function() command) {
    _closeCommandPanel();
    unawaited(_runSceneCommand(command));
  }

  double _normalizeViewportCoordinate(
    double logicalCoordinate,
    double logicalExtent,
  ) {
    if (logicalExtent <= 0) {
      return 0.0;
    }

    final normalizedCoordinate = logicalCoordinate / logicalExtent;
    return math.max(0.0, math.min(1.0, normalizedCoordinate));
  }

  void _handleViewportSizeChanged(
    Size logicalViewportSize,
    double devicePixelRatio,
  ) {
    if (logicalViewportSize.width <= 0 || logicalViewportSize.height <= 0) {
      return;
    }

    _lastLogicalViewportSize = logicalViewportSize;
    _lastDevicePixelRatio = devicePixelRatio;
    _updateRenderTargetSize();
  }

  _RenderTargetSize _resolveRenderTargetSize(
    Size logicalViewportSize,
    double devicePixelRatio,
  ) {
    final scaledDevicePixelRatio = devicePixelRatio * _effectiveRenderScale;
    var targetWidth = math.max(
      _minimumFrameWidth,
      (logicalViewportSize.width * scaledDevicePixelRatio).round(),
    );
    var targetHeight = math.max(
      _minimumFrameHeight,
      (logicalViewportSize.height * scaledDevicePixelRatio).round(),
    );

    targetWidth = math.min(_maximumFrameDimension, targetWidth);
    targetHeight = math.min(_maximumFrameDimension, targetHeight);

    targetWidth = _alignFrameDimension(targetWidth, _minimumFrameWidth);
    targetHeight = _alignFrameDimension(targetHeight, _minimumFrameHeight);

    return _RenderTargetSize(width: targetWidth, height: targetHeight);
  }

  bool _updateRenderTargetSize() {
    if (_lastLogicalViewportSize == Size.zero) {
      return false;
    }

    final nextRenderSize = _resolveRenderTargetSize(
      _lastLogicalViewportSize,
      _lastDevicePixelRatio,
    );

    if (nextRenderSize.width == _frameWidth &&
        nextRenderSize.height == _frameHeight) {
      return false;
    }

    if (!mounted) {
      _frameWidth = nextRenderSize.width;
      _frameHeight = nextRenderSize.height;
      _previewLine = _buildPreviewLine();
      return true;
    }

    setState(() {
      _frameWidth = nextRenderSize.width;
      _frameHeight = nextRenderSize.height;
      _previewLine = _buildPreviewLine();
    });
    _syncNativeViewportSize();
    return true;
  }

  int _alignFrameDimension(int value, int minimumValue) {
    if (value <= minimumValue) {
      return minimumValue;
    }

    final alignedValue =
        (value ~/ _frameDimensionAlignment) * _frameDimensionAlignment;
    return math.max(minimumValue, alignedValue);
  }

  Future<void> _runSceneCommand(String Function() command) async {
    if (_commandInFlight) {
      return;
    }

    setState(() {
      _commandInFlight = true;
    });

    try {
      final snapshot = _decodeSnapshot(command());
      if (!mounted) {
        return;
      }
      setState(() {
        _sceneSnapshot = snapshot;
        _viewportFeedback = TextureViewportFeedback.fromSceneSnapshot(snapshot);
      });
      _requestNativeFrame();
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = 'Scene command error: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _commandInFlight = false;
        });
      }
    }
  }

  Future<void> _promptRenameSelectedNode() async {
    if (_commandInFlight || !mounted) {
      return;
    }

    final selectedNode = _viewportFeedback?.selectedNode ?? _sceneSnapshot?.selectedNode;
    if (selectedNode == null) {
      return;
    }

    final submittedName = await showDialog<String>(
      context: context,
      builder: (dialogContext) {
        return _RenameNodeDialog(
          initialName: selectedNode.name,
        );
      },
    );

    if (!mounted || submittedName == null) {
      return;
    }

    await _runSceneCommand(
      () => renameNode(
        nodeId: BigInt.from(selectedNode.id),
        name: submittedName,
      ),
    );
  }

  Future<void> _promptCreateOperation() async {
    final selectedOperation = await _promptCreateNodeOption(
      title: 'Create Operation',
      optionKeyPrefix: 'operation-option',
      options: _operationOptions,
    );

    if (!mounted || selectedOperation == null) {
      return;
    }

    await _runSceneCommand(
      () => createOperation(operationId: selectedOperation.id),
    );
  }

  Future<void> _promptCreateModifier() async {
    final selectedModifier = await _promptCreateNodeOption(
      title: 'Create Modifier',
      optionKeyPrefix: 'modifier-option',
      options: _modifierOptions,
    );

    if (!mounted || selectedModifier == null) {
      return;
    }

    await _runSceneCommand(
      () => createModifier(modifierId: selectedModifier.id),
    );
  }

  Future<_CreateNodeOption?> _promptCreateNodeOption({
    required String title,
    required String optionKeyPrefix,
    required List<_CreateNodeOption> options,
  }) async {
    if (_commandInFlight || !mounted) {
      return null;
    }

    return showDialog<_CreateNodeOption>(
      context: context,
      builder: (dialogContext) {
        return _CreateNodeDialog(
          title: title,
          optionKeyPrefix: optionKeyPrefix,
          options: options,
        );
      },
    );
  }

  @override
  void dispose() {
    _textureEventSubscription?.cancel();
    _interactionCooldownTimer?.cancel();
    final activeTextureId = _textureId;
    if (activeTextureId != null) {
      unawaited(TextureBridge.instance.disposeTexture(activeTextureId));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final activeTextureId = _textureId;
    final snapshot = _sceneSnapshot;

    return Scaffold(
      appBar: AppBar(title: const Text('SDF Modeler Flutter Host')),
      body: Padding(
        padding: EdgeInsets.all(
          ShellLayout.forWidth(MediaQuery.sizeOf(context).width).screenPadding,
        ),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final shellLayout = ShellLayout.forWidth(constraints.maxWidth);
            final viewportCard = ViewportSurface(
              textureId: activeTextureId,
              onViewportSizeChanged: _handleViewportSizeChanged,
              onOrbitDrag: _handleViewportOrbitDrag,
              onPanDrag: _handleViewportPanDrag,
              onPrimaryTap: _handleViewportPrimaryTap,
              onHover: _handleViewportHover,
              onHoverExit: _handleViewportHoverExit,
              onScroll: _handleViewportScroll,
              onInteractionEnd: _handleViewportInteractionEnd,
              overlay: ViewportFeedbackOverlay(
                feedback: _viewportFeedback,
                interactionPhase: _interactionPhase,
                frameTimeMs: _lastNativeFrameTimeMs,
                framesPerSecond: _smoothedFramesPerSecond,
                droppedFrameCount: _droppedFrameCount,
              ),
            );

            if (shellLayout.useSidePanel) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(flex: 3, child: viewportCard),
                  SizedBox(width: shellLayout.panelGap),
                  ShellDesktopSidePanel(
                    width: shellLayout.inspectorPanelExtent,
                    child: _InspectorPanel(
                      shellLayout: shellLayout,
                      snapshot: snapshot,
                      viewportFeedback: _viewportFeedback,
                      statusLine: _statusLine,
                      versionLine: _versionLine,
                      previewLine: _previewLine,
                      commandInFlight: _commandInFlight,
                      adaptiveInteractionResolutionEnabled:
                          _adaptiveInteractionResolutionEnabled,
                      onToggleAdaptiveInteractionResolution:
                          _toggleAdaptiveInteractionResolution,
                      onRefresh: _refreshBridgeStatus,
                      onAddSphere: () => _runSceneCommand(addSphere),
                      onAddBox: () => _runSceneCommand(addBox),
                      onAddCylinder: () => _runSceneCommand(addCylinder),
                      onAddTorus: () => _runSceneCommand(addTorus),
                      onCreateOperation: _promptCreateOperation,
                      onCreateTransform: () => _runSceneCommand(createTransform),
                      onCreateModifier: _promptCreateModifier,
                      onRenameSelected: _promptRenameSelectedNode,
                      onDuplicateSelected: () =>
                          _runSceneCommand(duplicateSelected),
                      onUndo: () => _runSceneCommand(undo),
                      onRedo: () => _runSceneCommand(redo),
                      onDeleteSelected: () => _runSceneCommand(deleteSelected),
                      onSelectSceneNode: (nodeId) => _runSceneCommand(
                        () => selectNode(nodeId: BigInt.from(nodeId)),
                      ),
                      onToggleSceneNodeVisibility: (nodeId) => _runSceneCommand(
                        () => toggleNodeVisibility(nodeId: BigInt.from(nodeId)),
                      ),
                      onToggleSceneNodeLock: (nodeId) => _runSceneCommand(
                        () => toggleNodeLock(nodeId: BigInt.from(nodeId)),
                      ),
                      onFrameAll: () => _runSceneCommand(frameAll),
                      onResetScene: () => _runSceneCommand(resetScene),
                      onFocusSelected: () => _runSceneCommand(focusSelected),
                      onCameraFront: () => _runSceneCommand(cameraFront),
                      onCameraTop: () => _runSceneCommand(cameraTop),
                      onCameraRight: () => _runSceneCommand(cameraRight),
                      onCameraBack: () => _runSceneCommand(cameraBack),
                      onCameraLeft: () => _runSceneCommand(cameraLeft),
                      onCameraBottom: () => _runSceneCommand(cameraBottom),
                      onToggleProjection: () =>
                          _runSceneCommand(toggleOrthographic),
                    ),
                  ),
                ],
              );
            }

            return ShellStackedPaneLayout(
              viewport: viewportCard,
              modalPanel: _activeModalPanel == _BridgeModalPanel.commands
                  ? ShellModalPanel(
                      title: 'Workspace Commands',
                      onDismiss: _closeCommandPanel,
                      child: _CommandSheetContent(
                        commandInFlight: _commandInFlight,
                        viewportFeedback: _viewportFeedback,
                        snapshot: snapshot,
                        onAddSphere: () => _runModalSceneCommand(addSphere),
                        onAddBox: () => _runModalSceneCommand(addBox),
                        onAddCylinder: () =>
                            _runModalSceneCommand(addCylinder),
                        onAddTorus: () => _runModalSceneCommand(addTorus),
                        onCreateOperation: _promptCreateOperation,
                        onCreateTransform: () =>
                            _runModalSceneCommand(createTransform),
                        onCreateModifier: _promptCreateModifier,
                        onRenameSelected: _promptRenameSelectedNode,
                        onDuplicateSelected: () =>
                            _runModalSceneCommand(duplicateSelected),
                        onUndo: () => _runModalSceneCommand(undo),
                        onRedo: () => _runModalSceneCommand(redo),
                        onDeleteSelected: () =>
                            _runModalSceneCommand(deleteSelected),
                        onFrameAll: () => _runModalSceneCommand(frameAll),
                        onResetScene: () => _runModalSceneCommand(resetScene),
                        onFocusSelected: () =>
                            _runModalSceneCommand(focusSelected),
                        onCameraFront: () =>
                            _runModalSceneCommand(cameraFront),
                        onCameraTop: () => _runModalSceneCommand(cameraTop),
                        onCameraRight: () =>
                            _runModalSceneCommand(cameraRight),
                        onCameraBack: () => _runModalSceneCommand(cameraBack),
                        onCameraLeft: () => _runModalSceneCommand(cameraLeft),
                        onCameraBottom: () =>
                            _runModalSceneCommand(cameraBottom),
                        onToggleProjection: () =>
                            _runModalSceneCommand(toggleOrthographic),
                        onRefresh: () {
                          _closeCommandPanel();
                          _refreshBridgeStatus();
                        },
                      ),
                    )
                  : null,
              bottomSheetBuilder: (context, scrollController) {
                return _InspectorPanel(
                  shellLayout: shellLayout,
                  scrollController: scrollController,
                  snapshot: snapshot,
                  viewportFeedback: _viewportFeedback,
                  statusLine: _statusLine,
                  versionLine: _versionLine,
                  previewLine: _previewLine,
                  commandInFlight: _commandInFlight,
                  adaptiveInteractionResolutionEnabled:
                      _adaptiveInteractionResolutionEnabled,
                  onToggleAdaptiveInteractionResolution:
                      _toggleAdaptiveInteractionResolution,
                  onRefresh: _refreshBridgeStatus,
                  onAddSphere: () => _runSceneCommand(addSphere),
                  onAddBox: () => _runSceneCommand(addBox),
                  onAddCylinder: () => _runSceneCommand(addCylinder),
                  onAddTorus: () => _runSceneCommand(addTorus),
                  onCreateOperation: _promptCreateOperation,
                  onCreateTransform: () => _runSceneCommand(createTransform),
                  onCreateModifier: _promptCreateModifier,
                  onRenameSelected: _promptRenameSelectedNode,
                  onDuplicateSelected: () =>
                      _runSceneCommand(duplicateSelected),
                  onUndo: () => _runSceneCommand(undo),
                  onRedo: () => _runSceneCommand(redo),
                  onDeleteSelected: () => _runSceneCommand(deleteSelected),
                  onSelectSceneNode: (nodeId) => _runSceneCommand(
                    () => selectNode(nodeId: BigInt.from(nodeId)),
                  ),
                  onToggleSceneNodeVisibility: (nodeId) => _runSceneCommand(
                    () => toggleNodeVisibility(nodeId: BigInt.from(nodeId)),
                  ),
                  onToggleSceneNodeLock: (nodeId) => _runSceneCommand(
                    () => toggleNodeLock(nodeId: BigInt.from(nodeId)),
                  ),
                  onFrameAll: () => _runSceneCommand(frameAll),
                  onResetScene: () => _runSceneCommand(resetScene),
                  onFocusSelected: () => _runSceneCommand(focusSelected),
                  onCameraFront: () => _runSceneCommand(cameraFront),
                  onCameraTop: () => _runSceneCommand(cameraTop),
                  onCameraRight: () => _runSceneCommand(cameraRight),
                  onCameraBack: () => _runSceneCommand(cameraBack),
                  onCameraLeft: () => _runSceneCommand(cameraLeft),
                  onCameraBottom: () => _runSceneCommand(cameraBottom),
                  onToggleProjection: () =>
                      _runSceneCommand(toggleOrthographic),
                  onOpenCommandPanel: _openCommandPanel,
                );
              },
            );
          },
        ),
      ),
    );
  }
}

class _RenameNodeDialog extends StatefulWidget {
  const _RenameNodeDialog({required this.initialName});

  final String initialName;

  @override
  State<_RenameNodeDialog> createState() => _RenameNodeDialogState();
}

class _RenameNodeDialogState extends State<_RenameNodeDialog> {
  late final TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialName);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Rename Node'),
      content: TextField(
        key: const ValueKey('rename-node-field'),
        controller: _controller,
        autofocus: true,
        textInputAction: TextInputAction.done,
        onSubmitted: (value) => Navigator.of(context).pop(value),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          key: const ValueKey('rename-node-submit'),
          onPressed: () => Navigator.of(context).pop(_controller.text),
          child: const Text('Rename'),
        ),
      ],
    );
  }
}

class _CreateNodeDialog extends StatelessWidget {
  const _CreateNodeDialog({
    required this.title,
    required this.optionKeyPrefix,
    required this.options,
  });

  final String title;
  final String optionKeyPrefix;
  final List<_CreateNodeOption> options;

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(title),
      content: SizedBox(
        width: 360,
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxHeight: 420),
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                for (final option in options) ...[
                  OutlinedButton(
                    key: ValueKey('$optionKeyPrefix-${option.id}'),
                    onPressed: () => Navigator.of(context).pop(option),
                    child: Text(option.label),
                  ),
                  if (option != options.last)
                    const SizedBox(height: ShellTokens.controlGap),
                ],
              ],
            ),
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

class _InspectorPanel extends StatelessWidget {
  const _InspectorPanel({
    required this.shellLayout,
    required this.snapshot,
    required this.viewportFeedback,
    required this.statusLine,
    required this.versionLine,
    required this.previewLine,
    required this.commandInFlight,
    required this.adaptiveInteractionResolutionEnabled,
    required this.onToggleAdaptiveInteractionResolution,
    required this.onRefresh,
    required this.onAddSphere,
    required this.onAddBox,
    required this.onAddCylinder,
    required this.onAddTorus,
    required this.onCreateOperation,
    required this.onCreateTransform,
    required this.onCreateModifier,
    required this.onRenameSelected,
    required this.onDuplicateSelected,
    required this.onUndo,
    required this.onRedo,
    required this.onDeleteSelected,
    required this.onSelectSceneNode,
    required this.onToggleSceneNodeVisibility,
    required this.onToggleSceneNodeLock,
    required this.onFrameAll,
    required this.onResetScene,
    required this.onFocusSelected,
    required this.onCameraFront,
    required this.onCameraTop,
    required this.onCameraRight,
    required this.onCameraBack,
    required this.onCameraLeft,
    required this.onCameraBottom,
    required this.onToggleProjection,
    this.scrollController,
    this.onOpenCommandPanel,
  });

  final ShellLayout shellLayout;
  final ScrollController? scrollController;
  final AppSceneSnapshot? snapshot;
  final TextureViewportFeedback? viewportFeedback;
  final String statusLine;
  final String versionLine;
  final String previewLine;
  final bool commandInFlight;
  final bool adaptiveInteractionResolutionEnabled;
  final ValueChanged<bool> onToggleAdaptiveInteractionResolution;
  final VoidCallback onRefresh;
  final VoidCallback onAddSphere;
  final VoidCallback onAddBox;
  final VoidCallback onAddCylinder;
  final VoidCallback onAddTorus;
  final VoidCallback onCreateOperation;
  final VoidCallback onCreateTransform;
  final VoidCallback onCreateModifier;
  final VoidCallback onRenameSelected;
  final VoidCallback onDuplicateSelected;
  final VoidCallback onUndo;
  final VoidCallback onRedo;
  final VoidCallback onDeleteSelected;
  final ValueChanged<int> onSelectSceneNode;
  final ValueChanged<int> onToggleSceneNodeVisibility;
  final ValueChanged<int> onToggleSceneNodeLock;
  final VoidCallback onFrameAll;
  final VoidCallback onResetScene;
  final VoidCallback onFocusSelected;
  final VoidCallback onCameraFront;
  final VoidCallback onCameraTop;
  final VoidCallback onCameraRight;
  final VoidCallback onCameraBack;
  final VoidCallback onCameraLeft;
  final VoidCallback onCameraBottom;
  final VoidCallback onToggleProjection;
  final VoidCallback? onOpenCommandPanel;

  @override
  Widget build(BuildContext context) {
    final currentCamera = viewportFeedback?.camera ?? snapshot?.camera;
    final selectedNode = viewportFeedback?.selectedNode ?? snapshot?.selectedNode;
    final hoveredNode = viewportFeedback?.hoveredNode;
    final selectedNodeId = selectedNode?.id;
    final cameraControlsEnabled = !commandInFlight && currentCamera != null;
    final focusSelectedEnabled = !commandInFlight && selectedNode != null;
    final sceneCommandsEnabled = !commandInFlight;
    final undoEnabled = !commandInFlight && (snapshot?.history.canUndo ?? false);
    final redoEnabled = !commandInFlight && (snapshot?.history.canRedo ?? false);
    final renameSelectedEnabled = !commandInFlight && selectedNode != null;
    final duplicateSelectedEnabled = !commandInFlight && selectedNode != null;
    final deleteSelectedEnabled = !commandInFlight && selectedNode != null;
    final projectionButtonLabel = currentCamera?.orthographic ?? false
        ? 'Use Perspective'
        : 'Use Ortho';
    final showInlineCommandSections = shellLayout.useSidePanel;

    return ShellPanelSurface(
      child: ListView(
        controller: scrollController,
        children: [
          Text(statusLine),
          if (versionLine.isNotEmpty) ...[
            const SizedBox(height: ShellTokens.compactGap),
            Text(versionLine),
          ],
          const SizedBox(height: ShellTokens.compactGap),
          Text(previewLine),
          const SizedBox(height: ShellTokens.sectionGap),
          SwitchListTile.adaptive(
            contentPadding: EdgeInsets.zero,
            value: adaptiveInteractionResolutionEnabled,
            onChanged: onToggleAdaptiveInteractionResolution,
            title: const Text('Adaptive Interaction Resolution'),
            subtitle: const Text(
              'Lower the viewport render scale while navigating.',
            ),
          ),
          if (!showInlineCommandSections) ...[
            const SizedBox(height: ShellTokens.sectionGap),
            Text(
              'Workspace',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            ShellCommandStrip(
              children: [
                FilledButton.icon(
                  key: const ValueKey('open-command-panel'),
                  onPressed: onOpenCommandPanel,
                  icon: const Icon(Icons.dashboard_customize_outlined),
                  label: const Text('Commands'),
                ),
                OutlinedButton(
                  onPressed: sceneCommandsEnabled ? onFrameAll : null,
                  child: const Text('Frame All'),
                ),
                OutlinedButton(
                  onPressed: focusSelectedEnabled ? onFocusSelected : null,
                  child: const Text('Focus Selected'),
                ),
                OutlinedButton(
                  onPressed: cameraControlsEnabled ? onToggleProjection : null,
                  child: Text(projectionButtonLabel),
                ),
              ],
            ),
          ],
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Scene Tree',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          if (showInlineCommandSections) ...[
            const SizedBox(height: ShellTokens.controlGap),
            _SceneCommandButtons(
              sceneCommandsEnabled: sceneCommandsEnabled,
              deleteSelectedEnabled: deleteSelectedEnabled,
              onAddSphere: onAddSphere,
              onAddBox: onAddBox,
              onAddCylinder: onAddCylinder,
              onAddTorus: onAddTorus,
              onCreateOperation: onCreateOperation,
              onCreateTransform: onCreateTransform,
              onCreateModifier: onCreateModifier,
              renameSelectedEnabled: renameSelectedEnabled,
              onRenameSelected: onRenameSelected,
              duplicateSelectedEnabled: duplicateSelectedEnabled,
              onDuplicateSelected: onDuplicateSelected,
              undoEnabled: undoEnabled,
              onUndo: onUndo,
              redoEnabled: redoEnabled,
              onRedo: onRedo,
              onDeleteSelected: onDeleteSelected,
            ),
          ],
          const SizedBox(height: ShellTokens.controlGap),
          if (snapshot == null)
            const Text('Scene snapshot is still loading.')
          else
            SceneTreePanel(
              roots: snapshot!.sceneTreeRoots,
              selectedNodeId: selectedNodeId,
              enabled: !commandInFlight,
              onSelectNode: onSelectSceneNode,
              onToggleNodeVisibility: onToggleSceneNodeVisibility,
              onToggleNodeLock: onToggleSceneNodeLock,
            ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Camera',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          if (showInlineCommandSections) ...[
            const SizedBox(height: ShellTokens.controlGap),
            _CameraCommandButtons(
              cameraControlsEnabled: cameraControlsEnabled,
              focusSelectedEnabled: focusSelectedEnabled,
              projectionButtonLabel: projectionButtonLabel,
              onFocusSelected: onFocusSelected,
              onFrameAll: onFrameAll,
              onToggleProjection: onToggleProjection,
              onCameraFront: onCameraFront,
              onCameraTop: onCameraTop,
              onCameraRight: onCameraRight,
              onCameraBack: onCameraBack,
              onCameraLeft: onCameraLeft,
              onCameraBottom: onCameraBottom,
            ),
          ] else ...[
            const SizedBox(height: ShellTokens.controlGap),
            Text(
              'Primary camera commands stay in the command strip and modal panel so the tablet shell keeps the scene tree reachable.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Viewport Status',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          if (snapshot == null)
            const Text('Scene snapshot is still loading.')
          else ...[
            Text('Selected: ${selectedNode?.name ?? 'None'}'),
            Text('Hovered: ${hoveredNode?.name ?? 'None'}'),
            Text(
              'Tool: ${snapshot!.tool.activeToolLabel} - ${snapshot!.tool.shadingModeLabel}',
            ),
            if (currentCamera != null)
              Text(
                'Camera distance: ${currentCamera.distance.toStringAsFixed(2)} - ${currentCamera.orthographic ? 'Ortho' : 'Perspective'}',
              ),
            Text(
              'Scene nodes: ${snapshot!.stats.totalNodes} total - ${snapshot!.stats.visibleNodes} visible - ${snapshot!.stats.topLevelNodes} roots',
            ),
            Text(
              'SDF complexity: ${snapshot!.stats.sdfEvalComplexity} - Voxel memory: ${snapshot!.stats.voxelMemoryBytes} bytes',
            ),
            const SizedBox(height: ShellTokens.controlGap),
            if (showInlineCommandSections)
              Wrap(
                spacing: ShellTokens.controlGap,
                runSpacing: ShellTokens.controlGap,
                children: [
                  OutlinedButton(
                    onPressed: sceneCommandsEnabled ? onResetScene : null,
                    child: const Text('Reset Scene'),
                  ),
                  TextButton(
                    onPressed: onRefresh,
                    child: const Text('Re-run Ping'),
                  ),
                ],
              ),
          ],
        ],
      ),
    );
  }
}

class _SceneCommandButtons extends StatelessWidget {
  const _SceneCommandButtons({
    required this.sceneCommandsEnabled,
    required this.undoEnabled,
    required this.redoEnabled,
    required this.deleteSelectedEnabled,
    required this.onAddSphere,
    required this.onAddBox,
    required this.onAddCylinder,
    required this.onAddTorus,
    required this.onCreateOperation,
    required this.onCreateTransform,
    required this.onCreateModifier,
    required this.renameSelectedEnabled,
    required this.onRenameSelected,
    required this.duplicateSelectedEnabled,
    required this.onDuplicateSelected,
    required this.onUndo,
    required this.onRedo,
    required this.onDeleteSelected,
  });

  final bool sceneCommandsEnabled;
  final bool undoEnabled;
  final bool redoEnabled;
  final bool deleteSelectedEnabled;
  final VoidCallback onAddSphere;
  final VoidCallback onAddBox;
  final VoidCallback onAddCylinder;
  final VoidCallback onAddTorus;
  final VoidCallback onCreateOperation;
  final VoidCallback onCreateTransform;
  final VoidCallback onCreateModifier;
  final bool renameSelectedEnabled;
  final VoidCallback onRenameSelected;
  final bool duplicateSelectedEnabled;
  final VoidCallback onDuplicateSelected;
  final VoidCallback onUndo;
  final VoidCallback onRedo;
  final VoidCallback onDeleteSelected;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: ShellTokens.controlGap,
      runSpacing: ShellTokens.controlGap,
      children: [
        FilledButton(
          onPressed: sceneCommandsEnabled ? onAddSphere : null,
          child: const Text('Sphere'),
        ),
        OutlinedButton(
          onPressed: sceneCommandsEnabled ? onAddBox : null,
          child: const Text('Box'),
        ),
        OutlinedButton(
          onPressed: sceneCommandsEnabled ? onAddCylinder : null,
          child: const Text('Cylinder'),
        ),
        OutlinedButton(
          onPressed: sceneCommandsEnabled ? onAddTorus : null,
          child: const Text('Torus'),
        ),
        OutlinedButton(
          key: const ValueKey('create-operation-command'),
          onPressed: sceneCommandsEnabled ? onCreateOperation : null,
          child: const Text('Operation'),
        ),
        OutlinedButton(
          key: const ValueKey('create-transform-command'),
          onPressed: sceneCommandsEnabled ? onCreateTransform : null,
          child: const Text('Transform'),
        ),
        OutlinedButton(
          key: const ValueKey('create-modifier-command'),
          onPressed: sceneCommandsEnabled ? onCreateModifier : null,
          child: const Text('Modifier'),
        ),
        OutlinedButton(
          key: const ValueKey('rename-command'),
          onPressed: renameSelectedEnabled ? onRenameSelected : null,
          child: const Text('Rename'),
        ),
        OutlinedButton(
          key: const ValueKey('duplicate-command'),
          onPressed: duplicateSelectedEnabled ? onDuplicateSelected : null,
          child: const Text('Duplicate'),
        ),
        OutlinedButton(
          key: const ValueKey('undo-command'),
          onPressed: undoEnabled ? onUndo : null,
          child: const Text('Undo'),
        ),
        OutlinedButton(
          key: const ValueKey('redo-command'),
          onPressed: redoEnabled ? onRedo : null,
          child: const Text('Redo'),
        ),
        OutlinedButton(
          onPressed: deleteSelectedEnabled ? onDeleteSelected : null,
          child: const Text('Delete Selected'),
        ),
      ],
    );
  }
}

class _CameraCommandButtons extends StatelessWidget {
  const _CameraCommandButtons({
    required this.cameraControlsEnabled,
    required this.focusSelectedEnabled,
    required this.projectionButtonLabel,
    required this.onFocusSelected,
    required this.onFrameAll,
    required this.onToggleProjection,
    required this.onCameraFront,
    required this.onCameraTop,
    required this.onCameraRight,
    required this.onCameraBack,
    required this.onCameraLeft,
    required this.onCameraBottom,
  });

  final bool cameraControlsEnabled;
  final bool focusSelectedEnabled;
  final String projectionButtonLabel;
  final VoidCallback onFocusSelected;
  final VoidCallback onFrameAll;
  final VoidCallback onToggleProjection;
  final VoidCallback onCameraFront;
  final VoidCallback onCameraTop;
  final VoidCallback onCameraRight;
  final VoidCallback onCameraBack;
  final VoidCallback onCameraLeft;
  final VoidCallback onCameraBottom;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: ShellTokens.controlGap,
      runSpacing: ShellTokens.controlGap,
      children: [
        FilledButton(
          onPressed: focusSelectedEnabled ? onFocusSelected : null,
          child: const Text('Focus Selected'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onFrameAll : null,
          child: const Text('Frame All'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onToggleProjection : null,
          child: Text(projectionButtonLabel),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraFront : null,
          child: const Text('Front'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraTop : null,
          child: const Text('Top'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraRight : null,
          child: const Text('Right'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraBack : null,
          child: const Text('Back'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraLeft : null,
          child: const Text('Left'),
        ),
        OutlinedButton(
          onPressed: cameraControlsEnabled ? onCameraBottom : null,
          child: const Text('Bottom'),
        ),
      ],
    );
  }
}

class _CommandSheetContent extends StatelessWidget {
  const _CommandSheetContent({
    required this.commandInFlight,
    required this.viewportFeedback,
    required this.snapshot,
    required this.onAddSphere,
    required this.onAddBox,
    required this.onAddCylinder,
    required this.onAddTorus,
    required this.onCreateOperation,
    required this.onCreateTransform,
    required this.onCreateModifier,
    required this.onRenameSelected,
    required this.onDuplicateSelected,
    required this.onUndo,
    required this.onRedo,
    required this.onDeleteSelected,
    required this.onFrameAll,
    required this.onResetScene,
    required this.onFocusSelected,
    required this.onCameraFront,
    required this.onCameraTop,
    required this.onCameraRight,
    required this.onCameraBack,
    required this.onCameraLeft,
    required this.onCameraBottom,
    required this.onToggleProjection,
    required this.onRefresh,
  });

  final bool commandInFlight;
  final TextureViewportFeedback? viewportFeedback;
  final AppSceneSnapshot? snapshot;
  final VoidCallback onAddSphere;
  final VoidCallback onAddBox;
  final VoidCallback onAddCylinder;
  final VoidCallback onAddTorus;
  final VoidCallback onCreateOperation;
  final VoidCallback onCreateTransform;
  final VoidCallback onCreateModifier;
  final VoidCallback onRenameSelected;
  final VoidCallback onDuplicateSelected;
  final VoidCallback onUndo;
  final VoidCallback onRedo;
  final VoidCallback onDeleteSelected;
  final VoidCallback onFrameAll;
  final VoidCallback onResetScene;
  final VoidCallback onFocusSelected;
  final VoidCallback onCameraFront;
  final VoidCallback onCameraTop;
  final VoidCallback onCameraRight;
  final VoidCallback onCameraBack;
  final VoidCallback onCameraLeft;
  final VoidCallback onCameraBottom;
  final VoidCallback onToggleProjection;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    final currentCamera = viewportFeedback?.camera ?? snapshot?.camera;
    final selectedNode = viewportFeedback?.selectedNode ?? snapshot?.selectedNode;
    final sceneCommandsEnabled = !commandInFlight;
    final undoEnabled = !commandInFlight && (snapshot?.history.canUndo ?? false);
    final redoEnabled = !commandInFlight && (snapshot?.history.canRedo ?? false);
    final renameSelectedEnabled = !commandInFlight && selectedNode != null;
    final duplicateSelectedEnabled = !commandInFlight && selectedNode != null;
    final deleteSelectedEnabled = !commandInFlight && selectedNode != null;
    final cameraControlsEnabled = !commandInFlight && currentCamera != null;
    final focusSelectedEnabled = !commandInFlight && selectedNode != null;
    final projectionButtonLabel = currentCamera?.orthographic ?? false
        ? 'Use Perspective'
        : 'Use Ortho';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Scene Commands',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _SceneCommandButtons(
          sceneCommandsEnabled: sceneCommandsEnabled,
          deleteSelectedEnabled: deleteSelectedEnabled,
          onAddSphere: onAddSphere,
          onAddBox: onAddBox,
          onAddCylinder: onAddCylinder,
          onAddTorus: onAddTorus,
          onCreateOperation: onCreateOperation,
          onCreateTransform: onCreateTransform,
          onCreateModifier: onCreateModifier,
          renameSelectedEnabled: renameSelectedEnabled,
          onRenameSelected: onRenameSelected,
          duplicateSelectedEnabled: duplicateSelectedEnabled,
          onDuplicateSelected: onDuplicateSelected,
          undoEnabled: undoEnabled,
          onUndo: onUndo,
          redoEnabled: redoEnabled,
          onRedo: onRedo,
          onDeleteSelected: onDeleteSelected,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Camera Commands',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        _CameraCommandButtons(
          cameraControlsEnabled: cameraControlsEnabled,
          focusSelectedEnabled: focusSelectedEnabled,
          projectionButtonLabel: projectionButtonLabel,
          onFocusSelected: onFocusSelected,
          onFrameAll: onFrameAll,
          onToggleProjection: onToggleProjection,
          onCameraFront: onCameraFront,
          onCameraTop: onCameraTop,
          onCameraRight: onCameraRight,
          onCameraBack: onCameraBack,
          onCameraLeft: onCameraLeft,
          onCameraBottom: onCameraBottom,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Session',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        Wrap(
          spacing: ShellTokens.controlGap,
          runSpacing: ShellTokens.controlGap,
          children: [
            OutlinedButton(
              onPressed: sceneCommandsEnabled ? onResetScene : null,
              child: const Text('Reset Scene'),
            ),
            TextButton(
              onPressed: onRefresh,
              child: const Text('Re-run Ping'),
            ),
          ],
        ),
      ],
    );
  }
}

class _RenderTargetSize {
  const _RenderTargetSize({required this.width, required this.height});

  final int width;
  final int height;
}
