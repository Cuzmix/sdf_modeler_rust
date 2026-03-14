import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_tree_panel.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_bridge.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_event.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

class SdfModelerApp extends StatelessWidget {
  const SdfModelerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SDF Modeler Flutter',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
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
  static const Duration _interactionSharpnessDelay = Duration(milliseconds: 180);
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
      _interactionSharpnessDelay,
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
        padding: const EdgeInsets.all(16),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final useSidePanel = constraints.maxWidth >= 900;
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
            final inspectorPanel = _InspectorPanel(
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
              onToggleProjection: () => _runSceneCommand(toggleOrthographic),
            );

            if (useSidePanel) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(flex: 3, child: viewportCard),
                  const SizedBox(width: 16),
                  SizedBox(width: 320, child: inspectorPanel),
                ],
              );
            }

            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(child: viewportCard),
                const SizedBox(height: 16),
                SizedBox(height: 360, child: inspectorPanel),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _InspectorPanel extends StatelessWidget {
  const _InspectorPanel({
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
  });

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

  @override
  Widget build(BuildContext context) {
    final currentCamera = viewportFeedback?.camera ?? snapshot?.camera;
    final selectedNode = viewportFeedback?.selectedNode ?? snapshot?.selectedNode;
    final hoveredNode = viewportFeedback?.hoveredNode;
    final selectedNodeId = selectedNode?.id;
    final cameraControlsEnabled = !commandInFlight && currentCamera != null;
    final focusSelectedEnabled = !commandInFlight && selectedNode != null;
    final sceneCommandsEnabled = !commandInFlight;
    final deleteSelectedEnabled = !commandInFlight && selectedNode != null;
    final projectionButtonLabel = currentCamera?.orthographic ?? false
        ? 'Use Perspective'
        : 'Use Ortho';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            Text(statusLine),
            if (versionLine.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(versionLine),
            ],
            const SizedBox(height: 4),
            Text(previewLine),
            const SizedBox(height: 16),
            SwitchListTile.adaptive(
              contentPadding: EdgeInsets.zero,
              value: adaptiveInteractionResolutionEnabled,
              onChanged: onToggleAdaptiveInteractionResolution,
              title: const Text('Adaptive Interaction Resolution'),
              subtitle: const Text(
                'Lower the viewport render scale while navigating.',
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Scene Tree',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
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
                  onPressed: deleteSelectedEnabled ? onDeleteSelected : null,
                  child: const Text('Delete Selected'),
                ),
              ],
            ),
            const SizedBox(height: 12),
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
            const SizedBox(height: 16),
            Text(
              'Camera',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton(
                  onPressed: focusSelectedEnabled ? onFocusSelected : null,
                  child: const Text('Focus Selected'),
                ),
                OutlinedButton(
                  onPressed: sceneCommandsEnabled ? onFrameAll : null,
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
            ),
            const SizedBox(height: 16),
            Text(
              'Viewport Status',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
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
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
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
      ),
    );
  }
}

class _RenderTargetSize {
  const _RenderTargetSize({required this.width, required this.height});

  final int width;
  final int height;
}