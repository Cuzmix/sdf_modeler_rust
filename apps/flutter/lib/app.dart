import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/export/export_panel.dart';
import 'package:sdf_modeler_flutter/src/import/import_panel.dart';
import 'package:sdf_modeler_flutter/src/light/light_inspector_panel.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_tree_panel.dart';
import 'package:sdf_modeler_flutter/src/sculpt/sculpt_convert_panel.dart';
import 'package:sdf_modeler_flutter/src/sculpt/sculpt_session_panel.dart';
import 'package:sdf_modeler_flutter/src/session/document_session_panel.dart';
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
import 'package:sdf_modeler_flutter/src/viewport/viewport_tool_overlay.dart';

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

const List<_CreateNodeOption> _lightOptions = <_CreateNodeOption>[
  _CreateNodeOption(id: 'point', label: 'Point'),
  _CreateNodeOption(id: 'spot', label: 'Spot'),
  _CreateNodeOption(id: 'directional', label: 'Directional'),
  _CreateNodeOption(id: 'ambient', label: 'Ambient'),
];

const double _primitiveParameterMin = 0.01;
const double _primitiveParameterMax = 100.0;
const double _materialFactorMin = 0.0;
const double _materialFactorMax = 1.0;
const double _emissiveIntensityMax = 5.0;
const double _transformTranslationMin = -10.0;
const double _transformTranslationMax = 10.0;
const double _transformRotationDegreesMin = -180.0;
const double _transformRotationDegreesMax = 180.0;
const double _viewportTranslationNudge = 0.25;
const double _viewportRotationNudgeDegrees = 15.0;
const double _viewportScaleNudge = 0.25;
const double _viewportPivotNudge = 0.25;

String _formatVec3(AppVec3 value) {
  return [
    value.x.toStringAsFixed(2),
    value.y.toStringAsFixed(2),
    value.z.toStringAsFixed(2),
  ].join(', ');
}

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
  Timer? _workflowPollTimer;
  Timer? _interactionCooldownTimer;
  bool _workflowPollInFlight = false;
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
  String? _lastViewportHostError;
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
        _applySnapshotState(snapshot);
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

  void _syncWorkflowPolling(AppSceneSnapshot? snapshot) {
    final shouldPoll =
        (snapshot?.export.status.isInProgress ?? false) ||
        (snapshot?.import.status.isInProgress ?? false) ||
        (snapshot?.sculptConvert.status.isInProgress ?? false);
    if (shouldPoll) {
      _workflowPollTimer ??= Timer.periodic(
        const Duration(milliseconds: 150),
        (_) => _pollWorkflowSnapshot(),
      );
      return;
    }

    _workflowPollTimer?.cancel();
    _workflowPollTimer = null;
  }

  void _applySnapshotState(
    AppSceneSnapshot snapshot, {
    bool updateViewportFeedback = true,
  }) {
    _syncWorkflowPolling(snapshot);
    _sceneSnapshot = snapshot;
    if (updateViewportFeedback) {
      _viewportFeedback = TextureViewportFeedback.fromSceneSnapshot(snapshot);
    }
  }

  Future<void> _pollWorkflowSnapshot() async {
    if (!mounted || _workflowPollInFlight) {
      return;
    }

    _workflowPollInFlight = true;
    try {
      final snapshot = _decodeSnapshot(sceneSnapshotJson());
      if (!mounted) {
        return;
      }
      setState(() {
        _applySnapshotState(snapshot, updateViewportFeedback: false);
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = 'Workflow polling error: $error';
      });
    } finally {
      _workflowPollInFlight = false;
    }
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
    AppSceneSnapshot? refreshedSnapshot;
    String? snapshotRefreshError;
    final hostError = event.hostError;

    if (event.sceneStateChanged) {
      try {
        refreshedSnapshot = _decodeSnapshot(sceneSnapshotJson());
      } catch (error) {
        snapshotRefreshError = 'Scene snapshot refresh error: $error';
      }
    }

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
      _lastViewportHostError = hostError;
      if (refreshedSnapshot != null) {
        _applySnapshotState(refreshedSnapshot, updateViewportFeedback: false);
      }
      if (event.feedback != null) {
        _viewportFeedback = event.feedback;
      } else if (refreshedSnapshot != null) {
        _viewportFeedback = TextureViewportFeedback.fromSceneSnapshot(
          refreshedSnapshot,
        );
      }
      if (snapshotRefreshError != null) {
        _previewLine = snapshotRefreshError;
      } else if (hostError != null) {
        _previewLine = 'Viewport host error: $hostError';
      } else {
        _previewLine = _buildPreviewLine();
      }
    });
  }

  void debugHandleTextureEvent(TextureViewportEvent event) {
    _handleTextureEvent(event);
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
        _applySnapshotState(snapshot);
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

  void _runModalSceneCommand(
    String Function() command, {
    bool requestNativeFrame = true,
  }) {
    _closeCommandPanel();
    unawaited(
      _runSceneCommand(command, requestNativeFrame: requestNativeFrame),
    );
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

  Future<void> _runSceneCommand(
    String Function() command, {
    bool requestNativeFrame = true,
  }) async {
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
        _applySnapshotState(snapshot);
      });
      if (requestNativeFrame) {
        _requestNativeFrame();
      }
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

  Future<void> _newScene() {
    return _runSceneCommand(newScene);
  }

  Future<void> _openScene() {
    return _runSceneCommand(openScene);
  }

  Future<void> _openRecentScene(String path) {
    return _runSceneCommand(() => openRecentScene(path: path));
  }

  Future<void> _saveScene() {
    return _runSceneCommand(saveScene);
  }

  Future<void> _saveSceneAs() {
    return _runSceneCommand(saveSceneAs);
  }

  Future<void> _recoverAutosave() {
    return _runSceneCommand(recoverAutosave);
  }

  Future<void> _discardRecovery() {
    return _runSceneCommand(discardRecovery);
  }

  Future<void> _setExportResolution(int resolution) {
    return _runSceneCommand(
      () => setExportResolution(resolution: resolution),
      requestNativeFrame: false,
    );
  }

  Future<void> _setAdaptiveExport(bool enabled) {
    return _runSceneCommand(
      () => setAdaptiveExport(enabled: enabled),
      requestNativeFrame: false,
    );
  }

  Future<void> _startExport() {
    return _runSceneCommand(startExport, requestNativeFrame: false);
  }

  Future<void> _cancelExport() {
    return _runSceneCommand(cancelExport, requestNativeFrame: false);
  }

  Future<void> _openImportDialog() {
    return _runSceneCommand(openImportDialog, requestNativeFrame: false);
  }

  Future<void> _cancelImportDialog() {
    return _runSceneCommand(cancelImportDialog, requestNativeFrame: false);
  }

  Future<void> _setImportUseAuto(bool useAuto) {
    return _runSceneCommand(
      () => setImportUseAuto(useAuto: useAuto),
      requestNativeFrame: false,
    );
  }

  Future<void> _setImportResolution(int resolution) {
    return _runSceneCommand(
      () => setImportResolution(resolution: resolution),
      requestNativeFrame: false,
    );
  }

  Future<void> _startImport() {
    return _runSceneCommand(startImport, requestNativeFrame: false);
  }

  Future<void> _cancelImport() {
    return _runSceneCommand(cancelImport, requestNativeFrame: false);
  }

  Future<void> _openSculptConvertDialog() {
    return _runSceneCommand(
      openSculptConvertDialogForSelected,
      requestNativeFrame: false,
    );
  }

  Future<void> _cancelSculptConvertDialog() {
    return _runSceneCommand(
      cancelSculptConvertDialog,
      requestNativeFrame: false,
    );
  }

  Future<void> _setSculptConvertMode(String modeId) {
    return _runSceneCommand(
      () => setSculptConvertMode(modeId: modeId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSculptConvertResolution(int resolution) {
    return _runSceneCommand(
      () => setSculptConvertResolution(resolution: resolution),
      requestNativeFrame: false,
    );
  }

  Future<void> _startSculptConvert() {
    return _runSceneCommand(startSculptConvert, requestNativeFrame: false);
  }

  Future<void> _resumeSculptingSelected() {
    return _runSceneCommand(
      resumeSculptingSelected,
      requestNativeFrame: false,
    );
  }

  Future<void> _stopSculpting() {
    return _runSceneCommand(stopSculpting, requestNativeFrame: false);
  }

  Future<void> _setSculptBrushMode(String modeId) {
    return _runSceneCommand(
      () => setSculptBrushMode(modeId: modeId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSculptBrushRadius(double radius) {
    return _runSceneCommand(
      () => setSculptBrushRadius(radius: radius),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSculptBrushStrength(double strength) {
    return _runSceneCommand(
      () => setSculptBrushStrength(strength: strength),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSculptSymmetryAxis(String axisId) {
    return _runSceneCommand(
      () => setSculptSymmetryAxis(axisId: axisId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedSculptResolution(int resolution) {
    return _runSceneCommand(
      () => setSelectedSculptResolution(resolution: resolution),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightType(String lightTypeId) {
    return _runSceneCommand(
      () => setSelectedLightType(lightTypeId: lightTypeId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightColor(AppVec3 color) {
    return _runSceneCommand(
      () => setSelectedLightColor(
        red: color.x,
        green: color.y,
        blue: color.z,
      ),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightIntensity(double intensity) {
    return _runSceneCommand(
      () => setSelectedLightIntensity(intensity: intensity),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightRange(double range) {
    return _runSceneCommand(
      () => setSelectedLightRange(range: range),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightSpotAngle(double angleDegrees) {
    return _runSceneCommand(
      () => setSelectedLightSpotAngle(angleDegrees: angleDegrees),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightCastShadows(bool enabled) {
    return _runSceneCommand(
      () => setSelectedLightCastShadows(enabled: enabled),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightShadowSoftness(double softness) {
    return _runSceneCommand(
      () => setSelectedLightShadowSoftness(softness: softness),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightShadowColor(AppVec3 color) {
    return _runSceneCommand(
      () => setSelectedLightShadowColor(
        red: color.x,
        green: color.y,
        blue: color.z,
      ),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightVolumetric(bool enabled) {
    return _runSceneCommand(
      () => setSelectedLightVolumetric(enabled: enabled),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightVolumetricDensity(double density) {
    return _runSceneCommand(
      () => setSelectedLightVolumetricDensity(density: density),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightCookie(int cookieNodeId) {
    return _runSceneCommand(
      () => setSelectedLightCookie(cookieNodeId: BigInt.from(cookieNodeId)),
      requestNativeFrame: false,
    );
  }

  Future<void> _clearSelectedLightCookie() {
    return _runSceneCommand(clearSelectedLightCookie, requestNativeFrame: false);
  }

  Future<void> _setSelectedLightProximityMode(String modeId) {
    return _runSceneCommand(
      () => setSelectedLightProximityMode(modeId: modeId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightProximityRange(double range) {
    return _runSceneCommand(
      () => setSelectedLightProximityRange(range: range),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightArrayPattern(String patternId) {
    return _runSceneCommand(
      () => setSelectedLightArrayPattern(patternId: patternId),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightArrayCount(int count) {
    return _runSceneCommand(
      () => setSelectedLightArrayCount(count: count),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightArrayRadius(double radius) {
    return _runSceneCommand(
      () => setSelectedLightArrayRadius(radius: radius),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightArrayColorVariation(double value) {
    return _runSceneCommand(
      () => setSelectedLightArrayColorVariation(value: value),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightIntensityExpression(String expression) {
    return _runSceneCommand(
      () => setSelectedLightIntensityExpression(expression: expression),
      requestNativeFrame: false,
    );
  }

  Future<void> _setSelectedLightColorHueExpression(String expression) {
    return _runSceneCommand(
      () => setSelectedLightColorHueExpression(expression: expression),
      requestNativeFrame: false,
    );
  }

  Future<void> _setNodeLightMask(int nodeId, int mask) {
    return _runSceneCommand(
      () => setNodeLightMask(nodeId: BigInt.from(nodeId), mask: mask),
      requestNativeFrame: false,
    );
  }

  Future<void> _setNodeLightLinkEnabled(int nodeId, int lightId, bool enabled) {
    return _runSceneCommand(
      () => setNodeLightLinkEnabled(
        nodeId: BigInt.from(nodeId),
        lightId: BigInt.from(lightId),
        enabled: enabled,
      ),
      requestNativeFrame: false,
    );
  }

  Future<void> _setManipulatorMode(String modeId) {
    return _runSceneCommand(() => setManipulatorMode(modeId: modeId));
  }

  Future<void> _toggleManipulatorSpace() {
    return _runSceneCommand(toggleManipulatorSpace);
  }

  Future<void> _resetManipulatorPivot() {
    return _runSceneCommand(resetManipulatorPivot);
  }

  Future<void> _nudgeManipulatorPivot(String axisId, double direction) {
    final amount = direction * _viewportPivotNudge;
    final delta = switch (axisId) {
      'x' => (x: amount, y: 0.0, z: 0.0),
      'y' => (x: 0.0, y: amount, z: 0.0),
      _ => (x: 0.0, y: 0.0, z: amount),
    };
    return _runSceneCommand(
      () => nudgeManipulatorPivotOffset(x: delta.x, y: delta.y, z: delta.z),
    );
  }

  Future<void> _nudgeManipulatorAxis(
    String modeId,
    String axisId,
    double direction,
  ) {
    final amount = switch (modeId) {
      'rotate' => direction * _viewportRotationNudgeDegrees,
      'scale' => direction * _viewportScaleNudge,
      _ => direction * _viewportTranslationNudge,
    };
    final delta = switch (axisId) {
      'x' => (x: amount, y: 0.0, z: 0.0),
      'y' => (x: 0.0, y: amount, z: 0.0),
      _ => (x: 0.0, y: 0.0, z: amount),
    };

    return switch (modeId) {
      'rotate' => _runSceneCommand(
        () => nudgeSelectedRotationDegrees(
          deltaXDegrees: delta.x,
          deltaYDegrees: delta.y,
          deltaZDegrees: delta.z,
        ),
      ),
      'scale' => _runSceneCommand(
        () => nudgeSelectedScale(
          deltaX: delta.x,
          deltaY: delta.y,
          deltaZ: delta.z,
        ),
      ),
      _ => _runSceneCommand(
        () => nudgeSelectedTranslation(
          deltaX: delta.x,
          deltaY: delta.y,
          deltaZ: delta.z,
        ),
      ),
    };
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

  Future<void> _promptCreateLight() async {
    final selectedLight = await _promptCreateNodeOption(
      title: 'Create Light',
      optionKeyPrefix: 'light-option',
      options: _lightOptions,
    );

    if (!mounted || selectedLight == null) {
      return;
    }

    await _runSceneCommand(
      () => createLight(lightId: selectedLight.id),
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
    _workflowPollTimer?.cancel();
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
            final selectedNode =
                _viewportFeedback?.selectedNode ?? snapshot?.selectedNode;
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
                hostError: _lastViewportHostError,
              ),
              controlsOverlay: ViewportToolOverlay(
                tool: snapshot?.tool,
                hasSelection: selectedNode != null,
                enabled: !_commandInFlight,
                onSetManipulatorMode: (modeId) =>
                    unawaited(_setManipulatorMode(modeId)),
                onToggleManipulatorSpace: () =>
                    unawaited(_toggleManipulatorSpace()),
                onResetManipulatorPivot: () =>
                    unawaited(_resetManipulatorPivot()),
                onNudgeManipulatorAxis: (modeId, axisId, direction) =>
                    unawaited(_nudgeManipulatorAxis(modeId, axisId, direction)),
                onNudgeManipulatorPivot: (axisId, direction) =>
                    unawaited(_nudgeManipulatorPivot(axisId, direction)),
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
                      onCreateLight: _promptCreateLight,
                      onCreateSculpt: () => _runSceneCommand(createSculpt),
                      onRenameSelected: _promptRenameSelectedNode,
                      onSetPrimitiveParameter: (parameterKey, value) =>
                          _runSceneCommand(
                            () => setSelectedPrimitiveParameter(
                              parameterKey: parameterKey,
                              value: value,
                            ),
                          ),
                      onSetMaterialFloat: (fieldId, value) => _runSceneCommand(
                        () => setSelectedMaterialFloat(
                          fieldId: fieldId,
                          value: value,
                        ),
                      ),
                      onSetMaterialColor: (fieldId, color) => _runSceneCommand(
                        () => setSelectedMaterialColor(
                          fieldId: fieldId,
                          red: color.x,
                          green: color.y,
                          blue: color.z,
                        ),
                      ),
                      onSetTransformPosition: (x, y, z) => _runSceneCommand(
                        () => setSelectedTransformPosition(x: x, y: y, z: z),
                      ),
                      onSetTransformRotationDegrees: (x, y, z) =>
                          _runSceneCommand(
                            () => setSelectedTransformRotationDegrees(
                              xDegrees: x,
                              yDegrees: y,
                              zDegrees: z,
                            ),
                          ),
                      onSetTransformScale: (x, y, z) => _runSceneCommand(
                        () => setSelectedTransformScale(x: x, y: y, z: z),
                      ),
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
                      onNewScene: _newScene,
                      onOpenScene: _openScene,
                      onSaveScene: _saveScene,
                      onSaveSceneAs: _saveSceneAs,
                      onOpenRecentScene: _openRecentScene,
                      onRecoverAutosave: _recoverAutosave,
                      onDiscardRecovery: _discardRecovery,
                      onSetExportResolution: _setExportResolution,
                      onSetAdaptiveExport: _setAdaptiveExport,
                      onStartExport: _startExport,
                      onCancelExport: _cancelExport,
                      onOpenImportDialog: _openImportDialog,
                      onCancelImportDialog: _cancelImportDialog,
                      onSetImportUseAuto: _setImportUseAuto,
                      onSetImportResolution: _setImportResolution,
                      onStartImport: _startImport,
                      onCancelImport: _cancelImport,
                      onOpenSculptConvertDialog: _openSculptConvertDialog,
                      onCancelSculptConvertDialog: _cancelSculptConvertDialog,
                      onSetSculptConvertMode: _setSculptConvertMode,
                      onSetSculptConvertResolution: _setSculptConvertResolution,
                      onStartSculptConvert: _startSculptConvert,
                      onResumeSculptingSelected: _resumeSculptingSelected,
                      onStopSculpting: _stopSculpting,
                      onSetSculptBrushMode: _setSculptBrushMode,
                      onSetSculptBrushRadius: _setSculptBrushRadius,
                      onSetSculptBrushStrength: _setSculptBrushStrength,
                      onSetSculptSymmetryAxis: _setSculptSymmetryAxis,
                      onSetSelectedSculptResolution: _setSelectedSculptResolution,
                      onSetSelectedLightType: _setSelectedLightType,
                      onSetSelectedLightColor: _setSelectedLightColor,
                      onSetSelectedLightIntensity: _setSelectedLightIntensity,
                      onSetSelectedLightRange: _setSelectedLightRange,
                      onSetSelectedLightSpotAngle: _setSelectedLightSpotAngle,
                      onSetSelectedLightCastShadows:
                          _setSelectedLightCastShadows,
                      onSetSelectedLightShadowSoftness:
                          _setSelectedLightShadowSoftness,
                      onSetSelectedLightShadowColor:
                          _setSelectedLightShadowColor,
                      onSetSelectedLightVolumetric:
                          _setSelectedLightVolumetric,
                      onSetSelectedLightVolumetricDensity:
                          _setSelectedLightVolumetricDensity,
                      onSetSelectedLightCookie: _setSelectedLightCookie,
                      onClearSelectedLightCookie: _clearSelectedLightCookie,
                      onSetSelectedLightProximityMode:
                          _setSelectedLightProximityMode,
                      onSetSelectedLightProximityRange:
                          _setSelectedLightProximityRange,
                      onSetSelectedLightArrayPattern:
                          _setSelectedLightArrayPattern,
                      onSetSelectedLightArrayCount:
                          _setSelectedLightArrayCount,
                      onSetSelectedLightArrayRadius:
                          _setSelectedLightArrayRadius,
                      onSetSelectedLightArrayColorVariation:
                          _setSelectedLightArrayColorVariation,
                      onSetSelectedLightIntensityExpression:
                          _setSelectedLightIntensityExpression,
                      onSetSelectedLightColorHueExpression:
                          _setSelectedLightColorHueExpression,
                      onSetNodeLightMask: _setNodeLightMask,
                      onSetNodeLightLinkEnabled: _setNodeLightLinkEnabled,
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
                        onCreateLight: _promptCreateLight,
                        onCreateSculpt: () =>
                            _runModalSceneCommand(createSculpt),
                        onRenameSelected: _promptRenameSelectedNode,
                        onDuplicateSelected: () =>
                            _runModalSceneCommand(duplicateSelected),
                        onUndo: () => _runModalSceneCommand(undo),
                        onRedo: () => _runModalSceneCommand(redo),
                        onDeleteSelected: () =>
                            _runModalSceneCommand(deleteSelected),
                        onNewScene: () => _runModalSceneCommand(newScene),
                        onOpenScene: () => _runModalSceneCommand(openScene),
                        onSaveScene: () => _runModalSceneCommand(saveScene),
                        onSaveSceneAs: () => _runModalSceneCommand(saveSceneAs),
                        onOpenRecentScene: (path) =>
                            _runModalSceneCommand(() => openRecentScene(path: path)),
                        onRecoverAutosave: () =>
                            _runModalSceneCommand(recoverAutosave),
                        onDiscardRecovery: () =>
                            _runModalSceneCommand(discardRecovery),
                        onSetExportResolution: (resolution) =>
                            _runModalSceneCommand(
                              () => setExportResolution(resolution: resolution),
                              requestNativeFrame: false,
                            ),
                        onSetAdaptiveExport: (enabled) =>
                            _runModalSceneCommand(
                              () => setAdaptiveExport(enabled: enabled),
                              requestNativeFrame: false,
                            ),
                        onStartExport: () =>
                            _runModalSceneCommand(
                              startExport,
                              requestNativeFrame: false,
                            ),
                        onCancelExport: () =>
                            _runModalSceneCommand(
                              cancelExport,
                              requestNativeFrame: false,
                            ),
                        onOpenImportDialog: () =>
                            _runModalSceneCommand(
                              openImportDialog,
                              requestNativeFrame: false,
                            ),
                        onCancelImportDialog: () =>
                            _runModalSceneCommand(
                              cancelImportDialog,
                              requestNativeFrame: false,
                            ),
                        onSetImportUseAuto: (useAuto) =>
                            _runModalSceneCommand(
                              () => setImportUseAuto(useAuto: useAuto),
                              requestNativeFrame: false,
                            ),
                        onSetImportResolution: (resolution) =>
                            _runModalSceneCommand(
                              () => setImportResolution(resolution: resolution),
                              requestNativeFrame: false,
                            ),
                        onStartImport: () =>
                            _runModalSceneCommand(
                              startImport,
                              requestNativeFrame: false,
                            ),
                        onCancelImport: () =>
                            _runModalSceneCommand(
                              cancelImport,
                              requestNativeFrame: false,
                            ),
                        onOpenSculptConvertDialog: () =>
                            _runModalSceneCommand(
                              openSculptConvertDialogForSelected,
                              requestNativeFrame: false,
                            ),
                        onCancelSculptConvertDialog: () =>
                            _runModalSceneCommand(
                              cancelSculptConvertDialog,
                              requestNativeFrame: false,
                            ),
                        onSetSculptConvertMode: (modeId) =>
                            _runModalSceneCommand(
                              () => setSculptConvertMode(modeId: modeId),
                              requestNativeFrame: false,
                            ),
                        onSetSculptConvertResolution: (resolution) =>
                            _runModalSceneCommand(
                              () => setSculptConvertResolution(
                                resolution: resolution,
                              ),
                              requestNativeFrame: false,
                            ),
                        onStartSculptConvert: () =>
                            _runModalSceneCommand(
                              startSculptConvert,
                              requestNativeFrame: false,
                            ),
                        onResumeSculptingSelected: () =>
                            _runModalSceneCommand(
                              resumeSculptingSelected,
                              requestNativeFrame: false,
                            ),
                        onStopSculpting: () =>
                            _runModalSceneCommand(
                              stopSculpting,
                              requestNativeFrame: false,
                            ),
                        onSetSculptBrushMode: (modeId) =>
                            _runModalSceneCommand(
                              () => setSculptBrushMode(modeId: modeId),
                              requestNativeFrame: false,
                            ),
                        onSetSculptBrushRadius: (radius) =>
                            _runModalSceneCommand(
                              () => setSculptBrushRadius(radius: radius),
                              requestNativeFrame: false,
                            ),
                        onSetSculptBrushStrength: (strength) =>
                            _runModalSceneCommand(
                              () => setSculptBrushStrength(strength: strength),
                              requestNativeFrame: false,
                            ),
                        onSetSculptSymmetryAxis: (axisId) =>
                            _runModalSceneCommand(
                              () => setSculptSymmetryAxis(axisId: axisId),
                              requestNativeFrame: false,
                            ),
                        onSetSelectedSculptResolution: (resolution) =>
                            _runModalSceneCommand(
                              () => setSelectedSculptResolution(
                                resolution: resolution,
                              ),
                              requestNativeFrame: false,
                            ),
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
                  onCreateLight: _promptCreateLight,
                  onCreateSculpt: () => _runSceneCommand(createSculpt),
                  onRenameSelected: _promptRenameSelectedNode,
                  onSetPrimitiveParameter: (parameterKey, value) =>
                      _runSceneCommand(
                        () => setSelectedPrimitiveParameter(
                          parameterKey: parameterKey,
                          value: value,
                        ),
                      ),
                  onSetMaterialFloat: (fieldId, value) => _runSceneCommand(
                    () => setSelectedMaterialFloat(
                      fieldId: fieldId,
                      value: value,
                    ),
                  ),
                  onSetMaterialColor: (fieldId, color) => _runSceneCommand(
                    () => setSelectedMaterialColor(
                      fieldId: fieldId,
                      red: color.x,
                      green: color.y,
                      blue: color.z,
                    ),
                  ),
                  onSetTransformPosition: (x, y, z) => _runSceneCommand(
                    () => setSelectedTransformPosition(x: x, y: y, z: z),
                  ),
                  onSetTransformRotationDegrees: (x, y, z) => _runSceneCommand(
                    () => setSelectedTransformRotationDegrees(
                      xDegrees: x,
                      yDegrees: y,
                      zDegrees: z,
                    ),
                  ),
                  onSetTransformScale: (x, y, z) => _runSceneCommand(
                    () => setSelectedTransformScale(x: x, y: y, z: z),
                  ),
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
                  onNewScene: _newScene,
                  onOpenScene: _openScene,
                  onSaveScene: _saveScene,
                  onSaveSceneAs: _saveSceneAs,
                  onOpenRecentScene: _openRecentScene,
                  onRecoverAutosave: _recoverAutosave,
                  onDiscardRecovery: _discardRecovery,
                  onSetExportResolution: _setExportResolution,
                  onSetAdaptiveExport: _setAdaptiveExport,
                  onStartExport: _startExport,
                  onCancelExport: _cancelExport,
                  onOpenImportDialog: _openImportDialog,
                  onCancelImportDialog: _cancelImportDialog,
                  onSetImportUseAuto: _setImportUseAuto,
                  onSetImportResolution: _setImportResolution,
                  onStartImport: _startImport,
                  onCancelImport: _cancelImport,
                  onOpenSculptConvertDialog: _openSculptConvertDialog,
                  onCancelSculptConvertDialog: _cancelSculptConvertDialog,
                  onSetSculptConvertMode: _setSculptConvertMode,
                  onSetSculptConvertResolution: _setSculptConvertResolution,
                  onStartSculptConvert: _startSculptConvert,
                  onResumeSculptingSelected: _resumeSculptingSelected,
                  onStopSculpting: _stopSculpting,
                  onSetSculptBrushMode: _setSculptBrushMode,
                  onSetSculptBrushRadius: _setSculptBrushRadius,
                  onSetSculptBrushStrength: _setSculptBrushStrength,
                  onSetSculptSymmetryAxis: _setSculptSymmetryAxis,
                  onSetSelectedSculptResolution: _setSelectedSculptResolution,
                  onSetSelectedLightType: _setSelectedLightType,
                  onSetSelectedLightColor: _setSelectedLightColor,
                  onSetSelectedLightIntensity: _setSelectedLightIntensity,
                  onSetSelectedLightRange: _setSelectedLightRange,
                  onSetSelectedLightSpotAngle: _setSelectedLightSpotAngle,
                  onSetSelectedLightCastShadows:
                      _setSelectedLightCastShadows,
                  onSetSelectedLightShadowSoftness:
                      _setSelectedLightShadowSoftness,
                  onSetSelectedLightShadowColor:
                      _setSelectedLightShadowColor,
                  onSetSelectedLightVolumetric:
                      _setSelectedLightVolumetric,
                  onSetSelectedLightVolumetricDensity:
                      _setSelectedLightVolumetricDensity,
                  onSetSelectedLightCookie: _setSelectedLightCookie,
                  onClearSelectedLightCookie: _clearSelectedLightCookie,
                  onSetSelectedLightProximityMode:
                      _setSelectedLightProximityMode,
                  onSetSelectedLightProximityRange:
                      _setSelectedLightProximityRange,
                  onSetSelectedLightArrayPattern:
                      _setSelectedLightArrayPattern,
                  onSetSelectedLightArrayCount:
                      _setSelectedLightArrayCount,
                  onSetSelectedLightArrayRadius:
                      _setSelectedLightArrayRadius,
                  onSetSelectedLightArrayColorVariation:
                      _setSelectedLightArrayColorVariation,
                  onSetSelectedLightIntensityExpression:
                      _setSelectedLightIntensityExpression,
                  onSetSelectedLightColorHueExpression:
                      _setSelectedLightColorHueExpression,
                  onSetNodeLightMask: _setNodeLightMask,
                  onSetNodeLightLinkEnabled: _setNodeLightLinkEnabled,
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
    required this.onCreateLight,
    required this.onCreateSculpt,
    required this.onRenameSelected,
    required this.onSetPrimitiveParameter,
    required this.onSetMaterialFloat,
    required this.onSetMaterialColor,
    required this.onSetTransformPosition,
    required this.onSetTransformRotationDegrees,
    required this.onSetTransformScale,
    required this.onDuplicateSelected,
    required this.onUndo,
    required this.onRedo,
    required this.onDeleteSelected,
    required this.onSelectSceneNode,
    required this.onToggleSceneNodeVisibility,
    required this.onToggleSceneNodeLock,
    required this.onNewScene,
    required this.onOpenScene,
    required this.onSaveScene,
    required this.onSaveSceneAs,
    required this.onOpenRecentScene,
    required this.onRecoverAutosave,
    required this.onDiscardRecovery,
    required this.onSetExportResolution,
    required this.onSetAdaptiveExport,
    required this.onStartExport,
    required this.onCancelExport,
    required this.onOpenImportDialog,
    required this.onCancelImportDialog,
    required this.onSetImportUseAuto,
    required this.onSetImportResolution,
    required this.onStartImport,
    required this.onCancelImport,
    required this.onOpenSculptConvertDialog,
    required this.onCancelSculptConvertDialog,
    required this.onSetSculptConvertMode,
    required this.onSetSculptConvertResolution,
    required this.onStartSculptConvert,
    required this.onResumeSculptingSelected,
    required this.onStopSculpting,
    required this.onSetSculptBrushMode,
    required this.onSetSculptBrushRadius,
    required this.onSetSculptBrushStrength,
    required this.onSetSculptSymmetryAxis,
    required this.onSetSelectedSculptResolution,
    required this.onSetSelectedLightType,
    required this.onSetSelectedLightColor,
    required this.onSetSelectedLightIntensity,
    required this.onSetSelectedLightRange,
    required this.onSetSelectedLightSpotAngle,
    required this.onSetSelectedLightCastShadows,
    required this.onSetSelectedLightShadowSoftness,
    required this.onSetSelectedLightShadowColor,
    required this.onSetSelectedLightVolumetric,
    required this.onSetSelectedLightVolumetricDensity,
    required this.onSetSelectedLightCookie,
    required this.onClearSelectedLightCookie,
    required this.onSetSelectedLightProximityMode,
    required this.onSetSelectedLightProximityRange,
    required this.onSetSelectedLightArrayPattern,
    required this.onSetSelectedLightArrayCount,
    required this.onSetSelectedLightArrayRadius,
    required this.onSetSelectedLightArrayColorVariation,
    required this.onSetSelectedLightIntensityExpression,
    required this.onSetSelectedLightColorHueExpression,
    required this.onSetNodeLightMask,
    required this.onSetNodeLightLinkEnabled,
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
  final VoidCallback onCreateLight;
  final VoidCallback onCreateSculpt;
  final VoidCallback onRenameSelected;
  final void Function(String parameterKey, double value) onSetPrimitiveParameter;
  final void Function(String fieldId, double value) onSetMaterialFloat;
  final void Function(String fieldId, AppVec3 color) onSetMaterialColor;
  final void Function(double x, double y, double z) onSetTransformPosition;
  final void Function(double x, double y, double z)
  onSetTransformRotationDegrees;
  final void Function(double x, double y, double z) onSetTransformScale;
  final VoidCallback onDuplicateSelected;
  final VoidCallback onUndo;
  final VoidCallback onRedo;
  final VoidCallback onDeleteSelected;
  final ValueChanged<int> onSelectSceneNode;
  final ValueChanged<int> onToggleSceneNodeVisibility;
  final ValueChanged<int> onToggleSceneNodeLock;
  final VoidCallback onNewScene;
  final VoidCallback onOpenScene;
  final VoidCallback onSaveScene;
  final VoidCallback onSaveSceneAs;
  final ValueChanged<String> onOpenRecentScene;
  final VoidCallback onRecoverAutosave;
  final VoidCallback onDiscardRecovery;
  final ValueChanged<int> onSetExportResolution;
  final ValueChanged<bool> onSetAdaptiveExport;
  final VoidCallback onStartExport;
  final VoidCallback onCancelExport;
  final VoidCallback onOpenImportDialog;
  final VoidCallback onCancelImportDialog;
  final ValueChanged<bool> onSetImportUseAuto;
  final ValueChanged<int> onSetImportResolution;
  final VoidCallback onStartImport;
  final VoidCallback onCancelImport;
  final VoidCallback onOpenSculptConvertDialog;
  final VoidCallback onCancelSculptConvertDialog;
  final ValueChanged<String> onSetSculptConvertMode;
  final ValueChanged<int> onSetSculptConvertResolution;
  final VoidCallback onStartSculptConvert;
  final VoidCallback onResumeSculptingSelected;
  final VoidCallback onStopSculpting;
  final ValueChanged<String> onSetSculptBrushMode;
  final ValueChanged<double> onSetSculptBrushRadius;
  final ValueChanged<double> onSetSculptBrushStrength;
  final ValueChanged<String> onSetSculptSymmetryAxis;
  final ValueChanged<int> onSetSelectedSculptResolution;
  final ValueChanged<String> onSetSelectedLightType;
  final ValueChanged<AppVec3> onSetSelectedLightColor;
  final ValueChanged<double> onSetSelectedLightIntensity;
  final ValueChanged<double> onSetSelectedLightRange;
  final ValueChanged<double> onSetSelectedLightSpotAngle;
  final ValueChanged<bool> onSetSelectedLightCastShadows;
  final ValueChanged<double> onSetSelectedLightShadowSoftness;
  final ValueChanged<AppVec3> onSetSelectedLightShadowColor;
  final ValueChanged<bool> onSetSelectedLightVolumetric;
  final ValueChanged<double> onSetSelectedLightVolumetricDensity;
  final ValueChanged<int> onSetSelectedLightCookie;
  final VoidCallback onClearSelectedLightCookie;
  final ValueChanged<String> onSetSelectedLightProximityMode;
  final ValueChanged<double> onSetSelectedLightProximityRange;
  final ValueChanged<String> onSetSelectedLightArrayPattern;
  final ValueChanged<int> onSetSelectedLightArrayCount;
  final ValueChanged<double> onSetSelectedLightArrayRadius;
  final ValueChanged<double> onSetSelectedLightArrayColorVariation;
  final ValueChanged<String> onSetSelectedLightIntensityExpression;
  final ValueChanged<String> onSetSelectedLightColorHueExpression;
  final void Function(int nodeId, int mask) onSetNodeLightMask;
  final void Function(int nodeId, int lightId, bool enabled)
  onSetNodeLightLinkEnabled;
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
    final selectedNodeProperties = snapshot?.selectedNodeProperties;
    final hoveredNode = viewportFeedback?.hoveredNode;
    final selectedNodeId = selectedNode?.id;
    final cameraControlsEnabled = !commandInFlight && currentCamera != null;
    final focusSelectedEnabled = !commandInFlight && selectedNode != null;
    final sceneCommandsEnabled = !commandInFlight;
    final undoEnabled = !commandInFlight && (snapshot?.history.canUndo ?? false);
    final redoEnabled = !commandInFlight && (snapshot?.history.canRedo ?? false);
    final renameSelectedEnabled = !commandInFlight && selectedNode != null;
    final createSculptEnabled = !commandInFlight && selectedNode != null;
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
              onCreateLight: onCreateLight,
              createSculptEnabled: createSculptEnabled,
              onCreateSculpt: onCreateSculpt,
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
            'Node Basics',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          if (snapshot == null)
            const Text('Scene snapshot is still loading.')
          else
            _NodeBasicsPanel(
              properties: selectedNodeProperties,
              enabled: !commandInFlight,
              onRenameSelected: onRenameSelected,
              onSetPrimitiveParameter: onSetPrimitiveParameter,
              onSetMaterialFloat: onSetMaterialFloat,
              onSetMaterialColor: onSetMaterialColor,
              onToggleVisibility: selectedNodeProperties == null
                  ? null
                  : () => onToggleSceneNodeVisibility(
                      selectedNodeProperties.nodeId,
                    ),
              onToggleLock: selectedNodeProperties == null
                  ? null
                  : () => onToggleSceneNodeLock(selectedNodeProperties.nodeId),
            ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Light Inspector',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          LightInspectorPanel(
            properties: selectedNodeProperties,
            lightLinking: snapshot?.lightLinking,
            enabled: !commandInFlight,
            onSetLightType: onSetSelectedLightType,
            onSetLightColor: onSetSelectedLightColor,
            onSetLightIntensity: onSetSelectedLightIntensity,
            onSetLightRange: onSetSelectedLightRange,
            onSetLightSpotAngle: onSetSelectedLightSpotAngle,
            onSetLightCastShadows: onSetSelectedLightCastShadows,
            onSetLightShadowSoftness: onSetSelectedLightShadowSoftness,
            onSetLightShadowColor: onSetSelectedLightShadowColor,
            onSetLightVolumetric: onSetSelectedLightVolumetric,
            onSetLightVolumetricDensity: onSetSelectedLightVolumetricDensity,
            onSetLightCookie: onSetSelectedLightCookie,
            onClearLightCookie: onClearSelectedLightCookie,
            onSetLightProximityMode: onSetSelectedLightProximityMode,
            onSetLightProximityRange: onSetSelectedLightProximityRange,
            onSetLightArrayPattern: onSetSelectedLightArrayPattern,
            onSetLightArrayCount: onSetSelectedLightArrayCount,
            onSetLightArrayRadius: onSetSelectedLightArrayRadius,
            onSetLightArrayColorVariation:
                onSetSelectedLightArrayColorVariation,
            onSetLightIntensityExpression:
                onSetSelectedLightIntensityExpression,
            onSetLightColorHueExpression:
                onSetSelectedLightColorHueExpression,
            onSetNodeLightMask: onSetNodeLightMask,
            onSetNodeLightLinkEnabled: onSetNodeLightLinkEnabled,
          ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Transform Inspector',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          if (snapshot == null)
            const Text('Scene snapshot is still loading.')
          else
            _TransformInspectorPanel(
              properties: selectedNodeProperties,
              enabled: !commandInFlight,
              onSetTransformPosition: onSetTransformPosition,
              onSetTransformRotationDegrees: onSetTransformRotationDegrees,
              onSetTransformScale: onSetTransformScale,
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
          DocumentSessionPanel(
            document: snapshot?.document,
            enabled: !commandInFlight,
            onNewScene: onNewScene,
            onOpenScene: onOpenScene,
            onSaveScene: onSaveScene,
            onSaveSceneAs: onSaveSceneAs,
            onOpenRecentScene: onOpenRecentScene,
            onRecoverAutosave: onRecoverAutosave,
            onDiscardRecovery: onDiscardRecovery,
          ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Export',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          ExportPanel(
            export: snapshot?.export,
            enabled: !commandInFlight,
            onSetResolution: onSetExportResolution,
            onSetAdaptive: onSetAdaptiveExport,
            onStartExport: onStartExport,
            onCancelExport: onCancelExport,
          ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Import',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          ImportPanel(
            importSnapshot: snapshot?.import,
            enabled: !commandInFlight,
            onOpenImportDialog: onOpenImportDialog,
            onCancelImportDialog: onCancelImportDialog,
            onSetUseAuto: onSetImportUseAuto,
            onSetResolution: onSetImportResolution,
            onStartImport: onStartImport,
            onCancelImport: onCancelImport,
          ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Sculpt Convert',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          SculptConvertPanel(
            selectedNode: selectedNode,
            sculptConvertSnapshot: snapshot?.sculptConvert,
            enabled: !commandInFlight,
            onOpenDialog: onOpenSculptConvertDialog,
            onCancelDialog: onCancelSculptConvertDialog,
            onSetMode: onSetSculptConvertMode,
            onSetResolution: onSetSculptConvertResolution,
            onStartConvert: onStartSculptConvert,
          ),
          const SizedBox(height: ShellTokens.sectionGap),
          Text(
            'Sculpt Workflow',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          SculptSessionPanel(
            selectedNode: selectedNode,
            sculptSnapshot: snapshot?.sculpt,
            enabled: !commandInFlight,
            onCreateSculpt: onCreateSculpt,
            onResumeSelected: onResumeSculptingSelected,
            onStopSculpting: onStopSculpting,
            onSetBrushMode: onSetSculptBrushMode,
            onSetBrushRadius: onSetSculptBrushRadius,
            onSetBrushStrength: onSetSculptBrushStrength,
            onSetSymmetryAxis: onSetSculptSymmetryAxis,
            onSetResolution: onSetSelectedSculptResolution,
          ),
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

class _NodeBasicsPanel extends StatefulWidget {
  const _NodeBasicsPanel({
    required this.properties,
    required this.enabled,
    required this.onRenameSelected,
    required this.onSetPrimitiveParameter,
    required this.onSetMaterialFloat,
    required this.onSetMaterialColor,
    required this.onToggleVisibility,
    required this.onToggleLock,
  });

  final AppSelectedNodePropertiesSnapshot? properties;
  final bool enabled;
  final VoidCallback onRenameSelected;
  final void Function(String parameterKey, double value) onSetPrimitiveParameter;
  final void Function(String fieldId, double value) onSetMaterialFloat;
  final void Function(String fieldId, AppVec3 color) onSetMaterialColor;
  final VoidCallback? onToggleVisibility;
  final VoidCallback? onToggleLock;

  @override
  State<_NodeBasicsPanel> createState() => _NodeBasicsPanelState();
}

class _NodeBasicsPanelState extends State<_NodeBasicsPanel> {
  Map<String, double> _scalarDrafts = <String, double>{};
  Map<String, AppVec3> _colorDrafts = <String, AppVec3>{};

  @override
  void initState() {
    super.initState();
    _resetDrafts();
  }

  @override
  void didUpdateWidget(covariant _NodeBasicsPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.properties != oldWidget.properties) {
      _resetDrafts();
    }
  }

  void _resetDrafts() {
    final selectedProperties = widget.properties;
    if (selectedProperties == null) {
      _scalarDrafts = <String, double>{};
      _colorDrafts = <String, AppVec3>{};
      return;
    }

    final nextScalarDrafts = <String, double>{};
    final nextColorDrafts = <String, AppVec3>{};
    for (final parameter in selectedProperties.primitive?.parameters ?? const []) {
      nextScalarDrafts['primitive:${parameter.key}'] = parameter.value;
    }
    final material = selectedProperties.material;
    if (material != null) {
      nextScalarDrafts['material:roughness'] = material.roughness;
      nextScalarDrafts['material:metallic'] = material.metallic;
      nextScalarDrafts['material:fresnel'] = material.fresnel;
      nextScalarDrafts['material:emissive_intensity'] = material.emissiveIntensity;
      nextColorDrafts['material:color'] = material.color;
      nextColorDrafts['material:emissive'] = material.emissive;
    }
    _scalarDrafts = nextScalarDrafts;
    _colorDrafts = nextColorDrafts;
  }

  double _scalarValue(String key, double fallback) {
    return _scalarDrafts[key] ?? fallback;
  }

  void _setScalarValue(String key, double value) {
    setState(() {
      _scalarDrafts[key] = value;
    });
  }

  AppVec3 _colorValue(String key, AppVec3 fallback) {
    return _colorDrafts[key] ?? fallback;
  }

  void _setColorComponent(
    String key,
    AppVec3 fallback,
    int component,
    double value,
  ) {
    final current = _colorValue(key, fallback);
    final next = switch (component) {
      0 => AppVec3(x: value, y: current.y, z: current.z),
      1 => AppVec3(x: current.x, y: value, z: current.z),
      _ => AppVec3(x: current.x, y: current.y, z: value),
    };
    setState(() {
      _colorDrafts[key] = next;
    });
  }

  @override
  Widget build(BuildContext context) {
    final selectedProperties = widget.properties;
    if (selectedProperties == null) {
      return const Text('Select a node to inspect backend-owned properties.');
    }

    final primitive = selectedProperties.primitive;
    final material = selectedProperties.material;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              selectedProperties.name,
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            Text('Type: ${selectedProperties.kindLabel}'),
            const SizedBox(height: ShellTokens.controlGap),
            FilledButton.icon(
              key: const ValueKey('node-basics-rename'),
              onPressed: widget.enabled ? widget.onRenameSelected : null,
              icon: const Icon(Icons.edit_outlined),
              label: const Text('Rename'),
            ),
            const SizedBox(height: ShellTokens.controlGap),
            SwitchListTile.adaptive(
              key: const ValueKey('node-basics-visible-toggle'),
              contentPadding: EdgeInsets.zero,
              value: selectedProperties.visible,
              onChanged: widget.enabled && widget.onToggleVisibility != null
                  ? (_) => widget.onToggleVisibility!()
                  : null,
              title: const Text('Visible'),
              subtitle: const Text(
                'Keep this node active in the backend-rendered scene.',
              ),
            ),
            SwitchListTile.adaptive(
              key: const ValueKey('node-basics-lock-toggle'),
              contentPadding: EdgeInsets.zero,
              value: selectedProperties.locked,
              onChanged: widget.enabled && widget.onToggleLock != null
                  ? (_) => widget.onToggleLock!()
                  : null,
              title: const Text('Locked'),
              subtitle: const Text(
                'Prevent backend document commands from mutating this node.',
              ),
            ),
            if (primitive != null && primitive.parameters.isNotEmpty) ...[
              const SizedBox(height: ShellTokens.controlGap),
              Text(
                'Primitive Parameters',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              for (final parameter in primitive.parameters)
                _ScalarPropertyEditor(
                  key: ValueKey('primitive-parameter-${parameter.key}-slider'),
                  label: parameter.label,
                  value: _scalarValue(
                    'primitive:${parameter.key}',
                    parameter.value,
                  ),
                  min: _primitiveParameterMin,
                  max: _primitiveParameterMax,
                  enabled: widget.enabled,
                  onChanged: (value) => _setScalarValue(
                    'primitive:${parameter.key}',
                    value,
                  ),
                  onChangeEnd: (value) => widget.onSetPrimitiveParameter(
                    parameter.key,
                    value,
                  ),
                ),
            ],
            if (material != null) ...[
              const SizedBox(height: ShellTokens.controlGap),
              Text(
                'Material Basics',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              const SizedBox(height: ShellTokens.compactGap),
              _ColorPropertyEditor(
                label: 'Color',
                keyPrefix: 'material-color',
                value: _colorValue('material:color', material.color),
                enabled: widget.enabled,
                onChanged: (component, value) => _setColorComponent(
                  'material:color',
                  material.color,
                  component,
                  value,
                ),
                onChangeEnd: (_) => widget.onSetMaterialColor(
                  'color',
                  _colorValue('material:color', material.color),
                ),
              ),
              _ScalarPropertyEditor(
                key: const ValueKey('material-metallic-slider'),
                label: 'Metallic',
                value: _scalarValue('material:metallic', material.metallic),
                min: _materialFactorMin,
                max: _materialFactorMax,
                enabled: widget.enabled,
                onChanged: (value) =>
                    _setScalarValue('material:metallic', value),
                onChangeEnd: (value) =>
                    widget.onSetMaterialFloat('metallic', value),
              ),
              _ScalarPropertyEditor(
                key: const ValueKey('material-roughness-slider'),
                label: 'Roughness',
                value: _scalarValue('material:roughness', material.roughness),
                min: _materialFactorMin,
                max: _materialFactorMax,
                enabled: widget.enabled,
                onChanged: (value) =>
                    _setScalarValue('material:roughness', value),
                onChangeEnd: (value) =>
                    widget.onSetMaterialFloat('roughness', value),
              ),
              _ScalarPropertyEditor(
                key: const ValueKey('material-fresnel-slider'),
                label: 'Fresnel',
                value: _scalarValue('material:fresnel', material.fresnel),
                min: _materialFactorMin,
                max: _materialFactorMax,
                enabled: widget.enabled,
                onChanged: (value) =>
                    _setScalarValue('material:fresnel', value),
                onChangeEnd: (value) =>
                    widget.onSetMaterialFloat('fresnel', value),
              ),
              _ColorPropertyEditor(
                label: 'Emissive',
                keyPrefix: 'material-emissive',
                value: _colorValue('material:emissive', material.emissive),
                enabled: widget.enabled,
                onChanged: (component, value) => _setColorComponent(
                  'material:emissive',
                  material.emissive,
                  component,
                  value,
                ),
                onChangeEnd: (_) => widget.onSetMaterialColor(
                  'emissive',
                  _colorValue('material:emissive', material.emissive),
                ),
              ),
              _ScalarPropertyEditor(
                key: const ValueKey('material-emissive-intensity-slider'),
                label: 'Emissive Intensity',
                value: _scalarValue(
                  'material:emissive_intensity',
                  material.emissiveIntensity,
                ),
                min: _materialFactorMin,
                max: _emissiveIntensityMax,
                enabled: widget.enabled,
                onChanged: (value) =>
                    _setScalarValue('material:emissive_intensity', value),
                onChangeEnd: (value) =>
                    widget.onSetMaterialFloat('emissive_intensity', value),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _TransformInspectorPanel extends StatefulWidget {
  const _TransformInspectorPanel({
    required this.properties,
    required this.enabled,
    required this.onSetTransformPosition,
    required this.onSetTransformRotationDegrees,
    required this.onSetTransformScale,
  });

  final AppSelectedNodePropertiesSnapshot? properties;
  final bool enabled;
  final void Function(double x, double y, double z) onSetTransformPosition;
  final void Function(double x, double y, double z)
  onSetTransformRotationDegrees;
  final void Function(double x, double y, double z) onSetTransformScale;

  @override
  State<_TransformInspectorPanel> createState() =>
      _TransformInspectorPanelState();
}

class _TransformInspectorPanelState extends State<_TransformInspectorPanel> {
  Map<String, double> _drafts = <String, double>{};

  @override
  void initState() {
    super.initState();
    _resetDrafts();
  }

  @override
  void didUpdateWidget(covariant _TransformInspectorPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.properties != oldWidget.properties) {
      _resetDrafts();
    }
  }

  void _resetDrafts() {
    final transform = widget.properties?.transform;
    if (transform == null) {
      _drafts = <String, double>{};
      return;
    }

    _drafts = <String, double>{
      'position:x': transform.position.x,
      'position:y': transform.position.y,
      'position:z': transform.position.z,
      'rotation:x': transform.rotationDegrees.x,
      'rotation:y': transform.rotationDegrees.y,
      'rotation:z': transform.rotationDegrees.z,
      if (transform.scale != null) ...<String, double>{
        'scale:x': transform.scale!.x,
        'scale:y': transform.scale!.y,
        'scale:z': transform.scale!.z,
      },
    };
  }

  double _draftValue(String key, double fallback) {
    return _drafts[key] ?? fallback;
  }

  void _setDraftValue(String key, double value) {
    setState(() {
      _drafts[key] = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    final selectedProperties = widget.properties;
    final transform = selectedProperties?.transform;
    if (selectedProperties == null || transform == null) {
      return const Text(
        'Select a node to edit backend-owned translation, rotation, and scale values.',
      );
    }

    final scale = transform.scale;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(ShellTokens.panelPadding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '${selectedProperties.name} (${selectedProperties.kindLabel})',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: ShellTokens.compactGap),
            Text(
              'Direct edits stay on the Rust command path so later viewport tooling can reuse the same backend semantics.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: ShellTokens.controlGap),
            _Vector3PropertyEditor(
              label: transform.positionLabel,
              keyPrefix: 'transform-position',
              summary: _formatVec3(
                AppVec3(
                  x: _draftValue('position:x', transform.position.x),
                  y: _draftValue('position:y', transform.position.y),
                  z: _draftValue('position:z', transform.position.z),
                ),
              ),
              values: (
                _draftValue('position:x', transform.position.x),
                _draftValue('position:y', transform.position.y),
                _draftValue('position:z', transform.position.z),
              ),
              min: _transformTranslationMin,
              max: _transformTranslationMax,
              enabled: widget.enabled,
              onChanged: (axis, value) =>
                  _setDraftValue('position:$axis', value),
              onChangeEnd: () => widget.onSetTransformPosition(
                _draftValue('position:x', transform.position.x),
                _draftValue('position:y', transform.position.y),
                _draftValue('position:z', transform.position.z),
              ),
            ),
            _Vector3PropertyEditor(
              label: 'Rotation',
              keyPrefix: 'transform-rotation',
              summary:
                  '${_formatVec3(AppVec3(x: _draftValue('rotation:x', transform.rotationDegrees.x), y: _draftValue('rotation:y', transform.rotationDegrees.y), z: _draftValue('rotation:z', transform.rotationDegrees.z)))} deg',
              values: (
                _draftValue('rotation:x', transform.rotationDegrees.x),
                _draftValue('rotation:y', transform.rotationDegrees.y),
                _draftValue('rotation:z', transform.rotationDegrees.z),
              ),
              min: _transformRotationDegreesMin,
              max: _transformRotationDegreesMax,
              enabled: widget.enabled,
              onChanged: (axis, value) =>
                  _setDraftValue('rotation:$axis', value),
              onChangeEnd: () => widget.onSetTransformRotationDegrees(
                _draftValue('rotation:x', transform.rotationDegrees.x),
                _draftValue('rotation:y', transform.rotationDegrees.y),
                _draftValue('rotation:z', transform.rotationDegrees.z),
              ),
            ),
            if (scale != null)
              _Vector3PropertyEditor(
                label: 'Scale',
                keyPrefix: 'transform-scale',
                summary: _formatVec3(
                  AppVec3(
                    x: _draftValue('scale:x', scale.x),
                    y: _draftValue('scale:y', scale.y),
                    z: _draftValue('scale:z', scale.z),
                  ),
                ),
                values: (
                  _draftValue('scale:x', scale.x),
                  _draftValue('scale:y', scale.y),
                  _draftValue('scale:z', scale.z),
                ),
                min: _primitiveParameterMin,
                max: _primitiveParameterMax,
                enabled: widget.enabled,
                onChanged: (axis, value) => _setDraftValue('scale:$axis', value),
                onChangeEnd: () => widget.onSetTransformScale(
                  _draftValue('scale:x', scale.x),
                  _draftValue('scale:y', scale.y),
                  _draftValue('scale:z', scale.z),
                ),
              )
            else
              Text(
                'Scale stays fixed on this node type. Wrap geometry in a transform to edit scale directly.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }
}

class _Vector3PropertyEditor extends StatelessWidget {
  const _Vector3PropertyEditor({
    required this.label,
    required this.keyPrefix,
    required this.summary,
    required this.values,
    required this.min,
    required this.max,
    required this.enabled,
    required this.onChanged,
    required this.onChangeEnd,
  });

  final String label;
  final String keyPrefix;
  final String summary;
  final (double, double, double) values;
  final double min;
  final double max;
  final bool enabled;
  final void Function(String axis, double value) onChanged;
  final VoidCallback onChangeEnd;

  @override
  Widget build(BuildContext context) {
    final axes = <(String, double, String)>[
      ('X', values.$1, 'x'),
      ('Y', values.$2, 'y'),
      ('Z', values.$3, 'z'),
    ];

    return Padding(
      padding: const EdgeInsets.only(bottom: ShellTokens.controlGap),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$label: $summary',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: ShellTokens.compactGap),
          for (final axis in axes)
            _ScalarPropertyEditor(
              key: ValueKey('$keyPrefix-${axis.$3}-slider'),
              label: axis.$1,
              value: axis.$2,
              min: min,
              max: max,
              enabled: enabled,
              onChanged: (value) => onChanged(axis.$3, value),
              onChangeEnd: (_) => onChangeEnd(),
            ),
        ],
      ),
    );
  }
}

class _ScalarPropertyEditor extends StatelessWidget {
  const _ScalarPropertyEditor({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.enabled,
    required this.onChanged,
    required this.onChangeEnd,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final bool enabled;
  final ValueChanged<double> onChanged;
  final ValueChanged<double> onChangeEnd;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: ShellTokens.controlGap),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label: ${value.toStringAsFixed(2)}'),
          Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            onChanged: enabled ? onChanged : null,
            onChangeEnd: enabled ? onChangeEnd : null,
          ),
        ],
      ),
    );
  }
}

class _ColorPropertyEditor extends StatelessWidget {
  const _ColorPropertyEditor({
    required this.label,
    required this.keyPrefix,
    required this.value,
    required this.enabled,
    required this.onChanged,
    required this.onChangeEnd,
  });

  final String label;
  final String keyPrefix;
  final AppVec3 value;
  final bool enabled;
  final void Function(int component, double value) onChanged;
  final ValueChanged<AppVec3> onChangeEnd;

  @override
  Widget build(BuildContext context) {
    final components = <(String, double, String)>[
      ('R', value.x, 'red'),
      ('G', value.y, 'green'),
      ('B', value.z, 'blue'),
    ];

    return Padding(
      padding: const EdgeInsets.only(bottom: ShellTokens.controlGap),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$label: ${value.x.toStringAsFixed(2)}, ${value.y.toStringAsFixed(2)}, ${value.z.toStringAsFixed(2)}',
          ),
          for (final (index, component) in components.indexed)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(component.$1),
                Slider(
                  key: ValueKey('$keyPrefix-${component.$3}-slider'),
                  value: component.$2.clamp(
                    _materialFactorMin,
                    _materialFactorMax,
                  ),
                  min: _materialFactorMin,
                  max: _materialFactorMax,
                  onChanged: enabled
                      ? (nextValue) => onChanged(index, nextValue)
                      : null,
                  onChangeEnd: enabled ? (_) => onChangeEnd(value) : null,
                ),
              ],
            ),
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
    required this.onCreateLight,
    required this.createSculptEnabled,
    required this.onCreateSculpt,
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
  final VoidCallback onCreateLight;
  final bool createSculptEnabled;
  final VoidCallback onCreateSculpt;
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
          key: const ValueKey('create-light-command'),
          onPressed: sceneCommandsEnabled ? onCreateLight : null,
          child: const Text('Light'),
        ),
        OutlinedButton(
          key: const ValueKey('create-sculpt-command'),
          onPressed: createSculptEnabled ? onCreateSculpt : null,
          child: const Text('Sculpt'),
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
    required this.onCreateLight,
    required this.onCreateSculpt,
    required this.onRenameSelected,
    required this.onDuplicateSelected,
    required this.onUndo,
    required this.onRedo,
    required this.onDeleteSelected,
    required this.onNewScene,
    required this.onOpenScene,
    required this.onSaveScene,
    required this.onSaveSceneAs,
    required this.onOpenRecentScene,
    required this.onRecoverAutosave,
    required this.onDiscardRecovery,
    required this.onSetExportResolution,
    required this.onSetAdaptiveExport,
    required this.onStartExport,
    required this.onCancelExport,
    required this.onOpenImportDialog,
    required this.onCancelImportDialog,
    required this.onSetImportUseAuto,
    required this.onSetImportResolution,
    required this.onStartImport,
    required this.onCancelImport,
    required this.onOpenSculptConvertDialog,
    required this.onCancelSculptConvertDialog,
    required this.onSetSculptConvertMode,
    required this.onSetSculptConvertResolution,
    required this.onStartSculptConvert,
    required this.onResumeSculptingSelected,
    required this.onStopSculpting,
    required this.onSetSculptBrushMode,
    required this.onSetSculptBrushRadius,
    required this.onSetSculptBrushStrength,
    required this.onSetSculptSymmetryAxis,
    required this.onSetSelectedSculptResolution,
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
  final VoidCallback onCreateLight;
  final VoidCallback onCreateSculpt;
  final VoidCallback onRenameSelected;
  final VoidCallback onDuplicateSelected;
  final VoidCallback onUndo;
  final VoidCallback onRedo;
  final VoidCallback onDeleteSelected;
  final VoidCallback onNewScene;
  final VoidCallback onOpenScene;
  final VoidCallback onSaveScene;
  final VoidCallback onSaveSceneAs;
  final ValueChanged<String> onOpenRecentScene;
  final VoidCallback onRecoverAutosave;
  final VoidCallback onDiscardRecovery;
  final ValueChanged<int> onSetExportResolution;
  final ValueChanged<bool> onSetAdaptiveExport;
  final VoidCallback onStartExport;
  final VoidCallback onCancelExport;
  final VoidCallback onOpenImportDialog;
  final VoidCallback onCancelImportDialog;
  final ValueChanged<bool> onSetImportUseAuto;
  final ValueChanged<int> onSetImportResolution;
  final VoidCallback onStartImport;
  final VoidCallback onCancelImport;
  final VoidCallback onOpenSculptConvertDialog;
  final VoidCallback onCancelSculptConvertDialog;
  final ValueChanged<String> onSetSculptConvertMode;
  final ValueChanged<int> onSetSculptConvertResolution;
  final VoidCallback onStartSculptConvert;
  final VoidCallback onResumeSculptingSelected;
  final VoidCallback onStopSculpting;
  final ValueChanged<String> onSetSculptBrushMode;
  final ValueChanged<double> onSetSculptBrushRadius;
  final ValueChanged<double> onSetSculptBrushStrength;
  final ValueChanged<String> onSetSculptSymmetryAxis;
  final ValueChanged<int> onSetSelectedSculptResolution;
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
    final createSculptEnabled = !commandInFlight && selectedNode != null;
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
          onCreateLight: onCreateLight,
          createSculptEnabled: createSculptEnabled,
          onCreateSculpt: onCreateSculpt,
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
        DocumentSessionPanel(
          document: snapshot?.document,
          enabled: sceneCommandsEnabled,
          onNewScene: onNewScene,
          onOpenScene: onOpenScene,
          onSaveScene: onSaveScene,
          onSaveSceneAs: onSaveSceneAs,
          onOpenRecentScene: onOpenRecentScene,
          onRecoverAutosave: onRecoverAutosave,
          onDiscardRecovery: onDiscardRecovery,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Export',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        ExportPanel(
          export: snapshot?.export,
          enabled: sceneCommandsEnabled,
          onSetResolution: onSetExportResolution,
          onSetAdaptive: onSetAdaptiveExport,
          onStartExport: onStartExport,
          onCancelExport: onCancelExport,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Import',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        ImportPanel(
          importSnapshot: snapshot?.import,
          enabled: sceneCommandsEnabled,
          onOpenImportDialog: onOpenImportDialog,
          onCancelImportDialog: onCancelImportDialog,
          onSetUseAuto: onSetImportUseAuto,
          onSetResolution: onSetImportResolution,
          onStartImport: onStartImport,
          onCancelImport: onCancelImport,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Sculpt Convert',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        SculptConvertPanel(
          selectedNode: selectedNode,
          sculptConvertSnapshot: snapshot?.sculptConvert,
          enabled: sceneCommandsEnabled,
          onOpenDialog: onOpenSculptConvertDialog,
          onCancelDialog: onCancelSculptConvertDialog,
          onSetMode: onSetSculptConvertMode,
          onSetResolution: onSetSculptConvertResolution,
          onStartConvert: onStartSculptConvert,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
        Text(
          'Sculpt Workflow',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: ShellTokens.controlGap),
        SculptSessionPanel(
          selectedNode: selectedNode,
          sculptSnapshot: snapshot?.sculpt,
          enabled: sceneCommandsEnabled,
          onCreateSculpt: onCreateSculpt,
          onResumeSelected: onResumeSculptingSelected,
          onStopSculpting: onStopSculpting,
          onSetBrushMode: onSetSculptBrushMode,
          onSetBrushRadius: onSetSculptBrushRadius,
          onSetBrushStrength: onSetSculptBrushStrength,
          onSetSymmetryAxis: onSetSculptSymmetryAxis,
          onSetResolution: onSetSelectedSculptResolution,
        ),
        const SizedBox(height: ShellTokens.sectionGap),
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
