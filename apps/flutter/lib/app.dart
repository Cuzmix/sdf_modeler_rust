import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_bridge.dart';
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
  static const Duration _targetFrameInterval = Duration(milliseconds: 41);
  static const Duration _snapshotRefreshDelay = Duration(milliseconds: 120);
  static const int _defaultFrameWidth = 640;
  static const int _defaultFrameHeight = 360;
  static const int _minimumFrameWidth = 320;
  static const int _minimumFrameHeight = 180;
  static const int _maximumFrameDimension = 4096;
  static const int _frameDimensionAlignment = 8;
  static const double _minimumRenderScale = 0.6;
  static const double _maximumRenderScale = 1.0;
  static const double _renderScaleIncreaseStep = 0.05;
  static const double _renderScaleDecreaseStep = 0.1;
  static const int _minimumFramesBetweenScaleAdjustments = 12;
  static const double _scaleUpFrameLoadRatio = 0.6;
  static const double _scaleDownFrameLoadRatio = 0.95;

  String _statusLine = 'Checking Rust bridge...';
  String _versionLine = '';
  String _previewLine = 'Initializing viewport texture...';

  AppSceneSnapshot? _sceneSnapshot;
  int? _textureId;
  Timer? _renderTimer;
  Timer? _snapshotRefreshTimer;
  final Stopwatch _elapsed = Stopwatch();
  bool _renderInFlight = false;
  bool _commandInFlight = false;
  int _renderedFrames = 0;
  double _averageRenderMs = 0.0;
  int _frameWidth = _defaultFrameWidth;
  int _frameHeight = _defaultFrameHeight;
  double _renderScale = _maximumRenderScale;
  int _framesSinceScaleAdjustment = 0;
  Size _lastLogicalViewportSize = Size.zero;
  double _lastDevicePixelRatio = 1.0;

  @override
  void initState() {
    super.initState();
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

      if (!mounted) {
        return;
      }

      setState(() {
        _statusLine = 'Rust ping: $pingValue';
        _versionLine = 'Bridge crate version: $versionValue';
        _textureId = createdTextureId;
        _sceneSnapshot = snapshot;
        _previewLine = _buildPreviewLine();
      });

      _elapsed.start();
      _startRenderLoop();
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

  String _buildPreviewLine() {
    final scalePercent = (_renderScale * 100).round();
    final base =
        'Viewport render target: ${_frameWidth}x$_frameHeight at $scalePercent% scale';
    if (_renderedFrames == 0) {
      return '$base (real renderer).';
    }

    return '$base ($_renderedFrames frames, ${_averageRenderMs.toStringAsFixed(1)} ms render).';
  }

  void _startRenderLoop() {
    _scheduleNextFrame(Duration.zero);
  }

  void _scheduleNextFrame(Duration delay) {
    _renderTimer?.cancel();
    _renderTimer = Timer(delay, () {
      unawaited(_renderFrame());
    });
  }

  Duration _computeNextFrameDelay(Stopwatch frameStopwatch) {
    final remainingMicros =
        _targetFrameInterval.inMicroseconds -
        frameStopwatch.elapsedMicroseconds;
    if (remainingMicros <= 0) {
      return Duration.zero;
    }

    return Duration(microseconds: remainingMicros);
  }

  Future<void> _renderFrame() async {
    final activeTextureId = _textureId;
    if (_renderInFlight || activeTextureId == null) {
      return;
    }

    _renderInFlight = true;
    final frameStopwatch = Stopwatch()..start();
    var shouldScheduleNextFrame = false;

    try {
      final elapsedSeconds = _elapsed.elapsedMicroseconds / 1000000.0;
      final pixels = await renderPreviewFrame(
        width: _frameWidth,
        height: _frameHeight,
        timeSeconds: elapsedSeconds,
      );

      if (!mounted || _textureId != activeTextureId) {
        return;
      }

      await TextureBridge.instance.updateTexture(
        textureId: activeTextureId,
        width: _frameWidth,
        height: _frameHeight,
        pixels: pixels,
      );

      _renderedFrames += 1;
      final renderMs = frameStopwatch.elapsedMicroseconds / 1000.0;
      _averageRenderMs = _renderedFrames == 1
          ? renderMs
          : (_averageRenderMs * 0.85) + (renderMs * 0.15);
      _framesSinceScaleAdjustment += 1;
      _updateAdaptiveRenderScale();
      shouldScheduleNextFrame = true;

      if (mounted && _renderedFrames % 10 == 0) {
        setState(() {
          _previewLine = _buildPreviewLine();
        });
      }
    } catch (error) {
      _renderTimer?.cancel();
      if (mounted) {
        setState(() {
          _previewLine = 'Viewport update error: $error';
        });
      }
    } finally {
      _renderInFlight = false;
      if (shouldScheduleNextFrame && mounted && _textureId == activeTextureId) {
        _scheduleNextFrame(_computeNextFrameDelay(frameStopwatch));
      }
    }
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
      });
    } catch (error) {
      setState(() {
        _statusLine = 'Rust bridge error: $error';
        _versionLine = '';
      });
    }
  }

  void _refreshSceneSnapshot() {
    try {
      final snapshot = _decodeSnapshot(sceneSnapshotJson());
      if (!mounted) {
        return;
      }
      setState(() {
        _sceneSnapshot = snapshot;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = 'Scene snapshot error: $error';
      });
    }
  }

  void _scheduleSceneSnapshotRefresh() {
    _snapshotRefreshTimer?.cancel();
    _snapshotRefreshTimer = Timer(_snapshotRefreshDelay, _refreshSceneSnapshot);
  }

  void _runViewportCommand(VoidCallback command) {
    try {
      command();
      _scheduleSceneSnapshotRefresh();
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = 'Viewport input error: $error';
      });
    }
  }

  void _handleViewportOrbitDrag(Offset delta) {
    _runViewportCommand(() {
      orbitCamera(deltaX: delta.dx, deltaY: delta.dy);
    });
  }

  void _handleViewportPanDrag(Offset delta) {
    _runViewportCommand(() {
      panCamera(deltaX: delta.dx, deltaY: delta.dy);
    });
  }

  void _handleViewportScroll(double deltaY) {
    _runViewportCommand(() {
      zoomCamera(delta: deltaY);
    });
  }

  void _handleViewportPrimaryTap(
    Offset localPosition,
    Size logicalViewportSize,
  ) {
    try {
      final snapshot = _decodeSnapshot(
        selectNodeAtViewport(
          mouseX: _mapViewportCoordinate(
            localPosition.dx,
            logicalViewportSize.width,
            _frameWidth,
          ),
          mouseY: _mapViewportCoordinate(
            localPosition.dy,
            logicalViewportSize.height,
            _frameHeight,
          ),
          width: _frameWidth,
          height: _frameHeight,
          timeSeconds: _elapsed.elapsedMicroseconds / 1000000.0,
        ),
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _sceneSnapshot = snapshot;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _previewLine = 'Viewport pick error: $error';
      });
    }
  }

  void _handleViewportInteractionEnd() {
    _scheduleSceneSnapshotRefresh();
  }

  double _mapViewportCoordinate(
    double logicalCoordinate,
    double logicalExtent,
    int renderExtent,
  ) {
    if (logicalExtent <= 0 || renderExtent <= 1) {
      return 0.0;
    }

    final clampedCoordinate = logicalCoordinate.clamp(0.0, logicalExtent);
    return (clampedCoordinate / logicalExtent) * (renderExtent - 1);
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
    final scaledDevicePixelRatio = devicePixelRatio * _renderScale;
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
    return true;
  }

  void _updateAdaptiveRenderScale() {
    if (_framesSinceScaleAdjustment < _minimumFramesBetweenScaleAdjustments) {
      return;
    }

    final targetFrameMs = _targetFrameInterval.inMicroseconds / 1000.0;
    double? nextRenderScale;

    if (_averageRenderMs > targetFrameMs * _scaleDownFrameLoadRatio &&
        _renderScale > _minimumRenderScale) {
      nextRenderScale = math.max(
        _minimumRenderScale,
        _renderScale - _renderScaleDecreaseStep,
      );
    } else if (_averageRenderMs < targetFrameMs * _scaleUpFrameLoadRatio &&
        _renderScale < _maximumRenderScale) {
      nextRenderScale = math.min(
        _maximumRenderScale,
        _renderScale + _renderScaleIncreaseStep,
      );
    }

    if (nextRenderScale == null || nextRenderScale == _renderScale) {
      return;
    }

    _renderScale = nextRenderScale;
    _framesSinceScaleAdjustment = 0;
    final renderTargetChanged = _updateRenderTargetSize();
    if (!renderTargetChanged) {
      if (!mounted) {
        _previewLine = _buildPreviewLine();
        return;
      }

      setState(() {
        _previewLine = _buildPreviewLine();
      });
    }
  }

  int _alignFrameDimension(int value, int minimumValue) {
    if (value <= minimumValue) {
      return minimumValue;
    }

    final alignedValue = (value ~/ _frameDimensionAlignment) *
        _frameDimensionAlignment;
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
      });
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
    _renderTimer?.cancel();
    _snapshotRefreshTimer?.cancel();
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
              onScroll: _handleViewportScroll,
              onInteractionEnd: _handleViewportInteractionEnd,
            );
            final inspectorPanel = _InspectorPanel(
              snapshot: snapshot,
              statusLine: _statusLine,
              versionLine: _versionLine,
              previewLine: _previewLine,
              commandInFlight: _commandInFlight,
              onRefresh: _refreshBridgeStatus,
              onAddSphere: () => _runSceneCommand(addSphere),
              onFrameAll: () => _runSceneCommand(frameAll),
              onResetScene: () => _runSceneCommand(resetScene),
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
    required this.statusLine,
    required this.versionLine,
    required this.previewLine,
    required this.commandInFlight,
    required this.onRefresh,
    required this.onAddSphere,
    required this.onFrameAll,
    required this.onResetScene,
  });

  final AppSceneSnapshot? snapshot;
  final String statusLine;
  final String versionLine;
  final String previewLine;
  final bool commandInFlight;
  final VoidCallback onRefresh;
  final VoidCallback onAddSphere;
  final VoidCallback onFrameAll;
  final VoidCallback onResetScene;

  @override
  Widget build(BuildContext context) {
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
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton(
                  onPressed: commandInFlight ? null : onAddSphere,
                  child: const Text('Add Sphere'),
                ),
                OutlinedButton(
                  onPressed: commandInFlight ? null : onFrameAll,
                  child: const Text('Frame All'),
                ),
                OutlinedButton(
                  onPressed: commandInFlight ? null : onResetScene,
                  child: const Text('Reset Scene'),
                ),
                TextButton(onPressed: onRefresh, child: const Text('Re-run Ping')),
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
              Text('Selected: ${snapshot!.selectedNode?.name ?? 'None'}'),
              Text(
                'Tool: ${snapshot!.tool.activeToolLabel} • ${snapshot!.tool.shadingModeLabel}',
              ),
              Text(
                'Camera distance: ${snapshot!.camera.distance.toStringAsFixed(2)} • ${snapshot!.camera.orthographic ? 'Ortho' : 'Perspective'}',
              ),
              Text(
                'Scene nodes: ${snapshot!.stats.totalNodes} total • ${snapshot!.stats.visibleNodes} visible • ${snapshot!.stats.topLevelNodes} roots',
              ),
              Text(
                'SDF complexity: ${snapshot!.stats.sdfEvalComplexity} • Voxel memory: ${snapshot!.stats.voxelMemoryBytes} bytes',
              ),
              const SizedBox(height: 16),
              Text(
                'Top-Level Nodes',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              if (snapshot!.topLevelNodes.isEmpty)
                const Text('No top-level nodes in the current scene.')
              else
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: snapshot!.topLevelNodes
                      .map(
                        (node) => Chip(
                          label: Text('${node.name} • ${node.kindLabel}'),
                        ),
                      )
                      .toList(growable: false),
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