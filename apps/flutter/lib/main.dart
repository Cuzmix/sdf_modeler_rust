import 'dart:async';

import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_bridge.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const SdfModelerApp());
}

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
  static const int _frameWidth = 424;
  static const int _frameHeight = 240;

  String _statusLine = 'Checking Rust bridge...';
  String _versionLine = '';
  String _previewLine = 'Initializing preview texture...';

  int? _textureId;
  Timer? _renderTimer;
  final Stopwatch _elapsed = Stopwatch();
  bool _renderInFlight = false;
  int _renderedFrames = 0;

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

      if (!mounted) {
        return;
      }

      setState(() {
        _statusLine = 'Rust ping: $pingValue';
        _versionLine = 'Bridge crate version: $versionValue';
        _textureId = createdTextureId;
        _previewLine =
            'Preview texture is active (${_frameWidth}x$_frameHeight).';
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
        _previewLine = 'Preview stopped because initialization failed.';
      });
    }
  }

  void _startRenderLoop() {
    _renderTimer?.cancel();
    _renderTimer = Timer.periodic(const Duration(milliseconds: 66), (_) {
      _renderFrame();
    });
  }

  Future<void> _renderFrame() async {
    final activeTextureId = _textureId;
    if (_renderInFlight || activeTextureId == null) {
      return;
    }

    _renderInFlight = true;
    try {
      final elapsedSeconds = _elapsed.elapsedMicroseconds / 1000000.0;
      final pixels = renderPreviewFrame(
        width: _frameWidth,
        height: _frameHeight,
        timeSeconds: elapsedSeconds,
      );

      await TextureBridge.instance.updateTexture(
        textureId: activeTextureId,
        width: _frameWidth,
        height: _frameHeight,
        pixels: pixels,
      );

      _renderedFrames += 1;
      if (mounted && _renderedFrames % 10 == 0) {
        setState(() {
          _previewLine =
              'Preview stream active ($_renderedFrames frames submitted).';
        });
      }
    } catch (error) {
      _renderTimer?.cancel();
      if (mounted) {
        setState(() {
          _previewLine = 'Preview update error: $error';
        });
      }
    } finally {
      _renderInFlight = false;
    }
  }

  void _refreshBridgeStatus() {
    try {
      final pingValue = ping();
      final versionValue = bridgeVersion();
      setState(() {
        _statusLine = 'Rust ping: $pingValue';
        _versionLine = 'Bridge crate version: $versionValue';
      });
    } catch (error) {
      setState(() {
        _statusLine = 'Rust bridge error: $error';
        _versionLine = '';
      });
    }
  }

  @override
  void dispose() {
    _renderTimer?.cancel();
    final activeTextureId = _textureId;
    if (activeTextureId != null) {
      unawaited(TextureBridge.instance.disposeTexture(activeTextureId));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final activeTextureId = _textureId;

    return Scaffold(
      appBar: AppBar(title: const Text('SDF Modeler Flutter Host')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.white24),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Center(
                    child: AspectRatio(
                      aspectRatio: _frameWidth / _frameHeight,
                      child: activeTextureId == null
                          ? const Center(
                              child: Text(
                                'Preparing preview texture...',
                                style: TextStyle(color: Colors.white70),
                              ),
                            )
                          : Texture(textureId: activeTextureId),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Text(_statusLine),
            if (_versionLine.isNotEmpty) Text(_versionLine),
            Text(_previewLine),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              children: [
                FilledButton(
                  onPressed: _refreshBridgeStatus,
                  child: const Text('Re-run Rust Ping'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
