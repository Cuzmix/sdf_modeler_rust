import 'package:flutter/material.dart';
import '../core/gpu_service.dart';
import '../rust/bridge/gpu_context.dart';
import '../rust/bridge/scene_handle.dart';
import '../widgets/viewport/viewport_widget.dart';

/// Main editor screen — initialises GPU and shows the SDF viewport.
/// Future phases will add ScenePanel, PropertiesPanel, and Toolbar around it.
class EditorScreen extends StatefulWidget {
  const EditorScreen({super.key});

  @override
  State<EditorScreen> createState() => _EditorScreenState();
}

class _EditorScreenState extends State<EditorScreen> {
  GpuContext? _gpu;
  SceneHandle? _scene;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initGpu();
  }

  Future<void> _initGpu() async {
    try {
      final result = await initGpu();
      setState(() {
        _gpu = result.gpu;
        _scene = result.scene;
      });
    } catch (e) {
      setState(() {
        _error = 'GPU init failed: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SDF Modeler')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error != null) {
      return Center(
        child: Text(
          _error!,
          style: const TextStyle(color: Colors.red, fontSize: 16),
          textAlign: TextAlign.center,
        ),
      );
    }
    if (_gpu == null || _scene == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return ViewportWidget(gpu: _gpu!, scene: _scene!);
  }
}
