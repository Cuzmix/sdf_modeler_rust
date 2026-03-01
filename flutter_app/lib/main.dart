import 'package:flutter/material.dart';
import 'src/rust/frb_generated.dart';
import 'src/rust/bridge.dart';
import 'src/rust/bridge/gpu_context.dart';
import 'src/rust/bridge/scene_handle.dart';
import 'src/viewport_widget.dart';

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
      title: 'SDF Modeler',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blueGrey,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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
      final gpu = await createGpuContext();
      final scene = await createScene();
      await syncGpu(gpu: gpu, scene: scene);
      setState(() {
        _gpu = gpu;
        _scene = scene;
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
      appBar: AppBar(
        title: const Text('SDF Modeler'),
      ),
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
