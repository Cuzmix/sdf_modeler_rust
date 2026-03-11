import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';

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
  String _statusLine = 'Checking Rust bridge...';
  String _versionLine = '';

  @override
  void initState() {
    super.initState();
    _refreshBridgeStatus();
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
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SDF Modeler Flutter Host')),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(_statusLine, textAlign: TextAlign.center),
            if (_versionLine.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(_versionLine, textAlign: TextAlign.center),
              ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _refreshBridgeStatus,
              child: const Text('Re-run Rust Ping'),
            ),
          ],
        ),
      ),
    );
  }
}
