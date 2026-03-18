import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:sdf_modeler_flutter/app.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  ExternalLibrary? externalLibrary;
  if (!kIsWeb && Platform.isWindows) {
    final libraryPath =
        kReleaseMode
            ? 'rust/target/release/sdf_modeler_bridge.dll'
            : 'rust/target/debug/sdf_modeler_bridge.dll';
    externalLibrary = ExternalLibrary.open(libraryPath);
  }

  await RustLib.init(externalLibrary: externalLibrary);
  runApp(const SdfModelerApp());
}
