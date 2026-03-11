import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/main.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';

class _MockRustApi extends RustLibApi {
  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  Future<void> crateApiSimpleInitApp() async {}

  @override
  String crateApiSimplePing() => 'pong-test';

  @override
  Uint8List crateApiSimpleRenderPreviewFrame({
    required int width,
    required int height,
    required double timeSeconds,
  }) {
    return Uint8List(width * height * 4);
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const textureChannel = MethodChannel('sdf_modeler/texture');

  setUpAll(() {
    RustLib.initMock(api: _MockRustApi());

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, (call) async {
          switch (call.method) {
            case 'createTexture':
              return 7;
            case 'updateTexture':
              return null;
            case 'disposeTexture':
              return null;
            default:
              throw MissingPluginException('Unhandled method: ${call.method}');
          }
        });
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, null);
  });

  testWidgets('renders bridge status and preview stream', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 120));

    expect(find.textContaining('Rust ping: pong-test'), findsOneWidget);
    expect(
      find.textContaining('Bridge crate version: 0.1.0-test'),
      findsOneWidget,
    );
    expect(find.textContaining('Preview'), findsWidgets);
  });
}

