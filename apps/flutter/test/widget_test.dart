import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/main.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';

class _MockRustApi extends RustLibApi {
  @override
  String crateApiSimplePing() => 'pong-test';

  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  Future<void> crateApiSimpleInitApp() async {}
}

void main() {
  setUpAll(() {
    RustLib.initMock(api: _MockRustApi());
  });

  testWidgets('renders Rust bridge smoke values', (WidgetTester tester) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();

    expect(find.textContaining('Rust ping: pong-test'), findsOneWidget);
    expect(find.textContaining('Bridge crate version: 0.1.0-test'), findsOneWidget);
  });
}
