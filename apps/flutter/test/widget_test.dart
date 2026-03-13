import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/app.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

class _MockRustApi extends RustLibApi {
  static const String _baseSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';

  int orbitCalls = 0;
  int panCalls = 0;
  int zoomCalls = 0;
  int selectCalls = 0;

  void resetCounters() {
    orbitCalls = 0;
    panCalls = 0;
    zoomCalls = 0;
    selectCalls = 0;
  }

  @override
  String crateApiSimpleAddSphere() => _baseSnapshot;

  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  String crateApiSimpleFrameAll() => _baseSnapshot;

  @override
  Future<void> crateApiSimpleInitApp() async {}

  @override
  void crateApiSimpleOrbitCamera({
    required double deltaX,
    required double deltaY,
  }) {
    orbitCalls += 1;
  }

  @override
  void crateApiSimplePanCamera({
    required double deltaX,
    required double deltaY,
  }) {
    panCalls += 1;
  }

  @override
  String crateApiSimplePing() => 'pong-test';

  @override
  Future<Uint8List> crateApiSimpleRenderPreviewFrame({
    required int width,
    required int height,
    required double timeSeconds,
  }) async {
    return Uint8List(width * height * 4);
  }

  @override
  String crateApiSimpleResetScene() => _baseSnapshot;

  @override
  String crateApiSimpleSceneSnapshotJson() => _baseSnapshot;

  @override
  String crateApiSimpleSelectNodeAtViewport({
    required double mouseX,
    required double mouseY,
    required int width,
    required int height,
    required double timeSeconds,
  }) {
    selectCalls += 1;
    return _baseSnapshot;
  }

  @override
  void crateApiSimpleZoomCamera({required double delta}) {
    zoomCalls += 1;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const textureChannel = MethodChannel('sdf_modeler/texture');
  final mockApi = _MockRustApi();

  setUpAll(() {
    RustLib.initMock(api: mockApi);

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

  setUp(() {
    mockApi.resetCounters();
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, null);
  });

  testWidgets('renders bridge status, scene snapshot, and real viewport host', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));

    expect(find.textContaining('Rust ping: pong-test'), findsOneWidget);
    expect(
      find.textContaining('Bridge crate version: 0.1.0-test'),
      findsOneWidget,
    );
    expect(find.textContaining('Viewport Status'), findsOneWidget);
    expect(find.textContaining('Scene nodes: 7 total'), findsOneWidget);
  });

  testWidgets('routes viewport gestures to Rust commands', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));

    await tester.tapAt(viewportCenter);
    await tester.pump();

    final orbitGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await orbitGesture.moveBy(const Offset(24, 12));
    await orbitGesture.up();
    await tester.pump();

    final panGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kSecondaryMouseButton,
    );
    await panGesture.moveBy(const Offset(18, 10));
    await panGesture.up();
    await tester.pump();

    await tester.sendEventToBinding(
      PointerScrollEvent(
        position: viewportCenter,
        scrollDelta: const Offset(0, -24),
      ),
    );
    await tester.pump();

    expect(mockApi.selectCalls, 1);
    expect(mockApi.orbitCalls, greaterThan(0));
    expect(mockApi.panCalls, greaterThan(0));
    expect(mockApi.zoomCalls, 1);
  });
}