import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/app.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

class _MockRustApi extends RustLibApi {
  static const String _baseSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _orthoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":true,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';

  String currentSnapshot = _baseSnapshot;
  int focusSelectedCalls = 0;
  int cameraFrontCalls = 0;
  int toggleOrthographicCalls = 0;

  void resetState() {
    currentSnapshot = _baseSnapshot;
    focusSelectedCalls = 0;
    cameraFrontCalls = 0;
    toggleOrthographicCalls = 0;
  }

  @override
  String crateApiSimpleAddSphere() {
    currentSnapshot = _selectedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  String crateApiSimpleCameraBack() => currentSnapshot;

  @override
  String crateApiSimpleCameraBottom() => currentSnapshot;

  @override
  String crateApiSimpleCameraFront() {
    cameraFrontCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCameraLeft() => currentSnapshot;

  @override
  String crateApiSimpleCameraRight() => currentSnapshot;

  @override
  String crateApiSimpleCameraTop() => currentSnapshot;

  @override
  String crateApiSimpleFocusSelected() {
    focusSelectedCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleFrameAll() => currentSnapshot;

  @override
  Future<void> crateApiSimpleInitApp() async {}

  @override
  void crateApiSimpleOrbitCamera({
    required double deltaX,
    required double deltaY,
  }) {}

  @override
  void crateApiSimplePanCamera({
    required double deltaX,
    required double deltaY,
  }) {}

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
  String crateApiSimpleResetScene() {
    currentSnapshot = _baseSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSceneSnapshotJson() => currentSnapshot;

  @override
  String crateApiSimpleSelectNodeAtViewport({
    required double mouseX,
    required double mouseY,
    required int width,
    required int height,
    required double timeSeconds,
  }) {
    return currentSnapshot;
  }

  @override
  String crateApiSimpleToggleOrthographic() {
    toggleOrthographicCalls += 1;
    currentSnapshot = currentSnapshot == _orthoSnapshot
        ? _baseSnapshot
        : _orthoSnapshot;
    return currentSnapshot;
  }

  @override
  void crateApiSimpleZoomCamera({required double delta}) {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const textureChannel = MethodChannel('sdf_modeler/texture');
  const textureEventChannel = EventChannel('sdf_modeler/texture_events');
  final mockApi = _MockRustApi();

  var createTextureCalls = 0;
  var setTextureSizeCalls = 0;
  var requestFrameCalls = 0;
  var orbitCalls = 0;
  var panCalls = 0;
  var zoomCalls = 0;
  var pickCalls = 0;
  var hoverCalls = 0;
  var clearHoverCalls = 0;
  var disposeTextureCalls = 0;
  MockStreamHandlerEventSink? textureEventSink;

  void resetTextureChannelCounters() {
    createTextureCalls = 0;
    setTextureSizeCalls = 0;
    requestFrameCalls = 0;
    orbitCalls = 0;
    panCalls = 0;
    zoomCalls = 0;
    pickCalls = 0;
    hoverCalls = 0;
    clearHoverCalls = 0;
    disposeTextureCalls = 0;
  }

  setUpAll(() {
    RustLib.initMock(api: mockApi);

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, (call) async {
          switch (call.method) {
            case 'createTexture':
              createTextureCalls += 1;
              return 7;
            case 'setTextureSize':
              setTextureSizeCalls += 1;
              return null;
            case 'requestFrame':
              requestFrameCalls += 1;
              return null;
            case 'orbitCamera':
              orbitCalls += 1;
              return null;
            case 'panCamera':
              panCalls += 1;
              return null;
            case 'zoomCamera':
              zoomCalls += 1;
              return null;
            case 'pickNode':
              pickCalls += 1;
              return null;
            case 'hoverNode':
              hoverCalls += 1;
              return null;
            case 'clearHover':
              clearHoverCalls += 1;
              return null;
            case 'disposeTexture':
              disposeTextureCalls += 1;
              return null;
            case 'updateTexture':
              return null;
            default:
              throw MissingPluginException('Unhandled method: ${call.method}');
          }
        });

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockStreamHandler(
          textureEventChannel,
          MockStreamHandler.inline(
            onListen: (arguments, events) {
              textureEventSink = events;
            },
            onCancel: (arguments) {
              textureEventSink = null;
            },
          ),
        );
  });

  setUp(() {
    resetTextureChannelCounters();
    textureEventSink = null;
    mockApi.resetState();
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, null);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockStreamHandler(textureEventChannel, null);
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
    expect(find.text('Adaptive Interaction Resolution'), findsOneWidget);
    expect(find.text('Camera'), findsOneWidget);
    expect(createTextureCalls, 1);
    expect(setTextureSizeCalls, greaterThanOrEqualTo(0));
    expect(requestFrameCalls, greaterThan(0));
    expect(textureEventSink, isNotNull);

    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump();
    expect(disposeTextureCalls, 1);
  });

  testWidgets('shows FPS stats in the viewport overlay', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      const Directionality(
        textDirection: TextDirection.ltr,
        child: ViewportFeedbackOverlay(
          feedback: null,
          interactionPhase: 'interacting',
          frameTimeMs: 16.0,
          framesPerSecond: 62.5,
          droppedFrameCount: 0,
        ),
      ),
    );

    expect(find.textContaining('62.5 FPS', findRichText: true), findsOneWidget);
    expect(find.textContaining('16.0 ms', findRichText: true), findsOneWidget);
  });

  testWidgets('adaptive interaction resolution only drops scale when enabled', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));

    await tester.tap(find.text('Adaptive Interaction Resolution'));
    await tester.pump();

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));
    final orbitGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await orbitGesture.moveBy(const Offset(24, 12));
    await tester.pump();

    expect(find.textContaining('at 65% scale interactive'), findsOneWidget);

    await orbitGesture.up();
  });

  testWidgets('routes viewport gestures to native texture commands', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));
    final hoverGesture = await tester.createGesture(
      kind: PointerDeviceKind.mouse,
    );
    await hoverGesture.addPointer(location: viewportCenter);
    await hoverGesture.moveTo(viewportCenter);
    await tester.pump();
    await hoverGesture.moveTo(viewportCenter + const Offset(16, 8));
    await tester.pump();
    await hoverGesture.moveTo(const Offset(0, 0));
    await tester.pump();

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
    await hoverGesture.removePointer();

    expect(pickCalls, 1);
    expect(hoverCalls, greaterThan(0));
    expect(clearHoverCalls, greaterThan(0));
    expect(orbitCalls, greaterThan(0));
    expect(panCalls, greaterThan(0));
    expect(zoomCalls, 1);
  });

  testWidgets('routes inspector camera commands through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));

    await tester.scrollUntilVisible(
      find.text('Focus Selected'),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();

    await tester.tap(find.text('Focus Selected'));
    await tester.pump();
    await tester.tap(find.text('Front'));
    await tester.pump();
    await tester.tap(find.text('Use Ortho'));
    await tester.pump();

    expect(mockApi.focusSelectedCalls, 1);
    expect(mockApi.cameraFrontCalls, 1);
    expect(mockApi.toggleOrthographicCalls, 1);
    expect(find.text('Use Perspective'), findsOneWidget);
    expect(requestFrameCalls, greaterThan(0));
  });
}