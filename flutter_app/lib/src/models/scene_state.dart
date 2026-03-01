import '../rust/bridge/scene_handle.dart';

/// Dart-side wrapper around the opaque Rust [SceneHandle].
///
/// Future phases will add:
/// - selected node tracking
/// - dirty flag for GPU re-sync
/// - undo/redo snapshot stack
class SceneState {
  final SceneHandle handle;

  SceneState({required this.handle});
}
