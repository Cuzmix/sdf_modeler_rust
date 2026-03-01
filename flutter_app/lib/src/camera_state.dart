import 'dart:math';
import 'package:flutter/foundation.dart';

/// Pure-Dart camera state with orbit/pan/zoom.
/// Port of src/gpu/camera.rs — no FFI calls for gesture handling.
class CameraState extends ChangeNotifier {
  static const double _orbitSensitivity = 0.005;
  static const double _panSpeedFactor = 0.002;
  static const double _zoomSensitivity = 0.001;
  static const double _minDistance = 0.1;
  static const double _maxDistance = 100.0;
  static const double _pitchLimitRad = 89.0 * (pi / 180.0);

  double yaw = pi / 4; // 45 degrees
  double pitch = 0.4;
  double distance = 5.0;
  double targetX = 0.0;
  double targetY = 0.0;
  double targetZ = 0.0;
  double fov = 45.0 * (pi / 180.0);

  bool _dirty = true;
  bool get dirty => _dirty;
  void clearDirty() => _dirty = false;

  void orbit(double dx, double dy) {
    yaw += dx * _orbitSensitivity;
    pitch += dy * _orbitSensitivity;
    pitch = pitch.clamp(-_pitchLimitRad, _pitchLimitRad);
    _dirty = true;
    notifyListeners();
  }

  void pan(double dx, double dy) {
    // Eye relative to target (same as Rust Camera::eye() - target)
    final ex = distance * cos(yaw) * cos(pitch);
    final ey = distance * sin(pitch);
    final ez = distance * sin(yaw) * cos(pitch);

    // Forward = normalize(target - eye) = normalize(-eye_offset)
    final fLen = sqrt(ex * ex + ey * ey + ez * ez);
    final fx = -ex / fLen;
    final fy = -ey / fLen;
    final fz = -ez / fLen;

    // Right = normalize(forward x (0,1,0))
    var rx = fz; // fy*0 - fz*0 → simplified cross product
    const ry = 0.0;
    var rz = -fx;
    final rLen = sqrt(rx * rx + rz * rz);
    if (rLen > 1e-6) {
      rx /= rLen;
      rz /= rLen;
    }

    // Up = normalize(right x forward)
    final ux = ry * fz - rz * fy;
    final uy = rz * fx - rx * fz;
    final uz = rx * fy - ry * fx;

    final speed = distance * _panSpeedFactor;
    targetX -= rx * dx * speed;
    targetY -= ry * dx * speed;
    targetZ -= rz * dx * speed;

    targetX += ux * dy * speed;
    targetY += uy * dy * speed;
    targetZ += uz * dy * speed;

    _dirty = true;
    notifyListeners();
  }

  void zoom(double delta) {
    distance *= 1.0 - delta * _zoomSensitivity;
    distance = distance.clamp(_minDistance, _maxDistance);
    _dirty = true;
    notifyListeners();
  }
}
