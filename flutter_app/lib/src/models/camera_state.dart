import 'dart:math';
import 'package:flutter/foundation.dart';
import '../core/constants.dart';

/// Pure-Dart camera state with orbit/pan/zoom.
/// Port of src/gpu/camera.rs — no FFI calls for gesture handling.
class CameraState extends ChangeNotifier {
  double yaw = defaultYaw;
  double pitch = defaultPitch;
  double distance = defaultDistance;
  double targetX = 0.0;
  double targetY = 0.0;
  double targetZ = 0.0;
  double fov = defaultFov;

  bool _dirty = true;
  bool get dirty => _dirty;
  void clearDirty() => _dirty = false;

  void orbit(double dx, double dy) {
    yaw += dx * orbitSensitivity;
    pitch += dy * orbitSensitivity;
    pitch = pitch.clamp(-pitchLimitRad, pitchLimitRad);
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
    var rx = fz;
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

    final speed = distance * panSpeedFactor;
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
    distance *= 1.0 - delta * zoomSensitivity;
    distance = distance.clamp(minDistance, maxDistance);
    _dirty = true;
    notifyListeners();
  }
}
