import 'dart:math';

/// Shared constants extracted from camera and viewport code.
/// Centralised here so every widget uses the same tuning values.

// -- Camera sensitivity --
const double orbitSensitivity = 0.005;
const double panSpeedFactor = 0.002;
const double zoomSensitivity = 0.001;

// -- Camera limits --
const double minDistance = 0.1;
const double maxDistance = 100.0;
const double pitchLimitRad = 89.0 * (pi / 180.0);

// -- Camera defaults --
const double defaultYaw = pi / 4; // 45 degrees
const double defaultPitch = 0.4;
const double defaultDistance = 5.0;
const double defaultFov = 45.0 * (pi / 180.0);

// -- Viewport rendering --
const int maxTextureSize = 4096;
const double interactionScale = 0.5;
const double fullScale = 1.0;
const double interactionQuality = 1.0; // fast: skip AO/shadows
const double fullQuality = 0.0; // full quality
const int interactionDebounceMs = 100;
