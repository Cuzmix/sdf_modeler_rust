#ifndef RUNNER_NATIVE_VIEWPORT_CLIENT_H_
#define RUNNER_NATIVE_VIEWPORT_CLIENT_H_

#include <windows.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct NativeViewportCommands {
  float orbit_delta_x = 0.0f;
  float orbit_delta_y = 0.0f;
  float pan_delta_x = 0.0f;
  float pan_delta_y = 0.0f;
  float zoom_delta = 0.0f;
  float pick_normalized_x = 0.0f;
  float pick_normalized_y = 0.0f;
  float hover_normalized_x = 0.0f;
  float hover_normalized_y = 0.0f;
  uint32_t flags = 0;
};

struct NativeViewportVec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct NativeViewportCameraSnapshot {
  float yaw = 0.0f;
  float pitch = 0.0f;
  float roll = 0.0f;
  float distance = 0.0f;
  float fov_degrees = 0.0f;
  uint8_t orthographic = 0;
  NativeViewportVec3 target;
  NativeViewportVec3 eye;
};

struct NativeViewportString {
  uint8_t* ptr = nullptr;
  size_t len = 0;
  size_t capacity = 0;
};

struct NativeViewportNodeSnapshot {
  uint64_t id = 0;
  uint8_t visible = 0;
  uint8_t locked = 0;
  NativeViewportString name;
  NativeViewportString kind_label;
};

struct NativeViewportNode {
  uint64_t id = 0;
  bool visible = false;
  bool locked = false;
  std::string name;
  std::string kind_label;
};

struct NativeViewportFeedback {
  NativeViewportCameraSnapshot camera;
  std::optional<NativeViewportNode> selected_node;
  std::optional<NativeViewportNode> hovered_node;
};

struct NativeViewportRenderResult {
  std::vector<uint8_t> pixels;
  NativeViewportFeedback feedback;
  bool camera_animating = false;
};

class NativeViewportClient {
 public:
  static constexpr uint32_t kPickCommandFlag = 1u;
  static constexpr uint32_t kHoverCommandFlag = 1u << 1;
  static constexpr uint32_t kClearHoverCommandFlag = 1u << 2;

  static std::shared_ptr<NativeViewportClient> TryCreate();

  ~NativeViewportClient();

  NativeViewportClient(const NativeViewportClient&) = delete;
  NativeViewportClient& operator=(const NativeViewportClient&) = delete;

  std::optional<NativeViewportRenderResult> RenderFrame(
      int width,
      int height,
      float time_seconds,
      const NativeViewportCommands& commands) const;

 private:
  struct NativeViewportFrame {
    uint8_t* pixels_ptr;
    size_t pixels_len;
    size_t pixels_capacity;
    NativeViewportCameraSnapshot camera;
    uint8_t selected_node_present;
    NativeViewportNodeSnapshot selected_node;
    uint8_t hovered_node_present;
    NativeViewportNodeSnapshot hovered_node;
    uint8_t camera_animating;
  };

  using ProcessViewportFrameFn = NativeViewportFrame(__cdecl*)(
      uint32_t width,
      uint32_t height,
      float time_seconds,
      NativeViewportCommands commands);
  using FreeViewportFrameFn = void(__cdecl*)(NativeViewportFrame frame);

  NativeViewportClient(HMODULE module,
                       ProcessViewportFrameFn process_frame,
                       FreeViewportFrameFn free_frame);

  HMODULE module_;
  ProcessViewportFrameFn process_frame_;
  FreeViewportFrameFn free_frame_;
};

#endif  // RUNNER_NATIVE_VIEWPORT_CLIENT_H_
