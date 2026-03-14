#ifndef RUNNER_TEXTURE_BRIDGE_H_
#define RUNNER_TEXTURE_BRIDGE_H_

#include <flutter/binary_messenger.h>
#include <flutter/encodable_value.h>
#include <flutter/event_channel.h>
#include <flutter/event_sink.h>
#include <flutter/event_stream_handler_functions.h>
#include <flutter/method_channel.h>
#include <flutter/texture_registrar.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "native_viewport_client.h"

class TextureBridge {
 public:
  TextureBridge(flutter::BinaryMessenger* messenger,
                flutter::TextureRegistrar* texture_registrar);
  ~TextureBridge() = default;

  TextureBridge(const TextureBridge&) = delete;
  TextureBridge& operator=(const TextureBridge&) = delete;

 private:
  struct PendingCommands {
    float orbit_delta_x = 0.0f;
    float orbit_delta_y = 0.0f;
    float pan_delta_x = 0.0f;
    float pan_delta_y = 0.0f;
    float zoom_delta = 0.0f;
    bool pick_requested = false;
    float pick_normalized_x = 0.0f;
    float pick_normalized_y = 0.0f;
    bool hover_requested = false;
    float hover_normalized_x = 0.0f;
    float hover_normalized_y = 0.0f;
    bool clear_hover_requested = false;

    bool HasWork() const;
    NativeViewportCommands ToNativeCommands() const;
    void Reset();
  };

  class ManagedTexture {
   public:
    ManagedTexture(flutter::TextureRegistrar* texture_registrar,
                   std::shared_ptr<NativeViewportClient> native_viewport_client,
                   std::function<void(flutter::EncodableMap)> viewport_event_publisher,
                   int width,
                   int height);
    ~ManagedTexture();

    std::optional<int64_t> texture_id() const;
    bool UpdateFrame(int width, int height, const std::vector<uint8_t>& pixels);
    void Resize(int width, int height);
    void RequestFrame();
    void QueueOrbit(float delta_x, float delta_y);
    void QueuePan(float delta_x, float delta_y);
    void QueueZoom(float delta);
    void QueuePick(float normalized_x, float normalized_y);
    void QueueHover(float normalized_x, float normalized_y);
    void ClearHover();

   private:
    void TrackCoalescedFrameRequest();
    void PublishFrameEvent(int width,
                           int height,
                           double frame_time_ms,
                           const std::string& interaction_phase,
                           bool scene_state_changed,
                           const std::string& feedback_json,
                           const std::string& host_error);
    void RenderLoop();

    flutter::TextureRegistrar* texture_registrar_;
    std::shared_ptr<NativeViewportClient> native_viewport_client_;
    std::function<void(flutter::EncodableMap)> viewport_event_publisher_;
    std::unique_ptr<flutter::TextureVariant> texture_;
    std::optional<int64_t> texture_id_;

    int width_;
    int height_;
    int target_width_;
    int target_height_;

    std::vector<uint8_t> front_pixels_;
    std::vector<uint8_t> back_pixels_;
    bool has_pending_frame_ = false;

    FlutterDesktopPixelBuffer pixel_buffer_{};
    std::mutex pixel_mutex_;

    std::mutex render_state_mutex_;
    std::condition_variable render_state_cv_;
    PendingCommands pending_commands_;
    bool force_frame_ = true;
    bool stop_requested_ = false;
    bool interaction_active_ = false;
    bool scene_snapshot_refresh_pending_ = false;
    bool render_in_progress_ = false;
    int64_t rendered_frame_count_ = 0;
    int64_t dropped_frame_count_ = 0;
    std::chrono::steady_clock::time_point interaction_deadline_{};
    std::chrono::steady_clock::time_point render_start_time_{};
    std::thread render_thread_;
  };

  using MethodCall = flutter::MethodCall<flutter::EncodableValue>;
  using MethodResult = flutter::MethodResult<flutter::EncodableValue>;

  void HandleMethodCall(const MethodCall& call,
                        std::unique_ptr<MethodResult> result);
  void PublishViewportEvent(flutter::EncodableMap event);

  std::optional<int64_t> CreateTexture(int width, int height);
  bool UpdateTexture(int64_t texture_id,
                     int width,
                     int height,
                     const std::vector<uint8_t>& pixels);
  bool SetTextureSize(int64_t texture_id, int width, int height);
  bool RequestFrame(int64_t texture_id);
  bool OrbitCamera(int64_t texture_id, float delta_x, float delta_y);
  bool PanCamera(int64_t texture_id, float delta_x, float delta_y);
  bool ZoomCamera(int64_t texture_id, float delta);
  bool PickNode(int64_t texture_id, float normalized_x, float normalized_y);
  bool HoverNode(int64_t texture_id, float normalized_x, float normalized_y);
  bool ClearHover(int64_t texture_id);
  void DisposeTexture(int64_t texture_id);

  static const flutter::EncodableMap* ReadMap(
      const flutter::EncodableValue* value);
  static const flutter::EncodableValue* FindMapValue(
      const flutter::EncodableMap& map,
      const char* key);
  static std::optional<int64_t> ReadInt64(const flutter::EncodableValue* value);
  static std::optional<double> ReadDouble(const flutter::EncodableValue* value);
  static std::optional<std::vector<uint8_t>> ReadBytes(
      const flutter::EncodableValue* value);

  flutter::TextureRegistrar* texture_registrar_;
  std::shared_ptr<NativeViewportClient> native_viewport_client_;
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>>
      method_channel_;
  std::unique_ptr<flutter::EventChannel<flutter::EncodableValue>> event_channel_;
  std::mutex event_sink_mutex_;
  std::unique_ptr<flutter::EventSink<flutter::EncodableValue>> event_sink_;
  std::unordered_map<int64_t, std::unique_ptr<ManagedTexture>> textures_;
};

#endif  // RUNNER_TEXTURE_BRIDGE_H_
