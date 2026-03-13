#include "texture_bridge.h"

#include <flutter/standard_method_codec.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace {
constexpr const char* kTextureChannelName = "sdf_modeler/texture";
constexpr const char* kTextureEventChannelName = "sdf_modeler/texture_events";
constexpr int kMinDimension = 1;
constexpr int kMaxDimension = 4096;
constexpr auto kInteractionSharpnessDelay = std::chrono::milliseconds(180);

int ClampDimension(int value) {
  return std::clamp(value, kMinDimension, kMaxDimension);
}

size_t ByteCountForDimensions(int width, int height) {
  return static_cast<size_t>(width) * static_cast<size_t>(height) * 4U;
}

void InsertMapValue(flutter::EncodableMap& map,
                    const char* key,
                    flutter::EncodableValue value) {
  map[flutter::EncodableValue(std::string(key))] = std::move(value);
}
}  // namespace

bool TextureBridge::PendingCommands::HasWork() const {
  return orbit_delta_x != 0.0f || orbit_delta_y != 0.0f || pan_delta_x != 0.0f ||
         pan_delta_y != 0.0f || zoom_delta != 0.0f || pick_requested ||
         hover_requested || clear_hover_requested;
}

NativeViewportCommands TextureBridge::PendingCommands::ToNativeCommands() const {
  NativeViewportCommands commands;
  commands.orbit_delta_x = orbit_delta_x;
  commands.orbit_delta_y = orbit_delta_y;
  commands.pan_delta_x = pan_delta_x;
  commands.pan_delta_y = pan_delta_y;
  commands.zoom_delta = zoom_delta;
  commands.pick_normalized_x = pick_normalized_x;
  commands.pick_normalized_y = pick_normalized_y;
  commands.hover_normalized_x = hover_normalized_x;
  commands.hover_normalized_y = hover_normalized_y;
  if (pick_requested) {
    commands.flags |= NativeViewportClient::kPickCommandFlag;
  }
  if (hover_requested) {
    commands.flags |= NativeViewportClient::kHoverCommandFlag;
  }
  if (clear_hover_requested) {
    commands.flags |= NativeViewportClient::kClearHoverCommandFlag;
  }
  return commands;
}

void TextureBridge::PendingCommands::Reset() {
  orbit_delta_x = 0.0f;
  orbit_delta_y = 0.0f;
  pan_delta_x = 0.0f;
  pan_delta_y = 0.0f;
  zoom_delta = 0.0f;
  pick_requested = false;
  pick_normalized_x = 0.0f;
  pick_normalized_y = 0.0f;
  hover_requested = false;
  hover_normalized_x = 0.0f;
  hover_normalized_y = 0.0f;
  clear_hover_requested = false;
}

TextureBridge::ManagedTexture::ManagedTexture(
    flutter::TextureRegistrar* texture_registrar,
    std::shared_ptr<NativeViewportClient> native_viewport_client,
    std::function<void(flutter::EncodableMap)> viewport_event_publisher,
    int width,
    int height)
    : texture_registrar_(texture_registrar),
      native_viewport_client_(std::move(native_viewport_client)),
      viewport_event_publisher_(std::move(viewport_event_publisher)),
      width_(ClampDimension(width)),
      height_(ClampDimension(height)),
      target_width_(ClampDimension(width)),
      target_height_(ClampDimension(height)) {
  const size_t initial_byte_count = ByteCountForDimensions(width_, height_);
  front_pixels_.assign(initial_byte_count, uint8_t{0});
  back_pixels_.assign(initial_byte_count, uint8_t{0});

  texture_ = std::make_unique<flutter::TextureVariant>(flutter::PixelBufferTexture(
      [this](size_t, size_t) -> const FlutterDesktopPixelBuffer* {
        std::lock_guard<std::mutex> lock(pixel_mutex_);

        if (has_pending_frame_) {
          front_pixels_.swap(back_pixels_);
          has_pending_frame_ = false;
        }

        pixel_buffer_.buffer = front_pixels_.data();
        pixel_buffer_.width = static_cast<size_t>(width_);
        pixel_buffer_.height = static_cast<size_t>(height_);
        return &pixel_buffer_;
      }));

  const int64_t registered_texture_id =
      texture_registrar_->RegisterTexture(texture_.get());
  if (registered_texture_id >= 0) {
    texture_id_ = registered_texture_id;
  }

  if (texture_id_.has_value() && native_viewport_client_ != nullptr) {
    render_start_time_ = std::chrono::steady_clock::now();
    render_thread_ = std::thread([this]() { RenderLoop(); });
  }
}

TextureBridge::ManagedTexture::~ManagedTexture() {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    stop_requested_ = true;
  }
  render_state_cv_.notify_all();

  if (render_thread_.joinable()) {
    render_thread_.join();
  }

  if (texture_id_.has_value()) {
    texture_registrar_->UnregisterTexture(*texture_id_);
  }
}

std::optional<int64_t> TextureBridge::ManagedTexture::texture_id() const {
  return texture_id_;
}

bool TextureBridge::ManagedTexture::UpdateFrame(
    int width,
    int height,
    const std::vector<uint8_t>& pixels) {
  const int clamped_width = ClampDimension(width);
  const int clamped_height = ClampDimension(height);
  const size_t expected_size =
      ByteCountForDimensions(clamped_width, clamped_height);

  if (pixels.size() != expected_size) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(pixel_mutex_);

    if (width_ != clamped_width || height_ != clamped_height) {
      width_ = clamped_width;
      height_ = clamped_height;
      const size_t resized_size = ByteCountForDimensions(width_, height_);
      front_pixels_.assign(resized_size, uint8_t{0});
      back_pixels_.assign(resized_size, uint8_t{0});
    }

    std::copy(pixels.begin(), pixels.end(), back_pixels_.begin());
    has_pending_frame_ = true;
  }

  if (texture_id_.has_value()) {
    texture_registrar_->MarkTextureFrameAvailable(*texture_id_);
  }

  return true;
}

void TextureBridge::ManagedTexture::TrackCoalescedFrameRequest() {
  if (render_in_progress_ || force_frame_ || pending_commands_.HasWork()) {
    dropped_frame_count_ += 1;
  }
}

void TextureBridge::ManagedTexture::Resize(int width, int height) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    target_width_ = ClampDimension(width);
    target_height_ = ClampDimension(height);
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::RequestFrame() {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::QueueOrbit(float delta_x, float delta_y) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.orbit_delta_x += delta_x;
    pending_commands_.orbit_delta_y += delta_y;
    interaction_active_ = true;
    viewport_state_changed_pending_ = true;
    interaction_deadline_ = std::chrono::steady_clock::now() + kInteractionSharpnessDelay;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::QueuePan(float delta_x, float delta_y) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.pan_delta_x += delta_x;
    pending_commands_.pan_delta_y += delta_y;
    interaction_active_ = true;
    viewport_state_changed_pending_ = true;
    interaction_deadline_ = std::chrono::steady_clock::now() + kInteractionSharpnessDelay;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::QueueZoom(float delta) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.zoom_delta += delta;
    interaction_active_ = true;
    viewport_state_changed_pending_ = true;
    interaction_deadline_ = std::chrono::steady_clock::now() + kInteractionSharpnessDelay;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::QueuePick(float normalized_x, float normalized_y) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.pick_requested = true;
    pending_commands_.pick_normalized_x = std::clamp(normalized_x, 0.0f, 1.0f);
    pending_commands_.pick_normalized_y = std::clamp(normalized_y, 0.0f, 1.0f);
    pending_commands_.clear_hover_requested = false;
    viewport_state_changed_pending_ = true;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::QueueHover(float normalized_x, float normalized_y) {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.hover_requested = true;
    pending_commands_.hover_normalized_x = std::clamp(normalized_x, 0.0f, 1.0f);
    pending_commands_.hover_normalized_y = std::clamp(normalized_y, 0.0f, 1.0f);
    pending_commands_.clear_hover_requested = false;
    viewport_state_changed_pending_ = true;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::ClearHover() {
  {
    std::lock_guard<std::mutex> lock(render_state_mutex_);
    TrackCoalescedFrameRequest();
    pending_commands_.clear_hover_requested = true;
    pending_commands_.hover_requested = false;
    viewport_state_changed_pending_ = true;
    force_frame_ = true;
  }
  render_state_cv_.notify_all();
}

void TextureBridge::ManagedTexture::PublishFrameEvent(int width,
                                                      int height,
                                                      double frame_time_ms,
                                                      const std::string& interaction_phase,
                                                      bool viewport_state_changed,
                                                      const std::string& feedback_json) {
  if (!texture_id_.has_value()) {
    return;
  }

  flutter::EncodableMap event;
  InsertMapValue(event, "textureId", flutter::EncodableValue(*texture_id_));
  InsertMapValue(event, "frameWidth", flutter::EncodableValue(width));
  InsertMapValue(event, "frameHeight", flutter::EncodableValue(height));
  InsertMapValue(event, "frameTimeMs", flutter::EncodableValue(frame_time_ms));
  InsertMapValue(event, "frameCount", flutter::EncodableValue(rendered_frame_count_));
  InsertMapValue(event, "droppedFrameCount", flutter::EncodableValue(dropped_frame_count_));
  InsertMapValue(event, "interactionPhase",
                 flutter::EncodableValue(interaction_phase));
  InsertMapValue(event, "sceneStateChanged",
                 flutter::EncodableValue(viewport_state_changed));
  InsertMapValue(event, "feedbackJson", flutter::EncodableValue(feedback_json));
  viewport_event_publisher_(std::move(event));
}

void TextureBridge::ManagedTexture::RenderLoop() {
  while (true) {
    NativeViewportCommands commands;
    int render_width = 1;
    int render_height = 1;
    bool viewport_state_changed = false;
    std::string interaction_phase = "idle";

    {
      std::unique_lock<std::mutex> lock(render_state_mutex_);
      bool ended_interaction = false;
      while (true) {
        if (stop_requested_) {
          return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (force_frame_ || pending_commands_.HasWork()) {
          break;
        }

        if (interaction_active_) {
          if (now >= interaction_deadline_) {
            interaction_active_ = false;
            force_frame_ = true;
            ended_interaction = true;
            break;
          }

          render_state_cv_.wait_until(lock, interaction_deadline_);
          continue;
        }

        render_state_cv_.wait(lock);
      }

      commands = pending_commands_.ToNativeCommands();
      pending_commands_.Reset();
      render_width = target_width_;
      render_height = target_height_;
      viewport_state_changed = viewport_state_changed_pending_ || ended_interaction;
      viewport_state_changed_pending_ = false;
      render_in_progress_ = true;
      if (interaction_active_) {
        interaction_phase = "interacting";
      } else if (ended_interaction) {
        interaction_phase = "settling";
      }
      force_frame_ = false;
    }

    const auto render_elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - render_start_time_);
    const auto frame_start_time = std::chrono::steady_clock::now();
    const auto render_result = native_viewport_client_->RenderFrame(
        render_width, render_height, render_elapsed.count(), commands);
    const auto frame_end_time = std::chrono::steady_clock::now();

    bool camera_animating = false;
    {
      std::lock_guard<std::mutex> lock(render_state_mutex_);
      render_in_progress_ = false;
      if (render_result.has_value() && render_result->camera_animating) {
        force_frame_ = true;
        camera_animating = true;
      }
    }

    if (camera_animating) {
      render_state_cv_.notify_all();
    }

    if (!render_result.has_value()) {
      continue;
    }

    if (camera_animating && interaction_phase == "idle") {
      interaction_phase = "animating";
    }

    const bool publish_viewport_state_changed = viewport_state_changed || camera_animating;
    UpdateFrame(render_width, render_height, render_result->pixels);
    rendered_frame_count_ += 1;
    const double frame_time_ms =
        std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();
    PublishFrameEvent(render_width,
                      render_height,
                      frame_time_ms,
                      interaction_phase,
                      publish_viewport_state_changed,
                      render_result->feedback_json);
  }
}

TextureBridge::TextureBridge(flutter::BinaryMessenger* messenger,
                             flutter::TextureRegistrar* texture_registrar)
    : texture_registrar_(texture_registrar),
      native_viewport_client_(NativeViewportClient::TryCreate()) {
  method_channel_ = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
      messenger, kTextureChannelName, &flutter::StandardMethodCodec::GetInstance());
  event_channel_ = std::make_unique<flutter::EventChannel<flutter::EncodableValue>>(
      messenger, kTextureEventChannelName, &flutter::StandardMethodCodec::GetInstance());

  event_channel_->SetStreamHandler(
      std::make_unique<flutter::StreamHandlerFunctions<flutter::EncodableValue>>(
          [this](const flutter::EncodableValue*,
                 std::unique_ptr<flutter::EventSink<flutter::EncodableValue>>&& events)
              -> std::unique_ptr<flutter::StreamHandlerError<flutter::EncodableValue>> {
            std::lock_guard<std::mutex> lock(event_sink_mutex_);
            event_sink_ = std::move(events);
            return nullptr;
          },
          [this](const flutter::EncodableValue*)
              -> std::unique_ptr<flutter::StreamHandlerError<flutter::EncodableValue>> {
            std::lock_guard<std::mutex> lock(event_sink_mutex_);
            event_sink_.reset();
            return nullptr;
          }));

  method_channel_->SetMethodCallHandler(
      [this](const MethodCall& call, std::unique_ptr<MethodResult> result) {
        HandleMethodCall(call, std::move(result));
      });
}

void TextureBridge::HandleMethodCall(const MethodCall& call,
                                     std::unique_ptr<MethodResult> result) {
  const auto method_name = call.method_name();

  if (method_name == "createTexture") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "createTexture expects a map argument.");
      return;
    }

    const auto width = ReadInt64(FindMapValue(*arguments, "width"));
    const auto height = ReadInt64(FindMapValue(*arguments, "height"));

    if (!width.has_value() || !height.has_value()) {
      result->Error("invalid_args", "createTexture requires width and height.");
      return;
    }

    const auto texture_id =
        CreateTexture(static_cast<int>(*width), static_cast<int>(*height));
    if (!texture_id.has_value()) {
      result->Error("create_failed", "Failed to create native texture.");
      return;
    }

    result->Success(flutter::EncodableValue(*texture_id));
    return;
  }

  if (method_name == "updateTexture") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "updateTexture expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto width = ReadInt64(FindMapValue(*arguments, "width"));
    const auto height = ReadInt64(FindMapValue(*arguments, "height"));
    const auto pixels = ReadBytes(FindMapValue(*arguments, "pixels"));

    if (!texture_id.has_value() || !width.has_value() || !height.has_value() ||
        !pixels.has_value()) {
      result->Error(
          "invalid_args",
          "updateTexture requires textureId, width, height, and pixels.");
      return;
    }

    const bool updated = UpdateTexture(
        *texture_id, static_cast<int>(*width), static_cast<int>(*height), *pixels);
    if (!updated) {
      result->Error("update_failed", "Failed to update texture pixels.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "setTextureSize") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "setTextureSize expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto width = ReadInt64(FindMapValue(*arguments, "width"));
    const auto height = ReadInt64(FindMapValue(*arguments, "height"));
    if (!texture_id.has_value() || !width.has_value() || !height.has_value()) {
      result->Error("invalid_args", "setTextureSize requires textureId, width, and height.");
      return;
    }

    if (!SetTextureSize(*texture_id, static_cast<int>(*width), static_cast<int>(*height))) {
      result->Error("resize_failed", "Failed to update native texture size.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "requestFrame") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "requestFrame expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    if (!texture_id.has_value()) {
      result->Error("invalid_args", "requestFrame requires textureId.");
      return;
    }

    if (!RequestFrame(*texture_id)) {
      result->Error("request_failed", "Failed to request a native viewport frame.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "orbitCamera") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "orbitCamera expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto delta_x = ReadDouble(FindMapValue(*arguments, "deltaX"));
    const auto delta_y = ReadDouble(FindMapValue(*arguments, "deltaY"));
    if (!texture_id.has_value() || !delta_x.has_value() || !delta_y.has_value()) {
      result->Error("invalid_args", "orbitCamera requires textureId, deltaX, and deltaY.");
      return;
    }

    if (!OrbitCamera(*texture_id, static_cast<float>(*delta_x), static_cast<float>(*delta_y))) {
      result->Error("orbit_failed", "Failed to queue native orbit command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "panCamera") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "panCamera expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto delta_x = ReadDouble(FindMapValue(*arguments, "deltaX"));
    const auto delta_y = ReadDouble(FindMapValue(*arguments, "deltaY"));
    if (!texture_id.has_value() || !delta_x.has_value() || !delta_y.has_value()) {
      result->Error("invalid_args", "panCamera requires textureId, deltaX, and deltaY.");
      return;
    }

    if (!PanCamera(*texture_id, static_cast<float>(*delta_x), static_cast<float>(*delta_y))) {
      result->Error("pan_failed", "Failed to queue native pan command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "zoomCamera") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "zoomCamera expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto delta = ReadDouble(FindMapValue(*arguments, "delta"));
    if (!texture_id.has_value() || !delta.has_value()) {
      result->Error("invalid_args", "zoomCamera requires textureId and delta.");
      return;
    }

    if (!ZoomCamera(*texture_id, static_cast<float>(*delta))) {
      result->Error("zoom_failed", "Failed to queue native zoom command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "pickNode") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "pickNode expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto normalized_x = ReadDouble(FindMapValue(*arguments, "normalizedX"));
    const auto normalized_y = ReadDouble(FindMapValue(*arguments, "normalizedY"));
    if (!texture_id.has_value() || !normalized_x.has_value() || !normalized_y.has_value()) {
      result->Error("invalid_args", "pickNode requires textureId, normalizedX, and normalizedY.");
      return;
    }

    if (!PickNode(*texture_id,
                  static_cast<float>(*normalized_x),
                  static_cast<float>(*normalized_y))) {
      result->Error("pick_failed", "Failed to queue native pick command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "hoverNode") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "hoverNode expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    const auto normalized_x = ReadDouble(FindMapValue(*arguments, "normalizedX"));
    const auto normalized_y = ReadDouble(FindMapValue(*arguments, "normalizedY"));
    if (!texture_id.has_value() || !normalized_x.has_value() || !normalized_y.has_value()) {
      result->Error("invalid_args", "hoverNode requires textureId, normalizedX, and normalizedY.");
      return;
    }

    if (!HoverNode(*texture_id,
                   static_cast<float>(*normalized_x),
                   static_cast<float>(*normalized_y))) {
      result->Error("hover_failed", "Failed to queue native hover command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "clearHover") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "clearHover expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    if (!texture_id.has_value()) {
      result->Error("invalid_args", "clearHover requires textureId.");
      return;
    }

    if (!ClearHover(*texture_id)) {
      result->Error("hover_failed", "Failed to queue native clear-hover command.");
      return;
    }

    result->Success();
    return;
  }

  if (method_name == "disposeTexture") {
    const auto* arguments = ReadMap(call.arguments());
    if (arguments == nullptr) {
      result->Error("invalid_args", "disposeTexture expects a map argument.");
      return;
    }

    const auto texture_id = ReadInt64(FindMapValue(*arguments, "textureId"));
    if (!texture_id.has_value()) {
      result->Error("invalid_args", "disposeTexture requires textureId.");
      return;
    }

    DisposeTexture(*texture_id);
    result->Success();
    return;
  }

  result->NotImplemented();
}

void TextureBridge::PublishViewportEvent(flutter::EncodableMap event) {
  std::lock_guard<std::mutex> lock(event_sink_mutex_);
  if (event_sink_ == nullptr) {
    return;
  }

  event_sink_->Success(flutter::EncodableValue(event));
}

std::optional<int64_t> TextureBridge::CreateTexture(int width, int height) {
  if (native_viewport_client_ == nullptr) {
    return std::nullopt;
  }

  auto managed_texture = std::make_unique<ManagedTexture>(
      texture_registrar_,
      native_viewport_client_,
      [this](flutter::EncodableMap event) { PublishViewportEvent(std::move(event)); },
      width,
      height);
  const auto texture_id = managed_texture->texture_id();
  if (!texture_id.has_value()) {
    return std::nullopt;
  }

  textures_.emplace(*texture_id, std::move(managed_texture));
  return texture_id;
}

bool TextureBridge::UpdateTexture(int64_t texture_id,
                                  int width,
                                  int height,
                                  const std::vector<uint8_t>& pixels) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  return iterator->second->UpdateFrame(width, height, pixels);
}

bool TextureBridge::SetTextureSize(int64_t texture_id, int width, int height) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->Resize(width, height);
  return true;
}

bool TextureBridge::RequestFrame(int64_t texture_id) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->RequestFrame();
  return true;
}

bool TextureBridge::OrbitCamera(int64_t texture_id, float delta_x, float delta_y) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->QueueOrbit(delta_x, delta_y);
  return true;
}

bool TextureBridge::PanCamera(int64_t texture_id, float delta_x, float delta_y) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->QueuePan(delta_x, delta_y);
  return true;
}

bool TextureBridge::ZoomCamera(int64_t texture_id, float delta) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->QueueZoom(delta);
  return true;
}

bool TextureBridge::PickNode(int64_t texture_id,
                             float normalized_x,
                             float normalized_y) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->QueuePick(normalized_x, normalized_y);
  return true;
}

bool TextureBridge::HoverNode(int64_t texture_id,
                              float normalized_x,
                              float normalized_y) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->QueueHover(normalized_x, normalized_y);
  return true;
}

bool TextureBridge::ClearHover(int64_t texture_id) {
  const auto iterator = textures_.find(texture_id);
  if (iterator == textures_.end()) {
    return false;
  }

  iterator->second->ClearHover();
  return true;
}

void TextureBridge::DisposeTexture(int64_t texture_id) {
  textures_.erase(texture_id);
}

const flutter::EncodableMap* TextureBridge::ReadMap(
    const flutter::EncodableValue* value) {
  if (value == nullptr || !std::holds_alternative<flutter::EncodableMap>(*value)) {
    return nullptr;
  }

  return &std::get<flutter::EncodableMap>(*value);
}

const flutter::EncodableValue* TextureBridge::FindMapValue(
    const flutter::EncodableMap& map,
    const char* key) {
  const auto iterator = map.find(flutter::EncodableValue(std::string(key)));
  if (iterator == map.end()) {
    return nullptr;
  }

  return &iterator->second;
}

std::optional<int64_t> TextureBridge::ReadInt64(
    const flutter::EncodableValue* value) {
  if (value == nullptr) {
    return std::nullopt;
  }

  if (std::holds_alternative<int32_t>(*value)) {
    return static_cast<int64_t>(std::get<int32_t>(*value));
  }

  if (std::holds_alternative<int64_t>(*value)) {
    return std::get<int64_t>(*value);
  }

  return std::nullopt;
}

std::optional<double> TextureBridge::ReadDouble(
    const flutter::EncodableValue* value) {
  if (value == nullptr) {
    return std::nullopt;
  }

  if (std::holds_alternative<double>(*value)) {
    return std::get<double>(*value);
  }

  if (std::holds_alternative<int32_t>(*value)) {
    return static_cast<double>(std::get<int32_t>(*value));
  }

  if (std::holds_alternative<int64_t>(*value)) {
    return static_cast<double>(std::get<int64_t>(*value));
  }

  return std::nullopt;
}

std::optional<std::vector<uint8_t>> TextureBridge::ReadBytes(
    const flutter::EncodableValue* value) {
  if (value == nullptr || !std::holds_alternative<std::vector<uint8_t>>(*value)) {
    return std::nullopt;
  }

  return std::get<std::vector<uint8_t>>(*value);
}