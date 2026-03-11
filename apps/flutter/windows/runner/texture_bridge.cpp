#include "texture_bridge.h"

#include <flutter/standard_method_codec.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace {
constexpr const char* kTextureChannelName = "sdf_modeler/texture";
constexpr int kMinDimension = 1;
constexpr int kMaxDimension = 4096;

int ClampDimension(int value) {
  return std::clamp(value, kMinDimension, kMaxDimension);
}

size_t ByteCountForDimensions(int width, int height) {
  return static_cast<size_t>(width) * static_cast<size_t>(height) * 4U;
}
}  // namespace

TextureBridge::ManagedTexture::ManagedTexture(
    flutter::TextureRegistrar* texture_registrar,
    int width,
    int height)
    : texture_registrar_(texture_registrar),
      width_(ClampDimension(width)),
      height_(ClampDimension(height)) {
  const size_t initial_byte_count = ByteCountForDimensions(width_, height_);
  front_pixels_.assign(initial_byte_count, uint8_t{0});
  back_pixels_.assign(initial_byte_count, uint8_t{0});

  texture_ = std::make_unique<flutter::TextureVariant>(flutter::PixelBufferTexture(
      [this](size_t, size_t) -> const FlutterDesktopPixelBuffer* {
        std::lock_guard<std::mutex> lock(mutex_);

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
}

TextureBridge::ManagedTexture::~ManagedTexture() {
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
    std::lock_guard<std::mutex> lock(mutex_);

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

TextureBridge::TextureBridge(flutter::BinaryMessenger* messenger,
                             flutter::TextureRegistrar* texture_registrar)
    : texture_registrar_(texture_registrar) {
  method_channel_ = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
      messenger, kTextureChannelName, &flutter::StandardMethodCodec::GetInstance());

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

std::optional<int64_t> TextureBridge::CreateTexture(int width, int height) {
  auto managed_texture =
      std::make_unique<ManagedTexture>(texture_registrar_, width, height);
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

std::optional<std::vector<uint8_t>> TextureBridge::ReadBytes(
    const flutter::EncodableValue* value) {
  if (value == nullptr || !std::holds_alternative<std::vector<uint8_t>>(*value)) {
    return std::nullopt;
  }

  return std::get<std::vector<uint8_t>>(*value);
}
